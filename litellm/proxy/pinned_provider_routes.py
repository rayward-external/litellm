"""
Provider-pinned standard routes.

For each provider named in ``general_settings.pinned_provider_routes`` this
module registers two literal routes:

    POST /{provider}/v1/chat/completions
    POST /{provider}/v1/messages

Each route REPLACES the request's routing tags with exactly the server-side
namespaced tag ``pin:{provider}`` (client-supplied tags are dropped from the
routing decision) and then delegates to the SAME endpoint functions the
unified routes use (``chat_completion`` / ``anthropic_response``). Combined
with router-level ``enable_tag_filtering: true`` +
``tag_filtering_match_any: false`` (subset matching) and deployments tagged
``pin:<provider>``, the URL prefix — not the client — decides which
deployments are eligible to serve the call.

The routing tag set is REPLACED, not unioned: under strict subset matching
any EXTRA client tag would make the request tag set not-a-subset of the
pinned deployment's ``["pin:{provider}"]``, matching nothing and falling
through to the tag router's ``default``-deployment pool — an escape from the
pin (see ``get_deployments_for_tag`` in
``litellm/router_strategy/tag_based_routing.py``). Forcing the routing tags
to exactly the pin closes that: the pinned deployment always matches by
subset, the default fallback is unreachable, and a client-supplied
CONFLICTING ``pin:*`` tag can never widen or redirect. Spend attribution is
keyed by API-key hash, not tags, so dropping the client's own routing tags
here costs only optional analytics grouping — the correct security trade-off
for a pinned route (this REVERSES the earlier union-not-replace choice, for
pinned routes only; unified routes still union).

Enable via::

    general_settings:
      pinned_provider_routes: [azure, azure_ai, bedrock, vertex_ai, gemini, fireworks, baseten]

Without that setting the module registers nothing, so vanilla deployments
are entirely unaffected. ``openai`` and ``anthropic`` are refused as pinned
prefixes (``PINNED_PREFIX_DENYLIST``): their trees are live credentialed
pass-through surfaces that a pinned literal route would shadow.

Registration order is load-bearing: the provider pass-through catch-alls
(``/gemini/{endpoint:path}``, ``/bedrock/{endpoint:path}``,
``/azure/{endpoint:path}``, ``/vertex_ai/{endpoint:path}`` in
``litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py``) would
swallow the pinned literal paths if they matched first. This module runs at
config-load time — after every import-time ``include_router`` call — so a
plain ``include_router`` would append the pinned routes AFTER the
catch-alls and lose. Instead, the routes are spliced into
``app.router.routes`` immediately before the first existing route that
would otherwise match a pinned path, so the pinned literal route always
wins. The route-precedence test in
``tests/test_litellm/proxy/test_pinned_provider_routes.py`` guards this
against upstream reorderings.

The ``/v1/messages`` delegate is imported inside the handler:
``litellm.proxy.anthropic_endpoints.endpoints`` is a lazily attached module
(``litellm/proxy/_lazy_features.py``), and a module-level import here would
defeat that startup laziness.
"""

import asyncio
import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Optional

from fastapi import APIRouter, Depends, Request, Response
from fastapi.routing import APIRoute
from starlette.routing import Match

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.http_parsing_utils import (
    _read_request_body,
    _safe_set_request_parsed_body,
)
from litellm.router_strategy.tag_based_routing import (
    ORIGINAL_REQUEST_TAGS_KEY,
    ORIGINAL_REQUEST_TAGS_SNAPSHOT_TAKEN_KEY,
    PIN_TAG_PREFIX,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

# general_settings key that enables + configures this feature.
PINNED_PROVIDER_ROUTES_SETTING = "pinned_provider_routes"

# Server-side tag namespace (``PIN_TAG_PREFIX``, imported from the tag router).
# By convention, not enforcement on unified routes: a client may send
# "pin:<provider>" itself there and get the same pinning — acceptable, since
# under subset matching tags only ever narrow eligibility. On a PINNED route the
# prefix is authoritative and also disables the default-pool fallback in the tag
# router (get_deployments_for_tag), so a hard pin can never spill to ``default``.

# Body fields the base request processor also reads client-supplied ``tags``
# from when it builds the router-visible tag stream (besides the body-root
# ``tags`` list): the client ``metadata`` dict and, for
# ``LITELLM_METADATA_ROUTES``, ``litellm_metadata`` — see
# ``add_request_tag_to_metadata`` and its cross-merge in
# ``litellm/proxy/litellm_pre_call_utils.py``. A pinned route strips ``tags``
# from these so the replaced pin is the only routing tag.
_CLIENT_TAG_METADATA_FIELDS: tuple[str, ...] = ("metadata", "litellm_metadata")

# Internal, server-authoritative routing fields a client must NEVER supply. The
# router captures ``ORIGINAL_REQUEST_TAGS_KEY`` (its pre-merge tag snapshot) plus
# the ``..._SNAPSHOT_TAKEN_KEY`` sentinel itself; a client that pre-seeds either
# on a pinned route could otherwise re-open the pin (a /bedrock request smuggling
# ``_original_request_tags=["pin:openai"]`` and routing to OpenAI). The pinned
# route strips both from the body root and from every metadata field the base
# processor merges, so only the server-set pin ever reaches the router.
_CLIENT_STRIPPED_ROUTING_FIELDS: tuple[str, ...] = (
    ORIGINAL_REQUEST_TAGS_KEY,
    ORIGINAL_REQUEST_TAGS_SNAPSHOT_TAKEN_KEY,
)

# Route prefixes that are aliases of another provider's tag namespace:
# "/gemini/..." addresses the same deployments as "/vertex_ai/..." (both
# inject "pin:vertex_ai") — the deployments carry a single canonical tag.
PINNED_TAG_ALIASES: dict[str, str] = {"gemini": "vertex_ai"}

# Route prefixes accepted although they are not literal litellm provider
# names: "fireworks" deployments use custom_llm_provider "fireworks_ai",
# but the public route prefix and the pin tag keep the short name.
_EXTRA_ALLOWED_PREFIXES: frozenset[str] = frozenset({"fireworks"})

# Provider prefixes that must NEVER be pinned: "/openai/..." and
# "/anthropic/..." are live credentialed pass-through surfaces
# (LiteLLMRoutes.mapped_pass_through_routes + the passthrough routers), so a
# pinned literal route spliced ahead of them would shadow real client
# traffic on those trees. Warn + skip instead of registering.
PINNED_PREFIX_DENYLIST: frozenset[str] = frozenset({"openai", "anthropic"})

# ``app.state`` attribute under which this module keeps its per-app registry
# of the route objects it mounted (see ``_pinned_route_registry``). Named
# here so the proxy_server load-config gate can cheaply detect a prior
# registration — ``bool(getattr(app.state, PINNED_ROUTE_REGISTRY_STATE_ATTR,
# None))`` — WITHOUT importing this module, preserving the vanilla posture
# that a deployment which never configured pinned routes imports nothing.
PINNED_ROUTE_REGISTRY_STATE_ATTR = "pinned_provider_route_objects"

# Module path of the heavy, lazily-attached messages-dialect delegate
# (see litellm/proxy/_lazy_features.py). Imported off the event loop at
# registration time so the FIRST pinned /v1/messages request does not stall
# the loop for the 1-3 s the import takes.
_ANTHROPIC_ENDPOINTS_MODULE = "litellm.proxy.anthropic_endpoints.endpoints"


def get_pin_tag(provider: str) -> str:
    """The tag a pinned route injects for ``provider`` (alias-aware)."""
    return PIN_TAG_PREFIX + PINNED_TAG_ALIASES.get(provider, provider)


def _pinned_route_registry(app: "FastAPI") -> dict[int, tuple[str, APIRoute]]:
    """This module's app-scoped registry of the route OBJECTS it registered:
    ``id(route) -> (provider, route)``.

    Living on ``app.state`` keeps the registry per-app (no cross-app module
    state — two FastAPI instances each get their own ``State``), and holding
    the route object itself keeps a strong reference, so an id key can never
    be silently reused by a different live object while its entry exists.
    Starlette's ``State`` raises ``AttributeError`` for missing attributes,
    so first access creates the dict.
    """
    try:
        return getattr(app.state, PINNED_ROUTE_REGISTRY_STATE_ATTR)
    except AttributeError:
        registry: dict[int, tuple[str, APIRoute]] = {}
        setattr(app.state, PINNED_ROUTE_REGISTRY_STATE_ATTR, registry)
        return registry


def _known_provider_prefixes() -> frozenset[str]:
    provider_names = {str(getattr(p, "value", p)) for p in litellm.provider_list}
    return frozenset(provider_names | _EXTRA_ALLOWED_PREFIXES | set(PINNED_TAG_ALIASES))


async def _pin_request_tags(request: Request, pin_tag: str) -> None:
    """REPLACE the request's routing tags with exactly ``[pin_tag]`` and
    re-cache the body.

    On a pinned route the URL prefix — not the client — decides which
    deployments are eligible, so the routing tag set must be EXACTLY the pin
    regardless of client input. Under strict subset matching
    (``tag_filtering_match_any: false``) any EXTRA client tag would make the
    request tag set not-a-subset of the pinned deployment's
    ``["pin:{provider}"]``, matching nothing and falling through to the tag
    router's ``default``-deployment pool — an escape from the pin. Replacing
    (not unioning) the client's tags closes that, and a client-supplied
    CONFLICTING ``pin:*`` tag can never widen or redirect. Spend attribution
    is keyed by API-key hash (not tags), so this costs only optional
    analytics grouping.

    The base request processor builds the router-visible tag stream (landing
    in ``metadata`` or ``litellm_metadata`` per ``LITELLM_METADATA_ROUTES``)
    by merging client tags from BOTH the body-root ``tags`` list AND any
    client-supplied ``metadata`` / ``litellm_metadata`` ``tags`` — see
    ``LiteLLMProxyRequestSetup.add_request_tag_to_metadata`` and its
    ``litellm_metadata`` cross-merge in
    ``litellm/proxy/litellm_pre_call_utils.py``. To make the pin the ONLY
    routing tag, set body-root ``tags`` to the pin and strip ``tags`` from
    every client metadata field the merge reads.
    """
    data = await _read_request_body(request=request)
    # REPLACE (not union): the pin is the sole routing tag, whatever the
    # client sent at the body root.
    data["tags"] = [pin_tag]
    # Strip client-supplied tags AND the internal, server-authoritative routing
    # fields from the body root, so a client cannot pre-seed the router's
    # original-tags snapshot (re-opening the pin) or the default-fallback flag.
    for internal_field in _CLIENT_STRIPPED_ROUTING_FIELDS:
        data.pop(internal_field, None)
    # Same strip on every metadata field the base processor merges into the
    # routing stream, so none can survive the merge.
    for metadata_field in _CLIENT_TAG_METADATA_FIELDS:
        metadata = data.get(metadata_field)
        if isinstance(metadata, dict):
            metadata.pop("tags", None)
            for internal_field in _CLIENT_STRIPPED_ROUTING_FIELDS:
                metadata.pop(internal_field, None)
    # The pin tag itself disables the default-pool fallback in the tag router
    # (get_deployments_for_tag keys off the ``pin:`` prefix), so no separate
    # flag needs to ride the request — the pin can never spill to ``default``.
    # Re-store explicitly: the parsed-body cache snapshots accepted keys at
    # set time (see _safe_get_request_parsed_body), so an in-place mutation
    # that ADDS a "tags" key would be dropped on the next read.
    _safe_set_request_parsed_body(request=request, parsed_body=data)


def _make_pinned_chat_completion_handler(pin_tag: str) -> Callable:
    async def pinned_chat_completion(
        request: Request,
        fastapi_response: Response,
        user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    ):
        """Provider-pinned ``/v1/chat/completions``: inject the pin tag, then
        delegate to the exact endpoint function the unified route uses."""
        from litellm.proxy.proxy_server import chat_completion  # noqa: PLC0415

        await _pin_request_tags(request=request, pin_tag=pin_tag)
        return await chat_completion(
            request=request,
            fastapi_response=fastapi_response,
            model=None,
            user_api_key_dict=user_api_key_dict,
        )

    return pinned_chat_completion


def _make_pinned_anthropic_messages_handler(pin_tag: str) -> Callable:
    async def pinned_anthropic_messages(
        request: Request,
        fastapi_response: Response,
        user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    ):
        """Provider-pinned ``/v1/messages``: inject the pin tag, then delegate
        to the exact endpoint function the unified route uses.

        Import inside the handler on purpose: anthropic_endpoints.endpoints
        is lazily attached (litellm/proxy/_lazy_features.py) and a
        module-level import would eagerly load it at startup.
        """
        from litellm.proxy.anthropic_endpoints.endpoints import (  # noqa: PLC0415
            anthropic_response,
        )

        await _pin_request_tags(request=request, pin_tag=pin_tag)
        return await anthropic_response(
            fastapi_response=fastapi_response,
            request=request,
            user_api_key_dict=user_api_key_dict,
        )

    return pinned_anthropic_messages


def _existing_literal_post_route(app: "FastAPI", path: str) -> bool:
    """True if an exact-path POST APIRoute is already registered (idempotency
    on config reload; also refuses to double-register over a literal route)."""
    for route in app.router.routes:
        if isinstance(route, APIRoute) and route.path == path and "POST" in (route.methods or set()):
            return True
    return False


def _first_matching_route_index(app: "FastAPI", paths: list[str]) -> int:
    """Index of the first existing route that would match any of ``paths``
    (e.g. the provider pass-through catch-alls). The pinned routes must be
    inserted before it to win FastAPI's in-order route matching. Falls back
    to appending when nothing matches."""
    for idx, route in enumerate(app.router.routes):
        for path in paths:
            scope = {
                "type": "http",
                "method": "POST",
                "path": path,
                "root_path": "",
                "headers": [],
                "query_string": b"",
                "path_params": {},
            }
            match, _ = route.matches(scope)
            if match is not Match.NONE:
                return idx
    return len(app.router.routes)


def _schedule_messages_dialect_warmup() -> None:
    """Kick the heavy ``anthropic_endpoints`` import onto the default executor.

    Mirrors the executor pattern ``litellm/proxy/_lazy_features.py`` uses in
    ``_force_load`` (``loop.run_in_executor(None, importlib.import_module,
    ...)``): the import runs off the event loop, so the first pinned
    ``/v1/messages`` request finds the module already in ``sys.modules``
    instead of paying a multi-second, loop-blocking import inline. Outside a
    running loop (unit tests, sync callers) this is a silent no-op — the
    handler's inline import remains the correctness fallback either way.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.run_in_executor(None, importlib.import_module, _ANTHROPIC_ENDPOINTS_MODULE)


def _remove_stale_pinned_routes(app: "FastAPI", desired_paths: set[str]) -> list[str]:
    """Drop previously-registered pinned routes that are no longer configured.

    Re-init hygiene for config reloads: pinned routes are identified by
    OBJECT IDENTITY against the app-scoped ``app.state`` registry, so only
    routes THIS module registered on THIS app are ever removed — a user
    route mounted at the same path is never touched. Registry entries whose
    route object is no longer in ``app.router.routes`` (removed by someone
    else) are pruned. Returns the removed paths.
    """
    registry = _pinned_route_registry(app)
    removed: list[str] = []
    kept: list = []
    for route in app.router.routes:
        entry = registry.get(id(route))
        if entry is not None and entry[1].path not in desired_paths:
            removed.append(entry[1].path)
            del registry[id(route)]
            continue
        kept.append(route)
    if removed:
        app.router.routes[:] = kept
        app.openapi_schema = None  # rebuilt on next /openapi.json request
        verbose_proxy_logger.info("pinned_provider_routes: removed stale routes %s", removed)
    # Prune entries whose route object has left the route table entirely, so
    # the registry never keeps dead route objects (and their ids) alive.
    live_route_ids = {id(route) for route in app.router.routes}
    for stale_id in [route_id for route_id in registry if route_id not in live_route_ids]:
        del registry[stale_id]
    return removed


def initialize_pinned_provider_routes(
    app: "FastAPI",
    general_settings: Optional[dict],
) -> list[str]:
    """Register the pinned provider routes named in
    ``general_settings.pinned_provider_routes``. Returns the list of route
    paths registered by THIS call (empty when disabled / already registered).

    Also removes pinned routes registered by a PREVIOUS call that are no
    longer configured (config reload hygiene) — including all of them when
    the setting is absent or empty.
    """
    configured = (general_settings or {}).get(PINNED_PROVIDER_ROUTES_SETTING)
    if configured is not None and not isinstance(configured, list):
        verbose_proxy_logger.warning(
            "pinned_provider_routes: expected a list of provider names, got %r — ignoring.",
            type(configured).__name__,
        )
        # Treat an invalid value as "disabled": a reload that REJECTED the
        # setting must still drop any routes an earlier valid config registered,
        # rather than leaving stale endpoints live behind a rejected config.
        _remove_stale_pinned_routes(app, desired_paths=set())
        return []
    if not configured:
        # Disabled (setting absent or []): a reload that dropped the setting
        # must also drop any routes an earlier config registered.
        _remove_stale_pinned_routes(app, desired_paths=set())
        return []

    known_prefixes = _known_provider_prefixes()
    pinned_router = APIRouter()
    registered_paths: list[str] = []
    desired_paths: set[str] = set()
    provider_by_path: dict[str, str] = {}
    seen: set[str] = set()

    for provider in configured:
        if not isinstance(provider, str) or not provider:
            verbose_proxy_logger.warning("pinned_provider_routes: skipping non-string entry %r.", provider)
            continue
        if provider in seen:
            continue
        seen.add(provider)
        if provider in PINNED_PREFIX_DENYLIST:
            verbose_proxy_logger.warning(
                "pinned_provider_routes: refusing to pin %r — /%s/* is a live "
                "credentialed pass-through surface (mapped_pass_through_routes), "
                "and a pinned literal route would shadow it. Skipping.",
                provider,
                provider,
            )
            continue
        if provider not in known_prefixes:
            verbose_proxy_logger.warning(
                "pinned_provider_routes: unknown provider %r — no routes registered for it.",
                provider,
            )
            continue

        pin_tag = get_pin_tag(provider)
        chat_path = f"/{provider}/v1/chat/completions"
        messages_path = f"/{provider}/v1/messages"
        desired_paths.update((chat_path, messages_path))

        if not _existing_literal_post_route(app, chat_path):
            pinned_router.add_api_route(
                chat_path,
                _make_pinned_chat_completion_handler(pin_tag),
                methods=["POST"],
                name=f"pinned_{provider}_chat_completion",
                dependencies=[Depends(user_api_key_auth)],
                tags=["provider-pinned routes"],
            )
            registered_paths.append(chat_path)
            provider_by_path[chat_path] = provider
        if not _existing_literal_post_route(app, messages_path):
            pinned_router.add_api_route(
                messages_path,
                _make_pinned_anthropic_messages_handler(pin_tag),
                methods=["POST"],
                name=f"pinned_{provider}_messages",
                dependencies=[Depends(user_api_key_auth)],
                tags=["provider-pinned routes"],
            )
            registered_paths.append(messages_path)
            provider_by_path[messages_path] = provider

    # Config-reload hygiene: drop pinned routes from a previous call that are
    # no longer configured, BEFORE splicing in the new ones.
    _remove_stale_pinned_routes(app, desired_paths=desired_paths)

    if not registered_paths:
        return []

    # Splice before the first route (catch-all or otherwise) that would
    # swallow a pinned path — include_router alone would append AFTER the
    # pass-through catch-alls and lose the in-order match.
    insert_at = _first_matching_route_index(app, registered_paths)
    start = len(app.router.routes)
    app.include_router(pinned_router)
    new_routes = app.router.routes[start:]
    del app.router.routes[start:]
    app.router.routes[insert_at:insert_at] = new_routes
    app.openapi_schema = None  # rebuilt on next /openapi.json request

    # Record the actual mounted route objects in the app-scoped registry so
    # stale-route removal can later match them by object identity.
    registry = _pinned_route_registry(app)
    for route in new_routes:
        if isinstance(route, APIRoute):
            registry[id(route)] = (provider_by_path[route.path], route)

    if any(path.endswith("/v1/messages") for path in registered_paths):
        # Cold-start hygiene: pre-import the heavy messages-dialect delegate
        # off the event loop so the first pinned request doesn't stall.
        _schedule_messages_dialect_warmup()

    verbose_proxy_logger.info("pinned_provider_routes: registered %s", registered_paths)
    return registered_paths
