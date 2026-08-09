"""
Provider-pinned standard routes.

For each provider named in ``general_settings.pinned_provider_routes`` this
module registers two literal routes:

    POST /{provider}/v1/chat/completions
    POST /{provider}/v1/messages

Each route records the URL's provider in a SINGLE server-authoritative routing
signal — ``metadata[_pinned_provider_route]`` — and then delegates to the SAME
endpoint functions the unified routes use (``chat_completion`` /
``anthropic_response``). The tag router reads that signal with ABSOLUTE
priority (``_resolve_request_tags`` in
``litellm/router_strategy/tag_based_routing.py``): when it is present the
routing tag set is EXACTLY ``["pin:{provider}"]``, derived from the signal
alone and IGNORING ``tags`` / ``_original_request_tags`` / key+team tags for
the routing decision. Combined with router-level ``enable_tag_filtering: true``
+ ``tag_filtering_match_any: false`` (subset matching) and deployments tagged
``pin:<provider>``, the URL prefix — not the client — decides which deployments
are eligible to serve the call.

Because the pinned routing decision derives from ONE field with ONE trusted
source (the URL, set unconditionally by ``_pin_request_tags``), no client
input can change it — dict decoy, JSON-string-encoded metadata, key/team tag
pollution and spoofed pre-merge snapshots are all closed BY CONSTRUCTION, not
by a blocklist. The handler PRESERVES the client's legitimate body-root
attribution ``tags`` and appends the pin tag for SPEND ATTRIBUTION (key/team
tags are appended there too for analytics); routing never reads ``tags``. The
pin also disables the tag router's ``default``-pool fallback (see
``get_deployments_for_tag``), so an unmatched pin fails loud rather than
spilling to ``default``. Unified routes carry no signal, so their tag
routing/attribution is entirely unchanged.

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
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Request, Response
from fastapi.routing import APIRoute
from starlette.routing import Match

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
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
    PINNED_PROVIDER_ROUTE_KEY,
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

# The two metadata buckets the base request processor and the router key the
# routing tags off (besides the body-root ``tags`` list): ``metadata`` and,
# for ``LITELLM_METADATA_ROUTES``, ``litellm_metadata`` — see
# ``add_request_tag_to_metadata`` and its cross-merge in
# ``litellm/proxy/litellm_pre_call_utils.py``. On a pinned route exactly ONE is
# authoritative (the base processor's path-based ``_get_metadata_variable_name``
# selects it). ``_pin_request_tags`` parses either bucket if it arrived as a
# JSON string, strips the routing controls (``tags`` + the internal routing
# fields) from BOTH, and — for the non-authoritative bucket — MERGES its
# legitimate non-routing fields into the authoritative one (or keeps it, when it
# is the provider-facing Anthropic ``metadata`` on a ``litellm_metadata`` route)
# so nothing but routing controls is dropped. Routing itself no longer depends
# on any of this: the router reads the single trusted ``PINNED_PROVIDER_ROUTE_KEY``
# signal with absolute priority (see ``_resolve_request_tags``).
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


def assert_tag_filtering_enabled_for_pinned_routes(
    general_settings: dict | None,
    router_settings: dict | None,
) -> None:
    """Fail loud at startup when a pinned-provider-routes config would let a
    pinned request escape pin enforcement.

    The hard product contract of a pinned route — a pinned request is served
    ONLY by a ``pin:<provider>`` deployment, or fails loud — is enforced at
    request time in ``get_deployments_for_tag`` and re-asserted at the single
    commit chokepoint ``Router._update_kwargs_with_deployment`` (even with tag
    filtering off, the trusted pin signal forces the check). This assertion is
    the complementary STARTUP guard: it refuses to boot a config whose intent
    (pin the provider) contradicts a router/proxy policy that would bypass the
    router or disable tag filtering, rather than depending solely on the
    request-time enforcement. It closes two contradictions:

    1. ``general_settings.pass_through_all_models`` (finding 1): when enabled the
       proxy calls ``litellm.acompletion`` DIRECTLY for a model outside the
       router model list, bypassing the router entirely — so a pinned request
       would never be pin-enforced OR attributable. A pinned request must go
       through the router. Refuse to boot the two together.
    2. ``router_settings.enable_tag_filtering`` not genuinely enabled (finding 4):
       the router treats this setting with STRICT ``is True`` semantics
       (``get_deployments_for_tag`` / ``Router.__init__``), so a quoted
       ``enable_tag_filtering: "false"`` (or ``"true"``, ``"0"``, ``1``) is
       DISABLED at the router yet would pass a truthiness check here. Require the
       genuine boolean ``True`` so this guard agrees with the router EXACTLY and
       a string can never sneak past it.

    Names the exact conflicting setting so the operator can fix it. Prefers
    failing loud over silently serving unpinned traffic.

    No-ops when pinned routes are not configured (setting absent, empty, or an
    invalid non-list — the latter is separately rejected by the initializer).
    """
    configured = (general_settings or {}).get(PINNED_PROVIDER_ROUTES_SETTING)
    if not configured or not isinstance(configured, list):
        return

    # Finding 1: a pinned request must never take the pass-through-all-models
    # direct-call branch (route_llm_request.py), which skips the router and thus
    # the pin. Forbid the two together at boot.
    if (general_settings or {}).get("pass_through_all_models"):
        raise ValueError(
            f"{PINNED_PROVIDER_ROUTES_SETTING} is configured ({configured!r}) but "
            "general_settings.pass_through_all_models is enabled. A pass-through-all "
            "request calls litellm.acompletion directly, bypassing the router — so a "
            "pinned request would never be pin-enforced or attributable. A pinned "
            "request must go through the router. Disable pass_through_all_models or "
            f"remove {PINNED_PROVIDER_ROUTES_SETTING}."
        )

    # Finding 4: match the router's STRICT ``is True`` interpretation exactly — a
    # string "false"/"true"/"0" is treated as DISABLED by the router, so it must
    # not pass a truthiness check here.
    enable_tag_filtering = (router_settings or {}).get("enable_tag_filtering")
    if enable_tag_filtering is True:
        return
    raise ValueError(
        f"{PINNED_PROVIDER_ROUTES_SETTING} is configured ({configured!r}) but "
        f"router_settings.enable_tag_filtering is not the boolean true (got {enable_tag_filtering!r}). "
        "The router enables tag filtering only for the genuine boolean true (a quoted "
        '"false"/"true"/"0" is treated as disabled), so provider-pinned routes require it '
        "set to true — a pinned request is then restricted to its pin:<provider> deployments "
        "(or fails loud) and can never be served by an off-provider deployment. Set "
        "`router_settings: {enable_tag_filtering: true}` (unquoted) or remove "
        f"{PINNED_PROVIDER_ROUTES_SETTING}."
    )


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


def _sanitize_client_metadata_bucket(bucket: object) -> dict | None:
    """Parse a client metadata bucket to a dict (if it arrived JSON-encoded) and
    strip every routing control from it.

    Returns the sanitized dict, or ``None`` when the value cannot carry metadata
    (a non-dict, non-parseable string, or a JSON string that did not decode to a
    dict) — the caller then drops the field so it can neither carry routing nor
    steer the router's presence-based bucket selector.

    The JSON-string parse is the P1 close: a client can send the bucket as a
    STRING (``litellm_metadata: "{\\"_original_request_tags\\": [...]}"``), which
    ``add_litellm_data_to_request`` later parses into a dict. Sanitizing only the
    dict form would let the string form smuggle routing controls past this
    handler. Routing no longer depends on any of these controls (the router reads
    the trusted ``PINNED_PROVIDER_ROUTE_KEY`` signal), but they are still stripped
    so the decoy can never carry ``tags`` into SPEND ATTRIBUTION or reintroduce a
    stale snapshot.
    """
    if isinstance(bucket, str):
        parsed = safe_json_loads(bucket)
        bucket = parsed if isinstance(parsed, dict) else None
    if not isinstance(bucket, dict):
        return None
    bucket.pop("tags", None)
    for internal_field in _CLIENT_STRIPPED_ROUTING_FIELDS:
        bucket.pop(internal_field, None)
    bucket.pop(PINNED_PROVIDER_ROUTE_KEY, None)
    return bucket


async def _pin_request_tags(request: Request, pin_tag: str) -> None:
    """Record the trusted, URL-derived provider pin and re-cache the body.

    Root-cause design: the routing decision for a pinned request derives SOLELY
    from a single server-authoritative signal — ``PINNED_PROVIDER_ROUTE_KEY``,
    the provider named by the URL — which the tag router reads with ABSOLUTE
    priority (``_resolve_request_tags``). This handler sets that signal
    UNCONDITIONALLY from the URL, overwriting any client value in any form
    (dict or JSON-string, either bucket). Because the router ignores ``tags`` /
    ``_original_request_tags`` / key+team tags for a pinned routing decision,
    NO client input can change where a pinned request routes — the whole class
    of metadata-seam bypasses (dict decoy, string-encoded metadata, key/team
    tag pollution, spoofed snapshot) is closed by construction, not by a
    blocklist. On unified routes the signal is absent, so routing is unchanged.

    Alongside the signal this handler keeps the request well-formed:

    - Body-root ``tags``: the client's legitimate SPEND-ATTRIBUTION tags
      (``cost-center:*``, ``customer:*`` …) are PRESERVED and the server pin tag
      is appended (the base processor merges the result into ``metadata["tags"]``,
      where key/team tags are also appended). Routing NEVER reads ``tags`` — it
      derives SOLELY from the trusted ``PINNED_PROVIDER_ROUTE_KEY`` signal
      (``_resolve_request_tags`` ignores ``tags`` when the pin is present) — so
      preserving client tags cannot change where a pinned request routes; they
      flow only to attribution. Only client-supplied ``pin:*`` tags are dropped:
      the server owns pinning, and a stray client pin in SpendLogs would
      misattribute the provider. (This replaces the earlier destructive
      ``tags = [pin_tag]`` overwrite, which was only needed back when routing
      read ``tags`` and which discarded legitimate attribution tags.)
    - Both metadata buckets are parsed (if JSON-encoded) and stripped of routing
      controls. The NON-authoritative bucket's legitimate NON-routing fields are
      preserved — MERGED into the authoritative bucket when it is the
      litellm-internal ``litellm_metadata`` (so tracking like
      ``spend_logs_metadata`` survives), or KEPT in place when it is the
      provider-facing Anthropic ``metadata`` (so ``metadata.user_id`` survives).
      This replaces the earlier wholesale drop, which discarded those fields.
    - The trusted provider is also stashed on ``request.state`` so the base
      processor can re-assert the signal LAST (after it parses/merges the body),
      a belt-and-suspenders guard against a string-encoded bucket preempting it.
    """
    data = await _read_request_body(request=request)
    pinned_provider = pin_tag[len(PIN_TAG_PREFIX) :]

    # Body-root: PRESERVE the client's legitimate spend-attribution tags and
    # append the server pin tag. Drop only client-supplied ``pin:*`` tags (the
    # server owns pinning). Routing reads the trusted signal, not ``tags``, so
    # preserved client tags can never redirect a pinned request — they flow only
    # to SPEND ATTRIBUTION. Also drop any client-supplied internal routing fields
    # + the trusted pin signal (only the URL may set it).
    raw_client_tags = data.get("tags")
    preserved_client_tags = (
        [tag for tag in raw_client_tags if isinstance(tag, str) and not tag.startswith(PIN_TAG_PREFIX)]
        if isinstance(raw_client_tags, list)
        else []
    )
    data["tags"] = [*preserved_client_tags, pin_tag]
    for internal_field in _CLIENT_STRIPPED_ROUTING_FIELDS:
        data.pop(internal_field, None)
    data.pop(PINNED_PROVIDER_ROUTE_KEY, None)

    # Which bucket the base processor writes and the router reads: the SAME
    # path-based selector the processor uses. Import inline — importing
    # litellm_pre_call_utils at module load would eagerly pull the proxy import
    # graph and defeat this module's vanilla-until-configured posture.
    from litellm.proxy.litellm_pre_call_utils import (  # noqa: PLC0415
        _get_metadata_variable_name,
    )

    authoritative_field = _get_metadata_variable_name(request)
    decoy_field = "litellm_metadata" if authoritative_field == "metadata" else "metadata"

    # Parse (JSON-string → dict) and strip routing controls from BOTH buckets.
    for metadata_field in _CLIENT_TAG_METADATA_FIELDS:
        if metadata_field not in data:
            continue
        sanitized = _sanitize_client_metadata_bucket(data.get(metadata_field))
        if sanitized is None:
            data.pop(metadata_field, None)
        else:
            data[metadata_field] = sanitized

    # Preserve the non-authoritative bucket's legitimate NON-routing fields.
    if authoritative_field == "metadata":
        # The decoy is ``litellm_metadata``, a litellm-internal bucket the base
        # processor MERGES into ``metadata`` (spend_logs_metadata, tracking).
        # Mirror that merge so those fields survive, then REMOVE the decoy so the
        # router's presence-based selector cannot land on an untagged bucket.
        decoy = data.get(decoy_field)
        if isinstance(decoy, dict):
            authoritative = data.get(authoritative_field)
            if not isinstance(authoritative, dict):
                authoritative = {}
                data[authoritative_field] = authoritative
            for key, value in decoy.items():
                authoritative.setdefault(key, value)
        data.pop(decoy_field, None)
    # else: the authoritative bucket is ``litellm_metadata`` and the decoy is
    # the PROVIDER-FACING Anthropic ``metadata`` (``metadata.user_id`` etc.).
    # KEEP it (routing already stripped) so provider metadata survives — the
    # router's presence-based selector still picks ``litellm_metadata`` (present,
    # authoritative), so the kept ``metadata`` cannot steer routing.

    # Record the SINGLE trusted routing signal in the authoritative bucket.
    authoritative = data.get(authoritative_field)
    if not isinstance(authoritative, dict):
        authoritative = {}
        data[authoritative_field] = authoritative
    authoritative[PINNED_PROVIDER_ROUTE_KEY] = pinned_provider

    # Belt-and-suspenders: stash the trusted provider on request.state (a
    # server-only store) so the base processor re-asserts the signal LAST, after
    # it parses/merges the body (see add_litellm_data_to_request).
    setattr(request.state, PINNED_PROVIDER_ROUTE_KEY, pinned_provider)

    # Re-store explicitly: the parsed-body cache snapshots accepted keys at set
    # time (see _safe_get_request_parsed_body), so an in-place mutation that ADDS
    # a "tags" key would be dropped on the next read.
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
    general_settings: dict | None,
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
