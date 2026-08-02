"""
Use this to route requests between Teams

- If tags in request is a subset of tags in deployment, return deployment
- if deployments are set with default tags, return all default deployment
- If no default_deployments are set, return all deployments
"""

import re
from typing import TYPE_CHECKING, Any, Literal, NoReturn

import litellm
from litellm._logging import verbose_logger
from litellm.types.router import RouterErrors

# Neutral internal metadata key holding a snapshot of the CALLER's original
# request tags, captured once per request (before any per-deployment tag merge)
# by ``Router._update_kwargs_before_fallbacks``. Tag-based routing prefers this
# snapshot over the live ``metadata["tags"]`` so a selected deployment's tags —
# merged into ``metadata["tags"]`` by ``_update_kwargs_with_deployment`` for
# spend attribution, then reused across retries/fallbacks — can never leak back
# in as routing input on a later attempt (see get_deployments_for_tag).
ORIGINAL_REQUEST_TAGS_KEY = "_original_request_tags"

# Neutral internal sentinel marking that ``Router._update_kwargs_before_fallbacks``
# has already captured ``ORIGINAL_REQUEST_TAGS_KEY`` for THIS request. The snapshot
# is captured on the FIRST invocation only and OVERWRITTEN from the trusted live
# tags, so a client that smuggles a spoofed ``ORIGINAL_REQUEST_TAGS_KEY`` into
# metadata cannot pre-seed the routing snapshot (setdefault would have trusted it).
# Gating on this dedicated sentinel — never the spoofable field itself — preserves
# idempotency across the retry/fallback re-invocations that reuse the kwargs dict.
ORIGINAL_REQUEST_TAGS_SNAPSHOT_TAKEN_KEY = "_original_request_tags_snapshotted"

# Server-side namespace prefix for provider-pin routing tags. A request whose
# (original) tags carry ANY ``pin:``-prefixed tag is a HARD provider pin set by
# the proxy's provider-pinned routes (litellm/proxy/pinned_provider_routes.py):
# it must NEVER fall through to the ``default``-deployment pool — an empty tag
# match fails loud (``no_deployments_with_tag_routing``) instead, so a pin can
# never be silently served by a ``default``-tagged deployment, independent of
# any static config guard in another repo.
#
# Keying this off the pin tag itself (not a separate metadata flag) is
# deliberate: the pin tag already reaches the router-visible metadata reliably,
# whereas a companion flag would have to survive the proxy's PATH-based
# metadata-field selection vs the router's PRESENCE-based selection
# (get_metadata_variable_name_from_kwargs) — a divergence that silently drops a
# flag set in the wrong field. The pin policy stays in one place (the tag).
PIN_TAG_PREFIX = "pin:"

# The SINGLE server-authoritative routing signal for a provider-pinned request.
# Holds the provider whose pin the URL selected (alias-resolved, e.g. ``gemini``
# → ``vertex_ai``). It is set UNCONDITIONALLY from the URL by the proxy's pinned
# routes (litellm/proxy/pinned_provider_routes.py) — overwriting any client
# value in ANY form (dict or JSON-string, ``metadata`` or ``litellm_metadata``)
# and re-asserted last from ``request.state`` in ``add_litellm_data_to_request``
# so a string-encoded client bucket cannot preempt it. It is ALSO used as the
# ``request.state`` attribute name the pinned handler stashes the trusted
# provider under.
#
# When present, ``_resolve_request_tags`` derives the ROUTING tag set SOLELY
# from this signal — ``[pin:<provider>]`` — and ignores ``tags`` /
# ``_original_request_tags`` / key+team tags entirely for the routing decision.
# That closes the whole class of metadata-seam bypasses (dict decoy,
# string-encoded metadata, key/team tag pollution, spoofed snapshot) BY
# CONSTRUCTION: none of those inputs feed a pinned routing decision. ``tags``
# still flows untouched to SPEND ATTRIBUTION. On unified (non-pinned) routes the
# signal is absent, so normal tag routing/attribution is unchanged.
PINNED_PROVIDER_ROUTE_KEY = "_pinned_provider_route"

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


def _is_valid_deployment_tag_regex(
    tag_regexes: list[str],
    header_strings: list[str],
) -> str | None:
    """
    Test compiled regex patterns against "Header-Name: value" strings.

    Returns the first matching pattern string, or None if nothing matches.
    Compiles each pattern once (re's LRU cache) and logs invalid patterns once
    per pattern, not once per header string.
    """
    for pattern in tag_regexes:
        try:
            compiled = re.compile(pattern)
        except re.error:
            verbose_logger.warning("tag_regex: invalid pattern %r — skipping", pattern)
            continue
        for header_str in header_strings:
            if compiled.search(header_str):
                return pattern
    return None


def is_valid_deployment_tag(deployment_tags: list[str], request_tags: list[str], match_any: bool = True) -> bool:
    """
    Check if a tag is valid, the matching can be either any or all based on `match_any` flag
    """
    if not request_tags:
        return False

    dep_set = set(deployment_tags)
    req_set = set(request_tags)

    if match_any:
        is_valid_deployment = bool(dep_set & req_set)
    else:
        is_valid_deployment = req_set.issubset(dep_set)

    if is_valid_deployment:
        verbose_logger.debug(
            "adding deployment with tags: %s, request tags: %s for match_any=%s",
            deployment_tags,
            request_tags,
            match_any,
        )
        return True
    return False


def _match_deployment(
    deployment: Any,
    request_tags: list[str] | None,
    header_strings: list[str],
    match_any: bool,
) -> dict[str, str] | None:
    """
    Determine whether *deployment* matches the current request.

    Returns {"matched_via": ..., "matched_value": ...} if the deployment
    should be included, or None if it should be excluded.

    Priority:
      1. Exact tag match (respects match_any semantics).
      2. Regex match — skipped when match_any=False and the tag check already
         ran and failed, so the regex cannot override strict-tag policy.

    ``pin_enforced`` (P1 hard-pin guard): when the request carries a trusted
    provider pin, the ONLY acceptable match is exact membership of the
    ``pin:<provider>`` tag in the deployment's ``tags``. The ``tag_regex`` path
    (which evaluates CLIENT-controlled headers like User-Agent) is disabled
    entirely, so a deployment lacking the pin tag can never be selected for a
    pinned request via a spoofable User-Agent — the pin is the SOLE selector.
    """
    litellm_params = deployment.get("litellm_params", {})
    deployment_tags: list[str] | None = litellm_params.get("tags")
    deployment_tag_regex: list[str] | None = litellm_params.get("tag_regex")

    # 1. Exact tag match (existing behaviour). For a pinned request request_tags
    # is exactly ``[pin:<provider>]``, so this admits ONLY deployments that carry
    # the pin tag (under both match_any and subset semantics).
    if deployment_tags and request_tags:
        if is_valid_deployment_tag(deployment_tags, request_tags, match_any):
            matched_value = next(
                (t for t in deployment_tags if t in set(request_tags)),
                deployment_tags[0],
            )
            return {"matched_via": "tags", "matched_value": matched_value}

    # 2. Regex match against request headers.
    # When match_any=False and the deployment has plain tags, the strict tag
    # check either didn't run (no request tags) or failed (step 1 returned
    # None).  Block the regex path so it cannot circumvent the operator's
    # strict-tag policy. For a pinned request the regex path is blocked
    # UNCONDITIONALLY: a client-controlled User-Agent must never select a
    # deployment that lacks the pin tag.
    deployment_has_plain_tags = deployment_tags is not None and len(deployment_tags) > 0
    strict_tag_check_failed = not match_any and deployment_has_plain_tags
    if deployment_tag_regex and header_strings and not strict_tag_check_failed and not pin_enforced:
        regex_match = _is_valid_deployment_tag_regex(deployment_tag_regex, header_strings)
        if regex_match is not None:
            return {"matched_via": "tag_regex", "matched_value": regex_match}

    return None


def _split_tags(tags: list[str]) -> tuple[list[str], list[str]]:
    positive = [t for t in tags if not t.startswith("!")]
    excluded = [tag[1:] for tag in tags if tag.startswith("!") and len(tag) > 1]
    return positive, excluded


def _exclude_deployments(
    deployments: list[Any] | dict[Any, Any],
    excluded_set: frozenset[str],
) -> list[Any]:
    if not excluded_set:
        return list(deployments)
    return [d for d in deployments if not excluded_set.intersection(d.get("litellm_params", {}).get("tags") or [])]


def _raise_no_deployments_for_tags(model: str, request_tags: Any) -> NoReturn:
    """No deployment matches the request's tags: raise a typed, NON-retryable
    client error.

    ``litellm.BadRequestError`` carries ``status_code=400``, so
    ``Router.should_retry_this_error`` re-raises it immediately
    (``litellm._should_retry(400)`` is False) instead of burning
    ``num_retries`` on a request that can never succeed — a plain
    ``ValueError`` here previously cost multiple retry sleeps before
    surfacing. The message keeps the
    ``RouterErrors.no_deployments_with_tag_routing`` marker plus the model
    and tags so callers (and the proxy's ProxyException mapping) can name
    what was pinned.
    """
    raise litellm.BadRequestError(
        message=(f"{RouterErrors.no_deployments_with_tag_routing.value}. Passed model={model} and tags={request_tags}"),
        model=model,
        llm_provider="",
    )


def _require_candidates(
    candidates: list[Any],
    model: str,
    request_tags: Any,
) -> list[Any]:
    if not candidates:
        _raise_no_deployments_for_tags(model=model, request_tags=request_tags)
    return candidates


def _ban_only_base_pool(
    deployments: list[Any] | dict[Any, Any],
) -> list[Any]:
    # Mirrors untagged-request semantics so callers can't use !tags to escape the default pool.
    defaults = [d for d in deployments if "default" in (d.get("litellm_params", {}).get("tags") or [])]
    return defaults if defaults else list(deployments)


def _resolve_request_tags(metadata: dict[Any, Any]) -> Any:
    """Return the request tags to route on.

    ABSOLUTE-PRIORITY provider pin. If the trusted, URL-derived
    ``PINNED_PROVIDER_ROUTE_KEY`` signal is present, the routing tag set is
    EXACTLY ``["pin:<that provider>"]`` — derived from that single
    server-authoritative field alone, IGNORING ``tags``,
    ``ORIGINAL_REQUEST_TAGS_KEY`` and any key/team tags for the routing
    decision. The proxy's pinned routes set the signal from the URL and nowhere
    else, overwriting any client-supplied copy in any form, so no client input
    can change a pinned routing decision. (``metadata["tags"]`` is left intact
    for SPEND ATTRIBUTION.)

    Otherwise (unified routes — the signal is absent): prefer the pre-merge
    snapshot (``ORIGINAL_REQUEST_TAGS_KEY``) captured once per request by
    ``Router._update_kwargs_before_fallbacks``, so a selected deployment's own
    tags — merged into ``metadata["tags"]`` by ``_update_kwargs_with_deployment``
    for spend attribution, then reused as the kwargs dict is reused across
    retries/fallbacks — can never leak back in as routing input on a later
    attempt (which would reject every other-provider deployment). Falls back to
    the live ``metadata["tags"]`` for back-compat when no snapshot was taken
    (e.g. direct router use in tests that bypasses
    ``_update_kwargs_before_fallbacks``).
    """
    pinned_provider = metadata.get(PINNED_PROVIDER_ROUTE_KEY)
    if isinstance(pinned_provider, str) and pinned_provider:
        return [PIN_TAG_PREFIX + pinned_provider]
    if ORIGINAL_REQUEST_TAGS_KEY in metadata:
        return metadata.get(ORIGINAL_REQUEST_TAGS_KEY)
    return metadata.get("tags")


def _pinned_provider_from_kwargs(
    request_kwargs: dict[Any, Any] | None,
    metadata_variable_name: Literal["metadata", "litellm_metadata"],
) -> str | None:
    """Return the trusted, URL-derived provider pin from the routing metadata
    bucket, or ``None`` when the request is not pinned.

    Only the single server-authoritative ``PINNED_PROVIDER_ROUTE_KEY`` signal
    counts (set from the URL by the proxy's pinned routes, re-asserted from
    ``request.state``); a client can never set it. Used to enforce pin filtering
    even when ``enable_tag_filtering`` is off.
    """
    if not request_kwargs:
        return None
    metadata = request_kwargs.get(metadata_variable_name)
    if not isinstance(metadata, dict):
        return None
    pinned_provider = metadata.get(PINNED_PROVIDER_ROUTE_KEY)
    return pinned_provider if isinstance(pinned_provider, str) and pinned_provider else None


async def get_deployments_for_tag(
    llm_router_instance: LitellmRouter,
    model: str,  # used to raise the correct error
    healthy_deployments: list[Any] | dict[Any, Any],
    request_kwargs: dict[Any, Any] | None = None,
    metadata_variable_name: Literal["metadata", "litellm_metadata"] = "metadata",
):
    """
    Returns a list of deployments that match the requested model and tags in the request.

    Executes tag based filtering based on the tags in request metadata and the tags on the deployments

    Runs when the router-level `enable_tag_filtering` is True or the request carries
    `enable_tag_filtering=True` (set from key/team router_settings by the proxy).
    A request-level False never disables a router-level True, so per-request settings
    cannot escape an operator's global tag-routing policy.

    HARD PROVIDER PIN (P1): a trusted, URL-derived ``PINNED_PROVIDER_ROUTE_KEY``
    signal forces pin filtering REGARDLESS of ``enable_tag_filtering`` — read and
    applied BEFORE the enable_tag_filtering early-return below. So a pinned request
    is ALWAYS restricted to its ``pin:<provider>`` deployments (or fails loud),
    never served by an off-provider deployment just because tag filtering happens
    to be disabled. The proxy's startup guard
    (``assert_tag_filtering_enabled_for_pinned_routes``) is the belt; this is the
    suspenders — the pin holds even if that guard is bypassed.
    """
    pinned_provider = _pinned_provider_from_kwargs(request_kwargs, metadata_variable_name)
    request_enable_tag_filtering = request_kwargs.get("enable_tag_filtering") if request_kwargs else None
    if (
        pinned_provider is None
        and request_enable_tag_filtering is not True
        and llm_router_instance.enable_tag_filtering is not True
    ):
        return healthy_deployments

    if request_kwargs is None:
        verbose_logger.debug(
            "get_deployments_for_tag: request_kwargs is None returning healthy_deployments: %s",
            healthy_deployments,
        )
        return healthy_deployments

    if not healthy_deployments:
        verbose_logger.debug("get_deployments_for_tag: empty or None healthy_deployments; skipping tag filter")
        return healthy_deployments

    verbose_logger.debug("request metadata: %s", request_kwargs.get(metadata_variable_name))
    if metadata_variable_name in request_kwargs:
        metadata = request_kwargs[metadata_variable_name]
        request_tags = _resolve_request_tags(metadata)
        match_any = llm_router_instance.tag_filtering_match_any
        # A hard provider pin (any ``pin:`` tag) must never spill to the
        # ``default`` pool: an empty tag match fails loud rather than being
        # served by a ``default``-tagged deployment.
        disable_default_fallback = any(
            isinstance(t, str) and t.startswith(PIN_TAG_PREFIX) for t in (request_tags or [])
        )

        # Build header strings for regex matching from what the proxy already stores.
        # Currently we match against User-Agent; format matches "^User-Agent: claude-code/..."
        user_agent = metadata.get("user_agent", "")
        header_strings: list[str] = [f"User-Agent: {user_agent}"] if user_agent else []

        positive_tags, excluded_patterns = _split_tags(request_tags or [])

        excluded_set = frozenset(excluded_patterns)
        candidates = _exclude_deployments(healthy_deployments, excluded_set)

        has_regex_deployments = any(d.get("litellm_params", {}).get("tag_regex") for d in candidates)
        has_tag_filter = bool(positive_tags) or (bool(header_strings) and has_regex_deployments)
        ban_only = bool(excluded_set) and not has_tag_filter

        if ban_only:
            pool = _exclude_deployments(_ban_only_base_pool(healthy_deployments), excluded_set)
            return _require_candidates(pool, model, request_tags)

        new_healthy_deployments: list[Any] = []
        default_deployments: list[Any] = []

        if has_tag_filter:
            verbose_logger.debug(
                "get_deployments_for_tag routing: request_tags=%s user_agent=%s",
                request_tags,
                user_agent,
            )
            for deployment in candidates:
                deployment_tags = deployment.get("litellm_params", {}).get("tags")

                match_result = _match_deployment(
                    deployment=deployment,
                    request_tags=positive_tags,
                    header_strings=header_strings,
                    match_any=match_any,
                    # A hard pin disables tag_regex/User-Agent matching so only
                    # deployments carrying the pin tag are eligible.
                    pin_enforced=disable_default_fallback,
                )

                if match_result is not None:
                    verbose_logger.debug(
                        "tag routing match: deployment=%s matched_via=%s matched_value=%s",
                        deployment.get("model_name"),
                        match_result["matched_via"],
                        match_result["matched_value"],
                    )
                    if "tag_routing" not in metadata:
                        metadata["tag_routing"] = {
                            "matched_deployment": deployment.get("model_name"),
                            "matched_via": match_result["matched_via"],
                            "matched_value": match_result["matched_value"],
                            "request_tags": request_tags or [],
                            "user_agent": user_agent,
                        }
                    new_healthy_deployments.append(deployment)

                # Skip building the default-pool fallback entirely when this
                # request forbids it (a hard pin): with default_deployments left
                # empty, an unmatched pin falls through to the fail-loud raise
                # below instead of being served by a ``default`` deployment.
                if not disable_default_fallback and deployment_tags and "default" in deployment_tags:
                    default_deployments.append(deployment)

            if len(new_healthy_deployments) == 0 and len(default_deployments) == 0:
                _raise_no_deployments_for_tags(model=model, request_tags=request_tags)

            return new_healthy_deployments if len(new_healthy_deployments) > 0 else default_deployments

    # for Untagged requests use default deployments if set
    _default_deployments_with_tags = []
    for deployment in healthy_deployments:
        if "default" in deployment.get("litellm_params", {}).get("tags", []):
            _default_deployments_with_tags.append(deployment)

    if len(_default_deployments_with_tags) > 0:
        return _default_deployments_with_tags

    # if no default deployment is found, return healthy_deployments
    verbose_logger.debug(
        "no tier found in metadata, returning healthy_deployments: %s",
        healthy_deployments,
    )
    return healthy_deployments


def _get_tags_from_request_kwargs(
    request_kwargs: dict[Any, Any] | None = None,
    metadata_variable_name: Literal["metadata", "litellm_metadata"] = "metadata",
) -> list[str]:
    """
    Helper to get tags from request kwargs

    Args:
        request_kwargs: The request kwargs to get tags from

    Returns:
        List[str]: The tags from the request kwargs
    """
    if request_kwargs is None:
        return []
    if metadata_variable_name in request_kwargs:
        metadata = request_kwargs[metadata_variable_name] or {}
        tags = metadata.get("tags", [])
        return tags if tags is not None else []
    elif "litellm_params" in request_kwargs:
        litellm_params = request_kwargs["litellm_params"] or {}
        _metadata = litellm_params.get(metadata_variable_name, {}) or {}
        tags = _metadata.get("tags", [])
        return tags if tags is not None else []
    return []
