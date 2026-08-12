"""
Use this to route requests between Teams

- If tags in request is a subset of tags in deployment, return deployment
- if deployments are set with default tags, return all default deployment
- If no default_deployments are set, return all deployments
- A "!tag" excludes deployments carrying that tag; a "&tag" requires it
"""

import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, NoReturn

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
    *,
    pin_enforced: bool = False,
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


def _bare_tag_value(tag: str) -> str | None:
    # Mirrors _split_tags' own stripping rule exactly, so a confirmed value
    # compares equal to whatever required_set/excluded_set/positive_tags end up
    # holding for the same tag: a "&"/"!" marker is stripped only when something
    # follows it; a lone marker with nothing after it parses to nothing in any
    # of the three sets, so it must not become a confirmed value either.
    if tag.startswith(("&", "!")):
        return tag[1:] if len(tag) > 1 else None
    return tag


def _strip_routing_prefix(tags: Sequence[str], prefix: str) -> tuple[tuple[str, ...], frozenset[str]]:
    # Strips the configured routing-prefix marker from any tag carrying it, used
    # exactly as configured with no delimiter auto-appended, and separately
    # tracks the post-strip, post-marker-strip values that arrived prefixed: tags
    # whose routing intent the caller declared explicitly, exempt from the "maybe
    # foreign to this group" heuristics in _unknown_required_tag_hides_an_answer
    # and _tag_known_to_group below. Confirmed values are compared against
    # required_set/excluded_set downstream, which are themselves already stripped
    # of their "&"/"!" marker by _split_tags -- confirmed must match that same
    # bare form, not the raw post-prefix-strip value that still carries the
    # marker character. An empty prefix must return every tag unconfirmed, not
    # run every tag through str.startswith(""), which is trivially True for
    # every string and would mark everything confirmed.
    if not prefix:
        return tuple(tags), frozenset()
    rewritten: Final = tuple(t.removeprefix(prefix) for t in tags)
    confirmed: Final = frozenset(
        bare
        for bare in (_bare_tag_value(t.removeprefix(prefix)) for t in tags if t.startswith(prefix))
        if bare is not None
    )
    return rewritten, confirmed


def _split_tags(tags: Sequence[str]) -> tuple[tuple[str, ...], list[str], tuple[str, ...]]:
    required: Final = tuple(tag[1:] for tag in tags if tag.startswith("&") and len(tag) > 1)
    positive: Final = [
        t for t in tags if not t.startswith("!") and not t.startswith("&")
    ]  # mutable-ok: feeds _match_deployment's existing list[str]-typed request_tags param
    excluded: Final = tuple(tag[1:] for tag in tags if tag.startswith("!") and len(tag) > 1)
    return required, positive, excluded


def _exclude_deployments(
    deployments: Sequence[Any] | Mapping[Any, Any],
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


def _require_all_tags(
    deployments: Sequence[Any] | Mapping[Any, Any],
    required_set: frozenset[str],
) -> tuple[Any, ...]:
    if not required_set:
        return tuple(deployments)
    return tuple(d for d in deployments if required_set.issubset(d.get("litellm_params", {}).get("tags") or []))


def _default_tagged_pool(
    deployments: Sequence[Any] | Mapping[Any, Any],
) -> tuple[Any, ...]:
    defaults: Final = tuple(d for d in deployments if "default" in (d.get("litellm_params", {}).get("tags") or []))
    return defaults if defaults else tuple(deployments)


def _known_tag_values(deployments: Sequence[Any] | Mapping[Any, Any]) -> frozenset[str]:
    return frozenset(
        tag for d in deployments for tag in (d.get("litellm_params", MappingProxyType({})).get("tags") or ())
    )


def _unknown_required_tag_hides_an_answer(
    healthy_deployments: Sequence[Any] | Mapping[Any, Any],
    excluded_set: frozenset[str],
    required_set: frozenset[str],
    routing_confirmed: frozenset[str],
) -> bool:
    # A caller-invented "&" tag (one no deployment in this group has ever carried)
    # guarantees an empty required-AND result on its own, regardless of whether the
    # rest of the request's required tags were satisfiable. Dropping the unknown
    # tags and recomputing: if that reveals a specific, non-empty answer, the invented
    # tag was the actual cause of the exhaustion, and fail-open must not paper over
    # it. If every required tag is known, or none are, there's nothing hidden to
    # protect: either the caller made a real, honestly-unsatisfiable ask (fail-open
    # proceeds normally), or the whole required set is unrecognized noise with no
    # narrower answer to hide behind it. routing_confirmed (tag_routing_prefix)
    # counts as known too: the caller explicitly declared it a routing directive,
    # so it is never treated as invented noise regardless of deployment vocabulary.
    known_required: Final = required_set & (_known_tag_values(healthy_deployments) | routing_confirmed)
    if not known_required or known_required == required_set:
        return False
    allowed: Final = _exclude_deployments(healthy_deployments, excluded_set)
    return bool(_require_all_tags(allowed, known_required))


def _chain_allows_fail_open(
    healthy_deployments: Sequence[Any] | Mapping[Any, Any],
    excluded_set: frozenset[str],
    required_set: frozenset[str],
    routing_confirmed: frozenset[str],
) -> bool:
    if _unknown_required_tag_hides_an_answer(healthy_deployments, excluded_set, required_set, routing_confirmed):
        return False
    return any((d.get("model_info") or {}).get("allow_fail_open") is True for d in healthy_deployments)


def _trusted_only_pool(
    healthy_deployments: Sequence[Any] | Mapping[Any, Any],
    excluded_set: frozenset[str],
    required_set: frozenset[str],
    inherited_excluded_set: frozenset[str] | None,
    inherited_required_set: frozenset[str] | None,
) -> tuple[Any, ...]:
    # inherited_*_set is None only when this request carries no origin information
    # at all (e.g. direct SDK Router usage, bypassing the proxy layer that
    # populates metadata.inherited_tags) -- treat every constraint as
    # caller-controlled in that case (protected == empty), reproducing this
    # function's pre-provenance behavior exactly: an unconditional fall-open to the
    # full default-tagged pool, constraints discarded entirely. Otherwise, a tag
    # value is protected the moment it has ANY inherited backing, even when the
    # caller also happens to submit the identical value themselves -- set
    # membership can't distinguish "this value came from policy" from "this value
    # coincidentally matches policy," so presence in the inherited set (not
    # absence from a caller-supplied set) is what must gate discardability. This
    # is deliberately intersection with inherited_*_set, not subtraction of a
    # caller-supplied set: subtraction would let a caller strip an inherited
    # requirement's protection just by resubmitting its exact value alongside a
    # conflicting one (e.g. inherited "&region:eu" plus caller "&region:eu"
    # and "!region:eu" would otherwise cancel the inherited requirement out).
    trusted_excluded: Final = (
        frozenset[str]() if inherited_excluded_set is None else inherited_excluded_set & excluded_set
    )
    trusted_required: Final = (
        frozenset[str]() if inherited_required_set is None else inherited_required_set & required_set
    )
    return _require_all_tags(_exclude_deployments(healthy_deployments, trusted_excluded), trusted_required)


def _resolve_or_fail_open(
    pool: Sequence[Any],
    healthy_deployments: Sequence[Any] | Mapping[Any, Any],
    excluded_set: frozenset[str],
    required_set: frozenset[str],
    inherited_excluded_set: frozenset[str] | None,
    inherited_required_set: frozenset[str] | None,
    routing_confirmed: frozenset[str],
    model: str,
    request_tags: object,
) -> tuple[Any, ...]:
    if pool:
        return tuple(pool)
    if _chain_allows_fail_open(healthy_deployments, excluded_set, required_set, routing_confirmed):
        # Fall open only within whatever still satisfies whichever constraints
        # trace back to key/team policy. A constraint with no inherited backing at
        # all (or, when inherited_tags is unavailable, any constraint at all) can
        # be discarded; one inherited from key/team policy cannot -- if that alone
        # is unsatisfiable, raise instead of silently routing around it.
        trusted_pool: Final = _trusted_only_pool(
            healthy_deployments, excluded_set, required_set, inherited_excluded_set, inherited_required_set
        )
        if trusted_pool:
            return _default_tagged_pool(trusted_pool)
    _raise_no_deployments_for_tags(model=model, request_tags=request_tags)


def _resolve_constraint_only_pool(
    healthy_deployments: Sequence[Any] | Mapping[Any, Any],
    excluded_set: frozenset[str],
    required_set: frozenset[str],
    inherited_excluded_set: frozenset[str] | None,
    inherited_required_set: frozenset[str] | None,
    routing_confirmed: frozenset[str],
    model: str,
    request_tags: object,
) -> tuple[Any, ...]:
    pool: Final = (
        _require_all_tags(_exclude_deployments(healthy_deployments, excluded_set), required_set)
        if required_set
        else _exclude_deployments(_default_tagged_pool(healthy_deployments), excluded_set)
    )
    return _resolve_or_fail_open(
        pool,
        healthy_deployments,
        excluded_set,
        required_set,
        inherited_excluded_set,
        inherited_required_set,
        routing_confirmed,
        model,
        request_tags,
    )


def _all_deployments_or_fallback(
    llm_router_instance: LitellmRouter,
    model: str,
    fallback: Sequence[Any] | Mapping[Any, Any],
) -> Sequence[Any] | Mapping[Any, Any]:
    try:
        return llm_router_instance._get_all_deployments(model_name=model)
    except Exception:  # noqa: BLE001  # fail safe toward today's healthy-only behavior on lookup errors
        return fallback


def _chain_tag_filtering_override(
    llm_router_instance: LitellmRouter,
    model: str,
    healthy_deployments: Sequence[Any] | Mapping[Any, Any],
) -> bool | None:
    # Resolved from every deployment configured for this model group, not just the
    # ones that survived cooldown/health filtering (async_get_healthy_deployments
    # filters cooldowns before calling get_deployments_for_tag) -- otherwise the
    # sole deployment carrying this group's only explicit override loses its effect
    # the moment it's transiently unhealthy, silently falling back to the
    # router-wide default and letting an attacker disable a chain's tag policy by
    # repeatedly failing that one deployment into cooldown. Falls back to
    # healthy_deployments on a lookup error, preserving today's behavior rather
    # than crashing the request.
    all_deployments: Final = _all_deployments_or_fallback(llm_router_instance, model, healthy_deployments)
    for d in all_deployments:
        value = (d.get("model_info") or MappingProxyType({})).get("enable_tag_filtering")
        if value is not None:
            return value
    return None


def _inherited_constraint_sets(
    inherited_tags: object, routing_prefix: str
) -> tuple[frozenset[str] | None, frozenset[str] | None]:
    # None means no origin information is available at all (e.g. this request
    # bypassed the proxy layer that populates metadata.inherited_tags, as direct
    # SDK Router usage does) -- callers of this must treat that as "nothing is
    # protected," not "nothing is inherited," see _trusted_only_pool.
    # metadata.inherited_tags is a snapshot of whatever key/team/project policy
    # merged into "tags" *before* this request's own caller-supplied tags were
    # merged in on top (see litellm_pre_call_utils.py), so a value present here is
    # policy-backed regardless of whether the caller also happens to submit the
    # identical value. inherited_tags is stripped through the same routing_prefix
    # as the main request tags so a policy-inherited prefixed tag still matches
    # correctly against the (already-stripped) required_set/excluded_set computed
    # from request_tags.
    if not isinstance(inherited_tags, (list, tuple)):
        return None, None
    rewritten_inherited_tags: Final = _strip_routing_prefix(inherited_tags, routing_prefix)[0]
    inherited_required, _inherited_positive, inherited_excluded = _split_tags(rewritten_inherited_tags)
    return frozenset(inherited_required), frozenset(inherited_excluded)


def _tag_known_to_group(
    llm_router_instance: LitellmRouter,
    model: str,
    positive_tags: Sequence[str],
    routing_confirmed: frozenset[str],
) -> bool:
    tag_set: Final = frozenset(positive_tags)
    if tag_set & routing_confirmed:
        return True
    try:
        all_deployments: Final = llm_router_instance._get_all_deployments(model_name=model)
    except Exception:  # noqa: BLE001  # fail safe toward "unrecognized" so lookup errors preserve the existing silent-fallback behavior
        return False
    return any(
        tag_set.intersection(d.get("litellm_params", MappingProxyType({})).get("tags") or ()) for d in all_deployments
    )


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

    Runs when the effective enable_tag_filtering is True. Effective value: a
    request-level enable_tag_filtering=True (set from key/team router_settings by
    the proxy) always wins; otherwise model_info.enable_tag_filtering on this model
    group, if set on any of its deployments, overrides the router-wide default.
    A request-level False never disables either of those, so per-request settings
    cannot escape an operator's or a chain owner's tag-routing policy.

    HARD PROVIDER PIN (P1): a trusted, URL-derived ``PINNED_PROVIDER_ROUTE_KEY``
    signal forces pin filtering REGARDLESS of the effective enable_tag_filtering
    value above — read and applied BEFORE that early-return. So a pinned request
    is ALWAYS restricted to its ``pin:<provider>`` deployments (or fails loud),
    never served by an off-provider deployment just because tag filtering happens
    to be disabled. The proxy's startup guard
    (``assert_tag_filtering_enabled_for_pinned_routes``) is the belt; this is the
    suspenders — the pin holds even if that guard is bypassed.
    """
    if request_kwargs is None or not healthy_deployments:
        verbose_logger.debug(
            "get_deployments_for_tag: skipping tag filter (request_kwargs=%s, healthy_deployments=%s)",
            request_kwargs,
            healthy_deployments,
        )
        return healthy_deployments

    pinned_provider: Final = _pinned_provider_from_kwargs(request_kwargs, metadata_variable_name)
    request_enable_tag_filtering: Final = request_kwargs.get("enable_tag_filtering")
    chain_enable_tag_filtering: Final = _chain_tag_filtering_override(llm_router_instance, model, healthy_deployments)
    chain_default: Final = (
        chain_enable_tag_filtering
        if chain_enable_tag_filtering is not None
        else llm_router_instance.enable_tag_filtering
    )
    if pinned_provider is None and request_enable_tag_filtering is not True and chain_default is not True:
        return healthy_deployments

    verbose_logger.debug("request metadata: %s", request_kwargs.get(metadata_variable_name))
    if metadata_variable_name in request_kwargs:
        metadata: Final = request_kwargs[metadata_variable_name]
        request_tags: Final = _resolve_request_tags(metadata)
        # A hard provider pin (any ``pin:`` tag) must never spill to the
        # ``default`` pool: an empty tag match fails loud rather than being
        # served by a ``default``-tagged deployment.
        disable_default_fallback: Final = any(
            isinstance(t, str) and t.startswith(PIN_TAG_PREFIX) for t in (request_tags or [])
        )
        match_any: Final = llm_router_instance.tag_filtering_match_any
        routing_prefix: Final = llm_router_instance.tag_routing_prefix or ""

        # Build header strings for regex matching from what the proxy already stores.
        # Currently we match against User-Agent; format matches "^User-Agent: claude-code/..."
        user_agent = metadata.get("user_agent", "")
        header_strings: list[str] = [f"User-Agent: {user_agent}"] if user_agent else []

        # A tag_routing_prefix-marked tag is stripped before matching -- everything
        # downstream (_split_tags, deployment matching) works off the unprefixed
        # value, exactly as if the caller had sent it unprefixed -- and its
        # post-strip value is remembered in routing_confirmed as an explicit,
        # caller-declared routing directive, exempt from the "maybe foreign to this
        # group" heuristics that unprefixed tags still go through unchanged below.
        rewritten_tags, routing_confirmed = _strip_routing_prefix(request_tags or [], routing_prefix)
        required_tags, positive_tags, excluded_patterns = _split_tags(rewritten_tags)
        inherited_required_set, inherited_excluded_set = _inherited_constraint_sets(
            metadata.get("inherited_tags"), routing_prefix
        )

        excluded_set: Final = frozenset(excluded_patterns)
        required_set: Final = frozenset(required_tags)
        allowed_deployments: Final = _exclude_deployments(healthy_deployments, excluded_set)
        candidates: Final = _require_all_tags(allowed_deployments, required_set)

        has_regex_deployments: Final = any(d.get("litellm_params", {}).get("tag_regex") for d in candidates)
        has_positive_filter: Final = bool(positive_tags) or (
            bool(header_strings) and has_regex_deployments and not required_set
        )
        constraint_only: Final = (bool(excluded_set) or bool(required_set)) and not has_positive_filter

        if constraint_only:
            return _resolve_constraint_only_pool(
                healthy_deployments,
                excluded_set,
                required_set,
                inherited_excluded_set,
                inherited_required_set,
                routing_confirmed,
                model,
                request_tags,
            )

        new_healthy_deployments: list[Any] = []
        default_deployments: list[Any] = []

        if has_positive_filter:
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
                return _resolve_or_fail_open(
                    (),
                    healthy_deployments,
                    excluded_set,
                    required_set,
                    inherited_excluded_set,
                    inherited_required_set,
                    routing_confirmed,
                    model,
                    request_tags,
                )

            if (
                len(new_healthy_deployments) == 0
                and positive_tags
                and _tag_known_to_group(llm_router_instance, model, positive_tags, routing_confirmed)
            ):
                return _resolve_or_fail_open(
                    (),
                    healthy_deployments,
                    excluded_set,
                    required_set,
                    inherited_excluded_set,
                    inherited_required_set,
                    routing_confirmed,
                    model,
                    request_tags,
                )

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
