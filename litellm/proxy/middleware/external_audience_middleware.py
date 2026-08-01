"""
RAYWARD FORK PATCH — minimal response exposure for external callers.

Two policies, one audience gate, one module: response HEADERS are reduced to an
allowlist, and error BODIES are sanitized. The header half is described first
because it came first; the body half and the reasoning behind its different
shape are documented above ``ROUTING_DISCLOSURE_MARKERS`` further down.

WHY THIS EXISTS
---------------
An external party gets an API key and nothing else: they must not be able to
learn what gateway software we run, that we proxy at all, who we proxy to, or
what capacity we bought.

This cannot be solved at the load balancer. GCP url-map ``responseHeadersToRemove``
caps at 50 names, and LiteLLM emits 121 distinct response header names (measured
2026-07-31). Whichever 50 are listed, the rest reappear on a leg the selection did
not anticipate. The root cause is ``_get_llm_provider_headers()`` in
``litellm/litellm_core_utils/llm_response_utils/get_headers.py``, which re-publishes
EVERY upstream response header under an ``llm_provider-`` prefix with no allowlist
and no gating flag -- so the emitted name set is unbounded and provider-dependent.

WHY AN ALLOWLIST AND NOT A PREFIX DENYLIST
------------------------------------------
This started as a denylist over ``x-litellm-``, ``llm_provider-`` and bare
``x-ratelimit-``, on the reasoning that LiteLLM gives its leak a SHAPE: it puts
its own facts under one prefix and everything upstream under another, so matching
those two prefixes closes it. That is true of the native path and false of
pass-through. ``HttpPassThroughEndpointHelpers.get_response_headers()``
(``proxy/pass_through_endpoints/pass_through_endpoints.py``) forwards the
upstream's headers VERBATIM and UN-PREFIXED, excluding only seven framing names
-- so ``openai-version``, ``anthropic-organization-id``, ``cf-ray`` and whatever
the next provider invents sail straight through a prefix match. On that path the
leak has no shape at all, exactly like Bifrost's.

A denylist can only ever list what someone has already seen. So the policy is
inverted: keep a fixed set of standard HTTP / CORS / security / WebSocket
headers and drop everything else. A new provider, or an existing one that starts
sending a new header, is covered on the day it ships rather than the day someone
notices. The companion patch in the Bifrost fork is the same shape.

Two things fall out of the inversion for free. Bare ``x-ratelimit-*`` (OUR
purchased deployment quota, not the caller's key limit) is not in the allowlist,
so it goes without needing its own rule. And an upstream that sends its own
``x-usage-cost`` can no longer collide with the renamed header below: the
upstream copy is dropped, ours is written, and the caller sees exactly one value
rather than two joined into something non-numeric.

IF THIS PATCH IS DROPPED BY A REBASE
------------------------------------
External callers immediately see ``x-litellm-*`` (names our gateway software),
``llm_provider-*`` (proves we proxy, and carries the upstream's own request ids
and quota counters), and bare ``x-ratelimit-*`` holding OUR purchased deployment
quota. ``tests/test_litellm/proxy/middleware/test_external_audience_middleware.py``
fails if any part of this goes missing, including the registration in
``proxy_server.py``.

WHY A MIDDLEWARE AND NOT A PATCH AT get_headers.py
--------------------------------------------------
``get_custom_headers`` alone has 40+ call sites and does not receive the Request,
and pass-through / error paths build headers without going through it at all. A
pure-ASGI middleware sees the ``http.response.start`` message for EVERY HTTP
response -- streaming, non-streaming, and error -- so it is both the smallest and
the only complete seam. One module plus one registration line to re-apply.

SAFE FAILURE DIRECTION
----------------------
Absence of the audience header means INTERNAL, i.e. full headers. Only an explicit
``external`` value turns suppression on. If it were inverted, any failure to inject
the header would silently un-hide everything on the external path. With this
direction the worst a client can do by forging the header is suppress its own
response headers, which harms nobody.
"""

import json
import re
from collections.abc import Iterable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: Request header injected by the EXTERNAL load balancer's backend service.
#: Deliberately vendor-neutral -- AGENTS.md requires neutral header names in fork
#: patches, so it must never contain "rayward".
AUDIENCE_REQUEST_HEADER = "x-gateway-audience"
EXTERNAL_AUDIENCE = "external"

#: Renamed, not suppressed: this is the caller's OWN usage data, which they
#: legitimately need. The neutral names disclose nothing about the gateway.
#: Applied BEFORE the allowlist test below, which would otherwise drop all three.
#: Pairs rather than a dict so the table is immutable at import: nothing can
#: grow this at runtime, and a header cannot be renamed into existence later.
RENAMED_HEADERS: tuple[tuple[str, str], ...] = (
    ("x-litellm-response-cost", "x-usage-cost"),
    ("x-litellm-key-spend", "x-usage-spend"),
    ("x-litellm-key-max-budget", "x-usage-budget"),
)

#: The COMPLETE set of response headers an external caller receives. Anything not
#: named here is dropped, including every ``x-litellm-*`` and ``llm_provider-*``
#: name and every bare upstream header the pass-through path forwards verbatim.
#:
#: Membership test: does a standard HTTP client need it, and does it say nothing
#: about who we are or who we buy from? Lowercase; lookups normalize.
ALLOWED_HEADERS: frozenset[str] = frozenset(
    (
        # Message framing. uvicorn owns most of these; dropping one corrupts the
        # response rather than protecting anything.
        "content-type",
        "content-length",
        "content-encoding",
        "transfer-encoding",
        "connection",
        "date",
        "vary",
        "cache-control",
        "allow",
        # Backpressure. Unlike the vendor and deployment quota families this
        # carries no capacity figure -- just "come back later" -- so it stays.
        "retry-after",
        # CORS. Browser clients break without these. ``expose-headers`` is
        # rewritten rather than passed through (see below): LiteLLM advertises
        # LITELLM_UI_ALLOW_HEADERS in it, i.e. literal ``x-litellm-*`` names,
        # which is a disclosure even once those headers themselves are gone.
        "access-control-allow-origin",
        "access-control-allow-methods",
        "access-control-allow-headers",
        "access-control-allow-credentials",
        "access-control-expose-headers",
        "access-control-max-age",
        # Generic hardening, identical on any host.
        "x-frame-options",
        "x-content-type-options",
        "referrer-policy",
        "content-security-policy",
        "permissions-policy",
        "strict-transport-security",
        "x-robots-tag",
        # WebSocket handshake (``/v1/realtime``). Pure protocol:
        # ``sec-websocket-accept`` is a hash of the client's own nonce and
        # ``sec-websocket-protocol`` echoes what the client asked for, so none of
        # it describes us -- but a 101 missing them is rejected outright.
        "upgrade",
        "sec-websocket-accept",
        "sec-websocket-protocol",
        "sec-websocket-extensions",
        "sec-websocket-version",
        # uvicorn writes its own value at serialization time whether or not this
        # is set, so removing it here changes nothing on the wire. Listed so this
        # set stays an accurate description of what actually goes out; removing
        # ``server`` is the load balancer's job.
        "server",
    )
)

#: CORS advertises LITELLM_UI_ALLOW_HEADERS here, i.e. literal "x-litellm-*" names.
#: That is a disclosure even once the headers themselves are gone, so on the
#: external path the list is re-pointed at the neutral usage headers.
_EXPOSE_HEADERS_NAME = "access-control-expose-headers"
_EXTERNAL_EXPOSED_HEADERS = ", ".join(neutral for _, neutral in RENAMED_HEADERS).encode("latin-1")


def _is_external_audience(scope: Scope) -> bool:
    """
    True only if the audience header explicitly says "external".

    GCP ``custom_request_headers`` ADDS a header rather than replacing it, so when a
    client also sends one the value can arrive duplicated (two entries) or
    comma-joined into one. Every occurrence is checked and every token within it.
    """
    raw_headers: Iterable[tuple[bytes, bytes]] = scope.get("headers") or ()
    for raw_name, raw_value in raw_headers:
        if raw_name.decode("latin-1").lower() != AUDIENCE_REQUEST_HEADER:
            continue
        for token in raw_value.decode("latin-1").replace(",", " ").split():
            if token.lower() == EXTERNAL_AUDIENCE:
                return True
    return False


def _rewrite_header(raw_name: bytes, raw_value: bytes) -> tuple[bytes, bytes] | None:
    """One header's external form, or None to drop it entirely."""
    name = raw_name.decode("latin-1").lower()

    # Checked BEFORE the allowlist, which would otherwise drop all three. This
    # ordering is also what gives the gateway's value precedence over an upstream
    # that happens to send a header of the same neutral name: the upstream copy
    # reaches the allowlist test, is not in it, and is dropped.
    for original, neutral in RENAMED_HEADERS:
        if name == original:
            return (neutral.encode("latin-1"), raw_value)

    if name not in ALLOWED_HEADERS:
        return None

    if name == _EXPOSE_HEADERS_NAME:
        return (raw_name, _EXTERNAL_EXPOSED_HEADERS)

    return (raw_name, raw_value)


def apply_external_header_policy(
    headers: Iterable[tuple[bytes, bytes]],
) -> tuple[tuple[bytes, bytes], ...]:
    """Rewrite one response's raw ASGI header list for an external caller.

    Returns a tuple, not a list: the ASGI spec types ``headers`` as an iterable
    of pairs, and nothing is layered between this middleware and the server that
    could want to mutate it in place -- being registered outermost is what makes
    that true.
    """
    return tuple(
        rewritten for raw_name, raw_value in headers if (rewritten := _rewrite_header(raw_name, raw_value)) is not None
    )


# ---------------------------------------------------------------------------
# ERROR BODIES
#
# The header policy above closes the header route. Error BODIES are a second,
# independent route, and they leak worse: not our vendor's facts but OUR OWN
# routing decisions. Measured live on the external router, 2026-08-01, from a
# request whose only fault was a misspelled ``role``:
#
#   litellm.BadRequestError: Not allowed to access model due to tags
#   configuration. Passed model=gpt-5.6-luna-openai and
#   tags=['!pin:anthropic', '!pin:openai'] No fallback model group found for
#   original model_group=gpt-5.6-luna-openai.
#   Fallbacks=[{'gpt-5.4': ['gpt-5.4-openai']}, ... 7 groups ...]
#
# Four disclosures in one string: it NAMES THE GATEWAY SOFTWARE, publishes the
# whole internal fallback table, exposes internal model-group naming, and hands
# over the deny-tag mechanism -- syntax and values -- that is the only thing
# keeping this key on our hyperscaler credits rather than our direct vendor
# accounts. We were publishing the control to the party it constrains.
#
# It is not an edge case: 3 of 4 ordinary client mistakes reproduced it
# (temperature out of range, negative max_tokens, a bad role), so it fires on
# any 4xx from the primary leg during normal operation.
#
# WHY A DENYLIST HERE, WHEN THE HEADER POLICY IS DELIBERATELY AN ALLOWLIST
# -----------------------------------------------------------------------
# Header NAMES are a closed vocabulary, so "keep these, drop the rest" is
# expressible. An error MESSAGE is free text, and the useful part is written by
# the upstream: "max_tokens: Field required" is worth forwarding and cannot be
# enumerated in advance. An allowlist over free text degenerates to "return
# nothing", which is a real cost -- an external caller who cannot see why their
# request was rejected files a ticket we then answer by hand.
#
# So the policy is two independent controls, and a message must pass BOTH:
#
#   1. Markers -- known internal vocabulary. Catches the shapes we measured.
#   2. A LENGTH CAP -- shape, not vocabulary. Catches dumps whose wording we
#      have never seen, which is the failure mode a marker list has by
#      construction. Measured: the leak is 1150 chars; the legitimate 403 that
#      names the key's own model entitlement is 229; upstream validation
#      messages are far shorter still.
#
# Then a FINAL RE-CHECK over the serialized result, so a marker hiding in a
# field this code did not know to visit still fails closed.
# ---------------------------------------------------------------------------

#: Internal vocabulary that must never reach an external caller. Lowercase;
#: matched as substrings against the message AFTER the ``litellm.`` prefix is
#: stripped, so an upstream message LiteLLM merely wrapped is still forwarded.
ROUTING_DISCLOSURE_MARKERS: tuple[str, ...] = (
    "litellm",  # names the gateway software
    "model_group",  # internal routing vocabulary
    "fallback",  # the routing table itself
    "tags=",  # the deny-tag mechanism ...
    "tags configuration",  # ... and how it is described
    "deployment",  # Azure-shaped internal topology
    "api_base",  # upstream account URL
    "api_key",
)

#: A message longer than this is a dump, not an explanation. Sits between the
#: measured legitimate maximum (229) and the measured leak (1150), nearer the
#: former: anything in between is far likelier to be a dump than a useful
#: sentence, and the cost of being wrong is a generic message rather than a leak.
MAX_EXTERNAL_ERROR_MESSAGE_CHARS = 400

#: Keys whose string values are error prose. Visited at any depth so both
#: dialects are covered -- OpenAI's {"error":{"message":...}} and Anthropic's
#: {"type":"error","error":{"message":...}} -- without hardcoding either shape.
_MESSAGE_KEYS: frozenset[str] = frozenset(("message", "detail", "error_message"))

#: Strips a leading ``litellm.SomeError:`` so an upstream message that LiteLLM
#: only wrapped survives the marker test on its own merits.
_LITELLM_PREFIX = re.compile(r"^\s*litellm\.\w*(?:Error|Exception)\s*:\s*", re.IGNORECASE)

_GENERIC_CLIENT_ERROR = "Invalid request."
_GENERIC_SERVER_ERROR = "Internal error."


def _generic_message(status: int) -> str:
    return _GENERIC_CLIENT_ERROR if status < 500 else _GENERIC_SERVER_ERROR


def _generic_body(status: int) -> bytes:
    """The replacement body, assembled without a dict literal.

    Concatenation rather than ``json.dumps({...})`` so the module carries no
    mutable collection for this (the LIT002 ratchet), while the message itself
    still goes through ``json.dumps`` -- it is the only part that could ever need
    escaping, and hardcoding the escaped form would let it drift from
    ``_generic_message``.
    """
    return (
        b'{"error": {"message": '
        + json.dumps(_generic_message(status)).encode("utf-8")
        + b', "type": "invalid_request_error"}}'
    )


def sanitize_error_message(message: str, status: int) -> str:
    """One error string's external form."""
    stripped = _LITELLM_PREFIX.sub("", message, count=1).strip()
    if not stripped:
        return _generic_message(status)
    lowered = stripped.lower()
    if any(marker in lowered for marker in ROUTING_DISCLOSURE_MARKERS):
        return _generic_message(status)
    if len(stripped) > MAX_EXTERNAL_ERROR_MESSAGE_CHARS:
        return _generic_message(status)
    return stripped


def _sanitize_tree(root: object, status: int) -> object:
    """Rewrite every error-prose string in a decoded JSON body, in place.

    ITERATIVE, with an explicit stack, rather than the obvious recursion. Two
    reasons, and the CI gate that rejects unignored recursive functions is right
    on both counts:

    1. The input is an UPSTREAM-CONTROLLED error body. Recursion depth would be
       attacker-influenced, and a deeply nested document would raise
       RecursionError from inside a security control -- i.e. fail open unless
       every caller remembered to catch it.
    2. Depth is now bounded by the heap rather than the C stack.

    Mutates in place: the tree comes from ``json.loads`` on bytes we are about to
    discard, so nothing else can observe it, and rebuilding it would allocate a
    second copy of every error body for no benefit.
    """
    # A traversal worklist is the case a mutable collection exists for. The
    # immutable alternative -- rebuilding a tuple per visited node -- makes this
    # O(n^2) in document depth, and the body is upstream-supplied, so depth is
    # attacker-influenced. The list never escapes this function, so nothing else
    # can grow or rewrite it.
    #
    # The suppression must sit ON the offending line -- the checker ignores it
    # anywhere else -- AND must be short enough that the formatter does not split
    # the statement across lines, which moves the comment off the reported line
    # and silently un-suppresses it.
    stack = [root]  # mutable-ok: traversal worklist; a tuple rebuild is O(n^2) in depth
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key in _MESSAGE_KEYS and isinstance(value, str):
                    node[key] = sanitize_error_message(value, status)
                elif isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    stack.append(item)
    return root


def sanitize_error_body(raw: bytes, status: int) -> bytes:
    """One error response body's external form.

    Fails CLOSED in both uncertain cases: a body that does not parse but carries
    internal vocabulary is replaced, and so is a body that still carries it after
    the tree walk -- the latter being the guard against a field this code does
    not know to visit.
    """
    generic = _generic_body(status)

    def _is_dirty(candidate: bytes) -> bool:
        lowered = candidate.decode("utf-8", errors="replace").lower()
        return any(marker in lowered for marker in ROUTING_DISCLOSURE_MARKERS)

    try:
        decoded = json.loads(raw)
    except (ValueError, UnicodeDecodeError, RecursionError):
        # RecursionError is not hypothetical and is NOT a ValueError: CPython's
        # JSON decoder recurses, and this body is upstream-controlled. Letting it
        # escape would take down a security control with an unhandled exception.
        # Not JSON: an HTML error page or a plain-text 502 from something in
        # front of us. Only replaced if it actually carries internal vocabulary.
        return generic if _is_dirty(raw) else raw

    sanitized = json.dumps(_sanitize_tree(decoded, status)).encode("utf-8")
    return generic if _is_dirty(sanitized) else sanitized


def _with_content_length(headers: Iterable[tuple[bytes, bytes]], length: int) -> tuple[tuple[bytes, bytes], ...]:
    """Re-frame a held response start for a body whose length we just changed.

    ``transfer-encoding`` is dropped rather than preserved: the body is fully
    buffered by the time this runs, so it is sent as one complete chunk, and a
    stale chunked declaration alongside a content-length is a framing error that
    presents as a client hang.
    """
    kept = tuple(
        (name, value)
        for name, value in headers
        if name.decode("latin-1").lower() not in ("content-length", "transfer-encoding")
    )
    return kept + ((b"content-length", str(length).encode("latin-1")),)


class ExternalAudienceHeaderMiddleware:
    """
    Pure ASGI (never BaseHTTPMiddleware -- that degrades streaming).

    Applies BOTH external policies: the header allowlist, and the error-body
    sanitizer above.

    Must be registered LAST in proxy_server.py: Starlette makes the last-added
    middleware outermost, and this has to see the final header set produced by
    every inner middleware, CORS included.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _is_external_audience(scope):
            await self.app(scope, receive, send)
            return

        # Only set for a response being held for sanitization. A successful
        # response is forwarded chunk by chunk exactly as before -- buffering one
        # would break streaming, which is most of this proxy's traffic.
        held_start: Message | None = None
        buffered = bytearray()

        async def send_external(message: Message) -> None:
            nonlocal held_start

            if message["type"] == "http.response.start":
                # Emitted exactly once, before any body chunk, so this covers
                # streaming and error responses identically.
                raw_headers: Iterable[tuple[bytes, bytes]] = message.get("headers") or ()
                message["headers"] = apply_external_header_policy(raw_headers)

                if message.get("status", 200) >= 400:
                    # Held, not forwarded: content-length cannot be corrected
                    # once the start message is on the wire, and the sanitized
                    # body is a different length than the original.
                    held_start = message
                    return
                await send(message)
                return

            if message["type"] == "http.response.body" and held_start is not None:
                buffered.extend(message.get("body") or b"")
                if message.get("more_body", False):
                    return

                status = held_start.get("status", 500)
                sanitized = sanitize_error_body(bytes(buffered), status)
                held_start["headers"] = _with_content_length(held_start["headers"], len(sanitized))
                await send(held_start)
                held_start = None
                # Reuse the final body message rather than constructing a new
                # one: it is already the right shape, and this is the last chunk
                # so `more_body` must be False regardless of what it said.
                message["body"] = sanitized
                message["more_body"] = False
                await send(message)
                return

            await send(message)

        await self.app(scope, receive, send_external)
