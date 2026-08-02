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
from collections.abc import Iterable, MutableMapping

from anyio.lowlevel import checkpoint
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


# ---------------------------------------------------------------------------
# SUCCESS BODIES — provider attribution
#
# The error policy above stops us disclosing our own routing. A separate, milder
# disclosure survives on the SUCCESS path: LiteLLM passes several upstream-shaped
# fields through verbatim, and their NAMES identify the cloud we buy inference
# from. Measured live on router.trueward.ai:
#
#   gemini  vertex_ai_grounding_metadata, vertex_ai_url_context_metadata,
#           vertex_ai_safety_results, vertex_ai_citation_metadata  -> Vertex
#   gpt     latency_checkpoint                                     -> Azure
#   claude  "id":"msg_bdrk_01..."                                  -> Bedrock
#
# Every one was EMPTY or purely internal in every measurement: they tell the
# caller nothing and tell them who we buy from, which is the wrong way round.
#
# WHAT IS DELIBERATELY KEPT
#
# ``content_filter_results`` and ``prompt_filter_results`` are Azure-shaped too,
# and they STAY. They tell a caller their content was filtered and on which
# category -- the difference between "rejected, and here is why" and silent
# truncation. Removing them would be the over-sanitization failure the error
# policy is careful to avoid. Their retention is a KNOWING trade: a caller who
# studies field names can still infer Azure from them.
#
# STREAMING WAS NOT TOUCHED, AND THAT WAS WRONG. This comment used to read
# "STREAMING IS NOT TOUCHED, because it does not need to be. Measured: SSE chunks
# carry none of these fields". That measurement was real but NARROW: it covered
# Gemini and Azure on CHAT COMPLETIONS, where it still holds. It was never taken
# on a Bedrock-backed model over ``/v1/responses``, which is the one endpoint the
# leak-check suite never probed. Measured there on 2026-08-02 against
# router.trueward.ai with a real external-party key, streamed, 3 of 12 frames
# (``response.created``, ``response.in_progress``, ``response.completed``):
#
#   "model":"global.anthropic.claude-haiku-4-5-20251001-v1:0"
#
# ``anthropic.<model>-v1:0`` is Bedrock's id format, so the frame names AWS, and
# ``global.`` is our own inference-profile scope. The same request BUFFERED was
# clean, and chat-completions streaming was clean -- which is exactly why the
# narrow measurement read as general. Streamed ``/v1/messages`` leaked the milder
# form of the same thing: ``claude-haiku-4-5-20251001``, the dated snapshot,
# where the caller asked for the alias ``claude-haiku-4-5``.
#
# The model name is therefore rewritten ON THE STREAMING PATH, per chunk, without
# buffering -- see ``SSEModelRewriter``.
#
# THE BEDROCK MESSAGE ID IS REWRITTEN THERE TOO, and leaving it out was a second
# bug in the first version of this fix. The marker table above lists
# ``"id":"msg_bdrk_01..."`` as a measured cloud disclosure, and the BUFFERED path
# has stripped that prefix all along -- so the same streamed ``/v1/messages``
# request named AWS while its buffered twin did not. Measured on 2026-08-02
# against router.trueward.ai, streamed, AFTER the model rewrite was in place:
#
#   "id":"msg_bdrk_01Tvwe7PT19qAfSyY5VLp4Az"
#
# Unlike the model rewrite it needs nothing from the request, so it runs even
# when the caller's alias could not be determined.
#
# Of the three buffered rewrites, that is the only one that applies to chunks:
# nothing has measured a ``vertex_ai_*`` or a ``latency_checkpoint`` inside an
# SSE frame, and this module does not claim things it has not measured.
#
# Also note what ``strip_provider_attribution`` does NOT do: it has no model
# rewrite at all. Buffered bodies are clean because LiteLLM natively echoes the
# request's own model string back, not because anything here enforces it.

#: TOP-LEVEL keys removed from a successful external response. Each was measured
#: EMPTY, so nothing the caller can use is lost.
PROVIDER_ATTRIBUTION_KEYS: tuple[str, ...] = (
    "vertex_ai_grounding_metadata",
    "vertex_ai_url_context_metadata",
    "vertex_ai_safety_results",
    "vertex_ai_citation_metadata",
)

#: NESTED (parent, key) pairs. Listed separately because they are NOT top level
#: and a top-level pop silently misses them — which is exactly what the first
#: version of this patch did with ``latency_checkpoint``. Measured shape:
#:
#:   "usage": {"completion_tokens": 18, ..., "latency_checkpoint": {"engine_tbt_ms": 7, ...}}
#:
#: It is Azure's internal engine timing (TTFT, TBT, pre-inference), useful to
#: whoever operates the deployment and to nobody calling it.
PROVIDER_ATTRIBUTION_NESTED_KEYS: tuple[tuple[str, str], ...] = (("usage", "latency_checkpoint"),)

#: Bedrock mints message ids as ``msg_bdrk_<id>``; native Anthropic mints
#: ``msg_<id>``. Rewriting the prefix makes the response MORE faithful to the
#: dialect it claims to speak, not less. Message ids are not accepted as input on
#: /v1/messages, so there is nothing for a caller to round-trip and break.
_BEDROCK_ID_PREFIX = "msg_bdrk_"
_ANTHROPIC_ID_PREFIX = "msg_"

#: Raw-bytes probes for the fast path, built ONCE at import.
#:
#: This runs on every successful external response, so rebuilding the list per
#: call was both a per-request allocation on the hot path and two mutable-list
#: constructions against the LIT002 ratchet. A tuple over a generator is neither.
_ATTRIBUTION_PROBES: tuple[bytes, ...] = (
    tuple(key.encode("utf-8") for key in PROVIDER_ATTRIBUTION_KEYS)
    + tuple(key.encode("utf-8") for _, key in PROVIDER_ATTRIBUTION_NESTED_KEYS)
    + (_BEDROCK_ID_PREFIX.encode("utf-8"),)
)


def strip_provider_attribution(raw: bytes) -> bytes:
    """One successful response body's external form.

    Returns the input unchanged when there is nothing to remove -- the common
    case, and the one that has to stay cheap.
    """
    if not any(probe in raw for probe in _ATTRIBUTION_PROBES):
        return raw

    try:
        decoded = json.loads(raw)
    except (ValueError, UnicodeDecodeError, RecursionError):
        # Unlike the error path there is nothing dangerous to fail closed over:
        # these are attribution hints, not credentials or routing. Replacing a
        # caller's SUCCESSFUL response because we could not parse it would do
        # more harm than the disclosure it would prevent.
        return raw

    if not isinstance(decoded, dict):
        return raw

    for key in PROVIDER_ATTRIBUTION_KEYS:
        decoded.pop(key, None)

    for parent, key in PROVIDER_ATTRIBUTION_NESTED_KEYS:
        container = decoded.get(parent)
        if isinstance(container, dict):
            container.pop(key, None)

    message_id = decoded.get("id")
    if isinstance(message_id, str) and message_id.startswith(_BEDROCK_ID_PREFIX):
        decoded["id"] = _ANTHROPIC_ID_PREFIX + message_id[len(_BEDROCK_ID_PREFIX) :]

    return json.dumps(decoded).encode("utf-8")


# ---------------------------------------------------------------------------
# THE MODEL NAME, ON THE STREAMING PATH
#
# Rewriting ``model`` needs something the response does not carry: what the
# CALLER asked for. That comes from the request body, so this middleware captures
# it (see ``_capture_request_body``) and replays it to the app untouched.
#
# TWO RULES THIS CODE IS BUILT AROUND, both of which are easy to get wrong:
#
# 1. COMPARE BY EQUALITY, NEVER BY CONTAINMENT. The requested alias is a
#    SUBSTRING of the leaked wire id:
#
#        "claude-haiku-4-5" in "global.anthropic.claude-haiku-4-5-20251001-v1:0"
#
#    is True. Any "does the frame contain the alias?" check therefore passes on
#    the exact leak it is supposed to catch. Nothing here substring-matches a
#    model name: the frame is parsed and the field is compared with ``!=``.
#
# 2. AN ALLOWLIST OF ENVELOPE PATHS, NEVER A RECURSIVE WALK. A completion can
#    legitimately contain the word "model" -- in the caller's own text, in a tool
#    call's arguments, in a JSON document the model was asked to produce.
#    Rewriting every "model" key found by a document walk would silently corrupt
#    the caller's payload. Only the three envelopes that carry the RESPONSE's own
#    model are visited: the document root, ``response`` (/v1/responses events)
#    and ``message`` (Anthropic ``message_start``).
# ---------------------------------------------------------------------------

#: Request methods that can carry a body naming a model.
_MODEL_BEARING_METHODS: frozenset[str] = frozenset(("POST", "PUT", "PATCH"))

#: Content types never captured: file uploads. They can be arbitrarily large and
#: never carry a top-level JSON ``model``, so holding one in memory would buy
#: nothing. Prefix match, so ``multipart/form-data; boundary=...`` is covered.
_UNCAPTURED_CONTENT_TYPES: tuple[str, ...] = ("multipart/", "application/octet-stream")

#: Cap on the request body held in memory to learn the caller's model. Past it
#: the capture is abandoned and the model stays unknown -- which means no rewrite
#: (fail safe). The app still receives every byte either way; only OUR copy stops.
MAX_CAPTURED_REQUEST_BODY_BYTES = 1024 * 1024

#: Cap on the unterminated tail ``SSEModelRewriter`` will hold while waiting for a
#: newline. A real SSE line is a few KB; past this the upstream is not sending
#: line-terminated SSE, so the tail is released verbatim rather than stalling the
#: caller's stream forever.
MAX_PENDING_SSE_LINE_BYTES = 1024 * 1024

#: The ONLY nested envelopes visited, one level below the root. ``response`` is
#: the /v1/responses event envelope (response.created / .in_progress /
#: .completed); ``message`` is Anthropic's ``message_start``. Not a walk.
MODEL_ENVELOPE_KEYS: tuple[str, ...] = ("response", "message")

_SSE_DATA_FIELD = b"data:"
_SSE_DONE = b"[DONE]"


def _may_carry_a_model(scope: Scope) -> bool:
    """True when it is worth capturing this request body to learn its model."""
    method: str = scope.get("method") or ""
    if method.upper() not in _MODEL_BEARING_METHODS:
        return False

    raw_headers: Iterable[tuple[bytes, bytes]] = scope.get("headers") or ()
    for raw_name, raw_value in raw_headers:
        name = raw_name.decode("latin-1").lower()
        if name == "content-type":
            content_type = raw_value.decode("latin-1").strip().lower()
            if content_type.startswith(_UNCAPTURED_CONTENT_TYPES):
                return False
        elif name == "content-length":
            declared = raw_value.decode("latin-1").strip()
            if declared.isdigit() and int(declared) > MAX_CAPTURED_REQUEST_BODY_BYTES:
                return False
    return True


async def _capture_request_body(receive: Receive) -> tuple[bytes | None, Receive]:
    """Read the request body, and hand back a receive that replays it verbatim.

    The standard ASGI receive-replay: the middleware consumes the real receive
    channel, then gives the app a substitute that re-emits the exact messages it
    consumed and, once those run out, falls through to the real channel. Getting
    this wrong does not break streaming, it breaks EVERY request -- an app that
    never sees ``more_body: False`` hangs, and one that sees a synthesized
    message instead of the original loses whatever else the server put in it.

    Replayed messages accumulate into a TUPLE rather than a list: the count is
    the number of ASGI chunks the server split the body into (single digits for
    an LLM request), so the quadratic rebuild is unmeasurable, and it keeps a
    mutable buffer of caller data from existing at all.

    Returns ``None`` for the body -- not a partial one -- unless the whole body
    was read. A truncated body is worse than no body: it would parse as invalid
    JSON at best, and at worst as a DIFFERENT valid document.
    """
    replay: tuple[Message, ...] = ()
    captured = bytearray()
    complete = False

    while True:
        message = await receive()
        replay = replay + (message,)
        if message.get("type") != "http.request":
            # http.disconnect: the client went away mid-body. Nothing to learn.
            break
        captured.extend(message.get("body") or b"")
        if len(captured) > MAX_CAPTURED_REQUEST_BODY_BYTES:
            break
        if not message.get("more_body", False):
            complete = True
            break

    pending = iter(replay)

    async def replaying_receive() -> Message:
        # The checkpoint is NOT decoration, and it must come BEFORE the message
        # is taken off the iterator. A real receive always yields to the event
        # loop, and starlette's Request.is_disconnected() relies on exactly that:
        # it calls receive inside an ALREADY-CANCELLED anyio scope so that a
        # receive with nothing to deliver returns an empty message instead of
        # blocking. A replay that returns without ever awaiting has no
        # cancellation point, so is_disconnected() would consume a real body
        # chunk and throw it away -- silently truncating the request. With the
        # checkpoint first, cancellation lands before anything is consumed and
        # the substitute behaves exactly like the channel it stands in for.
        await checkpoint()
        held = next(pending, None)
        return await receive() if held is None else held

    return (bytes(captured) if complete else None), replaying_receive


def requested_model(raw: bytes | None) -> str | None:
    """The caller's own model string, or None when it cannot be determined.

    None is the fail-safe answer and every uncertain case returns it: no body, a
    body that does not parse, a body that is not an object, a ``model`` that is
    absent, not a string, or blank. A caller's working stream matters more than
    a hint, so "unknown" must mean "change nothing", never "guess".
    """
    if not raw:
        return None
    try:
        decoded: object = json.loads(raw)
    except (ValueError, UnicodeDecodeError, RecursionError):
        return None
    if not isinstance(decoded, dict):
        return None
    model = decoded.get("model")
    if isinstance(model, str) and model.strip():
        return model
    return None


def _envelopes(
    payload: object,
) -> tuple[MutableMapping[str, object], ...]:  # mutable-ok: these ARE the caller's frame, rewritten in place
    """The allowlisted objects in one frame: the root plus MODEL_ENVELOPE_KEYS.

    One level deep, and nothing else -- see rule 2 above for why this is not a
    recursive walk. Shared by both rewrites below so they cannot come to disagree
    about what an envelope is.
    """
    if not isinstance(payload, dict):
        return ()
    candidates = (payload, *(payload.get(key) for key in MODEL_ENVELOPE_KEYS))
    return tuple(item for item in candidates if isinstance(item, dict))


def rewrite_envelope_model(payload: object, alias: str) -> bool:
    """Point ``model`` at the caller's alias in the allowlisted envelopes.

    Returns True when something actually changed, so the caller can leave a frame
    byte-identical when there was nothing to do -- re-serializing every frame
    would churn key order and unicode escaping for no reason.
    """
    changed = False
    for envelope in _envelopes(payload):
        current = envelope.get("model")
        # EQUALITY. `alias in current` is True of the exact leak this closes.
        if isinstance(current, str) and current != alias:
            envelope["model"] = alias
            changed = True
    return changed


def rewrite_envelope_bedrock_id(payload: object) -> bool:
    """Restore the native Anthropic ``id`` shape in the allowlisted envelopes.

    Bedrock mints ``msg_bdrk_<id>``; native Anthropic mints ``msg_<id>``. The
    buffered path has always stripped the infix (see
    ``strip_provider_attribution``); this is the streaming counterpart, and its
    absence meant a streamed ``/v1/messages`` named AWS while the same request
    buffered did not.

    Takes no alias, and deliberately so: unlike the model rewrite it needs
    nothing from the request, so it still runs when the caller's model could not
    be determined. Prefix test, not containment -- an id that merely contains
    the infix somewhere is not Bedrock's shape and is left alone.
    """
    changed = False
    for envelope in _envelopes(payload):
        current = envelope.get("id")
        if isinstance(current, str) and current.startswith(_BEDROCK_ID_PREFIX):
            envelope["id"] = _ANTHROPIC_ID_PREFIX + current[len(_BEDROCK_ID_PREFIX) :]
            changed = True
    return changed


def rewrite_sse_line(line: bytes, alias: str | None) -> bytes:
    """One COMPLETE SSE line's external form, its newline terminator excluded.

    Returns the input unchanged unless it is a ``data:`` line holding a JSON
    object with a model to rewrite -- so comments, ``event:`` / ``retry:`` /
    ``id:`` fields, ``[DONE]``, blank lines and anything that does not parse all
    survive byte for byte. The field's own spacing is preserved too: ``data:{``
    with no space is valid SSE and some clients are strict about round-tripping.
    """
    field = line[:-1] if line.endswith(b"\r") else line
    if not field.startswith(_SSE_DATA_FIELD):
        return line

    value = field[len(_SSE_DATA_FIELD) :]
    payload = value.strip()
    if not payload or payload == _SSE_DONE:
        return line

    try:
        decoded: object = json.loads(payload)
    except (ValueError, UnicodeDecodeError, RecursionError):
        return line

    # Both, and NOT short-circuited: `a() or b()` would skip the id rewrite on
    # any frame whose model already matched. The model half is skipped entirely
    # when the caller's alias is unknown; the id half never needs one.
    changed = alias is not None and rewrite_envelope_model(decoded, alias)
    changed = rewrite_envelope_bedrock_id(decoded) or changed
    if not changed:
        return line

    spacing = value[: len(value) - len(value.lstrip())]
    rendered = _SSE_DATA_FIELD + spacing + json.dumps(decoded).encode("utf-8")
    return rendered + b"\r" if line.endswith(b"\r") else rendered


class SSEModelRewriter:
    """Rewrites ``model`` in an SSE stream AS IT PASSES. Never buffers the stream.

    ``_is_streaming``'s docstring is the constraint: SSE "must never be buffered".
    So this holds the only thing it has to -- the bytes after the last newline it
    has seen -- and emits every complete line immediately. A frame is not the
    unit: SSE is LINE oriented, a ``data:`` field is exactly one line, and lines
    are what this splits on. That makes frame reassembly a non-question: a frame
    split across ASGI chunks is just a line whose newline has not arrived yet,
    and both ``\\n`` and ``\\r\\n`` fall out of it for free (the ``\\r`` rides along
    as the last byte of the line and is put back).

    An ASGI chunk boundary can land anywhere, including mid-key or mid-escape, so
    nothing here may assume a chunk is a frame. The byte-by-byte split is a test
    case for exactly that reason.
    """

    def __init__(self, alias: str | None) -> None:
        self.alias = alias
        self.pending = bytearray()

    def feed(self, chunk: bytes, *, last: bool) -> bytes:
        """The external form of one ASGI body chunk. ``last`` flushes the tail."""
        self.pending.extend(chunk)
        out = bytearray()

        cut = self.pending.rfind(b"\n")
        if cut >= 0:
            complete = bytes(self.pending[: cut + 1])
            del self.pending[: cut + 1]
            # `complete` ends in a newline, so split() leaves a trailing empty
            # element that is not a line; [:-1] drops it.
            for line in complete.split(b"\n")[:-1]:
                out += rewrite_sse_line(line, self.alias) + b"\n"

        if last:
            # The stream ended without a final newline: what is held is a whole
            # line after all, so it gets rewritten rather than merely released.
            out += rewrite_sse_line(bytes(self.pending), self.alias)
            self.pending.clear()
        elif len(self.pending) > MAX_PENDING_SSE_LINE_BYTES:
            # Not line-terminated SSE. Release it verbatim -- an unterminated
            # fragment cannot be parsed, so there is nothing to rewrite, and
            # holding it would stall the caller's stream indefinitely.
            out += self.pending
            self.pending.clear()

        return bytes(out)


def _is_streaming(headers: Iterable[tuple[bytes, bytes]]) -> bool:
    """True for SSE, which must never be buffered."""
    for name, value in headers:
        if name.decode("latin-1").lower() == "content-type":
            return "text/event-stream" in value.decode("latin-1").lower()
    return False


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

    Applies ALL THREE external policies: the header allowlist, the error-body
    sanitizer, and the streamed model-name rewrite.

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

        # The caller's own model string, needed to rewrite a streamed response's
        # `model` back to what they asked for. Read from the REQUEST body, which
        # is then replayed to the app byte for byte. None whenever it cannot be
        # determined, and None means "change nothing" everywhere below.
        app_receive = receive
        alias: str | None = None
        if _may_carry_a_model(scope):
            captured, app_receive = await _capture_request_body(receive)
            alias = requested_model(captured)

        # Only set for a response being held for sanitization. A successful
        # response is forwarded chunk by chunk exactly as before -- buffering one
        # would break streaming, which is most of this proxy's traffic.
        held_start: Message | None = None
        buffered = bytearray()
        # Only set for a streamed SUCCESS whose caller alias we know. It rewrites
        # chunks in flight; it does not hold them. See SSEModelRewriter.
        rewriter: SSEModelRewriter | None = None

        async def send_external(message: Message) -> None:
            nonlocal held_start, rewriter

            if message["type"] == "http.response.start":
                # Emitted exactly once, before any body chunk, so this covers
                # streaming and error responses identically.
                raw_headers: Iterable[tuple[bytes, bytes]] = message.get("headers") or ()
                message["headers"] = apply_external_header_policy(raw_headers)

                status = message.get("status", 200)
                # Held, not forwarded: content-length cannot be corrected once
                # the start message is on the wire, and a rewritten body is a
                # different length than the original.
                #
                # Errors always. Successes only when they are NOT streaming --
                # buffering an SSE response would defeat streaming entirely, and
                # no measurement has put the attribution KEYS inside a chunk. The
                # model NAME is in there (see above), and it is handled without
                # buffering, per chunk, rather than by holding the response.
                if status >= 400 or not _is_streaming(message["headers"]):
                    held_start = message
                    return
                # A streamed success: forwarded immediately, never held. The
                # rewriter exists only when the caller's alias is known -- without
                # it every chunk goes out untouched, which is the fail-safe
                # direction (a hint survives; the caller's stream does not break).
                # Always, even when the alias is unknown: the Bedrock id rewrite
                # needs nothing from the request, so gating the whole rewriter on
                # the alias would leave `msg_bdrk_` shipping on any request whose
                # model could not be read.
                rewriter = SSEModelRewriter(alias)
                await send(message)
                return

            if message["type"] == "http.response.body" and rewriter is not None:
                last = not message.get("more_body", False)
                rewritten = rewriter.feed(message.get("body") or b"", last=last)
                if last:
                    rewriter = None
                elif not rewritten:
                    # Nothing complete yet. An empty non-final body message
                    # carries no information, so it is dropped rather than
                    # written out as a zero-length chunk.
                    return
                message["body"] = rewritten
                await send(message)
                return

            if message["type"] == "http.response.body" and held_start is not None:
                buffered.extend(message.get("body") or b"")
                if message.get("more_body", False):
                    return

                status = held_start.get("status", 500)
                if status >= 400:
                    sanitized = sanitize_error_body(bytes(buffered), status)
                else:
                    sanitized = strip_provider_attribution(bytes(buffered))
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

        await self.app(scope, app_receive, send_external)
