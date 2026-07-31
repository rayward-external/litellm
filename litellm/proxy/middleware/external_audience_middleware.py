"""
RAYWARD FORK PATCH — minimal response-header exposure for external callers.

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


class ExternalAudienceHeaderMiddleware:
    """
    Pure ASGI (never BaseHTTPMiddleware -- that degrades streaming).

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

        async def send_with_minimal_headers(message: Message) -> None:
            # "http.response.start" is emitted exactly once, before any body
            # chunk, so this covers streaming and error responses identically.
            if message["type"] == "http.response.start":
                raw_headers: Iterable[tuple[bytes, bytes]] = message.get("headers") or ()
                message["headers"] = apply_external_header_policy(raw_headers)
            await send(message)

        await self.app(scope, receive, send_with_minimal_headers)
