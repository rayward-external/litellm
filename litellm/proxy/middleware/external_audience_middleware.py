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

from typing import Dict, Iterable, List, Tuple

from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: Request header injected by the EXTERNAL load balancer's backend service.
#: Deliberately vendor-neutral -- AGENTS.md requires neutral header names in fork
#: patches, so it must never contain "rayward".
AUDIENCE_REQUEST_HEADER = "x-gateway-audience"
EXTERNAL_AUDIENCE = "external"

#: Renamed, not suppressed: this is the caller's OWN usage data, which they
#: legitimately need. The neutral names disclose nothing about the gateway.
#: Applied BEFORE the prefix check below, which would otherwise drop all three.
RENAMED_HEADERS: Dict[str, str] = {
    "x-litellm-response-cost": "x-usage-cost",
    "x-litellm-key-spend": "x-usage-spend",
    "x-litellm-key-max-budget": "x-usage-budget",
}

#: Names our gateway software ("x-litellm-") and proves we proxy, leaking the
#: upstream's own headers wholesale ("llm_provider-").
GATEWAY_PREFIXES: Tuple[str, ...] = ("x-litellm-", "llm_provider-")

#: Bare quota headers promoted by ``get_response_headers()``. These carry OUR
#: purchased deployment quota (e.g. Azure 500 rpm / 500k tpm) -- not the caller's
#: key limit -- so they are a capacity disclosure, and misleading as a client
#: backoff signal. The caller's real per-key counters are ``x-litellm-key-*`` and
#: are suppressed by GATEWAY_PREFIXES regardless.
UPSTREAM_QUOTA_PREFIXES: Tuple[str, ...] = ("x-ratelimit-",)

SUPPRESSED_PREFIXES: Tuple[str, ...] = GATEWAY_PREFIXES + UPSTREAM_QUOTA_PREFIXES

#: CORS advertises LITELLM_UI_ALLOW_HEADERS here, i.e. literal "x-litellm-*" names.
#: That is a disclosure even once the headers themselves are gone, so on the
#: external path the list is re-pointed at the neutral usage headers.
_EXPOSE_HEADERS_NAME = "access-control-expose-headers"
_EXTERNAL_EXPOSED_HEADERS = ", ".join(RENAMED_HEADERS.values()).encode("latin-1")


def _is_external_audience(scope: Scope) -> bool:
    """
    True only if the audience header explicitly says "external".

    GCP ``custom_request_headers`` ADDS a header rather than replacing it, so when a
    client also sends one the value can arrive duplicated (two entries) or
    comma-joined into one. Every occurrence is checked and every token within it.
    """
    raw_headers: Iterable[Tuple[bytes, bytes]] = scope.get("headers") or []
    for raw_name, raw_value in raw_headers:
        if raw_name.decode("latin-1").lower() != AUDIENCE_REQUEST_HEADER:
            continue
        for token in raw_value.decode("latin-1").replace(",", " ").split():
            if token.lower() == EXTERNAL_AUDIENCE:
                return True
    return False


def apply_external_header_policy(
    headers: Iterable[Tuple[bytes, bytes]],
) -> List[Tuple[bytes, bytes]]:
    """Rewrite one response's raw ASGI header list for an external caller."""
    filtered: List[Tuple[bytes, bytes]] = []
    for raw_name, raw_value in headers:
        name = raw_name.decode("latin-1").lower()

        renamed = RENAMED_HEADERS.get(name)
        if renamed is not None:
            filtered.append((renamed.encode("latin-1"), raw_value))
            continue

        if name.startswith(SUPPRESSED_PREFIXES):
            continue

        if name == _EXPOSE_HEADERS_NAME:
            filtered.append((raw_name, _EXTERNAL_EXPOSED_HEADERS))
            continue

        filtered.append((raw_name, raw_value))
    return filtered


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
                raw_headers: Iterable[Tuple[bytes, bytes]] = message.get("headers") or []
                message["headers"] = apply_external_header_policy(raw_headers)
            await send(message)

        await self.app(scope, receive, send_with_minimal_headers)
