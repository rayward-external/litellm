"""
Tests for ExternalAudienceHeaderMiddleware (RAYWARD FORK PATCH).

These tests are the guard that makes the patch survive upstream rebases: if the
middleware module or its registration in proxy_server.py is dropped, these fail
loudly instead of silently re-exposing gateway-identifying headers to external
callers.

What is asserted:
  * external  -> every x-litellm-* header is gone
  * external  -> every llm_provider-* header is gone
  * external  -> bare x-ratelimit-* (OUR purchased upstream quota) is gone
  * external  -> the three usage headers are RENAMED, values byte-identical
  * internal (header absent) -> byte-identical to unpatched behaviour
  * a forged client header cannot un-hide anything
  * streaming responses are covered
  * error responses (4xx/5xx) are covered
  * the registration in proxy_server.py is present and outermost
"""

import re
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from litellm.proxy.middleware.external_audience_middleware import (
    AUDIENCE_REQUEST_HEADER,
    ExternalAudienceHeaderMiddleware,
)

# A response header set modelled on what an external caller actually received
# through the LB on 2026-07-31, after the url-map had already stripped its 50.
LEAKY_HEADERS = {
    "x-litellm-call-id": "cid-123",
    "x-litellm-model-id": "azure-eastus-gpt-5",
    "x-litellm-model-api-base": "https://rayward-aoai.openai.azure.com",
    "x-litellm-version": "1.99.0",
    "x-litellm-attempted-retries": "1",
    "x-litellm-attempted-fallbacks": "0",
    "x-litellm-model-group": "gpt-5",
    "x-litellm-response-duration-ms": "812.5",
    "x-litellm-overhead-duration-ms": "11.2",
    "x-litellm-callback-duration-ms": "3.1",
    "x-litellm-key-tpm-limit": "None",
    "x-litellm-key-rpm-limit": "None",
    "llm_provider-x-request-id": "req_upstream_abc",
    "llm_provider-x-ratelimit-remaining-requests": "499",
    "llm_provider-x-ratelimit-remaining-tokens": "499910",
    "llm_provider-strict-transport-security": "max-age=31536000",
    "llm_provider-x-accel-buffering": "no",
    "llm_provider-content-type": "application/json",
    "x-ratelimit-limit-requests": "500",
    "x-ratelimit-remaining-requests": "499",
    "x-ratelimit-limit-tokens": "500000",
    "x-ratelimit-remaining-tokens": "499910",
    # Renamed, not suppressed: the caller's own usage data.
    "x-litellm-response-cost": "0.00031415",
    "x-litellm-key-spend": "12.5",
    "x-litellm-key-max-budget": "100.0",
    # Neutral header that must pass through untouched.
    "x-request-id": "client-supplied-trace",
}

USAGE_RENAMES = {
    "x-usage-cost": "0.00031415",
    "x-usage-spend": "12.5",
    "x-usage-budget": "100.0",
}

EXTERNAL = {AUDIENCE_REQUEST_HEADER: "external"}


async def _ok(request):
    return JSONResponse({"ok": True}, headers=LEAKY_HEADERS)


async def _stream(request):
    async def body():
        for chunk in (b'data: {"a":1}\n\n', b"data: [DONE]\n\n"):
            yield chunk

    return StreamingResponse(
        body(), media_type="text/event-stream", headers=LEAKY_HEADERS
    )


async def _client_error(request):
    return JSONResponse({"error": "bad request"}, status_code=400, headers=LEAKY_HEADERS)


async def _server_error(request):
    return JSONResponse({"error": "upstream blew up"}, status_code=500, headers=LEAKY_HEADERS)


def _client():
    app = Starlette(
        routes=[
            Route("/ok", _ok),
            Route("/stream", _stream),
            Route("/400", _client_error),
            Route("/500", _server_error),
        ]
    )
    app.add_middleware(ExternalAudienceHeaderMiddleware)
    return TestClient(app)


def _assert_nothing_disclosed(headers):
    """No header may name our gateway, prove we proxy, or state our quota."""
    names = [name.lower() for name in headers.keys()]
    assert not [n for n in names if "litellm" in n], names
    assert not [n for n in names if n.startswith("llm_provider-")], names
    assert not [n for n in names if n.startswith("x-ratelimit-")], names


# --------------------------------------------------------------------------
# The patch is only sound if it is pure ASGI and sees every response.
# --------------------------------------------------------------------------


def test_is_pure_asgi_not_base_http_middleware():
    """BaseHTTPMiddleware degrades streaming; this must be pure ASGI."""
    assert not issubclass(ExternalAudienceHeaderMiddleware, BaseHTTPMiddleware)
    assert "__call__" in ExternalAudienceHeaderMiddleware.__dict__


def test_audience_header_name_is_vendor_neutral():
    """AGENTS.md: fork patches use neutral header names."""
    assert "rayward" not in AUDIENCE_REQUEST_HEADER.lower()
    assert "litellm" not in AUDIENCE_REQUEST_HEADER.lower()


# --------------------------------------------------------------------------
# External: suppression
# --------------------------------------------------------------------------


def test_external_suppresses_x_litellm_headers():
    resp = _client().get("/ok", headers=EXTERNAL)
    for name in LEAKY_HEADERS:
        if name.startswith("x-litellm-"):
            assert name not in resp.headers, name


def test_external_suppresses_llm_provider_headers():
    resp = _client().get("/ok", headers=EXTERNAL)
    for name in LEAKY_HEADERS:
        if name.startswith("llm_provider-"):
            assert name not in resp.headers, name


def test_external_suppresses_bare_upstream_quota_headers():
    """Bare x-ratelimit-* is OUR purchased deployment quota, not the caller's."""
    resp = _client().get("/ok", headers=EXTERNAL)
    for name in ("x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
                 "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens"):
        assert name not in resp.headers, name


def test_external_discloses_nothing_at_all():
    resp = _client().get("/ok", headers=EXTERNAL)
    _assert_nothing_disclosed(resp.headers)


# --------------------------------------------------------------------------
# External: the three usage headers are renamed, values intact
# --------------------------------------------------------------------------


@pytest.mark.parametrize("neutral_name,expected_value", sorted(USAGE_RENAMES.items()))
def test_external_renames_usage_headers_with_values_intact(neutral_name, expected_value):
    resp = _client().get("/ok", headers=EXTERNAL)
    assert resp.headers[neutral_name] == expected_value


def test_external_keeps_neutral_headers_untouched():
    resp = _client().get("/ok", headers=EXTERNAL)
    assert resp.headers["x-request-id"] == "client-supplied-trace"


def test_external_rewrites_cors_expose_header_list():
    """CORS advertises literal x-litellm-* names; that is a disclosure too."""
    async def _with_expose(request):
        return JSONResponse(
            {},
            headers={
                "access-control-expose-headers": "x-litellm-semantic-filter,x-litellm-version"
            },
        )

    app2 = Starlette(routes=[Route("/e", _with_expose)])
    app2.add_middleware(ExternalAudienceHeaderMiddleware)
    resp = TestClient(app2).get("/e", headers=EXTERNAL)
    exposed = resp.headers["access-control-expose-headers"]
    assert "litellm" not in exposed
    assert exposed == "x-usage-cost, x-usage-spend, x-usage-budget"


# --------------------------------------------------------------------------
# Internal: byte-identical to unpatched behaviour (SAFE FAILURE DIRECTION)
# --------------------------------------------------------------------------


def test_internal_is_unchanged_when_header_absent():
    resp = _client().get("/ok")
    for name, value in LEAKY_HEADERS.items():
        assert resp.headers[name] == value, name
    for neutral_name in USAGE_RENAMES:
        assert neutral_name not in resp.headers


@pytest.mark.parametrize(
    "audience_value",
    ["internal", "", "EXTERNALS", "not-external", "externally", "unknown"],
)
def test_only_explicit_external_enables_suppression(audience_value):
    """
    Absence OR any non-"external" value means internal/full headers. If this were
    inverted, a failure to inject the header would silently un-hide everything.
    """
    resp = _client().get("/ok", headers={AUDIENCE_REQUEST_HEADER: audience_value})
    assert resp.headers["x-litellm-call-id"] == "cid-123"
    assert "x-usage-cost" not in resp.headers


# --------------------------------------------------------------------------
# A forged client header cannot un-hide anything
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forged_value",
    [
        "internal,external",   # GCP custom_request_headers ADDS, so ours is appended
        "external,internal",
        "internal, external",
        "spoofed external",
    ],
)
def test_client_cannot_unhide_by_forging_the_audience_header(forged_value):
    """Ours still appears in the value, so suppression stays ON."""
    resp = _client().get("/ok", headers={AUDIENCE_REQUEST_HEADER: forged_value})
    _assert_nothing_disclosed(resp.headers)
    assert resp.headers["x-usage-cost"] == "0.00031415"


def test_duplicate_audience_header_entries_still_external():
    """Two separate entries rather than one comma-joined value."""
    resp = _client().get(
        "/ok",
        headers=[(AUDIENCE_REQUEST_HEADER, "internal"), (AUDIENCE_REQUEST_HEADER, "external")],
    )
    _assert_nothing_disclosed(resp.headers)


def test_forging_external_only_suppresses_the_forgers_own_headers():
    """Worst case of a forged header: the client hides its own data. Harmless."""
    resp = _client().get("/ok", headers=EXTERNAL)
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


# --------------------------------------------------------------------------
# Streaming — the most common path. If ContextVars/state did not survive into
# the streaming task, the patch would silently do nothing here.
# --------------------------------------------------------------------------


def test_streaming_external_discloses_nothing():
    with _client().stream("GET", "/stream", headers=EXTERNAL) as resp:
        _assert_nothing_disclosed(resp.headers)
        assert resp.headers["x-usage-cost"] == "0.00031415"
        body = b"".join(resp.iter_bytes())
    assert b"[DONE]" in body


def test_streaming_internal_is_unchanged():
    with _client().stream("GET", "/stream") as resp:
        assert resp.headers["x-litellm-call-id"] == "cid-123"
        assert resp.headers["x-litellm-model-api-base"].startswith("https://")
        body = b"".join(resp.iter_bytes())
    assert b"[DONE]" in body


# --------------------------------------------------------------------------
# Error paths — a 4xx/5xx must not leak what a 200 hides.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path,status", [("/400", 400), ("/500", 500)])
def test_error_responses_disclose_nothing_to_external(path, status):
    resp = _client().get(path, headers=EXTERNAL)
    assert resp.status_code == status
    _assert_nothing_disclosed(resp.headers)


@pytest.mark.parametrize("path", ["/400", "/500"])
def test_error_responses_unchanged_for_internal(path):
    resp = _client().get(path)
    assert resp.headers["x-litellm-call-id"] == "cid-123"


def test_unhandled_exception_handler_response_is_covered():
    """An exception handler's response is built inside the middleware stack."""

    async def _boom(request):
        raise ValueError("kaboom")

    async def _handler(request, exc):
        return JSONResponse({"error": "internal"}, status_code=500, headers=LEAKY_HEADERS)

    app = Starlette(
        routes=[Route("/boom", _boom)], exception_handlers={ValueError: _handler}
    )
    app.add_middleware(ExternalAudienceHeaderMiddleware)
    resp = TestClient(app).get("/boom", headers=EXTERNAL)
    assert resp.status_code == 500
    _assert_nothing_disclosed(resp.headers)


# --------------------------------------------------------------------------
# Rebase guard: the registration itself must survive.
# --------------------------------------------------------------------------


def _proxy_server_source() -> str:
    import litellm

    path = Path(litellm.__file__).parent / "proxy" / "proxy_server.py"
    return path.read_text(encoding="utf-8")


def test_middleware_is_registered_in_proxy_server():
    source = _proxy_server_source()
    assert "from litellm.proxy.middleware.external_audience_middleware import" in source
    assert "app.add_middleware(ExternalAudienceHeaderMiddleware)" in source


def test_middleware_is_registered_last_so_it_is_outermost():
    """
    Starlette makes the last-added middleware outermost. This one must be
    outermost to observe the final header set from every inner middleware.
    """
    source = _proxy_server_source()
    registrations = re.findall(r"^app\.add_middleware\(\s*\n?\s*(\w+)", source, re.M)
    assert registrations, "no app.add_middleware(...) calls found in proxy_server.py"
    assert registrations[-1] == "ExternalAudienceHeaderMiddleware", registrations


def test_middleware_is_outermost_on_the_real_proxy_app():
    """
    Ground truth, not a source-text guard: middlewares can also be registered from
    helper functions (attach_lazy_features adds one), which a regex over
    proxy_server.py cannot see. Starlette's user_middleware[0] is the outermost.
    """
    from litellm.proxy.proxy_server import app

    registered = [m.cls.__name__ for m in app.user_middleware]
    assert registered[0] == "ExternalAudienceHeaderMiddleware", registered
