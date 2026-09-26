
from dotenv import load_dotenv

load_dotenv()
import io

# this file is to test litellm/proxy

import asyncio
import logging

import pytest
from fastapi import Request
from fastapi.routing import APIRoute, APIWebSocketRoute
from starlette.datastructures import URL, Headers, QueryParams

import litellm
from litellm.proxy._types import LiteLLMRoutes
from litellm.proxy.auth.auth_utils import get_request_route
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.proxy_server import app

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the desired logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def test_routes_on_litellm_proxy():
    """
    Goal of this test: Test that we have all the critical OpenAI Routes on the Proxy server Fast API router


    this prevents accidentelly deleting /threads, or /batches etc
    """
    # Force-load lazy features so the test sees the full route set. Continue
    # on per-feature import failure — the assertion below still catches
    # missing-route regressions.
    import importlib

    from litellm.proxy._lazy_features import LAZY_FEATURES

    registered_paths = [getattr(r, "path", "") for r in app.routes]
    for feat in LAZY_FEATURES:
        if any(rp.startswith(p) for p in feat.path_prefixes for rp in registered_paths):
            continue
        try:
            module = importlib.import_module(feat.module_path)
            feat.register_fn(app, module)
        except Exception as exc:
            print(f"warning: failed to force-load {feat.name}: {exc}")

    # fastapi 0.137+ stores included routers lazily as _IncludedRouter objects;
    # _iter_included_route_candidates flattens them with prefixes applied.
    try:
        from fastapi.routing import _iter_included_route_candidates

        route_iter = _iter_included_route_candidates(app.routes)
    except ImportError:
        route_iter = app.routes

    _all_routes = []
    for route in route_iter:
        _path_as_str = getattr(route, "path", None)
        if _path_as_str is None:
            continue
        _path_as_str = str(_path_as_str)
        if ":path" in _path_as_str:
            _path_as_str = _path_as_str.replace(":path", "")
        _all_routes.append(_path_as_str)

    print("ALL ROUTES on LiteLLM Proxy:", _all_routes)
    print("\n\n")
    print("ALL OPENAI ROUTES:", LiteLLMRoutes.openai_routes.value)

    for route in LiteLLMRoutes.openai_routes.value:
        # realtime routes - /realtime?model=gpt-4o
        if "realtime" in route:
            assert "/realtime" in _all_routes
        # wildcard patterns like /containers/* - check that base path exists
        elif RouteChecks._is_wildcard_pattern(pattern=route):
            # For wildcard patterns, check that the base path (without * and trailing /) exists
            base_path = route[:-1].rstrip(
                "/"
            )  # Remove the trailing * and any trailing /
            # Check if base path exists (e.g., /containers or /v1/containers)
            assert (
                base_path in _all_routes
            ), f"Wildcard pattern {route} requires base path {base_path} to exist"
        else:
            assert route in _all_routes


@pytest.mark.parametrize("path", ["/v1/responses", "/responses"])
def test_responses_api_registers_websocket_mode(path):
    """FORK PATCH GUARD (rayward-internal/llm-gateway-infra#645, #650).

    Responses WebSocket mode was disabled 2026-08-27 (#214) because every
    non-Vertex leg failed after the upgrade, then re-enabled (#650) once the
    upstream fixes were backported: BerriAI/litellm#34253 (preserve the
    caller's ``api_key`` instead of forcing a bearer token onto Bedrock),
    #33101 (stop leaking ``litellm_params`` into the provider request body),
    and #33463 (apply deployment defaults to websocket frames). Both aliases
    upstream decorates must stay registered, with the HTTPS POST route
    surviving alongside the WebSocket route on each of them.

    A ``-X theirs`` upstream sync could drop these decorators again with no
    conflict. When this test goes red, RE-APPLY the two ``@router.websocket``
    decorators on ``responses_websocket_endpoint``; never flip the assertion.
    See ``.github/fork-patches.txt``.
    """
    responses_routes = [route for route in app.routes if getattr(route, "path", None) == path]

    assert responses_routes, f"{path} is not registered at all"
    assert any(
        isinstance(route, APIRoute) and "POST" in route.methods for route in responses_routes
    ), f"HTTPS POST {path} must remain available"
    assert any(
        isinstance(route, APIWebSocketRoute) for route in responses_routes
    ), f"{path} must register a WebSocket route"


@pytest.mark.parametrize("path", ["/v1/realtime", "/realtime", "/openai/v1/realtime"])
def test_realtime_websocket_refuses_the_upgrade(path):
    """FORK PATCH GUARD (rayward-internal/llm-gateway-infra#646).

    No realtime-capable model is served on this deployment, so
    ``realtime_websocket_endpoint`` closes BEFORE ``accept()`` — uvicorn turns
    that into a definitive ``HTTP 403`` on the handshake, and no ``101`` is ever
    sent. Upstream accepted first and resolved the model afterwards, so a client
    got a handshake that succeeded and a socket that died on its first frame.

    The route deliberately stays REGISTERED: ``LiteLLMRoutes.openai_routes``
    lists the realtime paths and ``test_routes_on_litellm_proxy`` asserts a bare
    ``/realtime`` route exists whenever any realtime path is listed, so removing
    the decorators would make that upstream test lie.

    A ``-X theirs`` sync drops the refusal with no conflict. When this goes red,
    RE-APPLY it; never delete this guard instead. See ``.github/fork-patches.txt``.
    """
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    routes = [
        route
        for route in app.routes
        if isinstance(route, APIWebSocketRoute) and getattr(route, "path", None) == path
    ]
    assert routes, f"{path} must stay registered so LiteLLMRoutes.openai_routes stays honest"

    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect(path) as ws:
            ws.receive_text()
    assert excinfo.value.code == 1008, (
        f"{path} should refuse the upgrade with close code 1008 before accept(); "
        f"got {excinfo.value.code}"
    )


def test_vertex_ai_live_websocket_is_untouched():
    """Blast-radius guard: #646 is about realtime only.

    ``/vertex_ai/live`` is a different pass-through surface and must keep its
    WebSocket route, so a future sweep does not quietly take it with the others.
    """
    assert any(
        isinstance(route, APIWebSocketRoute) and getattr(route, "path", None) == "/vertex_ai/live"
        for route in app.routes
    ), "/vertex_ai/live must remain a registered WebSocket route"


@pytest.mark.parametrize(
    "route,expected",
    [
        # Test exact matches
        ("/chat/completions", True),
        ("/v1/chat/completions", True),
        ("/embeddings", True),
        ("/v1/models", True),
        ("/utils/token_counter", True),
        # Test routes with placeholders
        ("/engines/gpt-4/chat/completions", True),
        ("/openai/deployments/gpt-3.5-turbo/chat/completions", True),
        ("/threads/thread_49EIN5QF32s4mH20M7GFKdlZ", True),
        ("/v1/threads/thread_49EIN5QF32s4mH20M7GFKdlZ", True),
        ("/threads/thread_49EIN5QF32s4mH20M7GFKdlZ/messages", True),
        ("/v1/threads/thread_49EIN5QF32s4mH20M7GFKdlZ/runs", True),
        ("/v1/batches/123456", True),
        # Test non-OpenAI routes
        ("/some/random/route", False),
        ("/v2/chat/completions", False),
        ("/threads/invalid/format", False),
        ("/v1/non_existent_endpoint", False),
        # Bedrock Pass Through Routes
        ("/bedrock/model/cohere.command-r-v1:0/converse", True),
        ("/vertex-ai/model/text-embedding-004/embeddings", True),
        # LiteLLM native RAG routes
        ("/rag/ingest", True),
        ("/v1/rag/ingest", True),
        ("/rag/query", True),
        ("/v1/rag/query", True),
    ],
)
def test_is_llm_api_route(route: str, expected: bool):
    assert RouteChecks.is_llm_api_route(route) == expected


# Test-case for routes that are similar but should return False
@pytest.mark.parametrize(
    "route",
    [
        "/v1/threads/thread_id/invalid",
        "/threads/thread_id/invalid",
        "/v1/batches/123/invalid",
        "/engines/model/invalid/completions",
    ],
)
def test_is_llm_api_route_similar_but_false(route: str):
    assert RouteChecks.is_llm_api_route(route) is False


def test_anthropic_api_routes():
    # allow non proxy admins to call anthropic api routes
    assert RouteChecks.is_llm_api_route(route="/v1/messages") is True


def create_request(path: str, base_url: str = "http://testserver") -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "server": ("testserver", 80),
            "path": path,
            "query_string": b"",
            "headers": Headers().raw,
            "client": ("testclient", 50000),
            "root_path": URL(base_url).path,
        }
    )


def test_get_request_route_with_base_url():
    request = create_request(
        path="/genai/chat/completions", base_url="http://testserver/genai"
    )
    result = get_request_route(request)
    assert result == "/chat/completions"


def test_get_request_route_without_base_url():
    request = create_request("/chat/completions")
    result = get_request_route(request)
    assert result == "/chat/completions"


def test_get_request_route_with_nested_path():
    request = create_request(path="/embeddings", base_url="http://testserver/ishaan")
    result = get_request_route(request)
    assert result == "/embeddings"


def test_get_request_route_with_query_params():
    request = create_request(path="/genai/test", base_url="http://testserver/genai")
    request.scope["query_string"] = b"param=value"
    result = get_request_route(request)
    assert result == "/test"


def test_get_request_route_with_base_url_not_at_start():
    request = create_request("/api/genai/test")
    result = get_request_route(request)
    assert result == "/api/genai/test"


def _create_request_with_host_header(path: str, host_header: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "server": ("localhost", 4000),
            "path": path,
            "query_string": b"",
            "headers": [(b"host", host_header.encode())],
            "client": ("127.0.0.1", 50000),
            "root_path": "",
        }
    )


@pytest.mark.parametrize(
    "host_header",
    [
        "localhost/?x=1",
        "localhost:4000/?x=1",
        "localhost/#test",
        "localhost:4000/#test",
    ],
)
def test_get_request_route_not_bypassed_by_malformed_host(host_header: str):
    for protected_path in [
        "/health",
        "/user/new",
        "/key/generate",
        "/get/internal_user_settings",
    ]:
        request = _create_request_with_host_header(
            path=protected_path, host_header=host_header
        )
        result = get_request_route(request)
        assert (
            result == protected_path
        ), f"Host: {host_header!r} caused route {protected_path!r} to resolve as {result!r}"


# ---------------------------------------------------------------------------
# Regression tests for variant call sites that previously read request.url.path
# (Host-derived) instead of the ASGI scope path. Each test sends a Host header
# crafted to collapse url.path to a substring the call site's decision logic
# would match on, while scope["path"] is the real (unmatching) route.
# ---------------------------------------------------------------------------

_BYPASS_HOSTS = [
    "localhost/?x=1",
    "localhost:4000/?x=1",
    "localhost/#test",
    "localhost:4000/#test",
]


def _is_assistants(req):
    return RouteChecks._is_assistants_api_request(req)


def _metadata_var_name(req):
    from litellm.proxy.litellm_pre_call_utils import _get_metadata_variable_name

    return _get_metadata_variable_name(req)


def _vector_store_id_in_path(req):
    from litellm.proxy.common_utils.http_parsing_utils import (
        _add_vector_store_id_from_path,
    )

    data: dict = {}
    _add_vector_store_id_from_path(request_data=data, request=req)
    return "vector_store_id" in data


# (label, scope_path, host_suffix_template, predicate, expected) — host_suffix_template
# receives the host_header via %s substitution. The predicate is invoked on a Request
# whose scope["path"] is scope_path and whose Host header is the formatted suffix.
#
# The MCP entries (well_known_mcp_bypass, pkce_token_suffix) call
# get_request_route directly rather than the surrounding production handler
# (MCPRequestHandler.process_mcp_request / _mcp_oauth_user_api_key_auth) —
# those handlers require an ASGI scope plus MCP state to invoke, and the call
# sites do nothing with the path except feed it to this helper. The helper-
# level assertion is the relevant signal.
_CALL_SITES = [
    ("assistants_classification", "/key/generate", "%s/thread", _is_assistants, False),
    (
        "metadata_variable_name",
        "/chat/completions",
        "%s/thread",
        _metadata_var_name,
        "metadata",
    ),
    (
        "vector_store_id_extraction",
        "/key/generate",
        "%s/vector_stores/x/files",
        _vector_store_id_in_path,
        False,
    ),
    (
        "well_known_mcp_bypass",
        "/mcp/tools/call",
        "/.well-known/%s",
        lambda r: get_request_route(r).startswith("/.well-known/"),
        False,
    ),
    (
        "pkce_token_suffix",
        "/mcp/server-id/token",
        "%s",
        lambda r: get_request_route(r).rstrip("/").lower().endswith("/token"),
        True,
    ),
    (
        "spend_logs_v2_classification",
        "/spend/logs",
        "%s/spend/logs/v2",
        lambda r: "/spend/logs/v2" in get_request_route(r),
        False,
    ),
    ("health_route_echo", "/test", "%s", lambda r: get_request_route(r), "/test"),
]


@pytest.mark.parametrize("host_header", _BYPASS_HOSTS)
@pytest.mark.parametrize(
    "label,scope_path,host_suffix_template,predicate,expected",
    _CALL_SITES,
    ids=[c[0] for c in _CALL_SITES],
)
def test_call_site_uses_scope_path(
    label, scope_path, host_suffix_template, predicate, expected, host_header
):
    """Each call site that previously read request.url.path must now make its
    decision against scope["path"]. The Host header is crafted so url.path
    would resolve to a value that flips the decision under the old code."""
    request = _create_request_with_host_header(
        path=scope_path, host_header=host_suffix_template % host_header
    )
    assert predicate(request) == expected
