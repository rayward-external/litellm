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
  * streamed success bodies never publish an upstream wire model id (#487)
"""

import asyncio
import json
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
    MAX_CAPTURED_REQUEST_BODY_BYTES,
    ExternalAudienceHeaderMiddleware,
    SSEModelRewriter,
    rewrite_envelope_bedrock_id,
    rewrite_envelope_model,
    rewrite_sse_line,
)

#: The id Bedrock actually minted for a streamed /v1/messages call through
#: router.trueward.ai on 2026-08-02, and the native Anthropic shape it must be
#: republished as. Real rather than invented: the first version of this fix used
#: a placeholder `msg_1` here, and because a placeholder cannot carry the Bedrock
#: prefix, no test in this file could observe that the id was still naming AWS.
BEDROCK_MESSAGE_ID = "msg_bdrk_01Tvwe7PT19qAfSyY5VLp4Az"
NATIVE_MESSAGE_ID = "msg_01Tvwe7PT19qAfSyY5VLp4Az"

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

    return StreamingResponse(body(), media_type="text/event-stream", headers=LEAKY_HEADERS)


async def _client_error(request):
    return JSONResponse({"error": "bad request"}, status_code=400, headers=LEAKY_HEADERS)


async def _server_error(request):
    return JSONResponse({"error": "upstream blew up"}, status_code=500, headers=LEAKY_HEADERS)


def _echo_route(headers):
    """A route that returns exactly the given response headers."""

    async def route(request):
        return JSONResponse({"ok": True}, headers=headers)

    return route


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
    for name in (
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
    ):
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
    """Standard headers survive; x-request-id deliberately does not.

    Under the prefix denylist this asserted x-request-id passes through, on the
    reasoning that a correlation id says nothing about us. That holds for one we
    mint and not for one we forward: the pass-through path copies the upstream's
    response headers verbatim, and OpenAI sends its own x-request-id, so keeping
    the name means handing an external caller a VENDOR's request id. The
    allowlist drops it, and the caller loses nothing they can act on.
    """
    resp = _client().get("/ok", headers=EXTERNAL)
    assert resp.headers["content-type"].startswith("application/json")
    assert "x-request-id" not in resp.headers


def test_external_rewrites_cors_expose_header_list():
    """CORS advertises literal x-litellm-* names; that is a disclosure too."""

    async def _with_expose(request):
        return JSONResponse(
            {},
            headers={"access-control-expose-headers": "x-litellm-semantic-filter,x-litellm-version"},
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
        "internal,external",  # GCP custom_request_headers ADDS, so ours is appended
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

    app = Starlette(routes=[Route("/boom", _boom)], exception_handlers={ValueError: _handler})
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


# ── the pass-through hole the prefix denylist could not see ─────────────────


def test_external_drops_bare_upstream_headers():
    """The finding that turned this patch from a denylist into an allowlist.

    HttpPassThroughEndpointHelpers.get_response_headers() forwards the upstream's
    response headers VERBATIM and UN-PREFIXED, excluding only seven framing
    names. None of these carries `llm_provider-` or `x-litellm-`, so a prefix
    match saw nothing to do and every one of them reached the caller.
    """
    leaked = {
        "openai-version": "2020-10-01",
        "openai-organization": "rayward-ai",
        "openai-processing-ms": "412",
        "anthropic-organization-id": "b86e8a2d-ad5a-4d86-8432-d852a7c2fb39",
        "anthropic-ratelimit-tokens-limit": "12000000",
        "request-id": "req_011CdYdGULDUYXM2Z3DrVKkJ",
        "cf-ray": "a23608cf5daa4556-DFW",
        "x-ms-region": "East US",
        "azureml-model-session": "d123",
        "apim-request-id": "0f2f0e2c-1111-2222-3333-444455556666",
        "x-baseten-request-id": "bt-9",
        "fireworks-server-processing-time": "0.41",
    }
    app = Starlette(routes=[Route("/pt", _echo_route(leaked))])
    app.add_middleware(ExternalAudienceHeaderMiddleware)

    resp = TestClient(app).get("/pt", headers=EXTERNAL)

    for name in leaked:
        assert name not in resp.headers, f"{name} reached an external caller"


def test_external_upstream_cannot_collide_with_the_neutral_usage_headers():
    """An upstream sending x-usage-cost must not join ours into one value.

    Duplicate response headers are commonly comma-joined by clients, which turns
    a number into "999, 0.004" and lets the upstream obscure — or spoof — the
    caller's real spend. The gateway's value wins because the rename runs first
    and the upstream copy then fails the allowlist.
    """
    app = Starlette(
        routes=[
            Route(
                "/pt",
                _echo_route(
                    {
                        "x-usage-cost": "999",
                        "x-usage-spend": "999",
                        "x-usage-budget": "999",
                        "x-litellm-response-cost": "0.004",
                        "x-litellm-key-spend": "1.25",
                        "x-litellm-key-max-budget": "50.0",
                    }
                ),
            )
        ]
    )
    app.add_middleware(ExternalAudienceHeaderMiddleware)

    resp = TestClient(app).get("/pt", headers=EXTERNAL)

    assert resp.headers.get_list("x-usage-cost") == ["0.004"]
    assert resp.headers.get_list("x-usage-spend") == ["1.25"]
    assert resp.headers.get_list("x-usage-budget") == ["50.0"]


def test_external_keeps_the_websocket_handshake():
    """A 101 stripped of its handshake is rejected by every compliant client."""
    handshake = {
        "upgrade": "websocket",
        "sec-websocket-accept": "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=",
        "sec-websocket-protocol": "realtime",
        "sec-websocket-extensions": "permessage-deflate",
        "sec-websocket-version": "13",
    }
    app = Starlette(routes=[Route("/ws", _echo_route({**handshake, "x-litellm-version": "1.95.0"}))])
    app.add_middleware(ExternalAudienceHeaderMiddleware)

    resp = TestClient(app).get("/ws", headers=EXTERNAL)

    for name, value in handshake.items():
        assert resp.headers[name] == value
    assert "x-litellm-version" not in resp.headers


# ══════════════════════════════════════════════════════════════════════════════
# STREAMED SUCCESS BODIES — the model name (#487)
#
# Measured live 2026-08-02 against router.trueward.ai with a real external-party
# key. STREAMED /v1/responses on a Bedrock-backed model published the upstream
# wire id at `response.model` on 3 of 12 frames (response.created,
# response.in_progress, response.completed):
#
#     "model":"global.anthropic.claude-haiku-4-5-20251001-v1:0"
#
# `anthropic.<model>-v1:0` is Bedrock's id format and `global.` is our inference
# profile scope, so the frame names both the vendor and our own deployment
# topology. Buffered /v1/responses was CLEAN; chat-completions streaming was
# CLEAN. Streamed /v1/messages echoed the dated snapshot instead of the alias.
#
# THE SUBSTRING TRAP, which these tests exist to not fall into:
#
#     "claude-haiku-4-5" in "global.anthropic.claude-haiku-4-5-20251001-v1:0"
#
# is True. A `contains` assertion therefore PASSES on the exact leak. Every
# assertion below parses the frame and compares the model field by EQUALITY.
# ══════════════════════════════════════════════════════════════════════════════

REQUESTED_ALIAS = "claude-haiku-4-5"
#: Verbatim from the 2026-08-02 measurement.
BEDROCK_WIRE_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
#: Leak 2: milder, same cause — the dated snapshot rather than the alias.
DATED_SNAPSHOT = "claude-haiku-4-5-20251001"

_RESPONSES_FRAMES = (
    b'data: {"type":"response.created","sequence_number":0,"response":'
    b'{"id":"resp_a","object":"response","model":"' + BEDROCK_WIRE_ID.encode() + b'","status":"in_progress"}}\n\n',
    b'data: {"type":"response.output_text.delta","sequence_number":1,"delta":"hi"}\n\n',
    b'data: {"type":"response.completed","sequence_number":2,"response":'
    b'{"id":"resp_a","object":"response","model":"' + BEDROCK_WIRE_ID.encode() + b'","status":"completed"}}\n\n',
)

_MESSAGES_FRAMES = (
    b"event: message_start\n"
    b'data: {"type":"message_start","message":{"id":"' + BEDROCK_MESSAGE_ID.encode() + b'",'
    b'"type":"message","role":"assistant",'
    b'"model":"' + DATED_SNAPSHOT.encode() + b'","content":[]}}\n\n',
    b"event: content_block_delta\n"
    b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n',
)


def _sse_client(chunks, *, seen_bodies=None, path="/v1/responses"):
    """A POST route that streams `chunks` back verbatim as text/event-stream."""

    async def route(request):
        if seen_bodies is not None:
            seen_bodies.append(await request.body())

        async def body():
            for chunk in chunks:
                yield chunk

        return StreamingResponse(body(), media_type="text/event-stream", headers=LEAKY_HEADERS)

    app = Starlette(routes=[Route(path, route, methods=["POST"])])
    app.add_middleware(ExternalAudienceHeaderMiddleware)
    return TestClient(app)


def _post_stream(client, payload, *, path="/v1/responses", headers=EXTERNAL):
    """POST a JSON body and return the raw SSE bytes the caller receives."""
    kwargs = {"json": payload} if isinstance(payload, (dict, list)) else {"content": payload}
    request_headers = {"content-type": "application/json", **headers}
    with client.stream("POST", path, headers=request_headers, **kwargs) as resp:
        assert resp.status_code == 200
        return b"".join(resp.iter_bytes())


def _data_frames(raw: bytes):
    """Every `data:` payload in an SSE stream, PARSED — never substring-matched."""
    frames = []
    for line in raw.split(b"\n"):
        field = line.rstrip(b"\r")
        if not field.startswith(b"data:"):
            continue
        payload = field[len(b"data:") :].strip()
        if not payload or payload == b"[DONE]":
            continue
        frames.append(json.loads(payload))
    return frames


# ── leak 1: /v1/responses streamed, `response.model` ─────────────────────────


def test_streamed_responses_api_rewrites_response_model_to_the_requested_alias():
    raw = _post_stream(_sse_client(_RESPONSES_FRAMES), {"model": REQUESTED_ALIAS, "stream": True})

    envelopes = [frame["response"] for frame in _data_frames(raw) if "response" in frame]
    assert len(envelopes) == 2, raw
    for envelope in envelopes:
        # EQUALITY. `REQUESTED_ALIAS in envelope["model"]` is True of the leak.
        assert envelope["model"] == REQUESTED_ALIAS, raw


def test_streamed_responses_api_leaves_no_trace_of_the_bedrock_wire_id():
    """The wire id names AWS twice over: `anthropic.` and the `global.` profile."""
    raw = _post_stream(_sse_client(_RESPONSES_FRAMES), {"model": REQUESTED_ALIAS, "stream": True})

    assert BEDROCK_WIRE_ID.encode() not in raw
    assert b"anthropic." not in raw
    assert b"global." not in raw
    assert b"-v1:0" not in raw


def test_streamed_responses_api_preserves_every_other_field():
    """A sanitizer that eats the caller's payload is worse than the leak."""
    raw = _post_stream(_sse_client(_RESPONSES_FRAMES), {"model": REQUESTED_ALIAS, "stream": True})

    frames = _data_frames(raw)
    assert [frame["type"] for frame in frames] == [
        "response.created",
        "response.output_text.delta",
        "response.completed",
    ]
    assert [frame["sequence_number"] for frame in frames] == [0, 1, 2]
    assert frames[1]["delta"] == "hi"
    assert frames[0]["response"]["id"] == "resp_a"
    assert frames[2]["response"]["status"] == "completed"


# ── leak 2: /v1/messages streamed, `message.model` ───────────────────────────


def test_streamed_message_start_rewrites_message_model_to_the_requested_alias():
    raw = _post_stream(
        _sse_client(_MESSAGES_FRAMES, path="/v1/messages"),
        {"model": REQUESTED_ALIAS, "stream": True},
        path="/v1/messages",
    )

    starts = [frame["message"] for frame in _data_frames(raw) if frame.get("type") == "message_start"]
    assert len(starts) == 1, raw
    assert starts[0]["model"] == REQUESTED_ALIAS, raw
    assert DATED_SNAPSHOT.encode() not in raw
    # The SSE `event:` lines are not `data:` lines and must survive untouched.
    assert b"event: message_start\n" in raw
    assert b"event: content_block_delta\n" in raw


# ── chat completions: the root envelope ──────────────────────────────────────


def test_streamed_chat_completion_rewrites_the_root_model():
    frames = (
        b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"'
        + BEDROCK_WIRE_ID.encode()
        + b'","choices":[{"delta":{"content":"hi"}}]}\n\n',
        b"data: [DONE]\n\n",
    )
    raw = _post_stream(
        _sse_client(frames, path="/v1/chat/completions"),
        {"model": REQUESTED_ALIAS, "stream": True},
        path="/v1/chat/completions",
    )

    chunks = _data_frames(raw)
    assert len(chunks) == 1
    assert chunks[0]["model"] == REQUESTED_ALIAS
    assert chunks[0]["choices"][0]["delta"]["content"] == "hi"
    assert raw.endswith(b"data: [DONE]\n\n")


# ── the allowlist is an ALLOWLIST, not a recursive walk ──────────────────────


def test_model_rewrite_visits_only_the_allowlisted_envelope_paths():
    """A completion can legitimately contain the word "model" anywhere.

    Only the document root, `message` (Anthropic message_start) and `response`
    (/v1/responses events) are envelope paths. Everything else — including a
    model name the caller themselves asked about — is the caller's content.
    """
    frame = {
        "type": "response.output_item.done",
        "response": {"model": BEDROCK_WIRE_ID},
        "item": {"model": "the completion is talking about a model"},
        "delta": {"nested": {"model": "still the caller's content"}},
    }
    raw = _post_stream(
        _sse_client((b"data: " + json.dumps(frame).encode() + b"\n\n",)),
        {"model": REQUESTED_ALIAS, "stream": True},
    )

    (out,) = _data_frames(raw)
    assert out["response"]["model"] == REQUESTED_ALIAS
    assert out["item"]["model"] == "the completion is talking about a model"
    assert out["delta"]["nested"]["model"] == "still the caller's content"


def test_rewrite_envelope_model_is_not_recursive():
    """Unit-level twin of the test above, on the helper itself."""
    payload = {
        "model": BEDROCK_WIRE_ID,
        "response": {"model": BEDROCK_WIRE_ID},
        "message": {"model": DATED_SNAPSHOT},
        "choices": [{"delta": {"model": "deep"}}],
        "item": {"model": "deep"},
    }
    assert rewrite_envelope_model(payload, REQUESTED_ALIAS) is True
    assert payload["model"] == REQUESTED_ALIAS
    assert payload["response"]["model"] == REQUESTED_ALIAS
    assert payload["message"]["model"] == REQUESTED_ALIAS
    assert payload["choices"][0]["delta"]["model"] == "deep"
    assert payload["item"]["model"] == "deep"


def test_rewrite_envelope_model_reports_no_change_when_already_the_alias():
    payload = {"model": REQUESTED_ALIAS, "response": {"status": "completed"}}
    assert rewrite_envelope_model(payload, REQUESTED_ALIAS) is False


# ── an SSE frame split across ASGI chunks ────────────────────────────────────


def test_frame_split_across_asgi_chunks_is_reassembled_and_rewritten():
    """The wire id straddles two ASGI body messages, mid-JSON-key."""
    split = (
        b'data: {"type":"response.created","response":{"mod',
        b'el":"' + BEDROCK_WIRE_ID.encode() + b'","status":"in_progress"}}\n\n',
    )
    raw = _post_stream(_sse_client(split), {"model": REQUESTED_ALIAS, "stream": True})

    (out,) = _data_frames(raw)
    assert out["response"]["model"] == REQUESTED_ALIAS
    assert BEDROCK_WIRE_ID.encode() not in raw


def test_frame_split_byte_by_byte_is_reassembled_and_rewritten():
    """Worst case: one ASGI body message per byte."""
    whole = _RESPONSES_FRAMES[0]
    raw = _post_stream(
        _sse_client(tuple(whole[i : i + 1] for i in range(len(whole)))),
        {"model": REQUESTED_ALIAS, "stream": True},
    )

    (out,) = _data_frames(raw)
    assert out["response"]["model"] == REQUESTED_ALIAS
    assert BEDROCK_WIRE_ID.encode() not in raw


@pytest.mark.parametrize("terminator", [b"\n", b"\r\n"])
def test_rewriter_reassembles_across_arbitrary_splits(terminator):
    line = b'data: {"response":{"model":"' + BEDROCK_WIRE_ID.encode() + b'"}}'
    stream = line + terminator + terminator

    for cut in range(len(stream) + 1):
        rewriter = SSEModelRewriter(REQUESTED_ALIAS)
        out = rewriter.feed(stream[:cut], last=False) + rewriter.feed(stream[cut:], last=True)
        assert BEDROCK_WIRE_ID.encode() not in out, cut
        (frame,) = _data_frames(out)
        assert frame["response"]["model"] == REQUESTED_ALIAS, cut
        # Line terminators are preserved byte-for-byte.
        assert out.endswith(terminator + terminator), cut


def test_a_stream_with_no_trailing_newline_is_still_rewritten():
    frame = b'data: {"response":{"model":"' + BEDROCK_WIRE_ID.encode() + b'"}}'
    raw = _post_stream(_sse_client((frame,)), {"model": REQUESTED_ALIAS, "stream": True})

    (out,) = _data_frames(raw)
    assert out["response"]["model"] == REQUESTED_ALIAS


# ── non-JSON frames ──────────────────────────────────────────────────────────


def test_non_json_sse_frames_pass_through_byte_identical():
    frames = (
        b": ping\n\n",
        b"event: ping\ndata: not json at all\n\n",
        b"data: [DONE]\n\n",
        b"\n",
        b"data: \n\n",
        b"retry: 1000\n\n",
    )
    raw = _post_stream(_sse_client(frames), {"model": REQUESTED_ALIAS, "stream": True})

    assert raw == b"".join(frames)


@pytest.mark.parametrize(
    "line",
    [
        b"data: not json at all",
        b"data: [DONE]",
        b"data:",
        b"data: ",
        b"event: message_start",
        b": a comment",
        b"",
        b'data: "a bare json string"',
        b"data: [1, 2, 3]",
        b'data: {"model": 7}',
        b'data: {"response": "not a dict"}',
    ],
)
def test_rewrite_sse_line_leaves_unrewritable_lines_byte_identical(line):
    assert rewrite_sse_line(line, REQUESTED_ALIAS) == line
    assert rewrite_sse_line(line + b"\r", REQUESTED_ALIAS) == line + b"\r"


def test_rewrite_sse_line_preserves_the_data_field_spacing():
    """`data:{...}` with no space is valid SSE and must stay that way."""
    line = b'data:{"model":"' + BEDROCK_WIRE_ID.encode() + b'"}'
    out = rewrite_sse_line(line, REQUESTED_ALIAS)
    assert out.startswith(b"data:{")
    assert json.loads(out[len(b"data:") :])["model"] == REQUESTED_ALIAS


# ── fail safe: the requested model cannot be determined ──────────────────────


@pytest.mark.parametrize(
    "payload",
    [
        b"not json at all",
        b'{"stream": true}',
        b'{"model": 12}',
        b'{"model": ""}',
        b'{"model": "   "}',
        b'["model", "not-a-dict"]',
        b"",
    ],
)
def test_stream_is_untouched_when_the_requested_model_cannot_be_determined(payload):
    """A caller's working stream matters more than a hint. Fail SAFE, not empty."""
    raw = _post_stream(_sse_client(_RESPONSES_FRAMES), payload)
    assert raw == b"".join(_RESPONSES_FRAMES)


def test_stream_is_untouched_for_internal_callers():
    raw = _post_stream(_sse_client(_RESPONSES_FRAMES), {"model": REQUESTED_ALIAS}, headers={})
    assert raw == b"".join(_RESPONSES_FRAMES)


def test_stream_is_untouched_when_the_request_body_exceeds_the_capture_cap():
    """An oversized body is not held in memory, so the model is unknown: no rewrite."""
    payload = json.dumps({"model": REQUESTED_ALIAS, "pad": "x" * (MAX_CAPTURED_REQUEST_BODY_BYTES + 1024)}).encode()
    raw = _post_stream(_sse_client(_RESPONSES_FRAMES), payload)
    assert raw == b"".join(_RESPONSES_FRAMES)


# ── the request body must still reach the app, intact ────────────────────────


@pytest.mark.parametrize("size", [0, 1, 64_000, 300_000])
def test_downstream_app_still_receives_the_full_request_body(size):
    """Getting the receive replay wrong breaks EVERY request, not just streams."""
    seen = []
    payload = {"model": REQUESTED_ALIAS, "input": "y" * size}
    raw = _post_stream(_sse_client(_RESPONSES_FRAMES, seen_bodies=seen), payload)

    assert seen and json.loads(seen[0]) == payload
    assert [frame["response"]["model"] for frame in _data_frames(raw) if "response" in frame] == [
        REQUESTED_ALIAS,
        REQUESTED_ALIAS,
    ]


def test_downstream_app_receives_an_oversized_body_intact():
    """Over the capture cap the middleware stops holding, but must not truncate."""
    seen = []
    payload = {"model": REQUESTED_ALIAS, "pad": "z" * (MAX_CAPTURED_REQUEST_BODY_BYTES + 4096)}
    _post_stream(_sse_client(_RESPONSES_FRAMES, seen_bodies=seen), json.dumps(payload).encode())

    assert seen and json.loads(seen[0]) == payload


def test_a_disconnect_poll_before_the_body_is_read_does_not_eat_the_body():
    """`Request.is_disconnected()` calls receive inside an ALREADY-CANCELLED scope.

    A replay that returns a message without ever awaiting has no cancellation
    point, so that poll would swallow a real body chunk and the request would
    silently arrive truncated. The replay must yield to the event loop first.
    """
    seen = []

    async def route(request):
        assert await request.is_disconnected() is False
        assert await request.is_disconnected() is False
        seen.append(await request.body())
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/v1/responses", route, methods=["POST"])])
    app.add_middleware(ExternalAudienceHeaderMiddleware)

    payload = {"model": REQUESTED_ALIAS, "input": "w" * 5000}
    resp = TestClient(app).post("/v1/responses", json=payload, headers=EXTERNAL)

    assert resp.status_code == 200
    assert json.loads(seen[0]) == payload


def test_buffered_json_responses_still_work_end_to_end():
    """The non-streaming path must not regress from capturing the request body."""
    seen = []

    async def route(request):
        seen.append(await request.body())
        return JSONResponse({"model": REQUESTED_ALIAS, "ok": True}, headers=LEAKY_HEADERS)

    app = Starlette(routes=[Route("/v1/responses", route, methods=["POST"])])
    app.add_middleware(ExternalAudienceHeaderMiddleware)

    resp = TestClient(app).post("/v1/responses", json={"model": REQUESTED_ALIAS}, headers=EXTERNAL)

    assert resp.status_code == 200
    assert resp.json() == {"model": REQUESTED_ALIAS, "ok": True}
    assert json.loads(seen[0]) == {"model": REQUESTED_ALIAS}
    _assert_nothing_disclosed(resp.headers)


def test_multipart_upload_body_is_not_captured_but_arrives_intact():
    seen = []
    client = _sse_client(_RESPONSES_FRAMES, seen_bodies=seen)
    resp = client.post(
        "/v1/responses",
        files={"file": ("batch.jsonl", b'{"model":"' + REQUESTED_ALIAS.encode() + b'"}\n')},
        headers=EXTERNAL,
    )

    assert resp.status_code == 200
    assert b"batch.jsonl" in seen[0]
    # multipart is never parsed for a model, so the stream is forwarded untouched.
    assert resp.content == b"".join(_RESPONSES_FRAMES)


# ── SSE must never be buffered ───────────────────────────────────────────────


def test_sse_is_forwarded_chunk_by_chunk_and_never_buffered():
    """_is_streaming's docstring: streaming "must never be buffered".

    Asserted from inside the app: after it sends chunk N, the caller must already
    have received N body messages. A buffering middleware scores 0 until the end.
    """
    observed = []
    body_counts = []

    async def app(scope, receive, send):
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        for chunk in _RESPONSES_FRAMES:
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
            body_counts.append(len([m for m in observed if m["type"] == "http.response.body"]))
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def send(message):
        observed.append(message)

    async def receive():
        return {
            "type": "http.request",
            "body": json.dumps({"model": REQUESTED_ALIAS}).encode(),
            "more_body": False,
        }

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/responses",
        "headers": [
            (AUDIENCE_REQUEST_HEADER.encode(), b"external"),
            (b"content-type", b"application/json"),
        ],
    }
    asyncio.run(ExternalAudienceHeaderMiddleware(app)(scope, receive, send))

    assert body_counts == [1, 2, 3], body_counts
    raw = b"".join(m.get("body") or b"" for m in observed if m["type"] == "http.response.body")
    assert BEDROCK_WIRE_ID.encode() not in raw
    assert [frame["response"]["model"] for frame in _data_frames(raw) if "response" in frame] == [
        REQUESTED_ALIAS,
        REQUESTED_ALIAS,
    ]


def test_error_bodies_are_still_sanitized_when_the_request_body_was_captured():
    """The capture must not disturb the error path this module already guards."""

    async def route(request):
        await request.body()
        return JSONResponse(
            {"error": {"message": "litellm.BadRequestError: No fallback model group found"}},
            status_code=400,
            headers=LEAKY_HEADERS,
        )

    app = Starlette(routes=[Route("/v1/responses", route, methods=["POST"])])
    app.add_middleware(ExternalAudienceHeaderMiddleware)

    resp = TestClient(app).post("/v1/responses", json={"model": REQUESTED_ALIAS}, headers=EXTERNAL)

    assert resp.status_code == 400
    assert resp.json() == {"error": {"message": "Invalid request."}}
    assert b"litellm" not in resp.content
    assert b"fallback" not in resp.content


# ── the Bedrock message id, on the streaming path ────────────────────────────
#
# Found by adversarial review of the first version of this fix, then MEASURED
# live: after `message.model` was already being rewritten, a streamed
# /v1/messages call through router.trueward.ai still returned
# `msg_bdrk_01Tvwe7PT19qAfSyY5VLp4Az`. The buffered path had stripped that infix
# all along, so the same request named AWS streamed and did not buffered.


def test_streamed_message_start_strips_the_bedrock_id_prefix():
    raw = _post_stream(
        _sse_client(_MESSAGES_FRAMES, path="/v1/messages"),
        {"model": REQUESTED_ALIAS, "stream": True},
        path="/v1/messages",
    )

    starts = [frame["message"] for frame in _data_frames(raw) if frame.get("type") == "message_start"]
    assert len(starts) == 1, raw
    assert starts[0]["id"] == NATIVE_MESSAGE_ID, raw
    # The whole point: the prefix names Amazon and must not reach the caller.
    assert b"msg_bdrk_" not in raw


def test_bedrock_id_is_stripped_even_when_the_caller_model_is_unknown():
    """The id rewrite takes no alias, so an unreadable request must not gate it.

    Gating the whole rewriter on a known alias -- which the first version did --
    left `msg_bdrk_` shipping on any request whose model could not be read.
    """
    raw = _post_stream(
        _sse_client(_MESSAGES_FRAMES, path="/v1/messages"),
        {"stream": True},  # no `model` at all: alias is None
        path="/v1/messages",
    )

    starts = [frame["message"] for frame in _data_frames(raw) if frame.get("type") == "message_start"]
    assert starts[0]["id"] == NATIVE_MESSAGE_ID, raw
    assert b"msg_bdrk_" not in raw
    # The model is left ALONE, because there was no alias to point it at.
    assert starts[0]["model"] == DATED_SNAPSHOT, raw


def test_rewrite_envelope_bedrock_id_matches_the_prefix_not_a_substring():
    """An id that merely CONTAINS the infix is not Bedrock's shape."""
    payload = {"id": "msg_not_bdrk_msg_bdrk_tail"}
    assert rewrite_envelope_bedrock_id(payload) is False
    assert payload["id"] == "msg_not_bdrk_msg_bdrk_tail"


def test_rewrite_envelope_bedrock_id_visits_only_the_allowlisted_envelopes():
    """Same allowlist as the model rewrite -- root, `response`, `message`."""
    payload = {
        "id": BEDROCK_MESSAGE_ID,
        "message": {"id": BEDROCK_MESSAGE_ID},
        "response": {"id": BEDROCK_MESSAGE_ID},
        # Two levels down, and inside the caller's own content: NOT an envelope.
        "delta": {"nested": {"id": BEDROCK_MESSAGE_ID}},
        "content": [{"id": BEDROCK_MESSAGE_ID}],
    }
    assert rewrite_envelope_bedrock_id(payload) is True

    assert payload["id"] == NATIVE_MESSAGE_ID
    assert payload["message"]["id"] == NATIVE_MESSAGE_ID
    assert payload["response"]["id"] == NATIVE_MESSAGE_ID
    assert payload["delta"]["nested"]["id"] == BEDROCK_MESSAGE_ID
    assert payload["content"][0]["id"] == BEDROCK_MESSAGE_ID


def test_rewrite_envelope_bedrock_id_ignores_a_non_string_id():
    payload = {"id": 42, "message": {"id": None}}
    assert rewrite_envelope_bedrock_id(payload) is False
    assert payload["id"] == 42
    assert payload["message"]["id"] is None


def test_a_frame_needing_only_the_id_rewrite_is_still_rewritten():
    """Regression for a short-circuit: `model_rewrite() or id_rewrite()`.

    When the model already matches the alias the first call returns False, and
    an `or` chain would skip the id rewrite entirely.
    """
    line = (
        b'data: {"type":"message_start","message":{"id":"'
        + BEDROCK_MESSAGE_ID.encode()
        + b'","model":"'
        + REQUESTED_ALIAS.encode()
        + b'"}}'
    )
    out = rewrite_sse_line(line, REQUESTED_ALIAS)
    assert b"msg_bdrk_" not in out
    assert NATIVE_MESSAGE_ID.encode() in out
