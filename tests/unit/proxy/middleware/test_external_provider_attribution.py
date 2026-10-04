"""Guards for the external-audience SUCCESS-body policy.

The defect: a successful response told an external caller which cloud served it
— `vertex_ai_*` on Vertex, `latency_checkpoint` on Azure, a `msg_bdrk_` id on
Bedrock. Every one measured empty or internal-only, so they disclosed a supplier
and gave the caller nothing.
"""

import json

import pytest

from litellm.proxy.middleware.external_audience_middleware import (
    AUDIENCE_REQUEST_HEADER,
    EXTERNAL_AUDIENCE,
    PROVIDER_ATTRIBUTION_KEYS,
    PROVIDER_ATTRIBUTION_NESTED_KEYS,
    ExternalAudienceHeaderMiddleware,
    strip_provider_attribution,
)

# Measured on router.trueward.ai, 2026-08-01. Real payloads, not stubs.
GEMINI_BODY = {
    "id": "abc",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
    "model": "gemini-3.6-flash",
    "usage": {"total_tokens": 6},
    "vertex_ai_grounding_metadata": [],
    "vertex_ai_url_context_metadata": [],
    "vertex_ai_safety_results": [],
    "vertex_ai_citation_metadata": [],
}
AZURE_BODY = {
    "id": "chatcmpl-x",
    "choices": [{"index": 0, "message": {"content": "hi"}}],
    "usage": {
        "total_tokens": 30,
        "latency_checkpoint": {"engine_ttft_ms": 100, "service_ttlt_ms": 361},
    },
    "prompt_filter_results": [{"prompt_index": 0, "content_filter_results": {"hate": {"filtered": False}}}],
}
BEDROCK_BODY = {
    "id": "msg_bdrk_01TCRabhum91k3kMbNdF53aj",
    "type": "message",
    "content": [{"type": "text", "text": "hi"}],
    "model": "claude-haiku-4-5",
}


def test_vertex_field_names_are_removed():
    out = json.loads(strip_provider_attribution(json.dumps(GEMINI_BODY).encode()))
    for key in GEMINI_BODY:
        if key.startswith("vertex_ai_"):
            assert key not in out, f"{key} still names Vertex"
    # The completion must survive intact.
    assert out["choices"][0]["message"]["content"] == "hi"
    assert out["usage"]["total_tokens"] == 6


def test_bedrock_id_prefix_is_rewritten_to_the_anthropic_form():
    out = json.loads(strip_provider_attribution(json.dumps(BEDROCK_BODY).encode()))
    assert "bdrk" not in out["id"], f"id still names Bedrock: {out['id']}"
    # Rewritten, not deleted: the caller may quote it in a support request, and
    # `msg_<id>` is what native Anthropic returns anyway.
    assert out["id"] == "msg_01TCRabhum91k3kMbNdF53aj"


def test_content_filter_results_are_deliberately_KEPT():
    """The over-sanitization guard.

    These are Azure-shaped, so removing them would hide a supplier — but they
    tell the caller their content was filtered and on which category. Silent
    filtering is worse than the disclosure, so this is a knowing trade and it
    must not be "tidied up" later without that being a decision.
    """
    out = json.loads(strip_provider_attribution(json.dumps(AZURE_BODY).encode()))
    assert "prompt_filter_results" in out, "the caller lost the reason their content was filtered"
    assert out["prompt_filter_results"][0]["content_filter_results"]["hate"]["filtered"] is False


def test_internal_only_latency_telemetry_is_removed():
    """`latency_checkpoint` is NESTED under `usage`, not top level.

    The first version of this patch popped only top-level keys and therefore
    never removed it — and the first version of THIS test asserted
    `not in out.get("usage", {}) or "latency_checkpoint" not in out`, a vacuous
    `or` that passes whenever either side holds. Both bugs hid each other.
    Asserted here on the exact nested path, and nowhere else.
    """
    out = json.loads(strip_provider_attribution(json.dumps(AZURE_BODY).encode()))
    assert "latency_checkpoint" not in out["usage"], f"engine timings still present: {out['usage']}"
    assert "latency_checkpoint" not in json.dumps(out), "present somewhere else in the body"
    # The rest of usage is the caller's own token accounting and must survive.
    assert out["usage"]["total_tokens"] == 30


def test_untouched_body_is_returned_byte_identical():
    """The common case, and the one that must stay cheap.

    A body with nothing to strip must not even be re-serialized — round-tripping
    through json would reorder keys and change bytes for no reason.
    """
    raw = json.dumps({"id": "chatcmpl-x", "choices": [{"message": {"content": "hi"}}]}).encode()
    assert strip_provider_attribution(raw) is raw


def test_non_json_success_body_passes_through():
    """Unlike the error path this fails OPEN, on purpose.

    These are attribution hints, not credentials. Replacing a caller's
    successful response because we could not parse it would do more harm than
    the disclosure it prevents.
    """
    raw = b"\x89PNG\r\n\x1a\n binary-ish latency_checkpoint"
    assert strip_provider_attribution(raw) == raw


def test_every_declared_key_is_actually_removed():
    """Pins the declared list against the implementation.

    A key added to PROVIDER_ATTRIBUTION_KEYS but never popped would read as
    covered while still shipping.
    """
    body = {key: ["something"] for key in PROVIDER_ATTRIBUTION_KEYS}
    body["usage"] = {parent_key: {"x": 1} for _, parent_key in PROVIDER_ATTRIBUTION_NESTED_KEYS}
    body["choices"] = [{"message": {"content": "hi"}}]
    out = json.loads(strip_provider_attribution(json.dumps(body).encode()))
    still_present = [key for key in PROVIDER_ATTRIBUTION_KEYS if key in out]
    still_present += [k for _, k in PROVIDER_ATTRIBUTION_NESTED_KEYS if k in out.get("usage", {})]
    assert not still_present, f"declared but not removed: {still_present}"


# --------------------------------------------------------------------------
# ASGI wiring
# --------------------------------------------------------------------------


async def _drive(app_messages, *, external=True):
    headers = [(AUDIENCE_REQUEST_HEADER.encode(), EXTERNAL_AUDIENCE.encode())] if external else []
    scope = {"type": "http", "path": "/v1/chat/completions", "headers": headers}

    async def app(scope, receive, send):
        for message in app_messages:
            await send(dict(message))

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    await ExternalAudienceHeaderMiddleware(app)(scope, lambda: None, send)
    return sent


@pytest.mark.asyncio
async def test_success_body_is_scrubbed_and_reframed():
    body = json.dumps(GEMINI_BODY).encode()
    sent = await _drive(
        [
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            },
            {"type": "http.response.body", "body": body, "more_body": False},
        ]
    )
    start = next(m for m in sent if m["type"] == "http.response.start")
    delivered = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")

    assert b"vertex_ai_" not in delivered
    lengths = [v for k, v in start["headers"] if k.decode().lower() == "content-length"]
    assert lengths == [str(len(delivered)).encode()], "content-length does not describe the rewritten body"


@pytest.mark.asyncio
async def test_streaming_success_is_never_buffered():
    """The cost guard.

    Measurement says SSE chunks carry none of these fields, so buffering a
    stream would pay the entire cost of breaking streaming to buy nothing. Each
    chunk must reach the server as its own message, in order.
    """
    chunks = [
        {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]},
        {"type": "http.response.body", "body": b'data: {"a":1}\n\n', "more_body": True},
        {"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": False},
    ]
    sent = await _drive(chunks)
    assert [m["type"] for m in sent] == [c["type"] for c in chunks]
    assert sent[1]["body"] == b'data: {"a":1}\n\n'
    assert sent[2]["body"] == b"data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_internal_success_body_is_untouched():
    body = json.dumps(GEMINI_BODY).encode()
    sent = await _drive(
        [
            {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]},
            {"type": "http.response.body", "body": body, "more_body": False},
        ],
        external=False,
    )
    delivered = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert delivered == body, "internal caller lost fields they may depend on"
