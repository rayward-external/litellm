"""
Guards for the external-audience ERROR BODY policy.

The defect these exist for: a request whose only fault was a misspelled ``role``
returned our internal fallback table, our internal model-group naming, the
deny-tag mechanism that constrains the caller's own key, and the name of the
gateway software -- to an external party, on an ordinary 4xx.
"""

import json

import pytest

from litellm.proxy.middleware.external_audience_middleware import (
    AUDIENCE_REQUEST_HEADER,
    EXTERNAL_AUDIENCE,
    MAX_EXTERNAL_ERROR_MESSAGE_CHARS,
    ExternalAudienceHeaderMiddleware,
    sanitize_error_body,
    sanitize_error_message,
)

# The ACTUAL message measured on router.trueward.ai, 2026-08-01, from
# {"model":"gpt-5.6-luna","max_tokens":5,"messages":[{"role":"wizard",...}]}.
# Real payload rather than a hand-written stub: a synthetic fixture would have
# been written from the same understanding as the fix.
PROD_LEAK_MESSAGE = (
    "litellm.BadRequestError: Not allowed to access model due to tags configuration. "
    "Passed model=gpt-5.6-luna-openai and tags=['!pin:anthropic', '!pin:openai']"
    "No fallback model group found for original model_group=gpt-5.6-luna-openai. "
    "Fallbacks=[{'gpt-5.4': ['gpt-5.4-openai']}, {'gpt-5.4-mini': ['gpt-5.4-mini-openai']}, "
    "{'gpt-5.4-nano': ['gpt-5.4-nano-openai']}, {'gpt-5.5': ['gpt-5.5-openai']}, "
    "{'gpt-5.6-luna': ['gpt-5.6-luna-openai']}, {'gpt-5.6-sol': ['gpt-5.6-sol-openai']}, "
    "{'gpt-5.6-terra': ['gpt-5.6-terra-openai']}]. Received Model Group=gpt-5.6-luna-openai"
)

# Asserted individually so a failure names WHICH disclosure escaped.
LEAK_MARKERS = (
    "litellm",
    "fallback",
    "model_group",
    "tags=",
    "!pin:anthropic",
    "gpt-5.6-luna-openai",
    "Model Group",
)


def assert_no_leak(payload: str) -> None:
    lowered = payload.lower()
    for marker in LEAK_MARKERS:
        assert marker.lower() not in lowered, f"external error body still leaks {marker!r}: {payload}"


# --------------------------------------------------------------------------
# The message-level policy
# --------------------------------------------------------------------------


def test_production_leak_is_replaced():
    assert_no_leak(sanitize_error_message(PROD_LEAK_MESSAGE, 400))


def test_useful_upstream_validation_message_survives():
    # The whole reason this is not an allowlist: the caller must still be able to
    # see why their own request was rejected.
    assert sanitize_error_message("max_tokens: Field required", 400) == "max_tokens: Field required"


def test_upstream_message_wrapped_by_litellm_keeps_the_useful_half():
    # The prefix names the gateway; the remainder is the upstream's own words and
    # is worth forwarding. Stripping before testing is what allows both.
    got = sanitize_error_message(
        "litellm.BadRequestError: Invalid value: 'wizard'. Supported values are: 'system', 'user'.",
        400,
    )
    assert got == "Invalid value: 'wizard'. Supported values are: 'system', 'user'."
    assert "litellm" not in got.lower()


def test_key_entitlement_message_survives():
    # Measured at 229 chars and marker-free. It is the caller's OWN entitlement,
    # so it is deliberately under the cap -- this test pins that the cap was not
    # set so low it starts eating legitimate messages.
    message = (
        "key not allowed to access model. This key can only access "
        "models=['claude-fable-5', 'claude-opus-5', 'claude-haiku-4-5', 'gemini-3.6-flash', "
        "'gpt-5.6-sol', 'gpt-5.6-luna', 'gpt-5.6-terra', 'gpt-5.4-nano']. "
        "Tried to access no-such-model-xyz"
    )
    assert len(message) < MAX_EXTERNAL_ERROR_MESSAGE_CHARS
    assert sanitize_error_message(message, 403) == message


def test_long_marker_free_message_is_capped():
    # The control that catches dumps whose vocabulary we have never seen, which
    # is precisely what a marker list cannot do.
    #
    # The length is a LITERAL, deliberately. Writing it as
    # MAX_EXTERNAL_ERROR_MESSAGE_CHARS + 1 makes the test self-referential: it
    # then passes for ANY cap, including one raised to 10_000_000, because the
    # input grows with the thing under test. That version was written first and
    # a mutation run caught it -- it is the exact shape of a guard that looks
    # strict and asserts nothing.
    #
    # 900 is chosen from measurement: the production leak was 1150 chars, so a
    # cap that lets 900 through would not have caught it.
    assert sanitize_error_message("x" * 900, 400) == "Invalid request."


def test_cap_is_bracketed_by_the_two_measured_messages():
    # Together with test_key_entitlement_message_survives (229 chars, must pass)
    # this pins the cap into the measured window rather than to one number.
    # Below 229 it starts eating legitimate messages; at or above 1150 it stops
    # catching the dump that motivated it.
    assert 229 < MAX_EXTERNAL_ERROR_MESSAGE_CHARS < 1150


@pytest.mark.parametrize(
    "status,expected",
    [(400, "Invalid request."), (403, "Invalid request."), (500, "Internal error."), (502, "Internal error.")],
)
def test_generic_message_matches_status_class(status, expected):
    assert sanitize_error_message(PROD_LEAK_MESSAGE, status) == expected


# --------------------------------------------------------------------------
# The body-level policy
# --------------------------------------------------------------------------


def test_openai_dialect_body_is_sanitized():
    raw = json.dumps({"error": {"message": PROD_LEAK_MESSAGE, "type": None, "param": None, "code": "400"}}).encode()
    out = sanitize_error_body(raw, 400).decode()
    assert_no_leak(out)
    assert json.loads(out)["error"]["message"] == "Invalid request."


def test_anthropic_dialect_body_is_sanitized():
    # A different shape, not hardcoded anywhere: the tree walk visits "message"
    # at any depth, so both dialects are covered by one rule.
    raw = json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": PROD_LEAK_MESSAGE}}).encode()
    out = sanitize_error_body(raw, 400).decode()
    assert_no_leak(out)


def test_marker_in_an_unvisited_field_still_fails_closed():
    # The final re-check over the serialized result. "metadata" is not in
    # _MESSAGE_KEYS, so the tree walk leaves it alone and only the backstop
    # catches it.
    raw = json.dumps({"error": {"message": "bad request"}, "metadata": {"note": PROD_LEAK_MESSAGE}}).encode()
    out = sanitize_error_body(raw, 400).decode()
    assert_no_leak(out)
    assert json.loads(out)["error"]["message"] == "Invalid request."


def test_non_json_body_carrying_markers_is_replaced():
    out = sanitize_error_body(b"<html>litellm.BadRequestError: model_group failure</html>", 500).decode()
    assert_no_leak(out)


def test_non_json_body_without_markers_passes_through():
    # An upstream plain-text 502 is not ours to rewrite; replacing it would turn
    # this policy into an outage.
    raw = b"upstream gateway timeout"
    assert sanitize_error_body(raw, 504) == raw


def test_clean_json_error_is_untouched_apart_from_reserialization():
    raw = json.dumps({"error": {"message": "max_tokens: Field required", "type": "invalid_request_error"}}).encode()
    assert json.loads(sanitize_error_body(raw, 400)) == json.loads(raw)


# --------------------------------------------------------------------------
# The ASGI wiring
# --------------------------------------------------------------------------


async def _drive(app_messages, *, external: bool, path: str = "/v1/chat/completions"):
    """Run the middleware over a canned response, returning what reached the server."""
    headers = [(AUDIENCE_REQUEST_HEADER.encode(), EXTERNAL_AUDIENCE.encode())] if external else []
    scope = {"type": "http", "path": path, "headers": headers}

    async def app(scope, receive, send):
        for message in app_messages:
            await send(dict(message))

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    await ExternalAudienceHeaderMiddleware(app)(scope, lambda: None, send)
    return sent


@pytest.mark.asyncio
async def test_error_response_is_sanitized_end_to_end():
    body = json.dumps({"error": {"message": PROD_LEAK_MESSAGE}}).encode()
    sent = await _drive(
        [
            {"type": "http.response.start", "status": 400, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]},
            {"type": "http.response.body", "body": body, "more_body": False},
        ],
        external=True,
    )

    start = next(m for m in sent if m["type"] == "http.response.start")
    delivered = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert_no_leak(delivered.decode())

    # Content-length must describe the SANITIZED body. A stale one truncates or
    # hangs the client, and it is invisible to any assertion on the body alone.
    lengths = [v for k, v in start["headers"] if k.decode().lower() == "content-length"]
    assert lengths == [str(len(delivered)).encode()], f"content-length {lengths} != body {len(delivered)}"


@pytest.mark.asyncio
async def test_error_start_is_not_emitted_before_the_body_is_known():
    # Ordering matters: if the start went out first, its content-length would be
    # frozen at the pre-sanitization value.
    body = json.dumps({"error": {"message": PROD_LEAK_MESSAGE}}).encode()
    sent = await _drive(
        [
            {"type": "http.response.start", "status": 400, "headers": [(b"content-length", str(len(body)).encode())]},
            {"type": "http.response.body", "body": body[:20], "more_body": True},
            {"type": "http.response.body", "body": body[20:], "more_body": False},
        ],
        external=True,
    )
    assert [m["type"] for m in sent] == ["http.response.start", "http.response.body"]
    assert_no_leak(b"".join(m.get("body", b"") for m in sent).decode())


@pytest.mark.asyncio
async def test_successful_streaming_response_is_not_buffered():
    # Buffering a 200 would break streaming, which is most of this proxy's
    # traffic. Each chunk must reach the server as its own message.
    chunks = [
        {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]},
        {"type": "http.response.body", "body": b"data: {\"a\":1}\n\n", "more_body": True},
        {"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": False},
    ]
    sent = await _drive(chunks, external=True)
    assert [m["type"] for m in sent] == [c["type"] for c in chunks]
    assert sent[1]["body"] == b"data: {\"a\":1}\n\n"
    assert sent[2]["body"] == b"data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_internal_error_response_is_untouched():
    # Absence of the marker means INTERNAL. Internal traffic never traverses the
    # external LBs, so its behaviour must be unchanged by this patch existing.
    body = json.dumps({"error": {"message": PROD_LEAK_MESSAGE}}).encode()
    sent = await _drive(
        [
            {"type": "http.response.start", "status": 400, "headers": [(b"content-type", b"application/json")]},
            {"type": "http.response.body", "body": body, "more_body": False},
        ],
        external=False,
    )
    delivered = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert delivered == body, "internal caller lost the diagnostic detail they need"
