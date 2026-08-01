"""
/v1/messages must reject a missing required field as a 400 in the Anthropic
error shape — not as a 500 carrying the Python signature of the internal
handler.

Regression cover for the bug where a body omitting `max_tokens` was splatted
into `anthropic_messages(max_tokens: int, messages: List[Dict], model: str, …)`,
raised `TypeError` at the call boundary before any validation ran, and was
turned into:

    HTTP 500
    {"error":{"message":"anthropic_messages() missing 1 required positional
               argument: 'max_tokens'","type":"None","param":"None","code":"500"}}

Two things were wrong with that: the status class (a caller omitting a required
field is a 4xx) and the disclosure (an internal function name and signature
returned to whoever sent the request, external callers included).
"""

import pytest

from litellm.proxy.anthropic_endpoints.endpoints import (
    ANTHROPIC_MESSAGES_REQUIRED_FIELDS,
    _missing_required_anthropic_field,
)

VALID_BODY = {
    "model": "claude-haiku-4-5",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 16,
}


def _without(field: str) -> dict:
    return {k: v for k, v in VALID_BODY.items() if k != field}


def test_required_fields_match_the_anthropic_spec():
    """The Anthropic Messages API marks exactly these three as required."""
    assert set(ANTHROPIC_MESSAGES_REQUIRED_FIELDS) == {
        "model",
        "messages",
        "max_tokens",
    }


def test_valid_body_passes():
    assert _missing_required_anthropic_field(VALID_BODY) is None


@pytest.mark.parametrize("field", ["model", "messages", "max_tokens"])
def test_absent_required_field_is_reported(field):
    """The original bug: absent keys never reached validation at all."""
    assert _missing_required_anthropic_field(_without(field)) == field


@pytest.mark.parametrize("field", ["model", "messages", "max_tokens"])
def test_explicit_null_is_treated_as_absent(field):
    """`"max_tokens": null` must not reach the router.

    It used to fail deep in transformation, and the router then concatenated
    its own fallback diagnostics onto the message — disclosing the entire
    fallback chain, internal model-group names included, to the caller.
    """
    assert _missing_required_anthropic_field({**VALID_BODY, field: None}) == field


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({**VALID_BODY, "max_tokens": "16"}, id="max_tokens-wrong-type"),
        pytest.param({**VALID_BODY, "messages": "hello"}, id="messages-wrong-type"),
    ],
)
def test_present_but_wrong_type_is_left_to_downstream_validation(body):
    """Don't duplicate a check that already returns a correct 400.

    A present-but-wrong-type value reaches pydantic and already produces a
    proper `invalid_request_error`. This guard is only for the absent case,
    which skipped validation entirely.
    """
    assert _missing_required_anthropic_field(body) is None


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({**VALID_BODY, "max_tokens": 0}, id="max_tokens-zero"),
        pytest.param({**VALID_BODY, "messages": []}, id="messages-empty"),
    ],
)
def test_falsy_but_present_values_are_not_missing(body):
    """Presence is `is not None`, never truthiness.

    `0` and `[]` are present. Whether they are *semantically* valid is the
    downstream validator's call, not this guard's — a truthiness test here
    would reject them with the wrong error message.
    """
    assert _missing_required_anthropic_field(body) is None


def test_first_missing_field_is_reported_when_several_are_absent():
    assert _missing_required_anthropic_field({}) == "model"


# --- server-side defaults ------------------------------------------------
# The request processor fills `model` and `max_tokens` from the proxy's own
# settings before dispatch (common_request_processing.py:1195-1208):
# `completion_model`/`user_model` for the model, `user_max_tokens` for the cap.
# A proxy started with `--model` or `--max_tokens` legitimately serves a body
# that omits them, so validating the raw body ALONE would reject requests this
# deployment can answer.


def test_max_tokens_may_come_from_the_server_default():
    body = _without("max_tokens")
    assert _missing_required_anthropic_field(body) == "max_tokens"
    assert _missing_required_anthropic_field(body, server_max_tokens=4096) is None


def test_model_may_come_from_the_server_default():
    body = _without("model")
    assert _missing_required_anthropic_field(body) == "model"
    assert _missing_required_anthropic_field(body, server_model="gpt-4o") is None


def test_messages_has_no_server_default_and_stays_required():
    """`messages` is the request; nothing on the server can supply it."""
    body = _without("messages")
    assert (
        _missing_required_anthropic_field(
            body, server_model="gpt-4o", server_max_tokens=4096
        )
        == "messages"
    )


def test_server_defaults_do_not_mask_a_different_missing_field():
    assert (
        _missing_required_anthropic_field({"messages": []}, server_max_tokens=4096)
        == "model"
    )


def test_error_message_does_not_leak_internals():
    """The message a caller sees must name the field, not our call stack."""
    field = _missing_required_anthropic_field(_without("max_tokens"))
    message = f"{field}: Field required"
    assert message == "max_tokens: Field required"
    assert "anthropic_messages" not in message
    assert "positional argument" not in message


def test_envelope_is_anthropic_shaped():
    """The 400 body must be Anthropic's error shape, not LiteLLM's ProxyException.

    The old 500 carried `"type":"None","param":"None"`, so a client could not
    branch on it either.
    """
    from litellm.anthropic_interface.exceptions import AnthropicExceptionMapping

    body = AnthropicExceptionMapping.transform_to_anthropic_error(
        status_code=400,
        raw_message="max_tokens: Field required",
        request_id="req-123",
    )
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["message"] == "max_tokens: Field required"
    assert body["request_id"] == "req-123"
