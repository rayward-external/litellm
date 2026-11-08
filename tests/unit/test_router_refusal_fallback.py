"""Tests for the refusal -> content-policy-fallback bridge (rayward fork patch).

A model-level refusal is an HTTP 200 completion with finish_reason=stop and a
polite decline (e.g. "I'm sorry, but I cannot assist with that request."). Azure's
content filter never expresses that as finish_reason=content_filter, so without
this bridge such a refusal cannot fail over. When LITELLM_REFUSAL_FALLBACK_PATTERNS
is set (a JSON array of regexes), `_Driver._blocked` (litellm/router_rust/core.py)
treats a matching refusal like a content-policy violation so content_policy_fallbacks
can route it elsewhere, the same way it already treats finish_reason=content_filter.

The feature is OFF unless the env var is set — these tests assert the vanilla path
is unchanged when it is absent.

This file only covers the non-streaming path, restored onto `_Driver._blocked`
after upstream deleted litellm/router_python_core.py (the file this patch used to
live in). The streaming mid-response hold (`_RefusalStreamHold` /
`_ResponsesRefusalStreamHold`, litellm/router_rust/refusal_fallback.py) is restored
and tested here in isolation, but is NOT yet wired into a live stream -- see
.github/fork-patches.txt for what that follow-up needs.
"""

import json

import pytest

import litellm
from litellm.router import Router, _completion_matches_refusal_patterns
from litellm.types.utils import Choices, Message, ModelResponse


def _response(content, finish_reason="stop"):
    return ModelResponse(choices=[Choices(finish_reason=finish_reason, message=Message(content=content))])


def test_refusal_matcher_inert_when_env_unset(monkeypatch):
    monkeypatch.delenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", raising=False)
    assert _completion_matches_refusal_patterns(_response("I'm sorry, but I cannot assist with that request.")) is False


def test_refusal_matcher_matches_refusal_ignores_normal(monkeypatch):
    monkeypatch.setenv(
        "LITELLM_REFUSAL_FALLBACK_PATTERNS",
        json.dumps([r"i'?m sorry.*cannot assist", "i cannot help with"]),
    )
    assert _completion_matches_refusal_patterns(_response("I'm sorry, but I cannot assist with that request."))
    assert not _completion_matches_refusal_patterns(_response('{"extracted_final_answer":"0.186593","correct":"yes"}'))
    assert not _completion_matches_refusal_patterns(_response(None))


def test_refusal_matcher_preserves_commas_inside_regex(monkeypatch):
    """A JSON array means a pattern may contain commas — the comma is a literal in
    the regex, not a separator, so a partial phrase must NOT trigger it."""
    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", json.dumps(["I'm sorry, but I cannot"]))
    assert _completion_matches_refusal_patterns(_response("I'm sorry, but I cannot assist with that request."))
    assert not _completion_matches_refusal_patterns(_response("I'm sorry"))


def test_refusal_matcher_malformed_regex_never_raises(monkeypatch):
    """A malformed pattern must be skipped, never escape as a request failure; a
    valid sibling pattern still matches."""
    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", json.dumps(["[", "cannot assist"]))
    assert _completion_matches_refusal_patterns(_response("I'm sorry, but I cannot assist with that request."))
    # A non-JSON value is tolerated as a single pattern rather than raising.
    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", "cannot assist")
    assert _completion_matches_refusal_patterns(_response("...I cannot assist..."))


REFUSAL = "I'm sorry, but I cannot assist with that request."


def _refusal_router(fallback_response="This works!"):
    return Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5-20250929",
                    "api_key": "",
                    "mock_response": _response(REFUSAL),
                },
            },
            {
                "model_name": "m-fallback",
                "litellm_params": {
                    "model": "openai/my-fake-model",
                    "api_key": "",
                    "mock_response": fallback_response,
                },
            },
        ],
        content_policy_fallbacks=[{"m": ["m-fallback"]}],
    )


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_refusal_pattern_triggers_content_policy_fallback(monkeypatch, sync_mode):
    """End-to-end on `_Driver._blocked`: the same mechanism that already fails
    finish_reason=content_filter over to content_policy_fallbacks now also fires
    on a matching refusal."""
    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", json.dumps([r"i'?m sorry.*cannot assist"]))
    router = _refusal_router()
    if sync_mode:
        response = router.completion(model="m", messages=[{"role": "user", "content": "grade this"}])
    else:
        response = await router.acompletion(model="m", messages=[{"role": "user", "content": "grade this"}])
    assert response.model == "my-fake-model"


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_refusal_pattern_inert_when_env_unset(monkeypatch, sync_mode):
    """Vanilla behavior with the env unset: the refusal is returned unchanged."""
    monkeypatch.delenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", raising=False)
    router = _refusal_router()
    if sync_mode:
        response = router.completion(model="m", messages=[{"role": "user", "content": "grade this"}])
    else:
        response = await router.acompletion(model="m", messages=[{"role": "user", "content": "grade this"}])
    assert response.choices[0].message.content == REFUSAL


@pytest.mark.asyncio
async def test_refusal_pattern_normal_answer_unaffected_when_enabled(monkeypatch):
    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", json.dumps([r"i'?m sorry.*cannot assist"]))
    router = Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5-20250929",
                    "api_key": "",
                    "mock_response": "The verdict is correct.",
                },
            },
            {
                "model_name": "m-fallback",
                "litellm_params": {
                    "model": "openai/my-fake-model",
                    "api_key": "",
                    "mock_response": "Should not have called this.",
                },
            },
        ],
        content_policy_fallbacks=[{"m": ["m-fallback"]}],
    )
    response = await router.acompletion(model="m", messages=[{"role": "user", "content": "grade this"}])
    assert response.choices[0].message.content == "The verdict is correct."


@pytest.mark.asyncio
async def test_content_filter_finish_reason_still_triggers_fallback_without_env(monkeypatch):
    """The original content_filter path on `_Driver._blocked` must keep working
    with LITELLM_REFUSAL_FALLBACK_PATTERNS unset."""
    monkeypatch.delenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", raising=False)
    router = Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5-20250929",
                    "api_key": "",
                    "mock_response": _response("", finish_reason="content_filter"),
                },
            },
            {
                "model_name": "m-fallback",
                "litellm_params": {
                    "model": "openai/my-fake-model",
                    "api_key": "",
                    "mock_response": "This works!",
                },
            },
        ],
        content_policy_fallbacks=[{"m": ["m-fallback"]}],
    )
    response = await router.acompletion(model="m", messages=[{"role": "user", "content": "grade this"}])
    assert response.model == "my-fake-model"


# ---------------------------------------------------------------------------
# Stream hold mechanics, tested in isolation (NOT YET wired into a live stream
# -- see the module docstring and .github/fork-patches.txt).
# ---------------------------------------------------------------------------


def _stream_chunk(content=None, tool_calls=None, reasoning_content=None, refusal=None):
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    delta = Delta(content=content, tool_calls=tool_calls)
    if reasoning_content is not None:
        delta.reasoning_content = reasoning_content
    if refusal is not None:
        delta.refusal = refusal
    return ModelResponseStream(choices=[StreamingChoices(delta=delta)])


def test_stream_hold_releases_after_limit_and_on_tool_calls(monkeypatch):
    """A long non-refusal stream is released once the hold limit is hit, and a
    tool-call delta releases immediately (refusals are plain text)."""
    from litellm.router import _RefusalStreamHold

    hold = _RefusalStreamHold(patterns=["cannot assist"], hold_chars=10, model="m")
    assert hold.process(_stream_chunk("hello ")) == []
    released = hold.process(_stream_chunk("world!!!"))
    assert len(released) == 2  # both held chunks flushed once limit is reached
    assert hold.process(_stream_chunk("more")) != []  # passthrough afterwards

    hold = _RefusalStreamHold(patterns=["cannot assist"], hold_chars=1000, model="m")
    assert hold.process(_stream_chunk("thinking")) == []
    tool_call = {"id": "t1", "type": "function", "function": {"name": "f", "arguments": ""}}
    assert len(hold.process(_stream_chunk(None, tool_calls=[tool_call]))) == 2
    assert hold.flush() == []

    # Empty-choices chunks (e.g. a trailing usage chunk) are held, not crashed on.
    from litellm.types.utils import ModelResponseStream

    hold = _RefusalStreamHold(patterns=["cannot assist"], hold_chars=10, model="m")
    assert hold.process(ModelResponseStream(choices=[])) == []
    assert len(hold.flush()) == 1


def test_stream_hold_reasoning_deltas_advance_the_window():
    """Reasoning deltas can't match the visible refusal but must still advance
    the hold window — otherwise a reasoning model's whole thinking phase (and
    response) would be buffered to end-of-stream."""
    from litellm.router import _RefusalStreamHold

    hold = _RefusalStreamHold(patterns=["cannot assist"], hold_chars=10, model="m")
    assert hold.process(_stream_chunk(reasoning_content="let me think about it")) != []
    assert hold.active is False


def test_stream_hold_matches_openai_refusal_delta():
    """OpenAI structured-output refusals arrive in delta.refusal, not content."""
    from litellm.exceptions import MidStreamFallbackError
    from litellm.router import _RefusalStreamHold

    hold = _RefusalStreamHold(patterns=["cannot assist"], hold_chars=400, model="m")
    with pytest.raises(MidStreamFallbackError) as exc:
        hold.process(_stream_chunk(refusal="I cannot assist with that."))
    assert isinstance(exc.value.original_exception, litellm.ContentPolicyViolationError)
    assert exc.value.is_pre_first_chunk is True


def test_get_refusal_fallback_patterns_direct(monkeypatch):
    from litellm.router import _get_refusal_fallback_patterns

    monkeypatch.delenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", raising=False)
    assert _get_refusal_fallback_patterns() == []

    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", json.dumps(["a", "b"]))
    assert _get_refusal_fallback_patterns() == ["a", "b"]

    # Non-JSON value is tolerated as a single literal pattern, not raised.
    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", "not json[")
    assert _get_refusal_fallback_patterns() == ["not json["]

    # A JSON object (not an array) is also coerced to a single literal pattern.
    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", json.dumps({"a": 1}))
    assert _get_refusal_fallback_patterns() == [json.dumps({"a": 1})]

    # Blank/non-string entries in an otherwise valid array are dropped.
    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", json.dumps(["ok", "", 5, None]))
    assert _get_refusal_fallback_patterns() == ["ok"]


def test_text_matches_refusal_patterns_direct():
    from litellm.router import _text_matches_refusal_patterns

    assert _text_matches_refusal_patterns("I cannot assist", ["cannot assist"]) is True
    assert _text_matches_refusal_patterns("all good here", ["cannot assist"]) is False
    # A malformed regex among the patterns is skipped, not raised.
    assert _text_matches_refusal_patterns("I cannot assist", ["[", "cannot assist"]) is True
    assert _text_matches_refusal_patterns("I cannot assist", ["["]) is False


def test_refusal_stream_hold_chars_direct(monkeypatch):
    from litellm.router import _refusal_stream_hold_chars

    monkeypatch.delenv("LITELLM_REFUSAL_FALLBACK_STREAM_HOLD_CHARS", raising=False)
    assert _refusal_stream_hold_chars() == 400

    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_STREAM_HOLD_CHARS", "50")
    assert _refusal_stream_hold_chars() == 50

    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_STREAM_HOLD_CHARS", "0")
    assert _refusal_stream_hold_chars() == 0

    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_STREAM_HOLD_CHARS", "-5")
    assert _refusal_stream_hold_chars() == 0

    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_STREAM_HOLD_CHARS", "not-a-number")
    assert _refusal_stream_hold_chars() == 400


def test_stream_hold_release_direct():
    """_release() drains and clears the held buffer and disarms the hold."""
    from litellm.router import _RefusalStreamHold

    hold = _RefusalStreamHold(patterns=["cannot assist"], hold_chars=1000, model="m")
    hold.process(_stream_chunk("hello"))
    hold.process(_stream_chunk(" world"))
    assert len(hold._held) == 2

    released = hold._release()
    assert len(released) == 2
    assert hold.active is False
    assert hold._held == []

    # Calling again returns nothing further: the buffer was already drained.
    assert hold._release() == []


def test_stream_hold_delta_state_direct():
    """_delta_state() extracts (matchable text, reasoning length, saw tool call)
    from a single stream item for every delta shape it must handle."""
    from litellm.router import _RefusalStreamHold
    from litellm.types.utils import ModelResponseStream

    assert _RefusalStreamHold._delta_state(_stream_chunk("hello")) == ("hello", 0, False)
    assert _RefusalStreamHold._delta_state(_stream_chunk(refusal="I cannot help")) == ("I cannot help", 0, False)
    assert _RefusalStreamHold._delta_state(_stream_chunk(reasoning_content="thinking...")) == (
        "",
        len("thinking..."),
        False,
    )
    tool_call = {"id": "t1", "type": "function", "function": {"name": "f", "arguments": ""}}
    text, reasoning_len, saw_tool_call = _RefusalStreamHold._delta_state(_stream_chunk(None, tool_calls=[tool_call]))
    assert saw_tool_call is True
    # Empty choices (e.g. a trailing usage-only chunk) must not raise.
    assert _RefusalStreamHold._delta_state(ModelResponseStream(choices=[])) == ("", 0, False)


def test_stream_hold_raise_refusal_direct():
    from litellm.exceptions import MidStreamFallbackError
    from litellm.router import _RefusalStreamHold

    hold = _RefusalStreamHold(patterns=["cannot assist"], hold_chars=400, model="my-model")
    with pytest.raises(MidStreamFallbackError) as exc:
        hold._raise_refusal()
    assert exc.value.model == "my-model"
    assert exc.value.is_pre_first_chunk is True
    assert isinstance(exc.value.original_exception, litellm.ContentPolicyViolationError)


# ---------------------------------------------------------------------------
# Responses API streaming hold mechanics, also tested in isolation only.
# ---------------------------------------------------------------------------

from types import SimpleNamespace


def _ev(type_, **kw):
    return SimpleNamespace(type=type_, **kw)


def _incomplete_event(reason="content_filter"):
    return _ev(
        "response.incomplete",
        response=SimpleNamespace(incomplete_details=SimpleNamespace(reason=reason), usage=None),
    )


def _responses_hold(hold_chars=400):
    from litellm.router import _ResponsesRefusalStreamHold

    return _ResponsesRefusalStreamHold(patterns=["cannot assist"], hold_chars=hold_chars, model="gpt-5.5")


def test_responses_hold_raises_on_content_filter_terminal():
    """Azure jailbreak filter: 200 stream ending response.incomplete with
    reason=content_filter must be treated like a content-policy violation."""
    from litellm.exceptions import MidStreamFallbackError

    hold = _responses_hold()
    assert hold.process(_ev("response.created", response=SimpleNamespace())) == []
    assert hold.process(_ev("response.output_text.delta", delta="I'm")) == []
    with pytest.raises(MidStreamFallbackError) as exc:
        hold.process(_incomplete_event())
    assert isinstance(exc.value.original_exception, litellm.ContentPolicyViolationError)
    assert exc.value.is_pre_first_chunk is True

    # A non-filter incomplete (e.g. max_output_tokens) is NOT a block.
    hold = _responses_hold()
    assert hold.process(_incomplete_event(reason="max_output_tokens")) == []
    assert len(hold.flush()) == 1


def test_responses_hold_matches_refusal_text_delta():
    from litellm.exceptions import MidStreamFallbackError

    hold = _responses_hold()
    assert hold.process(_ev("response.output_text.delta", delta="I'm sorry, but I ")) == []
    with pytest.raises(MidStreamFallbackError):
        hold.process(_ev("response.output_text.delta", delta="cannot assist with that."))


def test_responses_hold_releases_on_reasoning_window_and_tool_calls():
    hold = _responses_hold(hold_chars=10)
    assert hold.process(_ev("response.reasoning_summary_text.delta", delta="thinking hard...")) != []
    assert hold.active is False

    hold = _responses_hold()
    released = hold.process(_ev("response.function_call_arguments.delta", delta='{"a"'))
    assert len(released) == 1
    assert hold.active is False


def test_responses_hold_blocked_terminal_with_dict_details_and_enum_type():
    """incomplete_details may arrive dict-shaped, and event.type may be the
    str-Enum member rather than a plain string — both must trigger."""
    from litellm.exceptions import MidStreamFallbackError
    from litellm.types.llms.openai import ResponsesAPIStreamEvents

    hold = _responses_hold()
    ev = _ev(
        ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE,
        response=SimpleNamespace(incomplete_details={"reason": "content_filter"}, usage=None),
    )
    with pytest.raises(MidStreamFallbackError):
        hold.process(ev)

    # Enum-typed text delta accumulates too.
    hold = _responses_hold()
    with pytest.raises(MidStreamFallbackError):
        hold.process(_ev(ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA, delta="I cannot assist."))


def test_is_blocked_terminal_direct():
    """Direct coverage of the _is_blocked_terminal hook on both classes."""
    from litellm.router import _RefusalStreamHold, _ResponsesRefusalStreamHold

    # Base hook: always False (chat has no blocked-terminal event shape).
    assert _RefusalStreamHold._is_blocked_terminal(_stream_chunk("x")) is False

    assert _ResponsesRefusalStreamHold._is_blocked_terminal(_incomplete_event()) is True
    assert _ResponsesRefusalStreamHold._is_blocked_terminal(_incomplete_event(reason="max_output_tokens")) is False
    assert (
        _ResponsesRefusalStreamHold._is_blocked_terminal(
            _ev("response.completed", response=SimpleNamespace(usage=None))
        )
        is False
    )


def test_warn_if_refusal_patterns_configured_for_stream(monkeypatch):
    """The streaming hold is not wired into a live stream yet (see the module docstring), so a
    streamed refusal is not caught while LITELLM_REFUSAL_FALLBACK_PATTERNS is set. Rather than
    stay silent about that gap, every streaming call logs a warning while the env var is set."""
    from litellm.router_rust import refusal_fallback

    warnings: list[str] = []
    monkeypatch.setattr(
        "litellm._logging.verbose_router_logger.warning",
        lambda msg, *args: warnings.append(msg % args if args else msg),
    )

    monkeypatch.delenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", raising=False)
    refusal_fallback.warn_if_refusal_patterns_configured_for_stream()
    assert warnings == []

    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", json.dumps(["cannot assist"]))
    refusal_fallback.warn_if_refusal_patterns_configured_for_stream()
    assert len(warnings) == 1
    assert "stream=True" in warnings[0]
