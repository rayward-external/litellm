"""Refusal -> content-policy-fallback bridge (rayward fork patch).

A model-level refusal is an HTTP 200 completion with finish_reason=stop and a polite
decline (e.g. "I'm sorry, but I cannot assist with that request."). Azure's content
filter never expresses that as finish_reason=content_filter, so without this bridge
such a refusal cannot fail over. When LITELLM_REFUSAL_FALLBACK_PATTERNS is set (a
JSON array of regexes), a matching refusal is treated like a content-policy
violation so content_policy_fallbacks can route it elsewhere.

OFF by default: only active when LITELLM_REFUSAL_FALLBACK_PATTERNS is set. Ported
from the deleted litellm/router_python_core.py onto its one surviving call site,
``_Driver._blocked`` in litellm/router_rust/core.py, for the non-streaming path. The
streaming mid-response hold (``_RefusalStreamHold``/``_ResponsesRefusalStreamHold``)
is restored here too, tested in isolation, but is NOT yet wired into
litellm/router_rust/invoke/streaming.py's live stream processing -- see
.github/fork-patches.txt for the removal condition and what wiring it up needs.
"""

from typing import Any, NoReturn

import litellm.exceptions as litellm_exceptions
from litellm.types.utils import ModelResponse

_REFUSAL_STREAM_HOLD_CHARS_DEFAULT = 400


def _get_refusal_fallback_patterns() -> list[str]:
    """Parse LITELLM_REFUSAL_FALLBACK_PATTERNS into a list of regex strings.
    Empty/unset env => [] (feature inert). A non-JSON value is tolerated as a
    single pattern rather than raising."""
    import json
    import os

    raw = os.environ.get("LITELLM_REFUSAL_FALLBACK_PATTERNS", "").strip()
    if not raw:
        return []

    try:
        patterns = json.loads(raw)
    except (ValueError, TypeError):
        patterns = [raw]
    if not isinstance(patterns, list):
        patterns = [raw]
    return [p for p in patterns if isinstance(p, str) and p.strip()]


def _text_matches_refusal_patterns(text: str, patterns: list[str]) -> bool:
    """A malformed regex is skipped, never raised, so it can't turn a successful
    completion into a request failure."""
    import re

    for pattern in patterns:
        try:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        except re.error:
            continue
    return False


def _completion_matches_refusal_patterns(response: ModelResponse) -> bool:
    """Return True if a successful (non-content_filter) completion is a model-level
    refusal that should be treated like a content-policy violation, so the router's
    content_policy_fallbacks can route it to an alternate deployment."""
    patterns = _get_refusal_fallback_patterns()
    if not patterns:
        return False

    # StreamingChoices has no `.message`; getattr keeps this total for either choice
    # type without a type: ignore.
    choices = getattr(response, "choices", None) or []
    if not choices:
        return False
    content = getattr(getattr(choices[0], "message", None), "content", None)
    if not content or not isinstance(content, str):
        return False

    return _text_matches_refusal_patterns(content, patterns)


def warn_if_refusal_patterns_configured_for_stream() -> None:
    """LITELLM_REFUSAL_FALLBACK_PATTERNS has no effect on a streamed response yet (see the module
    docstring): the stream hold classes below are ported but not wired into a live stream. Logged
    on every streaming call while the env var is set, so the gap stays visible instead of silent."""
    if not _get_refusal_fallback_patterns():
        return
    from litellm._logging import verbose_router_logger

    verbose_router_logger.warning(
        "LITELLM_REFUSAL_FALLBACK_PATTERNS is set but has no effect on streamed responses "
        "(stream=True): a streamed refusal is not failed over to content_policy_fallbacks. "
        "Only non-streaming completions are covered."
    )


def _refusal_stream_hold_chars() -> int:
    """Max chars of streamed text held back while a refusal is ruled out.
    Overridable via LITELLM_REFUSAL_FALLBACK_STREAM_HOLD_CHARS; 0 disables the
    stream hold (streamed refusals then pass through unrescued)."""
    import os

    raw = os.environ.get("LITELLM_REFUSAL_FALLBACK_STREAM_HOLD_CHARS", "").strip()
    if not raw:
        return _REFUSAL_STREAM_HOLD_CHARS_DEFAULT
    try:
        return max(0, int(raw))
    except ValueError:
        return _REFUSAL_STREAM_HOLD_CHARS_DEFAULT


class _RefusalStreamHold:
    """Holds back the head of a streaming response until a model-level refusal
    can be ruled out, so content_policy_fallbacks can replace the stream before
    the client has seen any refusal text (same idea as Bedrock Guardrails'
    synchronous stream-processing mode).

    Once `hold_chars` of text accumulate with no pattern match -- or a tool-call
    delta arrives (refusals are plain text) -- all buffered chunks are flushed
    and the rest of the stream passes through unchecked, so long responses only
    pay the hold cost on their head. On a match, raises MidStreamFallbackError
    wrapping a ContentPolicyViolationError with is_pre_first_chunk=True:
    nothing has been yielded yet, so the router reruns the original messages
    against the content-policy fallback group and streams a clean answer.

    NOT YET WIRED into the live stream (see module docstring) -- exercised here
    only by its own unit tests pending that follow-up."""

    def __init__(self, patterns: list[str], hold_chars: int, model: str):
        self.patterns = patterns
        self.hold_chars = hold_chars
        self.model = model
        self.active = bool(patterns) and hold_chars > 0
        self._held: list[Any] = []
        self._text = ""
        # Reasoning deltas can't contain the client-visible refusal text but must
        # still advance the hold window, otherwise a reasoning model's whole
        # thinking phase (and thus its entire response) would be buffered.
        self._reasoning_len = 0

    def process(self, item: Any) -> list[Any]:
        """Return the chunks safe to emit for this stream item (possibly []).
        Raises MidStreamFallbackError when the held text matches a pattern."""
        if not self.active:
            return [item]
        self._held.append(item)
        delta_text, reasoning_len, saw_tool_call = self._delta_state(item)
        self._reasoning_len += reasoning_len
        if delta_text:
            self._text += delta_text
            if _text_matches_refusal_patterns(self._text, self.patterns):
                self._raise_refusal()
        if self._is_blocked_terminal(item):
            self._raise_refusal()
        if saw_tool_call or len(self._text) + self._reasoning_len >= self.hold_chars:
            return self._release()
        return []

    def flush(self) -> list[Any]:
        """End of stream: a short completion held to the end never matched."""
        if not self.active:
            return []
        return self._release()

    def _release(self) -> list[Any]:
        self.active = False
        held, self._held = self._held, []
        return held

    @staticmethod
    def _delta_state(item: Any) -> tuple[str, int, bool]:
        """(matchable text, reasoning char count, saw tool/function call) for one
        stream item. delta.refusal (OpenAI structured-output refusals) counts as
        matchable text alongside delta.content."""
        choices = getattr(item, "choices", None) or []
        if not choices:
            return "", 0, False
        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None)
        refusal = getattr(delta, "refusal", None)
        reasoning = getattr(delta, "reasoning_content", None)
        text = (content if isinstance(content, str) else "") + (refusal if isinstance(refusal, str) else "")
        saw_tool_call = bool(getattr(delta, "tool_calls", None)) or bool(getattr(delta, "function_call", None))
        return text, len(reasoning) if isinstance(reasoning, str) else 0, saw_tool_call

    @staticmethod
    def _is_blocked_terminal(item: Any) -> bool:
        """Hook for stream shapes whose provider expresses a content-policy block
        as a terminal EVENT rather than an exception (Responses API). The chat
        completions path has no such event -- finish_reason=content_filter mid-
        stream surfaces through the non-streaming bridge or provider errors."""
        return False

    def _raise_refusal(self) -> "NoReturn":
        from litellm.exceptions import MidStreamFallbackError

        message = "Streamed response matched a refusal fallback pattern."
        raise MidStreamFallbackError(
            message=message,
            model=self.model,
            llm_provider="",
            original_exception=litellm_exceptions.ContentPolicyViolationError(
                message=message,
                model=self.model,
                llm_provider="",
            ),
            generated_content="",
            is_pre_first_chunk=True,
        )


class _ResponsesRefusalStreamHold(_RefusalStreamHold):
    """Refusal stream hold for the Responses API event stream. Two triggers:

    1. Refusal TEXT: `response.output_text.delta` / `response.refusal.delta`
       deltas accumulate as matchable text (same patterns as chat).
    2. Blocked TERMINAL event: a `response.incomplete` whose
       `incomplete_details.reason == "content_filter"` (Azure's jailbreak /
       content filter kills the stream with HTTP 200 + this event, never an
       exception) -- while chunks are still held, treat it exactly like a
       content-policy violation so the fallback group re-serves the request.

    Reasoning-summary deltas advance the hold window without being matchable;
    function-call argument deltas release immediately (refusals are text).
    ResponsesAPIStreamEvents is a str-Enum, so `event.type` compares directly
    against the literal event strings."""

    @staticmethod
    def _delta_state(item: Any) -> "tuple[str, int, bool]":
        event_type = getattr(item, "type", None)
        if event_type in ("response.output_text.delta", "response.refusal.delta"):
            delta = getattr(item, "delta", None)
            return (delta if isinstance(delta, str) else ""), 0, False
        if event_type == "response.reasoning_summary_text.delta":
            delta = getattr(item, "delta", None)
            return "", len(delta) if isinstance(delta, str) else 0, False
        if event_type in (
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        ):
            return "", 0, True
        return "", 0, False

    @staticmethod
    def _is_blocked_terminal(item: Any) -> bool:
        if getattr(item, "type", None) != "response.incomplete":
            return False
        response_obj = getattr(item, "response", None)
        details = getattr(response_obj, "incomplete_details", None)
        reason = getattr(details, "reason", None)
        if reason is None and isinstance(details, dict):
            reason = details.get("reason")
        return reason == "content_filter"
