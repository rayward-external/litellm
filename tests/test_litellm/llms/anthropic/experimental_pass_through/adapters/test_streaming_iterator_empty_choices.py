"""
Regression tests for Azure OpenAI streaming routed through `/v1/messages`.

Azure OpenAI's streaming emits at least one chunk with an EMPTY ``choices`` list
— its content-filter / prompt-annotations chunk, and/or a usage-only final chunk.
``azure_ai`` (Foundry) and other providers do not, which is why ONLY
``/azure/v1/messages`` streaming failed: ``AnthropicStreamWrapper`` dereferenced
``chunk.choices[0]`` without guarding the empty list, raising ``IndexError`` mid
-stream. That surfaced as a 500 error event and the terminal ``message_stop`` was
never sent (observed live: ``message_start`` -> ``content_block_start`` -> 500).

The adapter must instead skip a choice-less chunk (letting any ``usage`` it carries
still reach the final ``message_delta``) and always emit the full terminal
sequence: ``content_block_stop`` -> ``message_delta`` -> ``message_stop``.
"""

import asyncio
import json
from typing import AsyncIterator, List

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import (
    Delta,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)


def _content_chunk(content: str) -> ModelResponseStream:
    return ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(content=content), finish_reason=None)],
    )


def _empty_choices_chunk(usage: Usage = None) -> ModelResponseStream:
    """Mimic Azure's content-filter / usage-only chunk: an empty ``choices`` list."""
    chunk = ModelResponseStream(choices=[])
    if usage is not None:
        chunk.usage = usage
    return chunk


def _finish_chunk() -> ModelResponseStream:
    return ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(), finish_reason="stop")],
    )


def _azure_stream() -> List[ModelResponseStream]:
    """A representative Azure ``/v1/messages`` stream.

    content delta -> empty-choices content-filter chunk (no usage) -> finish chunk
    -> usage-only chunk (empty choices, carries usage).
    """
    return [
        _content_chunk("Hello from Azure."),
        _empty_choices_chunk(),  # content-filter chunk: empty choices, no usage
        _finish_chunk(),
        _empty_choices_chunk(usage=Usage(prompt_tokens=11, completion_tokens=7, total_tokens=18)),
    ]


def _collect_async(wrapper: AnthropicStreamWrapper) -> str:
    async def _run() -> str:
        out = []
        async for raw in wrapper.async_anthropic_sse_wrapper():
            out.append(raw.decode() if isinstance(raw, bytes) else raw)
        return "".join(out)

    return asyncio.run(_run())


def _events(sse: str) -> List[dict]:
    return [
        json.loads(line[len("data: ") :])
        for block in sse.split("\n\n")
        for line in block.splitlines()
        if line.startswith("data: ")
    ]


def test_azure_empty_choices_async_emits_full_terminal_sequence():
    """An Azure-style async stream with an empty-choices chunk must NOT raise
    IndexError and must end with the full terminal sequence."""

    async def _aiter() -> "AsyncIterator[ModelResponseStream]":
        for chunk in _azure_stream():
            yield chunk

    wrapper = AnthropicStreamWrapper(completion_stream=_aiter(), model="gpt-4o")
    sse = _collect_async(wrapper)

    types = [e.get("type") for e in _events(sse)]

    # No error event was emitted (the prod symptom was a 500 mid-stream).
    assert "error" not in types
    # Full, correctly ordered terminal sequence.
    assert "content_block_stop" in types
    assert "message_delta" in types
    assert "message_stop" in types
    assert types[-1] == "message_stop"
    stop_idx = types.index("content_block_stop")
    assert stop_idx < types.index("message_delta") < types.index("message_stop")


def test_azure_empty_choices_usage_reaches_message_delta():
    """Usage carried on the empty-choices final chunk must still reach the
    final ``message_delta`` (it must NOT be dropped)."""

    async def _aiter() -> "AsyncIterator[ModelResponseStream]":
        for chunk in _azure_stream():
            yield chunk

    wrapper = AnthropicStreamWrapper(completion_stream=_aiter(), model="gpt-4o")
    sse = _collect_async(wrapper)

    message_delta = next(e for e in _events(sse) if e.get("type") == "message_delta")
    assert message_delta["delta"]["stop_reason"] is not None
    assert message_delta["usage"]["input_tokens"] == 11
    assert message_delta["usage"]["output_tokens"] == 7


def test_azure_empty_choices_sync_emits_full_terminal_sequence():
    """The sync ``__next__`` path is guarded identically."""
    wrapper = AnthropicStreamWrapper(completion_stream=iter(_azure_stream()), model="gpt-4o")
    events = list(wrapper)
    types = [e.get("type") for e in events]

    assert "error" not in types
    assert types[-1] == "message_stop"
    assert types.index("content_block_stop") < types.index("message_delta") < types.index("message_stop")

    message_delta = next(e for e in events if e.get("type") == "message_delta")
    assert message_delta["usage"]["input_tokens"] == 11
    assert message_delta["usage"]["output_tokens"] == 7
