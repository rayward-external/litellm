import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import Final, cast

import pytest
from fastapi.responses import StreamingResponse

import litellm
from litellm.proxy.common_request_processing import create_response
from litellm.proxy.common_utils.sse_keepalive import (
    ANTHROPIC_PING_SSE_CHUNK,
    SSE_COMMENT_PING,
    SSE_COMMENT_PING_BYTES,
    UPSTREAM_IDLE_SSE_ERROR_TYPE,
    anthropic_upstream_idle_sse_chunk,
    resolve_ttft_keepalive_interval,
    split_complete_sse_frames,
    wrap_passthrough_sse_bytes_with_keepalive_pings,
    wrap_sse_stream_with_keepalive_pings,
)

MESSAGE_START_CHUNK: Final = 'data: {"type": "message_start"}\n\n'
TEXT_DELTA_CHUNK: Final = 'data: {"type": "content_block_delta"}\n\n'


async def _collect(stream: AsyncGenerator[str, None]) -> list[str]:
    return [chunk async for chunk in stream]


async def _drain_into(stream: AsyncGenerator[str, None], sink: list[str]) -> None:
    """Like ``_collect``, but what arrived survives the cancellation of a stream that never ends."""
    async for chunk in stream:
        sink.append(chunk)


# Distinguishes "the caller passed a disabling value" from "the caller never
# mentioned the cap at all", which are different code paths into the same state.
_CAP_ARGUMENT_OMITTED: Final = object()


@pytest.mark.parametrize("delimiter", [b"\n\n", b"\r\n\r\n", b"\r\r"])
def test_split_complete_sse_frames_recognizes_every_sse_frame_delimiter(delimiter: bytes):
    newline: Final = delimiter[: len(delimiter) // 2]
    frame: Final = b"event: response.created" + newline + b"data: {}" + delimiter
    tail: Final = b"data: partial"

    assert split_complete_sse_frames(frame + tail) == (frame, tail)


def test_split_complete_sse_frames_holds_bytes_with_no_complete_frame():
    assert split_complete_sse_frames(b"data: unterminated") == (b"", b"data: unterminated")


@pytest.mark.asyncio
async def test_pings_fill_mid_stream_silence_and_preserve_chunk_order():
    async def gappy_stream() -> AsyncGenerator[str, None]:
        yield MESSAGE_START_CHUNK
        await asyncio.sleep(0.3)
        yield TEXT_DELTA_CHUNK

    wrapped: Final = wrap_sse_stream_with_keepalive_pings(stream=gappy_stream(), ping_interval_seconds=0.05)
    collected: Final = [chunk async for chunk in wrapped]

    assert collected[0] == MESSAGE_START_CHUNK
    assert collected[-1] == TEXT_DELTA_CHUNK
    assert ANTHROPIC_PING_SSE_CHUNK in collected[1:-1]
    assert [chunk for chunk in collected if chunk != ANTHROPIC_PING_SSE_CHUNK] == [
        MESSAGE_START_CHUNK,
        TEXT_DELTA_CHUNK,
    ]


@pytest.mark.asyncio
async def test_ping_emitted_while_waiting_for_first_chunk():
    async def slow_start_stream() -> AsyncGenerator[str, None]:
        await asyncio.sleep(0.2)
        yield MESSAGE_START_CHUNK

    wrapped: Final = wrap_sse_stream_with_keepalive_pings(stream=slow_start_stream(), ping_interval_seconds=0.05)
    collected: Final = [chunk async for chunk in wrapped]

    assert collected[0] == ANTHROPIC_PING_SSE_CHUNK
    assert collected[-1] == MESSAGE_START_CHUNK


@pytest.mark.asyncio
async def test_no_pings_when_chunks_arrive_faster_than_interval():
    async def fast_stream() -> AsyncGenerator[str, None]:
        yield MESSAGE_START_CHUNK
        yield TEXT_DELTA_CHUNK
        yield TEXT_DELTA_CHUNK

    wrapped: Final = wrap_sse_stream_with_keepalive_pings(stream=fast_stream(), ping_interval_seconds=1.0)
    collected: Final = [chunk async for chunk in wrapped]

    assert collected == [MESSAGE_START_CHUNK, TEXT_DELTA_CHUNK, TEXT_DELTA_CHUNK]


@pytest.mark.asyncio
async def test_upstream_exception_propagates():
    async def failing_stream() -> AsyncGenerator[str, None]:
        yield MESSAGE_START_CHUNK
        raise ValueError("upstream broke")

    wrapped: Final = wrap_sse_stream_with_keepalive_pings(stream=failing_stream(), ping_interval_seconds=5.0)

    assert await wrapped.__anext__() == MESSAGE_START_CHUNK
    with pytest.raises(ValueError, match="upstream broke"):
        await wrapped.__anext__()


@pytest.mark.asyncio
async def test_aclose_mid_silence_cancels_upstream_and_runs_its_cleanup():
    upstream_cleaned_up: Final = asyncio.Event()

    async def hung_stream() -> AsyncGenerator[str, None]:
        try:
            yield MESSAGE_START_CHUNK
            await asyncio.Event().wait()
            yield TEXT_DELTA_CHUNK
        finally:
            upstream_cleaned_up.set()

    wrapped: Final = wrap_sse_stream_with_keepalive_pings(stream=hung_stream(), ping_interval_seconds=0.05)

    assert await wrapped.__anext__() == MESSAGE_START_CHUNK
    assert await wrapped.__anext__() == ANTHROPIC_PING_SSE_CHUNK
    await wrapped.aclose()

    assert upstream_cleaned_up.is_set()


@pytest.mark.asyncio
async def test_non_positive_interval_returns_stream_unwrapped():
    async def any_stream() -> AsyncGenerator[str, None]:
        yield MESSAGE_START_CHUNK

    stream: Final = any_stream()
    assert wrap_sse_stream_with_keepalive_pings(stream=stream, ping_interval_seconds=0) is stream
    await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_interval",
    [
        None,
        "abc",
        "",
        float("inf"),
        float("nan"),
        "-3",
        cast("float | str | None", [15]),
        cast("float | str | None", {"seconds": 15}),
    ],
)
async def test_invalid_config_interval_returns_stream_unwrapped(bad_interval: float | str | None):
    async def any_stream() -> AsyncGenerator[str, None]:
        yield MESSAGE_START_CHUNK

    stream: Final = any_stream()
    assert wrap_sse_stream_with_keepalive_pings(stream=stream, ping_interval_seconds=bad_interval) is stream
    await stream.aclose()


@pytest.mark.asyncio
async def test_numeric_string_interval_from_yaml_config_enables_pings():
    async def slow_start_stream() -> AsyncGenerator[str, None]:
        await asyncio.sleep(0.2)
        yield MESSAGE_START_CHUNK

    wrapped: Final = wrap_sse_stream_with_keepalive_pings(stream=slow_start_stream(), ping_interval_seconds="0.05")
    collected: Final = [chunk async for chunk in wrapped]

    assert collected[0] == ANTHROPIC_PING_SSE_CHUNK
    assert collected[-1] == MESSAGE_START_CHUNK


@pytest.mark.asyncio
async def test_create_response_streams_ping_first_for_slow_upstream():
    async def slow_start_stream() -> AsyncGenerator[str, None]:
        await asyncio.sleep(0.2)
        yield MESSAGE_START_CHUNK

    response: Final = await create_response(
        generator=wrap_sse_stream_with_keepalive_pings(stream=slow_start_stream(), ping_interval_seconds=0.05),
        media_type="text/event-stream",
        headers={},
    )

    assert isinstance(response, StreamingResponse)
    collected: Final = [chunk async for chunk in response.body_iterator]
    assert collected[0] == ANTHROPIC_PING_SSE_CHUNK
    assert collected[-1] == MESSAGE_START_CHUNK


# ---------------------------------------------------------------------------
# Upstream-idle cap.
#
# The per-write stall deadline (`litellm.stream_stalled_write_timeout_seconds`)
# can only fire when the peer applies backpressure. Measured behind a
# terminating proxy that buffers the response instead: the proxy kept accepting
# the app's writes for the whole run, every keepalive ping succeeded, no write
# ever stalled, and the requests ran to the platform's request cap having
# produced essentially no content - a hung upstream held open by our own pings.
# These cover the second, independent trigger for that shape, which measures
# upstream silence rather than anything about the transport.
#
# Timings are scaled down from the production defaults and every wait is
# bounded, because the defect under test is an unbounded wait: a test that hangs
# is a failure, not a slow pass.
# ---------------------------------------------------------------------------

# Small enough to keep the suite fast, large enough that several scheduler turns
# are never mistaken for silence.
IDLE_CAP_SECONDS: Final = 0.3
IDLE_PING_INTERVAL_SECONDS: Final = 0.02
TEST_DEADLINE_SECONDS: Final = 10.0


def _sse_event_payload(chunk: str) -> dict[str, object]:
    data_line: Final = next(line for line in chunk.splitlines() if line.startswith("data: "))
    return cast("dict[str, object]", json.loads(data_line[len("data: ") :]))


async def _hung_after(chunks: list[str], closed: asyncio.Event) -> AsyncGenerator[str, None]:
    """Deliver ``chunks``, then produce nothing ever again."""
    try:
        for chunk in chunks:
            yield chunk
        await asyncio.Event().wait()
    finally:
        closed.set()


def test_idle_error_chunk_is_a_parseable_anthropic_error_event():
    payload: Final = _sse_event_payload(anthropic_upstream_idle_sse_chunk(600.0))

    assert anthropic_upstream_idle_sse_chunk(600.0).startswith("event: error\n")
    assert payload["type"] == "error"
    assert cast("dict[str, object]", payload["error"])["type"] == UPSTREAM_IDLE_SSE_ERROR_TYPE
    assert "600s" in cast("str", cast("dict[str, object]", payload["error"])["message"])


@pytest.mark.asyncio
async def test_upstream_idle_cap_ends_a_hung_stream_with_a_terminal_error_event():
    closed: Final = asyncio.Event()
    wrapped: Final = wrap_sse_stream_with_keepalive_pings(
        stream=_hung_after([MESSAGE_START_CHUNK], closed),
        ping_interval_seconds=IDLE_PING_INTERVAL_SECONDS,
        max_upstream_idle_seconds=IDLE_CAP_SECONDS,
    )

    collected: Final = await asyncio.wait_for(_collect(wrapped), timeout=TEST_DEADLINE_SECONDS)

    assert collected[0] == MESSAGE_START_CHUNK
    assert collected[-1] == anthropic_upstream_idle_sse_chunk(IDLE_CAP_SECONDS)
    assert closed.is_set(), "the hung upstream was left open"


@pytest.mark.asyncio
async def test_keepalive_pings_do_not_reset_the_upstream_idle_cap():
    """The load-bearing property: the cap measures the UPSTREAM, not the wire.

    Pings are this proxy's own bytes and are written into exactly the silence the
    cap is counting. If emitting one restamped the clock the cap could never be
    reached on a ping-filled stream -- which is every stream it exists for -- and
    this collection would run until the test deadline instead of ending.
    """
    closed: Final = asyncio.Event()
    wrapped: Final = wrap_sse_stream_with_keepalive_pings(
        stream=_hung_after([MESSAGE_START_CHUNK], closed),
        ping_interval_seconds=IDLE_PING_INTERVAL_SECONDS,
        max_upstream_idle_seconds=IDLE_CAP_SECONDS,
    )

    started: Final = time.monotonic()
    collected: Final = await asyncio.wait_for(_collect(wrapped), timeout=TEST_DEADLINE_SECONDS)
    elapsed: Final = time.monotonic() - started

    pings: Final = [chunk for chunk in collected if chunk == ANTHROPIC_PING_SSE_CHUNK]
    assert len(pings) >= 5, "the cap must be reached across many ping cycles, not one"
    assert elapsed >= IDLE_CAP_SECONDS, "ended before the configured silence had elapsed"
    # Bounded well under `len(pings) * IDLE_PING_INTERVAL_SECONDS + IDLE_CAP_SECONDS`,
    # which is what a per-ping restamp would cost.
    assert elapsed < IDLE_CAP_SECONDS * 3


@pytest.mark.asyncio
async def test_slow_but_alive_upstream_is_not_capped():
    """Length is legitimate; only silence is not.

    The upstream here talks for four times the cap while never going quiet for
    longer than a fraction of it, and it is delivered whole.
    """

    async def slow_stream() -> AsyncGenerator[str, None]:
        for _ in range(8):
            await asyncio.sleep(IDLE_CAP_SECONDS / 2)
            yield TEXT_DELTA_CHUNK

    wrapped: Final = wrap_sse_stream_with_keepalive_pings(
        stream=slow_stream(),
        ping_interval_seconds=IDLE_PING_INTERVAL_SECONDS,
        max_upstream_idle_seconds=IDLE_CAP_SECONDS,
    )

    collected: Final = await asyncio.wait_for(_collect(wrapped), timeout=TEST_DEADLINE_SECONDS)

    assert [chunk for chunk in collected if chunk != ANTHROPIC_PING_SSE_CHUNK] == [TEXT_DELTA_CHUNK] * 8
    assert not any("event: error" in chunk for chunk in collected)


@pytest.mark.asyncio
async def test_a_consumer_that_stops_reading_is_not_counted_as_upstream_silence():
    """A slow reader is the other knob's business, not this one's.

    The consumer parks for longer than the whole cap between reads while the
    upstream answers well inside it. Timing the cap from anywhere but the start
    of the wait on the next upstream chunk folds that consumer pause into the
    measurement and fires on a perfectly healthy stream.

    The upstream is deliberately slower than the ping interval, so a ping - and
    with it the cap check - happens on every chunk; an instant upstream would
    skip the check entirely and the test would pass without measuring anything.
    """

    async def alive_upstream() -> AsyncGenerator[str, None]:
        for _ in range(3):
            await asyncio.sleep(IDLE_PING_INTERVAL_SECONDS * 3)
            yield TEXT_DELTA_CHUNK

    wrapped: Final = wrap_sse_stream_with_keepalive_pings(
        stream=alive_upstream(),
        ping_interval_seconds=IDLE_PING_INTERVAL_SECONDS,
        max_upstream_idle_seconds=IDLE_CAP_SECONDS,
    )

    collected: list[str] = []
    async for chunk in wrapped:
        collected.append(chunk)
        if chunk != ANTHROPIC_PING_SSE_CHUNK:
            await asyncio.sleep(IDLE_CAP_SECONDS * 1.5)

    assert [chunk for chunk in collected if chunk != ANTHROPIC_PING_SSE_CHUNK] == [TEXT_DELTA_CHUNK] * 3
    assert ANTHROPIC_PING_SSE_CHUNK in collected, "no ping fired, so the cap was never checked"
    assert not any("event: error" in chunk for chunk in collected)


def test_the_shipped_default_leaves_the_cap_off():
    """The mechanism ships; the number does not.

    The cap has to sit above the longest silence a deployment's own healthy
    streams produce, which is a fact about that deployment's traffic and not
    something this library can know. A non-None default here would end streams
    for every operator who never asked for one, so the value is opted into
    through config. Pinned, because changing it is a behaviour change for
    everybody rather than a tuning tweak.
    """
    assert litellm.stream_max_upstream_idle_seconds is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "disabled_cap",
    [_CAP_ARGUMENT_OMITTED, None, 0, "0", -1, "abc", float("nan")],
)
async def test_a_disabled_cap_leaves_the_stream_exactly_as_it_was(disabled_cap: object):
    """Off must mean untouched, not merely "does not fire".

    This is the default path, so it is the one that must be indistinguishable
    from the wrapper before the cap existed: a hung upstream is pinged forever,
    the only chunks on the wire are the upstream's own and the keepalive ping,
    no error frame is ever constructed, and nothing ends the stream.

    Covers the omitted argument alongside every spelling `coerce_keepalive_interval`
    rejects, so a coercion change cannot quietly arm the cap on a bad value.
    """
    closed: Final = asyncio.Event()
    wrapped: Final = wrap_sse_stream_with_keepalive_pings(
        stream=_hung_after([MESSAGE_START_CHUNK], closed),
        ping_interval_seconds=IDLE_PING_INTERVAL_SECONDS,
        **({} if disabled_cap is _CAP_ARGUMENT_OMITTED else {"max_upstream_idle_seconds": disabled_cap}),
    )
    collected: Final[list[str]] = []

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(_drain_into(wrapped, collected), timeout=IDLE_CAP_SECONDS * 3)

    assert collected[0] == MESSAGE_START_CHUNK
    assert set(collected[1:]) == {ANTHROPIC_PING_SSE_CHUNK}, "the disabled path put something new on the wire"
    assert len(collected) > 5, "the stream stopped being pinged well inside the window a cap would have used"


@pytest.mark.asyncio
async def test_a_caller_speaking_another_protocol_can_supply_its_own_error_frame():
    closed: Final = asyncio.Event()
    wrapped: Final = wrap_sse_stream_with_keepalive_pings(
        stream=_hung_after([MESSAGE_START_CHUNK], closed),
        ping_interval_seconds=IDLE_PING_INTERVAL_SECONDS,
        ping_chunk=SSE_COMMENT_PING,
        max_upstream_idle_seconds=IDLE_CAP_SECONDS,
        idle_error_chunk='data: {"error": "upstream idle"}\n\n',
    )

    collected: Final = await asyncio.wait_for(_collect(wrapped), timeout=TEST_DEADLINE_SECONDS)

    assert collected[-1] == 'data: {"error": "upstream idle"}\n\n'
    assert ANTHROPIC_PING_SSE_CHUNK not in collected


@pytest.mark.asyncio
async def test_cap_is_unreachable_when_keepalive_pings_are_disabled():
    """The measurement lives in the gap between two pings, so no pings, no cap."""
    closed: Final = asyncio.Event()
    stream: Final = _hung_after([MESSAGE_START_CHUNK], closed)

    wrapped: Final = wrap_sse_stream_with_keepalive_pings(
        stream=stream,
        ping_interval_seconds=0,
        max_upstream_idle_seconds=IDLE_CAP_SECONDS,
    )

    assert wrapped is stream
    await stream.aclose()


SSE_FRAME_BYTES: Final = b'event: content_block_delta\ndata: {"type": "content_block_delta"}\n\n'
BEDROCK_EVENT_STREAM_CONTENT_TYPE: Final = "application/vnd.amazon.eventstream"


@pytest.mark.asyncio
async def test_passthrough_ping_emitted_while_waiting_for_the_first_upstream_byte():
    async def slow_start_stream() -> AsyncGenerator[bytes, None]:
        await asyncio.sleep(0.2)
        yield SSE_FRAME_BYTES

    wrapped: Final = wrap_passthrough_sse_bytes_with_keepalive_pings(
        stream=slow_start_stream(),
        ping_interval_seconds=0.05,
        upstream_headers={"content-type": "text/event-stream"},
    )
    collected: Final = [chunk async for chunk in wrapped]

    assert collected[0] == SSE_COMMENT_PING_BYTES
    assert collected[-1] == SSE_FRAME_BYTES
    assert b"".join(c for c in collected if c != SSE_COMMENT_PING_BYTES) == SSE_FRAME_BYTES


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type", ["text/event-stream", "text/event-stream; charset=utf-8", "TEXT/Event-Stream"])
async def test_passthrough_wraps_every_spelling_of_the_sse_content_type(content_type: str):
    async def slow_start_stream() -> AsyncGenerator[bytes, None]:
        await asyncio.sleep(0.2)
        yield SSE_FRAME_BYTES

    wrapped: Final = wrap_passthrough_sse_bytes_with_keepalive_pings(
        stream=slow_start_stream(),
        ping_interval_seconds=0.05,
        upstream_headers={"content-type": content_type},
    )
    collected: Final = [chunk async for chunk in wrapped]

    assert SSE_COMMENT_PING_BYTES in collected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type",
    [BEDROCK_EVENT_STREAM_CONTENT_TYPE, "application/json", "application/x-ndjson", None, "text/event-streamish"],
)
async def test_passthrough_leaves_a_non_sse_transport_untouched(content_type: str | None):
    """A comment spliced into a binary transport (e.g. an AWS event stream) corrupts it."""

    async def any_stream() -> AsyncGenerator[bytes, None]:
        yield SSE_FRAME_BYTES

    stream: Final = any_stream()
    assert (
        wrap_passthrough_sse_bytes_with_keepalive_pings(
            stream=stream,
            ping_interval_seconds=0.05,
            upstream_headers={} if content_type is None else {"content-type": content_type},
        )
        is stream
    )
    await stream.aclose()


@pytest.mark.asyncio
async def test_passthrough_ping_is_never_spliced_into_a_half_delivered_frame():
    """Relayed chunks are raw transport reads, so an upstream can stall mid-frame."""

    async def stalls_mid_frame() -> AsyncGenerator[bytes, None]:
        yield b'event: content_block_delta\ndata: {"partial":'
        await asyncio.sleep(0.3)
        yield b"1}\n\n"

    wrapped: Final = wrap_passthrough_sse_bytes_with_keepalive_pings(
        stream=stalls_mid_frame(),
        ping_interval_seconds=0.05,
        upstream_headers={"content-type": "text/event-stream"},
    )
    collected: Final = [chunk async for chunk in wrapped]

    assert SSE_COMMENT_PING_BYTES not in collected
    assert b"".join(collected) == b'event: content_block_delta\ndata: {"partial":1}\n\n'


@pytest.mark.asyncio
async def test_passthrough_ping_resumes_once_the_stalled_frame_completes():
    async def stalls_mid_frame_then_at_boundary() -> AsyncGenerator[bytes, None]:
        yield b'event: content_block_delta\ndata: {"partial":'
        await asyncio.sleep(0.2)
        yield b"1}\n\n"
        await asyncio.sleep(0.2)
        yield SSE_FRAME_BYTES

    wrapped: Final = wrap_passthrough_sse_bytes_with_keepalive_pings(
        stream=stalls_mid_frame_then_at_boundary(),
        ping_interval_seconds=0.05,
        upstream_headers={"content-type": "text/event-stream"},
    )
    collected: Final = [chunk async for chunk in wrapped]

    ping_index: Final = collected.index(SSE_COMMENT_PING_BYTES)
    assert collected[:ping_index] == [b'event: content_block_delta\ndata: {"partial":', b"1}\n\n"]
    assert collected[-1] == SSE_FRAME_BYTES


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_interval", [None, 0, "abc", float("inf"), float("nan"), "-3"])
async def test_passthrough_invalid_or_disabled_interval_returns_stream_unwrapped(bad_interval: float | str | None):
    async def any_stream() -> AsyncGenerator[bytes, None]:
        yield SSE_FRAME_BYTES

    stream: Final = any_stream()
    assert (
        wrap_passthrough_sse_bytes_with_keepalive_pings(
            stream=stream,
            ping_interval_seconds=bad_interval,
            upstream_headers={"content-type": "text/event-stream"},
        )
        is stream
    )
    await stream.aclose()


@pytest.mark.asyncio
async def test_passthrough_aclose_mid_silence_cancels_upstream_and_runs_its_cleanup():
    upstream_cleaned_up: Final = asyncio.Event()

    async def hung_stream() -> AsyncGenerator[bytes, None]:
        try:
            yield SSE_FRAME_BYTES
            await asyncio.Event().wait()
        finally:
            upstream_cleaned_up.set()

    wrapped: Final = wrap_passthrough_sse_bytes_with_keepalive_pings(
        stream=hung_stream(),
        ping_interval_seconds=0.05,
        upstream_headers={"content-type": "text/event-stream"},
    )

    assert await wrapped.__anext__() == SSE_FRAME_BYTES
    assert await wrapped.__anext__() == SSE_COMMENT_PING_BYTES
    await wrapped.aclose()

    assert upstream_cleaned_up.is_set()


@pytest.mark.asyncio
async def test_passthrough_upstream_exception_propagates():
    async def failing_stream() -> AsyncGenerator[bytes, None]:
        yield SSE_FRAME_BYTES
        raise ValueError("upstream broke")

    wrapped: Final = wrap_passthrough_sse_bytes_with_keepalive_pings(
        stream=failing_stream(),
        ping_interval_seconds=5.0,
        upstream_headers={"content-type": "text/event-stream"},
    )

    assert await wrapped.__anext__() == SSE_FRAME_BYTES
    with pytest.raises(ValueError, match="upstream broke"):
        await wrapped.__anext__()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "split_frame",
    [
        (b'data: {"a": 1}\n', b"\n"),
        (b'data: {"a": 1}\r\n', b"\r\n"),
        (b'data: {"a": 1}\r', b"\n\r\n"),
        (b'data: {"a": 1}\r', b"\r"),
        (b'data: {"a": 1}\r\r', b""),
        (b'data: {"a": 1}\n\n', b""),
    ],
    ids=["lf-split", "crlf-split", "crlf-mixed-split", "cr-only-split", "cr-only-whole", "not-split"],
)
async def test_passthrough_sees_a_frame_delimiter_split_across_transport_chunks(split_frame):
    """A raw transport read can end mid-delimiter. Testing only the latest chunk
    would leave the stream looking permanently mid-frame, silently disabling the
    keepalive the operator configured."""

    async def split_delimiter_stream() -> AsyncGenerator[bytes, None]:
        for part in split_frame:
            if part:
                yield part
        await asyncio.sleep(0.3)
        yield SSE_FRAME_BYTES

    wrapped: Final = wrap_passthrough_sse_bytes_with_keepalive_pings(
        stream=split_delimiter_stream(),
        ping_interval_seconds=0.05,
        upstream_headers={"content-type": "text/event-stream"},
    )
    collected: Final = [chunk async for chunk in wrapped]

    assert SSE_COMMENT_PING_BYTES in collected
    assert b"".join(c for c in collected if c != SSE_COMMENT_PING_BYTES) == b"".join(split_frame) + SSE_FRAME_BYTES


def _deployment(keepalive_seconds=..., model="openai/gpt-4o"):
    params = {"model": model}
    if keepalive_seconds is not ...:
        params["keepalive_seconds"] = keepalive_seconds
    return {"model_name": "m", "litellm_params": params}


@pytest.mark.parametrize(
    "deployments, global_interval, expected, why",
    [
        ([], 30.0, 30.0, "no deployments known, the global applies"),
        ([_deployment()], 30.0, 30.0, "nothing configured, the global applies"),
        ([_deployment(0)], 30.0, None, "an operator's explicit 0 is a hard disable the global cannot lift"),
        ([_deployment("0")], 30.0, None, "the same, written as a yaml string"),
        ([_deployment(15)], 30.0, 15.0, "a deployment value wins over the global"),
        ([_deployment(15), _deployment(15)], 30.0, 15.0, "agreeing deployments are trusted"),
        ([_deployment(15), _deployment(60)], 30.0, 30.0, "disagreeing deployments fall back to the global"),
        ([_deployment(0), _deployment(30)], 30.0, 30.0, "a partial disable is not trusted before one is chosen"),
        ([_deployment(15)], None, 15.0, "a deployment value applies with no global set"),
        ([_deployment()], None, None, "nothing anywhere leaves it off"),
    ],
)
def test_ttft_interval_resolves_through_the_deployments_it_could_land_on(
    deployments, global_interval, expected, why
):
    assert resolve_ttft_keepalive_interval(deployments, global_interval) == expected, why
