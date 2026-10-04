"""Regression tests for a stalled streaming client on POST /v1/messages.

Measured production shape (llm-gateway-infra#688): a streaming `/v1/messages`
request whose client vanished at ~766s stayed alive on the backend for the full
3600s platform request cap, logged HTTP 200, and wrote more bytes than the hop
in front of it ever forwarded. Twenty-two occurrences in thirty days.

The cause is not the cleanup chain - that chain is complete, and covered by
`TestStreamCloseOnDisconnect` in test_common_request_processing.py. It is that
nothing ever *triggers* it. Starlette's `StreamingResponse` learns about a
client disconnect only from an ASGI `http.disconnect` message, and an ASGI
server only synthesizes one when the socket to its *immediate peer* closes.
When a terminating proxy sits in front (Cloud Run, ALB, nginx) that peer can
stop consuming while holding its own socket open: `receive()` then stays silent
forever and `await send(...)` parks in the server's flow-control drain.

Measured against the deployed pins (starlette 1.3.1 / uvicorn 0.51.0) with a raw
socket against a live uvicorn: a client that closes its socket is detected in
milliseconds; a client that merely stops reading is never detected at all, and
the response generator is still open and suspended mid-write minutes later.

These tests drive the streaming request the way `/v1/messages` does - through
`ProxyBaseLLMRequestProcessing.base_process_llm_request(route_type=
"anthropic_messages")`, which is the entire body of the `anthropic_response`
handler for a streaming request - and then run the response it returns as an
ASGI app against a client that stops reading. The ASGI layer is the only place
"the client stopped reading" can be expressed at all; entering at the processor
rather than at the HTTP router keeps the test off `litellm.proxy.proxy_server`,
whose import dominates the runtime of anything that touches it.
"""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import (
    ProxyBaseLLMRequestProcessing,
    create_response,
)

pytestmark = pytest.mark.asyncio


# The deadline the tests run with. Small enough to keep them fast, large enough
# that the several scheduler turns a healthy write takes are never mistaken for
# a stall.
STALL_TIMEOUT_SECONDS = 0.25
# Every assertion is bounded: the defect under test is an unbounded wait, so a
# test that hangs is a failure, not a slow pass.
TEST_DEADLINE_SECONDS = 10.0
# The upstream-idle cap the tests run with, and the ping cadence it is measured
# on. Both scaled down from the shipped defaults (600s / 15s) by the same order.
IDLE_CAP_SECONDS = 0.3
IDLE_PING_INTERVAL_SECONDS = 0.02


class _FakeAnthropicStream:
    """A slow upstream that records when it is closed.

    Shaped like the native Anthropic streaming iterator: an async iterator over
    SSE-ready chunk dicts, with the `aclose()` that
    `async_streaming_data_generator` calls on teardown.
    """

    def __init__(self) -> None:
        self.closed = asyncio.Event()
        self.chunks_yielded = 0

    def __aiter__(self) -> "_FakeAnthropicStream":
        return self

    async def __anext__(self) -> dict[str, Any]:
        self.chunks_yielded += 1
        # Slow enough to be a real generation, fast enough that the keepalive
        # ping wrapper is not what is being measured.
        await asyncio.sleep(0.01)
        return {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": f"tok{self.chunks_yielded}"},
        }

    async def aclose(self) -> None:
        self.closed.set()


class _HungAnthropicStream(_FakeAnthropicStream):
    """An upstream that delivers `chunks_before_hang` chunks and then goes quiet.

    The measured production shape: 29 of 33 orphaned requests delivered under
    4,746 total wire bytes across 555-884s, against ~5.5 KB/s for a healthy
    stream on the same route. They were not truncated long answers - they were
    hung upstreams held open by the proxy's own keepalive pings until the
    platform's request cap.
    """

    def __init__(self, chunks_before_hang: int = 0) -> None:
        super().__init__()
        self._chunks_before_hang = chunks_before_hang

    async def __anext__(self) -> dict[str, Any]:
        if self.chunks_yielded >= self._chunks_before_hang:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")
        return await super().__anext__()


class _StallingClient:
    """An ASGI client that accepts `read_chunks` body messages and then stops.

    It never closes: `receive()` stays silent for the life of the request, which
    is what the app sees when the hop in front of it stops consuming without
    tearing down its own connection. Nothing here ever sends `http.disconnect` -
    that is the whole point.
    """

    def __init__(self, read_chunks: int) -> None:
        self._read_chunks = read_chunks
        self.chunks_received: list[bytes] = []
        self.started = False
        self.stalled = asyncio.Event()

    async def receive(self) -> dict[str, Any]:
        # The transport has nothing to report, forever.
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def send(self, message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            self.started = True
            return
        if message["type"] != "http.response.body":
            return
        if len(self.chunks_received) < self._read_chunks:
            self.chunks_received.append(message.get("body", b""))
            return
        # Peer stopped reading: this write never drains.
        self.stalled.set()
        await asyncio.Event().wait()


def _asgi_scope() -> dict[str, Any]:
    # spec_version below 2.4 is what uvicorn 0.51.0 reports, which is the branch
    # of StreamingResponse.__call__ that relies entirely on receive().
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/v1/messages",
        "query_string": b"beta=true",
        "headers": [],
    }


def _logging_obj():
    """The real logging object, so response-header and cost assembly run for real."""
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    return LiteLLMLoggingObj(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="anthropic_messages",
        start_time=None,
        litellm_call_id="test-call-id",
        function_id="test-function-id",
    )


def _proxy_logging_obj() -> MagicMock:
    obj = MagicMock()
    obj.during_call_hook = AsyncMock(return_value=None)
    obj.update_request_status = AsyncMock(return_value=None)
    obj.post_call_failure_hook = AsyncMock(return_value=None)
    obj.post_call_response_headers_hook = AsyncMock(return_value={})
    obj._arelease_max_parallel_requests_on_disconnect = AsyncMock(return_value=None)
    obj.async_post_call_streaming_hook = AsyncMock(side_effect=lambda response, **_: response)
    # Returns the async iterable synchronously - it is consumed with `async for`,
    # not awaited.
    obj.async_post_call_streaming_iterator_hook = MagicMock(side_effect=lambda response, **_: response)
    return obj


async def _stream_v1_messages_to(
    client: "_StallingClient | _ReadingClient",
    upstream: _FakeAnthropicStream,
    proxy_logging_obj: MagicMock | None = None,
) -> dict[str, Any]:
    """Run the /v1/messages streaming path and serve its response to `client`.

    Returns the request data the run mutated, which is the dict the streaming
    teardown stamps its termination metadata onto and the one every logging
    callback reads.
    """
    data = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 4096,
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }
    processor = ProxyBaseLLMRequestProcessing(data=data)

    async def _fake_pre_call(self, **kwargs):
        return self.data, _logging_obj()

    async def _fake_llm_call() -> _FakeAnthropicStream:
        return upstream

    with (
        patch.object(  # test-quality-ok: the subject is the ASGI write path below routing; the real pre-call logic needs a live router and DB and would not change what is measured
            ProxyBaseLLMRequestProcessing,
            "common_processing_pre_call_logic",
            autospec=True,
            side_effect=_fake_pre_call,
        ),
        patch(  # test-quality-ok: this IS the boundary under test - it substitutes the provider connection whose closure the assertions are about
            "litellm.proxy.common_request_processing.route_request",
            new=AsyncMock(return_value=_fake_llm_call()),
        ),
    ):
        # A real Request over the same silent ASGI channel: the disconnect
        # bookkeeping in the streaming generator's teardown reads it, and a
        # MagicMock there would hide a real failure behind a swallowed aclose.
        request = Request(_asgi_scope(), receive=client.receive)

        response = await processor.base_process_llm_request(
            request=request,
            fastapi_response=MagicMock(),
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-test", user_id="u1"),
            route_type="anthropic_messages",
            proxy_logging_obj=proxy_logging_obj or _proxy_logging_obj(),
            general_settings={},
            proxy_config=MagicMock(),
            llm_router=None,
        )

    await response(_asgi_scope(), client.receive, client.send)
    return data


async def test_stalled_client_closes_the_upstream_anthropic_stream(monkeypatch):
    """The regression: no http.disconnect ever arrives, yet the upstream closes."""
    monkeypatch.setattr(litellm, "stream_stalled_write_timeout_seconds", STALL_TIMEOUT_SECONDS)
    upstream = _FakeAnthropicStream()
    client = _StallingClient(read_chunks=3)

    await asyncio.wait_for(
        _stream_v1_messages_to(client, upstream), timeout=TEST_DEADLINE_SECONDS
    )

    assert client.started, "the response must have opened before the client stalled"
    assert client.stalled.is_set(), "the test never reached the stalled write"
    assert upstream.closed.is_set(), (
        "the upstream Anthropic stream was left open after the client stopped reading"
    )


async def test_without_the_deadline_the_request_is_wedged(monkeypatch):
    """Pin the defect itself: with the deadline off, nothing ends the request.

    This is the pre-fix behaviour, and the reason the production requests ran to
    the platform cap. Kept so a change that silently disables the deadline shows
    up as a behaviour change rather than a quietly passing suite.
    """
    monkeypatch.setattr(litellm, "stream_stalled_write_timeout_seconds", None)
    upstream = _FakeAnthropicStream()
    client = _StallingClient(read_chunks=3)

    # Nothing in the app ends this request; only the test's own cancellation
    # does, which is the wedge the production requests sat in until Cloud Run's
    # 3600s cap fired.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(_stream_v1_messages_to(client, upstream), timeout=1.0)

    assert client.stalled.is_set()


async def test_stalled_client_on_openai_sse_path_closes_the_upstream(monkeypatch):
    """/v1/chat/completions and /v1/responses reach the same response object.

    Both build their SSE response through `create_response`, so the deadline
    covers them for the same reason and by the same code.
    """
    monkeypatch.setattr(litellm, "stream_stalled_write_timeout_seconds", STALL_TIMEOUT_SECONDS)
    closed = asyncio.Event()

    async def upstream():
        try:
            i = 0
            while True:
                i += 1
                yield f"data: {json.dumps({'id': i})}\n\n"
                await asyncio.sleep(0.01)
        finally:
            closed.set()

    response = await create_response(
        generator=upstream(), media_type="text/event-stream", headers={}
    )
    client = _StallingClient(read_chunks=3)

    await asyncio.wait_for(
        response(_asgi_scope(), client.receive, client.send),
        timeout=TEST_DEADLINE_SECONDS,
    )

    assert client.stalled.is_set()
    assert closed.is_set()


# ---------------------------------------------------------------------------
# Upstream-idle cap
#
# The deadline above needs the peer to apply backpressure. Measured behind a
# terminating proxy that buffers the response instead: the proxy kept accepting
# the app's writes for the whole run, every keepalive ping succeeded, no write
# ever stalled, and the requests ran to the platform's request cap having
# delivered essentially nothing - a hung upstream held open by our own pings.
#
# `litellm.stream_max_upstream_idle_seconds` is the second, independent trigger
# for that shape. It measures the model's side of the proxy, so it does not care
# what the transport does or does not report.
# ---------------------------------------------------------------------------


class _ReadingClient:
    """An ASGI client that reads the whole response and records every message.

    The opposite of `_StallingClient`: this peer always drains, so a write never
    stalls and the per-write deadline can never fire. Whatever ends the request
    here is the upstream-idle cap.
    """

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def receive(self) -> dict[str, Any]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def send(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    @property
    def body(self) -> str:
        return "".join(m.get("body", b"").decode() for m in self.messages if m["type"] == "http.response.body")

    @property
    def ended_cleanly(self) -> bool:
        """Whether Starlette finished the response rather than being torn down.

        A `StreamWriteStalled` raise abandons `stream_response` before its
        terminating empty-body frame, so this is False when the per-write
        deadline is what ended the request.
        """
        return bool(self.messages) and not self.messages[-1].get("more_body", False)


def _use_scaled_idle_cap(monkeypatch, cap_seconds: float | None) -> None:
    monkeypatch.setattr(litellm, "stream_max_upstream_idle_seconds", cap_seconds)
    monkeypatch.setattr(litellm, "anthropic_sse_ping_interval_seconds", IDLE_PING_INTERVAL_SECONDS)


async def test_hung_upstream_ends_with_a_terminal_error_event(monkeypatch):
    """The reframed regression: nothing about the client is wrong, the model is.

    `stream_stalled_write_timeout_seconds` is left at its shipped default, which
    is three orders of magnitude past this test's deadline - so if anything ends
    this request it is the idle cap, and the two triggers demonstrably coexist.
    """
    _use_scaled_idle_cap(monkeypatch, IDLE_CAP_SECONDS)
    upstream = _HungAnthropicStream(chunks_before_hang=1)
    client = _ReadingClient()

    await asyncio.wait_for(_stream_v1_messages_to(client, upstream), timeout=TEST_DEADLINE_SECONDS)

    assert "tok1" in client.body, "the real chunk the upstream did produce was lost"
    assert '"type": "error"' in client.body, "the client was left to infer the truncation"
    assert '"timeout_error"' in client.body
    assert upstream.closed.is_set(), "the hung upstream connection was left open"
    assert client.ended_cleanly, "torn down by the per-write deadline rather than ended by the idle cap"


async def test_without_the_idle_cap_a_hung_upstream_wedges_the_request(monkeypatch):
    """Pin the defect itself: a reading client and a dead model, and nothing ends it.

    This is the production shape - the request survived to the platform's request
    cap. Kept so a change that silently disables the cap shows up as a behaviour
    change rather than a quietly passing suite.
    """
    _use_scaled_idle_cap(monkeypatch, None)
    upstream = _HungAnthropicStream(chunks_before_hang=1)
    client = _ReadingClient()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(_stream_v1_messages_to(client, upstream), timeout=1.0)

    assert "tok1" in client.body, "the stream never opened, so nothing was under test"


async def test_a_hung_upstream_that_produced_nothing_refunds_the_budget_reservation(monkeypatch):
    """The teardown accounting the cap inherits, on the shape that dominates.

    29 of the 33 measured requests delivered no content at all, so the reservation
    taken up front is owed back in full: `async_streaming_data_generator` refunds
    only when no provider output was ever delivered, and the cap must reach that
    path rather than around it.
    """
    _use_scaled_idle_cap(monkeypatch, IDLE_CAP_SECONDS)
    refund = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.budget_reservation.release_budget_reservation_on_cancel",
        refund,
    )
    upstream = _HungAnthropicStream(chunks_before_hang=0)

    await asyncio.wait_for(_stream_v1_messages_to(_ReadingClient(), upstream), timeout=TEST_DEADLINE_SECONDS)

    assert refund.await_count == 1, "a request that produced nothing kept its reservation"
    assert upstream.closed.is_set()


async def test_delivered_output_is_not_refunded_when_the_idle_cap_fires(monkeypatch):
    """The other half of the same rule: a partial stream still owes its spend."""
    _use_scaled_idle_cap(monkeypatch, IDLE_CAP_SECONDS)
    refund = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.budget_reservation.release_budget_reservation_on_cancel",
        refund,
    )
    upstream = _HungAnthropicStream(chunks_before_hang=3)

    await asyncio.wait_for(_stream_v1_messages_to(_ReadingClient(), upstream), timeout=TEST_DEADLINE_SECONDS)

    assert refund.await_count == 0, "output was delivered, so nothing is owed back"


async def test_the_write_deadline_still_fires_with_the_idle_cap_enabled(monkeypatch):
    """The two triggers are independent: a live upstream and a client that left.

    The upstream here talks continuously, so the idle cap can never fire; the
    stall deadline must still end the request exactly as before.
    """
    monkeypatch.setattr(litellm, "stream_stalled_write_timeout_seconds", STALL_TIMEOUT_SECONDS)
    _use_scaled_idle_cap(monkeypatch, IDLE_CAP_SECONDS)
    upstream = _FakeAnthropicStream()
    client = _StallingClient(read_chunks=3)

    await asyncio.wait_for(_stream_v1_messages_to(client, upstream), timeout=TEST_DEADLINE_SECONDS)

    assert client.stalled.is_set(), "the test never reached the stalled write"
    assert upstream.closed.is_set()


class _FiniteAnthropicStream(_FakeAnthropicStream):
    """A healthy upstream that talks steadily for `chunks` chunks and then ends."""

    def __init__(self, chunks: int, gap_seconds: float) -> None:
        super().__init__()
        self._chunks = chunks
        self._gap_seconds = gap_seconds

    async def __anext__(self) -> dict[str, Any]:
        if self.chunks_yielded >= self._chunks:
            raise StopAsyncIteration
        await asyncio.sleep(self._gap_seconds)
        self.chunks_yielded += 1
        return {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": f"tok{self.chunks_yielded}"},
        }


def _buffering_proxy_logging_obj() -> MagicMock:
    """A proxy logging object whose iterator hook withholds every chunk to end of stream.

    The shape `streaming_buffer_until_moderated` produces: the hook consumes a
    perfectly healthy upstream continuously and deliberately yields nothing until
    its end-of-stream moderation pass has completed, then releases the originals.
    """
    obj = _proxy_logging_obj()

    async def _buffer_until_end_of_stream(response: Any) -> Any:
        held = [chunk async for chunk in response]
        for chunk in held:
            yield chunk

    obj.async_post_call_streaming_iterator_hook = MagicMock(
        side_effect=lambda response, **_: _buffer_until_end_of_stream(response)
    )
    return obj


async def test_a_buffering_guardrail_is_not_mistaken_for_a_silent_upstream(monkeypatch):
    """Measuring the processed stream measures the wrong thing.

    The wrapper sits above the post-call hooks, so a guardrail buffering until
    end-of-stream moderation makes a talking model look mute for the whole
    response - and the cap, reading that, would end a healthy stream and blame
    the provider. The upstream here withholds output for three times the cap
    while never being quiet, and must be delivered whole.
    """
    _use_scaled_idle_cap(monkeypatch, IDLE_CAP_SECONDS)
    upstream = _FiniteAnthropicStream(chunks=12, gap_seconds=IDLE_CAP_SECONDS / 4)
    client = _ReadingClient()

    await asyncio.wait_for(
        _stream_v1_messages_to(client, upstream, proxy_logging_obj=_buffering_proxy_logging_obj()),
        timeout=TEST_DEADLINE_SECONDS,
    )

    assert "tok12" in client.body, "the buffered response was truncated by the idle cap"
    assert '"timeout_error"' not in client.body, "a healthy upstream was ended as silent"
    assert client.ended_cleanly


async def test_a_dead_upstream_behind_the_same_buffering_guardrail_still_ends(monkeypatch):
    """The other half: buffering must not become a way to disarm the cap.

    Same hook, same silence on the wire - but nothing is being produced behind
    it, which is exactly the production shape. Without this the test above would
    pass just as well against a cap that had been switched off.
    """
    _use_scaled_idle_cap(monkeypatch, IDLE_CAP_SECONDS)
    upstream = _HungAnthropicStream(chunks_before_hang=1)
    client = _ReadingClient()

    await asyncio.wait_for(
        _stream_v1_messages_to(client, upstream, proxy_logging_obj=_buffering_proxy_logging_obj()),
        timeout=TEST_DEADLINE_SECONDS,
    )

    assert '"timeout_error"' in client.body, "a hung upstream survived because a hook was buffering"
    assert upstream.closed.is_set()


async def test_the_idle_cap_is_not_recorded_as_a_client_disconnect(monkeypatch):
    """#688 exists because the telemetry lied. This is the fix not re-telling it.

    The client here reads every byte and never goes away; the model is what
    failed. Recording that as a client disconnect - which is what the cap's own
    teardown does by default, since closing the generator is indistinguishable
    from a client leaving - would rebuild the same blind spot one layer up, with
    the proxy's own timeouts hidden inside the 499s.
    """
    _use_scaled_idle_cap(monkeypatch, IDLE_CAP_SECONDS)
    upstream = _HungAnthropicStream(chunks_before_hang=1)

    data = await asyncio.wait_for(
        _stream_v1_messages_to(_ReadingClient(), upstream), timeout=TEST_DEADLINE_SECONDS
    )

    metadata = data["metadata"]
    assert metadata.get("stream_ended_upstream_idle") is True, "the cap's own termination went unrecorded"
    assert "client_disconnected" not in metadata, "the proxy's timeout was filed as the client leaving"
    assert metadata["error_information"]["error_code"] == "504"
    assert metadata["error_information"]["error_class"] == "UpstreamStreamIdle"


async def test_a_client_that_stops_reading_is_still_recorded_as_a_client_disconnect(monkeypatch):
    """The contrast that makes the flag above worth reading.

    A separate signal that fired on every teardown would separate nothing, so the
    real disconnect - live upstream, client gone - has to keep its own 499.
    """
    monkeypatch.setattr(litellm, "stream_stalled_write_timeout_seconds", STALL_TIMEOUT_SECONDS)
    _use_scaled_idle_cap(monkeypatch, IDLE_CAP_SECONDS)
    upstream = _FakeAnthropicStream()
    client = _StallingClient(read_chunks=3)

    data = await asyncio.wait_for(_stream_v1_messages_to(client, upstream), timeout=TEST_DEADLINE_SECONDS)

    metadata = data["metadata"]
    assert metadata.get("client_disconnected") is True
    assert "stream_ended_upstream_idle" not in metadata, "a client that left was blamed on the upstream"
    assert metadata["error_information"]["error_code"] == "499"
