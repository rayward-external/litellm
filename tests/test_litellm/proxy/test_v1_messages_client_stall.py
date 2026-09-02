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


async def _stream_v1_messages_to(client: _StallingClient, upstream: _FakeAnthropicStream) -> None:
    """Run the /v1/messages streaming path and serve its response to `client`."""
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
            proxy_logging_obj=_proxy_logging_obj(),
            general_settings={},
            proxy_config=MagicMock(),
            llm_router=None,
        )

    await response(_asgi_scope(), client.receive, client.send)


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
