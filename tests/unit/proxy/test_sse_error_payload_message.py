import asyncio
import json

import httpx
import pytest
from fastapi import status

import litellm
from litellm.proxy.common_request_processing import (
    _anthropic_late_failure_frames,
    create_response,
    open_sse_before_first_byte,
    proxy_exception_from_route_error,
    sse_error_payload,
)

_CREDIT_TEXT = "Your credit balance is too low to access the Anthropic API."


def _provider_400() -> litellm.BadRequestError:
    return litellm.BadRequestError(
        message=f"AnthropicException - {_CREDIT_TEXT}",
        model="claude-x",
        llm_provider="anthropic",
        response=httpx.Response(400, request=httpx.Request("POST", "https://example.invalid")),
    )


def test_sse_error_payload_carries_provider_message():
    exc = _provider_400()
    error_status, error_obj = sse_error_payload(exc)
    assert error_status == 400
    assert _CREDIT_TEXT in str(error_obj["message"])


def test_sse_error_payload_matches_non_streaming_body():
    exc = _provider_400()
    _, error_obj = sse_error_payload(exc)
    assert dict(error_obj) == proxy_exception_from_route_error(exc).to_dict()


def test_sse_error_payload_keeps_placeholder_for_plain_exception():
    _, error_obj = sse_error_payload(ValueError("internal detail"))
    assert error_obj["message"] == "Error processing stream start"


@pytest.mark.asyncio
async def test_openai_late_failure_frame_carries_provider_message():
    async def slow_failure():
        await asyncio.sleep(0.2)
        raise _provider_400()

    response = await open_sse_before_first_byte(slow_failure(), ping_interval_seconds=0.05)
    collected = [chunk async for chunk in response.body_iterator]
    frame = json.loads(collected[-2].decode().removeprefix("data: ").strip())
    assert _CREDIT_TEXT in frame["error"]["message"]


@pytest.mark.asyncio
async def test_first_chunk_failure_frame_carries_provider_message():
    async def gen():
        raise _provider_400()
        yield ""  # pragma: no cover

    response = await create_response(gen(), "text/event-stream", {})
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    collected = [chunk async for chunk in response.body_iterator]
    frame = json.loads(collected[0].removeprefix("data: ").strip())
    assert _CREDIT_TEXT in frame["error"]["message"]


def test_anthropic_late_failure_frame_already_carries_provider_message():
    frames = _anthropic_late_failure_frames(_provider_400())
    assert _CREDIT_TEXT in "".join(frames)
