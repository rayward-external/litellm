import json

import pytest

from litellm.google_genai.adapters.transformation import GoogleGenAIStreamWrapper


def test_google_genai_sse_wrapper_skips_done_sentinel():
    payload = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "OK"}],
                }
            }
        ]
    }

    wrapper = GoogleGenAIStreamWrapper(
        iter(
            [
                payload,
                b"data: [DONE]\n\n",
            ]
        )
    )

    chunks = list(wrapper.google_genai_sse_wrapper())

    assert chunks == [f"data: {json.dumps(payload)}\n\n".encode()]


@pytest.mark.asyncio
async def test_async_google_genai_sse_wrapper_skips_done_sentinel():
    payload = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "OK"}],
                }
            }
        ]
    }

    async def completion_stream():
        yield payload
        yield b"data: [DONE]\n\n"

    wrapper = GoogleGenAIStreamWrapper(completion_stream())
    chunks = [chunk async for chunk in wrapper.async_google_genai_sse_wrapper()]

    assert chunks == [f"data: {json.dumps(payload)}\n\n".encode()]
