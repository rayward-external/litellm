"""Azure AI Foundry xAI models report reasoning tokens additively:
``completion_tokens`` is visible text only, ``reasoning_tokens`` is extra and
``total = prompt + completion + reasoning``. The provider boundary folds reasoning into
``completion_tokens`` (OpenAI convention) so cost math and client-visible usage agree."""

import json
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.cost_calculator import completion_cost
from litellm.llms.azure_ai.chat.transformation import AzureAIChatCompletionResponseIterator, AzureAIStudioConfig
from litellm.types.utils import ModelResponse

MODEL: Final = "grok-4.3"
OUTPUT_RATE: Final = litellm.model_cost["azure_ai/grok-4.3"]["output_cost_per_token"]
INPUT_RATE: Final = litellm.model_cost["azure_ai/grok-4.3"]["input_cost_per_token"]


def _body(usage: dict) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
        "usage": usage,
    }


def _transform(usage: dict) -> ModelResponse:
    raw = httpx.Response(200, json=_body(usage), request=httpx.Request("POST", "https://example.invalid"))
    return AzureAIStudioConfig().transform_response(
        model=MODEL,
        raw_response=raw,
        model_response=ModelResponse(),
        logging_obj=MagicMock(),
        request_data={},
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )


ADDITIVE: Final = {
    "prompt_tokens": 11,
    "completion_tokens": 369,
    "total_tokens": 5902,
    "completion_tokens_details": {"reasoning_tokens": 5522},
}


def test_non_streaming_additive_usage_is_folded():
    usage = _transform(ADDITIVE).usage
    assert usage.completion_tokens == 369 + 5522
    assert usage.prompt_tokens == 11
    assert usage.total_tokens == 5902
    assert usage.completion_tokens_details.reasoning_tokens == 5522


def test_cost_bills_visible_plus_reasoning_at_output_rate():
    response = _transform(ADDITIVE)
    cost = completion_cost(completion_response=response, model=f"azure_ai/{MODEL}")
    assert cost == pytest.approx(11 * INPUT_RATE + (369 + 5522) * OUTPUT_RATE)


def test_openai_convention_usage_is_untouched():
    usage = _transform(
        {
            "prompt_tokens": 11,
            "completion_tokens": 5891,
            "total_tokens": 5902,
            "completion_tokens_details": {"reasoning_tokens": 5522},
        }
    ).usage
    assert usage.completion_tokens == 5891
    assert usage.total_tokens == 5902
    cost = completion_cost(completion_response=_transform_again(usage), model=f"azure_ai/{MODEL}")
    assert cost == pytest.approx(11 * INPUT_RATE + 5891 * OUTPUT_RATE)


def _transform_again(usage) -> ModelResponse:
    response = ModelResponse()
    response.usage = usage
    return response


def test_zero_reasoning_is_untouched():
    usage = _transform(
        {
            "prompt_tokens": 11,
            "completion_tokens": 20,
            "total_tokens": 31,
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
    ).usage
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 31


def test_missing_reasoning_details_is_untouched():
    usage = _transform({"prompt_tokens": 11, "completion_tokens": 20, "total_tokens": 31}).usage
    assert usage.completion_tokens == 20


def _stream_chunk(usage: dict | None) -> dict:
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": MODEL,
        "choices": [],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _handler() -> AzureAIChatCompletionResponseIterator:
    handler = AzureAIStudioConfig().get_model_response_iterator(streaming_response=iter([]), sync_stream=True)
    assert isinstance(handler, AzureAIChatCompletionResponseIterator)
    return handler


def test_streaming_final_chunk_additive_usage_is_folded():
    chunk = _stream_chunk(ADDITIVE)
    parsed = _handler().chunk_parser(chunk)
    assert parsed.usage.completion_tokens == 369 + 5522
    assert parsed.usage.total_tokens == 5902


def test_streaming_openai_convention_and_content_chunks_untouched():
    parsed = _handler().chunk_parser(
        _stream_chunk(
            {
                "prompt_tokens": 11,
                "completion_tokens": 5891,
                "total_tokens": 5902,
                "completion_tokens_details": {"reasoning_tokens": 5522},
            }
        )
    )
    assert parsed.usage.completion_tokens == 5891
    assert _handler().chunk_parser(_stream_chunk(None)) is not None


def test_streaming_end_to_end_through_stream_wrapper():
    lines = [
        "data: " + json.dumps({**_stream_chunk(None), "choices": [{"index": 0, "delta": {"content": "hi"}}]}),
        "data: " + json.dumps({**_stream_chunk(None), "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        "data: " + json.dumps(_stream_chunk(ADDITIVE)),
        "data: [DONE]",
    ]
    handler = AzureAIStudioConfig().get_model_response_iterator(streaming_response=iter(lines), sync_stream=True)
    chunks = list(handler)
    usages = [c.usage for c in chunks if getattr(c, "usage", None) is not None]
    assert [u.completion_tokens for u in usages] == [369 + 5522]
