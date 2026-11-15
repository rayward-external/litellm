"""A /v1/messages call that succeeds on its retry after a 503 keeps the success's cost on the shared
logging object. Anthropic `/v1/messages` runs on the Rust path only, whose HTTP client respx cannot see,
so the upstream is a loopback server on a real socket."""

import datetime
from collections.abc import Iterator

import httpx
import pytest

import litellm
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.router import Router
from tests.unit.test_router.loopback_upstream import LoopbackUpstream, loopback_upstream

_OK_BODY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5",
    "content": [{"type": "text", "text": "OK"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1000, "output_tokens": 10},
}


def _router(api_base: str) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "claude-group",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5",
                    "api_key": "sk-test",
                    "api_base": api_base,
                },
            }
        ],
        num_retries=1,
        retry_after=0,
    )


def _shared_logging_obj(request: dict):
    logging_obj, _ = litellm.utils.function_setup(
        original_function="anthropic_messages",
        rules_obj=litellm.utils.Rules(),
        start_time=datetime.datetime.now(),
        **request,
    )
    return logging_obj


@pytest.fixture
def upstream() -> Iterator[LoopbackUpstream]:
    with loopback_upstream() as served:
        yield served


@pytest.mark.asyncio
async def test_v1_messages_success_after_a_503_retry_keeps_its_cost_header(monkeypatch, upstream: LoopbackUpstream):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    answers = iter(
        [
            httpx.Response(503, json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}),
            httpx.Response(200, json=_OK_BODY),
        ]
    )
    upstream.mock(side_effect=lambda _request: next(answers))
    request = {
        "model": "claude-group",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
        "litellm_call_id": "call-1118",
    }
    logging_obj = _shared_logging_obj(request)

    response = await _router(upstream.api_base).aanthropic_messages(**request, litellm_logging_obj=logging_obj)

    assert upstream.call_count == 2
    stale = logging_obj.model_call_details.get("response_cost")
    header_cost = ProxyBaseLLMRequestProcessing._response_cost_from_logging_obj(
        response=response, logging_obj=logging_obj
    )
    assert stale != 0, "the failed attempt's zeroed cost is still on the shared logging object"
    assert isinstance(header_cost, float) and header_cost > 0
