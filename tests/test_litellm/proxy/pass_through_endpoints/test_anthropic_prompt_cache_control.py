import asyncio
import copy

from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    ANTHROPIC_PROMPT_CACHE_TTL_ENV,
    ANTHROPIC_PROMPT_CACHE_TTL_HEADER,
    _apply_anthropic_prompt_cache_control_to_request,
    apply_anthropic_prompt_cache_control,
    filter_anthropic_prompt_cache_control_headers,
)
from litellm.types.utils import LlmProviders


class DummyState:
    pass


class DummyRequest:
    def __init__(self, body, headers=None, method="POST"):
        self.method = method
        self.headers = headers or {}
        self.scope = {}
        self.state = DummyState()
        self._body = body

    async def body(self):
        import orjson

        return orjson.dumps(self._body)


def test_apply_anthropic_prompt_cache_control_injects_top_level_one_hour(monkeypatch):
    monkeypatch.setenv(ANTHROPIC_PROMPT_CACHE_TTL_ENV, "1h")
    body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hello"}],
    }

    changed = apply_anthropic_prompt_cache_control(body, headers={})

    assert changed is True
    assert body["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_apply_anthropic_prompt_cache_control_preserves_client_control(monkeypatch):
    monkeypatch.setenv(ANTHROPIC_PROMPT_CACHE_TTL_ENV, "1h")
    body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "cache_control": {"type": "ephemeral", "ttl": "5m"},
        "messages": [{"role": "user", "content": "hello"}],
    }
    original = copy.deepcopy(body)

    changed = apply_anthropic_prompt_cache_control(body, headers={})

    assert changed is False
    assert body == original


def test_apply_anthropic_prompt_cache_control_preserves_nested_client_control(
    monkeypatch,
):
    monkeypatch.setenv(ANTHROPIC_PROMPT_CACHE_TTL_ENV, "1h")
    body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hello",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    }
    original = copy.deepcopy(body)

    changed = apply_anthropic_prompt_cache_control(body, headers={})

    assert changed is False
    assert body == original


def test_apply_anthropic_prompt_cache_control_header_can_disable_env_default(
	monkeypatch,
):
    monkeypatch.setenv(ANTHROPIC_PROMPT_CACHE_TTL_ENV, "1h")
    body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hello"}],
    }

    changed = apply_anthropic_prompt_cache_control(
        body,
        headers={ANTHROPIC_PROMPT_CACHE_TTL_HEADER: "off"},
    )

    assert changed is False
    assert "cache_control" not in body


def test_filter_anthropic_prompt_cache_control_headers_strips_control_headers():
    headers = {
        "authorization": "Bearer sk-test",
        ANTHROPIC_PROMPT_CACHE_TTL_HEADER: "1h",
        "x-anthropic-prompt-cache-workload": "eval",
    }

    filtered = filter_anthropic_prompt_cache_control_headers(headers)

    assert filtered == {"authorization": "Bearer sk-test"}


def test_anthropic_passthrough_mutates_messages_request(monkeypatch):
    monkeypatch.setenv(ANTHROPIC_PROMPT_CACHE_TTL_ENV, "1h")
    request = DummyRequest(
        {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
        },
        headers={"content-type": "application/json"},
    )

    asyncio.run(
        _apply_anthropic_prompt_cache_control_to_request(
            request,
            LlmProviders.ANTHROPIC.value,
            "/v1/messages",
        )
    )

    assert request.state.litellm_pass_through_custom_body["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }


def test_non_anthropic_passthrough_is_unchanged(monkeypatch):
    monkeypatch.setenv(ANTHROPIC_PROMPT_CACHE_TTL_ENV, "1h")
    request = DummyRequest(
        {
            "model": "some-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
        headers={"content-type": "application/json"},
    )

    asyncio.run(
        _apply_anthropic_prompt_cache_control_to_request(
            request,
            LlmProviders.MISTRAL.value,
            "/v1/messages",
        )
    )

    assert not hasattr(request.state, "litellm_pass_through_custom_body")
