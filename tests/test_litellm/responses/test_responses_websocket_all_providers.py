"""
Unit tests to verify that all providers support Responses API WebSocket mode.

Tests that:
1. All providers with ResponsesAPIConfig support websocket mode
2. Providers with native websocket support use direct connection
3. Providers without native websocket support use ManagedResponsesWebSocketHandler
"""

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
from litellm.llms.databricks.responses.transformation import (
    DatabricksResponsesAPIConfig,
)
from litellm.llms.github_copilot.responses.transformation import (
    GithubCopilotResponsesAPIConfig,
)
from litellm.llms.hosted_vllm.responses.transformation import (
    HostedVLLMResponsesAPIConfig,
)
from litellm.llms.litellm_proxy.responses.transformation import (
    LiteLLMProxyResponsesAPIConfig,
)
from litellm.llms.manus.responses.transformation import ManusResponsesAPIConfig
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.llms.openrouter.responses.transformation import (
    OpenRouterResponsesAPIConfig,
)
from litellm.llms.perplexity.responses.transformation import PerplexityResponsesConfig
from litellm.llms.volcengine.responses.transformation import (
    VolcEngineResponsesAPIConfig,
)
from litellm.llms.xai.responses.transformation import XAIResponsesAPIConfig


class TestResponsesAPIWebSocketSupport:
    """Test that all providers have websocket support configured correctly"""

    def test_openai_supports_native_websocket(self):
        """OpenAI should support native websocket"""
        config = OpenAIResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is True
        ), "OpenAI should support native websocket"

    def test_azure_supports_native_websocket(self):
        """Azure should support native websocket"""
        config = AzureOpenAIResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is True
        ), "Azure should support native websocket"

    def test_azure_websocket_url_uses_v1_path(self):
        """Azure WebSocket URL must use /openai/v1/responses (no api-version)"""
        config = AzureOpenAIResponsesAPIConfig()
        url = config.get_websocket_url(
            api_base="https://myresource.cognitiveservices.azure.com",
            litellm_params={"api_version": "2025-04-01-preview"},
        )
        assert url == "wss://myresource.cognitiveservices.azure.com/openai/v1/responses"
        assert "api-version" not in url

    def test_azure_websocket_url_strips_existing_path(self):
        """api_base that already contains /openai/responses must be cleaned"""
        config = AzureOpenAIResponsesAPIConfig()
        url = config.get_websocket_url(
            api_base="https://myresource.cognitiveservices.azure.com/openai/responses",
            litellm_params={},
        )
        assert url == "wss://myresource.cognitiveservices.azure.com/openai/v1/responses"

    def test_azure_websocket_url_strips_query_params(self):
        config = AzureOpenAIResponsesAPIConfig()
        url = config.get_websocket_url(
            api_base="https://myresource.cognitiveservices.azure.com/openai/responses?api-version=2024-05-01-preview",
            litellm_params={},
        )
        assert url == "wss://myresource.cognitiveservices.azure.com/openai/v1/responses"

    def test_azure_websocket_url_requires_api_base(self):
        config = AzureOpenAIResponsesAPIConfig()
        with pytest.raises(ValueError, match='api_base is required for Azure WebSocket'):
            config.get_websocket_url(api_base=None, litellm_params={})

    def test_azure_model_not_in_websocket_url(self):
        """Azure sends the model in the body, so it must not be appended to the URL"""
        assert AzureOpenAIResponsesAPIConfig().model_in_websocket_url() is False

    def test_openai_default_websocket_url_converts_scheme(self):
        """The base get_websocket_url default converts the HTTP endpoint to wss://"""
        config = OpenAIResponsesAPIConfig()
        url = config.get_websocket_url(
            api_base="https://api.openai.com/v1", litellm_params={}
        )
        assert url == "wss://api.openai.com/v1/responses"

    def test_openai_model_in_websocket_url_default(self):
        assert OpenAIResponsesAPIConfig().model_in_websocket_url() is True

    @pytest.mark.asyncio
    async def test_openai_websocket_forwards_explicit_api_key(self, monkeypatch):
        from unittest.mock import AsyncMock

        import litellm
        from litellm.responses import main as responses_main

        websocket_handler = AsyncMock()
        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.setattr(litellm, "openai_key", None)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(
            responses_main.base_llm_http_handler,
            "async_responses_websocket",
            websocket_handler,
        )

        await responses_main._aresponses_websocket.__wrapped__(
            model="gpt-4o",
            websocket=MagicMock(),
            api_key="explicit-api-key",
            litellm_logging_obj=MagicMock(),
        )

        assert websocket_handler.await_args.kwargs["api_key"] == "explicit-api-key"

    @pytest.mark.asyncio
    async def test_bedrock_websocket_does_not_leak_openai_key_as_bearer_token(
        self, monkeypatch
    ):
        """
        Regression test for rayward-internal/llm-gateway-infra#645.

        A Bedrock deployment carries no `api_key` of its own (boto3 reads AWS
        credentials from the environment/IAM role), so `_aresponses_websocket`
        must leave `resolved_api_key` at `None` for it, exactly like the HTTP
        path leaves `litellm_params.api_key` at `None` for bedrock.
        `bedrock/base_aws_llm.py`'s `get_request_headers` treats ANY non-None
        `api_key` as an AWS bearer token and skips SigV4 entirely, so a
        left-over generic fallback (previously `litellm.api_key or
        litellm.openai_key or get_secret_str("OPENAI_API_KEY")`) here fails
        Bedrock with "Invalid API Key format: Must start with pre-defined
        prefix".
        """
        from unittest.mock import AsyncMock

        import litellm
        from litellm.responses import main as responses_main

        websocket_handler = AsyncMock()
        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.setattr(litellm, "openai_key", None)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-should-never-reach-bedrock")
        monkeypatch.setattr(
            responses_main.base_llm_http_handler,
            "async_responses_websocket",
            websocket_handler,
        )

        await responses_main._aresponses_websocket.__wrapped__(
            model="bedrock/anthropic.claude-haiku-4-5-v1:0",
            websocket=MagicMock(),
            litellm_logging_obj=MagicMock(),
        )

        assert websocket_handler.await_args.kwargs["api_key"] is None

    @pytest.mark.asyncio
    async def test_azure_websocket_honors_router_resolved_custom_llm_provider(
        self, monkeypatch
    ):
        """
        Regression test for rayward-internal/llm-gateway-infra#650's Azure
        WS-bridge 404.

        Azure OpenAI deployments deliberately configure a BARE
        litellm_params.model (no "azure/" prefix) and rely entirely on the
        separate custom_llm_provider field for routing (see
        gcp/modules/gateway_cloud_run/litellm.config.yaml.tmpl in
        llm-gateway-infra: "Keep model_name and litellm_params.model bare").
        The router already resolves and forwards custom_llm_provider="azure"
        for such a deployment, but `_aresponses_websocket` used to call
        `litellm.get_llm_provider(model=model, ...)` without it, forcing a
        fresh re-derivation from the bare model string alone. "gpt-5.5"
        matches litellm's own OpenAI model-name heuristic, so the connection
        silently got provider="openai" and OpenAIResponsesAPIConfig instead
        of Azure's -- an OpenAI-shaped request against an Azure api_base,
        which Azure answers with `OpenAIException - 404 Resource not found`
        (the exact string measured live 2026-08-27). Passing the
        already-resolved custom_llm_provider through must keep it "azure".
        """
        from unittest.mock import AsyncMock

        from litellm.responses import main as responses_main

        websocket_handler = AsyncMock()
        monkeypatch.setattr(
            responses_main.base_llm_http_handler,
            "async_responses_websocket",
            websocket_handler,
        )

        await responses_main._aresponses_websocket.__wrapped__(
            model="gpt-5.5",
            websocket=MagicMock(),
            api_base="https://myresource.openai.azure.com",
            api_key="sk-azure-key",
            api_version="2025-04-01-preview",
            custom_llm_provider="azure",
            litellm_logging_obj=MagicMock(),
        )

        assert websocket_handler.await_args.kwargs["custom_llm_provider"] == "azure"
        assert isinstance(
            websocket_handler.await_args.kwargs["responses_api_provider_config"],
            AzureOpenAIResponsesAPIConfig,
        )

    @pytest.mark.asyncio
    async def test_azure_ai_foundry_websocket_does_not_crash_on_bare_model_name(
        self, monkeypatch
    ):
        """
        Regression test for rayward-internal/llm-gateway-infra#650's
        kimi-k2.6 "no frames at all" symptom.

        Azure AI Foundry deployments use the same bare-model-name +
        custom_llm_provider="azure_ai" config shape as Azure OpenAI (see
        gcp/modules/gateway_cloud_run/litellm.config.yaml.tmpl in
        llm-gateway-infra). Unlike "gpt-5.5", a bare model like "kimi-k2.6"
        matches NONE of litellm.get_llm_provider's model-name heuristics, so
        re-deriving the provider from scratch (ignoring the router's
        already-resolved custom_llm_provider) raised an uncaught
        `litellm.BadRequestError: LLM Provider NOT provided` before a single
        WebSocket frame was ever sent -- measured live as "no frames at all".
        Passing the already-resolved custom_llm_provider through must let
        this connection set up without raising.
        """
        from unittest.mock import AsyncMock

        from litellm.responses import main as responses_main

        websocket_handler = AsyncMock()
        monkeypatch.setattr(
            responses_main.base_llm_http_handler,
            "async_responses_websocket",
            websocket_handler,
        )

        await responses_main._aresponses_websocket.__wrapped__(
            model="kimi-k2.6",
            websocket=MagicMock(),
            api_base="https://rayward-foundry-east-aif.cognitiveservices.azure.com/models",
            api_key="sk-foundry-key",
            api_version="2024-05-01-preview",
            custom_llm_provider="azure_ai",
            litellm_logging_obj=MagicMock(),
        )

        assert websocket_handler.await_args.kwargs["custom_llm_provider"] == "azure_ai"
        assert websocket_handler.await_args.kwargs["model"] == "kimi-k2.6"

    def test_xai_uses_managed_websocket(self):
        """XAI should use managed websocket handler"""
        config = XAIResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "XAI should use managed websocket handler"

    def test_github_copilot_uses_managed_websocket(self):
        """GitHub Copilot should use managed websocket handler"""
        config = GithubCopilotResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "GitHub Copilot should use managed websocket handler"

    def test_chatgpt_uses_managed_websocket(self):
        """ChatGPT should use managed websocket handler"""
        config = ChatGPTResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "ChatGPT should use managed websocket handler"

    def test_litellm_proxy_uses_managed_websocket(self):
        """LiteLLM Proxy should use managed websocket handler"""
        config = LiteLLMProxyResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "LiteLLM Proxy should use managed websocket handler"

    def test_volcengine_uses_managed_websocket(self):
        """VolcEngine should use managed websocket handler"""
        config = VolcEngineResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "VolcEngine should use managed websocket handler"

    def test_manus_uses_managed_websocket(self):
        """Manus should use managed websocket handler"""
        config = ManusResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "Manus should use managed websocket handler"

    def test_perplexity_uses_managed_websocket(self):
        """Perplexity should use managed websocket handler"""
        config = PerplexityResponsesConfig()
        assert (
            config.supports_native_websocket() is False
        ), "Perplexity should use managed websocket handler"

    def test_databricks_uses_managed_websocket(self):
        """Databricks should use managed websocket handler"""
        config = DatabricksResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "Databricks should use managed websocket handler"

    def test_openrouter_uses_managed_websocket(self):
        """OpenRouter should use managed websocket handler"""
        config = OpenRouterResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "OpenRouter should use managed websocket handler"

    def test_hosted_vllm_uses_managed_websocket(self):
        """Hosted vLLM should use managed websocket handler"""
        config = HostedVLLMResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "Hosted vLLM should use managed websocket handler"


class TestManagedWebSocketHandlerIntegration:
    """Test that ManagedResponsesWebSocketHandler is properly integrated"""

    @pytest.mark.asyncio
    async def test_managed_handler_instantiation(self):
        """Test that ManagedResponsesWebSocketHandler can be instantiated"""
        from unittest.mock import MagicMock

        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        mock_websocket = MagicMock()
        mock_logging_obj = Logging(
            model="test-model",
            messages=[],
            stream=True,
            call_type="aresponses",
            start_time=0,
            litellm_call_id="test-id",
            function_id="test-func",
        )

        handler = ManagedResponsesWebSocketHandler(
            websocket=mock_websocket,
            model="test-model",
            logging_obj=mock_logging_obj,
            user_api_key_dict=None,
            litellm_metadata={},
            api_key="test-key",
            api_base="https://api.example.com",
            timeout=30.0,
            custom_llm_provider="test_provider",
        )

        assert handler.model == "test-model"
        assert handler.api_key == "test-key"
        assert handler.api_base == "https://api.example.com"
        assert handler.timeout == 30.0
        assert handler.custom_llm_provider == "test_provider"

    @pytest.mark.asyncio
    async def test_frame_alias_resolves_to_connection_model(self, monkeypatch):
        """
        A response.create frame that repeats the public model alias must reach
        litellm.aresponses with the router-resolved deployment model, not the
        raw alias (which fails in get_llm_provider). Regression for codex
        WebSocket sessions against managed providers like bedrock_mantle.
        """
        import json
        from unittest.mock import AsyncMock, MagicMock

        import litellm
        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        captured: dict = {}

        async def fake_aresponses(*args, **kwargs):
            captured["model"] = kwargs.get("model")

            async def _empty():
                return
                yield

            return _empty()

        monkeypatch.setattr(litellm, "aresponses", fake_aresponses)

        mock_websocket = MagicMock()
        mock_websocket.send_text = AsyncMock()

        handler = ManagedResponsesWebSocketHandler(
            websocket=mock_websocket,
            model="bedrock_mantle/openai.gpt-5.5",
            logging_obj=Logging(
                model="bedrock_mantle/openai.gpt-5.5",
                messages=[],
                stream=True,
                call_type="aresponses",
                start_time=0,
                litellm_call_id="test-id",
                function_id="test-func",
            ),
            litellm_metadata={"model_group": "gpt-5.5-mantle"},
        )

        frame = json.dumps(
            {
                "type": "response.create",
                "model": "gpt-5.5-mantle",
                "input": [],
            }
        )
        await handler._process_response_create(frame)

        assert captured["model"] == "bedrock_mantle/openai.gpt-5.5"

    @pytest.mark.asyncio
    async def test_warmup_frame_skips_provider_and_sends_synthetic_ack(
        self, monkeypatch
    ):
        """
        A generate=false warmup frame (codex prewarm) carries empty input that
        managed HTTP providers reject. It must not call the provider, and should
        emit synthetic response.created/completed events so Codex can proceed.
        """
        import json
        from unittest.mock import AsyncMock, MagicMock

        import litellm
        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        called = False

        async def fail_aresponses(*args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("provider must not be called for a warmup frame")

        monkeypatch.setattr(litellm, "aresponses", fail_aresponses)

        mock_websocket = MagicMock()
        mock_websocket.send_text = AsyncMock()

        handler = ManagedResponsesWebSocketHandler(
            websocket=mock_websocket,
            model="bedrock_mantle/openai.gpt-5.5",
            logging_obj=Logging(
                model="bedrock_mantle/openai.gpt-5.5",
                messages=[],
                stream=True,
                call_type="aresponses",
                start_time=0,
                litellm_call_id="test-id",
                function_id="test-func",
            ),
            litellm_metadata={"model_group": "gpt-5.5-mantle"},
        )

        frame = json.dumps(
            {
                "type": "response.create",
                "model": "gpt-5.5-mantle",
                "generate": False,
                "input": [],
            }
        )
        await handler._process_response_create(frame)

        assert called is False
        assert mock_websocket.send_text.call_count == 2
        events = [
            json.loads(call.args[0]) for call in mock_websocket.send_text.call_args_list
        ]
        assert events[0]["type"] == "response.created"
        assert events[0]["response"]["status"] == "in_progress"
        assert events[1]["type"] == "response.completed"
        assert events[1]["response"]["status"] == "completed"
        assert events[1]["response"]["output"] == []
        assert events[1]["response"]["model"] == "gpt-5.5-mantle"

    @pytest.mark.asyncio
    async def test_warmup_previous_response_id_not_forwarded_to_provider(
        self, monkeypatch
    ):
        import json
        from unittest.mock import AsyncMock, MagicMock

        import litellm
        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        captured: dict = {}

        async def fake_aresponses(*args, **kwargs):
            captured.update(kwargs)

            async def _empty():
                return
                yield

            return _empty()

        monkeypatch.setattr(litellm, "aresponses", fake_aresponses)

        mock_websocket = MagicMock()
        mock_websocket.send_text = AsyncMock()

        handler = ManagedResponsesWebSocketHandler(
            websocket=mock_websocket,
            model="bedrock_mantle/openai.gpt-5.5",
            logging_obj=Logging(
                model="bedrock_mantle/openai.gpt-5.5",
                messages=[],
                stream=True,
                call_type="aresponses",
                start_time=0,
                litellm_call_id="test-id",
                function_id="test-func",
            ),
            litellm_metadata={"model_group": "gpt-5.5-mantle"},
        )

        await handler._process_response_create(
            json.dumps(
                {
                    "type": "response.create",
                    "model": "gpt-5.5-mantle",
                    "generate": False,
                    "input": [],
                }
            )
        )
        warmup_id = json.loads(mock_websocket.send_text.call_args_list[1].args[0])[
            "response"
        ]["id"]

        await handler._process_response_create(
            json.dumps(
                {
                    "type": "response.create",
                    "model": "gpt-5.5-mantle",
                    "previous_response_id": warmup_id,
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Hi"}],
                        }
                    ],
                }
            )
        )

        assert "previous_response_id" not in captured


class TestChunkTransformation:
    """Test chunk serialization and transformation for WebSocket streaming"""

    def test_serialize_chunk_with_dict(self):
        """Test serialization of dict chunks"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.created",
            "response": {"id": "resp_456", "status": "in_progress"},
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.created" in serialized
        assert "resp_456" in serialized

    def test_serialize_chunk_handles_invalid_json(self):
        """Test that chunks with circular references are handled"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        # Create object with circular reference
        obj = {"a": 1}
        obj["self"] = obj  # type: ignore

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(obj)
        assert serialized is None

    def test_extract_output_messages_with_text_content(self):
        """Test extraction of output messages with text content"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello world"}],
                    }
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 1
        assert messages[0]["type"] == "message"
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"][0]["text"] == "Hello world"

    def test_extract_output_messages_with_multiple_content_parts(self):
        """Test extraction with multiple content parts"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Part 1. "},
                            {"type": "output_text", "text": "Part 2."},
                        ],
                    }
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 1
        assert messages[0]["content"][0]["text"] == "Part 1. Part 2."

    def test_extract_output_messages_with_function_calls(self):
        """Test that function calls are preserved"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "function_call",
                        "id": "call_123",
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    }
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 1
        assert messages[0]["type"] == "function_call"
        assert messages[0]["id"] == "call_123"
        assert messages[0]["name"] == "get_weather"

    def test_extract_output_messages_filters_empty_text(self):
        """Test that messages with empty text are filtered out"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": ""}],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Valid text"}],
                    },
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 1
        assert messages[0]["content"][0]["text"] == "Valid text"

    def test_extract_output_messages_handles_non_dict_items(self):
        """Test that non-dict items are skipped"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    "invalid_string",
                    None,
                    123,
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Valid"}],
                    },
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 1
        assert messages[0]["content"][0]["text"] == "Valid"

    def test_input_to_messages_with_string(self):
        """Test conversion of string input to messages"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        messages = ManagedResponsesWebSocketHandler._input_to_messages("Hello world")
        assert len(messages) == 1
        assert messages[0]["type"] == "message"
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0]["type"] == "input_text"
        assert messages[0]["content"][0]["text"] == "Hello world"

    def test_input_to_messages_with_list(self):
        """Test conversion of list input to messages"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        input_list = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Question"}],
            }
        ]

        messages = ManagedResponsesWebSocketHandler._input_to_messages(input_list)
        assert len(messages) == 1
        assert messages[0]["type"] == "message"
        assert messages[0]["content"][0]["text"] == "Question"

    def test_input_to_messages_filters_non_dict_items(self):
        """Test that non-dict items in list input are filtered"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        input_list = [
            "invalid_string",
            None,
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Valid"}],
            },
        ]

        messages = ManagedResponsesWebSocketHandler._input_to_messages(input_list)
        assert len(messages) == 1
        assert messages[0]["content"][0]["text"] == "Valid"

    def test_input_to_messages_handles_empty_input(self):
        """Test that empty input returns empty list"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        assert ManagedResponsesWebSocketHandler._input_to_messages(None) == []
        assert ManagedResponsesWebSocketHandler._input_to_messages([]) == []
        assert ManagedResponsesWebSocketHandler._input_to_messages({}) == []


class TestUpdateProxyRequest:
    """Regression tests for ManagedResponsesWebSocketHandler._update_proxy_request.

    The managed WebSocket path calls ``litellm.aresponses(model=..., **call_kwargs)``.
    ``litellm_params`` is not a Responses API request field, so passing it as a
    top-level kwarg leaks it into the provider request body and providers that
    forbid extra inputs (e.g. Anthropic) reject the call with
    ``litellm_params: Extra inputs are not permitted``. The request-tracking data
    must ride along as ``proxy_server_request`` instead, which litellm consumes
    internally and never forwards to the provider.
    """

    def test_does_not_inject_litellm_params_kwarg(self):
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        call_kwargs = {
            "input": "hello",
            "store": True,
            "litellm_metadata": {
                "proxy_server_request": {"headers": {}, "body": {}},
            },
        }

        ManagedResponsesWebSocketHandler._update_proxy_request(
            call_kwargs, "anthropic/claude-sonnet-4-5"
        )

        assert "litellm_params" not in call_kwargs
        assert call_kwargs["proxy_server_request"]["body"]["model"] == (
            "anthropic/claude-sonnet-4-5"
        )
        assert call_kwargs["proxy_server_request"]["body"]["input"] == "hello"

    def test_proxy_server_request_matches_metadata(self):
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        call_kwargs = {
            "input": "hi",
            "litellm_metadata": {"proxy_server_request": {"body": {}}},
        }

        ManagedResponsesWebSocketHandler._update_proxy_request(call_kwargs, "gpt-4o")

        assert (
            call_kwargs["proxy_server_request"]
            == call_kwargs["litellm_metadata"]["proxy_server_request"]
        )


class TestWebSocketEventTypes:
    """Test that all WebSocket event types are properly handled with dict-based chunks"""

    def test_serialize_response_created_event_dict(self):
        """Test serialization of response.created event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.created",
            "response_id": "resp_123",
            "response": {
                "id": "resp_123",
                "object": "response",
                "status": "in_progress",
                "created_at": 1234567890,
            },
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.created" in serialized
        assert "resp_123" in serialized

    def test_serialize_response_in_progress_event_dict(self):
        """Test serialization of response.in_progress event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {"type": "response.in_progress", "response_id": "resp_123"}

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.in_progress" in serialized

    def test_serialize_output_item_added_event_dict(self):
        """Test serialization of response.output_item.added event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.output_item.added",
            "response_id": "resp_123",
            "item_id": "msg_456",
            "output_index": 0,
            "item": {"type": "message", "role": "assistant"},
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.output_item.added" in serialized
        assert "msg_456" in serialized

    def test_serialize_output_text_delta_event_dict(self):
        """Test serialization of response.output_text.delta event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.output_text.delta",
            "response_id": "resp_123",
            "item_id": "msg_456",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello",
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.output_text.delta" in serialized
        assert "Hello" in serialized

    def test_serialize_output_text_done_event_dict(self):
        """Test serialization of response.output_text.done event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.output_text.done",
            "response_id": "resp_123",
            "item_id": "msg_456",
            "output_index": 0,
            "content_index": 0,
            "text": "Hello world",
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.output_text.done" in serialized
        assert "Hello world" in serialized

    def test_serialize_content_part_done_event_dict(self):
        """Test serialization of response.content_part.done event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.content_part.done",
            "response_id": "resp_123",
            "item_id": "msg_456",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "Complete text"},
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.content_part.done" in serialized

    def test_serialize_output_item_done_event_dict(self):
        """Test serialization of response.output_item.done event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.output_item.done",
            "response_id": "resp_123",
            "item_id": "msg_456",
            "output_index": 0,
            "item": {"type": "message", "role": "assistant", "status": "completed"},
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.output_item.done" in serialized
        assert "msg_456" in serialized

    def test_serialize_response_completed_event_dict(self):
        """Test serialization of response.completed event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.completed",
            "response_id": "resp_123",
            "response": {
                "id": "resp_123",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Done"}],
                    }
                ],
            },
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.completed" in serialized
        assert "resp_123" in serialized

    def test_serialize_response_failed_event_dict(self):
        """Test serialization of response.failed event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.failed",
            "response_id": "resp_123",
            "response": {
                "id": "resp_123",
                "status": "failed",
                "status_details": {"error": {"message": "Rate limit exceeded"}},
            },
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.failed" in serialized
        assert "Rate limit exceeded" in serialized

    def test_serialize_response_incomplete_event_dict(self):
        """Test serialization of response.incomplete event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.incomplete",
            "response_id": "resp_123",
            "response": {
                "id": "resp_123",
                "status": "incomplete",
                "status_details": {"reason": "max_output_tokens"},
            },
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.incomplete" in serialized
        assert "max_output_tokens" in serialized


class TestMultiTurnSessionHistory:
    """Test multi-turn conversation handling via session history"""

    def test_extract_output_messages_preserves_multiple_messages(self):
        """Test that multiple output messages are all preserved"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "First message"}],
                    },
                    {
                        "type": "function_call",
                        "id": "call_123",
                        "name": "get_weather",
                        "arguments": "{}",
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Second message"}],
                    },
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 3
        assert messages[0]["content"][0]["text"] == "First message"
        assert messages[1]["type"] == "function_call"
        assert messages[2]["content"][0]["text"] == "Second message"

    def test_input_to_messages_with_mixed_content_types(self):
        """Test input conversion with mixed content types"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        input_list = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Question"},
                    {"type": "input_image", "image_url": "https://example.com/img.png"},
                ],
            }
        ]

        messages = ManagedResponsesWebSocketHandler._input_to_messages(input_list)
        assert len(messages) == 1
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][0]["type"] == "input_text"
        assert messages[0]["content"][1]["type"] == "input_image"

    def test_extract_output_messages_with_mixed_text_types(self):
        """Test that both 'output_text' and 'text' types are extracted"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Part 1"},
                            {"type": "text", "text": "Part 2"},
                        ],
                    }
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 1
        assert messages[0]["content"][0]["text"] == "Part 1Part 2"

    def test_extract_response_id_from_completed_event(self):
        """Test extraction of response ID from completed event"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {"id": "resp_abc123", "status": "completed"},
        }

        response_id = ManagedResponsesWebSocketHandler._extract_response_id(
            completed_event
        )
        assert response_id == "resp_abc123"

    def test_extract_response_id_handles_missing_response(self):
        """Test that missing response dict returns None"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {"type": "response.completed"}

        response_id = ManagedResponsesWebSocketHandler._extract_response_id(
            completed_event
        )
        assert response_id is None


class TestWebSocketErrorHandling:
    """Test error handling in WebSocket mode"""

    @pytest.mark.asyncio
    async def test_managed_handler_handles_invalid_json(self):
        """Test that invalid JSON in response.create is handled gracefully"""
        from unittest.mock import AsyncMock, MagicMock

        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        mock_websocket = MagicMock()
        mock_websocket.send_text = AsyncMock()
        mock_websocket.recv = AsyncMock(return_value="invalid json {{{")

        mock_logging_obj = Logging(
            model="test-model",
            messages=[],
            stream=True,
            call_type="aresponses",
            start_time=0,
            litellm_call_id="test-id",
            function_id="test-func",
        )

        handler = ManagedResponsesWebSocketHandler(
            websocket=mock_websocket,
            model="test-model",
            logging_obj=mock_logging_obj,
        )

        # Process invalid JSON
        await handler._process_response_create("invalid json {{{")

        # Should have sent an error event
        mock_websocket.send_text.assert_called_once()
        error_event = mock_websocket.send_text.call_args[0][0]
        assert "error" in error_event
        assert "Invalid JSON" in error_event


class TestWebSocketProjectQuotaEnforcement:
    """VERIA regression: the connection-level pre-call hook only runs once,
    but a WebSocket connection accepts many response.create frames. Every
    frame must be checked against any registered project ITPM/OTPM quota
    callback, not just the first one."""

    @pytest.mark.asyncio
    async def test_managed_handler_blocks_frame_rejected_by_quota_callback(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        import litellm
        from litellm.exceptions import RateLimitError
        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        aresponses_called = False

        async def fake_aresponses(*args, **kwargs):
            nonlocal aresponses_called
            aresponses_called = True

        monkeypatch.setattr(litellm, "aresponses", fake_aresponses)

        quota_callback = MagicMock()
        quota_callback.enforce_project_io_token_quota_for_frame = AsyncMock(
            side_effect=RateLimitError(message="project OTPM exceeded", llm_provider="", model="")
        )

        mock_websocket = MagicMock()
        mock_websocket.send_text = AsyncMock()
        mock_logging_obj = Logging(
            model="test-model",
            messages=[],
            stream=True,
            call_type="aresponses",
            start_time=0,
            litellm_call_id="test-id",
            function_id="test-func",
        )
        handler = ManagedResponsesWebSocketHandler(
            websocket=mock_websocket,
            model="test-model",
            logging_obj=mock_logging_obj,
            quota_callbacks=[quota_callback],
        )

        await handler._process_response_create(json.dumps({"type": "response.create", "input": "hi"}))

        quota_callback.enforce_project_io_token_quota_for_frame.assert_awaited_once()
        assert aresponses_called is False
        mock_websocket.send_text.assert_called_once()
        error_event = mock_websocket.send_text.call_args[0][0]
        assert "rate_limit_exceeded" in error_event

    @pytest.mark.asyncio
    async def test_managed_handler_forwards_frame_allowed_by_quota_callback(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        import litellm
        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        aresponses_called = False

        async def fake_aresponses(*args, **kwargs):
            nonlocal aresponses_called
            aresponses_called = True

            async def _empty():
                return
                yield

            return _empty()

        monkeypatch.setattr(litellm, "aresponses", fake_aresponses)

        quota_callback = MagicMock()
        quota_callback.enforce_project_io_token_quota_for_frame = AsyncMock(return_value=None)

        mock_websocket = MagicMock()
        mock_websocket.send_text = AsyncMock()
        mock_logging_obj = Logging(
            model="test-model",
            messages=[],
            stream=True,
            call_type="aresponses",
            start_time=0,
            litellm_call_id="test-id",
            function_id="test-func",
        )
        handler = ManagedResponsesWebSocketHandler(
            websocket=mock_websocket,
            model="test-model",
            logging_obj=mock_logging_obj,
            quota_callbacks=[quota_callback],
        )

        await handler._process_response_create(json.dumps({"type": "response.create", "input": "hi"}))

        quota_callback.enforce_project_io_token_quota_for_frame.assert_awaited_once()
        assert aresponses_called is True

    @pytest.mark.asyncio
    async def test_native_handler_blocks_frame_rejected_by_quota_callback(self):
        from unittest.mock import AsyncMock, MagicMock

        from litellm.exceptions import RateLimitError
        from litellm.responses.streaming_iterator import ResponsesWebSocketStreaming

        quota_callback = MagicMock()
        quota_callback.enforce_project_io_token_quota_for_frame = AsyncMock(
            side_effect=RateLimitError(message="project OTPM exceeded", llm_provider="", model="")
        )

        mock_backend_ws = MagicMock()
        mock_backend_ws.send = AsyncMock()
        mock_websocket = MagicMock()
        mock_websocket.send_text = AsyncMock()

        handler = ResponsesWebSocketStreaming(
            websocket=mock_websocket,
            backend_ws=mock_backend_ws,
            logging_obj=MagicMock(),
            authorized_model="gpt-4o",
            quota_callbacks=[quota_callback],
        )

        allowed = await handler._enforce_or_reject_frame(
            json.dumps({"type": "response.create", "input": "hi"})
        )

        assert allowed is False
        mock_backend_ws.send.assert_not_called()
        mock_websocket.send_text.assert_called_once()
        assert "rate_limit_exceeded" in mock_websocket.send_text.call_args[0][0]

    @pytest.mark.asyncio
    async def test_native_handler_forwards_frame_allowed_by_quota_callback(self):
        from unittest.mock import AsyncMock, MagicMock

        from litellm.responses.streaming_iterator import ResponsesWebSocketStreaming

        quota_callback = MagicMock()
        quota_callback.enforce_project_io_token_quota_for_frame = AsyncMock(return_value=None)

        handler = ResponsesWebSocketStreaming(
            websocket=MagicMock(),
            backend_ws=MagicMock(),
            logging_obj=MagicMock(),
            authorized_model="gpt-4o",
            quota_callbacks=[quota_callback],
        )

        allowed = await handler._enforce_or_reject_frame(
            json.dumps({"type": "response.create", "input": "hi"})
        )

        assert allowed is True
        quota_callback.enforce_project_io_token_quota_for_frame.assert_awaited_once()


class TestNativeWebSocketGuardrails:
    @pytest.mark.asyncio
    async def test_response_create_injects_authorized_model(self):
        import json
        from unittest.mock import MagicMock

        from litellm.responses.streaming_iterator import ResponsesWebSocketStreaming

        handler = ResponsesWebSocketStreaming(
            websocket=MagicMock(),
            backend_ws=MagicMock(),
            logging_obj=MagicMock(),
            authorized_model="authorized-deployment",
        )

        flat_message = await handler._mask_response_create(
            json.dumps({"type": "response.create", "input": "hi"})
        )
        nested_message = await handler._mask_response_create(
            json.dumps({"type": "response.create", "response": {"input": "hi"}})
        )

        assert json.loads(flat_message)["model"] == "authorized-deployment"
        assert (
            json.loads(nested_message)["response"]["model"] == "authorized-deployment"
        )

    @pytest.mark.asyncio
    async def test_native_websocket_merges_deployment_defaults(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from litellm.responses.main import _aresponses_websocket

        class FakeBackendWebSocket:
            def __init__(self):
                self.send = AsyncMock()
                self.close = AsyncMock()

            async def recv(self, decode=False):
                await asyncio.Future()

        backend_websocket = FakeBackendWebSocket()

        class FakeConnect:
            def __init__(self, url, **kwargs):
                pass

            async def __aenter__(self):
                return backend_websocket

            async def __aexit__(self, *args):
                pass

        first_message = json.dumps(
            {
                "type": "response.create",
                "model": "gpt-4o-mini",
                "input": "hi",
                "service_tier": "default",
            }
        )
        websocket = MagicMock()
        websocket.receive_text = AsyncMock(side_effect=RuntimeError("disconnect"))

        with patch("websockets.connect", FakeConnect):
            await _aresponses_websocket.__wrapped__(
                model="openai/gpt-4o-mini",
                websocket=websocket,
                api_key="sk-test",
                litellm_logging_obj=MagicMock(),
                first_message=first_message,
                reasoning_effort="high",
                service_tier="priority",
                extra_body={"provider_default": "configured"},
            )

        sent_message = json.loads(backend_websocket.send.await_args.args[0])
        assert sent_message["reasoning"] == {"effort": "high"}
        assert sent_message["service_tier"] == "default"
        assert sent_message["provider_default"] == "configured"

    @pytest.mark.asyncio
    async def test_completed_event_with_null_response_passes_through(self):
        from unittest.mock import MagicMock

        from litellm.responses.streaming_iterator import ResponsesWebSocketStreaming

        class Guardrail:
            def get_presidio_settings_from_request_data(self, request_data):
                return None

            def _unmask_pii_text(self, text, pii_tokens):
                return text

        event = '{"type":"response.completed","response":null}'
        guardrail = Guardrail()
        handler = ResponsesWebSocketStreaming(
            websocket=MagicMock(),
            backend_ws=MagicMock(),
            logging_obj=MagicMock(),
            request_data={"metadata": {"pii_tokens": {"<TOKEN_1>": "secret"}}},
            guardrail_callbacks=[guardrail],
            output_guardrail_callbacks=[guardrail],
        )

        assert handler._unmask_response_event(event) == event
        assert await handler._mask_response_completed(event) == event

    @pytest.mark.asyncio
    async def test_output_masking_suppresses_delta_without_calling_presidio(self):
        import json
        from unittest.mock import AsyncMock, MagicMock

        import websockets.exceptions

        from litellm.responses.streaming_iterator import ResponsesWebSocketStreaming

        class RecordingGuardrail:
            def __init__(self):
                self.check_pii_calls = []

            def get_presidio_settings_from_request_data(self, request_data):
                return None

            def _unmask_pii_text(self, text, pii_tokens):
                return text

            async def check_pii(
                self, text, output_parse_pii, presidio_config, request_data
            ):
                self.check_pii_calls.append(text)
                return text

        class FakeBackendWS:
            def __init__(self, events):
                self._events = list(events)

            async def recv(self, decode=False):
                if self._events:
                    return self._events.pop(0)
                raise websockets.exceptions.ConnectionClosed(None, None)

        guardrail = RecordingGuardrail()
        client_ws = MagicMock()
        client_ws.send_text = AsyncMock()
        logging_obj = MagicMock()
        logging_obj.dispatch_success_handlers = AsyncMock()

        delta_event = json.dumps(
            {"type": "response.output_text.delta", "delta": "alice@example.com"}
        )
        completed_event = json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        {
                            "content": [
                                {"type": "output_text", "text": "alice@example.com"}
                            ]
                        }
                    ]
                },
            }
        )

        handler = ResponsesWebSocketStreaming(
            websocket=client_ws,
            backend_ws=FakeBackendWS([delta_event, completed_event]),
            logging_obj=logging_obj,
            output_guardrail_callbacks=[guardrail],
        )

        await handler.backend_to_client()

        # The delta event must be suppressed without ever invoking Presidio,
        # so check_pii is called exactly once (for the completed event only).
        assert guardrail.check_pii_calls == ["alice@example.com"]
        client_ws.send_text.assert_called_once()
        sent_payload = client_ws.send_text.call_args[0][0]
        assert json.loads(sent_payload)["type"] == "response.completed"

    @pytest.mark.asyncio
    async def test_output_masking_suppresses_text_bearing_done_events(self):
        import json
        from unittest.mock import AsyncMock, MagicMock

        import websockets.exceptions

        from litellm.responses.streaming_iterator import ResponsesWebSocketStreaming

        class MaskingGuardrail:
            def __init__(self):
                self.check_pii_calls = []

            def get_presidio_settings_from_request_data(self, request_data):
                return None

            def _unmask_pii_text(self, text, pii_tokens):
                return text

            async def check_pii(
                self, text, output_parse_pii, presidio_config, request_data
            ):
                self.check_pii_calls.append(text)
                return text.replace("alice@example.com", "<EMAIL_ADDRESS>")

        class FakeBackendWS:
            def __init__(self, events):
                self._events = list(events)

            async def recv(self, decode=False):
                if self._events:
                    return self._events.pop(0)
                raise websockets.exceptions.ConnectionClosed(None, None)

        guardrail = MaskingGuardrail()
        client_ws = MagicMock()
        client_ws.send_text = AsyncMock()
        logging_obj = MagicMock()
        logging_obj.dispatch_success_handlers = AsyncMock()

        done_events = [
            json.dumps(
                {"type": "response.output_text.done", "text": "alice@example.com"}
            ),
            json.dumps(
                {
                    "type": "response.content_part.done",
                    "part": {"type": "output_text", "text": "alice@example.com"},
                }
            ),
            json.dumps(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "alice@example.com"}
                        ],
                    },
                }
            ),
        ]
        completed_event = json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        {
                            "content": [
                                {"type": "output_text", "text": "alice@example.com"}
                            ]
                        }
                    ]
                },
            }
        )

        handler = ResponsesWebSocketStreaming(
            websocket=client_ws,
            backend_ws=FakeBackendWS(done_events + [completed_event]),
            logging_obj=logging_obj,
            output_guardrail_callbacks=[guardrail],
        )

        await handler.backend_to_client()

        # Text-bearing done events carry the full output before response.completed
        # arrives; they must be suppressed so unmasked PII never reaches the
        # client, and Presidio is only invoked for response.completed.
        assert guardrail.check_pii_calls == ["alice@example.com"]
        client_ws.send_text.assert_called_once()
        sent_payload = client_ws.send_text.call_args[0][0]
        assert json.loads(sent_payload)["type"] == "response.completed"
        assert "alice@example.com" not in sent_payload
        assert "<EMAIL_ADDRESS>" in sent_payload


class _FakeWSGuardrail:
    """Presidio-like guardrail double for the WebSocket masking hooks.

    ``check_pii`` replaces each known PII string with its token. When
    ``output_parse_pii`` is True (input masking) the token->original map is
    persisted into ``request_data["metadata"]["pii_tokens"]`` so the response
    path can reverse it. ``_unmask_pii_text`` performs that reversal.
    """

    def __init__(self, mask_map=None):
        self.mask_map = mask_map or {"alice@example.com": "<EMAIL_ADDRESS_1>"}
        self.output_parse_pii = True
        self.apply_to_output = True

    def get_presidio_settings_from_request_data(self, request_data):
        return None

    async def check_pii(self, text, output_parse_pii, presidio_config, request_data):
        masked = text
        tokens = {}
        for original, token in self.mask_map.items():
            if original in masked:
                masked = masked.replace(original, token)
                tokens[token] = original
        if output_parse_pii and tokens:
            metadata = request_data.setdefault("metadata", {})
            metadata.setdefault("pii_tokens", {}).update(tokens)
        return masked

    def _unmask_pii_text(self, text, pii_tokens):
        for token, original in pii_tokens.items():
            text = text.replace(token, original)
        return text


def _make_streaming(**kwargs):
    from unittest.mock import MagicMock

    from litellm.responses.streaming_iterator import ResponsesWebSocketStreaming

    kwargs.setdefault("websocket", MagicMock())
    kwargs.setdefault("backend_ws", MagicMock())
    kwargs.setdefault("logging_obj", MagicMock())
    return ResponsesWebSocketStreaming(**kwargs)


class TestNativeWebSocketGuardrailMasking:
    """Exercises the input/output PII masking hooks on ResponsesWebSocketStreaming."""

    @pytest.mark.asyncio
    async def test_mask_response_create_flat_string_input(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(
            request_data={},
            guardrail_callbacks=[guardrail],
            authorized_model="auth-model",
        )

        masked = await handler._mask_response_create(
            json.dumps(
                {"type": "response.create", "input": "email alice@example.com now"}
            )
        )
        obj = json.loads(masked)

        assert obj["model"] == "auth-model"
        assert obj["input"] == "email <EMAIL_ADDRESS_1> now"
        assert handler.request_data["metadata"]["pii_tokens"] == {
            "<EMAIL_ADDRESS_1>": "alice@example.com"
        }

    @pytest.mark.asyncio
    async def test_mask_response_create_list_content_string(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(request_data={}, guardrail_callbacks=[guardrail])

        masked = await handler._mask_response_create(
            json.dumps(
                {
                    "type": "response.create",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": "ping alice@example.com",
                        }
                    ],
                }
            )
        )
        obj = json.loads(masked)

        assert obj["input"][0]["content"] == "ping <EMAIL_ADDRESS_1>"

    @pytest.mark.asyncio
    async def test_mask_response_create_input_text_blocks(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(request_data={}, guardrail_callbacks=[guardrail])

        masked = await handler._mask_response_create(
            json.dumps(
                {
                    "type": "response.create",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "alice@example.com"},
                                {"type": "input_image", "image_url": "http://x"},
                            ],
                        }
                    ],
                }
            )
        )
        obj = json.loads(masked)
        blocks = obj["input"][0]["content"]

        assert blocks[0]["text"] == "<EMAIL_ADDRESS_1>"
        assert blocks[1]["image_url"] == "http://x"

    @pytest.mark.asyncio
    async def test_mask_response_create_function_call_output_string(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(request_data={}, guardrail_callbacks=[guardrail])

        masked = await handler._mask_response_create(
            json.dumps(
                {
                    "type": "response.create",
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "call_1",
                            "output": "tool returned alice@example.com",
                        }
                    ],
                }
            )
        )
        obj = json.loads(masked)

        assert obj["input"][0]["output"] == "tool returned <EMAIL_ADDRESS_1>"
        assert handler.request_data["metadata"]["pii_tokens"] == {
            "<EMAIL_ADDRESS_1>": "alice@example.com"
        }

    @pytest.mark.asyncio
    async def test_mask_response_create_function_call_output_blocks(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(request_data={}, guardrail_callbacks=[guardrail])

        masked = await handler._mask_response_create(
            json.dumps(
                {
                    "type": "response.create",
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "call_1",
                            "output": [
                                {"type": "output_text", "text": "alice@example.com"},
                                {"type": "input_image", "image_url": "http://x"},
                            ],
                        }
                    ],
                }
            )
        )
        obj = json.loads(masked)
        blocks = obj["input"][0]["output"]

        assert blocks[0]["text"] == "<EMAIL_ADDRESS_1>"
        assert blocks[1]["image_url"] == "http://x"

    @pytest.mark.asyncio
    async def test_mask_response_create_nested_shape(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(
            request_data={},
            guardrail_callbacks=[guardrail],
            authorized_model="auth-model",
        )

        masked = await handler._mask_response_create(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {"input": "alice@example.com", "model": "spoofed"},
                }
            )
        )
        obj = json.loads(masked)

        assert obj["response"]["model"] == "auth-model"
        assert obj["response"]["input"] == "<EMAIL_ADDRESS_1>"

    @pytest.mark.asyncio
    async def test_mask_response_create_flat_instructions(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(request_data={}, guardrail_callbacks=[guardrail])

        masked = await handler._mask_response_create(
            json.dumps(
                {
                    "type": "response.create",
                    "input": "hi",
                    "instructions": "reply to alice@example.com",
                }
            )
        )
        obj = json.loads(masked)

        assert obj["instructions"] == "reply to <EMAIL_ADDRESS_1>"
        assert handler.request_data["metadata"]["pii_tokens"] == {
            "<EMAIL_ADDRESS_1>": "alice@example.com"
        }

    @pytest.mark.asyncio
    async def test_mask_response_create_nested_instructions(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(request_data={}, guardrail_callbacks=[guardrail])

        masked = await handler._mask_response_create(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "input": "hi",
                        "instructions": "email alice@example.com",
                    },
                }
            )
        )
        obj = json.loads(masked)

        assert obj["response"]["instructions"] == "email <EMAIL_ADDRESS_1>"
        assert handler.request_data["metadata"]["pii_tokens"] == {
            "<EMAIL_ADDRESS_1>": "alice@example.com"
        }

    @pytest.mark.asyncio
    async def test_mask_response_create_non_create_unchanged(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(
            request_data={},
            guardrail_callbacks=[guardrail],
            authorized_model="auth-model",
        )

        message = json.dumps({"type": "response.cancel", "input": "alice@example.com"})
        assert await handler._mask_response_create(message) == message

    @pytest.mark.asyncio
    async def test_mask_response_create_invalid_json_unchanged(self):
        handler = _make_streaming(
            request_data={}, guardrail_callbacks=[_FakeWSGuardrail()]
        )
        assert await handler._mask_response_create("not json {{{") == "not json {{{"

    @pytest.mark.asyncio
    async def test_mask_response_create_model_only_without_guardrails(self):
        handler = _make_streaming(request_data={}, authorized_model="auth-model")

        masked = await handler._mask_response_create(
            json.dumps({"type": "response.create", "input": "alice@example.com"})
        )
        obj = json.loads(masked)

        assert obj["model"] == "auth-model"
        assert obj["input"] == "alice@example.com"

    @pytest.mark.asyncio
    async def test_mask_response_create_no_op_without_model_or_guardrails(self):
        handler = _make_streaming(request_data={})
        message = json.dumps({"type": "response.create", "input": "alice@example.com"})
        assert await handler._mask_response_create(message) == message

    @pytest.mark.asyncio
    async def test_mask_response_create_list_with_non_dict_item(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(request_data={}, guardrail_callbacks=[guardrail])

        masked = await handler._mask_response_create(
            json.dumps(
                {
                    "type": "response.create",
                    "input": [
                        "not-a-dict",
                        {
                            "type": "message",
                            "role": "user",
                            "content": "alice@example.com",
                        },
                    ],
                }
            )
        )
        obj = json.loads(masked)
        assert obj["input"][0] == "not-a-dict"
        assert obj["input"][1]["content"] == "<EMAIL_ADDRESS_1>"

    def test_enforce_authorized_model_no_authorized_model(self):
        handler = _make_streaming(request_data={})
        assert handler._enforce_authorized_model({"model": "anything"}) is False

    def test_enforce_authorized_model_nested_with_top_level_model(self):
        handler = _make_streaming(request_data={}, authorized_model="auth-model")
        msg = {"response": {"model": "spoofed"}, "model": "also-spoofed"}
        assert handler._enforce_authorized_model(msg) is True
        assert msg["response"]["model"] == "auth-model"
        assert msg["model"] == "auth-model"

    @pytest.mark.asyncio
    async def test_unmask_response_event_completed(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(
            request_data={
                "metadata": {"pii_tokens": {"<EMAIL_ADDRESS_1>": "alice@example.com"}}
            },
            guardrail_callbacks=[guardrail],
        )

        event = json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        {
                            "content": [
                                {"type": "output_text", "text": "to <EMAIL_ADDRESS_1>"}
                            ]
                        }
                    ]
                },
            }
        )
        unmasked = json.loads(handler._unmask_response_event(event))
        assert (
            unmasked["response"]["output"][0]["content"][0]["text"]
            == "to alice@example.com"
        )

    @pytest.mark.asyncio
    async def test_unmask_response_event_delta(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(
            request_data={
                "metadata": {"pii_tokens": {"<EMAIL_ADDRESS_1>": "alice@example.com"}}
            },
            guardrail_callbacks=[guardrail],
        )

        event = json.dumps(
            {"type": "response.output_text.delta", "delta": "<EMAIL_ADDRESS_1>"}
        )
        unmasked = json.loads(handler._unmask_response_event(event))
        assert unmasked["delta"] == "alice@example.com"

    def test_unmask_response_event_no_tokens_unchanged(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(request_data={}, guardrail_callbacks=[guardrail])
        event = json.dumps(
            {"type": "response.output_text.delta", "delta": "<EMAIL_ADDRESS_1>"}
        )
        assert handler._unmask_response_event(event) == event

    def test_unmask_response_event_no_guardrails_unchanged(self):
        handler = _make_streaming(
            request_data={"metadata": {"pii_tokens": {"<EMAIL_ADDRESS_1>": "x"}}}
        )
        event = json.dumps({"type": "response.completed", "response": {}})
        assert handler._unmask_response_event(event) == event

    def test_unmask_response_event_invalid_json_unchanged(self):
        handler = _make_streaming(
            request_data={"metadata": {"pii_tokens": {"<EMAIL_ADDRESS_1>": "x"}}},
            guardrail_callbacks=[_FakeWSGuardrail()],
        )
        assert handler._unmask_response_event("not json {{{") == "not json {{{"

    def test_unmask_response_event_non_dict_response_unchanged(self):
        handler = _make_streaming(
            request_data={"metadata": {"pii_tokens": {"<EMAIL_ADDRESS_1>": "x"}}},
            guardrail_callbacks=[_FakeWSGuardrail()],
        )
        event = json.dumps({"type": "response.completed", "response": ["bad-shape"]})
        assert handler._unmask_response_event(event) == event

    def test_unmask_response_event_malformed_output_items_unchanged(self):
        handler = _make_streaming(
            request_data={
                "metadata": {"pii_tokens": {"<EMAIL_ADDRESS_1>": "alice@example.com"}}
            },
            guardrail_callbacks=[_FakeWSGuardrail()],
        )
        event = json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        "not-a-dict",
                        {"content": "not-a-list"},
                        {"content": ["not-a-dict-block"]},
                    ]
                },
            }
        )
        assert handler._unmask_response_event(event) == event

    def test_unmask_response_event_other_event_type_unchanged(self):
        handler = _make_streaming(
            request_data={"metadata": {"pii_tokens": {"<EMAIL_ADDRESS_1>": "x"}}},
            guardrail_callbacks=[_FakeWSGuardrail()],
        )
        event = json.dumps(
            {"type": "response.in_progress", "delta": "<EMAIL_ADDRESS_1>"}
        )
        assert handler._unmask_response_event(event) == event

    @pytest.mark.asyncio
    async def test_mask_response_completed_event(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(
            request_data={}, output_guardrail_callbacks=[guardrail]
        )

        event = json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        {
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "contact alice@example.com",
                                }
                            ]
                        }
                    ]
                },
            }
        )
        masked = json.loads(await handler._mask_response_completed(event))
        assert (
            masked["response"]["output"][0]["content"][0]["text"]
            == "contact <EMAIL_ADDRESS_1>"
        )

    @pytest.mark.asyncio
    async def test_mask_response_completed_masks_function_call_arguments(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(
            request_data={}, output_guardrail_callbacks=[guardrail]
        )

        event = json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        {
                            "type": "function_call",
                            "name": "send_email",
                            "arguments": '{"to": "alice@example.com"}',
                        }
                    ]
                },
            }
        )
        masked = json.loads(await handler._mask_response_completed(event))
        assert (
            masked["response"]["output"][0]["arguments"]
            == '{"to": "<EMAIL_ADDRESS_1>"}'
        )

    @pytest.mark.asyncio
    async def test_mask_response_completed_masks_reasoning_summary(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(
            request_data={}, output_guardrail_callbacks=[guardrail]
        )

        event = json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        {
                            "type": "reasoning",
                            "summary": [
                                {
                                    "type": "summary_text",
                                    "text": "user is alice@example.com",
                                }
                            ],
                        }
                    ]
                },
            }
        )
        masked = json.loads(await handler._mask_response_completed(event))
        assert (
            masked["response"]["output"][0]["summary"][0]["text"]
            == "user is <EMAIL_ADDRESS_1>"
        )

    @pytest.mark.asyncio
    async def test_mask_response_completed_delta_unchanged(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(
            request_data={}, output_guardrail_callbacks=[guardrail]
        )

        event = json.dumps(
            {"type": "response.output_text.delta", "delta": "alice@example.com"}
        )
        assert await handler._mask_response_completed(event) == event

    @pytest.mark.asyncio
    async def test_mask_response_completed_no_guardrails_unchanged(self):
        handler = _make_streaming(request_data={})
        event = json.dumps(
            {"type": "response.output_text.delta", "delta": "alice@example.com"}
        )
        assert await handler._mask_response_completed(event) == event

    @pytest.mark.asyncio
    async def test_mask_response_completed_invalid_json_unchanged(self):
        handler = _make_streaming(
            request_data={}, output_guardrail_callbacks=[_FakeWSGuardrail()]
        )
        assert await handler._mask_response_completed("not json {{{") == "not json {{{"

    @pytest.mark.asyncio
    async def test_mask_response_completed_malformed_unchanged(self):
        handler = _make_streaming(
            request_data={}, output_guardrail_callbacks=[_FakeWSGuardrail()]
        )
        event = json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        "not-a-dict",
                        {"content": "not-a-list"},
                        {"content": ["not-a-dict-block"]},
                    ]
                },
            }
        )
        assert await handler._mask_response_completed(event) == event

    @pytest.mark.asyncio
    async def test_mask_response_completed_non_dict_response_unchanged(self):
        handler = _make_streaming(
            request_data={}, output_guardrail_callbacks=[_FakeWSGuardrail()]
        )
        event = json.dumps({"type": "response.completed", "response": ["bad"]})
        assert await handler._mask_response_completed(event) == event

    @pytest.mark.asyncio
    async def test_client_to_backend_masks_and_enforces_model(self):
        from unittest.mock import AsyncMock

        guardrail = _FakeWSGuardrail()
        backend_ws = MagicMock()
        backend_ws.send = AsyncMock()
        websocket = MagicMock()
        websocket.receive_text = AsyncMock(
            side_effect=[
                json.dumps(
                    {"type": "response.create", "input": "ping alice@example.com"}
                ),
                Exception("stop"),
            ]
        )

        handler = _make_streaming(
            websocket=websocket,
            backend_ws=backend_ws,
            request_data={},
            first_message=json.dumps(
                {"type": "response.create", "input": "alice@example.com"}
            ),
            guardrail_callbacks=[guardrail],
            authorized_model="auth-model",
        )

        await handler.client_to_backend()

        assert backend_ws.send.await_count == 2
        first_sent = json.loads(backend_ws.send.await_args_list[0][0][0])
        assert first_sent["model"] == "auth-model"
        assert first_sent["input"] == "<EMAIL_ADDRESS_1>"
        second_sent = json.loads(backend_ws.send.await_args_list[1][0][0])
        assert second_sent["model"] == "auth-model"
        assert second_sent["input"] == "ping <EMAIL_ADDRESS_1>"
        assert handler.request_data["metadata"]["pii_tokens"] == {
            "<EMAIL_ADDRESS_1>": "alice@example.com"
        }

    @pytest.mark.asyncio
    async def test_backend_to_client_suppresses_deltas_and_masks_completed(self):
        from unittest.mock import AsyncMock

        import websockets.exceptions  # noqa: F401  (lazy submodule must be importable)

        guardrail = _FakeWSGuardrail()
        websocket = MagicMock()
        websocket.send_text = AsyncMock()
        backend_ws = MagicMock()
        backend_ws.recv = AsyncMock(
            side_effect=[
                json.dumps(
                    {"type": "response.output_text.delta", "delta": "alice@example.com"}
                ),
                json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "output": [
                                {
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": "contact alice@example.com",
                                        }
                                    ]
                                }
                            ]
                        },
                    }
                ),
                Exception("stop"),
            ]
        )
        logging_obj = MagicMock()
        logging_obj.dispatch_success_handlers = AsyncMock()

        handler = _make_streaming(
            websocket=websocket,
            backend_ws=backend_ws,
            logging_obj=logging_obj,
            request_data={},
            output_guardrail_callbacks=[guardrail],
        )

        await handler.backend_to_client()

        websocket.send_text.assert_awaited_once()
        forwarded = json.loads(websocket.send_text.await_args[0][0])
        assert forwarded["type"] == "response.completed"
        assert (
            forwarded["response"]["output"][0]["content"][0]["text"]
            == "contact <EMAIL_ADDRESS_1>"
        )

    @pytest.mark.asyncio
    async def test_backend_to_client_suppresses_function_call_arguments_done(self):
        from unittest.mock import AsyncMock

        import websockets.exceptions  # noqa: F401  (lazy submodule must be importable)

        guardrail = _FakeWSGuardrail()
        websocket = MagicMock()
        websocket.send_text = AsyncMock()
        backend_ws = MagicMock()
        backend_ws.recv = AsyncMock(
            side_effect=[
                json.dumps(
                    {
                        "type": "response.function_call_arguments.done",
                        "arguments": '{"to": "alice@example.com"}',
                    }
                ),
                json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "output": [
                                {
                                    "type": "function_call",
                                    "name": "send_email",
                                    "arguments": '{"to": "alice@example.com"}',
                                }
                            ]
                        },
                    }
                ),
                Exception("stop"),
            ]
        )
        logging_obj = MagicMock()
        logging_obj.dispatch_success_handlers = AsyncMock()

        handler = _make_streaming(
            websocket=websocket,
            backend_ws=backend_ws,
            logging_obj=logging_obj,
            request_data={},
            output_guardrail_callbacks=[guardrail],
        )

        await handler.backend_to_client()

        # The unmasked function-call arguments must never reach the client; only
        # the masked response.completed is forwarded.
        websocket.send_text.assert_awaited_once()
        sent_payload = websocket.send_text.await_args[0][0]
        forwarded = json.loads(sent_payload)
        assert forwarded["type"] == "response.completed"
        assert (
            forwarded["response"]["output"][0]["arguments"]
            == '{"to": "<EMAIL_ADDRESS_1>"}'
        )
        assert "alice@example.com" not in sent_payload

    @pytest.mark.asyncio
    async def test_backend_to_client_suppresses_reasoning_summary_text_done(self):
        from unittest.mock import AsyncMock

        import websockets.exceptions  # noqa: F401  (lazy submodule must be importable)

        guardrail = _FakeWSGuardrail()
        websocket = MagicMock()
        websocket.send_text = AsyncMock()
        backend_ws = MagicMock()
        backend_ws.recv = AsyncMock(
            side_effect=[
                json.dumps(
                    {
                        "type": "response.reasoning_summary_text.done",
                        "text": "contact alice@example.com",
                    }
                ),
                json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "output": [
                                {
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": "done",
                                        }
                                    ]
                                }
                            ]
                        },
                    }
                ),
                Exception("stop"),
            ]
        )
        logging_obj = MagicMock()
        logging_obj.dispatch_success_handlers = AsyncMock()

        handler = _make_streaming(
            websocket=websocket,
            backend_ws=backend_ws,
            logging_obj=logging_obj,
            request_data={},
            output_guardrail_callbacks=[guardrail],
        )

        await handler.backend_to_client()

        # The reasoning-summary done event carries the full reasoning text before
        # response.completed arrives; it must be suppressed so unmasked PII never
        # reaches the client.
        websocket.send_text.assert_awaited_once()
        sent_payload = websocket.send_text.await_args[0][0]
        assert json.loads(sent_payload)["type"] == "response.completed"
        assert "alice@example.com" not in sent_payload

    @pytest.mark.asyncio
    async def test_backend_to_client_suppresses_reasoning_summary_part_done(self):
        from unittest.mock import AsyncMock

        import websockets.exceptions  # noqa: F401  (lazy submodule must be importable)

        guardrail = _FakeWSGuardrail()
        websocket = MagicMock()
        websocket.send_text = AsyncMock()
        backend_ws = MagicMock()
        backend_ws.recv = AsyncMock(
            side_effect=[
                json.dumps(
                    {
                        "type": "response.reasoning_summary_part.done",
                        "part": {
                            "type": "summary_text",
                            "text": "user is alice@example.com",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "output": [
                                {
                                    "type": "reasoning",
                                    "summary": [
                                        {
                                            "type": "summary_text",
                                            "text": "user is alice@example.com",
                                        }
                                    ],
                                }
                            ]
                        },
                    }
                ),
                Exception("stop"),
            ]
        )
        logging_obj = MagicMock()
        logging_obj.dispatch_success_handlers = AsyncMock()

        handler = _make_streaming(
            websocket=websocket,
            backend_ws=backend_ws,
            logging_obj=logging_obj,
            request_data={},
            output_guardrail_callbacks=[guardrail],
        )

        await handler.backend_to_client()

        # The reasoning-summary part-done event carries the full reasoning text
        # before response.completed arrives; it must be suppressed, and the
        # reasoning summary in response.completed must itself be masked.
        websocket.send_text.assert_awaited_once()
        sent_payload = websocket.send_text.await_args[0][0]
        forwarded = json.loads(sent_payload)
        assert forwarded["type"] == "response.completed"
        assert (
            forwarded["response"]["output"][0]["summary"][0]["text"]
            == "user is <EMAIL_ADDRESS_1>"
        )
        assert "alice@example.com" not in sent_payload


class TestWebSocketChunkTypes:
    """Test handling of different chunk types from streaming responses"""

    def test_serialize_function_call_chunk(self):
        """Test serialization of function call chunks"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.function_call.added",
            "response_id": "resp_123",
            "item_id": "call_456",
            "output_index": 0,
            "call_id": "call_456",
            "name": "get_weather",
            "arguments": "",
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.function_call.added" in serialized
        assert "get_weather" in serialized

    def test_serialize_function_call_arguments_delta(self):
        """Test serialization of function call arguments delta"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.function_call_arguments.delta",
            "response_id": "resp_123",
            "item_id": "call_456",
            "output_index": 0,
            "call_id": "call_456",
            "delta": '{"location"',
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.function_call_arguments.delta" in serialized
        assert "location" in serialized

    def test_serialize_function_call_arguments_done(self):
        """Test serialization of function call arguments done"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.function_call_arguments.done",
            "response_id": "resp_123",
            "item_id": "call_456",
            "output_index": 0,
            "call_id": "call_456",
            "arguments": '{"location": "Paris"}',
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.function_call_arguments.done" in serialized
        assert "Paris" in serialized

    def test_serialize_reasoning_content_delta(self):
        """Test serialization of reasoning content delta"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.reasoning_content.delta",
            "response_id": "resp_123",
            "item_id": "msg_456",
            "output_index": 0,
            "content_index": 0,
            "delta": "Thinking step 1...",
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.reasoning_content.delta" in serialized
        assert "Thinking step 1" in serialized

    def test_serialize_reasoning_content_done(self):
        """Test serialization of reasoning content done"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.reasoning_content.done",
            "response_id": "resp_123",
            "item_id": "msg_456",
            "output_index": 0,
            "content_index": 0,
            "reasoning_content": "Complete reasoning...",
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.reasoning_content.done" in serialized
        assert "Complete reasoning" in serialized

    def test_extract_output_messages_preserves_multiple_messages(self):
        """Test that multiple output messages are all preserved"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "First message"}],
                    },
                    {
                        "type": "function_call",
                        "id": "call_123",
                        "name": "get_weather",
                        "arguments": "{}",
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Second message"}],
                    },
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 3
        assert messages[0]["content"][0]["text"] == "First message"
        assert messages[1]["type"] == "function_call"
        assert messages[2]["content"][0]["text"] == "Second message"

    def test_input_to_messages_with_mixed_content_types(self):
        """Test input conversion with mixed content types"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        input_list = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Question"},
                    {"type": "input_image", "image_url": "https://example.com/img.png"},
                ],
            }
        ]

        messages = ManagedResponsesWebSocketHandler._input_to_messages(input_list)
        assert len(messages) == 1
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][0]["type"] == "input_text"
        assert messages[0]["content"][1]["type"] == "input_image"

    def test_extract_output_messages_with_mixed_text_types(self):
        """Test that both 'output_text' and 'text' types are extracted"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Part 1"},
                            {"type": "text", "text": "Part 2"},
                        ],
                    }
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 1
        assert messages[0]["content"][0]["text"] == "Part 1Part 2"


class TestNativeWebSocketUrlConstruction:
    """Test that native WebSocket URLs include the model query parameter.

    These tests mock websockets.connect so they exercise the actual URL-building
    code inside BaseLLMHTTPHandler.async_responses_websocket rather than
    reimplementing the logic themselves.
    """

    @pytest.mark.asyncio
    async def test_openai_ws_url_includes_model(self):
        """Handler must pass ?model= in the URL to the backend WebSocket."""
        from unittest.mock import AsyncMock, MagicMock, patch

        captured_urls = []

        class FakeConnect:
            def __init__(self, url, **kwargs):
                captured_urls.append(url)

            async def __aenter__(self):
                raise Exception("stop")

            async def __aexit__(self, *args):
                pass

        mock_config = MagicMock(spec=OpenAIResponsesAPIConfig)
        mock_config.supports_native_websocket.return_value = True
        mock_config.get_websocket_url.return_value = "wss://api.openai.com/v1/responses"
        mock_config.validate_environment.return_value = {}

        mock_logging = MagicMock()
        mock_logging.pre_call = MagicMock()

        from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

        handler = BaseLLMHTTPHandler()

        mock_ws = MagicMock()
        mock_ws.close = AsyncMock()

        with patch("websockets.connect", FakeConnect):
            await handler.async_responses_websocket(
                model="gpt-4o-mini",
                websocket=mock_ws,
                logging_obj=mock_logging,
                responses_api_provider_config=mock_config,
                api_key="sk-test",
            )

        assert len(captured_urls) == 1
        from urllib.parse import parse_qs, urlparse

        qs = parse_qs(urlparse(captured_urls[0]).query)
        assert qs.get("model") == [
            "gpt-4o-mini"
        ], f"Expected model in URL, got: {captured_urls[0]}"

    @pytest.mark.asyncio
    async def test_ws_url_preserves_existing_params_and_adds_model(self):
        """When api_base already has query params, model is added alongside them."""
        from unittest.mock import AsyncMock, MagicMock, patch

        captured_urls = []

        class FakeConnect:
            def __init__(self, url, **kwargs):
                captured_urls.append(url)

            async def __aenter__(self):
                raise Exception("stop")

            async def __aexit__(self, *args):
                pass

        mock_config = MagicMock(spec=OpenAIResponsesAPIConfig)
        mock_config.supports_native_websocket.return_value = True
        mock_config.get_websocket_url.return_value = (
            "wss://custom.example.com/v1/responses?api-version=2024-05-01"
        )
        mock_config.validate_environment.return_value = {}

        mock_logging = MagicMock()
        mock_logging.pre_call = MagicMock()

        from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

        handler = BaseLLMHTTPHandler()
        mock_ws = MagicMock()
        mock_ws.close = AsyncMock()

        with patch("websockets.connect", FakeConnect):
            await handler.async_responses_websocket(
                model="gpt-4o",
                websocket=mock_ws,
                logging_obj=mock_logging,
                responses_api_provider_config=mock_config,
                api_key="sk-test",
            )

        assert len(captured_urls) == 1
        from urllib.parse import parse_qs, urlparse

        qs = parse_qs(urlparse(captured_urls[0]).query)
        assert qs.get("model") == [
            "gpt-4o"
        ], f"model missing from URL: {captured_urls[0]}"
        assert qs.get("api-version") == [
            "2024-05-01"
        ], f"existing param lost: {captured_urls[0]}"

    @pytest.mark.asyncio
    async def test_ws_passes_litellm_params_to_get_websocket_url(self):
        """Deployment api_version must reach get_websocket_url (Azure WS URL)."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_config = MagicMock(spec=OpenAIResponsesAPIConfig)
        mock_config.supports_native_websocket.return_value = True
        mock_config.get_websocket_url.return_value = (
            "wss://example.openai.azure.com/openai/v1/responses"
        )
        mock_config.validate_environment.return_value = {}

        mock_logging = MagicMock()
        mock_logging.pre_call = MagicMock()

        from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

        handler = BaseLLMHTTPHandler()
        mock_ws = MagicMock()
        mock_ws.close = AsyncMock()

        class FakeConnect:
            def __init__(self, url, **kwargs):
                pass

            async def __aenter__(self):
                raise Exception("stop")

            async def __aexit__(self, *args):
                pass

        with patch("websockets.connect", FakeConnect):
            await handler.async_responses_websocket(
                model="gpt-5.3-codex",
                websocket=mock_ws,
                logging_obj=mock_logging,
                responses_api_provider_config=mock_config,
                api_key="sk-test",
                api_base="https://example.openai.azure.com",
                api_version="2025-04-01-preview",
            )

        mock_config.get_websocket_url.assert_called_once()
        _, call_kwargs = mock_config.get_websocket_url.call_args
        assert call_kwargs["litellm_params"]["api_version"] == "2025-04-01-preview"

    @pytest.mark.asyncio
    async def test_native_websocket_handshake_failure_falls_back_to_managed_bridge(  # test-quality-ok: the regression IS which internal path runs after a failed native handshake — a faked HTTP boundary cannot tell "native failed, then bridged" apart from "native was never attempted", so the bridge collaborator is the observable under test. The caller-visible half (client socket never closed, never with 1011) is asserted below.
        self,
    ):
        """
        Regression test for rayward-internal/llm-gateway-infra#645's "HTTP 404
        during connection" row.

        Azure exposes `supports_native_websocket() == True`, but our Azure
        deployments don't expose a real `wss://` Responses endpoint, so the
        handshake fails (websockets raises `InvalidStatus`, not the deprecated
        `InvalidStatusCode` the old code caught, so it used to fall into the
        generic `except Exception` and close the client's already-accepted
        WebSocket with code 1011). The handshake failure must instead bridge
        through `ManagedResponsesWebSocketHandler`, since the client hasn't
        received a single frame yet and every other managed provider already
        proves that bridge works end-to-end.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        class FakeConnect:
            def __init__(self, url, **kwargs):
                pass

            async def __aenter__(self):
                raise Exception("server rejected WebSocket connection: HTTP 404")

            async def __aexit__(self, *args):
                pass

        mock_config = MagicMock(spec=AzureOpenAIResponsesAPIConfig)
        mock_config.supports_native_websocket.return_value = True
        mock_config.get_websocket_url.return_value = (
            "wss://myresource.cognitiveservices.azure.com/openai/v1/responses"
        )
        mock_config.model_in_websocket_url.return_value = False
        mock_config.validate_environment.return_value = {}

        mock_logging = MagicMock()
        mock_logging.pre_call = MagicMock()

        from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

        handler = BaseLLMHTTPHandler()
        mock_ws = MagicMock()
        mock_ws.close = AsyncMock()

        fake_managed_handler = MagicMock()
        fake_managed_handler.run = AsyncMock()
        fake_managed_handler_cls = MagicMock(return_value=fake_managed_handler)

        with patch("websockets.connect", FakeConnect), patch(  # test-quality-ok: the bridge class is the seam that tells "native failed, then bridged" apart from "native never tried"; a faked HTTP boundary cannot distinguish them. Client-observable half asserted below.
            "litellm.responses.streaming_iterator.ManagedResponsesWebSocketHandler",
            fake_managed_handler_cls,
        ):
            await handler.async_responses_websocket(
                model="gpt-5.5",
                websocket=mock_ws,
                logging_obj=mock_logging,
                responses_api_provider_config=mock_config,
                api_key="sk-azure-test",
                api_base="https://myresource.cognitiveservices.azure.com",
                custom_llm_provider="azure",
            )

        fake_managed_handler_cls.assert_called_once()
        assert fake_managed_handler_cls.call_args.kwargs["model"] == "gpt-5.5"
        fake_managed_handler.run.assert_awaited_once()
        # The caller-observable regression: before the fix the client's already-accepted
        # socket was closed with 1011 ("server rejected WebSocket connection: HTTP 404").
        mock_ws.close.assert_not_awaited()
        assert not [
            call for call in mock_ws.close.await_args_list if 1011 in call.args or call.kwargs.get("code") == 1011
        ], "client socket must never be closed with 1011 after a native handshake failure"


<<<<<<< HEAD
class TestWebSocketRateLimitEnforcementMechanism:
    """Mechanism tests for rayward-internal/llm-gateway-infra#657's
    non-native-provider (managed-path) leg -- ``ManagedResponsesWebSocketHandler``,
    used by Fireworks/Vertex/Bedrock-Mantle/etc. Native providers (Azure,
    OpenAI) take a completely different code path (``ResponsesWebSocketStreaming``)
    with its own, separately-tracked usage-accounting gap -- NOT covered by
    this fix or these tests.

    Measured on production 2026-08-28: two large WS turns (17,731 then
    ~13,000 tokens) billed the calling key exactly $0. A live re-check
    (corrected after an initial mis-ordered read of the raw log) established
    that the production run's 3 WebSocket rounds all preceded the 2 HTTPS
    control rounds -- so it does NOT show a fresh WS connection bypassing an
    *already*-exhausted TPM cap. Also corrected: DB spend logging itself
    (``_PROXY_track_cost_callback``) is NOT broken by the metadata collision
    -- it resolves identity via ``get_litellm_metadata_from_kwargs``, which
    is resilient to it (confirmed empirically, see
    ``TestWebSocketProxyIdentityMetadataMerge``).

    What this class documents, with file:line, is the STATIC mechanism by
    which a metadata collision reaches the ACTIVE rate limiter:
    - ``litellm/proxy/response_api_endpoints/endpoints.py``'s
      ``responses_websocket_endpoint`` calls
      ``ProxyBaseLLMRequestProcessing.common_processing_pre_call_logic``,
      which calls ``proxy_logging_obj.pre_call_hook(..., call_type=route_type)``
      (``litellm/proxy/common_request_processing.py:1966``) exactly once,
      before the WebSocket accepts any ``response.create`` frame.
    - ``ProxyLogging.pre_call_hook`` (``litellm/proxy/utils.py:1634+``)
      iterates ``litellm.callbacks`` (via ``_callback_capabilities()``) and
      calls ``_callback.async_pre_call_hook(user_api_key_dict, cache, data,
      call_type)`` for every registered CustomLogger that overrides it.
    - The registered rate limiter is ``_PROXY_MaxParallelRequestsHandler_v3``
      (``litellm/proxy/hooks/__init__.py:21`` -- the *default*
      ``"parallel_request_limiter"`` entry, unless
      ``LEGACY_MULTI_INSTANCE_RATE_LIMITING=true``), added to
      ``litellm.callbacks`` at startup via ``ProxyLogging._add_proxy_hooks``
      (``litellm/proxy/utils.py:786-806``). Its
      ``get_rate_limiter_for_call_type`` (``parallel_request_limiter_v3.py:
      2865-2870``) special-cases ONLY ``"acreate_batch"`` -- there is no
      exclusion for ``"_aresponses_websocket"``, so the generic per-key
      TPM/RPM check runs for it exactly as for any HTTP call type.
    - Because ``_PROXY_MaxParallelRequestsHandler_v3`` is a plain
      ``litellm.callbacks`` entry (not something only the HTTP request path
      invokes), its ``async_log_success_event``
      (``parallel_request_limiter_v3.py:4396+``) fires from the SDK's
      generic success-callback dispatch for ANY ``litellm.aresponses()``
      call -- including the per-turn calls
      ``ManagedResponsesWebSocketHandler._stream_and_forward`` makes deep
      inside a WS session. It reads ``standard_logging_object["metadata"]
      ["user_api_key_hash"]`` (``parallel_request_limiter_v3.py:4157``) to
      attribute the turn's tokens to the right counter -- the exact field
      the ``_inject_credentials`` fix protects from being dropped.
    - Its ``async_log_success_event`` (``parallel_request_limiter_v3.py:4396+``)
      reads ``standard_logging_object["metadata"]["user_api_key_hash"]``
      (``:4157``) to attribute a turn's tokens -- the exact field the
      ``_inject_credentials`` fix protects from being dropped by a colliding
      client ``metadata`` object (verified deterministically in
      ``TestWebSocketProxyIdentityMetadataMerge`` against
      ``litellm_params["metadata"]``, which ``standard_logging_object``
      construction reads from).

    NOT covered here: an end-to-end assertion that a WS turn's real usage
    makes a *subsequent* connection's ``pre_call_hook`` raise
    ``RateLimitError``. I attempted this against the real, registered
    ``_PROXY_MaxParallelRequestsHandler_v3`` and a real ``InternalUsageCache``
    and could not get a reliable, reproducible result: its reservation/
    correction accounting (``claim_request_stash_for_data`` /
    ``async_increment_reservation_aware_tokens``) behaved inconsistently
    across runs in ways not fully root-caused within the time available --
    sometimes the turn's actual usage never registered at all, sometimes
    only a small pre-call "floor" reservation persisted. Rather than present
    a flaky or misleading end-to-end test, that specific claim is RETRACTED
    as unverified; what stands, verified and deterministic, is the
    metadata-merge protection itself (``TestWebSocketProxyIdentityMetadataMerge``)
    and the mechanism trace above.
    """

    def test_parallel_request_limiter_v3_has_no_websocket_exclusion(self):
        """Guard the mechanism claim above: only acreate_batch gets a
        call-type-specific rate limiter: _aresponses_websocket must fall
        through to the generic per-key/team/user/org check."""
        from litellm import DualCache
        from litellm.proxy.hooks.parallel_request_limiter_v3 import (
            _PROXY_MaxParallelRequestsHandler_v3,
        )
        from litellm.proxy.utils import InternalUsageCache

        limiter = _PROXY_MaxParallelRequestsHandler_v3(InternalUsageCache(dual_cache=DualCache()))
        assert limiter.get_rate_limiter_for_call_type(call_type="_aresponses_websocket") is None
        assert limiter.get_rate_limiter_for_call_type(call_type="acreate_batch") is not None


class TestWebSocketProxyIdentityMetadataMerge:
    """Regression tests for rayward-internal/llm-gateway-infra#657's
    non-native-provider (managed-path) leg.

    NOT about DB spend logging: ``_PROXY_track_cost_callback`` resolves
    identity via ``get_litellm_metadata_from_kwargs`` (litellm/litellm_core_utils/
    core_helpers.py:280-297), which prefers ``litellm_params["litellm_metadata"]``
    over ``["metadata"]`` and even back-fills missing ``user_api_key*`` keys
    from ``metadata`` (``add_missing_spend_metadata_to_litellm_metadata``,
    same file :230-241) -- confirmed empirically against the real callback,
    with and without this fix, both resolve ``user_api_key`` correctly.

    What this DOES fix: a ``response.create`` frame may carry its own
    top-level ``metadata`` object -- a legal Responses API request field
    (``ResponsesAPIOptionalRequestParams.metadata``) that
    ``_build_base_call_kwargs`` forwards verbatim from the client (e.g. a
    Codex session tag). ``litellm.utils.function_setup`` only copies our
    ``user_api_key``-carrying ``litellm_metadata`` into
    ``litellm_params["metadata"]`` when ``kwargs["metadata"]`` is falsy, so a
    truthy client metadata dict wins and drops ``user_api_key`` from
    ``litellm_params["metadata"]``. The ACTIVE rate limiter,
    ``_PROXY_MaxParallelRequestsHandler_v3``
    (litellm/proxy/hooks/parallel_request_limiter_v3.py:4157), reads
    ``standard_logging_object["metadata"]["user_api_key_hash"]`` to attribute
    a turn's tokens to the right TPM/RPM counter, and that field is NOT
    protected by the ``get_litellm_metadata_from_kwargs`` merge -- so the
    collision breaks cross-connection TPM/RPM enforcement for non-native
    providers (Fireworks, Vertex, Bedrock-Mantle, etc., anything routed
    through ``ManagedResponsesWebSocketHandler``). See
    ``TestWebSocketRateLimitEnforcementMechanism`` for the file:line
    mechanism trace against the real registered limiter; these tests check
    the underlying merge mechanism directly and deterministically.

    Native providers (Azure, OpenAI) do not go through
    ``ManagedResponsesWebSocketHandler`` at all -- see
    ``ResponsesWebSocketStreaming`` and the tracked native-path usage
    accounting gap (a separate change, not covered by this fix).
    """

    @staticmethod
    def _install_streaming_shim(monkeypatch):
        """Wrap the REAL litellm.aresponses so the real @client decorator,
        the real litellm.utils.function_setup metadata merge, and the real
        registered success callbacks all run -- only the stream/non-stream
        boundary is faked (mock_response does not support stream=True)."""
        import litellm

        real_aresponses = litellm.aresponses

        async def streaming_shim(*args, **kwargs):
            kwargs.pop("stream", None)
            response = await real_aresponses(*args, **kwargs)

            async def _one_chunk():
                yield response

            return _one_chunk()

        monkeypatch.setattr(litellm, "aresponses", streaming_shim)

    @pytest.mark.asyncio
    async def test_ws_turn_keeps_user_api_key_in_litellm_params_metadata_despite_client_metadata(self, monkeypatch):
        """litellm_params["metadata"] (read directly by standard_logging_object
        construction, and hence by the real rate limiter -- see
        TestWebSocketRateLimitClosesAcrossConnections) must keep user_api_key
        even when the client's own frame carries a colliding metadata object."""
        import asyncio
        from unittest.mock import AsyncMock

        import litellm
        from litellm.integrations.custom_logger import CustomLogger
        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        class MetadataCapture(CustomLogger):
            def __init__(self):
                self.metadata: dict | None = None
                self.event = asyncio.Event()

            async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
                self.metadata = kwargs.get("litellm_params", {}).get("metadata") or {}
                self.event.set()

        spy = MetadataCapture()
        # monkeypatch.setattr auto-reverts at teardown (and the autouse
        # isolate_litellm_state fixture in tests/test_litellm/conftest.py
        # would restore it anyway) -- no manual restore needed either way.
        monkeypatch.setattr(litellm, "callbacks", [spy])
        self._install_streaming_shim(monkeypatch)

        mock_websocket = MagicMock()
        mock_websocket.send_text = AsyncMock()

        handler = ManagedResponsesWebSocketHandler(
            websocket=mock_websocket,
            model="gpt-4o",
            logging_obj=Logging(
                model="gpt-4o",
                messages=[],
                stream=True,
                call_type="aresponses",
                start_time=0,
                litellm_call_id="test-ws-metadata-id",
                function_id="test-func",
            ),
            litellm_metadata={
                "user_api_key": "sk-hashed-test-key",
                "user_api_key_team_id": "team-1",
            },
            mock_response="hello",
        )

        # The client's OWN metadata object -- e.g. a Codex session tag --
        # riding alongside the proxy's litellm_metadata on the same frame.
        frame = json.dumps(
            {
                "type": "response.create",
                "model": "gpt-4o",
                "input": "hi",
                "metadata": {"codex_session_id": "abc123"},
            }
        )

        await handler._process_response_create(frame)
        await asyncio.wait_for(spy.event.wait(), timeout=5.0)
        assert spy.metadata is not None, "success callback was never invoked"
        assert spy.metadata.get("user_api_key") == "sk-hashed-test-key", (
            f"user_api_key missing from litellm_params['metadata']: {spy.metadata!r} "
            "-- the rate limiter's standard_logging_object read depends on this"
        )

    @pytest.mark.asyncio
    async def test_ws_turn_preserves_client_metadata_as_requester_metadata(self, monkeypatch):
        """The client's own metadata must not be discarded -- only nested
        under litellm_metadata so it no longer collides with the proxy's."""
        import asyncio
=======
class TestNativeWebSocketPerTurnCostAccounting:
    """Regression tests for rayward-internal/llm-gateway-infra#657's native-provider
    leg (Azure, OpenAI -- anything where responses_api_provider_config.supports_native_websocket()
    is True, so llm_http_handler.py routes to ResponsesWebSocketStreaming instead of
    ManagedResponsesWebSocketHandler).

    Measured in production 2026-08-28 (real DB query against LiteLLM_SpendLogs,
    2026-08-28 04:00-04:45 UTC window): 11 `_aresponses_websocket` rows, one per
    WebSocket session (11 sessions, 200+ response.completed frames), every row
    `spend=0`, `total_tokens=prompt_tokens=completion_tokens=0`,
    `metadata.cost_breakdown` all-zero, `status='success'` (not skipped --
    genuinely written as zero). Root cause: `_log_messages()` used to dispatch
    `self.messages` -- a raw Python list of stored WS event dicts -- as the
    "response" to `self.logging_obj.dispatch_success_handlers`, once, in
    `backend_to_client`'s `finally:`, i.e. only at connection close.
    `_get_assembled_streaming_response` (litellm_logging.py:3670-3693) has no
    branch for a bare list and returns None, so response_cost/usage always
    computed as 0/0 -- confirmed by driving the exact pre-fix code with a real
    registered spy CustomLogger.

    Fix: as each response.completed/failed/incomplete frame is forwarded (inside
    backend_to_client's loop, not at close), build the correctly-typed
    ResponseCompletedEvent/ResponseFailedEvent/ResponseIncompleteEvent from it
    and dispatch it on a FRESH per-turn LiteLLMLoggingObj. This lands on
    _get_assembled_streaming_response's EXISTING correct branch for those three
    types (litellm_logging.py:3672-3673) -- no logging-layer change needed, only
    the caller needs to build the right typed object and hand it a fresh
    logging_obj.

    Fresh-per-turn is not optional: reusing the connection-level logging_obj
    across turns was verified (by running, not reading) to silently drop every
    turn after the first once stream=True, because dispatch_success_handlers's
    own dedup guard (model_call_details["has_dispatched_final_stream_success"])
    is keyed on the logging_obj instance, not the call. See
    test_three_turn_session_costs_exactly_three_turns below.

    Row cardinality: this deliberately changes one row per SESSION into one row
    per TURN. Verified safe: SpendLogs' primary key (request_id) prefers the
    response's own `id` (get_spend_logs_id, spend_tracking_utils.py:190-201)
    over litellm_call_id, and every real provider response.completed event
    carries a genuinely distinct response id per turn -- and this fix ALSO
    generates a fresh litellm_call_id per turn as a second independent source
    of uniqueness. See test_distinct_turns_get_distinct_request_ids.
    """

    @staticmethod
    def _register_spy_and_fanout(spy):
        """Register *spy* the way production startup does (litellm.callbacks),
        and perform the SAME litellm.callbacks -> litellm._async_success_callback
        fan-out litellm.utils.function_setup performs on its first call per
        process -- which, in the real endpoint, already happened once for the
        connection at Phase 1 (common_processing_pre_call_logic) before any
        per-turn dispatch. Restoration of litellm.callbacks (and
        _async_success_callback) is handled by the autouse isolate_litellm_state
        fixture (tests/test_litellm/conftest.py) -- do not add a manual restore
        here or at call sites."""
        import uuid as _uuid
        from datetime import datetime as _dt

        import litellm as _litellm
        from litellm.utils import Rules as _Rules

        _litellm.callbacks = [spy]
        _litellm.utils.function_setup(
            original_function="_aresponses_websocket",
            rules_obj=_Rules(),
            start_time=_dt.now(),
            model="gpt-5.6-sol",
            litellm_call_id=str(_uuid.uuid4()),
        )

    @staticmethod
    def _completed_event(resp_id, total_tokens, model="gpt-5.6-sol", event_type="response.completed"):
        return json.dumps(
            {
                "type": event_type,
                "response": {
                    "id": resp_id,
                    "object": "response",
                    "created_at": 0,
                    "status": "completed" if event_type == "response.completed" else "incomplete",
                    "model": model,
                    "output": [
                        {
                            "type": "message",
                            "id": f"msg_{resp_id}",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "hi", "annotations": []}],
                        }
                    ],
                    "usage": {
                        "input_tokens": total_tokens // 2,
                        "output_tokens": total_tokens - total_tokens // 2,
                        "total_tokens": total_tokens,
                    },
                },
            }
        )

    @pytest.mark.asyncio
    async def test_three_turn_session_costs_exactly_three_turns(self, monkeypatch):
        """Fails before the fix (0 costed events -- $0 spend, matching production);
        must show exactly 3, not 4 (a leftover close-time dispatch) or 6 (double
        counting response.created alongside response.completed)."""
        from unittest.mock import AsyncMock

        import websockets.exceptions  # noqa: F401  (lazy submodule must be importable)

        import litellm
        from litellm.integrations.custom_logger import CustomLogger
        from litellm.proxy._types import UserAPIKeyAuth

        class Spy(CustomLogger):
            def __init__(self):
                self.events = []

            async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
                sl = kwargs.get("standard_logging_object") or {}
                self.events.append((sl.get("response_cost"), sl.get("total_tokens")))

        spy = Spy()
        self._register_spy_and_fanout(spy)

        websocket = MagicMock()
        websocket.send_text = AsyncMock()
        backend_ws = MagicMock()
        backend_ws.recv = AsyncMock(
            side_effect=[
                json.dumps({"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}}),
                self._completed_event("resp_1", 100),
                json.dumps({"type": "response.created", "response": {"id": "resp_2", "status": "in_progress"}}),
                self._completed_event("resp_2", 200),
                json.dumps({"type": "response.created", "response": {"id": "resp_3", "status": "in_progress"}}),
                self._completed_event("resp_3", 300),
                Exception("stop"),
            ]
        )

        user_api_key_dict = UserAPIKeyAuth(api_key="sk-native-3turn-key")
        handler = _make_streaming(
            websocket=websocket,
            backend_ws=backend_ws,
            user_api_key_dict=user_api_key_dict,
            request_data={
                "litellm_metadata": {
                    "user_api_key": user_api_key_dict.api_key,
                    "user_api_key_hash": user_api_key_dict.api_key,
                }
            },
            authorized_model="gpt-5.6-sol",
        )

        await handler.backend_to_client()
        await asyncio.sleep(0.3)

        assert len(spy.events) == 3, f"expected exactly 3 costed turns, got {len(spy.events)}: {spy.events}"
        costs, tokens = zip(*spy.events)
        assert list(tokens) == [100, 200, 300], f"per-turn usage must be distinct and non-cumulative: {tokens}"
        assert all(c > 0 for c in costs), f"every turn must be costed above $0: {costs}"
        assert len(set(costs)) == 3, f"three distinct turns must produce three distinct costs: {costs}"

    @pytest.mark.asyncio
    async def test_response_failed_and_incomplete_are_also_costed(self, monkeypatch):
        """A failed or incomplete turn still consumed real tokens and must still
        be attributed -- not just response.completed."""
        from unittest.mock import AsyncMock

        import websockets.exceptions  # noqa: F401

        import litellm
        from litellm.integrations.custom_logger import CustomLogger
        from litellm.proxy._types import UserAPIKeyAuth

        class Spy(CustomLogger):
            def __init__(self):
                self.events = []

            async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
                sl = kwargs.get("standard_logging_object") or {}
                self.events.append((sl.get("response_cost"), sl.get("total_tokens")))

        spy = Spy()
        self._register_spy_and_fanout(spy)

        websocket = MagicMock()
        websocket.send_text = AsyncMock()
        backend_ws = MagicMock()
        backend_ws.recv = AsyncMock(
            side_effect=[
                self._completed_event("resp_failed_1", 50, event_type="response.failed"),
                self._completed_event("resp_incomplete_1", 75, event_type="response.incomplete"),
                Exception("stop"),
            ]
        )

        user_api_key_dict = UserAPIKeyAuth(api_key="sk-native-failed-key")
        handler = _make_streaming(
            websocket=websocket,
            backend_ws=backend_ws,
            user_api_key_dict=user_api_key_dict,
            request_data={
                "litellm_metadata": {
                    "user_api_key": user_api_key_dict.api_key,
                    "user_api_key_hash": user_api_key_dict.api_key,
                }
            },
            authorized_model="gpt-5.6-sol",
        )

        await handler.backend_to_client()
        await asyncio.sleep(0.3)

        assert len(spy.events) == 2, f"failed AND incomplete turns must both be costed: {spy.events}"
        tokens = sorted(t for _, t in spy.events)
        assert tokens == [50, 75]

    @pytest.mark.asyncio
    async def test_turn_cost_attributes_to_the_correct_key(self, monkeypatch):
        """Verify key identity reaches the callback on the native path too --
        this class builds request_data differently from the managed path
        (llm_http_handler.py:6519-6521, litellm_metadata only if truthy), so it
        is not safe to assume the managed-path fix's verification carries over."""
        from unittest.mock import AsyncMock

        import websockets.exceptions  # noqa: F401

        import litellm
        from litellm.integrations.custom_logger import CustomLogger
        from litellm.proxy._types import UserAPIKeyAuth

        class Spy(CustomLogger):
            def __init__(self):
                self.metadata = None

            async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
                sl = kwargs.get("standard_logging_object") or {}
                self.metadata = sl.get("metadata")

        spy = Spy()
        self._register_spy_and_fanout(spy)

        websocket = MagicMock()
        websocket.send_text = AsyncMock()
        backend_ws = MagicMock()
        backend_ws.recv = AsyncMock(side_effect=[self._completed_event("resp_key_1", 42), Exception("stop")])

        user_api_key_dict = UserAPIKeyAuth(api_key="sk-native-attribution-key")
        handler = _make_streaming(
            websocket=websocket,
            backend_ws=backend_ws,
            user_api_key_dict=user_api_key_dict,
            request_data={
                "litellm_metadata": {
                    "user_api_key": user_api_key_dict.api_key,
                    "user_api_key_hash": user_api_key_dict.api_key,
                    "requester_ip_address": "203.0.113.7",
                }
            },
            authorized_model="gpt-5.6-sol",
        )

        await handler.backend_to_client()
        await asyncio.sleep(0.3)

        assert spy.metadata is not None, "success callback was never invoked"
        assert spy.metadata.get("user_api_key_hash") == user_api_key_dict.api_key
        assert spy.metadata.get("requester_ip_address") == "203.0.113.7", (
            "requester_ip_address must thread through from the connection-level "
            "litellm_metadata (add_litellm_data_to_request already sets it there); "
            "the old once-at-close raw-list dispatch could not populate ANY "
            "metadata field, including this one"
        )

    @pytest.mark.asyncio
    async def test_distinct_turns_get_distinct_request_ids(self, monkeypatch):
        """SpendLogs' primary key prefers response_obj["id"] over litellm_call_id
        (get_spend_logs_id, spend_tracking_utils.py:190-201). Verify both are
        independently distinct per turn -- a real constraint-violation risk if
        either were reused across turns on the same connection."""
        from unittest.mock import AsyncMock

        import websockets.exceptions  # noqa: F401

        import litellm
        from litellm.integrations.custom_logger import CustomLogger
        from litellm.proxy._types import UserAPIKeyAuth

        class Spy(CustomLogger):
            def __init__(self):
                self.ids = []

            async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
                self.ids.append((getattr(response_obj, "id", None), kwargs.get("litellm_call_id")))

        spy = Spy()
        self._register_spy_and_fanout(spy)

        websocket = MagicMock()
        websocket.send_text = AsyncMock()
        backend_ws = MagicMock()
        backend_ws.recv = AsyncMock(
            side_effect=[
                self._completed_event("resp_a", 10),
                self._completed_event("resp_b", 20),
                Exception("stop"),
            ]
        )

        user_api_key_dict = UserAPIKeyAuth(api_key="sk-native-ids-key")
        handler = _make_streaming(
            websocket=websocket,
            backend_ws=backend_ws,
            user_api_key_dict=user_api_key_dict,
            request_data={
                "litellm_metadata": {
                    "user_api_key": user_api_key_dict.api_key,
                    "user_api_key_hash": user_api_key_dict.api_key,
                }
            },
            authorized_model="gpt-5.6-sol",
        )

        await handler.backend_to_client()
        await asyncio.sleep(0.3)

        assert len(spy.ids) == 2
        response_ids, call_ids = zip(*spy.ids)
        assert len(set(response_ids)) == 2, f"response ids must be distinct per turn: {response_ids}"
        assert len(set(call_ids)) == 2, f"litellm_call_id must be distinct per turn: {call_ids}"

    @pytest.mark.asyncio
    async def test_non_terminal_events_are_not_costed(self, monkeypatch):
        """response.created and delta events carry no final usage and must
        never trigger a dispatch."""
        from unittest.mock import AsyncMock

        import websockets.exceptions  # noqa: F401

        import litellm
        from litellm.integrations.custom_logger import CustomLogger
        from litellm.proxy._types import UserAPIKeyAuth

        class Spy(CustomLogger):
            def __init__(self):
                self.call_count = 0

            async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
                self.call_count += 1

        spy = Spy()
        self._register_spy_and_fanout(spy)

        websocket = MagicMock()
        websocket.send_text = AsyncMock()
        backend_ws = MagicMock()
        backend_ws.recv = AsyncMock(
            side_effect=[
                json.dumps({"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}}),
                json.dumps({"type": "response.output_text.delta", "delta": "hi"}),
                Exception("stop"),
            ]
        )

        user_api_key_dict = UserAPIKeyAuth(api_key="sk-native-noncost-key")
        handler = _make_streaming(
            websocket=websocket,
            backend_ws=backend_ws,
            user_api_key_dict=user_api_key_dict,
            request_data={
                "litellm_metadata": {
                    "user_api_key": user_api_key_dict.api_key,
                    "user_api_key_hash": user_api_key_dict.api_key,
                }
            },
            authorized_model="gpt-5.6-sol",
        )

        await handler.backend_to_client()
        await asyncio.sleep(0.3)

        assert spy.call_count == 0

    @pytest.mark.asyncio
    async def test_turn_cost_unaffected_by_output_pii_masking(self, monkeypatch):
        """apply_to_output masking must not corrupt or block cost computation --
        the masked (client-forwarded) text is what gets read, but usage/cost
        come from the response's own usage object, never the text content."""
        from unittest.mock import AsyncMock

        import websockets.exceptions  # noqa: F401

        import litellm
        from litellm.integrations.custom_logger import CustomLogger
        from litellm.proxy._types import UserAPIKeyAuth

        class Spy(CustomLogger):
            def __init__(self):
                self.events = []

            async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
                sl = kwargs.get("standard_logging_object") or {}
                self.events.append((sl.get("response_cost"), sl.get("total_tokens")))

        spy = Spy()
        self._register_spy_and_fanout(spy)

        guardrail = _FakeWSGuardrail()
        websocket = MagicMock()
        websocket.send_text = AsyncMock()
        backend_ws = MagicMock()
        backend_ws.recv = AsyncMock(
            side_effect=[
                json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "resp_pii_1",
                            "object": "response",
                            "created_at": 0,
                            "status": "completed",
                            "model": "gpt-5.6-sol",
                            "output": [
                                {
                                    "type": "message",
                                    "id": "msg_pii_1",
                                    "status": "completed",
                                    "role": "assistant",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": "contact alice@example.com please",
                                            "annotations": [],
                                        }
                                    ],
                                },
                            ],
                            "usage": {"input_tokens": 50, "output_tokens": 60, "total_tokens": 110},
                        },
                    }
                ),
                Exception("stop"),
            ]
        )

        user_api_key_dict = UserAPIKeyAuth(api_key="sk-native-pii-key")
        handler = _make_streaming(
            websocket=websocket,
            backend_ws=backend_ws,
            user_api_key_dict=user_api_key_dict,
            request_data={
                "litellm_metadata": {
                    "user_api_key": user_api_key_dict.api_key,
                    "user_api_key_hash": user_api_key_dict.api_key,
                }
            },
            output_guardrail_callbacks=[guardrail],
            authorized_model="gpt-5.6-sol",
        )

        await handler.backend_to_client()
        await asyncio.sleep(0.3)

        websocket.send_text.assert_awaited_once()
        forwarded = json.loads(websocket.send_text.await_args[0][0])
        forwarded_text = forwarded["response"]["output"][0]["content"][0]["text"]
        assert "alice@example.com" not in forwarded_text, "client must receive the masked text"
        assert forwarded_text == "contact <EMAIL_ADDRESS_1> please"

        assert len(spy.events) == 1, "the masked turn must still be costed exactly once"
        assert spy.events[0][1] == 110, "usage must come from the response's usage object, unaffected by masking"
        assert spy.events[0][0] > 0

    @pytest.mark.asyncio
    async def test_drain_waits_for_a_turn_dispatched_just_before_teardown(self):
        """Regression for the fire-and-forget task-ownership gap: the client
        disconnects right after its final response.completed (the realistic
        trigger is Cloud Run scaling the instance down, or a deploy, killing
        an in-flight task -- not the event loop itself stopping, which a
        pending asyncio task usually survives in production). Teardown must
        drain self._pending_cost_tasks with a bounded wait, or the LAST
        (often largest) turn of every session is the one most likely to lose
        its billing.

        Tests _drain_pending_cost_tasks directly rather than the full
        bidirectional_forward orchestration: bidirectional_forward's
        PRE-EXISTING (not introduced by this fix) "cancel forward_task if not
        done" logic races against backend_to_client's own progress in a mock
        setup with no real ordering guarantees, which would make an
        integration-level version of this test flaky for reasons unrelated to
        the drain itself. Deliberately asserts with NO extra sleep/await
        after the drain call -- a test that slept afterwards would pass
        whether or not the drain exists, which is exactly why the original
        3-turn test (which does sleep) could not have caught this."""
        import litellm
        from litellm.integrations.custom_logger import CustomLogger
        from litellm.proxy._types import UserAPIKeyAuth

        class SlowSpy(CustomLogger):
            """Simulates a success handler whose write has not landed yet
            when the connection tears down -- e.g. an in-flight DB write."""

            def __init__(self):
                self.events = []

            async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
                await asyncio.sleep(0.05)
                sl = kwargs.get("standard_logging_object") or {}
                self.events.append((sl.get("response_cost"), sl.get("total_tokens")))

        spy = SlowSpy()
        self._register_spy_and_fanout(spy)

        user_api_key_dict = UserAPIKeyAuth(api_key="sk-native-teardown-key")
        handler = _make_streaming(
            websocket=MagicMock(),
            backend_ws=MagicMock(),
            user_api_key_dict=user_api_key_dict,
            request_data={
                "litellm_metadata": {
                    "user_api_key": user_api_key_dict.api_key,
                    "user_api_key_hash": user_api_key_dict.api_key,
                }
            },
            authorized_model="gpt-5.6-sol",
        )

        # The connection's LAST turn, dispatched an instant before the
        # (simulated) client disconnect / instance teardown.
        handler._dispatch_turn_cost(self._completed_event("resp_final_turn", 999))
        assert len(handler._pending_cost_tasks) == 1

        # NO sleep here -- the assertions below must be satisfied by the
        # drain itself, not by luck or a test delay.
        await handler._drain_pending_cost_tasks()

        assert len(spy.events) == 1, (
            f"the final turn's cost dispatch must be drained, not lost at teardown: {spy.events}"
        )
        assert spy.events[0][1] == 999
        assert spy.events[0][0] > 0
        assert len(handler._pending_cost_tasks) == 0, "drained tasks must be discarded from the registry"

    @pytest.mark.asyncio
    async def test_bidirectional_forward_drains_pending_cost_tasks_before_returning(self):
        """Wiring check: bidirectional_forward's finally: must actually call
        the drain (not just exist as a dead method) before it returns."""
        from unittest.mock import AsyncMock

        websocket = MagicMock()
        backend_ws = MagicMock()
        backend_ws.recv = AsyncMock(side_effect=Exception("stop"))
        backend_ws.close = AsyncMock()
        websocket.receive_text = AsyncMock(side_effect=Exception("client disconnected"))

        handler = _make_streaming(websocket=websocket, backend_ws=backend_ws)
        handler._drain_pending_cost_tasks = AsyncMock()

        await handler.bidirectional_forward()

        handler._drain_pending_cost_tasks.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pending_cost_tasks_hold_strong_references(self):
        """_dispatch_turn_cost must retain the created task on the instance,
        not just create_task() and let it go -- asyncio keeps only a weak
        reference to a bare create_task() result, so an unretained task can
        be garbage-collected mid-run."""
>>>>>>> origin/litellm_internal_staging
        from unittest.mock import AsyncMock

        import litellm
        from litellm.integrations.custom_logger import CustomLogger
<<<<<<< HEAD
        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        class MetadataCapture(CustomLogger):
            def __init__(self):
                self.metadata: dict | None = None
                self.event = asyncio.Event()

            async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
                self.metadata = kwargs.get("litellm_params", {}).get("metadata") or {}
                self.event.set()

        spy = MetadataCapture()
        # monkeypatch.setattr auto-reverts at teardown (and the autouse
        # isolate_litellm_state fixture in tests/test_litellm/conftest.py
        # would restore it anyway) -- no manual restore needed either way.
        monkeypatch.setattr(litellm, "callbacks", [spy])
        self._install_streaming_shim(monkeypatch)

        mock_websocket = MagicMock()
        mock_websocket.send_text = AsyncMock()

        handler = ManagedResponsesWebSocketHandler(
            websocket=mock_websocket,
            model="gpt-4o",
            logging_obj=Logging(
                model="gpt-4o",
                messages=[],
                stream=True,
                call_type="aresponses",
                start_time=0,
                litellm_call_id="test-ws-metadata-id-2",
                function_id="test-func",
            ),
            litellm_metadata={"user_api_key": "sk-hashed-test-key"},
            mock_response="hello",
        )

        frame = json.dumps(
            {
                "type": "response.create",
                "model": "gpt-4o",
                "input": "hi",
                "metadata": {"codex_session_id": "abc123"},
            }
        )

        await handler._process_response_create(frame)
        await asyncio.wait_for(spy.event.wait(), timeout=5.0)
        assert spy.metadata is not None
        assert spy.metadata.get("requester_metadata", {}).get("codex_session_id") == "abc123", (
            "client-supplied metadata must survive, nested under requester_metadata"
        )
=======
        from litellm.proxy._types import UserAPIKeyAuth

        class Spy(CustomLogger):
            async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
                pass

        spy = Spy()
        self._register_spy_and_fanout(spy)

        user_api_key_dict = UserAPIKeyAuth(api_key="sk-native-strongref-key")
        handler = _make_streaming(
            websocket=MagicMock(),
            backend_ws=MagicMock(),
            user_api_key_dict=user_api_key_dict,
            request_data={
                "litellm_metadata": {
                    "user_api_key": user_api_key_dict.api_key,
                    "user_api_key_hash": user_api_key_dict.api_key,
                }
            },
            authorized_model="gpt-5.6-sol",
        )

        assert handler._pending_cost_tasks == set()
        handler._dispatch_turn_cost(self._completed_event("resp_ref_1", 10))
        assert len(handler._pending_cost_tasks) == 1, (
            "the created task must be retained in self._pending_cost_tasks immediately"
        )
        await asyncio.sleep(0.2)
        assert handler._pending_cost_tasks == set(), "a completed task must be discarded via add_done_callback"
>>>>>>> origin/litellm_internal_staging
