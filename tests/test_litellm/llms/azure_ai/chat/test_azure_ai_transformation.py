import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from litellm.llms.azure_ai.azure_model_router.transformation import (
    AzureModelRouterConfig,
)
from litellm.llms.azure_ai.chat.transformation import AzureAIStudioConfig


@pytest.mark.asyncio
async def test_get_openai_compatible_provider_info():
    """
    Test that Azure AI requests are formatted correctly with the proper endpoint and parameters
    for both synchronous and asynchronous calls
    """
    config = AzureAIStudioConfig()

    (
        api_base,
        dynamic_api_key,
        custom_llm_provider,
    ) = config._get_openai_compatible_provider_info(
        model="azure_ai/gpt-4o-mini",
        api_base="https://my-base",
        api_key="my-key",
        custom_llm_provider="azure_ai",
    )

    assert custom_llm_provider == "azure"


def test_azure_ai_validate_environment():
    config = AzureAIStudioConfig()
    headers = config.validate_environment(
        headers={},
        model="azure_ai/gpt-4o-mini",
        messages=[],
        optional_params={},
        litellm_params={},
    )
    assert headers["Content-Type"] == "application/json"


def test_azure_ai_validate_environment_with_api_key():
    """
    Test that when api_key is provided, it is set in the api-key header
    for Azure Foundry endpoints (.services.ai.azure.com).
    """
    config = AzureAIStudioConfig()
    headers = config.validate_environment(
        headers={},
        model="Kimi-K2.5",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="test-api-key",
        api_base="https://my-endpoint.services.ai.azure.com",
    )
    assert headers["api-key"] == "test-api-key"
    assert headers["Content-Type"] == "application/json"


def test_azure_ai_validate_environment_with_azure_ad_token():
    """
    Test that when no api_key is provided but Azure AD credentials are available,
    the Authorization header is set with a Bearer token.

    Regression test for https://github.com/BerriAI/litellm/issues/20759
    """
    import litellm

    config = AzureAIStudioConfig()
    with (
        patch(
            "litellm.llms.azure.common_utils.get_azure_ad_token",
            return_value="fake-azure-ad-token",
        ),
        patch(
            "litellm.llms.azure.common_utils.get_secret_str",
            return_value=None,
        ),
        patch.object(litellm, "api_key", None),
        patch.object(litellm, "azure_key", None),
    ):
        headers = config.validate_environment(
            headers={},
            model="Kimi-K2.5",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base="https://my-endpoint.services.ai.azure.com",
        )
    assert headers.get("Authorization") == "Bearer fake-azure-ad-token"
    assert "api-key" not in headers
    assert headers["Content-Type"] == "application/json"


def test_azure_ai_grok_stop_parameter_handling():
    """
    Test that Grok models properly handle stop parameter filtering in Azure AI Studio.
    """
    config = AzureAIStudioConfig()

    # Test Grok model detection
    assert config._supports_stop_reason("grok-4-fast") is False
    assert config._supports_stop_reason("grok-4.3") is False
    assert config._supports_stop_reason("grok-4") is False
    assert config._supports_stop_reason("grok-3-mini") is False
    assert config._supports_stop_reason("grok-code-fast") is False
    assert config._supports_stop_reason("gpt-4") is True

    # Test supported parameters for Grok models
    for model in ("grok-4-fast", "grok-4.3"):
        grok_params = config.get_supported_openai_params(model)
        assert (
            "stop" not in grok_params
        ), "Grok models should not support stop parameter"

    # Test supported parameters for non-Grok models
    gpt_params = config.get_supported_openai_params("gpt-4")
    assert "stop" in gpt_params, "GPT models should support stop parameter"


def test_azure_model_router_response_shows_actual_model():
    """
    Test that Azure Model Router returns the actual model used in the response,
    not the router model.

    According to the documentation, when using Azure Model Router, the response
    should show the actual model that handled the request (e.g., gpt-5-nano-2025-08-07)
    rather than the router model (e.g., model-router).

    Regression test for: Azure Model Router should show actual model in response
    """
    from httpx import Response

    from litellm.llms.base_llm.chat.transformation import LiteLLMLoggingObj
    from litellm.types.utils import ModelResponse

    config = AzureModelRouterConfig()

    # Mock raw response from Azure that includes the actual model used
    raw_response_json = {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-5-nano-2025-08-07",  # Actual model used by the router
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }

    # Create mock Response object
    mock_response = MagicMock(spec=Response)
    mock_response.json.return_value = raw_response_json
    mock_response.text = json.dumps(raw_response_json)
    mock_response.headers = {}

    # Create ModelResponse object
    model_response = ModelResponse()

    # Create mock logging object with required methods
    logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    logging_obj.post_call = MagicMock()
    logging_obj.model_call_details = {}

    # Call transform_response with router model
    result = config.transform_response(
        model="model-router",  # This is the router model (without prefix)
        raw_response=mock_response,
        model_response=model_response,
        logging_obj=logging_obj,
        request_data={},
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={"model": "azure_ai/model-router"},  # Original request model
        encoding=None,
        api_key="test-key",
        json_mode=False,
    )

    # Verify that the response contains the actual model used, not the router model
    assert result.model == "azure_ai/gpt-5-nano-2025-08-07", (
        f"Expected model to be 'azure_ai/gpt-5-nano-2025-08-07' (actual model used), "
        f"but got '{result.model}'"
    )


def test_drop_tool_level_extra_fields_strips_copilot_mcp_server_name():
    """
    Regression test: Azure AI returns 400 when tools contain copilot_mcp_server_name.
    LiteLLM should strip the field and retry automatically.
    """
    import httpx

    config = AzureAIStudioConfig()

    error_text = json.dumps(
        {
            "error": {
                "message": "2 request validation errors: Extra inputs are not permitted, field: 'tools[0].copilot_mcp_server_name', value: 'github-mcp-server'; Extra inputs are not permitted, field: 'tools[1].copilot_mcp_server_name', value: 'ide'"
            }
        }
    )
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.text = error_text
    mock_response.json.return_value = json.loads(error_text)
    mock_response.status_code = 400
    e = httpx.HTTPStatusError(
        message="400", request=MagicMock(), response=mock_response
    )

    assert config._error_has_tool_level_extra_fields(error_text) is True
    assert (
        config.should_retry_llm_api_inside_llm_translation_on_http_error(e, {}) is True
    )

    request_data = {
        "model": "FW-Kimi-K2.6",
        "messages": [{"role": "user", "content": "Say hi."}],
        "tools": [
            {
                "type": "function",
                "copilot_mcp_server_name": "github-mcp-server",
                "function": {
                    "name": "github_search_code",
                    "description": "Search code",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "copilot_mcp_server_name": "ide",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
    }

    result = config.transform_request_on_unprocessable_entity_error(e, request_data)

    for tool in result["tools"]:
        assert "copilot_mcp_server_name" not in tool
    assert result["tools"][0]["type"] == "function"
    assert result["tools"][1]["function"]["name"] == "read_file"


def test_azure_ai_reasoning_effort_declared_via_supports_reasoning():
    """
    reasoning_effort was being silently stripped by litellm's own drop_params
    for every azure_ai-routed model, including ones litellm's own cost map
    already marks supports_reasoning=True (azure_ai/kimi-k2.6,
    azure_ai/deepseek-v4-pro) -- get_supported_openai_params never consulted
    that flag at all, so map_openai_params (a pure allow-list passthrough,
    see _map_openai_params in llms/openai/openai.py) never let the caller's
    value through to the wire. Verified live against a production gateway
    fronting both LiteLLM and Bifrost: Bifrost forwards reasoning_effort to
    these same Azure AI Foundry deployments and the upstream endpoints
    genuinely honor it (rayward-internal/llm-gateway-infra#290).

    grok models resolve through the "xai" provider's cost-map entries, not
    "azure_ai" -- those are better-maintained (xai/grok-4.3 -> True vs.
    azure_ai/grok-4.3 -> False in the current cost map) and this file already
    delegates grok's stop-token capability to XAIChatConfig for the same
    reason (see _supports_stop_reason above); reasoning follows the same
    delegation.
    """
    config = AzureAIStudioConfig()

    # Cost map already says these support reasoning -- the gate must actually
    # look, not just always omit the param.
    assert "reasoning_effort" in config.get_supported_openai_params("kimi-k2.6")
    assert "reasoning_effort" in config.get_supported_openai_params("deepseek-v4-pro")

    # grok resolves via the xai/ cost-map entries (delegated, like stop-token
    # support above), not the azure_ai/ ones.
    assert "reasoning_effort" in config.get_supported_openai_params("grok-4.3")

    # A model with no reasoning capability anywhere in the cost map must not
    # get the param -- this isn't "always allow", it's "declare what's true".
    assert "reasoning_effort" not in config.get_supported_openai_params("gpt-4")


def test_azure_ai_reasoning_effort_flows_through_as_a_plain_value():
    """map_openai_params for azure_ai is inherited, unmodified, from
    OpenAIConfig._map_openai_params -- a pure allow-list passthrough with no
    value translation (`if param in supported_openai_params: optional_params[param] = value`).
    Once declared supported, the value must reach optional_params UNCHANGED --
    there is deliberately no new mapping logic for this, since the upstream
    Azure AI Foundry endpoints already accept the literal OpenAI-style string
    (confirmed live via Bifrost, which forwards it as-is)."""
    config = AzureAIStudioConfig()
    optional_params = config.map_openai_params(
        non_default_params={"reasoning_effort": "high"},
        optional_params={},
        model="kimi-k2.6",
        drop_params=False,
    )
    assert optional_params == {"reasoning_effort": "high"}


def test_azure_ai_reasoning_check_is_additive_not_exclusive_for_grok():
    """codex review, PR #150: the xai/ delegation was written as EXCLUSIVE
    ("if grok in model, ONLY check xai/") rather than additive, which
    regressed a real deployment: azure_ai/global/grok-3-mini has
    supports_reasoning=true in litellm's own cost map with no corresponding
    xai/global/grok-3-mini entry to fall back to, so forcing the xai/ lookup
    for any "grok" in the model name lost data the azure_ai/ entry already
    had. The check must consult azure_ai/ first and only reach for xai/ as an
    additional (not replacement) signal.
    """
    config = AzureAIStudioConfig()

    # Azure's own cost-map entry already knows this one -- must not be
    # thrown away by forcing an xai/ lookup that doesn't exist for it.
    assert "reasoning_effort" in config.get_supported_openai_params("global/grok-3-mini")

    # The original working case must still work: azure_ai/grok-4.3 has no
    # entry, xai/grok-4.3 does.
    assert "reasoning_effort" in config.get_supported_openai_params("grok-4.3")
