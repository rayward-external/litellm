import os
import pytest
from unittest.mock import patch
from litellm.llms.baseten.chat import BasetenConfig


class TestBasetenRouting:
    """Test Baseten routing logic"""

    def test_routing_logic(self):
        """Test routing between Model API and dedicated deployments"""
        config = BasetenConfig()

        # Dedicated deployment (8-character alphanumeric)
        assert (
            config.get_api_base_for_model("abcd1234")
            == "https://model-abcd1234.api.baseten.co/environments/production/sync/v1"
        )

        # Model API (non-8-character)
        assert (
            config.get_api_base_for_model("openai/gpt-oss-120b")
            == "https://inference.baseten.co/v1"
        )


class TestBasetenModelAPI:
    """Test Baseten Model API inference"""

    @patch.dict(os.environ, {"BASETEN_API_KEY": "test-key"})
    def test_model_api_inference(self):
        """Test Model API inference with basic parameters"""
        config = BasetenConfig()

        # Test parameter mapping
        non_default_params = {"max_tokens": 100, "temperature": 0.7, "top_p": 0.9}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="openai/gpt-oss-120b",
            drop_params=False,
        )

        assert result["max_tokens"] == 100
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9

        # Test provider info
        api_base, api_key = config._get_openai_compatible_provider_info(
            None, "test-key"
        )
        assert api_base == "https://inference.baseten.co/v1"
        assert api_key == "test-key"


if __name__ == "__main__":
    pytest.main([__file__])


class TestBasetenReasoningEffort:
    """reasoning_effort was silently dropped for every Baseten model -- the
    param list simply never included it, and drop_params strips anything not
    on that list before the request is built. Baseten hosts arbitrary,
    user-supplied model deployments (its own docstring: "Reference:
    https://inference.baseten.co/v1" -- there is no enumerable Baseten
    catalog in litellm's cost map the way there is for a curated provider),
    so this is declared unconditionally, matching every other parameter in
    this same list (temperature, tools, etc. are all forwarded regardless of
    whether the specific deployed model supports them -- Baseten's own
    endpoint is the one source of truth on capability, not litellm).
    Confirmed live: the actual zai-org/GLM-5.2 deployment behind this
    gateway's `glm-5.2` model genuinely honors a subset of reasoning_effort
    values (accepts high/xhigh/max, rejects low/medium/minimal with its own
    400) -- proof the parameter reaches a real, opinionated upstream, not a
    black hole (rayward-internal/llm-gateway-infra#290)."""

    def test_reasoning_effort_is_declared(self):
        config = BasetenConfig()
        assert "reasoning_effort" in config.get_supported_openai_params(model="zai-org/GLM-5.2")

    def test_reasoning_effort_flows_through_as_a_plain_value(self):
        """map_openai_params here is a pure allow-list passthrough (`elif
        param in supported_openai_params: optional_params[param] = value`) --
        no translation exists or is needed, since Baseten's own endpoint
        already accepts the literal OpenAI-style string."""
        config = BasetenConfig()
        optional_params = config.map_openai_params(
            non_default_params={"reasoning_effort": "high"},
            optional_params={},
            model="zai-org/GLM-5.2",
            drop_params=False,
        )
        assert optional_params == {"reasoning_effort": "high"}
