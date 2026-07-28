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
        assert config.get_api_base_for_model("openai/gpt-oss-120b") == "https://inference.baseten.co/v1"


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
        api_base, api_key = config._get_openai_compatible_provider_info(None, "test-key")
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


class TestBasetenReasoningEffortIsNotTranslated:
    """Regression guard for the removal of the reasoning_effort -> `thinking`
    translation.

    This file briefly resolved the caller's reasoning_effort into Baseten's
    native binary switch (`thinking={"type": "enabled"|"disabled"}`), deleting
    the caller's own value in the process. Probed directly against
    https://inference.baseten.co/v1/chat/completions with a real key on
    2026-07-28, that was destroying real capability:

      * moonshotai/Kimi-K3 -- none/minimal/low/high/max all return 200 (a real
        graded ladder; `none` yields 0 reasoning tokens) and a bogus value 400s.
        Collapsing five distinct levels to one on/off bit threw the ladder away.
      * zai-org/GLM-5.2 -- none/high/max return 200; minimal/low genuinely 400.

    So the gateway now forwards the literal value and lets the provider's own
    vocabulary decide. GLM's 400 on minimal/low is the ACCEPTED consequence:
    it is the provider's real answer, delivered faithfully rather than hidden
    behind a substitution. These tests assert on the actual dict
    map_openai_params returns, and pin BOTH halves of the contract -- the value
    survives verbatim, and no `thinking` key is injected anywhere."""

    @pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max"])
    def test_every_effort_level_is_forwarded_verbatim(self, effort):
        """Including the levels the old translation flattened: "none" became
        {"type": "disabled"} and every other level became {"type": "enabled"},
        so the caller's actual choice never reached the provider."""
        config = BasetenConfig()
        optional_params = config.map_openai_params(
            non_default_params={"reasoning_effort": effort},
            optional_params={},
            model="moonshotai/Kimi-K3",
            drop_params=False,
        )
        assert optional_params == {"reasoning_effort": effort}

    def test_no_thinking_key_is_injected_anywhere(self):
        """The translation wrote the switch into extra_body (not top-level, to
        avoid a TypeError in the OpenAI SDK). Nothing may write it now -- not
        top-level, not in extra_body, which must not even be created."""
        config = BasetenConfig()
        optional_params = config.map_openai_params(
            non_default_params={"reasoning_effort": "high"},
            optional_params={},
            model="zai-org/GLM-5.2",
            drop_params=False,
        )
        assert "thinking" not in optional_params
        assert "extra_body" not in optional_params
        assert optional_params == {"reasoning_effort": "high"}

    def test_thinking_is_no_longer_a_declared_param(self):
        """`thinking` was declared supported only to carry the translation. With
        the translation gone it is not a Baseten param, so drop_params should
        treat it like any other unknown key."""
        config = BasetenConfig()
        assert "thinking" not in config.get_supported_openai_params(model="zai-org/GLM-5.2")

    def test_reasoning_effort_does_not_disturb_neighbouring_params(self):
        """The translation used to rewrite the whole reasoning branch; confirm
        the plain allow-list passthrough leaves everything else exactly as it
        was, including the max_completion_tokens -> max_tokens rename."""
        config = BasetenConfig()
        optional_params = config.map_openai_params(
            non_default_params={
                "max_completion_tokens": 256,
                "temperature": 0.2,
                "reasoning_effort": "low",
            },
            optional_params={},
            model="moonshotai/Kimi-K3",
            drop_params=False,
        )
        assert optional_params == {
            "max_tokens": 256,
            "temperature": 0.2,
            "reasoning_effort": "low",
        }
