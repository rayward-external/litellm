import os
from typing import Final
from unittest.mock import patch

import httpx
import pytest
import respx

import litellm
from litellm.llms.baseten.chat import BasetenConfig
from litellm.utils import get_valid_models


class TestBasetenRouting:
    """Test Baseten routing logic"""

    def test_routing_logic(self):
        """Test routing between Model API and dedicated deployments"""
        config: Final = BasetenConfig()

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
        config: Final = BasetenConfig()

        # Test parameter mapping
        non_default_params: Final = {"max_tokens": 100, "temperature": 0.7, "top_p": 0.9}

        result: Final = config.map_openai_params(
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


class TestBasetenModelDiscovery:
    """Test Baseten model listing for `baseten/*` wildcard expansion and /v1/models"""

    def test_models_by_provider_comes_from_registry(self):
        """Static listing uses the model cost map, not the retired 2023 deployment IDs."""
        baseten_models: Final = litellm.models_by_provider["baseten"]
        servable_registry_models: Final = {
            key
            for key, value in litellm.model_cost.items()
            if value.get("litellm_provider") == "baseten" and not litellm._is_past_deprecation_date(value)
        }

        assert servable_registry_models
        assert servable_registry_models <= set(baseten_models)
        for retired_id in ("qvv0xeq", "q841o8w", "31dxrj3"):
            assert retired_id not in baseten_models

    def test_models_by_provider_skips_deprecated_models(self):
        """Retired Model APIs stay in the cost map for spend tracking but are not listed."""
        assert litellm.model_cost["baseten/zai-org/GLM-5"]["deprecation_date"]
        assert "baseten/zai-org/GLM-5" not in litellm.models_by_provider["baseten"]

    def test_cost_map_reload_drops_a_newly_retired_model(self):
        """A refreshed cost map that marks a listed model retired removes it without a restart."""
        model: Final = "baseten/acme/reload-probe"
        try:
            litellm.add_known_models({model: {"litellm_provider": "baseten"}})
            assert model in litellm.models_by_provider["baseten"]

            litellm.add_known_models({model: {"litellm_provider": "baseten", "deprecation_date": "2000-01-01"}})

            assert model not in litellm.models_by_provider["baseten"]
        finally:
            litellm.baseten_models.discard(model)
            litellm.add_known_models({})

    @pytest.mark.parametrize(
        "value, expected",
        [
            ({}, False),
            ({"deprecation_date": "2000-01-01"}, True),
            ({"deprecation_date": "2999-01-01"}, False),
            ({"deprecation_date": "not-a-date"}, False),
        ],
    )
    def test_is_past_deprecation_date(self, value, expected):
        assert litellm._is_past_deprecation_date(value) is expected

    def test_get_models_calls_model_api(self, respx_mock: respx.MockRouter) -> None:
        route: Final = respx_mock.get("https://inference.baseten.co/v1/models").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": "zai-org/GLM-5", "object": "model"},
                        {"id": "moonshotai/Kimi-K2.5", "object": "model"},
                    ],
                },
            )
        )

        models: Final = BasetenConfig().get_models(api_key="test-key")

        assert models == ["baseten/zai-org/GLM-5", "baseten/moonshotai/Kimi-K2.5"]
        assert route.calls.last.request.headers["Authorization"] == "Bearer test-key"

    def test_get_models_respects_custom_api_base(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get("https://custom.baseten.example/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "zai-org/GLM-5"}]})
        )

        models: Final = BasetenConfig().get_models(api_key="test-key", api_base="https://custom.baseten.example/v1/")

        assert models == ["baseten/zai-org/GLM-5"]

    @patch.dict(os.environ, {}, clear=True)
    def test_get_models_requires_api_key(self):
        with pytest.raises(ValueError, match="BASETEN_API_KEY"):
            BasetenConfig().get_models()

    def test_get_models_raises_on_error_status(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get("https://inference.baseten.co/v1/models").mock(
            return_value=httpx.Response(403, text="forbidden")
        )

        with pytest.raises(ValueError, match="403"):
            BasetenConfig().get_models(api_key="test-key")

    @patch.dict(os.environ, {"BASETEN_API_KEY": "test-key"})
    def test_get_valid_models_check_provider_endpoint(self, respx_mock: respx.MockRouter) -> None:
        """check_provider_endpoint=True returns the live catalog instead of the static list."""
        respx_mock.get("https://inference.baseten.co/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "brand-new/model"}]})
        )

        models: Final = get_valid_models(check_provider_endpoint=True, custom_llm_provider="baseten")

        assert models == ["baseten/brand-new/model"]


class TestBasetenReasoningEffort:
    def test_reasoning_effort_passthrough_with_drop_params(self):
        config: Final = BasetenConfig()

        result: Final = config.map_openai_params(
            non_default_params={"reasoning_effort": "high", "max_tokens": 100},
            optional_params={},
            model="deepseek-ai/DeepSeek-V4-Flash-0731",
            drop_params=True,
        )

        assert result["reasoning_effort"] == "high"
        assert result["max_tokens"] == 100

    def test_reasoning_effort_none_passthrough(self):
        config: Final = BasetenConfig()

        result: Final = config.map_openai_params(
            non_default_params={"reasoning_effort": "none"},
            optional_params={},
            model="moonshotai/Kimi-K3",
            drop_params=True,
        )

        assert result["reasoning_effort"] == "none"


if __name__ == "__main__":
    pytest.main([__file__])


class TestBasetenReasoningEffortDeclaredUnconditionally:
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
