import os
import pytest
from unittest.mock import patch
import litellm.utils
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


REASONING_MODEL = "some-org/some-reasoning-model"


class TestBasetenReasoningEffort:
    """Reasoning intent has to be *translated* into Baseten's native `thinking`
    switch, not forwarded as an OpenAI-style `reasoning_effort` string.

    Measured against the live https://inference.baseten.co/v1 endpoint:
    `thinking={"type": "enabled"}` reliably produces reasoning tokens and
    `{"type": "disabled"}` reliably suppresses them on the reasoning-capable
    deployments served there, while `reasoning_effort` is either ignored
    outright (zero reasoning tokens, no error) or accepted for only a narrow
    subset of values and rejected with an upstream HTTP 400 for the rest. The
    switch is binary -- reasoning on or off -- so there is no graded effort
    ladder to build, and the raw `reasoning_effort` key must never reach the
    provider."""

    def test_reasoning_params_are_declared(self):
        """Both must be declared or litellm's own drop_params strips the
        caller's reasoning intent before map_openai_params ever sees it."""
        config = BasetenConfig()
        supported = config.get_supported_openai_params(model=REASONING_MODEL)
        assert "reasoning_effort" in supported
        assert "thinking" in supported

    def test_reasoning_effort_high_enables_thinking_and_is_not_forwarded(self):
        config = BasetenConfig()
        optional_params = config.map_openai_params(
            non_default_params={"reasoning_effort": "high"},
            optional_params={},
            model=REASONING_MODEL,
            drop_params=False,
        )
        assert optional_params["extra_body"]["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in optional_params
        assert "thinking" not in optional_params

    def test_reasoning_effort_none_disables_thinking(self):
        """The effort value "none" is litellm's standard OpenAI-style sentinel
        for "reasoning off"; Baseten has no matching effort level, so it becomes
        the disabled switch."""
        config = BasetenConfig()
        optional_params = config.map_openai_params(
            non_default_params={"reasoning_effort": "none"},
            optional_params={},
            model=REASONING_MODEL,
            drop_params=False,
        )
        assert optional_params["extra_body"]["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in optional_params
        assert "thinking" not in optional_params

    @pytest.mark.parametrize("effort", ["minimal", "low", "medium", "xhigh", "max"])
    def test_every_other_effort_enables_thinking_without_emitting_the_effort(self, effort):
        """The regression guard: "low"/"medium" et al are values the upstream
        rejects with a 400 when sent literally. Whatever graded effort a caller
        asks for, only the binary switch may leave here."""
        config = BasetenConfig()
        optional_params = config.map_openai_params(
            non_default_params={"reasoning_effort": effort},
            optional_params={},
            model=REASONING_MODEL,
            drop_params=False,
        )
        assert optional_params == {"extra_body": {"thinking": {"type": "enabled"}}}

    def test_explicit_thinking_wins_over_reasoning_effort(self):
        """A caller who names the native switch was specific -- honour it."""
        config = BasetenConfig()
        optional_params = config.map_openai_params(
            non_default_params={
                "thinking": {"type": "disabled"},
                "reasoning_effort": "high",
            },
            optional_params={},
            model=REASONING_MODEL,
            drop_params=False,
        )
        assert optional_params == {"extra_body": {"thinking": {"type": "disabled"}}}

    def test_explicit_thinking_passes_through_unchanged(self):
        config = BasetenConfig()
        optional_params = config.map_openai_params(
            non_default_params={"thinking": {"type": "enabled"}},
            optional_params={},
            model=REASONING_MODEL,
            drop_params=False,
        )
        assert optional_params == {"extra_body": {"thinking": {"type": "enabled"}}}

    def test_no_reasoning_params_adds_no_thinking_key(self):
        """Default-off behaviour is preserved: absent any reasoning intent, the
        deployment's own default stands and nothing is injected."""
        config = BasetenConfig()
        optional_params = config.map_openai_params(
            non_default_params={"max_tokens": 100, "temperature": 0.7},
            optional_params={},
            model=REASONING_MODEL,
            drop_params=False,
        )
        assert "thinking" not in optional_params
        assert "extra_body" not in optional_params
        assert optional_params == {"max_tokens": 100, "temperature": 0.7}

    def test_other_params_are_unaffected_by_the_translation(self):
        config = BasetenConfig()
        optional_params = config.map_openai_params(
            non_default_params={
                "max_completion_tokens": 256,
                "temperature": 0.2,
                "reasoning_effort": "low",
            },
            optional_params={},
            model=REASONING_MODEL,
            drop_params=False,
        )
        assert optional_params == {
            "max_tokens": 256,
            "temperature": 0.2,
            "extra_body": {"thinking": {"type": "enabled"}},
        }


class TestBasetenThinkingReachesTheWire:
    """The tests above assert on the dict `map_openai_params` returns. That is an
    INTERMEDIATE structure, and asserting on it alone is how this feature shipped
    broken once already.

    Baseten is dispatched through the OpenAI *SDK*, not a raw JSON POST:
    litellm.main's `custom_llm_provider == "baseten"` branch reaches
    `_complete_custom_openai` -> `openai_chat_completions.completion`, and
    `OpenAIGPTConfig.transform_request` splats optional_params directly into the
    kwargs of `client.chat.completions.create(**data)`. That method has an explicit
    signature with no `thinking` parameter and no `**kwargs`, so emitting `thinking`
    at the TOP LEVEL of optional_params raises TypeError before any HTTP call and
    litellm re-wraps it as a 500 — a hard outage on exactly the requests the
    translation exists to fix, while every intermediate-dict test stays green.

    These tests therefore assert on the request body actually handed to the SDK."""

    @staticmethod
    def _request_body(**kwargs):
        """The body litellm would hand to the OpenAI SDK for a baseten call."""
        optional_params = litellm.utils.get_optional_params(
            model=REASONING_MODEL,
            custom_llm_provider="baseten",
            **kwargs,
        )
        return BasetenConfig().transform_request(
            model=REASONING_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

    def test_thinking_is_never_a_top_level_sdk_kwarg(self):
        """The exact shape that 500s: `thinking` must not appear beside `model`
        and `messages`, because those become create() kwargs."""
        body = self._request_body(reasoning_effort="high")
        assert "thinking" not in body, (
            "thinking reached the top level of the SDK call — "
            "Completions.create() has no such parameter and this will TypeError"
        )
        assert body["extra_body"]["thinking"] == {"type": "enabled"}

    def test_raw_reasoning_effort_never_reaches_the_provider(self):
        """Upstream rejects most effort values with a 400; only the switch may go."""
        body = self._request_body(reasoning_effort="low")
        assert "reasoning_effort" not in body
        assert "reasoning_effort" not in body.get("extra_body", {})
        assert body["extra_body"]["thinking"] == {"type": "enabled"}

    def test_effort_none_sends_the_disabled_switch(self):
        body = self._request_body(reasoning_effort="none")
        assert body["extra_body"]["thinking"] == {"type": "disabled"}

    def test_no_reasoning_intent_sends_no_thinking_at_all(self):
        """Default-off is preserved end to end, not just in the intermediate dict."""
        body = self._request_body(temperature=0.5)
        assert "thinking" not in body
        assert "thinking" not in body.get("extra_body", {})

    def test_every_sdk_kwarg_is_accepted_by_the_openai_client(self):
        """Guards the whole class of bug rather than this one key: every top-level
        key we emit must be a real parameter of Completions.create()."""
        import inspect

        from openai.resources.chat.completions import Completions

        signature = inspect.signature(Completions.create)
        accepts_var_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        body = self._request_body(reasoning_effort="high", temperature=0.2)
        unknown = [k for k in body if k not in signature.parameters]
        assert accepts_var_kwargs or not unknown, f"these keys would TypeError in Completions.create(): {unknown}"
