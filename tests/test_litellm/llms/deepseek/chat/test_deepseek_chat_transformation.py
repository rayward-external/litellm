import pytest

from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig

REASONING_MODEL = "deepseek-reasoner"

# DeepSeek's own enum, as named verbatim in its 400 body on an unknown variant:
#   unknown variant `X`, expected one of `none`, `minimal`, `low`, `medium`,
#   `high`, `xhigh`, `max`
DEEPSEEK_REASONING_EFFORT_LEVELS = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)


def _map(non_default_params: dict, model: str = REASONING_MODEL) -> dict:
    return DeepSeekChatConfig().map_openai_params(
        non_default_params=non_default_params,
        optional_params={},
        model=model,
        drop_params=False,
    )


@pytest.mark.parametrize("effort", DEEPSEEK_REASONING_EFFORT_LEVELS)
def test_graded_reasoning_effort_is_forwarded_alongside_the_thinking_switch(effort):
    result = _map({"reasoning_effort": effort})

    assert result["reasoning_effort"] == effort
    expected_switch = "disabled" if effort == "none" else "enabled"
    assert result["thinking"] == {"type": expected_switch}


def test_litellm_default_sentinel_is_not_put_on_the_wire():
    # completion() accepts reasoning_effort="default" (litellm/main.py) as "no explicit
    # level". DeepSeek has no such level, so forwarding it would turn a request that
    # works today into a 400. It still drives the thinking switch, as before.
    result = _map({"reasoning_effort": "default"})

    assert "reasoning_effort" not in result
    assert result["thinking"] == {"type": "enabled"}


def test_out_of_vocabulary_reasoning_effort_is_forwarded_not_policed():
    # The value must reach DeepSeek so DeepSeek's own 400 names the vocabulary;
    # re-checking the enum here would be a second copy that can disagree with it.
    result = _map({"reasoning_effort": "BOGUS_LEVEL"})

    assert result["reasoning_effort"] == "BOGUS_LEVEL"
    assert result["thinking"] == {"type": "enabled"}


@pytest.mark.parametrize("switch", ["enabled", "disabled"])
def test_explicit_thinking_still_wins_the_switch_and_the_effort_rides_along(switch):
    result = _map({"thinking": {"type": switch}, "reasoning_effort": "max"})

    assert result["thinking"] == {"type": switch}
    assert result["reasoning_effort"] == "max"


@pytest.mark.parametrize(
    "non_default_params",
    [{}, {"reasoning_effort": None}, {"thinking": {"type": "enabled"}}],
)
def test_reasoning_effort_is_absent_when_the_caller_sent_none(non_default_params):
    result = _map(non_default_params)

    assert "reasoning_effort" not in result


def test_graded_effort_reaches_the_request_body():
    config = DeepSeekChatConfig()
    optional_params = config.map_openai_params(
        non_default_params={"reasoning_effort": "max"},
        optional_params={},
        model=REASONING_MODEL,
        drop_params=False,
    )

    body = config.transform_request(
        model=REASONING_MODEL,
        messages=[{"role": "user", "content": "hi"}],
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert body["reasoning_effort"] == "max"
    assert body["thinking"] == {"type": "enabled"}


@pytest.mark.parametrize("effort", DEEPSEEK_REASONING_EFFORT_LEVELS)
def test_thinking_mode_detection_is_unchanged_by_forwarding_the_effort(effort):
    # _thinking_mode_active reads the derived `thinking` switch and gates the
    # multi-turn reasoning_content echo, so the switch must survive alongside
    # the forwarded value rather than being replaced by it.
    config = DeepSeekChatConfig()
    optional_params = config.map_openai_params(
        non_default_params={"reasoning_effort": effort},
        optional_params={},
        model=REASONING_MODEL,
        drop_params=False,
    )

    active = config._thinking_mode_active(model=REASONING_MODEL, optional_params=optional_params)

    assert active is (effort != "none")


def _function_tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object"}},
    }


def test_drop_unsupported_tools_keeps_function_tools_only():
    optional_params = {
        "tools": [
            _function_tool("shell"),
            {"type": "namespace", "name": "container.exec"},
            _function_tool("apply_patch"),
        ],
        "tool_choice": "auto",
    }

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert [tool["function"]["name"] for tool in result["tools"]] == [
        "shell",
        "apply_patch",
    ]
    assert all(tool["type"] == "function" for tool in result["tools"])
    assert result["tool_choice"] == "auto"


def test_drop_unsupported_tools_drops_dangling_tool_choice_when_none_survive():
    optional_params = {
        "tools": [{"type": "namespace", "name": "container.exec"}],
        "tool_choice": "required",
        "parallel_tool_calls": True,
        "temperature": 0.2,
    }

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert "tools" not in result
    assert "tool_choice" not in result
    assert "parallel_tool_calls" not in result
    assert result["temperature"] == 0.2


def test_drop_unsupported_tools_is_noop_for_function_only():
    optional_params = {
        "tools": [_function_tool("shell")],
        "tool_choice": "auto",
    }

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert result is optional_params


def test_drop_unsupported_tools_is_noop_without_tools():
    optional_params = {"temperature": 0.7}

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert result is optional_params


def test_transform_request_strips_unsupported_tools_from_body():
    config = DeepSeekChatConfig()
    body = config.transform_request(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={
            "tools": [
                _function_tool("shell"),
                {"type": "namespace", "name": "container.exec"},
            ],
            "tool_choice": "auto",
        },
        litellm_params={},
        headers={},
    )

    assert [tool["type"] for tool in body["tools"]] == ["function"]
    assert body["tools"][0]["function"]["name"] == "shell"


async def test_async_transform_request_strips_unsupported_tools_from_body():
    config = DeepSeekChatConfig()
    body = await config.async_transform_request(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={
            "tools": [
                _function_tool("shell"),
                {"type": "namespace", "name": "container.exec"},
            ],
            "tool_choice": "auto",
        },
        litellm_params={},
        headers={},
    )

    assert [tool["type"] for tool in body["tools"]] == ["function"]
    assert body["tools"][0]["function"]["name"] == "shell"


def test_thinking_mode_active_bool_thinking_returns_false_without_crashing():
    config = DeepSeekChatConfig()
    assert config._thinking_mode_active(model="deepseek-reasoner", optional_params={"thinking": True}) is False


class TestDeepSeekThinkingParams:
    """Test thinking and reasoning_effort parameter handling for DeepSeek."""

    def setup_method(self):
        self.config = DeepSeekChatConfig()
        self.model = "deepseek-reasoner"

    def test_get_supported_openai_params_includes_thinking(self):
        """Test that thinking and reasoning_effort are in supported params."""
        params = self.config.get_supported_openai_params(self.model)
        assert "thinking" in params
        assert "reasoning_effort" in params

    def test_map_thinking_enabled(self):
        """Test that thinking={"type": "enabled"} is passed through correctly."""
        non_default_params = {"thinking": {"type": "enabled"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_thinking_with_budget_tokens_strips_budget(self):
        """Test that budget_tokens is stripped from thinking param (DeepSeek doesn't support it)."""
        non_default_params = {"thinking": {"type": "enabled", "budget_tokens": 2048}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # Should strip budget_tokens, only pass type
        assert result["thinking"] == {"type": "enabled"}
        assert "budget_tokens" not in result.get("thinking", {})

    def test_map_reasoning_effort_medium(self):
        """Test that reasoning_effort='medium' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "medium"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_low(self):
        """Test that reasoning_effort='low' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "low"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_high(self):
        """Test that reasoning_effort='high' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "high"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_none_does_not_enable_thinking(self):
        """Test that reasoning_effort='none' does not enable thinking."""
        non_default_params = {"reasoning_effort": "none"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "disabled"}

    def test_map_reasoning_effort_null_does_not_enable_thinking(self):
        """Test that reasoning_effort=None does not enable thinking."""
        non_default_params = {"reasoning_effort": None}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

    def test_thinking_takes_precedence_over_reasoning_effort(self):
        """Test that thinking param takes precedence when both are provided."""
        non_default_params = {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # thinking should be set, reasoning_effort should not override
        assert result["thinking"] == {"type": "enabled"}

    def test_invalid_thinking_type_ignored(self):
        """Test that invalid thinking type values are ignored."""
        non_default_params = {"thinking": {"type": "invalid"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

    def test_thinking_none_value_ignored(self):
        """Test that thinking=None is ignored."""
        non_default_params = {"thinking": None}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

    def test_drop_unsupported_tools_removes_dangling_tool_choice(self):
        optional_params = {
            "tools": [
                {"type": "namespace", "name": "local_shell"},
                {"type": "function", "function": {"name": "get_weather"}},
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "local_shell"},
            },
            "parallel_tool_calls": True,
        }

        result = self.config._drop_unsupported_tools(optional_params)

        assert result["tools"] == [
            {"type": "function", "function": {"name": "get_weather"}}
        ]
        assert "tool_choice" not in result
        assert result["parallel_tool_calls"] is True
