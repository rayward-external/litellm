"""Unit tests for /v1/messages/batches per-request reasoning_effort translation.

The batch-create path forwards each request's ``params`` to Anthropic (or stages
it for Bedrock). Anthropic's messages API has no ``reasoning_effort`` field and
adaptive-thinking models reject ``thinking.type == "enabled"``, so
``mb._translate_reasoning_effort_params`` rewrites the field into the native
shape before it is forwarded. These tests exercise that translation directly,
mocking the model-capability lookup (no live calls / no network).
"""

from unittest.mock import patch

import pytest

import litellm.proxy.anthropic_endpoints.messages_batches as mb
from litellm.constants import DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET
from litellm.llms.anthropic.chat.transformation import AnthropicConfig


def _mock_capability(monkeypatch, *, known: bool, adaptive: bool):
    """Pin the two capability probes the translation depends on.

    ``_map_reasoning_effort`` itself branches on ``_is_adaptive_thinking_model``,
    so pinning that one classmethod keeps the thinking shape and the
    output_config branch consistent without touching the real model map.
    """
    monkeypatch.setattr(mb, "_anthropic_model_is_known", lambda model: known)
    monkeypatch.setattr(
        AnthropicConfig,
        "_is_adaptive_thinking_model",
        staticmethod(lambda *a, **k: adaptive),
    )


def _base_params(**extra):
    params = {
        "model": "claude-opus-4-8",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "hi"}],
    }
    params.update(extra)
    return params


def test_reasoning_effort_high_on_adaptive_model(monkeypatch):
    _mock_capability(monkeypatch, known=True, adaptive=True)
    params = _base_params(reasoning_effort="high")

    out = mb._translate_reasoning_effort_params(params)

    assert "reasoning_effort" not in out
    assert out["thinking"] == {"type": "adaptive"}
    assert out["output_config"] == {"effort": "high"}
    # untouched fields survive
    assert out["model"] == "claude-opus-4-8"
    assert out["max_tokens"] == 128
    assert out["messages"] == [{"role": "user", "content": "hi"}]
    # original object is not mutated
    assert params["reasoning_effort"] == "high"


def test_reasoning_effort_on_legacy_model_maps_to_enabled_budget(monkeypatch):
    _mock_capability(monkeypatch, known=True, adaptive=False)
    # max_tokens comfortably above the fixed high budget: budget survives uncapped.
    params = _base_params(model="claude-haiku-4-5", reasoning_effort="high", max_tokens=8192)

    out = mb._translate_reasoning_effort_params(params)

    assert "reasoning_effort" not in out
    assert out["thinking"] == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    }
    assert "output_config" not in out


def test_legacy_budget_capped_below_max_tokens(monkeypatch):
    """Anthropic/Bedrock require max_tokens > budget_tokens: a fixed per-effort
    budget larger than the record's max_tokens must be clamped to max_tokens-1
    (mirrors the chat path's _cap_thinking_budget_to_max_tokens)."""
    _mock_capability(monkeypatch, known=True, adaptive=False)
    params = _base_params(model="claude-haiku-4-5", reasoning_effort="high", max_tokens=2048)

    out = mb._translate_reasoning_effort_params(params)

    assert "reasoning_effort" not in out
    assert out["thinking"] == {"type": "enabled", "budget_tokens": 2047}
    assert out["max_tokens"] == 2048


def test_legacy_budget_unfittable_drops_thinking(monkeypatch):
    """max_tokens at/below the minimum thinking budget cannot fit ANY thinking:
    drop thinking entirely (same decision as the chat path) instead of
    forwarding a guaranteed-400 shape upstream."""
    _mock_capability(monkeypatch, known=True, adaptive=False)
    params = _base_params(model="claude-haiku-4-5", reasoning_effort="high", max_tokens=128)

    out = mb._translate_reasoning_effort_params(params)

    assert "reasoning_effort" not in out
    assert "thinking" not in out
    assert "output_config" not in out


def test_caller_provided_thinking_and_output_config_not_clobbered(monkeypatch):
    _mock_capability(monkeypatch, known=True, adaptive=True)
    params = _base_params(
        reasoning_effort="high",
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
    )

    out = mb._translate_reasoning_effort_params(params)

    # reasoning_effort dropped, caller's native fields win verbatim
    assert "reasoning_effort" not in out
    assert out["thinking"] == {"type": "adaptive"}
    assert out["output_config"] == {"effort": "low"}


def test_caller_provided_legacy_thinking_not_clobbered(monkeypatch):
    _mock_capability(monkeypatch, known=True, adaptive=False)
    params = _base_params(
        model="claude-haiku-4-5",
        reasoning_effort="high",
        thinking={"type": "enabled", "budget_tokens": 9999},
    )

    out = mb._translate_reasoning_effort_params(params)

    assert "reasoning_effort" not in out
    assert out["thinking"] == {"type": "enabled", "budget_tokens": 9999}
    assert "output_config" not in out


def test_no_reasoning_effort_is_byte_identical_passthrough(monkeypatch):
    # Must not even consult the capability lookup — nothing to translate.
    monkeypatch.setattr(
        mb,
        "_anthropic_model_is_known",
        lambda model: pytest.fail("capability lookup must not run without reasoning_effort"),
    )
    params = _base_params(thinking={"type": "adaptive"})

    out = mb._translate_reasoning_effort_params(params)

    assert out is params  # same object, no copy, no mutation


def test_unknown_model_is_untouched_passthrough(monkeypatch):
    _mock_capability(monkeypatch, known=False, adaptive=False)
    params = _base_params(model="some-unmapped-model", reasoning_effort="high")

    out = mb._translate_reasoning_effort_params(params)

    assert out is params  # fail open: forwarded untranslated, same object


def test_reasoning_effort_none_on_adaptive_omits_thinking(monkeypatch):
    _mock_capability(monkeypatch, known=True, adaptive=True)
    params = _base_params(reasoning_effort="none")

    out = mb._translate_reasoning_effort_params(params)

    assert "reasoning_effort" not in out
    assert "thinking" not in out  # adaptive can't disable thinking → omit, don't fabricate
    assert "output_config" not in out


def test_reasoning_effort_none_on_legacy_omits_thinking(monkeypatch):
    _mock_capability(monkeypatch, known=True, adaptive=False)
    params = _base_params(model="claude-haiku-4-5", reasoning_effort="none")

    out = mb._translate_reasoning_effort_params(params)

    assert "reasoning_effort" not in out
    assert "thinking" not in out


def test_reasoning_effort_dict_shape_is_coerced(monkeypatch):
    _mock_capability(monkeypatch, known=True, adaptive=True)
    params = _base_params(reasoning_effort={"effort": "medium", "summary": "concise"})

    out = mb._translate_reasoning_effort_params(params)

    assert "reasoning_effort" not in out
    assert out["thinking"] == {"type": "adaptive"}
    assert out["output_config"] == {"effort": "medium"}


def test_non_dict_params_passthrough(monkeypatch):
    monkeypatch.setattr(
        mb,
        "_anthropic_model_is_known",
        lambda model: pytest.fail("capability lookup must not run on non-dict params"),
    )
    assert mb._translate_reasoning_effort_params(None) is None
    sentinel = "not-a-dict"
    assert mb._translate_reasoning_effort_params(sentinel) is sentinel
