"""Tests for pass-through admission control (cost-tracking enforcement).

The guard exists because pass-through forwards a client request upstream using
the proxy's own credentials. An unpriced route bills the upstream account and
records $0 against the caller's key, which both defeats budgets and corrupts
any reconciliation that splits a provider invoice by gateway-computed cost.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy.pass_through_endpoints.passthrough_admission import (  # noqa: E402
    PassthroughAdmissionError,
    _normalize_path,
    enforce_passthrough_admission,
    find_matching_capability,
)

ANTHROPIC_MESSAGES = {
    "provider": "anthropic",
    "methods": ["POST"],
    "path": "/v1/messages",
    "model_source": "body",
}
BEDROCK_CONVERSE = {
    "provider": "bedrock",
    "methods": ["POST"],
    "path": "/model/{model_id}/converse",
    "model_source": "path:model_id",
}
CAPABILITIES = [ANTHROPIC_MESSAGES, BEDROCK_CONVERSE]

ENABLED = {
    "passthrough_require_cost_tracking": True,
    "passthrough_capabilities": CAPABILITIES,
}


def _enforce(path, method="POST", provider="anthropic", body=None, settings=None):
    enforce_passthrough_admission(
        general_settings=settings if settings is not None else ENABLED,
        provider=provider,
        method=method,
        path=path,
        request_body=body if body is not None else {"model": "claude-sonnet-5"},
    )


# ---------------------------------------------------------------------------
# Disabled by default — existing deployments must not change behaviour.
# ---------------------------------------------------------------------------


def test_no_op_when_setting_absent():
    _enforce("/v1/anything/at/all", settings={})


def test_no_op_when_setting_false():
    _enforce(
        "/v1/anything",
        settings={"passthrough_require_cost_tracking": False, "passthrough_capabilities": []},
    )


# ---------------------------------------------------------------------------
# The holes a prefix-based load balancer structurally cannot close.
# Each of these was a real finding; they are the reason this guard exists.
# ---------------------------------------------------------------------------


def test_registered_capability_is_allowed():
    _enforce("/v1/messages")


def test_subtree_of_a_registered_path_is_denied():
    # /v1/messages/batches rides along on any prefix rule for /v1/messages.
    # Batch work is billed later with no job-to-key ledger, so it must not pass.
    with pytest.raises(PassthroughAdmissionError) as exc:
        _enforce("/v1/messages/batches")
    assert "not a registered capability" in str(exc.value.message)


def test_free_sibling_endpoint_is_denied():
    # count_tokens is free. It would emit a legitimate $0 row that is
    # indistinguishable from "billed but unpriced" without route classification.
    with pytest.raises(PassthroughAdmissionError):
        _enforce("/v1/messages/count_tokens")


def test_method_is_enforced():
    # GET on the same path is object management (e.g. listing stored
    # completions) — free, and a prefix rule cannot express the distinction.
    with pytest.raises(PassthroughAdmissionError):
        _enforce("/v1/messages", method="GET")


def test_path_placeholder_matches_exactly_one_segment():
    _enforce(
        "/model/anthropic.claude-sonnet-5/converse",
        provider="bedrock",
        body={},
    )
    with pytest.raises(PassthroughAdmissionError):
        _enforce("/model/a/b/converse", provider="bedrock", body={})


def test_sibling_operation_under_placeholder_is_denied():
    # invoke-with-bidirectional-stream is a separate inference operation
    # outside the verified costing surface.
    with pytest.raises(PassthroughAdmissionError):
        _enforce(
            "/model/anthropic.claude-sonnet-5/invoke-with-bidirectional-stream",
            provider="bedrock",
            body={},
        )


def test_provider_is_enforced():
    with pytest.raises(PassthroughAdmissionError):
        _enforce("/v1/messages", provider="openai")


# ---------------------------------------------------------------------------
# Pricing: a registered route still has to resolve to a real price.
# ---------------------------------------------------------------------------


def test_unpriced_model_is_denied():
    with pytest.raises(PassthroughAdmissionError) as exc:
        _enforce("/v1/messages", body={"model": "definitely-not-a-real-model-xyz"})
    assert "no explicit price entry" in str(exc.value.message)


def test_missing_model_is_denied():
    with pytest.raises(PassthroughAdmissionError):
        _enforce("/v1/messages", body={})


def test_require_priced_model_can_be_disabled_for_genuinely_free_routes():
    settings = {
        "passthrough_require_cost_tracking": True,
        "passthrough_capabilities": [
            {**ANTHROPIC_MESSAGES, "path": "/v1/free", "require_priced_model": False}
        ],
    }
    _enforce("/v1/free", body={}, settings=settings)


def test_model_from_path_is_priced_check_too():
    # Bedrock reads the model from the URL, not the body. A raw ARN has no
    # price-map entry, so it must be refused rather than recorded at $0.
    with pytest.raises(PassthroughAdmissionError):
        _enforce(
            "/model/arn:aws:bedrock:us-east-1:1234:application-inference-profile%2Fabc/converse",
            provider="bedrock",
            body={},
        )


# ---------------------------------------------------------------------------
# Path handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [("//v1//messages", "/v1/messages"), ("/v1/messages/", "/v1/messages"), ("", "/"), ("/", "/")],
)
def test_normalize_path(raw, expected):
    assert _normalize_path(raw) == expected


def test_duplicate_slashes_do_not_bypass_matching():
    _enforce("//v1//messages")


def test_percent_encoded_separator_is_not_decoded():
    # Decoding %2F would let a caller synthesise a path matching a narrower
    # template than the one the upstream actually routes.
    with pytest.raises(PassthroughAdmissionError):
        _enforce("/v1/messages%2Fbatches")


def test_unknown_capability_shape_is_ignored_not_crashed():
    settings = {
        "passthrough_require_cost_tracking": True,
        "passthrough_capabilities": ["not-a-dict", ANTHROPIC_MESSAGES],
    }
    _enforce("/v1/messages", settings=settings)


def test_non_list_capabilities_is_a_config_error():
    with pytest.raises(PassthroughAdmissionError) as exc:
        _enforce("/v1/messages", settings={
            "passthrough_require_cost_tracking": True,
            "passthrough_capabilities": {"provider": "anthropic"},
        })
    assert exc.value.status_code == 500


def test_empty_capability_list_denies_everything():
    # Fail closed: enabling enforcement with nothing registered must not
    # silently allow all traffic.
    with pytest.raises(PassthroughAdmissionError):
        _enforce("/v1/messages", settings={
            "passthrough_require_cost_tracking": True,
            "passthrough_capabilities": [],
        })


def test_find_matching_capability_returns_the_match_object():
    capability, match = find_matching_capability(
        CAPABILITIES, "bedrock", "POST", "/model/my-model/converse"
    )
    assert capability is BEDROCK_CONVERSE
    assert match is not None and match.group("model_id") == "my-model"
