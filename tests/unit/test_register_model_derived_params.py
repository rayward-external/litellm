"""
Regression: ``register_model`` must not persist a ``supported_openai_params``
list it derived from a router deployment id.

Custom-priced deployments register under their deployment id. ``get_model_info``
cannot resolve an id like ``bedrock-claude-pool-1`` as a model name, so it
derives the provider's minimal default list. Persisting that snapshot made
``Router._set_model_group_info`` advertise it for the whole group (last
deployment wins), so the model hub's ``supported_openai_params`` varied with
whichever deployments a process had already registered.
"""

import pytest


@pytest.fixture(autouse=True)
def _restore_model_cost():
    import litellm

    original = dict(litellm.model_cost)
    try:
        yield
    finally:
        litellm.model_cost.clear()
        litellm.model_cost.update(original)


BEDROCK_MODEL = "anthropic.claude-sonnet-4-5-20250929-v1:0"


def _register_request_pricing(deployment_id: str) -> None:
    """What the proxy does on every request to a custom-priced deployment."""
    import litellm.main as litellm_main

    litellm_main._register_custom_pricing_for_request(
        model=BEDROCK_MODEL,
        custom_llm_provider="bedrock",
        kwargs={
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 1.5e-5,
            "litellm_metadata": {"model_info": {"id": deployment_id}},
        },
        model_info={"id": deployment_id},
    )


def test_derived_params_are_not_persisted_for_a_deployment_id():
    import litellm

    _register_request_pricing("bedrock-claude-pool-1")

    entry = litellm.model_cost["bedrock-claude-pool-1"]
    assert "supported_openai_params" not in entry
    assert entry["input_cost_per_token"] == 3e-6


def test_explicit_params_are_still_persisted():
    import litellm

    litellm.register_model(
        {
            "bedrock-claude-pool-2": {
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 5e-6,
                "litellm_provider": "bedrock",
                "supported_openai_params": ["max_tokens", "tools"],
            }
        },
        custom_llm_provider="bedrock",
    )

    assert litellm.model_cost["bedrock-claude-pool-2"]["supported_openai_params"] == ["max_tokens", "tools"]


def test_group_info_is_not_narrowed_by_a_custom_priced_leg():
    import litellm

    router = litellm.Router(
        model_list=[
            {
                "model_name": "claude-pool",
                "litellm_params": {"model": "anthropic/claude-sonnet-4-5-20250929", "api_key": "sk-test"},
                "model_info": {"id": "anthropic-claude-pool"},
            },
            {
                "model_name": "claude-pool",
                "litellm_params": {
                    "model": f"bedrock/{BEDROCK_MODEL}",
                    "aws_region_name": "us-east-1",
                    "input_cost_per_token": 3e-6,
                    "output_cost_per_token": 1.5e-5,
                },
                "model_info": {"id": "bedrock-claude-pool"},
            },
        ]
    )
    before = router.get_model_group_info("claude-pool").supported_openai_params
    _register_request_pricing("bedrock-claude-pool")
    after = router.get_model_group_info("claude-pool").supported_openai_params

    assert before and "tools" in before
    assert sorted(after) == sorted(before)
