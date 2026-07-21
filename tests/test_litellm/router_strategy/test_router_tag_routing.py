#### What this tests ####
# This tests litellm router

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))  # Adds the parent directory to the system path
import logging
import os


import litellm
from litellm._logging import verbose_logger


@pytest.mark.asyncio()
async def test_router_free_paid_tier():
    """
    Pass list of orgs in 1 model definition,
    expect a unique deployment for each to be created
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["free"],
                },
                "model_info": {"id": "very-cheap-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["paid"],
                },
                "model_info": {"id": "very-expensive-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        # this should pick model with id == very-cheap-model
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Tell me a joke."}],
            metadata={"tags": ["free"]},
            mock_response="Tell me a joke.",
        )

        response_extra_info = response._hidden_params

        assert response_extra_info["model_id"] == "very-cheap-model"

    for _ in range(5):
        # this should pick model with id == very-cheap-model
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Tell me a joke."}],
            metadata={"tags": ["paid"]},
            mock_response="Tell me a joke.",
        )

        response_extra_info = response._hidden_params

        assert response_extra_info["model_id"] == "very-expensive-model"


@pytest.mark.asyncio()
async def test_router_free_paid_tier_embeddings():
    """
    Pass list of orgs in 1 model definition,
    expect a unique deployment for each to be created
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["free"],
                    "mock_response": ["1", "2", "3"],
                },
                "model_info": {"id": "very-cheap-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["paid"],
                    "mock_response": ["1", "2", "3"],
                },
                "model_info": {"id": "very-expensive-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["default"],
                    "mock_response": ["1", "2", "3"],
                },
                "model_info": {"id": "default-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        # this should pick model with id == very-cheap-model
        response = await router.aembedding(
            model="gpt-4",
            input="Tell me a joke.",
            metadata={"tags": ["free"]},
            mock_response=[1, 2, 3],
        )

        response_extra_info = response._hidden_params

        assert response_extra_info["model_id"] == "very-cheap-model"

    for _ in range(5):
        # this should pick model with id == very-expensive-model
        response = await router.aembedding(
            model="gpt-4",
            input="Tell me a joke.",
            metadata={"tags": ["paid"]},
            mock_response=[1, 2, 3],
        )

        response_extra_info = response._hidden_params

        assert response_extra_info["model_id"] == "very-expensive-model"


@pytest.mark.asyncio()
async def test_default_tagged_deployments():
    """
    - only use default deployment for untagged requests
    - if a request has tag "default", use default deployment
    """

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["default"],
                },
                "model_info": {"id": "default-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                },
                "model_info": {"id": "default-model-2"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["teamA"],
                },
                "model_info": {"id": "very-expensive-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        # Untagged request, this should pick model with id == "default-model"
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Tell me a joke."}],
            mock_response="Tell me a joke.",
        )

        response_extra_info = response._hidden_params

        assert response_extra_info["model_id"] == "default-model"

    for _ in range(5):
        # requests tagged with "default", this should pick model with id == "default-model"
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Tell me a joke."}],
            metadata={"tags": ["default"]},
            mock_response="Tell me a joke.",
        )

        response_extra_info = response._hidden_params

        assert response_extra_info["model_id"] == "default-model"

    for _ in range(5):
        # requests with invalid tags, this should pick model with id == "default-model"
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Tell me a joke."}],
            metadata={"tags": ["invalid-tag"]},
            mock_response="Tell me a joke.",
        )

        response_extra_info = response._hidden_params

        assert response_extra_info["model_id"] == "default-model"


@pytest.mark.asyncio()
async def test_error_from_tag_routing():
    """
    Tests the correct error raised when no deployments found for tag
    """
    verbose_logger.setLevel(logging.DEBUG)
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                },
                "model_info": {"id": "default-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                },
                "model_info": {"id": "default-model-2"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["teamA"],
                },
                "model_info": {"id": "very-expensive-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    try:
        await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Tell me a joke."}],
            metadata={"tags": ["paid"]},
            mock_response="Tell me a joke.",
        )

        pytest.fail("this should have failed - expected it to fail")
    except Exception as e:
        from litellm.types.router import RouterErrors

        assert RouterErrors.no_deployments_with_tag_routing.value in str(e)
        pass


def test_tag_routing_with_list_of_tags():
    """
    Test that the router can handle a list of tags with match_any behavior
    """
    from litellm.router_strategy.tag_based_routing import is_valid_deployment_tag

    assert is_valid_deployment_tag(["teamA", "teamB"], ["teamA"])
    assert is_valid_deployment_tag(["teamA", "teamB"], ["teamA", "teamB"])
    assert is_valid_deployment_tag(["teamA", "teamB"], ["teamA", "teamC"])
    assert is_valid_deployment_tag(["teamA"], ["teamA", "teamB"])
    assert not is_valid_deployment_tag(["teamA", "teamB"], ["teamC"])
    assert not is_valid_deployment_tag(["teamA", "teamB"], [])
    assert not is_valid_deployment_tag(["default"], ["teamA"])


def test_tag_routing_with_list_of_tags_match_all():
    """
    Test that the router can handle a list of tags with match_all behavior
    """
    from litellm.router_strategy.tag_based_routing import is_valid_deployment_tag

    assert is_valid_deployment_tag(["teamA", "teamB"], ["teamA"], match_any=False)
    assert is_valid_deployment_tag(["teamA", "teamB"], ["teamA", "teamB"], match_any=False)
    assert not is_valid_deployment_tag(["teamA", "teamB", "teamC"], ["teamA", "teamD"], match_any=False)
    assert not is_valid_deployment_tag(["teamA"], ["teamA", "teamB"], match_any=False)
    assert not is_valid_deployment_tag(["teamA", "teamB"], ["teamA", "teamC"], match_any=False)
    assert not is_valid_deployment_tag(["teamA", "teamB"], [], match_any=False)
    assert not is_valid_deployment_tag(["default"], ["teamA"], match_any=False)


def test_strict_tag_routing_without_request_tags_blocks_header_regex_fallback():
    """
    When tag_filtering_match_any=False, deployments with plain tags must require
    those request tags before header regex can match. A spoofed User-Agent must
    not route to a tagged deployment when the request has no tags.
    """
    from litellm.router_strategy.tag_based_routing import _match_deployment

    deployment = {
        "model_name": "restricted-model",
        "litellm_params": {
            "model": "gpt-4o",
            "tags": ["internal"],
            "tag_regex": ["^User-Agent: internal-tool"],
        },
    }

    assert (
        _match_deployment(
            deployment=deployment,
            request_tags=None,
            header_strings=["User-Agent: internal-tool"],
            match_any=False,
        )
        is None
    )


@pytest.mark.asyncio()
async def test_router_free_paid_tier_with_responses_api():
    """
    Pass list of orgs in 1 model definition,
    expect a unique deployment for each to be created
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["free"],
                },
                "model_info": {"id": "very-cheap-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["paid"],
                },
                "model_info": {"id": "very-expensive-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        # this should pick model with id == very-cheap-model
        response = await router.aresponses(
            model="gpt-4",
            input="Tell me a joke.",
            litellm_metadata={"tags": ["free"]},
            mock_response="Tell me a joke.",
        )

        response_extra_info = response._hidden_params

        assert response_extra_info["model_id"] == "very-cheap-model"

    for _ in range(5):
        # this should pick model with id == very-cheap-model
        response = await router.aresponses(
            model="gpt-4",
            input="Tell me a joke.",
            litellm_metadata={"tags": ["paid"]},
            mock_response="Tell me a joke.",
        )

        response_extra_info = response._hidden_params

        assert response_extra_info["model_id"] == "very-expensive-model"


def test_get_tags_from_request_kwargs_none():
    from litellm.router_strategy.tag_based_routing import _get_tags_from_request_kwargs

    # None request kwargs should safely return empty list
    assert _get_tags_from_request_kwargs(None) == []


def test_get_tags_from_request_kwargs_various_inputs():
    from litellm.router_strategy.tag_based_routing import _get_tags_from_request_kwargs

    # Direct "metadata" path
    assert _get_tags_from_request_kwargs({"metadata": {"tags": ["free"]}}) == ["free"]
    assert _get_tags_from_request_kwargs({"metadata": {"tags": []}}) == []
    assert _get_tags_from_request_kwargs({"metadata": {"tags": None}}) == []
    assert _get_tags_from_request_kwargs({"metadata": {}}) == []
    assert _get_tags_from_request_kwargs({"metadata": None}) == []

    # Indirect via "litellm_params" - metadata inside
    assert _get_tags_from_request_kwargs({"litellm_params": {"metadata": {"tags": ["paid"]}}}) == ["paid"]
    assert _get_tags_from_request_kwargs({"litellm_params": {"metadata": None}}) == []
    assert _get_tags_from_request_kwargs({"litellm_params": {}}) == []

    # Alternate metadata variable name: "litellm_metadata"
    assert _get_tags_from_request_kwargs(
        {"litellm_metadata": {"tags": ["alt"]}},
        metadata_variable_name="litellm_metadata",
    ) == ["alt"]
    assert _get_tags_from_request_kwargs(
        {"litellm_params": {"litellm_metadata": {"tags": ["nested-alt"]}}},
        metadata_variable_name="litellm_metadata",
    ) == ["nested-alt"]

    # No relevant keys present
    assert _get_tags_from_request_kwargs({"foo": "bar"}) == []


# --- _split_tags unit tests ---


def test_split_tags_positive_only():
    from litellm.router_strategy.tag_based_routing import _split_tags

    positive, excluded = _split_tags(["paid", "teamA"])
    assert positive == ["paid", "teamA"]
    assert excluded == []


def test_split_tags_negation_only():
    from litellm.router_strategy.tag_based_routing import _split_tags

    positive, excluded = _split_tags(["!provider:anthropic"])
    assert positive == []
    assert excluded == ["provider:anthropic"]


def test_split_tags_mixed():
    from litellm.router_strategy.tag_based_routing import _split_tags

    positive, excluded = _split_tags(["paid", "!provider:anthropic", "!inference:cerebras"])
    assert positive == ["paid"]
    assert len(excluded) == 2


def test_split_tags_bare_bang_skipped():
    from litellm.router_strategy.tag_based_routing import _split_tags

    # A bare "!" with nothing after it is not a valid negation tag; skip it
    positive, excluded = _split_tags(["paid", "!"])
    assert positive == ["paid"]
    assert excluded == []


def test_split_tags_empty():
    from litellm.router_strategy.tag_based_routing import _split_tags

    positive, excluded = _split_tags([])
    assert positive == []
    assert excluded == []


# --- get_deployments_for_tag negation integration tests ---


@pytest.mark.asyncio()
async def test_negation_excludes_matching_deployments():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic", "model:claude-sonnet-4-6"],
                },
                "model_info": {"id": "anthropic-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:openai", "model:gpt-4o"],
                },
                "model_info": {"id": "openai-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!provider:anthropic"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "openai-model"


@pytest.mark.asyncio()
async def test_negation_multiple_tags_exclude_multiple_providers():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic"],
                },
                "model_info": {"id": "anthropic-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:openai"],
                },
                "model_info": {"id": "openai-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:vertex"],
                },
                "model_info": {"id": "vertex-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!provider:anthropic", "!provider:openai"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "vertex-model"


@pytest.mark.asyncio()
async def test_negation_with_positive_tag():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["paid", "provider:anthropic"],
                },
                "model_info": {"id": "anthropic-paid"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["paid", "provider:openai"],
                },
                "model_info": {"id": "openai-paid"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["free", "provider:openai"],
                },
                "model_info": {"id": "openai-free"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["paid", "!provider:anthropic"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "openai-paid"


@pytest.mark.asyncio()
async def test_negation_all_excluded_raises():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic"],
                },
                "model_info": {"id": "anthropic-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    with pytest.raises(Exception) as exc_info:
        await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!provider:anthropic"]},
            mock_response="hi",
        )

    from litellm.types.router import RouterErrors

    assert RouterErrors.no_deployments_with_tag_routing.value in str(exc_info.value)


@pytest.mark.asyncio()
async def test_negation_ban_only_cannot_escape_default_pool():
    # A ban-only request must not route to tagged deployments outside the default pool.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["default"],
                },
                "model_info": {"id": "default-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["paid"],
                },
                "model_info": {"id": "paid-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    # Sending only "!default" must NOT route to the paid deployment.
    # The base pool for ban-only is the default pool; banning the only
    # default deployment should raise rather than falling through to paid.
    with pytest.raises(Exception) as exc_info:
        await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!default"]},
            mock_response="hi",
        )

    from litellm.types.router import RouterErrors

    assert RouterErrors.no_deployments_with_tag_routing.value in str(exc_info.value)


@pytest.mark.asyncio()
async def test_negation_ban_only_respects_default_pool():
    # A ban-only request stays within the default pool; non-default deployments
    # remain unreachable even when the negation tag is unrelated to the default.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["default"],
                },
                "model_info": {"id": "default-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["paid"],
                },
                "model_info": {"id": "paid-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    # "!paid" bans the paid deployment, but the base pool for ban-only is
    # already restricted to defaults; default-model must still be returned.
    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!paid"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "default-model"


@pytest.mark.asyncio()
async def test_negation_untagged_deployment_kept():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic"],
                },
                "model_info": {"id": "anthropic-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                },
                "model_info": {"id": "untagged-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!provider:anthropic"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "untagged-model"


@pytest.mark.asyncio()
async def test_negation_literal_only_no_partial_match():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic-haiku"],
                },
                "model_info": {"id": "anthropic-haiku-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:openai"],
                },
                "model_info": {"id": "openai-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    # "!provider:anthropic" should NOT match "provider:anthropic-haiku" — exact tag match only
    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!provider:anthropic"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] in (
            "anthropic-haiku-model",
            "openai-model",
        )


@pytest.mark.asyncio()
async def test_negation_regex_pattern_treated_as_literal():
    # "!provider:(anthropic|openai)" looks like a regex but is treated as a literal string.
    # It does NOT exclude deployments tagged "provider:anthropic" or "provider:openai".
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic"],
                },
                "model_info": {"id": "anthropic-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:openai"],
                },
                "model_info": {"id": "openai-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    # The regex-like string matches no deployment tag literally, so all
    # candidates survive and both model IDs are reachable.
    seen_ids = set()
    for _ in range(10):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!provider:(anthropic|openai)"]},
            mock_response="hi",
        )
        seen_ids.add(response._hidden_params["model_id"])

    assert seen_ids == {"anthropic-model", "openai-model"}


@pytest.mark.asyncio()
async def test_positive_tags_unchanged_by_negation():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["free"],
                },
                "model_info": {"id": "free-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["paid"],
                },
                "model_info": {"id": "paid-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["free"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "free-model"


@pytest.mark.asyncio()
async def test_negation_skips_banned_group_and_uses_fallback():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic"],
                },
                "model_info": {"id": "anthropic-primary"},
            },
            {
                "model_name": "fallback",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:openai"],
                },
                "model_info": {"id": "openai-fallback"},
            },
        ],
        fallbacks=[{"primary": ["fallback"]}],
        enable_tag_filtering=True,
    )

    response = await router.acompletion(
        model="primary",
        messages=[{"role": "user", "content": "hi"}],
        metadata={"tags": ["!provider:anthropic"]},
        mock_response="hi",
    )
    assert response._hidden_params["model_id"] == "openai-fallback"


@pytest.mark.asyncio()
async def test_negation_exhausts_entire_fallback_chain():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic"],
                },
                "model_info": {"id": "anthropic-primary"},
            },
            {
                "model_name": "fallback",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic"],
                },
                "model_info": {"id": "anthropic-fallback"},
            },
        ],
        fallbacks=[{"primary": ["fallback"]}],
        enable_tag_filtering=True,
    )

    with pytest.raises(Exception) as exc_info:
        await router.acompletion(
            model="primary",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!provider:anthropic"]},
            mock_response="hi",
        )

    from litellm.types.router import RouterErrors

    assert RouterErrors.no_deployments_with_tag_routing.value in str(exc_info.value)


@pytest.mark.asyncio()
async def test_tag_regex_survives_when_negation_removes_other_deployment():
    # Negation removes a plain-tagged deployment; the surviving tag_regex deployment
    # is still matched by User-Agent and selected.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tag_regex": ["^User-Agent: claude-code\\/"],
                },
                "model_info": {"id": "claude-code-deployment"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic"],
                },
                "model_info": {"id": "anthropic-deployment"},
            },
        ],
        enable_tag_filtering=True,
        tag_filtering_match_any=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!provider:anthropic"], "user_agent": "claude-code/1.2.3"},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "claude-code-deployment"


@pytest.mark.asyncio()
async def test_negation_removes_tag_regex_deployment_falls_to_ban_only():
    # When a negation tag removes the only tag_regex deployment, no regex deployments
    # remain in the candidate pool. has_tag_filter becomes False, ban_only fires,
    # and the remaining plain-tagged deployment is returned via the ban-only path.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tag_regex": ["^User-Agent: claude-code\\/"],
                    "tags": ["group:claude"],
                },
                "model_info": {"id": "claude-code-deployment"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:openai"],
                },
                "model_info": {"id": "openai-deployment"},
            },
        ],
        enable_tag_filtering=True,
        tag_filtering_match_any=True,
    )

    # !group:claude removes the tag_regex deployment from candidates, so no regex
    # deployments remain. The ban-only path fires and returns the openai deployment.
    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!group:claude"], "user_agent": "claude-code/1.2.3"},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "openai-deployment"


@pytest.mark.asyncio()
async def test_request_level_enable_tag_filtering_applies_when_global_off():
    """
    A request carrying enable_tag_filtering=True (set by the proxy from key/team
    router_settings) must activate tag filtering even when the router-level flag
    is off. Without this, a team's "Enable Tag Filtering" toggle saved in the UI
    is silently ignored at request time.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["teamA"],
                },
                "model_info": {"id": "team-a-deployment"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["teamB"],
                },
                "model_info": {"id": "team-b-deployment"},
            },
        ],
        enable_tag_filtering=False,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["teamA"]},
            enable_tag_filtering=True,
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "team-a-deployment"

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["teamB"]},
            enable_tag_filtering=True,
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "team-b-deployment"


@pytest.mark.asyncio()
async def test_request_level_enable_tag_filtering_false_cannot_disable_global():
    """
    A request-level enable_tag_filtering=False must not bypass a router-level
    True: tag filtering can be an operator-level restriction on which
    deployments a caller may reach, so per-request settings may only scope
    down, never escape the global policy.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["teamA"],
                },
                "model_info": {"id": "team-a-deployment"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["teamB"],
                },
                "model_info": {"id": "team-b-deployment"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["teamA"]},
            enable_tag_filtering=False,
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "team-a-deployment"


# ---------------------------------------------------------------------------
# Regression: a SELECTED deployment's own tags must never leak into the routing
# decision on a later attempt.
#
# Upstream (PR #20769) merges the selected deployment's litellm_params["tags"]
# into metadata["tags"] for spend attribution, in _update_kwargs_with_deployment.
# That kwargs dict is REUSED across retries and fallbacks, so on the next attempt
# get_deployments_for_tag would re-read the leaked deployment tag as if the caller
# had sent it — filtering an untagged request down to just the failing provider
# and breaking cross-provider fallback / same-group weighted retry.
#
# The fix snapshots the caller's ORIGINAL request tags once, before any
# per-deployment merge (Router._update_kwargs_before_fallbacks), into
# ORIGINAL_REQUEST_TAGS_KEY; get_deployments_for_tag prefers that snapshot.
# Reverting the snapshot-preference in get_deployments_for_tag makes
# test_deployment_tags_do_not_leak_into_cross_provider_fallback fail.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_deployment_tags_do_not_leak_into_cross_provider_fallback():
    """UNTAGGED request whose tagged primary group fails must fall back across
    providers. Before the fix, the primary deployment's `pin:azure` tag leaked
    into metadata["tags"] and the fallback group (tagged `pin:openai`) was
    rejected as not-a-subset, so the untagged request failed instead of failing
    over. This is the MUTATION-sensitive test."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake",
                    "tags": ["pin:azure"],
                    "mock_response": Exception("simulated azure failure"),
                },
                "model_info": {"id": "azure-dep"},
            },
            {
                "model_name": "secondary",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake",
                    "tags": ["pin:openai"],
                    "mock_response": "OK-FROM-OPENAI",
                },
                "model_info": {"id": "openai-dep"},
            },
        ],
        fallbacks=[{"primary": ["secondary"]}],
        enable_tag_filtering=True,
        tag_filtering_match_any=False,
        num_retries=0,
    )

    # No tags in the request: the pin tags belong to the DEPLOYMENTS, not the
    # caller. The untagged request must be free to fall back to `secondary`.
    response = await router.acompletion(
        model="primary",
        messages=[{"role": "user", "content": "hi"}],
        metadata={},
    )
    assert response._hidden_params["model_id"] == "openai-dep"


@pytest.mark.asyncio()
async def test_deployment_tags_do_not_leak_into_same_group_weighted_retry():
    """UNTAGGED request in a single model group whose weight-forced primary leg
    always fails must retry onto the healthy other-provider leg. Before the fix
    the failing leg's `pin:azure` tag leaked and filtered the retry down to just
    that leg, so the healthy `pin:openai` leg became unreachable."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake",
                    "tags": ["pin:azure"],
                    "weight": 1000,
                    "mock_response": Exception("fail azure"),
                },
                "model_info": {"id": "azure-dep"},
            },
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake",
                    "tags": ["pin:openai"],
                    "weight": 1,
                    "mock_response": "OK-RETRY-OPENAI",
                },
                "model_info": {"id": "openai-dep"},
            },
        ],
        enable_tag_filtering=True,
        tag_filtering_match_any=False,
        num_retries=3,
        routing_strategy="simple-shuffle",
        # Weighted intra-group failover excludes the just-failed deployment on
        # retry; that exclusion re-selects through get_deployments_for_tag, so it
        # only reaches the healthy leg once the leaked tag no longer filters it out.
        enable_weighted_failover=True,
    )

    response = await router.acompletion(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        metadata={},
    )
    assert response._hidden_params["model_id"] == "openai-dep"


@pytest.mark.asyncio()
async def test_genuine_pin_still_enforced_and_fails_loud_when_no_matching_leg():
    """The snapshot must NOT weaken a genuine pin. A request whose ORIGINAL tags
    are `[pin:bedrock]` (as the proxy's pinned route sets pre-routing) must still
    filter strictly to a bedrock leg — and, when no bedrock leg exists, fail loud
    with the tag-routing error rather than silently spilling to another provider."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake",
                    "tags": ["pin:azure"],
                    "mock_response": "should-never-be-reached",
                },
                "model_info": {"id": "azure-dep"},
            },
        ],
        enable_tag_filtering=True,
        tag_filtering_match_any=False,
        num_retries=0,
    )

    with pytest.raises(Exception) as exc_info:
        await router.acompletion(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            # The pin tag is present in metadata pre-routing, exactly as the
            # proxy's pinned handler sets it; the snapshot must capture it so
            # strict subset matching still applies.
            metadata={"tags": ["pin:bedrock"]},
        )

    from litellm.types.router import RouterErrors

    assert RouterErrors.no_deployments_with_tag_routing.value in str(exc_info.value)


# ---------------------------------------------------------------------------
# Codex re-review P1-A (part 2): a client-smuggled ORIGINAL_REQUEST_TAGS_KEY
# must never be trusted as the routing snapshot. Router._update_kwargs_before_
# fallbacks OVERWRITES the snapshot from the trusted live tags on the FIRST
# invocation (setdefault would have honored the spoof). Reverting the overwrite
# back to setdefault makes test_spoofed_original_request_tags_snapshot_is_
# overwritten route to the spoofed pin instead of failing loud.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_spoofed_original_request_tags_snapshot_is_overwritten():
    """A spoofed ``_original_request_tags`` in the caller's metadata must be
    overwritten by the router from the trusted live ``tags`` on the first
    invocation. Live pin is ``pin:bedrock``; the spoof claims ``pin:azure``.
    Routing must honor ``pin:bedrock`` and fail loud — never serve the
    ``pin:azure`` leg the spoof points at."""
    from litellm.router_strategy.tag_based_routing import ORIGINAL_REQUEST_TAGS_KEY

    router = litellm.Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake",
                    "tags": ["pin:azure"],
                    "mock_response": "should-never-be-reached",
                },
                "model_info": {"id": "azure-dep"},
            },
        ],
        enable_tag_filtering=True,
        tag_filtering_match_any=False,
        num_retries=0,
    )

    with pytest.raises(Exception) as exc_info:
        await router.acompletion(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            # Live tags are the server-set pin; the client also smuggled a
            # conflicting pre-merge snapshot pointing at the azure leg.
            metadata={"tags": ["pin:bedrock"], ORIGINAL_REQUEST_TAGS_KEY: ["pin:azure"]},
        )

    from litellm.types.router import RouterErrors

    assert RouterErrors.no_deployments_with_tag_routing.value in str(exc_info.value)


# ---------------------------------------------------------------------------
# Codex re-review P1-B: a hard provider pin (any ``pin:`` tag) must NEVER fall
# through to the ``default``-deployment pool — the fork must not depend on a
# static config guard in another repo to forbid ``default`` tags. get_deployments
# _for_tag skips the default fallback for pinned requests. Reverting the
# ``not disable_default_fallback`` guard makes test_pin_never_falls_back_to_
# default_pool serve the default deployment instead of failing loud.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_pin_never_falls_back_to_default_pool():
    """A ``pin:bedrock`` request with a ``default``-tagged deployment present but
    NO bedrock leg must fail loud — the pin can never be served by the default
    pool."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake",
                    "tags": ["default"],
                    "mock_response": "should-never-be-reached-default",
                },
                "model_info": {"id": "default-model"},
            },
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake",
                    "tags": ["pin:azure"],
                    "mock_response": "should-never-be-reached-azure",
                },
                "model_info": {"id": "azure-dep"},
            },
        ],
        enable_tag_filtering=True,
        tag_filtering_match_any=False,
        num_retries=0,
    )

    with pytest.raises(Exception) as exc_info:
        await router.acompletion(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["pin:bedrock"]},
        )

    from litellm.types.router import RouterErrors

    assert RouterErrors.no_deployments_with_tag_routing.value in str(exc_info.value)


@pytest.mark.asyncio()
async def test_non_pin_tag_still_falls_back_to_default_pool_unchanged():
    """The pin guard is scoped to ``pin:`` tags: a NON-pin tagged request with no
    exact match still falls back to the ``default`` pool (vanilla behavior). This
    guards against the guard over-firing on all tagged requests."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["default"],
                },
                "model_info": {"id": "default-model"},
            },
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["teamA"],
                },
                "model_info": {"id": "team-a"},
            },
        ],
        enable_tag_filtering=True,
        tag_filtering_match_any=False,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["team-does-not-exist"]},
            mock_response="ok",
        )
        assert response._hidden_params["model_id"] == "default-model"


@pytest.mark.asyncio()
async def test_untagged_request_still_uses_default_pool_unchanged():
    """Vanilla behavior preserved: an UNTAGGED request with a ``default``-tagged
    deployment still routes to it (the pin guard never fires without a pin)."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["default"],
                },
                "model_info": {"id": "default-model"},
            },
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["pin:azure"],
                },
                "model_info": {"id": "azure-dep"},
            },
        ],
        enable_tag_filtering=True,
        tag_filtering_match_any=False,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            metadata={},
            mock_response="ok",
        )
        assert response._hidden_params["model_id"] == "default-model"


# ---------------------------------------------------------------------------
# Codex R4 convergence: the pinned routing decision derives SOLELY from the
# trusted PINNED_PROVIDER_ROUTE_KEY signal (the provider from the URL), read by
# get_deployments_for_tag with absolute priority. metadata["tags"] (incl. the
# appended key/team tags) is IGNORED for routing but still carried for spend
# attribution. This structurally closes the key/team-tag 400 (P2) and the
# spoofed-snapshot / forged-tag bypasses (P1): none of them feed the pinned
# routing decision.
#
# Mutation-verify: drop the PINNED_PROVIDER_ROUTE_KEY branch in
# _resolve_request_tags -> test_pinned_signal_routes_despite_team_tags 400s
# (team:x re-enters routing and breaks the subset match) and
# test_pinned_signal_overrides_forged_client_routing routes to / fails on the
# forged pin instead of the trusted one.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_pinned_signal_routes_despite_team_tags():
    """A pinned request whose metadata also carries an appended team tag
    (``metadata["tags"] == ["pin:bedrock", "team:x"]``) must still route to the
    pin leg under strict subset matching — the trusted signal makes the routing
    tag set EXACTLY ``["pin:bedrock"]``, so the team tag never breaks the subset
    match (previously a 400). The team tag remains in ``metadata["tags"]`` for
    attribution."""
    from litellm.router_strategy.tag_based_routing import PINNED_PROVIDER_ROUTE_KEY

    router = litellm.Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake",
                    "tags": ["pin:bedrock"],
                },
                "model_info": {"id": "bedrock-dep"},
            },
        ],
        enable_tag_filtering=True,
        tag_filtering_match_any=False,
        num_retries=0,
    )

    metadata = {"tags": ["pin:bedrock", "team:x"], PINNED_PROVIDER_ROUTE_KEY: "bedrock"}
    response = await router.acompletion(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        metadata=metadata,
        mock_response="ok",
    )
    assert response._hidden_params["model_id"] == "bedrock-dep"
    # Attribution tags untouched: the team tag is still present for spend.
    assert "team:x" in metadata["tags"]


@pytest.mark.asyncio()
async def test_pinned_signal_overrides_forged_client_routing():
    """Every client-controllable routing input points at ``azure``; the trusted
    signal points at ``bedrock``. Routing must honor the signal (bedrock leg),
    proving no client field can redirect a pinned request."""
    from litellm.router_strategy.tag_based_routing import (
        ORIGINAL_REQUEST_TAGS_KEY,
        PINNED_PROVIDER_ROUTE_KEY,
    )

    router = litellm.Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake",
                    "tags": ["pin:bedrock"],
                },
                "model_info": {"id": "bedrock-dep"},
            },
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake",
                    "tags": ["pin:azure"],
                    "mock_response": "should-never-be-reached-azure",
                },
                "model_info": {"id": "azure-dep"},
            },
        ],
        enable_tag_filtering=True,
        tag_filtering_match_any=False,
        num_retries=0,
    )

    response = await router.acompletion(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        metadata={
            "tags": ["pin:azure"],
            ORIGINAL_REQUEST_TAGS_KEY: ["pin:azure"],
            PINNED_PROVIDER_ROUTE_KEY: "bedrock",
        },
        mock_response="ok",
    )
    assert response._hidden_params["model_id"] == "bedrock-dep"


@pytest.mark.asyncio()
async def test_pinned_signal_still_fails_loud_when_no_matching_leg():
    """The trusted signal must not weaken a genuine pin: a ``bedrock`` signal
    with only an ``azure`` leg (and a ``default`` leg present) fails loud — the
    pin still disables the default-pool fallback."""
    from litellm.router_strategy.tag_based_routing import PINNED_PROVIDER_ROUTE_KEY
    from litellm.types.router import RouterErrors

    router = litellm.Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake",
                    "tags": ["default"],
                    "mock_response": "should-never-be-reached-default",
                },
                "model_info": {"id": "default-model"},
            },
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake",
                    "tags": ["pin:azure"],
                    "mock_response": "should-never-be-reached-azure",
                },
                "model_info": {"id": "azure-dep"},
            },
        ],
        enable_tag_filtering=True,
        tag_filtering_match_any=False,
        num_retries=0,
    )

    with pytest.raises(Exception) as exc_info:
        await router.acompletion(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["pin:bedrock"], PINNED_PROVIDER_ROUTE_KEY: "bedrock"},
        )
    assert RouterErrors.no_deployments_with_tag_routing.value in str(exc_info.value)


@pytest.mark.asyncio()
async def test_unified_route_unchanged_without_signal():
    """Requirement 4: with no pinned signal present, tag routing is identical to
    vanilla — a team-tagged request still routes to its team leg by subset."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake",
                    "tags": ["team:x"],
                },
                "model_info": {"id": "team-dep"},
            },
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake",
                    "tags": ["default"],
                },
                "model_info": {"id": "default-dep"},
            },
        ],
        enable_tag_filtering=True,
        tag_filtering_match_any=False,
        num_retries=0,
    )

    response = await router.acompletion(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        metadata={"tags": ["team:x"]},
        mock_response="ok",
    )
    assert response._hidden_params["model_id"] == "team-dep"


# ---------------------------------------------------------------------------
# P1a: a trusted provider pin must be ENFORCED even when enable_tag_filtering is
# off. get_deployments_for_tag reads the pin signal BEFORE the enable_tag_
# filtering early-return, so a pinned request is always restricted to its
# pin:<provider> legs (or fails loud) — never served by an off-provider leg.
#
# Mutation-verify: restore the original early-return
# (`if request_enable_tag_filtering is not True and llm_router.enable_tag_
# filtering is not True: return healthy_deployments`, dropping the pinned_provider
# guard) and test_pin_enforced_with_tag_filtering_disabled would route to the
# off-provider leg / stop failing loud.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_pin_enforced_with_tag_filtering_disabled():
    """enable_tag_filtering=False WITH a trusted pin present: the request must
    still be restricted to the pin:<provider> leg, never the off-provider one."""
    from litellm.router_strategy.tag_based_routing import PINNED_PROVIDER_ROUTE_KEY

    router = litellm.Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake",
                    "tags": ["pin:bedrock"],
                },
                "model_info": {"id": "bedrock-dep"},
            },
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake",
                    "tags": ["pin:openai"],
                    "mock_response": "should-never-be-reached-openai",
                },
                "model_info": {"id": "openai-dep"},
            },
        ],
        enable_tag_filtering=False,  # OFF — the pin must hold anyway
        tag_filtering_match_any=False,
        num_retries=0,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["pin:bedrock"], PINNED_PROVIDER_ROUTE_KEY: "bedrock"},
            mock_response="ok",
        )
        assert response._hidden_params["model_id"] == "bedrock-dep"


@pytest.mark.asyncio()
async def test_pin_fails_loud_with_tag_filtering_disabled_and_no_matching_leg():
    """enable_tag_filtering=False, a trusted bedrock pin, but only an off-provider
    leg present: must FAIL LOUD, never silently serve the off-provider leg."""
    from litellm.router_strategy.tag_based_routing import PINNED_PROVIDER_ROUTE_KEY
    from litellm.types.router import RouterErrors

    router = litellm.Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake",
                    "tags": ["pin:openai"],
                    "mock_response": "should-never-be-reached-openai",
                },
                "model_info": {"id": "openai-dep"},
            },
        ],
        enable_tag_filtering=False,
        tag_filtering_match_any=False,
        num_retries=0,
    )

    with pytest.raises(Exception) as exc_info:
        await router.acompletion(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["pin:bedrock"], PINNED_PROVIDER_ROUTE_KEY: "bedrock"},
        )
    assert RouterErrors.no_deployments_with_tag_routing.value in str(exc_info.value)


def test_pinned_provider_from_kwargs_only_trusts_the_signal():
    from litellm.router_strategy.tag_based_routing import (
        PINNED_PROVIDER_ROUTE_KEY,
        _pinned_provider_from_kwargs,
    )

    assert _pinned_provider_from_kwargs({"metadata": {PINNED_PROVIDER_ROUTE_KEY: "bedrock"}}, "metadata") == "bedrock"
    # A pin: TAG alone (no trusted signal) is NOT the trusted pin signal.
    assert _pinned_provider_from_kwargs({"metadata": {"tags": ["pin:bedrock"]}}, "metadata") is None
    assert _pinned_provider_from_kwargs(None, "metadata") is None
    assert _pinned_provider_from_kwargs({"metadata": None}, "metadata") is None
    assert _pinned_provider_from_kwargs({"metadata": {PINNED_PROVIDER_ROUTE_KEY: ""}}, "metadata") is None
    assert (
        _pinned_provider_from_kwargs({"litellm_metadata": {PINNED_PROVIDER_ROUTE_KEY: "vertex_ai"}}, "litellm_metadata")
        == "vertex_ai"
    )


# ---------------------------------------------------------------------------
# P1b: a client-controlled User-Agent (tag_regex) must NEVER select a deployment
# that lacks the pin tag for a pinned request. _match_deployment disables the
# regex path when pin_enforced, so the pin tag is the SOLE selector.
#
# Mutation-verify: drop the `and not pin_enforced` guard in _match_deployment and
# test_pin_ignores_tag_regex_user_agent_match routes to / is rescued by the regex
# deployment.
# ---------------------------------------------------------------------------


def test_match_deployment_pin_enforced_blocks_regex():
    """Unit: with pin_enforced, a regex-only (no plain tag) deployment whose
    tag_regex matches the User-Agent must NOT match."""
    from litellm.router_strategy.tag_based_routing import _match_deployment

    deployment = {
        "model_name": "m",
        "litellm_params": {"model": "openai/gpt-4o", "tag_regex": ["^User-Agent: claude-code"]},
    }
    # Without the pin, the regex matches (baseline).
    assert (
        _match_deployment(
            deployment=deployment,
            request_tags=["pin:bedrock"],
            header_strings=["User-Agent: claude-code/1.2.3"],
            match_any=True,
            pin_enforced=False,
        )
        is not None
    )
    # With the pin enforced, the regex path is disabled -> no match.
    assert (
        _match_deployment(
            deployment=deployment,
            request_tags=["pin:bedrock"],
            header_strings=["User-Agent: claude-code/1.2.3"],
            match_any=True,
            pin_enforced=True,
        )
        is None
    )


@pytest.mark.asyncio()
async def test_pin_ignores_tag_regex_user_agent_match():
    """A pinned bedrock request whose User-Agent matches an off-provider
    deployment's tag_regex must route to the pin:bedrock leg, NOT the regex leg."""
    from litellm.router_strategy.tag_based_routing import PINNED_PROVIDER_ROUTE_KEY

    router = litellm.Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake",
                    "tag_regex": ["^User-Agent: claude-code\\/"],
                    "mock_response": "should-never-be-reached-regex",
                },
                "model_info": {"id": "regex-dep"},
            },
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake",
                    "tags": ["pin:bedrock"],
                },
                "model_info": {"id": "bedrock-dep"},
            },
        ],
        enable_tag_filtering=True,
        tag_filtering_match_any=True,  # regex would normally be allowed to match
        num_retries=0,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            metadata={
                "tags": ["pin:bedrock"],
                PINNED_PROVIDER_ROUTE_KEY: "bedrock",
                "user_agent": "claude-code/1.2.3",
            },
            mock_response="ok",
        )
        assert response._hidden_params["model_id"] == "bedrock-dep"


@pytest.mark.asyncio()
async def test_pin_with_only_regex_deployment_fails_loud():
    """When the ONLY deployment is a regex (User-Agent) leg lacking the pin tag,
    a pinned request must fail loud — the spoofable User-Agent cannot rescue it."""
    from litellm.router_strategy.tag_based_routing import PINNED_PROVIDER_ROUTE_KEY
    from litellm.types.router import RouterErrors

    router = litellm.Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake",
                    "tag_regex": ["^User-Agent: claude-code\\/"],
                    "mock_response": "should-never-be-reached-regex",
                },
                "model_info": {"id": "regex-dep"},
            },
        ],
        enable_tag_filtering=True,
        tag_filtering_match_any=True,
        num_retries=0,
    )

    with pytest.raises(Exception) as exc_info:
        await router.acompletion(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            metadata={
                "tags": ["pin:bedrock"],
                PINNED_PROVIDER_ROUTE_KEY: "bedrock",
                "user_agent": "claude-code/1.2.3",
            },
        )
    assert RouterErrors.no_deployments_with_tag_routing.value in str(exc_info.value)


# ---------------------------------------------------------------------------
# P2: on a fallback the reused kwargs dict must not ACCUMULATE each attempt's
# deployment tags in metadata["tags"] — the successful SpendLogs row's
# request_tags (read from metadata["tags"]) must reflect ONLY the winning
# deployment's tags + the caller baseline, never a prior failed attempt's tag.
#
# Mutation-verify: revert _update_kwargs_with_deployment to append onto the live
# tags (drop the ORIGINAL_REQUEST_TAGS_KEY baseline rebuild) and the second call
# yields ["pin:azure", "pin:openai"] instead of ["pin:openai"].
# ---------------------------------------------------------------------------


def _minimal_router_for_kwargs_merge() -> "litellm.Router":
    return litellm.Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
                "model_info": {"id": "seed"},
            }
        ],
    )


def test_update_kwargs_with_deployment_replaces_prior_attempt_tag_untagged_caller():
    """UNTAGGED caller, Azure attempt then OpenAI fallback reuse the SAME kwargs:
    the winning attribution tags must be exactly the winning deployment's tag."""
    from litellm.router_strategy.tag_based_routing import ORIGINAL_REQUEST_TAGS_KEY

    router = _minimal_router_for_kwargs_merge()
    azure_dep = {
        "model_name": "m",
        "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake", "tags": ["pin:azure"]},
        "model_info": {"id": "azure-dep"},
    }
    openai_dep = {
        "model_name": "m",
        "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake", "tags": ["pin:openai"]},
        "model_info": {"id": "openai-dep"},
    }
    # Snapshot as _update_kwargs_before_fallbacks would leave it for an untagged caller.
    kwargs = {"metadata": {ORIGINAL_REQUEST_TAGS_KEY: [], "tags": []}}

    router._update_kwargs_with_deployment(deployment=azure_dep, kwargs=kwargs)
    assert kwargs["metadata"]["tags"] == ["pin:azure"]

    # Fallback reuses the SAME kwargs dict; must REPLACE, not accumulate.
    router._update_kwargs_with_deployment(deployment=openai_dep, kwargs=kwargs)
    assert kwargs["metadata"]["tags"] == ["pin:openai"], (
        "winning SpendLogs row would carry the failed attempt's tag (double-attribution)"
    )


def test_update_kwargs_with_deployment_keeps_caller_tags_and_swaps_deployment_tag():
    """Caller/auth tags in the snapshot survive; only the per-deployment tag swaps
    across attempts."""
    from litellm.router_strategy.tag_based_routing import ORIGINAL_REQUEST_TAGS_KEY

    router = _minimal_router_for_kwargs_merge()
    azure_dep = {
        "model_name": "m",
        "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake", "tags": ["pin:azure"]},
        "model_info": {"id": "azure-dep"},
    }
    openai_dep = {
        "model_name": "m",
        "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake", "tags": ["pin:openai"]},
        "model_info": {"id": "openai-dep"},
    }
    kwargs = {"metadata": {ORIGINAL_REQUEST_TAGS_KEY: ["team:x"], "tags": ["team:x"]}}

    router._update_kwargs_with_deployment(deployment=azure_dep, kwargs=kwargs)
    assert kwargs["metadata"]["tags"] == ["team:x", "pin:azure"]

    router._update_kwargs_with_deployment(deployment=openai_dep, kwargs=kwargs)
    assert kwargs["metadata"]["tags"] == ["team:x", "pin:openai"]


@pytest.mark.asyncio()
async def test_fallback_winning_row_request_tags_exclude_failed_deployment_tag():
    """End-to-end: Azure leg fails, OpenAI fallback serves. The successful row's
    request_tags (captured off the standard logging payload) must contain the
    winning pin:openai tag and NOT the failed attempt's pin:azure tag."""
    import asyncio as _asyncio

    from litellm.integrations.custom_logger import CustomLogger

    captured: dict = {}

    class _CaptureLogger(CustomLogger):
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            slo = kwargs.get("standard_logging_object") or {}
            captured["request_tags"] = slo.get("request_tags")

    logger = _CaptureLogger()
    _prev_callbacks = litellm.callbacks
    litellm.callbacks = [logger]
    try:
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "primary",
                    "litellm_params": {
                        "model": "openai/gpt-4o",
                        "api_key": "fake",
                        "tags": ["pin:azure"],
                        "mock_response": Exception("simulated azure failure"),
                    },
                    "model_info": {"id": "azure-dep"},
                },
                {
                    "model_name": "secondary",
                    "litellm_params": {
                        "model": "openai/gpt-4o",
                        "api_key": "fake",
                        "tags": ["pin:openai"],
                        "mock_response": "OK-FROM-OPENAI",
                    },
                    "model_info": {"id": "openai-dep"},
                },
            ],
            fallbacks=[{"primary": ["secondary"]}],
            enable_tag_filtering=True,
            tag_filtering_match_any=False,
            num_retries=0,
        )

        response = await router.acompletion(
            model="primary",
            messages=[{"role": "user", "content": "hi"}],
            metadata={},
        )
        assert response._hidden_params["model_id"] == "openai-dep"

        # Async success logging is scheduled as a task; poll briefly for it.
        for _ in range(40):
            if "request_tags" in captured:
                break
            await _asyncio.sleep(0.05)

        assert "request_tags" in captured, "success logging did not fire"
        request_tags = captured["request_tags"] or []
        assert "pin:openai" in request_tags, request_tags
        assert "pin:azure" not in request_tags, (
            f"failed Azure attempt's tag leaked into the winning row: {request_tags}"
        )
    finally:
        litellm.callbacks = _prev_callbacks
