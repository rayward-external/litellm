import asyncio
import json
from typing import Final, NoReturn
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_utils.cooldown_handlers import mark_advisor_orchestration_failure
from litellm.router_utils.fallback_event_handlers import (
    AttemptedFallbackTargets,
    _trigger_cooldown_for_failed_deployment,
    clear_pre_routing_selection,
    fallback_attempt_key,
    get_fallback_model_group,
    get_pre_routing_selection,
    record_pre_routing_selection,
    run_async_fallback,
)
from litellm.types.router import RouterRateLimitError


class StreamingWrapper:
    def __init__(self):
        self._hidden_params = {"additional_headers": {}}


class FakeRouter:
    fallback_access_check = None

    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        return StreamingWrapper()


class AlwaysFailRouter:
    fallback_access_check = None

    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        raise RuntimeError("fallback model also failed")


@pytest.mark.asyncio
async def test_run_async_fallback_adds_errors_when_opted_in():
    response = await run_async_fallback(
        litellm_router=FakeRouter(),
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
        include_fallback_errors=True,
    )

    additional_headers = response._hidden_params["additional_headers"]
    assert additional_headers["x-litellm-attempted-fallbacks"] == 1
    assert json.loads(additional_headers["x-litellm-fallback-errors"]) == [
        {
            "message": "upstream limited request",
            "type": "RuntimeError",
            "param": None,
            "code": None,
        }
    ]


@pytest.mark.asyncio
async def test_run_async_fallback_omits_errors_without_opt_in():
    response = await run_async_fallback(
        litellm_router=FakeRouter(),
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    additional_headers = response._hidden_params["additional_headers"]
    assert additional_headers["x-litellm-attempted-fallbacks"] == 1
    assert "x-litellm-fallback-errors" not in additional_headers


@pytest.mark.asyncio
async def test_run_async_fallback_raises_when_all_fallbacks_fail():
    with pytest.raises(RuntimeError, match="fallback model also failed"):
        await run_async_fallback(
            litellm_router=AlwaysFailRouter(),
            fallback_model_group=["fallback-model"],
            original_model_group="primary-model",
            original_exception=RuntimeError("original request failed"),
            max_fallbacks=3,
            fallback_depth=0,
            include_fallback_errors=True,
        )


class RecordingRouter:
    fallback_access_check = None

    def __init__(self):
        self.received_kwargs = None

    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        self.received_kwargs = kwargs
        return StreamingWrapper()


@pytest.mark.asyncio
async def test_run_async_fallback_forwards_include_fallback_errors_to_nested_call():
    """A nested fallback (multi-hop) must keep collecting errors, so the opt-in
    flag has to reach the nested async_function_with_fallbacks call."""
    router = RecordingRouter()
    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
        include_fallback_errors=True,
    )

    assert router.received_kwargs.get("include_fallback_errors") is True


@pytest.mark.asyncio
async def test_run_async_fallback_does_not_forward_flag_without_opt_in():
    router = RecordingRouter()
    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    assert "include_fallback_errors" not in router.received_kwargs


@pytest.mark.asyncio
async def test_run_async_fallback_skips_original_model_group():
    response = await run_async_fallback(
        litellm_router=FakeRouter(),
        fallback_model_group=["primary-model", "fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("original failed"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    assert response._hidden_params["additional_headers"]["x-litellm-attempted-fallbacks"] == 1


class AttemptRecordingRouter:
    fallback_access_check = None

    def __init__(self):
        self.attempted_model_groups = []
        self.received_kwargs = None

    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        self.attempted_model_groups.append(kwargs.get("model"))
        self.received_kwargs = kwargs
        return StreamingWrapper()


async def _acreate_batch(*args, **kwargs):
    raise AssertionError("only used for its __name__")


async def _acreate_file(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("only used for its __name__")


async def _acancel_batch(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("only used for its __name__")


async def _acompletion(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("only used for its __name__")


async def _ageneric_api_call_with_fallbacks_helper(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("only used for its __name__")


async def acreate_fine_tuning_job(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("only used for its __name__")


async def aretrieve_fine_tuning_job(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("only used for its __name__")


async def afile_content(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("only used for its __name__")


@pytest.mark.asyncio
async def test_run_async_fallback_keeps_uploaded_file_requests_in_their_model_group():
    """An input_file_id only exists under the credentials of the group it was uploaded
    to, so a cross-group fallback can only fail with the wrong provider's error."""
    router = AttemptRecordingRouter()
    owning_provider_error = RuntimeError("openai connection error")

    with pytest.raises(RuntimeError, match="openai connection error"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=["azure-group"],
            original_model_group="openai-group",
            original_exception=owning_provider_error,
            max_fallbacks=3,
            fallback_depth=0,
            model="openai-group",
            input_file_id="file-owned-by-openai",
            original_function=_acreate_batch,
        )

    assert router.attempted_model_groups == []


@pytest.mark.asyncio
async def test_run_async_fallback_keeps_fine_tuning_requests_in_their_model_group():
    router = AttemptRecordingRouter()

    with pytest.raises(RuntimeError, match="openai connection error"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=["azure-group"],
            original_model_group="openai-group",
            original_exception=RuntimeError("openai connection error"),
            max_fallbacks=3,
            fallback_depth=0,
            model="openai-group",
            training_file="file-owned-by-openai",
            original_function=_ageneric_api_call_with_fallbacks_helper,
            original_generic_function=acreate_fine_tuning_job,
        )

    assert router.attempted_model_groups == []


@pytest.mark.asyncio
async def test_run_async_fallback_allows_same_model_group_retry_for_uploaded_file_requests():
    """Order-based fallbacks stay inside the owning group, so they must still run."""
    router = AttemptRecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=[{"model": "openai-group", "_target_order": 2}],
        original_model_group="openai-group",
        original_exception=RuntimeError("first deployment failed"),
        max_fallbacks=3,
        fallback_depth=0,
        model="openai-group",
        input_file_id="file-owned-by-openai",
        original_function=_acreate_batch,
    )

    assert router.attempted_model_groups == ["openai-group"]


@pytest.mark.asyncio
async def test_run_async_fallback_keeps_file_creation_in_its_model_group():
    """A file created for batches lands in the account of the deployment that stored it,
    and its id is only usable against the model group the caller named. A cross-group
    fallback silently stores the file with the wrong provider."""
    router = AttemptRecordingRouter()

    with pytest.raises(RuntimeError, match="azure connection error"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=["openai-group"],
            original_model_group="azure-group",
            original_exception=RuntimeError("azure connection error"),
            max_fallbacks=3,
            fallback_depth=0,
            model="azure-group",
            original_function=_acreate_file,
        )

    assert router.attempted_model_groups == []


@pytest.mark.asyncio
async def test_run_async_fallback_allows_same_model_group_retry_for_file_creation():
    router = AttemptRecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=[{"model": "azure-group", "_target_order": 2}],
        original_model_group="azure-group",
        original_exception=RuntimeError("first deployment failed"),
        max_fallbacks=3,
        fallback_depth=0,
        model="azure-group",
        original_function=_acreate_file,
    )

    assert router.attempted_model_groups == ["azure-group"]


@pytest.mark.asyncio
async def test_run_async_fallback_still_crosses_model_groups_without_an_uploaded_file():
    router = AttemptRecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["azure-group"],
        original_model_group="openai-group",
        original_exception=RuntimeError("openai connection error"),
        max_fallbacks=3,
        fallback_depth=0,
        model="openai-group",
    )

    assert router.attempted_model_groups == ["azure-group"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resource_key", "handler_kwargs"),
    [
        ("batch_id", {"original_function": _acancel_batch}),
        (
            "file_id",
            {
                "original_function": _ageneric_api_call_with_fallbacks_helper,
                "original_generic_function": afile_content,
            },
        ),
        (
            "fine_tuning_job_id",
            {
                "original_function": _ageneric_api_call_with_fallbacks_helper,
                "original_generic_function": aretrieve_fine_tuning_job,
            },
        ),
    ],
)
async def test_run_async_fallback_keeps_provider_scoped_ids_in_their_model_group(
    resource_key: str, handler_kwargs: dict
):
    """A batch, file, or fine-tuning job id only exists under the credentials of the group
    that issued it, so a cross-group fallback asks a provider about an id it never saw.
    Generic API calls carry the real handler in original_generic_function, so the pin
    must recognize it there too."""
    router = AttemptRecordingRouter()

    with pytest.raises(RuntimeError, match="openai connection error"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=["azure-group"],
            original_model_group="openai-group",
            original_exception=RuntimeError("openai connection error"),
            max_fallbacks=3,
            fallback_depth=0,
            model="openai-group",
            **{resource_key: "owned-by-openai"},
            **handler_kwargs,
        )

    assert router.attempted_model_groups == []


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_key", ["batch_id", "file_id", "fine_tuning_job_id"])
async def test_run_async_fallback_ignores_stray_resource_ids_on_completion_calls(resource_key: str):
    """A caller-supplied top-level field like file_id on a chat completion is application
    data, never a provider resource reference, so it must not cost the request its
    cross-group fallbacks."""
    router = AttemptRecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["azure-group"],
        original_model_group="openai-group",
        original_exception=RuntimeError("openai connection error"),
        max_fallbacks=3,
        fallback_depth=0,
        model="openai-group",
        original_function=_acompletion,
        **{resource_key: "caller-app-data"},
    )

    assert router.attempted_model_groups == ["azure-group"]


@pytest.mark.asyncio
async def test_run_async_fallback_allows_same_model_group_retry_for_batch_cancel():
    router = AttemptRecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=[{"model": "openai-group", "_target_order": 2}],
        original_model_group="openai-group",
        original_exception=RuntimeError("first deployment failed"),
        max_fallbacks=3,
        fallback_depth=0,
        model="openai-group",
        batch_id="owned-by-openai",
        original_function=_acancel_batch,
    )

    assert router.attempted_model_groups == ["openai-group"]


@pytest.mark.asyncio
async def test_run_async_fallback_handles_explicitly_none_metadata():
    """/v1/batches always sets `metadata`, and sets it to None when the caller sent
    none, so setdefault() on it hands back None instead of a dict."""
    router = AttemptRecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["azure-group"],
        original_model_group="openai-group",
        original_exception=RuntimeError("openai connection error"),
        max_fallbacks=3,
        fallback_depth=0,
        model="openai-group",
        metadata=None,
    )

    assert router.received_kwargs["metadata"] == {
        "model_group": "azure-group",
        "attempted_fallbacks": 1,
        "original_model_group": "openai-group",
    }


@pytest.mark.asyncio
async def test_run_async_fallback_records_batch_model_group_outside_provider_metadata():
    """`metadata` on a batch request is forwarded to the provider and stored on the
    batch, so the router's own model_group belongs in litellm_metadata."""
    router = AttemptRecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=[{"model": "openai-group", "_target_order": 2}],
        original_model_group="openai-group",
        original_exception=RuntimeError("first deployment failed"),
        max_fallbacks=3,
        fallback_depth=0,
        model="openai-group",
        input_file_id="file-owned-by-openai",
        metadata={"caller": "nightly-job"},
        litellm_metadata={"model_group": "openai-group"},
        original_function=_acreate_batch,
    )

    assert router.received_kwargs["metadata"] == {"caller": "nightly-job"}
    assert router.received_kwargs["litellm_metadata"]["model_group"] == "openai-group"


class AccessCheckedRouter(AttemptRecordingRouter):
    def __init__(self, allowed_models: frozenset[str]):
        super().__init__()
        self.allowed_models = allowed_models
        self.access_checks = []

    async def fallback_access_check(self, *, model, request_kwargs, llm_router):
        self.access_checks.append((model, request_kwargs["metadata"]["user_api_key"], llm_router is self))
        return model in self.allowed_models


@pytest.mark.asyncio
async def test_run_async_fallback_skips_targets_the_access_check_rejects():
    router = AccessCheckedRouter(allowed_models=frozenset({"allowed-model"}))

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=[
            {"model": "secret-model", "messages": [{"role": "user", "content": "hi"}]},
            "allowed-model",
        ],
        original_model_group="primary-model",
        original_exception=RuntimeError("primary failed"),
        max_fallbacks=3,
        fallback_depth=0,
        model="primary-model",
        metadata={"user_api_key": "hashed"},
    )

    assert router.attempted_model_groups == ["allowed-model"]
    assert router.access_checks == [
        ("secret-model", "hashed", True),
        ("allowed-model", "hashed", True),
    ]


@pytest.mark.asyncio
async def test_run_async_fallback_raises_original_error_when_no_target_is_authorized():
    router = AccessCheckedRouter(allowed_models=frozenset())

    with pytest.raises(RuntimeError, match="primary failed"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=["secret-model", "other-secret-model"],
            original_model_group="primary-model",
            original_exception=RuntimeError("primary failed"),
            max_fallbacks=3,
            fallback_depth=0,
            model="primary-model",
            metadata={"user_api_key": "hashed"},
        )

    assert router.attempted_model_groups == []
    assert [model for model, _, _ in router.access_checks] == ["secret-model", "other-secret-model"]


@pytest.mark.asyncio
async def test_run_async_fallback_does_not_consult_access_check_for_same_model_group_retries():
    router = AccessCheckedRouter(allowed_models=frozenset())

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=[{"model": "primary-model", "_target_order": 2}],
        original_model_group="primary-model",
        original_exception=RuntimeError("first order level failed"),
        max_fallbacks=3,
        fallback_depth=0,
        model="primary-model",
        metadata={"user_api_key": "hashed"},
    )

    assert router.attempted_model_groups == ["primary-model"]
    assert router.access_checks == []


class RecordingFailRouter:
    fallback_access_check = None

    def __init__(self):
        self.attempted_models = []

    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        self.attempted_models.append(kwargs.get("model"))
        raise RuntimeError("fallback model also failed")


@pytest.mark.asyncio
async def test_run_async_fallback_skips_model_group_already_attempted():
    """A fallback graph that loops back on itself must not re-attempt a model group that
    already failed for this request. Every group in a cycle fails identically, so
    revisiting one multiplies the work and the error output without any chance of
    succeeding."""
    router = RecordingFailRouter()

    with pytest.raises(RuntimeError, match="original failed"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=["already-attempted"],
            original_model_group="primary-model",
            original_exception=RuntimeError("original failed"),
            max_fallbacks=3,
            fallback_depth=0,
            attempted_targets=AttemptedFallbackTargets(frozenset({"already-attempted"})),
        )

    assert router.attempted_models == []


@pytest.mark.asyncio
async def test_run_async_fallback_attempts_a_repeated_target_once():
    router = RecordingFailRouter()

    with pytest.raises(RuntimeError, match="fallback model also failed"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=["fallback-model", "fallback-model", "other-model"],
            original_model_group="primary-model",
            original_exception=RuntimeError("original failed"),
            max_fallbacks=5,
            fallback_depth=0,
        )

    assert router.attempted_models == ["fallback-model", "other-model"]


@pytest.mark.asyncio
async def test_run_async_fallback_forwards_attempted_model_groups_to_nested_call():
    """The nested call is where the next hop of the walk decides what to skip, so the
    accumulated set has to reach it, carrying both the group that just failed and the
    target being attempted."""
    router = RecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("original failed"),
        max_fallbacks=3,
        fallback_depth=0,
        attempted_targets=AttemptedFallbackTargets(frozenset({"earlier-model"})),
    )

    assert router.received_kwargs["attempted_targets"].keys == frozenset(
        {"earlier-model", "primary-model", "fallback-model"}
    )


@pytest.mark.asyncio
async def test_run_async_fallback_can_target_the_requested_group_when_a_pre_router_replaced_it():
    """The requested group was never called when a pre-router selected a tier, so a
    tier fallback may legitimately target that originally requested group."""
    router = RecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["requested-model"],
        original_model_group="requested-model",
        original_exception=RuntimeError("selected tier failed"),
        max_fallbacks=3,
        fallback_depth=0,
        model="requested-model",
        metadata={"pre_routing_selected_model": "selected-tier"},
    )

    assert router.received_kwargs["model"] == "requested-model"
    assert router.received_kwargs["attempted_targets"].keys == frozenset({"selected-tier", "requested-model"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entry",
    [
        {"model": "primary-model", "_target_order": 2},
        {"model": "primary-model", "_excluded_deployment_ids": ["dep-1"]},
    ],
)
async def test_run_async_fallback_still_retargets_the_same_group_via_dict_entry(entry):
    """Order-based fallback and weighted intra-group failover both re-target the group that
    just failed, selecting a different set of deployments inside it. Those entries are dicts
    rather than plain names and must survive a guard that skips already-attempted names."""
    router = RecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=[entry],
        original_model_group="primary-model",
        original_exception=RuntimeError("original failed"),
        max_fallbacks=3,
        fallback_depth=0,
        attempted_targets=AttemptedFallbackTargets(frozenset({"primary-model"})),
    )

    assert router.received_kwargs["model"] == "primary-model"


@pytest.mark.asyncio
async def test_run_async_fallback_skips_a_repeated_dict_target():
    """A client-side fallback list names its targets with dicts, and that list is re-walked
    at every level of the recursion, so an entry that carries no request override has to be
    recognised as the same attempt as the bare name."""
    router = RecordingFailRouter()

    with pytest.raises(RuntimeError, match="original failed"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=[{"model": "already-attempted"}],
            original_model_group="primary-model",
            original_exception=RuntimeError("original failed"),
            max_fallbacks=3,
            fallback_depth=0,
            attempted_targets=AttemptedFallbackTargets(frozenset({"already-attempted"})),
        )

    assert router.attempted_models == []


@pytest.mark.asyncio
async def test_run_async_fallback_attempts_a_repeated_dict_target_once():
    router = RecordingFailRouter()
    entry = {"model": "fallback-model", "messages": [{"role": "user", "content": "shorter"}]}

    with pytest.raises(RuntimeError, match="fallback model also failed"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=[entry, entry, {"model": "other-model"}],
            original_model_group="primary-model",
            original_exception=RuntimeError("original failed"),
            max_fallbacks=5,
            fallback_depth=0,
        )

    assert router.attempted_models == ["fallback-model", "other-model"]


@pytest.mark.asyncio
async def test_run_async_fallback_keeps_a_request_override_distinct_from_the_bare_name():
    """The documented use of the client-side form is to retry a group with different request
    params, so an entry carrying an override must survive even when the bare name of that
    same group has already been attempted."""
    router = RecordingFailRouter()

    with pytest.raises(RuntimeError, match="fallback model also failed"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=[{"model": "already-attempted", "messages": [{"role": "user", "content": "shorter"}]}],
            original_model_group="primary-model",
            original_exception=RuntimeError("original failed"),
            max_fallbacks=3,
            fallback_depth=0,
            attempted_targets=AttemptedFallbackTargets(frozenset({"already-attempted"})),
        )

    assert router.attempted_models == ["already-attempted"]


@pytest.mark.parametrize(
    "target, expected",
    [
        ("group-a", "group-a"),
        ({"model": "group-a"}, "group-a"),
        (None, None),
        (["group-a"], None),
    ],
)
def test_fallback_attempt_key_identity(target, expected):
    """A bare name and a `{"model": name}` entry are the same attempt. A shape with no
    usable identity returns None and is never skipped, so an unrecognised entry keeps
    today's behaviour rather than being silently dropped."""
    assert fallback_attempt_key(target) == expected


def test_fallback_attempt_key_gives_a_param_only_entry_its_own_identity():
    """An entry with no `model` re-targets the group currently being attempted with
    different request params, so it is a distinct attempt and still needs an identity."""
    key = fallback_attempt_key({"messages": [{"role": "user", "content": "shorter"}]})

    assert key is not None
    assert key != fallback_attempt_key({"messages": [{"role": "user", "content": "other"}]})


def test_fallback_attempt_key_separates_overrides_from_the_bare_name():
    bare = fallback_attempt_key("group-a")
    override = fallback_attempt_key({"model": "group-a", "messages": [{"role": "user", "content": "x"}]})
    other_override = fallback_attempt_key({"model": "group-a", "messages": [{"role": "user", "content": "y"}]})
    order_retarget = fallback_attempt_key({"model": "group-a", "_target_order": 2})

    assert len({bare, override, other_override, order_retarget}) == 4


def test_fallback_attempt_key_is_stable_across_key_order():
    assert fallback_attempt_key({"model": "group-a", "_target_order": 2}) == fallback_attempt_key(
        {"_target_order": 2, "model": "group-a"}
    )


def test_get_fallback_model_group_does_not_mutate_fallbacks():
    """A string fallback must be resolved without mutating the caller's
    fallbacks list, which is the live router config shared across requests."""
    fallbacks = [{"gpt-3.5-turbo": ["claude-3-haiku"]}, "gpt-4o-mini"]

    fallback_model_group, _ = get_fallback_model_group(fallbacks=fallbacks, model_group="unmatched-model")

    assert fallback_model_group == ["gpt-4o-mini"]
    assert fallbacks == [{"gpt-3.5-turbo": ["claude-3-haiku"]}, "gpt-4o-mini"]


class TestTriggerCooldownForFailedDeployment:
    def test_calls_set_cooldown_deployments_with_stamped_deployment_id(self):
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        exc.failed_deployment_id = "fallback-deployment"

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(litellm_router=mock_router, kwargs={}, exception=exc)

            mock_set_cooldown.assert_called_once()
            call_kwargs = mock_set_cooldown.call_args[1]
            assert call_kwargs["deployment"] == "fallback-deployment"
            assert call_kwargs["original_exception"] is exc

    def test_does_not_trust_caller_supplied_metadata_bucket(self):
        """A metadata bucket can't reliably be told apart from a caller-supplied
        one without knowing this call's function_name, so a client with
        permission to set metadata must not be able to get an arbitrary
        deployment cooled down by forging a deployment_model_name marker."""
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        kwargs = {
            "metadata": {
                "model_info": {"id": "attacker-chosen-deployment"},
                "deployment_model_name": "gpt-4",
            }
        }

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(litellm_router=mock_router, kwargs=kwargs, exception=exc)

            mock_set_cooldown.assert_not_called()

    def test_increments_failure_counter_before_cooldown_check(self):
        """The fallback path must feed the same per-minute failure counter the
        primary path uses, or repeated fallback failures never accumulate
        toward the default percent-fail-rate cooldown threshold."""
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        exc.failed_deployment_id = "fallback-deployment"

        with (
            patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown,
            patch(
                "litellm.router_utils.fallback_event_handlers.increment_deployment_failures_for_current_minute"
            ) as mock_increment,
        ):
            _trigger_cooldown_for_failed_deployment(litellm_router=mock_router, kwargs={}, exception=exc)

            mock_increment.assert_called_once_with(
                litellm_router_instance=mock_router, deployment_id="fallback-deployment"
            )
            mock_set_cooldown.assert_called_once()

    def test_no_op_when_deployment_id_missing(self):
        mock_router = MagicMock()

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(
                litellm_router=mock_router, kwargs={}, exception=RuntimeError("no metadata")
            )

            mock_set_cooldown.assert_not_called()

    def test_skipped_for_advisor_orchestration_failure(self):
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        exc.failed_deployment_id = "fallback-deployment"
        mark_advisor_orchestration_failure(exc)

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(litellm_router=mock_router, kwargs={}, exception=exc)

            mock_set_cooldown.assert_not_called()

    def test_uses_deployment_litellm_params_cooldown_time_override(self):
        mock_router = MagicMock()
        mock_router.cooldown_time = 300.0
        mock_router.get_model_info.return_value = {"litellm_params": {"cooldown_time": 30.0}}

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        exc.failed_deployment_id = "fallback-deployment"

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(litellm_router=mock_router, kwargs={}, exception=exc)

            call_kwargs = mock_set_cooldown.call_args[1]
            assert call_kwargs["time_to_cooldown"] == 30.0

    def test_uses_response_header_when_no_deployment_config(self):
        """Precedence must match Router.deployment_callback_on_failure's primary
        path: deployment config, then the response's Retry-After header, then the
        router default."""
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = {"litellm_params": {}}

        exc = RuntimeError("upstream error")
        exc.failed_deployment_id = "fallback-deployment"
        exc.litellm_response_headers = httpx.Headers({"retry-after": "45"})

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(litellm_router=mock_router, kwargs={}, exception=exc)

            call_kwargs = mock_set_cooldown.call_args[1]
            assert call_kwargs["time_to_cooldown"] == 45

    def test_silently_catches_exceptions(self):
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = RuntimeError("upstream error")
        exc.failed_deployment_id = "fallback-deployment"

        with patch(
            "litellm.router_utils.fallback_event_handlers._set_cooldown_deployments",
            side_effect=RuntimeError("cooldown error"),
        ):
            _trigger_cooldown_for_failed_deployment(litellm_router=mock_router, kwargs={}, exception=exc)

    def test_skips_request_scoped_404_on_generic_api_call(self):
        """A generic API call (files/batches/threads/rerank/...) forwards a caller-supplied
        resource id, so a 404 there means "that id doesn't exist", not "this deployment is
        unhealthy". Without this guard, a single bad id would 404 every deployment in the
        fallback chain and cool all of them down from one request."""
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.NotFoundError("not found", "openai", "gpt-4")
        exc.failed_deployment_id = "fallback-deployment"

        with (
            patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown,
            patch(
                "litellm.router_utils.fallback_event_handlers.increment_deployment_failures_for_current_minute"
            ) as mock_increment,
        ):
            _trigger_cooldown_for_failed_deployment(
                litellm_router=mock_router,
                kwargs={"original_generic_function": MagicMock()},
                exception=exc,
            )

            mock_set_cooldown.assert_not_called()
            mock_increment.assert_not_called()

    def test_still_cools_down_404_outside_generic_api_call(self):
        """The request-scoped-404 guard is scoped to generic API calls only: a 404 on a
        regular completion fallback (no original_generic_function in kwargs) must still
        cool down the deployment as before."""
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.NotFoundError("not found", "openai", "gpt-4")
        exc.failed_deployment_id = "fallback-deployment"

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(litellm_router=mock_router, kwargs={}, exception=exc)

            mock_set_cooldown.assert_called_once()

    def test_skips_client_side_timeout_408(self):
        """The proxy's x-litellm-timeout header lets a caller set an arbitrarily short
        timeout, which litellm.Timeout reports as status 408 regardless of the
        deployment's actual health. Without this guard, a caller could force a 408 on
        every deployment in the fallback chain from a single request."""
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.Timeout(message="timeout", model="gpt-4", llm_provider="openai")
        exc.failed_deployment_id = "fallback-deployment"

        with (
            patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown,
            patch(
                "litellm.router_utils.fallback_event_handlers.increment_deployment_failures_for_current_minute"
            ) as mock_increment,
        ):
            _trigger_cooldown_for_failed_deployment(
                litellm_router=mock_router,
                kwargs={"client_side_timeout": True},
                exception=exc,
            )

            mock_set_cooldown.assert_not_called()
            mock_increment.assert_not_called()

    def test_still_cools_down_408_without_client_side_timeout_flag(self):
        """The client-side-timeout guard is scoped to caller-supplied timeouts only: a
        408 that did not come from x-litellm-timeout (no client_side_timeout in kwargs)
        must still cool down the deployment as before."""
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.Timeout(message="timeout", model="gpt-4", llm_provider="openai")
        exc.failed_deployment_id = "fallback-deployment"

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(litellm_router=mock_router, kwargs={}, exception=exc)

            mock_set_cooldown.assert_called_once()


class TestRunAsyncFallbackTriggersCooldown:
    class RouterWithLoggingKwarg:
        fallback_access_check = None

        def __init__(self):
            self.cooldown_time = 60.0

        def log_retry(self, kwargs, e):
            return kwargs

        def get_model_info(self, id):
            return None

        async def async_function_with_fallbacks(self, *args, **kwargs):
            raise RuntimeError("fallback model also failed")

    def _logging_obj(self, has_logged_async_failure: bool) -> MagicMock:
        logging_obj = MagicMock()
        logging_obj.model_call_details = {"has_logged_async_failure": has_logged_async_failure}
        return logging_obj

    @pytest.mark.asyncio
    async def test_triggers_cooldown_when_has_logged_async_failure_is_true(self):
        with patch(
            "litellm.router_utils.fallback_event_handlers._trigger_cooldown_for_failed_deployment"
        ) as mock_trigger:
            with pytest.raises(RuntimeError, match="fallback model also failed"):
                await run_async_fallback(
                    litellm_router=self.RouterWithLoggingKwarg(),
                    fallback_model_group=["fallback-model"],
                    original_model_group="primary-model",
                    original_exception=RuntimeError("original request failed"),
                    max_fallbacks=3,
                    fallback_depth=0,
                    litellm_logging_obj=self._logging_obj(has_logged_async_failure=True),
                )

            mock_trigger.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_trigger_cooldown_when_has_logged_async_failure_is_false(self):
        """This is the exact dead-code scenario the bug fix addresses: before it,
        the normal failure callback runs for the first attempt in a fallback chain
        (has_logged_async_failure is still False at that point), so no explicit
        trigger is needed there."""
        with patch(
            "litellm.router_utils.fallback_event_handlers._trigger_cooldown_for_failed_deployment"
        ) as mock_trigger:
            with pytest.raises(RuntimeError, match="fallback model also failed"):
                await run_async_fallback(
                    litellm_router=self.RouterWithLoggingKwarg(),
                    fallback_model_group=["fallback-model"],
                    original_model_group="primary-model",
                    original_exception=RuntimeError("original request failed"),
                    max_fallbacks=3,
                    fallback_depth=0,
                    litellm_logging_obj=self._logging_obj(has_logged_async_failure=False),
                )

            mock_trigger.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_trigger_cooldown_when_no_logging_obj_present(self):
        with patch(
            "litellm.router_utils.fallback_event_handlers._trigger_cooldown_for_failed_deployment"
        ) as mock_trigger:
            with pytest.raises(RuntimeError, match="fallback model also failed"):
                await run_async_fallback(
                    litellm_router=self.RouterWithLoggingKwarg(),
                    fallback_model_group=["fallback-model"],
                    original_model_group="primary-model",
                    original_exception=RuntimeError("original request failed"),
                    max_fallbacks=3,
                    fallback_depth=0,
                )

            mock_trigger.assert_not_called()


@pytest.mark.asyncio
async def test_run_async_fallback_stamps_fallback_info_into_metadata():
    """Spend logs are built from the request metadata of the nested call, so the
    fallback signal has to be stamped there before recursing."""
    router = RecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("original failed"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    metadata = router.received_kwargs["metadata"]
    assert metadata["attempted_fallbacks"] == 1
    assert metadata["original_model_group"] == "primary-model"
    assert metadata["model_group"] == "fallback-model"


@pytest.mark.asyncio
async def test_run_async_fallback_preserves_original_model_group_on_nested_fallback():
    """A second-level fallback receives the first fallback target as its
    original_model_group argument, so the first-stamped value must survive the hop."""
    router = RecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["second-fallback"],
        original_model_group="first-fallback",
        original_exception=RuntimeError("first fallback failed"),
        max_fallbacks=3,
        fallback_depth=1,
        metadata={"attempted_fallbacks": 1, "original_model_group": "primary-model"},
    )

    metadata = router.received_kwargs["metadata"]
    assert metadata["attempted_fallbacks"] == 2
    assert metadata["original_model_group"] == "primary-model"


class TestPreRoutingSelectionCarriesToFallbacks:
    """#38832: a complexity/auto router picks a tier behind the router name, but fallback
    lookup kept using the router name, so the tier's configured chain never ran."""

    def test_selection_is_recorded_in_the_metadata_bucket(self):
        kwargs = {"model": "smart-router", "metadata": {}}
        record_pre_routing_selection(kwargs, "tier1")
        assert kwargs["metadata"]["pre_routing_selected_model"] == "tier1"
        assert get_pre_routing_selection(kwargs) == "tier1"

    def test_selection_is_recorded_in_the_litellm_metadata_bucket(self):
        kwargs = {"model": "smart-router", "litellm_metadata": {}}
        record_pre_routing_selection(kwargs, "tier2")
        assert get_pre_routing_selection(kwargs) == "tier2"

    def test_a_bucket_survives_the_kwargs_copy_that_fallbacks_run_on(self):
        """The bucket is shared by reference, which is the whole reason this works."""
        outer = {"model": "smart-router", "metadata": {}}
        inner = {**outer}
        record_pre_routing_selection(inner, "tier1")
        assert get_pre_routing_selection(outer) == "tier1"

    def test_no_selection_reads_as_none(self):
        assert get_pre_routing_selection({"model": "smart-router", "metadata": {}}) is None
        assert get_pre_routing_selection({"model": "smart-router"}) is None

    def test_missing_kwargs_is_a_no_op(self):
        """A caller with no kwargs must not raise, and must not leak the selection anywhere."""
        record_pre_routing_selection(None, "tier1")

        assert get_pre_routing_selection({}) is None

    def test_a_non_dict_bucket_is_ignored(self):
        kwargs = {"model": "smart-router", "metadata": "not-a-dict"}
        record_pre_routing_selection(kwargs, "tier1")
        assert get_pre_routing_selection(kwargs) is None

    def test_fallbacks_resolve_against_the_selected_tier(self):
        """The lookup the router performs, keyed on the tier rather than the router name."""
        fallbacks = [{"tier1": ["backup-a", "backup-b"]}, {"tier2": ["backup-c"]}]
        assert get_fallback_model_group(fallbacks=fallbacks, model_group="tier1")[0] == ["backup-a", "backup-b"]
        assert get_fallback_model_group(fallbacks=fallbacks, model_group="smart-router")[0] is None


class TestPreRoutingSelectionIsPerHop:
    """#38832 review: the buckets also carry whatever the caller sent, and a fallback hop
    inherits the previous hop's tier, so a hop must start without a selection."""

    def test_a_caller_supplied_selection_is_dropped(self):
        kwargs = {"model": "plain", "metadata": {"pre_routing_selected_model": "tier1"}}

        clear_pre_routing_selection(kwargs)

        assert get_pre_routing_selection(kwargs) is None
        assert "pre_routing_selected_model" not in kwargs["metadata"]

    def test_both_buckets_are_cleared(self):
        kwargs = {
            "metadata": {"pre_routing_selected_model": "tier1"},
            "litellm_metadata": {"pre_routing_selected_model": "tier2"},
        }

        clear_pre_routing_selection(kwargs)

        assert get_pre_routing_selection(kwargs) is None

    def test_the_rest_of_the_bucket_is_left_alone(self):
        kwargs = {"metadata": {"pre_routing_selected_model": "tier1", "tags": ["a"]}}

        clear_pre_routing_selection(kwargs)

        assert kwargs["metadata"] == {"tags": ["a"]}

    def test_clearing_is_a_no_op_without_a_usable_bucket(self):
        kwargs = {"model": "plain", "metadata": "not-a-dict"}

        clear_pre_routing_selection(None)
        clear_pre_routing_selection(kwargs)

        assert kwargs == {"model": "plain", "metadata": "not-a-dict"}

    def test_a_selection_recorded_after_clearing_is_kept(self):
        """Clearing runs before routing, so the hook's own write must survive it."""
        kwargs = {"model": "smart-router", "metadata": {"pre_routing_selected_model": "stale"}}

        clear_pre_routing_selection(kwargs)
        record_pre_routing_selection(kwargs, "tier1")

        assert get_pre_routing_selection(kwargs) == "tier1"


class TestOrderedFallbackLookupGroups:
    def test_tier_first_then_requested_group_deduped(self):
        from litellm.router_utils.fallback_event_handlers import (
            PRE_ROUTING_SELECTED_MODEL_KEY,
            fallback_lookup_groups,
        )

        kwargs = {"litellm_metadata": {PRE_ROUTING_SELECTED_MODEL_KEY: "tier1"}}
        assert fallback_lookup_groups(kwargs, "smart-router") == ("tier1", "smart-router")
        assert fallback_lookup_groups(kwargs, "tier1") == ("tier1",)
        assert fallback_lookup_groups({}, "smart-router") == ("smart-router",)
        assert fallback_lookup_groups({}, None) == ()

    def test_session_remap_keeps_the_bound_router_between_tier_and_requested_group(self):
        from litellm.router_utils.fallback_event_handlers import (
            PRE_ROUTING_SELECTED_MODEL_KEY,
            fallback_lookup_groups,
        )

        kwargs = {
            "litellm_metadata": {
                PRE_ROUTING_SELECTED_MODEL_KEY: "tier1",
                "model_group": "smart-router",
            }
        }

        assert fallback_lookup_groups(kwargs, "requested-model") == (
            "tier1",
            "smart-router",
            "requested-model",
        )
        assert fallback_lookup_groups({"metadata": {"model_group": []}}, "requested-model") == ("requested-model",)

    def test_first_resolving_group_wins_and_generic_idx_survives_a_miss(self):
        from litellm.router_utils.fallback_event_handlers import (
            get_fallback_model_group_for_lookup_groups,
        )

        fallbacks = [{"tier1": ["backup-a"]}, {"smart-router": ["backup-b"]}, {"*": ["backup-c"]}]
        assert get_fallback_model_group_for_lookup_groups(fallbacks, ("tier1", "smart-router")) == (["backup-a"], None)
        assert get_fallback_model_group_for_lookup_groups(fallbacks, ("tier9", "smart-router")) == (["backup-b"], None)
        assert get_fallback_model_group_for_lookup_groups(fallbacks, ("tier9", "no-such")) == (["backup-c"], 2)
        assert get_fallback_model_group_for_lookup_groups([{"tier1": ["backup-a"]}], ("no", "nope")) == (None, None)


class _SyntheticUpstream400(Exception):
    """Deep-copyable stand-in for a provider's deterministic 400."""

    status_code = 400


class _SyntheticUpstream500(Exception):
    """Deep-copyable stand-in for a provider's 500."""

    status_code = 500


class _SyntheticUpstream429(Exception):
    """Deep-copyable stand-in for a provider's genuine rate limit."""

    status_code = 429


class _SyntheticUpstream408(Exception):
    """Deep-copyable stand-in for a provider timeout, which the retry policy treats as transient."""

    status_code = 408


class TestOrderLadderPreservesUpstreamErrorForPinnedRequests:
    """A provider-pinned request on a pooled group must surface the provider's real
    error, not the selection error of an order level the pin already emptied.

    The order ladder builds its levels from the FULL model group, while selection
    applies tag filtering first. So a request pinned to the provider that owns
    orders 1 and 2 is still handed order 3 as a fallback target -- an order whose
    only deployment carries a different pin and was removed before the order
    filter ran. That hop raises RouterRateLimitError ("No deployments available"),
    and run_async_fallback used to raise THAT, turning a deterministic upstream 400
    into a retryable 429 and destroying the provider's message.
    """

    UPSTREAM_MESSAGE = "upstream rejected parameter 'reasoning_effort' (synthetic-400-marker)"
    SPILL_MESSAGE = "spill deployment exploded (synthetic-500-marker)"

    @staticmethod
    def _deployment(dep_id: str, *, order: int, tags: list[str], mock_response: object) -> dict:
        return {
            "model_name": "pooled-model",
            "litellm_params": {
                "model": f"openai/{dep_id}",
                "api_key": "synthetic-key",
                "mock_response": mock_response,
                "order": order,
                "tags": tags,
            },
            "model_info": {"id": dep_id},
        }

    @classmethod
    def _router(cls, *, spill_tags: list[str], spill_mock_response: object) -> litellm.Router:
        # Router deep-copies litellm_params, and litellm's own exception classes
        # cannot be reconstructed by copy.deepcopy, so the upstream failures are plain
        # exceptions carrying a status_code: the mock path maps them to the matching
        # litellm exception type (400 -> BadRequestError, 500 -> InternalServerError)
        # with the message preserved.
        return litellm.Router(
            model_list=[
                cls._deployment(
                    "primary-a", order=1, tags=["pin:azure"], mock_response=_SyntheticUpstream400(cls.UPSTREAM_MESSAGE)
                ),
                cls._deployment(
                    "primary-b", order=2, tags=["pin:azure"], mock_response=_SyntheticUpstream400(cls.UPSTREAM_MESSAGE)
                ),
                cls._deployment("spill", order=3, tags=spill_tags, mock_response=spill_mock_response),
            ],
            enable_tag_filtering=True,
            tag_filtering_match_any=False,
            num_retries=0,
        )

    @pytest.mark.asyncio
    async def test_pinned_request_surfaces_the_upstream_400_not_a_selection_429(self):
        router = self._router(
            spill_tags=["pin:openai"],
            spill_mock_response="spill must never be selected by a pin:azure request",
        )

        with pytest.raises(litellm.BadRequestError) as excinfo:
            await router.acompletion(
                model="pooled-model",
                messages=[{"role": "user", "content": "hi"}],
                metadata={"tags": ["pin:azure"]},
            )

        assert self.UPSTREAM_MESSAGE in str(excinfo.value)
        assert excinfo.value.status_code == 400
        assert not isinstance(excinfo.value, litellm.RateLimitError)

    @pytest.mark.asyncio
    async def test_ladder_that_exhausts_on_a_real_error_still_surfaces_the_last_hop(self):
        """Inverse guard: the swap applies ONLY when the final error is the selection
        error. When order 3 is selectable and fails with its own error, that error is
        what the caller sees -- the original 400 is not smuggled back in."""
        router = self._router(
            spill_tags=["pin:azure"],
            spill_mock_response=_SyntheticUpstream500(self.SPILL_MESSAGE),
        )

        with pytest.raises(litellm.InternalServerError) as excinfo:
            await router.acompletion(
                model="pooled-model",
                messages=[{"role": "user", "content": "hi"}],
                metadata={"tags": ["pin:azure"]},
            )

        assert self.SPILL_MESSAGE in str(excinfo.value)
        assert self.UPSTREAM_MESSAGE not in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_ladder_cooled_down_by_real_429s_still_surfaces_the_rate_limit(self):
        """The swap must not fire when the selection error is GENUINE.

        RouterRateLimitError has two causes: the pin emptied the target order
        (structural, cooldown_list=[]), or every deployment at that order is cooling
        down from real upstream 429s (cooldown_list non-empty) and would serve the
        request in cooldown_time seconds. Reporting the second as the original 400
        would tell a client with 429-backoff that the request is permanently bad.
        """
        router: Final = litellm.Router(
            model_list=[
                self._deployment(
                    "primary-a", order=1, tags=["pin:azure"], mock_response=_SyntheticUpstream400(self.UPSTREAM_MESSAGE)
                ),
                self._deployment(
                    "secondary-1", order=2, tags=["pin:azure"], mock_response=_SyntheticUpstream429("rate limited")
                ),
                self._deployment(
                    "secondary-2", order=2, tags=["pin:azure"], mock_response=_SyntheticUpstream429("rate limited")
                ),
            ],
            enable_tag_filtering=True,
            tag_filtering_match_any=False,
            num_retries=0,
            cooldown_time=30,
        )
        messages: Final = [{"role": "user", "content": "hi"}]
        metadata: Final = {"tags": ["pin:azure"]}

        # Warm-up: each pinned request 400s on order 1, ladders to order 2, and meets a
        # genuine 429 there, which puts the order-2 deployments into cooldown.
        for _ in range(6):
            with pytest.raises((litellm.BadRequestError, litellm.RateLimitError, RouterRateLimitError)):
                await router.acompletion(model="pooled-model", messages=messages, metadata=metadata)
        await asyncio.sleep(0.2)

        with pytest.raises(RouterRateLimitError) as excinfo:
            await router.acompletion(model="pooled-model", messages=messages, metadata=metadata)

        assert excinfo.value.cooldown_list, "order-2 deployments were expected to be in cooldown"
        assert self.UPSTREAM_MESSAGE not in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_unrelated_cooldown_elsewhere_does_not_suppress_the_fix(self):
        """cooldown_list is router-wide, not target-order-scoped.

        A deployment in a DIFFERENT model group cooling down from real 429s must not
        make the pinned request's structural case look like a genuine rate limit.
        """
        router: Final = litellm.Router(
            model_list=[
                self._deployment(
                    "primary-a", order=1, tags=["pin:azure"], mock_response=_SyntheticUpstream400(self.UPSTREAM_MESSAGE)
                ),
                self._deployment(
                    "primary-b", order=2, tags=["pin:azure"], mock_response=_SyntheticUpstream400(self.UPSTREAM_MESSAGE)
                ),
                self._deployment("spill", order=3, tags=["pin:openai"], mock_response="never selected by pin:azure"),
                {
                    "model_name": "other-model",
                    "litellm_params": {
                        "model": "openai/other-1",
                        "api_key": "synthetic-key",
                        "mock_response": _SyntheticUpstream429("rate limited"),
                    },
                    "model_info": {"id": "other-1"},
                },
                {
                    "model_name": "other-model",
                    "litellm_params": {
                        "model": "openai/other-2",
                        "api_key": "synthetic-key",
                        "mock_response": _SyntheticUpstream429("rate limited"),
                    },
                    "model_info": {"id": "other-2"},
                },
            ],
            enable_tag_filtering=True,
            tag_filtering_match_any=False,
            num_retries=0,
            cooldown_time=30,
        )
        messages: Final = [{"role": "user", "content": "hi"}]
        for _ in range(6):
            with pytest.raises((litellm.BadRequestError, litellm.RateLimitError, RouterRateLimitError)):
                await router.acompletion(model="other-model", messages=messages)
        await asyncio.sleep(0.2)

        with pytest.raises(litellm.BadRequestError) as excinfo:
            await router.acompletion(model="pooled-model", messages=messages, metadata={"tags": ["pin:azure"]})

        assert self.UPSTREAM_MESSAGE in str(excinfo.value)
        assert not isinstance(excinfo.value, litellm.RateLimitError)

    @pytest.mark.asyncio
    async def test_callback_emptied_order_is_not_the_structural_case(self):
        """An order emptied by a filter callback, not by the pin, keeps the selection error.

        The order-2 deployment carries the request's own pin, so it survives the tag
        filter; something else removed it. That is not the case this fix is for.
        """

        class _DropOrder2(CustomLogger):
            async def async_filter_deployments(
                self, model, healthy_deployments, messages, request_kwargs=None, parent_otel_span=None
            ):
                return [d for d in healthy_deployments if d.get("litellm_params", {}).get("order") != 2]

        router: Final = litellm.Router(
            model_list=[
                self._deployment(
                    "primary-a", order=1, tags=["pin:azure"], mock_response=_SyntheticUpstream400(self.UPSTREAM_MESSAGE)
                ),
                self._deployment("primary-b", order=2, tags=["pin:azure"], mock_response="would succeed if selectable"),
            ],
            enable_tag_filtering=True,
            tag_filtering_match_any=False,
            num_retries=0,
        )
        drop_order_2: Final = _DropOrder2()
        litellm.callbacks.append(drop_order_2)
        try:
            with pytest.raises(RouterRateLimitError) as excinfo:
                await router.acompletion(
                    model="pooled-model",
                    messages=[{"role": "user", "content": "hi"}],
                    metadata={"tags": ["pin:azure"]},
                )
            assert self.UPSTREAM_MESSAGE not in str(excinfo.value)
        finally:
            litellm.callbacks.remove(drop_order_2)

    @pytest.mark.asyncio
    async def test_team_scope_is_applied_before_the_tag_probe(self):
        """Selection removes other teams' deployments before tag filtering; so must the probe.

        The only order-2 deployment carries the request's pin but belongs to another
        team, so for this request the level was structurally empty. A probe that
        checks tags alone sees a match there and wrongly keeps the selection 429.
        """
        router: Final = litellm.Router(
            model_list=[
                {
                    "model_name": "pooled-model",
                    "litellm_params": {
                        "model": "openai/primary-a",
                        "api_key": "synthetic-key",
                        "mock_response": _SyntheticUpstream400(self.UPSTREAM_MESSAGE),
                        "order": 1,
                        "tags": ["pin:azure"],
                    },
                    "model_info": {"id": "primary-a", "team_id": "team-a"},
                },
                {
                    "model_name": "pooled-model",
                    "litellm_params": {
                        "model": "openai/other-team",
                        "api_key": "synthetic-key",
                        "mock_response": "another team's deployment must never serve team-a",
                        "order": 2,
                        "tags": ["pin:azure"],
                    },
                    "model_info": {"id": "other-team", "team_id": "team-b"},
                },
            ],
            enable_tag_filtering=True,
            tag_filtering_match_any=False,
            num_retries=0,
        )

        with pytest.raises(litellm.BadRequestError) as excinfo:
            await router.acompletion(
                model="pooled-model",
                messages=[{"role": "user", "content": "hi"}],
                metadata={"tags": ["pin:azure"], "user_api_key_team_id": "team-a"},
            )

        assert self.UPSTREAM_MESSAGE in str(excinfo.value)
        assert "another team" not in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_retryable_original_status_is_not_swapped(self):
        """408 is transient under litellm._should_retry; it must not be reported as permanent."""
        router: Final = self._router(
            spill_tags=["pin:openai"],
            spill_mock_response="never selected by pin:azure",
        )
        router.model_list[0]["litellm_params"]["mock_response"] = _SyntheticUpstream408("upstream timeout")
        router.model_list[1]["litellm_params"]["mock_response"] = _SyntheticUpstream408("upstream timeout")

        with pytest.raises(RouterRateLimitError):
            await router.acompletion(
                model="pooled-model",
                messages=[{"role": "user", "content": "hi"}],
                metadata={"tags": ["pin:azure"]},
            )
