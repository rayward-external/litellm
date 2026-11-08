"""Deployment-level attribution tags, winning-deployment-only (rayward fork patch), on
``update_kwargs_with_deployment`` (litellm/router_rust/invoke/shaping.py), restored after the
2026-11-08 sync's router_python_core.py deletion dropped it.

The request's kwargs dict is reused across retries/fallbacks, so merging each serving
deployment's own tags into the live ``metadata["tags"]`` would accumulate every failed
attempt's tags onto the one that finally succeeds (and the same for the credential-name
tag). The fix rebuilds ``tags`` from the caller's ``ROUTING_REQUEST_TAGS_METADATA_KEY``
snapshot plus only the current deployment's own tags on every attempt.
"""

from collections.abc import Callable, Mapping
from typing import Final

import pytest

from litellm.router_core import RouterCore
from tests.unit.test_router.core.construction_precall_helpers import CallbackHost
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment, rate_limit

MakeCore = Callable[..., RouterCore]
MESSAGES: Final = [{"role": "user", "content": "hi"}]


def _metadata(kwargs: Mapping[str, object], name: str = "metadata") -> Mapping[str, object]:
    bucket: Final = kwargs[name]
    assert isinstance(bucket, Mapping)
    return bucket


@pytest.fixture
def host() -> ScriptedHost:
    return CallbackHost()


async def test_retry_tags_do_not_accumulate_across_deployments(make_core: MakeCore, host: ScriptedHost) -> None:
    """A request that falls back from a tagged deployment to another tagged deployment must
    carry only the SERVING deployment's tags on each attempt, not the union of both."""
    host.answer("openai/d1", rate_limit())
    core: Final = make_core(
        model_list=[
            deployment("g1", "d1", tags=["t1"]),
            deployment("g2", "d2", tags=["t2"]),
        ],
        fallbacks=[{"g1": ["g2"]}],
        num_retries=0,
    )
    await core.acompletion(model="g1", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]  # every core serves acompletion
    assert len(host.calls) == 2
    first_metadata = _metadata(host.calls[0].kwargs)
    second_metadata = _metadata(host.calls[1].kwargs)
    assert first_metadata["tags"] == ["t1"]
    assert second_metadata["tags"] == ["t2"]
