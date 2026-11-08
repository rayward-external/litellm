"""``RustCore._tag_routable_ids`` (litellm/router_rust/core.py), new in the 2026-11-08 sync's
router_python_core.py deletion, lists the cooling deployments tag routing would still send a
request to. ``get_deployments_for_tag`` ends a pass either by raising ``ValueError`` (tag routing
sends the request to none of the rest) or, once every candidate is exhausted, by raising
``litellm.BadRequestError`` via ``_raise_no_deployments_for_tags``
(litellm/router_strategy/tag_based_routing.py). ``litellm.BadRequestError`` subclasses
``openai.BadRequestError``, not ``ValueError``, so an ``except ValueError`` clause alone misses it
and the error escapes ``_cooldown_deployment_info``/``_failure_error`` as a non-retryable 400
instead of the usual ``RouterRateLimitError`` the caller expects.
"""

from collections.abc import Mapping
from typing import Final

import litellm
import litellm.router_strategy.tag_based_routing as tag_based_routing
from litellm.router_rust.core import RustCore


def test_tag_routable_ids_catches_bad_request_error(monkeypatch) -> None:
    def _raises_tag_exhausted(**kwargs: object) -> object:
        raise litellm.BadRequestError(message="no deployments for tags", model="m", llm_provider="")

    monkeypatch.setattr(tag_based_routing, "get_deployments_for_tag", _raises_tag_exhausted)

    core: Final = RustCore.__new__(RustCore)
    deployment: Final[Mapping[str, object]] = {"model_name": "m", "model_info": {"id": "d1"}}
    result = core._tag_routable_ids(  # pyright: ignore[reportPrivateUsage]  # exercising the method directly
        group="m",
        pool=[deployment],
        routable=[deployment],
        cooled=frozenset({"d1"}),
        request={"model": "m", "messages": []},
    )
    assert result == frozenset()


def test_tag_routable_ids_still_catches_value_error(monkeypatch) -> None:
    """The pre-existing ValueError path (tag routing sends this request to none of
    the rest, short of the full-exhaustion BadRequestError above) must keep working."""

    def _raises_value_error(**kwargs: object) -> object:
        raise ValueError("no match")

    monkeypatch.setattr(tag_based_routing, "get_deployments_for_tag", _raises_value_error)

    core: Final = RustCore.__new__(RustCore)
    deployment: Final[Mapping[str, object]] = {"model_name": "m", "model_info": {"id": "d1"}}
    result = core._tag_routable_ids(  # pyright: ignore[reportPrivateUsage]  # exercising the method directly
        group="m",
        pool=[deployment],
        routable=[deployment],
        cooled=frozenset({"d1"}),
        request={"model": "m", "messages": []},
    )
    assert result == frozenset()
