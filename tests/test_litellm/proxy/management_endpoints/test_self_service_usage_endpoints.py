"""Guards for GET /v1/usage — the self-service usage read.

The two tests that matter here are structural, not behavioural. This route is
reachable by an external key holder with no SSO identity, so its safety rests on
two properties that are easy to destroy accidentally in a later edit:

  1. The handler takes NO caller-supplied identity parameter, which makes a
     cross-tenant read unrepresentable rather than merely filtered.
  2. The response is a frozen allowlist, asserted on the SERIALISED BODY.

(2) is asserted on the body specifically because pydantic silently drops unknown
kwargs — a test that inspects a model instance can pass while the wire carries a
field nobody approved. That is the #387 lesson, and it applies verbatim here.
"""

import inspect
from unittest.mock import patch

import pytest

from litellm.proxy._types import LiteLLMRoutes, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import MODEL_DISCOVERY_ROUTES
from litellm.proxy.management_endpoints.self_service_usage_endpoints import (
    UsageResponse,
    _budget_windows,
    _model_rollup,
    get_self_usage,
)

# Every field the wire is allowed to carry. Adding one is a deliberate act.
ALLOWED_TOP_LEVEL = {"key_alias", "budgets", "models"}
ALLOWED_BUDGET = {"duration", "max_budget", "spent", "remaining", "reset_at"}
ALLOWED_MODEL = {
    "model",
    "spend",
    "prompt_tokens",
    "completion_tokens",
    "api_requests",
    "successful_requests",
    "failed_requests",
}

# Fields whose presence would leak someone else's data or an employee's identity.
FORBIDDEN_ANYWHERE = {
    "metadata",  # carries requested_by_email — a Rayward employee address
    "team_id",
    "user_id",
    "credential_set_id",
    "token",
    "api_key",
    "key",
    "requested_by_email",
}


def _all_keys(obj: object) -> set:
    """Every dict key anywhere in a nested structure, for exact-match checks."""
    keys: set = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            keys |= _all_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            keys |= _all_keys(item)
    return keys


def test_handler_accepts_no_caller_supplied_identity() -> None:
    """A `?key=` added later for support triage must fail the build.

    The scoping guarantee is that identity comes only from the presented
    credential. If the signature ever grows a second parameter, that guarantee
    is gone whether or not the body remembers to filter on it.
    """
    params = set(inspect.signature(get_self_usage).parameters) - {"user_api_key_dict"}
    assert params == set(), (
        f"get_self_usage grew parameter(s) {sorted(params)}. Identity must derive "
        f"ONLY from user_api_key_dict, so a cross-tenant read stays unrepresentable "
        f"rather than merely filtered."
    )
    for banned in ("key", "api_key", "user_id", "team_id"):
        assert banned not in inspect.signature(get_self_usage).parameters


def test_response_body_is_a_frozen_allowlist() -> None:
    """Asserted on the SERIALISED body, not the model instance."""
    body: dict[str, object] = UsageResponse(
        key_alias="umass-1d6b0754",
        budgets=[
            {  # type: ignore[list-item]
                "duration": "1d",
                "max_budget": 2000.0,
                "spent": 12.5,
                "remaining": 1987.5,
                "reset_at": "2026-08-01T00:00:00+00:00",
            }
        ],
        models=[{"model": "claude-sonnet-5", "spend": 12.5}],  # type: ignore[list-item]
    ).model_dump()

    assert set(body) == ALLOWED_TOP_LEVEL
    assert set(body["budgets"][0]) == ALLOWED_BUDGET
    assert set(body["models"][0]) == ALLOWED_MODEL

    # Exact key match, walked recursively — NOT a substring scan of the
    # stringified body. A substring check reports "token" as leaked because
    # "prompt_tokens" contains it, which is a false positive that would train
    # the next person to weaken the assertion instead of trusting it.
    leaked = FORBIDDEN_ANYWHERE & _all_keys(body)
    assert leaked == set(), f"forbidden field(s) reached the response body: {sorted(leaked)}"


@pytest.mark.asyncio
async def test_budget_windows_use_per_window_counters_not_lifetime_spend() -> None:
    """The lifetime `spend` field must not be substituted for a window counter.

    Subtracting a lifetime total from a monthly cap reports STOP on a budget with
    plenty left, as soon as the key outlives one window. This asserts we read the
    same counter key that enforcement writes, so what we report and what gets
    enforced cannot diverge.
    """
    token = UserAPIKeyAuth(
        token="hash-abc",
        api_key="hash-abc",
        spend=9_999_999.0,  # absurd lifetime total; must be ignored entirely
        budget_limits=[
            {"budget_duration": "1d", "max_budget": 100.0},
            {"budget_duration": "1mo", "max_budget": 1000.0},
        ],
    )

    seen: list[str] = []

    async def fake_spend(counter_key: str, **kwargs: object) -> float:
        seen.append(counter_key)
        return 25.0

    with patch("litellm.proxy.proxy_server.get_current_spend", new=fake_spend):
        windows = await _budget_windows(token)

    assert seen == [
        "spend:key:hash-abc:window:1d",
        "spend:key:hash-abc:window:1mo",
    ], "must read the same per-window counters _virtual_key_multi_budget_check writes"

    assert [w.spent for w in windows] == [25.0, 25.0]
    assert [w.remaining for w in windows] == [75.0, 975.0]
    assert all(w.spent != token.spend for w in windows)


@pytest.mark.asyncio
async def test_remaining_is_floored_at_zero_when_over_budget() -> None:
    """An over-spent window reports 0 remaining, never a negative number."""
    token = UserAPIKeyAuth(token="t", api_key="t", budget_limits=[{"budget_duration": "1d", "max_budget": 10.0}])

    async def over(counter_key: str, **kwargs: object) -> float:
        return 42.0

    with patch("litellm.proxy.proxy_server.get_current_spend", new=over):
        windows = await _budget_windows(token)

    assert windows[0].spent == 42.0
    assert windows[0].remaining == 0.0


@pytest.mark.asyncio
async def test_model_rollup_filters_on_api_key_alone() -> None:
    """No user or team predicate — that is what makes cross-tenant leakage
    unrepresentable rather than a filter we have to keep correct."""
    captured: dict[str, object] = {}

    class FakeTable:
        async def group_by(self, **kwargs: object) -> list[dict[str, object]]:
            captured.update(kwargs)
            return [
                {"model": "gpt-5.5", "_sum": {"spend": 1.0, "prompt_tokens": 10}},
                {"model": "claude-sonnet-5", "_sum": {"spend": 5.0, "prompt_tokens": 20}},
            ]

    class FakeDB:
        litellm_dailyuserspend = FakeTable()

    class FakeClient:
        db = FakeDB()

    with patch("litellm.proxy.proxy_server.prisma_client", FakeClient()):
        rows = await _model_rollup("hash-abc")

    assert captured["where"] == {"api_key": "hash-abc"}, (
        "the WHERE clause must be api_key alone — adding a user_id/team_id "
        "predicate reintroduces the class of bug upstream #19194 describes"
    )
    # highest spend first, so the expensive model is the one you see
    assert [r.model for r in rows] == ["claude-sonnet-5", "gpt-5.5"]
    assert rows[0].spend == 5.0
    assert rows[1].prompt_tokens == 10


@pytest.mark.asyncio
async def test_model_rollup_is_empty_without_a_db_rather_than_raising() -> None:
    with patch("litellm.proxy.proxy_server.prisma_client", None):
        assert await _model_rollup("hash-abc") == ()


def test_route_survives_an_exhausted_budget() -> None:
    """Registered for the budget skip, so the read works exactly when needed.

    Without this the holder learns they are over budget from the same 429 that
    hides how far over they are and when the window resets.
    """
    assert "/v1/usage" in MODEL_DISCOVERY_ROUTES


def test_route_is_self_managed() -> None:
    """The one role-independent branch — required for a key with user_id=None."""
    assert "/v1/usage" in LiteLLMRoutes.self_managed_routes.value


def test_route_is_not_an_llm_api_route() -> None:
    """Must NOT be an LLM route, or _virtual_key_max_budget_check fires on it
    and the read dies on an exhausted key-level budget."""
    from litellm.proxy.auth.route_checks import RouteChecks

    assert RouteChecks.is_llm_api_route(route="/v1/usage") is False
