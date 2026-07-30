"""Tests for GET /v1/usage — the self-service usage read.

The security properties are tested harder than the happy path, because the happy
path fails loudly and the security properties fail silently. Two in particular:

- the handler must accept no caller-supplied identity parameter, so pointing it
  at another tenant is unrepresentable rather than merely filtered;
- the response is a frozen allowlist, so a later change cannot widen it into
  leaking `metadata` (which carries the minting employee's email on
  broker-issued keys), `team_id` or `user_id`.
"""

import inspect
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from fastapi.testclient import TestClient

from litellm.proxy._types import LiteLLMRoutes, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import MODEL_DISCOVERY_ROUTES
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints import self_usage_endpoints
from litellm.proxy.proxy_server import app

KEY_HASH = "a" * 64
OTHER_HASH = "b" * 64


def _auth(budget_limits=None, spend=1.25, alias="umass-1d6b0754"):
    return UserAPIKeyAuth(
        api_key=KEY_HASH,
        token=KEY_HASH,
        key_alias=alias,
        spend=spend,
        budget_limits=budget_limits,
        team_id="team-external",
        user_id="hunter",
    )


def _row(model, spend, prompt=0, completion=0, requests=0):
    row = MagicMock()
    row.model = model
    row.spend = spend
    row.prompt_tokens = prompt
    row.completion_tokens = completion
    row.cache_read_input_tokens = 0
    row.cache_creation_input_tokens = 0
    row.api_requests = requests
    row.successful_requests = requests
    row.failed_requests = 0
    return row


@pytest.fixture
def client_with_key():
    def _make(auth_obj, rows=None, captured=None):
        app.dependency_overrides[user_api_key_auth] = lambda: auth_obj

        prisma = MagicMock()

        async def _find_many(where=None, **kwargs):
            if captured is not None:
                captured["where"] = where
            return rows or []

        prisma.db.litellm_dailyuserspend.find_many = _find_many
        patcher = patch("litellm.proxy.proxy_server.prisma_client", prisma)
        patcher.start()
        return TestClient(app), patcher

    yield _make
    app.dependency_overrides.pop(user_api_key_auth, None)


# ── the scoping property ────────────────────────────────────────────────────


def test_the_handler_accepts_no_caller_supplied_identity_parameter():
    """The property that makes a cross-tenant read unrepresentable.

    Filtering by the authenticated key is not enough on its own: a later change
    that adds `?key=` "just for support triage" would silently destroy the
    guarantee while every other test kept passing. So the signature itself is
    pinned.
    """
    params = set(inspect.signature(self_usage_endpoints.self_usage).parameters)
    forbidden = {"key", "api_key", "user_id", "team_id", "token", "key_alias", "end_user"}
    assert not (params & forbidden), f"identity parameter(s) exposed: {params & forbidden}"
    assert params == {"start_date", "end_date", "user_api_key_dict"}


def test_the_model_rollup_filters_on_the_calling_key_alone(client_with_key):
    """Narrower than the user-scoped daily activity endpoint, by construction.

    Filtering on api_key alone means a second key belonging to the same user
    cannot appear -- upstream's cross-tenant leak in /user/daily/activity cannot
    apply to us by construction rather than by luck.
    """
    captured = {}
    client, patcher = client_with_key(_auth(), rows=[_row("gpt-5.6-sol", 1.0)], captured=captured)
    try:
        response = client.get("/v1/usage")
    finally:
        patcher.stop()

    assert response.status_code == 200
    where = captured["where"]
    assert where["api_key"] == KEY_HASH
    for leak in ("user_id", "team_id"):
        assert leak not in where, f"{leak} must not widen the query"


# ── the response allowlist ──────────────────────────────────────────────────


def test_the_response_never_carries_another_tenants_fields(client_with_key):
    """Asserted on the RESPONSE, not the source object.

    The auth object handed to the handler deliberately carries team_id and
    user_id here; the test is worthless if it only checks that we did not ask
    for them.
    """
    client, patcher = client_with_key(_auth(), rows=[_row("gpt-5.6-sol", 1.0)])
    try:
        body = client.get("/v1/usage").json()
    finally:
        patcher.stop()

    assert set(body) == {
        "key_alias",
        "spend",
        "budgets",
        "by_model",
        "start_date",
        "end_date",
        "as_of",
    }
    serialized = str(body)
    for leak in ("team-external", "hunter", "metadata", "requested_by_email"):
        assert leak not in serialized, f"response leaked {leak}"


# ── budget windows ──────────────────────────────────────────────────────────


def test_budget_windows_are_reported_with_spend_and_headroom(client_with_key):
    windows = [
        {"budget_duration": "1d", "max_budget": 50.0, "reset_at": "2026-07-31T00:00:00+00:00"},
        {"budget_duration": "1mo", "max_budget": 1000.0, "reset_at": "2026-08-30T00:00:00+00:00"},
    ]
    client, patcher = client_with_key(_auth(budget_limits=windows))
    try:
        with patch.object(
            self_usage_endpoints, "_window_spend", new=AsyncMock(side_effect=[12.0, 227.25])
        ):
            body = client.get("/v1/usage").json()
    finally:
        patcher.stop()

    daily, monthly = body["budgets"]
    assert daily["budget_duration"] == "1d"
    assert daily["spend"] == 12.0
    assert daily["remaining"] == 38.0
    assert daily["resets_at"] == "2026-07-31T00:00:00+00:00"
    assert monthly["remaining"] == pytest.approx(772.75)


def test_a_key_with_no_windows_reports_an_empty_list_not_unlimited(client_with_key):
    """No windows on the key does not mean no cap -- a team cap may still bind.

    Reporting "unlimited" here would be the optimistic error, and optimistic is
    the direction that costs someone money.
    """
    client, patcher = client_with_key(_auth(budget_limits=None))
    try:
        body = client.get("/v1/usage").json()
    finally:
        patcher.stop()
    assert body["budgets"] == []


def test_window_spend_prefers_the_larger_of_counter_and_spend_logs():
    """A cold Redis counter reads 0.0 for a window; the log sum is the floor.

    Under-reporting spend overstates remaining budget, so the larger value wins.
    """
    import asyncio

    window = {"budget_duration": "1mo", "max_budget": 500.0}

    from litellm.proxy.db import spend_counter_reseed

    with patch(
        "litellm.proxy.proxy_server.get_current_spend", new=AsyncMock(return_value=0.0)
    ), patch.object(
        spend_counter_reseed.SpendCounterReseed,
        "window_from_spend_logs",
        new=AsyncMock(return_value=227.25),
    ), patch(
        "litellm.proxy.proxy_server.prisma_client", MagicMock()
    ):
        result = asyncio.run(self_usage_endpoints._window_spend(token=KEY_HASH, window=window))

    assert result == pytest.approx(227.25), "the spend-log floor must win over a cold counter"


def test_window_spend_survives_a_counter_read_failure():
    """Redis being unreachable must degrade the number, not fail the report."""
    import asyncio

    window = {"budget_duration": "1d", "max_budget": 50.0}
    with patch(
        "litellm.proxy.proxy_server.get_current_spend",
        new=AsyncMock(side_effect=RuntimeError("redis down")),
    ), patch("litellm.proxy.proxy_server.prisma_client", None):
        result = asyncio.run(self_usage_endpoints._window_spend(token=KEY_HASH, window=window))
    assert result == 0.0


# ── rollup shape ────────────────────────────────────────────────────────────


def test_models_are_returned_most_expensive_first(client_with_key):
    """The question behind this endpoint is "what is eating my budget"."""
    rows = [
        _row("gemini-3.6-flash", 31.05),
        _row("gpt-5.6-sol", 142.80),
        _row("gpt-5.6-terra", 68.10),
    ]
    client, patcher = client_with_key(_auth(), rows=rows)
    try:
        body = client.get("/v1/usage").json()
    finally:
        patcher.stop()

    assert [row["model"] for row in body["by_model"]] == [
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gemini-3.6-flash",
    ]


def test_rows_for_the_same_model_are_summed_across_days(client_with_key):
    rows = [_row("gpt-5.6-sol", 10.0, prompt=100, requests=2) for _ in range(3)]
    client, patcher = client_with_key(_auth(), rows=rows)
    try:
        body = client.get("/v1/usage").json()
    finally:
        patcher.stop()

    assert len(body["by_model"]) == 1
    assert body["by_model"][0]["spend"] == pytest.approx(30.0)
    assert body["by_model"][0]["prompt_tokens"] == 300
    assert body["by_model"][0]["api_requests"] == 6


# ── date range ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "params,expected",
    [
        ({"start_date": "not-a-date"}, 400),
        ({"start_date": "2026-07-30", "end_date": "2026-07-01"}, 400),
        ({"start_date": "2020-01-01", "end_date": "2026-07-30"}, 400),
    ],
)
def test_bad_date_ranges_are_refused(client_with_key, params, expected):
    client, patcher = client_with_key(_auth())
    try:
        assert client.get("/v1/usage", params=params).status_code == expected
    finally:
        patcher.stop()


# ── registration ────────────────────────────────────────────────────────────


def test_the_route_is_self_managed():
    """Role-agnostic branch: a key with user_id=None must still reach it."""
    assert "/v1/usage" in LiteLLMRoutes.self_managed_routes.value


def test_the_route_survives_budget_exhaustion():
    """The meter must not go dark exactly when the caller has run out.

    The team budget check is not route-gated, so without this exemption an
    exhausted team budget 403s every route -- including the one that would tell
    the key holder they are out.
    """
    assert "/v1/usage" in MODEL_DISCOVERY_ROUTES
