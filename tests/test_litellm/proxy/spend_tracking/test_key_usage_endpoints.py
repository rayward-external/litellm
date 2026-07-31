"""Guards for `GET /v1/usage` (#405).

The endpoint's whole value rests on three properties, each of which is easy to
destroy with a well-intentioned edit. These tests pin them.
"""

import inspect

import pytest

from litellm.proxy._types import LiteLLMRoutes
from litellm.proxy.auth import auth_checks
from litellm.proxy.spend_tracking import key_usage_endpoints


# ── 1. Scoping: a cross-tenant read must be UNREPRESENTABLE ──────────────────
# Not "filtered correctly" -- unrepresentable. There must be no argument an
# attacker can vary. The natural future edit is a staff-convenience `?key=` for
# support triage, which would silently turn this into a filtered read while
# every behavioural test kept passing.

_FORBIDDEN_PARAMS = {
    "key",
    "api_key",
    "user_id",
    "team_id",
    "customer_id",
    "end_user_id",
    "organization_id",
    "token",
}


def test_handler_accepts_no_caller_supplied_identity():
    params = set(inspect.signature(key_usage_endpoints.key_usage).parameters)
    leaked = params & _FORBIDDEN_PARAMS
    assert not leaked, (
        f"GET /v1/usage grew caller-supplied identity parameter(s) {sorted(leaked)}. "
        f"Scope must come ONLY from the authenticated key (user_api_key_dict), so "
        f"that a cross-tenant read cannot be expressed at all. If support triage "
        f"needs to read another key's usage, that belongs on an admin route with "
        f"its own authorization -- not here."
    )


def test_handler_takes_only_the_auth_dependency():
    params = list(inspect.signature(key_usage_endpoints.key_usage).parameters)
    assert params == ["user_api_key_dict"], (
        f"GET /v1/usage should take exactly one parameter, the auth dependency; "
        f"got {params}. Any additional parameter is a scope widening."
    )


# ── 2. The read must survive budget exhaustion ───────────────────────────────
# This is the defect that motivated the endpoint: when the team exhausts its
# budget, the holder loses the ability to see THAT they have run out. An
# external key holder has no identity on our IdP, so no other surface exists.


def test_v1_usage_is_exempt_from_budget_checks():
    assert "/v1/usage" in auth_checks.BUDGET_EXEMPT_READ_ROUTES, (
        "/v1/usage must skip budget checks, or it 400s exactly when it is needed "
        "-- the route that reports exhaustion cannot itself be gated on it."
    )


def test_budget_exemption_stays_narrower_than_info_routes():
    """An exhausted budget must not reach side-effectful routes (#27923).

    The exemption is a union with MODEL_DISCOVERY_ROUTES precisely so it cannot
    quietly grow into the much wider info_routes set.
    """
    exempt = auth_checks.BUDGET_EXEMPT_READ_ROUTES
    assert auth_checks.MODEL_DISCOVERY_ROUTES <= exempt
    assert exempt - auth_checks.MODEL_DISCOVERY_ROUTES == {"/v1/usage"}, (
        "the budget exemption grew beyond /v1/usage; every addition must be a "
        "read-only, side-effect-free route, argued for on its own"
    )
    for side_effectful in ("/health/services", "/key/generate", "/key/info"):
        assert side_effectful not in exempt


# ── 3. Route registration ────────────────────────────────────────────────────


def test_registered_as_self_managed_route():
    """self_managed_routes is the one role-independent branch in route_checks,
    so it is what makes this work for a key with user_id=None as well as a
    class-A key."""
    assert "/v1/usage" in LiteLLMRoutes.self_managed_routes.value


def test_route_is_registered_on_the_router():
    paths = {r.path for r in key_usage_endpoints.router.routes}
    assert "/v1/usage" in paths


# ── 4. Response allowlist ────────────────────────────────────────────────────
# Asserted against the RESPONSE, not a stored row: FakeRepo never executes SQL
# and pydantic silently drops unknown kwargs, so asserting on the model proves
# nothing about the bytes that leave (#387).

_FORBIDDEN_RESPONSE_KEYS = {
    "metadata",  # carries requested_by_email -- an employee's address
    "team_id",
    "user_id",
    "credential_set_id",
    "key",
    "api_key",
    "token",
}


@pytest.mark.asyncio
async def test_response_carries_no_forbidden_fields(monkeypatch):
    from litellm.proxy._types import UserAPIKeyAuth

    async def _no_windows(_valid_token):
        return []

    async def _no_models(_api_key):
        return []

    monkeypatch.setattr(key_usage_endpoints, "_window_budgets", _no_windows)
    monkeypatch.setattr(key_usage_endpoints, "_per_model_rollup", _no_models)

    token = UserAPIKeyAuth(
        api_key="sk-hashed",
        key_alias="external-party",
        key_name="sk-...abcd",
        spend=12.5,
        max_budget=100.0,
        user_id="employee-who-minted-it",
        team_id="External",
        metadata={"requested_by_email": "someone@openrefinery.ai"},
    )

    body = await key_usage_endpoints.key_usage(user_api_key_dict=token)

    leaked = set(body) & _FORBIDDEN_RESPONSE_KEYS
    assert not leaked, (
        f"GET /v1/usage response leaked {sorted(leaked)}. metadata carries the "
        f"minting employee's email; team_id/user_id are not the caller's to see."
    )
    assert body["lifetime_spend"] == 12.5
    assert "budget_windows" in body and "models" in body


@pytest.mark.asyncio
async def test_lifetime_spend_is_not_named_like_a_window(monkeypatch):
    """`spend` on the token is a LIFETIME counter, a different quantity from a
    window's spend. Naming it `spend` invites a client to subtract it from a
    monthly cap and report STOP on a budget with plenty left, as soon as the
    key outlives one window."""
    from litellm.proxy._types import UserAPIKeyAuth

    async def _no_windows(_valid_token):
        return []

    async def _no_models(_api_key):
        return []

    monkeypatch.setattr(key_usage_endpoints, "_window_budgets", _no_windows)
    monkeypatch.setattr(key_usage_endpoints, "_per_model_rollup", _no_models)

    body = await key_usage_endpoints.key_usage(
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-hashed", spend=3.0)
    )
    assert "spend" not in body, (
        "a bare `spend` key is ambiguous between lifetime and window totals; "
        "use lifetime_spend"
    )
    assert body["lifetime_spend"] == 3.0
