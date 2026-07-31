"""
SELF-SERVICE USAGE

GET /v1/usage — lets a virtual-key holder read their OWN usage: per-window
budget state and a per-model rollup. Nothing else.

Why this exists
---------------
An external collaborator holds a gateway key and no SSO identity, so every
human surface (admin UI, portal) is shut to them. They can already measure
cost per request from response headers, but two questions are unanswerable
client-side:

  * "how much budget do I have left" — the binding cap is a budget *window*
    (`budget_limits`), whose counters live in Redis and are never echoed to
    the caller.
  * "where did it go" — a per-model breakdown only exists server-side.

Scoping (the security property this file is built around)
---------------------------------------------------------
Everything is derived from ``user_api_key_dict.api_key`` — the sha256 token
hash of the presented credential. The handler accepts **no caller-supplied
identity parameter**: no ``key=``, ``user_id=``, ``team_id=``, ``api_key=``.
A cross-tenant read is therefore *unrepresentable*, not merely filtered.

That is a stronger guarantee than "we remember to filter", and it is the whole
reason this route can be public. ``tests/test_litellm/proxy/management_endpoints/
test_self_service_usage_endpoints.py`` asserts the signature accepts no such
parameter, so a later staff-convenience ``?key=`` for support triage fails the
build instead of silently destroying the property.

The response is a **frozen allowlist**, asserted against the response body
rather than the stored row: pydantic silently drops unknown kwargs, so a test
that inspects the model instance can pass while the wire carries something
else. Never returned: ``metadata`` (it carries ``requested_by_email``, a
Rayward employee's address), ``team_id``, ``user_id``, ``credential_set_id``.

Related: upstream ships the same self-service shape on the Bifrost side with
no admin auth ("Self-service endpoint — no admin auth, VK in header is the
credential"), which is the precedent this follows.
"""

from collections.abc import Mapping, Sequence
from typing import Annotated
from datetime import datetime
from decimal import Decimal
from types import MappingProxyType

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()

# Read-only stand-in so an absent aggregate does not construct a fresh dict
# per row (LIT002) just to be read once.
_NO_TOTALS: Mapping[str, object] = MappingProxyType({})  # mutable-ok: frozen at construction, never mutated


class UsageBudgetWindow(BaseModel):
    """One budget window's state. `spent`/`remaining` are for THIS window only."""

    duration: str = Field(description="Window duration as configured, e.g. '1d', '1w', '1mo'.")
    max_budget: float | None = Field(default=None, description="Cap for this window in USD.")
    spent: float = Field(default=0.0, description="Spend accumulated inside the current window.")
    remaining: float | None = Field(
        default=None,
        description="max_budget - spent, floored at 0. Null when the window has no finite cap.",
    )
    reset_at: str | None = Field(default=None, description="ISO-8601 timestamp at which this window's counter resets.")


class UsageModelRow(BaseModel):
    """Per-model rollup for the calling key."""

    model: str | None = None
    spend: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0


class UsageResponse(BaseModel):
    """The complete wire contract. Adding a field here is a deliberate act —
    see the allowlist test before doing it."""

    key_alias: str | None = None
    budgets: Sequence[UsageBudgetWindow] = Field(default=())
    models: Sequence[UsageModelRow] = Field(default=())


def _as_int(value: object) -> int:
    """BigInt columns arrive as int, str or Decimal depending on the driver.

    Narrowed explicitly rather than typed Any: the driver-dependence is exactly
    the reason to enumerate what may arrive instead of waving it through.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, Decimal)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _as_float(value: object) -> float:
    """Same contract as _as_int, for the Float columns."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


async def _budget_windows(valid_token: UserAPIKeyAuth) -> tuple[UsageBudgetWindow, ...]:
    """Read each configured window's own counter.

    Deliberately NOT derived from ``valid_token.spend``: that field is a
    LIFETIME counter and never resets, so subtracting it from a monthly cap
    reports "budget exhausted" on a key that has plenty left as soon as the key
    outlives one window. Each window has its own counter, keyed exactly as
    ``_virtual_key_multi_budget_check`` writes it, so what we report here is the
    same number that enforcement will act on.
    """
    budget_limits = getattr(valid_token, "budget_limits", None)
    if not budget_limits:
        return ()

    from litellm.proxy.proxy_server import get_current_spend
    from litellm.proxy.spend_tracking.budget_reservation import get_budget_window_start

    windows: list[UsageBudgetWindow] = []  # mutable-ok: local accumulator, never escapes; returned as a tuple
    for window in budget_limits:
        w: Mapping[str, object] = window if isinstance(window, dict) else window.model_dump()
        duration = w.get("budget_duration")
        if duration is None:
            continue
        raw_max_budget = w.get("max_budget")
        max_budget = _as_float(raw_max_budget) if raw_max_budget is not None else None

        spent = await get_current_spend(
            counter_key=f"spend:key:{valid_token.token}:window:{duration}",
            fallback_spend=0.0,
            max_budget=max_budget,
            window_entity_type="Key",
            window_entity_id=valid_token.token,
            window_start=get_budget_window_start(w),
        )

        remaining = max(0.0, max_budget - _as_float(spent)) if max_budget is not None else None

        # reset_at is a datetime from the DB but a string once the window has
        # been through a model_dump/JSON round trip, so normalise both shapes.
        raw_reset_at = w.get("reset_at")
        if isinstance(raw_reset_at, datetime):
            reset_at = raw_reset_at.isoformat()
        elif isinstance(raw_reset_at, str):
            reset_at = raw_reset_at
        else:
            reset_at = None

        windows.append(
            UsageBudgetWindow(
                duration=str(duration),
                max_budget=max_budget,
                spent=_as_float(spent),
                remaining=remaining,
                reset_at=reset_at,
            )
        )
    return tuple(windows)


async def _model_rollup(api_key: str) -> tuple[UsageModelRow, ...]:
    """Aggregate LiteLLM_DailyUserSpend for this key.

    Filtered on ``api_key`` ALONE — strictly narrower than
    ``/user/daily/activity``'s user scope. Upstream BerriAI/litellm #19194
    (cross-team usage leakage, closed "not planned") cannot apply here by
    construction rather than by luck: there is no user or team predicate to get
    wrong, because there is no user or team predicate.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return ()

    rows = await prisma_client.db.litellm_dailyuserspend.group_by(
        by=["model"],  # mutable-ok: prisma query arg, passed to the driver, never retained
        where={"api_key": api_key},  # mutable-ok: prisma query arg; api_key ALONE is the scoping guarantee
        sum={  # mutable-ok: prisma query arg, passed to the driver, never retained
            "spend": True,
            "prompt_tokens": True,
            "completion_tokens": True,
            "api_requests": True,
            "successful_requests": True,
            "failed_requests": True,
        },
    )

    rollup: list[UsageModelRow] = []  # mutable-ok: local accumulator, never escapes; returned as a tuple
    for row in rows or ():
        totals = row.get("_sum") or _NO_TOTALS
        rollup.append(
            UsageModelRow(
                model=row.get("model"),
                spend=_as_float(totals.get("spend")),
                prompt_tokens=_as_int(totals.get("prompt_tokens")),
                completion_tokens=_as_int(totals.get("completion_tokens")),
                api_requests=_as_int(totals.get("api_requests")),
                successful_requests=_as_int(totals.get("successful_requests")),
                failed_requests=_as_int(totals.get("failed_requests")),
            )
        )
    rollup.sort(key=lambda r: r.spend, reverse=True)
    return tuple(rollup)


@router.get(
    "/v1/usage",
    tags=["usage"],  # mutable-ok: FastAPI requires a list here
    response_model=UsageResponse,
    summary="Read your own key's budget state and per-model usage",
)
async def get_self_usage(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> UsageResponse:
    """Return the calling key's per-window budgets and per-model rollup.

    Takes NO parameters. The identity is the presented credential and cannot be
    overridden by the caller — see this module's docstring.
    """
    api_key = user_api_key_dict.api_key or user_api_key_dict.token

    budgets = await _budget_windows(user_api_key_dict)
    models = await _model_rollup(api_key) if api_key else ()

    return UsageResponse(
        key_alias=user_api_key_dict.key_alias,
        budgets=budgets,
        models=models,
    )
