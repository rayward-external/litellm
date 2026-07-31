"""`GET /v1/usage` — a key holder's view of its OWN spend.

Why this exists
---------------
An external collaborator holds a gateway key and no identity on our IdP, so
every human surface is shut to them. The distributable `gateway-cost` skill
closed most of that gap client-side, but two things are structurally
impossible from outside:

* **Remaining budget.** The cap that actually binds is a budget *window*
  (`budget_limits`), whose counter lives in Redis. No client can see it, and
  the lifetime `spend` field is a different quantity — subtracting it from a
  monthly cap reports STOP on a budget with plenty left as soon as the key
  outlives one window.
* **Per-model / per-day breakdown.** Only the server has it.

Scoping — the load-bearing property
-----------------------------------
Everything is derived from ``user_api_key_dict`` alone. This handler accepts
**no caller-supplied identity parameter** — no ``key``, ``user_id``,
``team_id``, ``api_key``. That makes a cross-tenant read *unrepresentable*
rather than merely filtered, which is a much stronger guarantee than a
correct ``WHERE`` clause: there is no argument an attacker can vary.

``tests/test_key_usage_endpoint.py`` asserts that property against the
signature, because the natural future edit — a ``?key=`` for support triage —
would destroy it silently while every existing test kept passing.

The per-model rollup queries ``LiteLLM_DailyUserSpend`` filtered on
``api_key`` **alone**. That is strictly narrower than ``/user/daily/activity``'s
user scope, so BerriAI/litellm#19194 (cross-team usage leakage) cannot apply
here by construction rather than by luck.

Response fields are an explicit allowlist. Never ``metadata`` (it carries
``requested_by_email``, an employee address), ``team_id``, ``user_id`` or
``credential_set_id``.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.utils import handle_exception_on_proxy

router = APIRouter()


async def _window_budgets(valid_token: UserAPIKeyAuth) -> List[Dict[str, Any]]:
    """Per-window cap, spend, remaining and window start.

    Mirrors ``_virtual_key_window_budget_check`` exactly — same counter key,
    same ``get_current_spend`` call — so what a caller reads is what the
    enforcement path will act on. Reimplementing the key format here would
    let the two drift, and a usage endpoint that disagrees with enforcement is
    worse than none.
    """
    from litellm.proxy.proxy_server import get_current_spend
    from litellm.proxy.spend_tracking.budget_reservation import (
        get_budget_window_start,
    )

    if not valid_token.budget_limits:
        return []

    out: List[Dict[str, Any]] = []
    for window in valid_token.budget_limits:
        w: dict = window if isinstance(window, dict) else window.model_dump()
        counter_key = f"spend:key:{valid_token.token}:window:{w['budget_duration']}"
        window_start = get_budget_window_start(w)
        spent = await get_current_spend(
            counter_key=counter_key,
            fallback_spend=0.0,
            max_budget=w["max_budget"],
            window_entity_type="Key",
            window_entity_id=valid_token.token,
            window_start=window_start,
        )
        max_budget = w["max_budget"]
        out.append(
            {
                "budget_duration": w["budget_duration"],
                "max_budget": max_budget,
                "spend": spent,
                # Clamped at 0: a window can end up marginally over its cap
                # between the enforcing check and this read, and reporting a
                # negative remaining budget reads as a bug to the caller.
                "remaining": (
                    max(0.0, max_budget - spent) if max_budget is not None else None
                ),
                "window_start": window_start.isoformat() if window_start else None,
            }
        )
    return out


async def _per_model_rollup(api_key: str) -> List[Dict[str, Any]]:
    """Spend and tokens per model for this key, newest-first by spend.

    Filtered on ``api_key`` alone — never user_id or team_id, which is what
    makes this strictly narrower than /user/daily/activity.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return []

    rows = await prisma_client.db.litellm_dailyuserspend.group_by(
        by=["model"],
        where={"api_key": api_key},
        sum={
            "spend": True,
            "prompt_tokens": True,
            "completion_tokens": True,
            "api_requests": True,
            "successful_requests": True,
            "failed_requests": True,
        },
    )

    rollup: List[Dict[str, Any]] = []
    for row in rows:
        totals = row.get("_sum") or {}
        rollup.append(
            {
                "model": row.get("model"),
                "spend": float(totals.get("spend") or 0.0),
                "prompt_tokens": int(totals.get("prompt_tokens") or 0),
                "completion_tokens": int(totals.get("completion_tokens") or 0),
                "api_requests": int(totals.get("api_requests") or 0),
                "successful_requests": int(totals.get("successful_requests") or 0),
                "failed_requests": int(totals.get("failed_requests") or 0),
            }
        )
    rollup.sort(key=lambda r: r["spend"], reverse=True)
    return rollup


@router.get(
    "/v1/usage",
    tags=["budget & spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def key_usage(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Return THIS key's usage. Scoped entirely by the presented credential.

    Deliberately parameterless. See the module docstring: adding any caller
    -supplied identity argument turns an unrepresentable cross-tenant read
    into a merely-filtered one.
    """
    try:
        response: Dict[str, Any] = {
            # The masked form (last 4 chars) so a caller can tell which key
            # answered without the response echoing the secret back.
            "key_alias": user_api_key_dict.key_alias,
            "key_name": user_api_key_dict.key_name,
            # Lifetime, NOT a window. Named so no client mistakes it for
            # "spend this month" and subtracts it from a monthly cap.
            "lifetime_spend": user_api_key_dict.spend,
            "max_budget": user_api_key_dict.max_budget,
            "budget_windows": await _window_budgets(user_api_key_dict),
            "models": [],
        }

        if user_api_key_dict.api_key:
            response["models"] = await _per_model_rollup(user_api_key_dict.api_key)

        return response
    except Exception as e:
        verbose_proxy_logger.exception("litellm.proxy.key_usage(): %s", str(e))
        raise handle_exception_on_proxy(e)
