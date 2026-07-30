"""Self-service usage read for the holder of a virtual key.

A key holder can see what their own key has spent, broken down by model, and how
much of each budget window is left. Nothing else. No admin credential, no master
key, no database access.

Why this has to live server-side. Everything a client can observe is either the
wrong quantity or unavailable:

- The binding cap on a key is a budget *window* (a daily and/or monthly limit),
  and each window is metered by its own counter, `spend:key:<token>:window:<dur>`.
  No response header carries the cap or that counter.
- The one spend figure a client can see, `x-litellm-key-spend`, is the token's
  cumulative total. Subtracting it from a monthly cap answers a question nobody
  asked, and gets it wrong once the key outlives a single window.
- Per-model attribution lives in `LiteLLM_DailyUserSpend`, which a key holder
  cannot query.

The scoping rule, which is the whole security design: every value returned is
derived from `user_api_key_dict.api_key` and nothing else. This module
deliberately accepts no caller-supplied identity parameter -- no `key`, no
`user_id`, no `team_id`. That makes a cross-tenant read *unrepresentable* rather
than merely filtered, so it cannot be reintroduced by a later change that adds a
convenience parameter for support triage. A test asserts the absence.
"""

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed
from litellm.types.utils import LiteLLMPydanticObjectBase

router = APIRouter()

DEFAULT_LOOKBACK_DAYS = 30
MAX_LOOKBACK_DAYS = 90


class SelfUsageBudgetWindow(LiteLLMPydanticObjectBase):
    """One budget window on the key, with its own spend and reset."""

    budget_duration: str
    max_budget: float
    spend: float
    remaining: float
    resets_at: Optional[str] = None


class SelfUsageModelRow(LiteLLMPydanticObjectBase):
    """What one model cost this key over the requested range."""

    model: str
    spend: float
    prompt_tokens: int
    completion_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    api_requests: int
    successful_requests: int
    failed_requests: int


class SelfUsageResponse(LiteLLMPydanticObjectBase):
    """The complete response shape.

    This is a frozen allowlist, and the freeze is the point. The key row carries
    `metadata` (which on broker-minted keys holds the email of the employee who
    minted it), `team_id`, `user_id` and internal ids; none of them belong to the
    key holder and none of them appear here. A test asserts the field set so a
    later change cannot widen it by accident.
    """

    key_alias: Optional[str] = None
    spend: float
    budgets: List[SelfUsageBudgetWindow]
    by_model: List[SelfUsageModelRow]
    start_date: str
    end_date: str
    as_of: str


def _coerce_window(window: Any) -> Dict[str, Any]:
    """Budget windows arrive as dicts, pydantic objects, or JSON strings.

    `UserAPIKeyAuth.budget_limits` is typed `Optional[List[dict]]`, but every
    other consumer in the codebase defends against the other two shapes because
    they do occur in practice. Matching that tolerance here is cheaper than
    being the one reader that raises.
    """
    if isinstance(window, dict):
        return window
    model_dump = getattr(window, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    return {}


async def _window_spend(token: str, window: Dict[str, Any]) -> float:
    """Spend on one budget window, read without side effects.

    `get_current_spend` is called with `max_budget=None` on purpose. Passing the
    cap turns it into an enforcement path: it repairs the counter, writes back to
    Redis, and can raise a 503 when the value cannot be verified. A read-only
    usage endpoint must not mutate cache state and must not fail because Redis
    blinked.

    The cost of that choice is the cold-counter trap: with `max_budget=None` a
    window counter that Redis has never seen returns the fallback (0.0) rather
    than the truth, because the reseed path deliberately refuses window keys --
    they have no DB row. So the spend-log sum is computed as an independent floor
    and the larger of the two is reported. Under-reporting spend here would
    overstate someone's remaining budget, which is the one direction that costs
    them money.
    """
    from litellm.proxy.proxy_server import get_current_spend
    from litellm.proxy.spend_tracking.budget_reservation import get_budget_window_start

    window_start = get_budget_window_start(window)
    counter_key = f"spend:key:{token}:window:{window['budget_duration']}"

    counter_spend = 0.0
    try:
        counter_spend = await get_current_spend(
            counter_key=counter_key,
            fallback_spend=0.0,
            max_budget=None,
            window_entity_type="Key",
            window_entity_id=token,
            window_start=window_start,
        )
    except Exception:
        # A counter read failure must not fail the whole report -- the spend-log
        # floor below is the more authoritative number anyway.
        counter_spend = 0.0

    # Imported at module scope, NOT inside the try below. An ImportError here
    # would mean the floor silently never runs, and a broad except would hide
    # that forever -- which is exactly what happened while writing this: the
    # import was on the wrong path and every reported window spend would have
    # quietly been the cold-counter zero.
    logged_spend = 0.0
    if window_start is not None:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is not None:
            try:
                logged = await SpendCounterReseed.window_from_spend_logs(
                    prisma_client=prisma_client,
                    entity_type="Key",
                    entity_id=token,
                    window_start=window_start,
                )
                logged_spend = float(logged or 0.0)
            except Exception:
                # A query failure degrades to the counter value rather than
                # failing the report. An import failure cannot land here.
                logged_spend = 0.0

    return max(counter_spend, logged_spend)


async def _budgets_for_key(user_api_key_dict: UserAPIKeyAuth) -> List[SelfUsageBudgetWindow]:
    """Every budget window on this key, with spend and headroom.

    Returns an empty list when the key carries no windows -- which is a real
    state and must not be reported as "no limit", because a team-level cap may
    still bind. The caller is told what this key has, not what it is free to do.
    """
    token = user_api_key_dict.token or user_api_key_dict.api_key
    if not token or not user_api_key_dict.budget_limits:
        return []

    windows: List[SelfUsageBudgetWindow] = []
    for raw in user_api_key_dict.budget_limits:
        window = _coerce_window(raw)
        duration = window.get("budget_duration")
        cap = window.get("max_budget")
        if duration is None or cap is None:
            continue
        spend = await _window_spend(token=token, window=window)
        cap_float = float(cap)
        # An infinite cap is a real configuration; reporting `inf` remaining
        # would serialize as a non-JSON value, so it is reported as None.
        remaining = cap_float - spend if math.isfinite(cap_float) else None
        reset_at = window.get("reset_at")
        windows.append(
            SelfUsageBudgetWindow(
                budget_duration=str(duration),
                max_budget=cap_float,
                spend=spend,
                remaining=remaining if remaining is not None else 0.0,
                resets_at=str(reset_at) if reset_at else None,
            )
        )
    return windows


async def _usage_by_model(
    api_key_hash: str, start_date: str, end_date: str
) -> List[SelfUsageModelRow]:
    """Per-model rollup for THIS key alone.

    Filtered on `api_key` and nothing else. That is strictly narrower than the
    user-scoped daily-activity endpoint, which returns every key belonging to a
    user, and it is why a second key under the same user cannot appear here.
    `LiteLLM_DailyUserSpend.api_key` holds the same sha256 hash as
    `UserAPIKeyAuth.api_key`, so no re-hashing is needed, and the column is
    indexed.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    rows = await prisma_client.db.litellm_dailyuserspend.find_many(
        where={
            "api_key": api_key_hash,
            "date": {"gte": start_date, "lte": end_date},
        },
    )

    totals: Dict[str, Dict[str, float]] = {}
    for row in rows:
        model = getattr(row, "model", None) or "unknown"
        bucket = totals.setdefault(
            model,
            {
                "spend": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "api_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
            },
        )
        bucket["spend"] += float(getattr(row, "spend", 0.0) or 0.0)
        for field in (
            "prompt_tokens",
            "completion_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "api_requests",
            "successful_requests",
            "failed_requests",
        ):
            bucket[field] += int(getattr(row, field, 0) or 0)

    return [
        SelfUsageModelRow(
            model=model,
            spend=values["spend"],
            prompt_tokens=int(values["prompt_tokens"]),
            completion_tokens=int(values["completion_tokens"]),
            cache_read_input_tokens=int(values["cache_read_input_tokens"]),
            cache_creation_input_tokens=int(values["cache_creation_input_tokens"]),
            api_requests=int(values["api_requests"]),
            successful_requests=int(values["successful_requests"]),
            failed_requests=int(values["failed_requests"]),
        )
        # Most expensive first: the question behind this endpoint is almost
        # always "what is eating my budget".
        for model, values in sorted(totals.items(), key=lambda kv: -kv[1]["spend"])
    ]


@router.get(
    "/v1/usage",
    tags=["budget & spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=SelfUsageResponse,
)
async def self_usage(
    start_date: Optional[str] = Query(
        default=None, description="YYYY-MM-DD, inclusive. Defaults to 30 days ago."
    ),
    end_date: Optional[str] = Query(
        default=None, description="YYYY-MM-DD, inclusive. Defaults to today (UTC)."
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> SelfUsageResponse:
    """What has MY key spent, and how much of my budget is left.

    Scoped entirely to the calling key. There is deliberately no parameter that
    names a key, user or team: the only identity this handler can express is the
    one that authenticated, so it cannot be pointed at anyone else.
    """
    api_key_hash = user_api_key_dict.api_key
    if not api_key_hash:
        raise HTTPException(
            status_code=401,
            detail={"error": "No authenticated key on this request."},
        )

    today = datetime.now(timezone.utc).date()
    resolved_end = end_date or today.isoformat()
    resolved_start = start_date or (today - timedelta(days=DEFAULT_LOOKBACK_DAYS)).isoformat()

    try:
        parsed_start = datetime.strptime(resolved_start, "%Y-%m-%d").date()
        parsed_end = datetime.strptime(resolved_end, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={"error": "start_date and end_date must be YYYY-MM-DD."},
        )
    if parsed_start > parsed_end:
        raise HTTPException(
            status_code=400,
            detail={"error": "start_date must be on or before end_date."},
        )
    # Bounded so one request cannot ask the daily table for an unbounded scan.
    if (parsed_end - parsed_start).days > MAX_LOOKBACK_DAYS:
        raise HTTPException(
            status_code=400,
            detail={"error": f"range must be {MAX_LOOKBACK_DAYS} days or fewer."},
        )

    budgets = await _budgets_for_key(user_api_key_dict)
    by_model = await _usage_by_model(
        api_key_hash=api_key_hash,
        start_date=resolved_start,
        end_date=resolved_end,
    )

    return SelfUsageResponse(
        key_alias=user_api_key_dict.key_alias,
        spend=float(user_api_key_dict.spend or 0.0),
        budgets=budgets,
        by_model=by_model,
        start_date=resolved_start,
        end_date=resolved_end,
        as_of=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )
