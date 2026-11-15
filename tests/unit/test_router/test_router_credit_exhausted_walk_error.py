"""A walk that ends with no deployment left because the legs left are held out on a spent prepaid credit
raises the request's own error, not "No deployments available, Try again in N seconds": waiting does not
bring credit back. A walk that ran out on a real rate limit keeps the router's error."""

from collections.abc import Awaitable, Callable, Mapping
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.router import Router
from litellm.router_rust.surface.health import held_out_on_spent_credential
from litellm.types.router import RouterRateLimitError

API_BASE: Final = "https://anthropic.example.invalid"
FIRST_KEY: Final = "sk-first-leg"
SECOND_KEY: Final = "sk-second-leg"
SPENT_KEY: Final = "sk-spent"
OTHER_SPENT_KEY: Final = "sk-other-spent"
CREDIT_COOLDOWN_SECONDS: Final = 3600.0
CREDIT_TEXT: Final = (
    "Your credit balance is too low to access the Anthropic API. "
    "Please go to Plans & Billing to upgrade or purchase credits."
)
FIRST_LEG_REJECTION: Final = "first leg rejected the request"
SECOND_LEG_REJECTION: Final = "second leg rejected the request"
SECOND_LEG_OUTAGE: Final = "second leg is unavailable"
KEYS: Final = (FIRST_KEY, SECOND_KEY, SPENT_KEY, OTHER_SPENT_KEY)


def _error(status: int, error_type: str, message: str) -> httpx.Response:
    return httpx.Response(status, json={"type": "error", "error": {"type": error_type, "message": message}})


SPENT: Final = _error(400, "invalid_request_error", CREDIT_TEXT)
RATE_LIMITED: Final = _error(429, "rate_limit_error", "Number of request tokens has exceeded your rate limit")


def _leg(model_id: str, api_key: str, order: int) -> dict[str, object]:
    return {
        "model_name": "claude",
        "litellm_params": {"model": "anthropic/claude-x", "api_key": api_key, "api_base": API_BASE, "order": order},
        "model_info": {"id": model_id},
    }


def _router(*legs: tuple[str, str, int]) -> Router:
    return Router(
        model_list=[_leg(model_id, api_key, order) for model_id, api_key, order in legs],
        num_retries=2,
        retry_after=0,
        cooldown_time=30.0,
    )


async def _chat(router: Router, stream: bool) -> None:
    response: Final = await router.acompletion(
        model="claude", messages=[{"role": "user", "content": "hi"}], stream=stream
    )
    if stream:
        _ = [chunk async for chunk in response]


async def _messages(router: Router, stream: bool) -> None:
    response: Final = await router.aanthropic_messages(
        model="claude", messages=[{"role": "user", "content": "hi"}], max_tokens=10, stream=stream
    )
    if stream:
        _ = [chunk async for chunk in response]


async def _raised(send: Callable[[Router, bool], Awaitable[None]], router: Router, stream: bool) -> Exception:
    """The error the request raised: which class it is, is what each test asserts."""
    try:
        await send(router, stream)
    except Exception as error:  # noqa: BLE001  # any class: the test asserts which one
        return error
    raise AssertionError("the request was served")


def _calls_by_key(route: respx.Route, since: int = 0) -> dict[str, int]:
    """The upstream calls per key from call ``since`` on."""
    keys: Final = [call.request.headers["x-api-key"] for call in list(route.calls)[since:]]
    return {key: keys.count(key) for key in KEYS if key in keys}


@pytest.fixture
def anthropic_upstream(monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter) -> respx.Route:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "suppress_debug_info", True)
    monkeypatch.setattr(litellm, "credit_exhausted_cooldown_time", CREDIT_COOLDOWN_SECONDS)
    return respx_mock.post(f"{API_BASE}/v1/messages")


def _answer(answers: Mapping[str, httpx.Response]) -> Callable[[httpx.Request], httpx.Response]:
    def answer(request: httpx.Request) -> httpx.Response:
        return answers[request.headers["x-api-key"]]

    return answer


ROUTES: Final = pytest.mark.parametrize("send", [_chat, _messages], ids=["chat", "messages"])
STREAMS: Final = pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])


@pytest.mark.asyncio
@ROUTES
@STREAMS
async def test_a_bad_request_that_walks_onto_a_spent_leg_answers_with_the_bad_request(
    send: Callable[[Router, bool], Awaitable[None]], stream: bool, anthropic_upstream: respx.Route
) -> None:
    anthropic_upstream.mock(
        side_effect=_answer(
            {
                FIRST_KEY: _error(400, "invalid_request_error", FIRST_LEG_REJECTION),
                SECOND_KEY: _error(400, "invalid_request_error", SECOND_LEG_REJECTION),
                SPENT_KEY: SPENT,
            }
        )
    )
    router: Final = _router(("first", FIRST_KEY, 0), ("second", SECOND_KEY, 1), ("spent", SPENT_KEY, 2))

    first: Final = await _raised(send, router, stream)
    first_calls: Final = _calls_by_key(anthropic_upstream)
    called: Final = anthropic_upstream.call_count
    # Within the hold-out the spent leg is skipped without a call, and the answer is the same.
    again: Final = await _raised(send, router, stream)

    assert first_calls == {FIRST_KEY: 1, SECOND_KEY: 1, SPENT_KEY: 1}
    assert _calls_by_key(anthropic_upstream, called) == {FIRST_KEY: 1, SECOND_KEY: 1}
    for raised in (first, again):
        assert isinstance(raised, litellm.BadRequestError)
        assert raised.status_code == 400
        assert SECOND_LEG_REJECTION in str(raised)


@pytest.mark.asyncio
@ROUTES
@STREAMS
async def test_a_server_error_that_walks_onto_a_spent_leg_answers_with_the_server_error(
    send: Callable[[Router, bool], Awaitable[None]], stream: bool, anthropic_upstream: respx.Route
) -> None:
    anthropic_upstream.mock(
        side_effect=_answer({SECOND_KEY: _error(503, "api_error", SECOND_LEG_OUTAGE), SPENT_KEY: SPENT})
    )
    router: Final = _router(("second", SECOND_KEY, 1), ("spent", SPENT_KEY, 2))

    first: Final = await _raised(send, router, stream)
    first_calls: Final = _calls_by_key(anthropic_upstream)
    called: Final = anthropic_upstream.call_count
    again: Final = await _raised(send, router, stream)

    assert first_calls == {SECOND_KEY: 3, SPENT_KEY: 1}
    # The second request cools the failing leg (allowed fails) and never calls the spent one.
    assert SPENT_KEY not in _calls_by_key(anthropic_upstream, called)
    for raised in (first, again):
        assert not isinstance(raised, RouterRateLimitError | litellm.RateLimitError)
        assert getattr(raised, "status_code", None) == 503
        assert SECOND_LEG_OUTAGE in str(raised)


@pytest.mark.asyncio
@ROUTES
@STREAMS
async def test_every_leg_spent_answers_with_the_credit_error(
    send: Callable[[Router, bool], Awaitable[None]], stream: bool, anthropic_upstream: respx.Route
) -> None:
    anthropic_upstream.mock(side_effect=_answer({SPENT_KEY: SPENT, OTHER_SPENT_KEY: SPENT}))
    router: Final = _router(("spent", SPENT_KEY, 0), ("other-spent", OTHER_SPENT_KEY, 1))

    first: Final = await _raised(send, router, stream)
    first_calls: Final = _calls_by_key(anthropic_upstream)
    called: Final = anthropic_upstream.call_count
    again: Final = await _raised(send, router, stream)

    assert first_calls == {SPENT_KEY: 1, OTHER_SPENT_KEY: 1}
    assert _calls_by_key(anthropic_upstream, called) == {}
    assert CREDIT_TEXT in str(first)
    assert "prepaid credit" in str(again)
    for raised in (first, again):
        assert isinstance(raised, litellm.CreditBalanceExhaustedError)
        assert raised.status_code == 429
        assert "Try again in" not in str(raised)


@pytest.mark.asyncio
async def test_a_bad_request_onto_a_spent_leg_answers_with_the_bad_request_without_a_credit_cooldown(
    anthropic_upstream: respx.Route, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "credit_exhausted_cooldown_time", None)
    anthropic_upstream.mock(
        side_effect=_answer({FIRST_KEY: _error(400, "invalid_request_error", FIRST_LEG_REJECTION), SPENT_KEY: SPENT})
    )
    router: Final = _router(("first", FIRST_KEY, 0), ("spent", SPENT_KEY, 1))

    raised: Final = await _raised(_chat, router, False)

    assert isinstance(raised, litellm.BadRequestError)
    assert FIRST_LEG_REJECTION in str(raised)


def test_a_sync_bad_request_onto_a_spent_leg_answers_with_the_bad_request(anthropic_upstream: respx.Route) -> None:
    anthropic_upstream.mock(
        side_effect=_answer({FIRST_KEY: _error(400, "invalid_request_error", FIRST_LEG_REJECTION), SPENT_KEY: SPENT})
    )
    router: Final = _router(("first", FIRST_KEY, 0), ("spent", SPENT_KEY, 1))

    for _ in range(2):
        with pytest.raises(litellm.BadRequestError) as raised:
            _ = router.completion(model="claude", messages=[{"role": "user", "content": "hi"}])
        assert FIRST_LEG_REJECTION in str(raised.value)
    assert _calls_by_key(anthropic_upstream) == {FIRST_KEY: 2, SPENT_KEY: 1}


@pytest.mark.asyncio
@ROUTES
async def test_rate_limited_legs_keep_the_no_deployments_error(
    send: Callable[[Router, bool], Awaitable[None]], anthropic_upstream: respx.Route
) -> None:
    anthropic_upstream.mock(side_effect=_answer({FIRST_KEY: RATE_LIMITED, SECOND_KEY: RATE_LIMITED}))
    router: Final = _router(("first", FIRST_KEY, 0), ("second", SECOND_KEY, 0))

    first: Final = await _raised(send, router, False)
    again: Final = await _raised(send, router, False)

    assert type(first) is litellm.RateLimitError
    assert isinstance(again, RouterRateLimitError)
    assert "Try again in 30" in str(again)


@pytest.mark.asyncio
@ROUTES
async def test_a_rate_limited_leg_beside_a_spent_leg_keeps_the_no_deployments_error(
    send: Callable[[Router, bool], Awaitable[None]], anthropic_upstream: respx.Route
) -> None:
    anthropic_upstream.mock(side_effect=_answer({FIRST_KEY: RATE_LIMITED, SPENT_KEY: SPENT}))
    router: Final = _router(("first", FIRST_KEY, 0), ("spent", SPENT_KEY, 1))

    first: Final = await _raised(send, router, False)
    first_calls: Final = _calls_by_key(anthropic_upstream)
    called: Final = anthropic_upstream.call_count
    # The rate-limited leg is cooled on its 429 and the spent leg held out: neither is called.
    again: Final = await _raised(send, router, False)

    assert first_calls == {FIRST_KEY: 1, SPENT_KEY: 1}
    assert _calls_by_key(anthropic_upstream, called) == {}
    assert isinstance(first, RouterRateLimitError)
    assert isinstance(again, RouterRateLimitError)


@pytest.mark.asyncio
@ROUTES
@STREAMS
async def test_another_router_on_the_shared_cooldowns_answers_with_the_bad_request(
    send: Callable[[Router, bool], Awaitable[None]],
    stream: bool,
    anthropic_upstream: respx.Route,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anthropic_upstream.mock(
        side_effect=_answer(
            {
                FIRST_KEY: _error(400, "invalid_request_error", FIRST_LEG_REJECTION),
                SECOND_KEY: _error(400, "invalid_request_error", SECOND_LEG_REJECTION),
                SPENT_KEY: SPENT,
            }
        )
    )
    legs: Final = (("first", FIRST_KEY, 0), ("second", SECOND_KEY, 1), ("spent", SPENT_KEY, 2))
    # Two instances on one cooldown store, as instances behind one Redis: only the first sees the credit error.
    seen_it: Final = _router(*legs)
    other: Final = _router(*legs)
    monkeypatch.setattr(other.cooldown_cache, "cooldown_store", seen_it.cooldown_cache.cooldown_store)

    _ = await _raised(send, seen_it, stream)
    called: Final = anthropic_upstream.call_count
    raised: Final = await _raised(send, other, stream)

    shared: Final = dict(other.cooldown_cache.get_active_cooldowns(["spent"], parent_otel_span=None))
    assert held_out_on_spent_credential(shared["spent"])
    assert float(shared["spent"]["cooldown_time"]) == CREDIT_COOLDOWN_SECONDS
    assert _calls_by_key(anthropic_upstream, called) == {FIRST_KEY: 1, SECOND_KEY: 1}
    assert isinstance(raised, litellm.BadRequestError)
    assert SECOND_LEG_REJECTION in str(raised)
    assert "Try again in" not in str(raised)
