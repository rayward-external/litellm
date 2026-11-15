"""A walk that ends with no deployment left because the legs left are held out on a spent prepaid credit
raises the request's own error, not "No deployments available, Try again in N seconds": waiting does not
bring credit back. A walk that ran out on a real rate limit keeps the router's error.

The chat route runs on Python's httpx transport, mocked with respx. Anthropic `/v1/messages` runs on the Rust
path only, whose HTTP client respx cannot see, so its cases answer from a loopback upstream on a real socket."""

import json
from collections.abc import Awaitable, Callable, Iterator, Mapping
from typing import Final

import httpx
import pytest

import litellm
from litellm.router import Router
from litellm.router_rust.surface.health import held_out_on_spent_credential
from litellm.types.router import RouterRateLimitError
from tests.unit.test_router.loopback_upstream import RespxUpstream, Upstream, loopback_upstream

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


def _leg(api_base: str, model_id: str, api_key: str, order: int) -> dict[str, object]:
    return {
        "model_name": "claude",
        "litellm_params": {"model": "anthropic/claude-x", "api_key": api_key, "api_base": api_base, "order": order},
        "model_info": {"id": model_id},
    }


def _router(api_base: str, *legs: tuple[str, str, int]) -> Router:
    return Router(
        model_list=[_leg(api_base, model_id, api_key, order) for model_id, api_key, order in legs],
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


def _calls_by_key(route: Upstream, since: int = 0) -> dict[str, int]:
    """The upstream calls per key from call ``since`` on."""
    keys: Final = [call.request.headers["x-api-key"] for call in list(route.calls)[since:]]
    return {key: keys.count(key) for key in KEYS if key in keys}


@pytest.fixture
def anthropic_upstream(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[Upstream]:
    """``respx`` mocks Python's httpx transport; ``loopback`` serves the Rust path from a real socket."""
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "suppress_debug_info", True)
    monkeypatch.setattr(litellm, "credit_exhausted_cooldown_time", CREDIT_COOLDOWN_SECONDS)
    if request.param == "respx":
        yield RespxUpstream(request.getfixturevalue("respx_mock"), "/v1/messages")
        return
    with loopback_upstream() as upstream:
        yield upstream


def _answer(answers: Mapping[str, httpx.Response]) -> Callable[[httpx.Request], httpx.Response]:
    def answer(request: httpx.Request) -> httpx.Response:
        return answers[request.headers["x-api-key"]]

    return answer


ROUTES: Final = pytest.mark.parametrize(
    ("send", "anthropic_upstream"),
    [(_chat, "respx"), (_messages, "loopback")],
    ids=["chat", "messages"],
    indirect=["anthropic_upstream"],
)
CHAT: Final = pytest.mark.parametrize("anthropic_upstream", ["respx"], ids=["chat"], indirect=True)
STREAMS: Final = pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])


@pytest.mark.asyncio
@ROUTES
@STREAMS
async def test_a_bad_request_that_walks_onto_a_spent_leg_answers_with_the_bad_request(
    send: Callable[[Router, bool], Awaitable[None]], stream: bool, anthropic_upstream: Upstream
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
    router: Final = _router(
        anthropic_upstream.api_base, ("first", FIRST_KEY, 0), ("second", SECOND_KEY, 1), ("spent", SPENT_KEY, 2)
    )

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
    send: Callable[[Router, bool], Awaitable[None]], stream: bool, anthropic_upstream: Upstream
) -> None:
    anthropic_upstream.mock(
        side_effect=_answer({SECOND_KEY: _error(503, "api_error", SECOND_LEG_OUTAGE), SPENT_KEY: SPENT})
    )
    router: Final = _router(anthropic_upstream.api_base, ("second", SECOND_KEY, 1), ("spent", SPENT_KEY, 2))

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
    send: Callable[[Router, bool], Awaitable[None]], stream: bool, anthropic_upstream: Upstream
) -> None:
    anthropic_upstream.mock(side_effect=_answer({SPENT_KEY: SPENT, OTHER_SPENT_KEY: SPENT}))
    router: Final = _router(anthropic_upstream.api_base, ("spent", SPENT_KEY, 0), ("other-spent", OTHER_SPENT_KEY, 1))

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
@CHAT
async def test_a_bad_request_onto_a_spent_leg_answers_with_the_bad_request_without_a_credit_cooldown(
    anthropic_upstream: Upstream, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "credit_exhausted_cooldown_time", None)
    anthropic_upstream.mock(
        side_effect=_answer({FIRST_KEY: _error(400, "invalid_request_error", FIRST_LEG_REJECTION), SPENT_KEY: SPENT})
    )
    router: Final = _router(anthropic_upstream.api_base, ("first", FIRST_KEY, 0), ("spent", SPENT_KEY, 1))

    raised: Final = await _raised(_chat, router, False)

    assert isinstance(raised, litellm.BadRequestError)
    assert FIRST_LEG_REJECTION in str(raised)


@CHAT
def test_a_sync_bad_request_onto_a_spent_leg_answers_with_the_bad_request(anthropic_upstream: Upstream) -> None:
    anthropic_upstream.mock(
        side_effect=_answer({FIRST_KEY: _error(400, "invalid_request_error", FIRST_LEG_REJECTION), SPENT_KEY: SPENT})
    )
    router: Final = _router(anthropic_upstream.api_base, ("first", FIRST_KEY, 0), ("spent", SPENT_KEY, 1))

    for _ in range(2):
        with pytest.raises(litellm.BadRequestError) as raised:
            _ = router.completion(model="claude", messages=[{"role": "user", "content": "hi"}])
        assert FIRST_LEG_REJECTION in str(raised.value)
    assert _calls_by_key(anthropic_upstream) == {FIRST_KEY: 2, SPENT_KEY: 1}


@pytest.mark.asyncio
@ROUTES
async def test_rate_limited_legs_keep_the_no_deployments_error(
    send: Callable[[Router, bool], Awaitable[None]], anthropic_upstream: Upstream
) -> None:
    anthropic_upstream.mock(side_effect=_answer({FIRST_KEY: RATE_LIMITED, SECOND_KEY: RATE_LIMITED}))
    router: Final = _router(anthropic_upstream.api_base, ("first", FIRST_KEY, 0), ("second", SECOND_KEY, 0))

    first: Final = await _raised(send, router, False)
    again: Final = await _raised(send, router, False)

    assert type(first) is litellm.RateLimitError
    assert isinstance(again, RouterRateLimitError)
    assert "Try again in 30" in str(again)


@pytest.mark.asyncio
@ROUTES
async def test_a_rate_limited_leg_beside_a_spent_leg_keeps_the_no_deployments_error(
    send: Callable[[Router, bool], Awaitable[None]], anthropic_upstream: Upstream
) -> None:
    anthropic_upstream.mock(side_effect=_answer({FIRST_KEY: RATE_LIMITED, SPENT_KEY: SPENT}))
    router: Final = _router(anthropic_upstream.api_base, ("first", FIRST_KEY, 0), ("spent", SPENT_KEY, 1))

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
    anthropic_upstream: Upstream,
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
    seen_it: Final = _router(anthropic_upstream.api_base, *legs)
    other: Final = _router(anthropic_upstream.api_base, *legs)
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


STREAM_OUTAGE: Final = "stream overloaded before any content"


@pytest.mark.asyncio
@ROUTES
async def test_a_stream_that_fails_before_content_onto_a_spent_leg_answers_with_the_stream_failure(
    send: Callable[[Router, bool], Awaitable[None]], anthropic_upstream: Upstream
) -> None:
    failing_stream: Final = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=(
            "event: error\n"
            f"data: {json.dumps({'type': 'error', 'error': {'type': 'overloaded_error', 'message': STREAM_OUTAGE}})}\n\n"
        ).encode(),
    )
    anthropic_upstream.mock(side_effect=_answer({FIRST_KEY: failing_stream, SPENT_KEY: SPENT}))
    router: Final = _router(anthropic_upstream.api_base, ("first", FIRST_KEY, 0), ("spent", SPENT_KEY, 1))

    # The recovery walk after the stream's failure reaches only the spent leg.
    raised: Final = await _raised(send, router, True)

    assert _calls_by_key(anthropic_upstream) == {FIRST_KEY: 3, SPENT_KEY: 1}
    assert not isinstance(raised, litellm.CreditBalanceExhaustedError | RouterRateLimitError)
    assert getattr(raised, "status_code", None) == 500
    assert STREAM_OUTAGE in str(raised)


@pytest.mark.asyncio
@ROUTES
async def test_a_spent_group_whose_fallback_group_is_rate_limited_keeps_the_no_deployments_error(
    send: Callable[[Router, bool], Awaitable[None]], anthropic_upstream: Upstream
) -> None:
    anthropic_upstream.mock(
        side_effect=_answer(
            {SPENT_KEY: SPENT, OTHER_SPENT_KEY: SPENT, FIRST_KEY: RATE_LIMITED, SECOND_KEY: RATE_LIMITED}
        )
    )
    api_base: Final = anthropic_upstream.api_base
    spent_group: Final = [_leg(api_base, "spent", SPENT_KEY, 0), _leg(api_base, "other-spent", OTHER_SPENT_KEY, 1)]
    fallback_group: Final = [
        {**_leg(api_base, model_id, api_key, 0), "model_name": "claude-fallback"}
        for model_id, api_key in (("first", FIRST_KEY), ("second", SECOND_KEY))
    ]
    router: Final = Router(
        model_list=[*spent_group, *fallback_group],
        fallbacks=[{"claude": ["claude-fallback"]}],
        num_retries=2,
        retry_after=0,
        cooldown_time=30.0,
    )

    _ = await _raised(send, router, False)
    called: Final = anthropic_upstream.call_count
    # Both groups are now cooling: the requested one on its spent credit, its fallback on a rate limit.
    again: Final = await _raised(send, router, False)

    assert _calls_by_key(anthropic_upstream, called) == {}
    assert isinstance(again, RouterRateLimitError)
    assert not isinstance(again, litellm.CreditBalanceExhaustedError)
