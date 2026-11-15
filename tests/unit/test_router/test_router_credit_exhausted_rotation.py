"""Two keys of one provider at the same order, a third leg at the next: when the first key's account is out
of credit the same request is served by its sibling, and every deployment on the spent key is held out.

The chat route runs on Python's httpx transport, mocked with respx. Anthropic `/v1/messages` runs on the Rust
path only, whose HTTP client respx cannot see, so its cases answer from a loopback upstream on a real socket."""

import json
from collections.abc import Awaitable, Callable, Iterator, Mapping
from typing import Final

import httpx
import pytest

import litellm
from litellm.models.credentials import CredentialItem
from litellm.redis.stores.router_cooldowns import RouterCooldowns
from litellm.router import Router
from litellm.types.router import RetryPolicy

from tests.unit.test_router.loopback_upstream import RespxUpstream, Upstream, loopback_upstream

SPENT_KEY: Final = "sk-spent"
FUNDED_KEY: Final = "sk-funded"
NEXT_ORDER_KEY: Final = "sk-next-order"
ROUTER_COOLDOWN_SECONDS: Final = 30.0
CREDIT_COOLDOWN_SECONDS: Final = 3600.0
CREDIT_TEXT: Final = (
    "Your credit balance is too low to access the Anthropic API. "
    "Please go to Plans & Billing to upgrade or purchase credits."
)
ANSWER: Final = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-x",
    "content": [{"type": "text", "text": "hi"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
STREAMED_ANSWER: Final = b"".join(
    f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
    for event in (
        {"type": "message_start", "message": {**ANSWER, "content": [], "stop_reason": None}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    )
)


def _anthropic_error(message: str) -> httpx.Response:
    return httpx.Response(400, json={"type": "error", "error": {"type": "invalid_request_error", "message": message}})


def _upstream(spent_key_answer: httpx.Response) -> Callable[[httpx.Request], httpx.Response]:
    def answer(request: httpx.Request) -> httpx.Response:
        if request.headers["x-api-key"] == SPENT_KEY:
            return spent_key_answer
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=STREAMED_ANSWER)
        return httpx.Response(200, json=ANSWER)

    return answer


def _leg(
    api_base: str,
    model_id: str,
    api_key: str,
    order: int,
    weight: int,
    model_name: str = "claude",
    leg_settings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    settings: Final = leg_settings or {}
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": "anthropic/claude-x",
            "api_key": api_key,
            "api_base": api_base,
            "order": order,
            "weight": weight,
            **settings.get("litellm_params", {}),
        },
        "model_info": {"id": model_id, **settings.get("model_info", {})},
    }


def _router(
    api_base: str, leg_settings: Mapping[str, Mapping[str, object]] | None = None, **router_settings: object
) -> Router:
    # Weight 0 on the funded key makes the spent key the first pick; a sibling holds it otherwise.
    return Router(
        model_list=[
            _leg(api_base, "spent", SPENT_KEY, order=0, weight=1, leg_settings=leg_settings),
            _leg(api_base, "funded", FUNDED_KEY, order=0, weight=0, leg_settings=leg_settings),
            _leg(api_base, "next-order", NEXT_ORDER_KEY, order=1, weight=1),
            _leg(api_base, "spent-elsewhere", SPENT_KEY, order=0, weight=1, model_name="other-model"),
        ],
        num_retries=2,
        retry_after=0,
        cooldown_time=ROUTER_COOLDOWN_SECONDS,
        **router_settings,
    )


async def _chat_text(router: Router, stream: bool) -> str:
    response: Final = await router.acompletion(
        model="claude", messages=[{"role": "user", "content": "hi"}], stream=stream
    )
    if not stream:
        return response.choices[0].message.content
    return "".join([chunk.choices[0].delta.content or "" async for chunk in response])


async def _messages_text(router: Router, stream: bool) -> str:
    response: Final = await router.aanthropic_messages(
        model="claude", messages=[{"role": "user", "content": "hi"}], max_tokens=10, stream=stream
    )
    if not stream:
        return "".join(block["text"] for block in response["content"])
    events: Final = [
        json.loads(line.removeprefix(b"data: "))
        for line in b"".join([chunk async for chunk in response]).splitlines()
        if line.startswith(b"data: ")
    ]
    return "".join(event["delta"]["text"] for event in events if event["type"] == "content_block_delta")


def _calls_by_key(route: Upstream) -> dict[str, int]:
    keys: Final = [call.request.headers["x-api-key"] for call in route.calls]
    return {key: keys.count(key) for key in (SPENT_KEY, FUNDED_KEY, NEXT_ORDER_KEY)}


def _cooldowns(router: Router) -> dict[str, float]:
    active: Final = router.cooldown_cache.get_active_cooldowns(
        model_ids=["spent", "funded", "next-order", "spent-elsewhere"], parent_otel_span=None
    )
    return {model_id: float(value["cooldown_time"]) for model_id, value in active}


@pytest.fixture
def anthropic_upstream(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[Upstream]:
    """``respx`` mocks Python's httpx transport; ``loopback`` serves the Rust path from a real socket."""
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "suppress_debug_info", True)
    if request.param == "respx":
        yield RespxUpstream(request.getfixturevalue("respx_mock"), "/v1/messages")
        return
    with loopback_upstream() as upstream:
        yield upstream


ROUTES: Final = pytest.mark.parametrize(
    ("served_text", "anthropic_upstream"),
    [(_chat_text, "respx"), (_messages_text, "loopback")],
    ids=["chat", "messages"],
    indirect=["anthropic_upstream"],
)
CHAT: Final = pytest.mark.parametrize("anthropic_upstream", ["respx"], ids=["chat"], indirect=True)
MESSAGES: Final = pytest.mark.parametrize("anthropic_upstream", ["loopback"], ids=["messages"], indirect=True)


@pytest.mark.asyncio
@ROUTES
@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
async def test_a_spent_key_hands_the_request_to_its_sibling_and_leaves_with_its_credential(
    served_text: Callable[[Router, bool], Awaitable[str]],
    stream: bool,
    anthropic_upstream: Upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "credit_exhausted_cooldown_time", CREDIT_COOLDOWN_SECONDS)
    anthropic_upstream.mock(side_effect=_upstream(_anthropic_error(CREDIT_TEXT)))
    router: Final = _router(anthropic_upstream.api_base)

    text: Final = await served_text(router, stream)

    assert text == "hi"
    assert _calls_by_key(anthropic_upstream) == {SPENT_KEY: 1, FUNDED_KEY: 1, NEXT_ORDER_KEY: 0}
    assert _cooldowns(router) == {"spent": CREDIT_COOLDOWN_SECONDS, "spent-elsewhere": CREDIT_COOLDOWN_SECONDS}


@pytest.mark.asyncio
@MESSAGES
async def test_a_retry_policy_and_a_per_leg_cooldown_still_rotate_and_hold_the_spent_key_out(
    anthropic_upstream: Upstream, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "credit_exhausted_cooldown_time", CREDIT_COOLDOWN_SECONDS)
    anthropic_upstream.mock(side_effect=_upstream(_anthropic_error(CREDIT_TEXT)))
    router: Final = _router(
        anthropic_upstream.api_base,
        leg_settings={
            "litellm_params": {"cooldown_time": 60},
            "model_info": {"allowed_fails_policy": {"RateLimitErrorAllowedFails": 0, "BadRequestErrorAllowedFails": 2}},
        },
        retry_policy=RetryPolicy(RateLimitErrorRetries=5, BadRequestErrorRetries=0, AuthenticationErrorRetries=0),
    )

    text: Final = await _messages_text(router, stream=True)

    assert text == "hi"
    assert _calls_by_key(anthropic_upstream) == {SPENT_KEY: 1, FUNDED_KEY: 1, NEXT_ORDER_KEY: 0}
    assert _cooldowns(router) == {"spent": CREDIT_COOLDOWN_SECONDS, "spent-elsewhere": CREDIT_COOLDOWN_SECONDS}


@pytest.mark.asyncio
@CHAT
async def test_without_a_credit_cooldown_the_spent_key_keeps_the_routers_cooldown(
    anthropic_upstream: Upstream, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "credit_exhausted_cooldown_time", None)
    anthropic_upstream.mock(side_effect=_upstream(_anthropic_error(CREDIT_TEXT)))
    router: Final = _router(anthropic_upstream.api_base)

    text: Final = await _chat_text(router, stream=False)

    assert text == "hi"
    assert _cooldowns(router) == {"spent": ROUTER_COOLDOWN_SECONDS, "spent-elsewhere": ROUTER_COOLDOWN_SECONDS}


@pytest.mark.asyncio
@ROUTES
async def test_a_bad_request_never_moves_to_a_sibling_key_or_cools_the_leg(
    served_text: Callable[[Router, bool], Awaitable[str]],
    anthropic_upstream: Upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "credit_exhausted_cooldown_time", CREDIT_COOLDOWN_SECONDS)
    anthropic_upstream.mock(side_effect=_upstream(_anthropic_error("max_tokens: Field required")))
    router: Final = _router(anthropic_upstream.api_base)

    _ = await served_text(router, False)

    # The order chain's own fallback (unchanged): no retry within the order, the next order answers.
    assert _calls_by_key(anthropic_upstream) == {SPENT_KEY: 1, FUNDED_KEY: 0, NEXT_ORDER_KEY: 1}
    assert _cooldowns(router) == {}


@pytest.mark.asyncio
@CHAT
async def test_a_shared_credential_name_with_its_own_key_is_not_held_out_with_the_spent_key(
    anthropic_upstream: Upstream, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "credit_exhausted_cooldown_time", CREDIT_COOLDOWN_SECONDS)
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="shared", credential_values={"api_key": "sk-from-credential"}, credential_info={}
            )
        ],
    )
    anthropic_upstream.mock(side_effect=_upstream(_anthropic_error(CREDIT_TEXT)))
    named: Final = {"litellm_params": {"litellm_credential_name": "shared"}}
    router: Final = Router(
        model_list=[
            # Both order-0 legs name the same credential, each overriding its key: only the spent key ran dry.
            _leg(anthropic_upstream.api_base, "spent", SPENT_KEY, order=0, weight=1, leg_settings=named),
            _leg(anthropic_upstream.api_base, "funded", FUNDED_KEY, order=0, weight=0, leg_settings=named),
            _leg(anthropic_upstream.api_base, "next-order", NEXT_ORDER_KEY, order=1, weight=1),
            _leg(
                anthropic_upstream.api_base, "spent-elsewhere", SPENT_KEY, order=0, weight=1, model_name="other-model"
            ),
        ],
        num_retries=2,
        retry_after=0,
        cooldown_time=ROUTER_COOLDOWN_SECONDS,
    )

    text: Final = await _chat_text(router, stream=False)

    assert text == "hi"
    assert _calls_by_key(anthropic_upstream) == {SPENT_KEY: 1, FUNDED_KEY: 1, NEXT_ORDER_KEY: 0}
    assert _cooldowns(router) == {"spent": CREDIT_COOLDOWN_SECONDS, "spent-elsewhere": CREDIT_COOLDOWN_SECONDS}


class _AsyncOnlyStore:
    """The router's cooldown store with every blocking call refused and recorded."""

    def __init__(self, store: RouterCooldowns) -> None:
        self._store: Final = store
        self.blocking_calls: Final[list[str]] = []

    def __getattr__(self, name: str) -> object:
        if name in {"put", "active_sync", "minimum", "increment_allowed_fails", "snapshot"}:
            self.blocking_calls.append(name)
            raise AssertionError(f"blocking cooldown-store call {name} on the async path")
        return getattr(self._store, name)


@pytest.mark.asyncio
@CHAT
async def test_an_async_request_holds_the_spent_key_out_without_a_blocking_store_call(
    anthropic_upstream: Upstream, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "credit_exhausted_cooldown_time", CREDIT_COOLDOWN_SECONDS)
    anthropic_upstream.mock(side_effect=_upstream(_anthropic_error(CREDIT_TEXT)))
    router: Final = _router(anthropic_upstream.api_base)
    store: Final = _AsyncOnlyStore(router.cooldown_cache.cooldown_store)
    monkeypatch.setattr(router.cooldown_cache, "cooldown_store", store)

    text: Final = await _chat_text(router, stream=False)

    assert text == "hi"
    assert store.blocking_calls == []
    active: Final = await router.cooldown_cache.async_get_active_cooldowns(
        ["spent", "funded", "next-order", "spent-elsewhere"], parent_otel_span=None
    )
    assert {model_id: float(value["cooldown_time"]) for model_id, value in active} == {
        "spent": CREDIT_COOLDOWN_SECONDS,
        "spent-elsewhere": CREDIT_COOLDOWN_SECONDS,
    }
