import asyncio
import functools
import json
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from contextlib import suppress
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, ParamSpec
from urllib.parse import quote, urlparse

import anyio
import httpx
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    AZURE_SPEECH_CUSTOM_LLM_PROVIDER,
    BACKGROUND_RESPONSE_SETTLEMENT_TIMEOUT_SECONDS,
    USAGE_ONLY_SETTLEMENT_KEY,
)
from litellm.interactions.background_cost_polling import rebuild_logging_obj
from litellm.litellm_core_utils.core_helpers import bind_budget_reservation_to_callbacks
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._types import PassThroughEndpointLoggingResultValues, PassThroughEndpointLoggingTypedDict
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)
from litellm.types.utils import LlmProviders, StandardPassThroughResponseObject, Usage

from .background_response_creator import (
    CREATOR_BILL_DEFERRED,
    NOT_A_SETTLEMENT,
    SETTLED_BEFORE,
    SETTLES_FOR_CALLER,
    BackgroundResponseCreateContext,
    BackgroundResponseCreator,
    BackgroundResponseCreatorStore,
    BackgroundResponseSettlementDecision,
    CreatorLookupFailed,
    PrismaBackgroundResponseCreatorStore,
    SettlesForCreator,
    background_response_creator,
    leaves_the_caller_unbilled,
)
from .background_response_polling import (
    BACKGROUND_RESPONSE_SWEEP_INTERVAL_SECONDS,
    BackgroundResponsePoller,
    BackgroundResponsePollRequest,
    Fetched,
    PollHeaders,
    is_pollable_response_id,
    poll_request_for_create,
    pollable_at,
    response_url,
    sweep_background_response_polls,
    sweep_periodically,
)
from .background_response_settlement import (
    BackgroundResponseSettlementStore,
    Claimed,
    ClaimedBefore,
    ClaimUnrecorded,
    PrismaBackgroundResponseSettlementStore,
)
from .llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)
from .llm_provider_handlers.assembly_passthrough_logging_handler import (
    AssemblyAIPassthroughLoggingHandler,
)
from .llm_provider_handlers.cohere_passthrough_logging_handler import (
    CoherePassthroughLoggingHandler,
    is_cohere_embed_route,
)
from .llm_provider_handlers.cursor_passthrough_logging_handler import (
    CursorPassthroughLoggingHandler,
)
from .llm_provider_handlers.deepgram_listen_passthrough_logging_handler import (
    DeepgramListenPassthroughLoggingHandler,
)
from .llm_provider_handlers.fal_ai_passthrough_logging_handler import (
    FalAIPassthroughLoggingHandler,
)
from .llm_provider_handlers.gemini_passthrough_logging_handler import (
    GeminiPassthroughLoggingHandler,
)
from .llm_provider_handlers.tinyfish_passthrough_logging_handler import (
    TinyFishPassthroughLoggingHandler,
    is_tinyfish_agent_url,
)
from .llm_provider_handlers.transcribe_passthrough_logging_handler import (
    TRANSCRIBE_CUSTOM_LLM_PROVIDER,
    PassThroughLogDispatch,
    TranscribePassthroughLoggingHandler,
)
from .llm_provider_handlers.vertex_passthrough_logging_handler import (
    VertexPassthroughLoggingHandler,
)
from .managed_id_codec import decode as decode_managed_id
from .upstream_usage_headers import has_upstream_reported_usage

if TYPE_CHECKING:
    from .pass_through_endpoints import TeamCallbackWiring

CreatorLoggingWiring = Callable[[BackgroundResponseCreateContext], Awaitable["TeamCallbackWiring | None"]]

cohere_passthrough_logging_handler: Final = CoherePassthroughLoggingHandler()
_RESPONSE_BODY: Final = TypeAdapter(dict[str, object])


class NormalizedPassThroughLoggingPayload(TypedDict):
    standard_logging_response_object: ReadOnly[PassThroughEndpointLoggingResultValues | None]
    kwargs: ReadOnly[dict[str, object]]


def _upstream_response_id(response_id: str) -> str:
    managed_id: Final = decode_managed_id(response_id)
    return response_id if managed_id is None else managed_id.raw_provider_id


def _created_model(request_body: Mapping[str, object], response_body: Mapping[str, object]) -> str:
    return next(
        (model for model in (request_body.get("model"), response_body.get("model")) if isinstance(model, str)), ""
    )


def _response_time(response_body: Mapping[str, object], field: str, start_time: datetime) -> datetime | None:
    timestamp: Final = response_body.get(field)
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        return None
    return datetime.fromtimestamp(timestamp, tz=start_time.tzinfo)


def _run_window(response_body: Mapping[str, object], start_time: datetime) -> tuple[datetime, datetime]:
    completed: Final = _response_time(response_body, "completed_at", start_time)
    end: Final = completed if completed is not None and completed >= start_time else datetime.now(tz=start_time.tzinfo)
    created: Final = _response_time(response_body, "created_at", start_time)
    return (created if created is not None and created <= end else end), end


def _creator_litellm_params(context: BackgroundResponseCreateContext, spend_only: bool) -> Mapping[str, object]:
    metadata: Final = dict(context.metadata)
    return {"metadata": metadata, "no-log": True} if spend_only else {"metadata": metadata}


def _wires_logging(wiring: "TeamCallbackWiring | None") -> bool:
    return wiring is not None and bool(wiring.success_callbacks or wiring.failure_callbacks or wiring.logging_kwargs)


async def proxy_creator_logging_wiring(
    context: BackgroundResponseCreateContext, timeout_seconds: float = BACKGROUND_RESPONSE_SETTLEMENT_TIMEOUT_SECONDS
) -> "TeamCallbackWiring | None":
    from litellm.proxy.auth.auth_checks import get_key_object
    from litellm.proxy.proxy_server import prisma_client, proxy_config, user_api_key_cache

    from .pass_through_endpoints import resolve_team_callback_wiring

    key_hash: Final = context.metadata.get("user_api_key")
    if not isinstance(key_hash, str):
        return None
    try:
        creator_key: Final = await asyncio.wait_for(
            get_key_object(hashed_token=key_hash, prisma_client=prisma_client, user_api_key_cache=user_api_key_cache),
            timeout=timeout_seconds,
        )
    except (TimeoutError, asyncio.TimeoutError):
        raise
    except Exception as e:  # noqa: BLE001  # an unresolved creator key bills spend only instead of failing the bill
        verbose_proxy_logger.warning("Could not load the key that created a background response: %s", e)
        return None
    return resolve_team_callback_wiring(creator_key, proxy_config, "background response creator bill")


async def complete_despite_cancellation(work: Coroutine[object, object, None]) -> None:
    pending: Final = asyncio.create_task(work)
    try:
        await asyncio.shield(pending)
    except asyncio.CancelledError:
        with anyio.CancelScope(shield=True):
            while not pending.done():
                with suppress(asyncio.CancelledError):
                    await asyncio.shield(pending)
            pending.result()
        raise


_HandlerParams = ParamSpec("_HandlerParams")


def _finishes_despite_cancellation_when_finalizing(
    handler: Callable[_HandlerParams, Coroutine[object, object, None]],
) -> Callable[_HandlerParams, Coroutine[object, object, None]]:
    @functools.wraps(handler)
    async def handle(
        *args: _HandlerParams.args,
        **kwargs: _HandlerParams.kwargs,  # kwargs-ok: ParamSpec forwards the wrapped handler's exact signature
    ) -> None:
        if kwargs.get("finalize_before_dispatch") is True:
            await complete_despite_cancellation(handler(*args, **kwargs))
            return
        await handler(*args, **kwargs)

    return handle


# Usage shapes understood by the generic fallback, in probe order. Each entry is
# (container key, prompt-token field, completion-token field); a `None` container
# means the fields sit at the top level of the response body.
#
# These three cover essentially every LLM API in the wild: the OpenAI wire
# protocol and its many compatible upstreams, the Anthropic Messages shape, and
# Google's Gemini/Vertex shape.
_GENERIC_USAGE_SHAPES = (
    ("usage", "prompt_tokens", "completion_tokens"),  # OpenAI + compatibles
    ("usage", "input_tokens", "output_tokens"),  # Anthropic
    ("usageMetadata", "promptTokenCount", "candidatesTokenCount"),  # Gemini
)


def _coerce_token_count(value: Any) -> int | None:
    """Return a non-negative int token count, or None if the value isn't one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    count = int(value)
    return count if count >= 0 else None


def extract_generic_usage(response_body: Any) -> tuple | None:
    """Best-effort (prompt, completion, cache_read, cache_creation) token counts.

    Cost tracking for pass-through is otherwise an allow-list: a provider
    without a bespoke handler records $0 while its request is still billed
    against our upstream credentials. This makes a recognisable usage block
    sufficient to be priced, so pricing is the default for future providers
    rather than the exception.

    `prompt` INCLUDES any cache_read tokens (the OpenAI-wire convention);
    `cache_read`/`cache_creation` are broken out so the pricer can apply the
    discounted rates. Pricing every token flat at the input rate is not a safe
    simplification: DeepSeek bills cache hits at ~10% of the input rate, so a
    flat calc confidently OVERCHARGES the caller's key — worse than $0,
    because nothing looks wrong.

    Returns None when no shape matches — the caller must then leave the call
    unpriced rather than invent a number.
    """
    if not isinstance(response_body, dict):
        return None
    for container_key, prompt_field, completion_field in _GENERIC_USAGE_SHAPES:
        container = response_body.get(container_key) if container_key else response_body
        if not isinstance(container, dict):
            continue
        if prompt_field not in container and completion_field not in container:
            continue
        prompt_tokens = _coerce_token_count(container.get(prompt_field, 0))
        completion_tokens = _coerce_token_count(container.get(completion_field, 0))
        if prompt_tokens is None or completion_tokens is None:
            continue

        cache_read = 0
        cache_creation = 0
        if prompt_field == "prompt_tokens":
            # OpenAI wire: cached tokens are INSIDE prompt_tokens, detailed in
            # prompt_tokens_details. DeepSeek's dialect uses a top-level
            # prompt_cache_hit_tokens (also included in prompt_tokens).
            details = container.get("prompt_tokens_details")
            cache_read = (
                (_coerce_token_count(details.get("cached_tokens", 0)) or 0) if isinstance(details, dict) else 0
            ) or (_coerce_token_count(container.get("prompt_cache_hit_tokens", 0)) or 0)
        elif prompt_field == "input_tokens":
            # Anthropic: cache tokens are OUTSIDE input_tokens. Fold them in so
            # `prompt` means the same thing across shapes — without this they
            # were priced at $0.
            cache_read = _coerce_token_count(container.get("cache_read_input_tokens", 0)) or 0
            cache_creation = _coerce_token_count(container.get("cache_creation_input_tokens", 0)) or 0
            prompt_tokens += cache_read + cache_creation
        elif prompt_field == "promptTokenCount":
            # Gemini: cachedContentTokenCount is inside promptTokenCount;
            # thinking tokens are OUTSIDE candidatesTokenCount but billed at
            # the output rate — without this they were priced at $0.
            cache_read = _coerce_token_count(container.get("cachedContentTokenCount", 0)) or 0
            completion_tokens += _coerce_token_count(container.get("thoughtsTokenCount", 0)) or 0

        if prompt_tokens == 0 and completion_tokens == 0:
            # A usage block of all zeros carries no billable signal; keep
            # probing in case another shape is populated.
            continue
        return prompt_tokens, completion_tokens, cache_read, cache_creation
    return None


def _resolve_generic_price(model: str, custom_llm_provider: str | None) -> tuple | None:
    """Per-token input/output rates for `model`, or None if it isn't priced.

    Deliberately strict: only a model that resolves to a real price-map entry
    carrying per-token rates gets a cost. Anything else is left unpriced — an
    invented number corrupts invoice reconciliation more thoroughly than a
    missing one, because it looks authoritative.

    Models priced per second / per image (audio, image generation) resolve with
    zero per-token rates; those belong to dedicated handlers, so they are
    skipped here rather than recorded as a misleading $0.00.
    """
    from litellm.utils import get_model_info

    # When a provider IS configured, its price map is authoritative — no
    # fallback to a provider-less lookup. A self-hosted upstream (vllm) serving
    # a model whose name collides with a price-map entry ("gpt-4o") must stay
    # unpriced, not be billed at the real provider's rates.
    providers = (custom_llm_provider,) if custom_llm_provider else (None,)
    for provider in providers:
        try:
            model_info = get_model_info(model=model, custom_llm_provider=provider)
        except Exception:  # noqa: BLE001  # price lookup is best-effort; try the next provider
            continue
        input_rate = model_info.get("input_cost_per_token") or 0
        output_rate = model_info.get("output_cost_per_token") or 0
        if input_rate or output_rate:
            return (
                input_rate,
                output_rate,
                model_info.get("cache_read_input_token_cost"),
                model_info.get("cache_creation_input_token_cost"),
            )
    return None


def _safe_response_text(httpx_response: httpx.Response) -> str:
    """
    Streamed passthrough responses are relayed to the client without being read
    into memory, so accessing .text on them raises ResponseNotRead. Their body is
    intentionally uninspected; log an empty string instead of failing the row.
    """
    try:
        return httpx_response.text
    except httpx.ResponseNotRead:
        return ""


def _websocket_frames(
    response_body: Mapping[str, object] | Sequence[Mapping[str, object]] | None,
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        message for message in (response_body if isinstance(response_body, list) else ()) if isinstance(message, dict)
    )


def _adopt_handler_usage(logging_obj: LiteLLMLoggingObj, handler_kwargs: Mapping[str, object]) -> None:
    """Cost finalization reads model_call_details, so without the handler's usage there it logs zero tokens."""
    usage: Final = handler_kwargs.get("combined_usage_object")
    if isinstance(usage, Usage):
        logging_obj.model_call_details["combined_usage_object"] = usage  # rebind-ok: finalization reads it
    payload: Final = handler_kwargs.get("standard_logging_object")
    if isinstance(payload, dict):
        logging_obj.model_call_details["standard_logging_object"] = payload  # rebind-ok: finalization reads it


class PassThroughEndpointLogging:
    def __init__(
        self,
        transcribe_handler: TranscribePassthroughLoggingHandler | None = None,
        log_dispatch: PassThroughLogDispatch | None = None,
        background_response_settlement: BackgroundResponseSettlementStore | None = None,
        background_response_creators: BackgroundResponseCreatorStore | None = None,
        background_response_poller: BackgroundResponsePoller | None = None,
        creator_logging_wiring: CreatorLoggingWiring = proxy_creator_logging_wiring,
    ):
        self.creator_logging_wiring: Final = creator_logging_wiring
        self.transcribe_passthrough_logging_handler: Final = (
            transcribe_handler if transcribe_handler is not None else TranscribePassthroughLoggingHandler()
        )
        self._injected_log_dispatch: Final = log_dispatch
        self.background_response_settlement: Final[BackgroundResponseSettlementStore] = (
            background_response_settlement
            if background_response_settlement is not None
            else PrismaBackgroundResponseSettlementStore()
        )
        self.background_response_creators: Final[BackgroundResponseCreatorStore] = (
            background_response_creators
            if background_response_creators is not None
            else PrismaBackgroundResponseCreatorStore()
        )
        self.background_response_poller: Final = (
            background_response_poller if background_response_poller is not None else BackgroundResponsePoller()
        )
        self.TRACKED_VERTEX_METHOD_ROUTES = (
            "generateContent",
            "streamGenerateContent",
            "predict",
            "rawPredict",
            "streamRawPredict",
            "search",
            "predictLongRunning",
            "embedContent",
            "batchEmbedContents",
        )
        self.TRACKED_VERTEX_RESOURCE_ROUTES = ("batchPredictionJobs",)

        # Anthropic
        self.TRACKED_ANTHROPIC_ROUTES = ["/messages", "/v1/messages/batches"]

        # Cohere
        self.TRACKED_COHERE_ROUTES = ["/v2/chat"]
        self.assemblyai_passthrough_logging_handler = AssemblyAIPassthroughLoggingHandler()

        # Langfuse
        self.TRACKED_LANGFUSE_ROUTES = ["/langfuse/"]

        # Gemini
        self.TRACKED_GEMINI_ROUTES = [
            "generateContent",
            "streamGenerateContent",
            "batchGenerateContent",
            "predictLongRunning",
        ]

        # Cursor Cloud Agents
        self.TRACKED_CURSOR_ROUTES = [
            "/v0/agents",
            "/v0/me",
            "/v0/models",
            "/v0/repositories",
        ]

        # Vertex AI Live API WebSocket
        self.TRACKED_VERTEX_AI_LIVE_ROUTES = ["/vertex_ai/live"]
        self.TRACKED_OPENAI_WEBSOCKET_ROUTES: Final = ("/openai/", "/openai_passthrough/")

    @property
    def _log_dispatch(self) -> PassThroughLogDispatch:
        return self._injected_log_dispatch if self._injected_log_dispatch is not None else self.handle_logging

    async def handle_logging(
        self,
        logging_obj: LiteLLMLoggingObj,
        standard_logging_response_object: object,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        finalize_before_dispatch: bool = False,
        **kwargs: object,  # kwargs-ok: forwards the existing pass-through logging contract
    ) -> None:
        """Log pass-through success via the shared async dispatch path."""
        # Always reached from pass_through_async_success_handler, which runs in
        # an async context. call_type is "pass_through_endpoint" here, so the
        # passthrough guard in dispatch_success_handlers already forces the
        # async handler to run; pass prefer_async_handlers explicitly to match
        # the streaming sibling (_route_streaming_logging_to_handler) and keep
        # async-only loggers (e.g. the proxy spend logger) firing regardless of
        # how the call-type classification evolves.
        if finalize_before_dispatch:
            from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
            from litellm.proxy.spend_tracking.cost_finalization import finalize_proxy_response_cost

            async def finalize_and_enqueue() -> None:
                await finalize_proxy_response_cost(logging_obj, standard_logging_response_object)
                GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(
                    async_coroutine=self.handle_logging(
                        logging_obj=logging_obj,
                        standard_logging_response_object=standard_logging_response_object,
                        result=result,
                        start_time=start_time,
                        end_time=end_time,
                        cache_hit=cache_hit,
                        finalize_before_dispatch=False,
                        **kwargs,
                    )
                )
                bind_budget_reservation_to_callbacks(logging_obj.litellm_params)

            await complete_despite_cancellation(finalize_and_enqueue())
            return

        await logging_obj.dispatch_success_handlers(
            result=(json.dumps(result) if isinstance(result, dict) else standard_logging_response_object),
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
            prefer_async_handlers=True,
            **kwargs,
        )

    _handle_logging = handle_logging

    async def background_response_settlement_for(
        self, settled_response_id: str, response_body: Mapping[str, object], url_route: str
    ) -> BackgroundResponseSettlementDecision:
        from .llm_provider_handlers.openai_passthrough_logging_handler import targets_openai_api

        upstream_response_id: Final = _upstream_response_id(settled_response_id)
        creator: Final = (
            await self.background_response_creators.creator(upstream_response_id)
            if targets_openai_api(url_route)
            else None
        )
        if isinstance(creator, CreatorLookupFailed):
            return CREATOR_BILL_DEFERRED
        match await self.background_response_settlement.claim_outcome(
            upstream_response_id, release_if_late=creator is not None
        ):
            case ClaimedBefore():
                return SETTLED_BEFORE
            case Claimed() if creator is not None:
                return SettlesForCreator(creator=creator, response_body=response_body)
            case ClaimUnrecorded() if creator is not None:
                return CREATOR_BILL_DEFERRED
            case Claimed() | ClaimUnrecorded():
                return SETTLES_FOR_CALLER

    async def background_response_settlement_of_call(
        self,
        method: str,
        url_route: str,
        response_fields: Mapping[str, object] | None,
    ) -> BackgroundResponseSettlementDecision:
        from .llm_provider_handlers.openai_passthrough_logging_handler import (
            OpenAIPassthroughLoggingHandler,
        )

        if response_fields is None or not OpenAIPassthroughLoggingHandler.is_openai_responses_route(url_route):
            return NOT_A_SETTLEMENT
        settled_response_id: Final = OpenAIPassthroughLoggingHandler.background_response_settled_by_call(
            method, url_route, response_fields
        )
        if settled_response_id is None:
            return NOT_A_SETTLEMENT
        return await self.background_response_settlement_for(settled_response_id, response_fields, url_route)

    async def settle_for_creator(self, settlement: SettlesForCreator) -> None:
        await self._bill_creator_once(settlement.creator, settlement.response_body)

    async def settle_polled_background_response(
        self, creator: BackgroundResponseCreator, response_body: Mapping[str, object]
    ) -> bool:
        from .llm_provider_handlers.openai_passthrough_logging_handler import is_finished_background_response

        if not is_finished_background_response(response_body):
            await self.background_response_creators.forget(creator.response_id)
            return True
        match await self.background_response_settlement.claim_outcome(creator.response_id, release_if_late=True):
            case Claimed():
                return await self._bill_creator_once(creator, response_body)
            case ClaimedBefore():
                return True
            case ClaimUnrecorded():
                return False

    async def _bill_creator_once(self, creator: BackgroundResponseCreator, response_body: Mapping[str, object]) -> bool:
        if not await self.bill_background_response_creator(creator, response_body):
            await self.background_response_settlement.release(creator.response_id)
            return False
        await self.background_response_creators.forget(creator.response_id)
        return True

    async def bill_background_response_creator(
        self, creator: BackgroundResponseCreator, response_body: Mapping[str, object]
    ) -> bool:
        try:
            return await self._bill_background_response_creator(creator, response_body)
        except Exception:  # noqa: BLE001  # a failed creator bill is retried, so it must not fail the read or the poll
            verbose_proxy_logger.exception(
                "Could not bill background response %s to the key that created it", creator.response_id
            )
            return False

    async def _bill_background_response_creator(
        self, creator: BackgroundResponseCreator, response_body: Mapping[str, object]
    ) -> bool:
        context: Final = creator.context
        url_route: Final = f"{creator.upstream_url}/{quote(creator.response_id, safe='')}"
        wiring: Final = await self.creator_logging_wiring(context) if context.key_or_team_logging else None
        rewired: Final = _wires_logging(wiring)
        run_started_at, run_ended_at = _run_window(response_body, context.start_time)
        litellm_params: Final = _creator_litellm_params(
            context, spend_only=context.no_log or (context.key_or_team_logging and not rewired)
        )
        logging_obj: Final = rebuild_logging_obj(
            model=context.model,
            custom_llm_provider=context.custom_llm_provider,
            call_type="pass_through_endpoint",
            litellm_call_id=context.litellm_call_id,
            function_id=context.litellm_call_id,
            litellm_trace_id=context.litellm_trace_id,
            start_time=run_started_at,
            litellm_params=litellm_params,
            dynamic_success_callbacks=None if wiring is None else wiring.success_callbacks,
            dynamic_failure_callbacks=None if wiring is None else wiring.failure_callbacks,
            logging_kwargs=None if wiring is None else wiring.logging_kwargs,
        )
        if context.disabled_callbacks:
            logging_obj.standard_callback_dynamic_params["litellm_disabled_callbacks"] = list(
                context.disabled_callbacks
            )
        passthrough_logging_payload: Final = PassthroughStandardLoggingPayload(
            url=url_route, request_body={"model": context.model}, request_method="GET"
        )
        logging_obj.model_call_details["passthrough_logging_payload"] = passthrough_logging_payload
        logging_obj.model_call_details[USAGE_ONLY_SETTLEMENT_KEY] = True
        return await self._log_llm_pass_through(
            httpx_response=httpx.Response(200, json=response_body, request=httpx.Request("GET", url_route)),
            response_body=dict(response_body),
            request_body={"model": context.model},
            logging_obj=logging_obj,
            url_route=url_route,
            result="",
            start_time=run_started_at,
            end_time=run_ended_at,
            cache_hit=False,
            passthrough_logging_payload=passthrough_logging_payload,
            custom_llm_provider=context.custom_llm_provider,
            finalize_before_dispatch=True,
            background_response_settled_before=False,
            forwarded={
                "litellm_params": litellm_params,
                "call_type": "pass_through_endpoint",
                "litellm_call_id": context.litellm_call_id,
                "passthrough_logging_payload": passthrough_logging_payload,
            },
        )

    async def register_background_response_create(
        self,
        create_request: httpx.Request,
        response_fields: Mapping[str, object] | None,
        request_body: Mapping[str, object],
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        start_time: datetime,
        custom_llm_provider: str | None,
    ) -> None:
        from .llm_provider_handlers.openai_passthrough_logging_handler import (
            OpenAIPassthroughLoggingHandler,
        )

        if (
            custom_llm_provider != LlmProviders.OPENAI.value
            or response_fields is None
            or not OpenAIPassthroughLoggingHandler.creates_pollable_background_response(
                create_request.method, url_route, request_body, response_fields
            )
        ):
            return
        response_id: Final = _upstream_response_id(str(response_fields["id"]))
        if not is_pollable_response_id(response_id):
            verbose_proxy_logger.warning(
                "Not tracking background response %r for billing: its id cannot be part of a URL path", response_id
            )
            return
        creator: Final = background_response_creator(
            logging_obj=logging_obj,
            response_id=response_id,
            create_url=create_request.url,
            model=_created_model(request_body, response_fields),
            custom_llm_provider=custom_llm_provider,
            start_time=start_time,
        )
        await self.background_response_creators.register(creator, self.background_response_poller.lease_until())
        self.background_response_poller.start(
            creator,
            poll_request_for_create(create_request, response_id),
            self.settle_polled_background_response,
            self.background_response_creators,
        )

    async def sweep_background_response_polls(
        self, api_base: str, poll_headers: PollHeaders
    ) -> tuple["asyncio.Task[None]", ...]:
        return await sweep_background_response_polls(
            creators=self.background_response_creators,
            poller=self.background_response_poller,
            settle=self.settle_polled_background_response,
            api_base=api_base,
            poll_headers=poll_headers,
        )

    async def sweep_background_response_polls_periodically(self, api_base: str, poll_headers: PollHeaders) -> None:
        await sweep_periodically(
            lambda: self.sweep_background_response_polls(api_base, poll_headers),
            BACKGROUND_RESPONSE_SWEEP_INTERVAL_SECONDS,
        )

    async def settle_background_response_before_delete(
        self, method: str, target_url: str, headers: Mapping[str, str]
    ) -> None:
        from .llm_provider_handlers.openai_passthrough_logging_handler import targets_openai_api

        if method != "DELETE" or not targets_openai_api(target_url):
            return
        try:
            await self._settle_background_response_before_delete(httpx.URL(target_url), headers)
        except Exception:  # noqa: BLE001  # settling first must never block or fail the delete it runs before
            verbose_proxy_logger.exception("Could not settle background response %s before deleting it", target_url)

    async def _settle_background_response_before_delete(
        self, target_url: httpx.URL, headers: Mapping[str, str]
    ) -> None:
        from .llm_provider_handlers.openai_passthrough_logging_handler import (
            deleted_response_id,
            is_finished_background_response,
        )

        response_id: Final = deleted_response_id(target_url.path)
        if response_id is None:
            return
        creator: Final = await self.background_response_creators.creator(_upstream_response_id(response_id))
        if not isinstance(creator, BackgroundResponseCreator) or not pollable_at(creator, target_url):
            return
        match await self.background_response_poller.fetch(
            BackgroundResponsePollRequest(
                url=response_url(httpx.URL(creator.upstream_url), creator.response_id), headers=tuple(headers.items())
            )
        ):
            case Fetched(body=body) if is_finished_background_response(body):
                await self.settle_polled_background_response(creator, body)
            case _:
                return

    def normalize_llm_passthrough_logging_payload(
        self,
        httpx_response: httpx.Response,
        response_body: Mapping[str, object] | list[dict[str, object]] | None,
        request_body: dict[str, object],
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        custom_llm_provider: str | None = None,
        background_response_settled_before: bool = False,
        **kwargs,
    ) -> NormalizedPassThroughLoggingPayload:
        standard_logging_response_object: PassThroughEndpointLoggingResultValues | None = None

        if self.is_gemini_route(url_route, custom_llm_provider):
            gemini_passthrough_logging_handler_result = GeminiPassthroughLoggingHandler.gemini_passthrough_handler(
                httpx_response=httpx_response,
                response_body=response_body if isinstance(response_body, dict) else {},
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                request_body=request_body,
                **kwargs,
            )
            standard_logging_response_object = gemini_passthrough_logging_handler_result["result"]
            kwargs = gemini_passthrough_logging_handler_result["kwargs"]
        elif self.is_vertex_route(url_route):
            vertex_passthrough_logging_handler_result = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
                httpx_response=httpx_response,
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                request_body=request_body,
                **kwargs,
            )
            standard_logging_response_object = vertex_passthrough_logging_handler_result["result"]
            kwargs = vertex_passthrough_logging_handler_result["kwargs"]
        elif self.is_anthropic_route(url_route):
            anthropic_passthrough_logging_handler_result: Final = (
                AnthropicPassthroughLoggingHandler.anthropic_passthrough_handler(
                    httpx_response=httpx_response,
                    response_body=response_body if isinstance(response_body, dict) else {},
                    logging_obj=logging_obj,
                    url_route=url_route,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    cache_hit=cache_hit,
                    request_body=request_body,
                    **kwargs,
                )
            )

            standard_logging_response_object = anthropic_passthrough_logging_handler_result["result"]
            kwargs = anthropic_passthrough_logging_handler_result["kwargs"]
        elif self.is_cohere_route(url_route):
            cohere_passthrough_logging_handler_result = cohere_passthrough_logging_handler.cohere_passthrough_handler(
                httpx_response=httpx_response,
                response_body=response_body if isinstance(response_body, dict) else {},
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                request_body=request_body,
                **kwargs,
            )
            standard_logging_response_object = cohere_passthrough_logging_handler_result["result"]
            kwargs = cohere_passthrough_logging_handler_result["kwargs"]
        elif self.is_openai_route(url_route, custom_llm_provider):
            openai_dispatch_result: Final = self._dispatch_openai_passthrough(
                httpx_response=httpx_response,
                response_body=response_body,
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                request_body=request_body,
                custom_llm_provider=custom_llm_provider,
                kwargs=kwargs,
                background_response_settled_before=background_response_settled_before,
            )
            standard_logging_response_object = openai_dispatch_result["result"]  # rebind-ok: elif-chain
            kwargs = openai_dispatch_result["kwargs"]  # rebind-ok: elif-chain contract
        elif self.is_cursor_route(url_route, custom_llm_provider):
            cursor_passthrough_logging_handler_result = CursorPassthroughLoggingHandler.cursor_passthrough_handler(
                httpx_response=httpx_response,
                response_body=response_body if isinstance(response_body, dict) else {},
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                request_body=request_body,
                **kwargs,
            )
            standard_logging_response_object = cursor_passthrough_logging_handler_result["result"]
            kwargs = cursor_passthrough_logging_handler_result["kwargs"]
        elif self.is_comprehend_medical_route(custom_llm_provider):
            from .llm_provider_handlers.comprehend_medical_passthrough_logging_handler import (
                ComprehendMedicalPassthroughLoggingHandler,
            )

            comprehend_medical_handler_result: Final = (
                ComprehendMedicalPassthroughLoggingHandler.comprehend_medical_passthrough_handler(
                    httpx_response=httpx_response,
                    logging_obj=logging_obj,
                    url_route=url_route,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    cache_hit=cache_hit,
                    request_body=request_body,
                    **kwargs,
                )
            )
            standard_logging_response_object = comprehend_medical_handler_result["result"]  # rebind-ok: elif-chain
            kwargs = comprehend_medical_handler_result["kwargs"]  # rebind-ok: elif-chain contract
        elif self.is_tinyfish_route(url_route, custom_llm_provider):
            tinyfish_handler_result: Final = TinyFishPassthroughLoggingHandler.tinyfish_passthrough_handler(
                httpx_response=httpx_response,
                response_body=response_body if isinstance(response_body, dict) else None,
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                request_body=request_body,
                **kwargs,
            )
            standard_logging_response_object = tinyfish_handler_result["result"]  # rebind-ok: elif-chain
            kwargs = tinyfish_handler_result["kwargs"]  # rebind-ok: elif-chain contract

        elif self.is_azure_speech_route(custom_llm_provider):
            from .llm_provider_handlers.azure_speech_passthrough_logging_handler import (
                AzureSpeechPassthroughLoggingHandler,
            )

            azure_speech_handler_result: Final = AzureSpeechPassthroughLoggingHandler.azure_speech_passthrough_handler(
                httpx_response=httpx_response,
                response_body=response_body,
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                request_body=request_body,
                **kwargs,
            )
            standard_logging_response_object = azure_speech_handler_result["result"]  # rebind-ok: elif-chain
            kwargs = azure_speech_handler_result["kwargs"]  # rebind-ok: elif-chain contract
        elif self.is_transcribe_route(custom_llm_provider):
            transcribe_handler_result: Final = TranscribePassthroughLoggingHandler.transcribe_passthrough_handler(
                httpx_response=httpx_response,
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                request_body=request_body,
                **kwargs,
            )
            standard_logging_response_object = transcribe_handler_result["result"]  # rebind-ok: elif-chain
            kwargs = transcribe_handler_result["kwargs"]  # rebind-ok: elif-chain contract
        elif (
            self.is_typesafe_route(custom_llm_provider)
            or custom_llm_provider in ("eu_jev", "laya", "bespoke")
            or self.is_openrouter_decisions_route(url_route, custom_llm_provider)
        ):
            from .llm_provider_handlers.typesafe_passthrough_logging_handler import (
                TypeSafePassthroughLoggingHandler,
            )

            typesafe_handler_result: Final = TypeSafePassthroughLoggingHandler.typesafe_passthrough_handler(
                httpx_response=httpx_response,
                response_body=response_body if isinstance(response_body, dict) else MappingProxyType({}),
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                request_body=request_body,
                custom_llm_provider=custom_llm_provider or "",
                **kwargs,
            )
            standard_logging_response_object = typesafe_handler_result["result"]
            kwargs = typesafe_handler_result["kwargs"]

        elif self.is_vertex_ai_live_route(url_route):
            from .llm_provider_handlers.vertex_ai_live_passthrough_logging_handler import (
                VertexAILivePassthroughLoggingHandler,
            )

            vertex_ai_live_handler: Final = VertexAILivePassthroughLoggingHandler()

            # For WebSocket responses, response_body should be a list of messages
            websocket_messages: Final[Sequence[Mapping[str, object]]] = (
                response_body if isinstance(response_body, list) else []
            )

            vertex_ai_live_handler_result: Final = vertex_ai_live_handler.vertex_ai_live_passthrough_handler(
                websocket_messages=websocket_messages,
                logging_obj=logging_obj,
                url_route=url_route,
                start_time=start_time,
                end_time=end_time,
                request_body=request_body,
                **kwargs,
            )

            standard_logging_response_object = vertex_ai_live_handler_result["result"]
            kwargs = vertex_ai_live_handler_result["kwargs"]
        elif (
            websocket_relay_result := self._websocket_relay_logging_result(
                url_route=url_route,
                websocket_frames=_websocket_frames(response_body),
                logging_obj=logging_obj,
                httpx_response=httpx_response,
                start_time=start_time,
                kwargs=kwargs,
            )
        ) is not None:
            standard_logging_response_object = websocket_relay_result["result"]  # rebind-ok: elif-chain
            kwargs = websocket_relay_result["kwargs"]  # rebind-ok: elif-chain contract
        elif FalAIPassthroughLoggingHandler.is_fal_ai_route(url_route, custom_llm_provider):
            fal_ai_handler_result: Final = FalAIPassthroughLoggingHandler().fal_ai_passthrough_handler(
                response_body=response_body if isinstance(response_body, dict) else MappingProxyType({}),
                request_body=request_body,
                logging_obj=logging_obj,
                url_route=url_route,
                kwargs=kwargs,
            )
            standard_logging_response_object = fal_ai_handler_result["result"]  # rebind-ok: elif-chain
            kwargs = fal_ai_handler_result["kwargs"]  # rebind-ok: elif-chain contract
        _adopt_handler_usage(logging_obj, _RESPONSE_BODY.validate_python(kwargs))
        return {
            "standard_logging_response_object": standard_logging_response_object,
            "kwargs": kwargs,
        }

    def _websocket_relay_logging_result(
        self,
        url_route: str,
        websocket_frames: tuple[Mapping[str, object], ...],
        logging_obj: LiteLLMLoggingObj,
        httpx_response: httpx.Response,
        start_time: datetime,
        kwargs: Mapping[str, object],
    ) -> PassThroughEndpointLoggingTypedDict | None:
        if self.is_openai_websocket_route(url_route):
            from .llm_provider_handlers.openai_websocket_passthrough_logging_handler import (
                OpenAIWebsocketPassthroughLoggingHandler,
            )

            return OpenAIWebsocketPassthroughLoggingHandler().openai_websocket_passthrough_handler(
                websocket_messages=websocket_frames,
                logging_obj=logging_obj,
                url_route=url_route,
                start_time=start_time,
                kwargs=kwargs,
            )
        if DeepgramListenPassthroughLoggingHandler.is_deepgram_listen_route(url_route):
            return DeepgramListenPassthroughLoggingHandler().deepgram_listen_passthrough_handler(
                websocket_messages=websocket_frames,
                logging_obj=logging_obj,
                upstream_url=str(httpx_response.request.url),
                kwargs=kwargs,
            )
        return None

    @_finishes_despite_cancellation_when_finalizing
    async def pass_through_async_success_handler(
        self,
        httpx_response: httpx.Response,
        response_body: dict[str, object] | list[dict[str, object]] | None,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: dict[str, object],
        passthrough_logging_payload: PassthroughStandardLoggingPayload,
        custom_llm_provider: str | None = None,
        finalize_before_dispatch: bool = False,
        **kwargs: object,
    ) -> None:
        logging_obj.model_call_details["passthrough_logging_payload"] = passthrough_logging_payload
        if self.is_tinyfish_route(url_route, custom_llm_provider):
            # polls and cancels never write spend rows; run-async bills once from the background poller
            if not TinyFishPassthroughLoggingHandler.should_log_request(httpx_response.request.method, url_route):
                return
            if TinyFishPassthroughLoggingHandler.is_run_async_route(url_route):
                TinyFishPassthroughLoggingHandler.start_async_run_billing(
                    response_body=response_body if isinstance(response_body, dict) else None,
                    logging_obj=logging_obj,
                    result=result,
                    start_time=start_time,
                    cache_hit=cache_hit,
                    **kwargs,
                )
                return
        if self.is_assemblyai_route(url_route) and not self.is_azure_speech_route(custom_llm_provider):
            if AssemblyAIPassthroughLoggingHandler.should_log_request(httpx_response.request.method) is not True:
                return
            self.assemblyai_passthrough_logging_handler.assemblyai_passthrough_logging_handler(
                httpx_response=httpx_response,
                response_body=response_body if isinstance(response_body, dict) else {},
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                **kwargs,
            )
            return
        elif self.is_langfuse_route(url_route):
            # Don't log langfuse pass-through requests
            return
        elif self.is_transcribe_route(custom_llm_provider) and TranscribePassthroughLoggingHandler.is_priced_job_start(
            httpx_response
        ):
            self.transcribe_passthrough_logging_handler.schedule_priced_job_logging(
                httpx_response=httpx_response,
                response_body=response_body if isinstance(response_body, dict) else None,
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                request_body=request_body,
                log=self._log_dispatch,
                standard_pass_through_logging_payload=passthrough_logging_payload,
                **kwargs,
            )
            return
        else:
            from .llm_provider_handlers.openai_passthrough_logging_handler import (
                OpenAIPassthroughLoggingHandler,
            )

            response_fields: Final = (
                _RESPONSE_BODY.validate_python(response_body)
                if isinstance(response_body, dict)
                and OpenAIPassthroughLoggingHandler.is_openai_responses_route(url_route)
                else None
            )
            settlement: Final = await self.background_response_settlement_of_call(
                httpx_response.request.method, url_route, response_fields
            )
            if isinstance(settlement, SettlesForCreator):
                await self.settle_for_creator(settlement)
            await self._log_llm_pass_through(
                httpx_response=httpx_response,
                response_body=response_body,
                request_body=request_body,
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                passthrough_logging_payload=passthrough_logging_payload,
                custom_llm_provider=custom_llm_provider,
                finalize_before_dispatch=finalize_before_dispatch,
                background_response_settled_before=leaves_the_caller_unbilled(settlement),
                forwarded=kwargs,
            )
            await self.register_background_response_create(
                create_request=httpx_response.request,
                response_fields=response_fields,
                request_body=request_body,
                logging_obj=logging_obj,
                url_route=url_route,
                start_time=start_time,
                custom_llm_provider=custom_llm_provider,
            )

    async def _log_llm_pass_through(
        self,
        httpx_response: httpx.Response,
        response_body: dict[str, object] | list[dict[str, object]] | None,  # mutable-ok: normalize takes dicts
        request_body: dict[str, object],  # mutable-ok: normalize takes dicts
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        passthrough_logging_payload: PassthroughStandardLoggingPayload,
        custom_llm_provider: str | None,
        finalize_before_dispatch: bool,
        background_response_settled_before: bool,
        forwarded: Mapping[str, object],
    ) -> bool:
        try:
            normalized_llm_passthrough_logging_payload: Final = self.normalize_llm_passthrough_logging_payload(
                httpx_response=httpx_response,
                response_body=response_body,
                request_body=request_body,
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                custom_llm_provider=custom_llm_provider,
                background_response_settled_before=background_response_settled_before,
                **forwarded,
            )
        except Exception:
            if not finalize_before_dispatch:
                raise
            verbose_proxy_logger.exception("Unable to prepare pass-through success logging")
            return False
        normalized_response_object: Final = normalized_llm_passthrough_logging_payload[
            "standard_logging_response_object"
        ]
        standard_logging_response_object: Final[PassThroughEndpointLoggingResultValues] = (
            StandardPassThroughResponseObject(response=_safe_response_text(httpx_response))
            if normalized_response_object is None
            else normalized_response_object
        )
        kwargs: Final = self._set_cost_per_request(
            logging_obj=logging_obj,
            passthrough_logging_payload=passthrough_logging_payload,
            kwargs=normalized_llm_passthrough_logging_payload["kwargs"] or {},
        )
        await self._log_dispatch(
            logging_obj=logging_obj,
            finalize_before_dispatch=finalize_before_dispatch,
            standard_logging_response_object=standard_logging_response_object,
            result=result,
            start_time=start_time,
            end_time=end_time,
            cache_hit=cache_hit,
            standard_pass_through_logging_payload=passthrough_logging_payload,
            **kwargs,
        )
        return True

    def is_vertex_route(self, url_route: str) -> bool:
        if any(f":{method}" in url_route for method in self.TRACKED_VERTEX_METHOD_ROUTES):
            return True
        if any(resource in url_route for resource in self.TRACKED_VERTEX_RESOURCE_ROUTES):
            return True
        return VertexPassthroughLoggingHandler.is_vertex_interactions_route(url_route)

    def is_anthropic_route(self, url_route: str):
        for route in self.TRACKED_ANTHROPIC_ROUTES:
            if route in url_route:
                return True
        return False

    def is_cohere_route(self, url_route: str) -> bool:
        return is_cohere_embed_route(url_route) or any(route in url_route for route in self.TRACKED_COHERE_ROUTES)

    def is_assemblyai_route(self, url_route: str):
        parsed_url: Final = urlparse(url_route)
        if parsed_url.hostname == "api.assemblyai.com" or "/transcript" in parsed_url.path:
            return True
        return False

    def is_comprehend_medical_route(self, custom_llm_provider: str | None) -> bool:
        return custom_llm_provider == "comprehendmedical"

    def is_tinyfish_route(self, url_route: str, custom_llm_provider: str | None) -> bool:
        return custom_llm_provider == "tinyfish" or is_tinyfish_agent_url(url_route)

    def is_azure_speech_route(self, custom_llm_provider: str | None) -> bool:
        return custom_llm_provider == AZURE_SPEECH_CUSTOM_LLM_PROVIDER

    def is_transcribe_route(self, custom_llm_provider: str | None) -> bool:
        return custom_llm_provider == TRANSCRIBE_CUSTOM_LLM_PROVIDER

    def is_typesafe_route(self, custom_llm_provider: str | None) -> bool:
        return custom_llm_provider == "typesafe"

    def is_openrouter_decisions_route(self, url_route: str, custom_llm_provider: str | None) -> bool:
        return custom_llm_provider == "openrouter" and urlparse(url_route).path.endswith("/alpha/decisions")

    def is_langfuse_route(self, url_route: str):
        parsed_url: Final = urlparse(url_route)
        for route in self.TRACKED_LANGFUSE_ROUTES:
            if route in parsed_url.path:
                return True
        return False

    def is_vertex_ai_live_route(self, url_route: str):
        """Check if the URL route is a Vertex AI Live API WebSocket route."""
        if not url_route:
            return False
        for route in self.TRACKED_VERTEX_AI_LIVE_ROUTES:
            if route in url_route:
                return True
        return False

    def is_openai_websocket_route(self, url_route: str) -> bool:
        return url_route.startswith(self.TRACKED_OPENAI_WEBSOCKET_ROUTES)

    def is_cursor_route(self, url_route: str, custom_llm_provider: str | None = None):
        """Check if the URL route is a Cursor Cloud Agents API route."""
        if custom_llm_provider == "cursor":
            return True
        parsed_url: Final = urlparse(url_route)
        if parsed_url.hostname and "api.cursor.com" in parsed_url.hostname:
            return True
        for route in self.TRACKED_CURSOR_ROUTES:
            if route in url_route:
                path = parsed_url.path if parsed_url.scheme else url_route
                if path.startswith("/v0/"):
                    return custom_llm_provider == "cursor"
        return False

    def is_openai_route(self, url_route: str, custom_llm_provider: str | None = None):
        """Check if this pass-through call speaks the OpenAI wire protocol.

        Keys off `custom_llm_provider` first — the same way `is_gemini_route`
        and `is_cursor_route` do — so that ANY OpenAI-compatible upstream
        (Fireworks, Groq, Together, ...) reaches the OpenAI handler, whose cost
        math is already provider-agnostic. A hostname allow-list alone excluded
        them, and every such route recorded $0 while still billing our upstream
        account.

        Falls back to the URL-aware classification so OpenAI/Azure routes with
        no configured provider are unchanged, and non-OpenAI Azure Cognitive
        Services (Speech, Vision, Language, ...) sharing the
        `*.cognitiveservices.azure.com` / `*.openai.azure.com` domains are still
        excluded by the path-marker guard.
        """
        if not url_route:
            return False
        from .common_utils import is_openai_wire_compatible_route

        return is_openai_wire_compatible_route(url_route, custom_llm_provider)

    def is_gemini_route(self, url_route: str, custom_llm_provider: str | None = None):
        """Check if the URL route is a Gemini API route."""
        if custom_llm_provider != "gemini":
            return False
        if VertexPassthroughLoggingHandler.is_interactions_route(url_route):
            return True
        for route in self.TRACKED_GEMINI_ROUTES:
            if route in url_route:
                return True
        return False

    def _is_supported_openai_endpoint(self, url_route: str) -> bool:
        """Check if the OpenAI endpoint is supported by the passthrough logging handler.

        The Responses API route is included because
        `openai_passthrough_handler` has a dedicated `elif is_responses:`
        branch that knows how to extract usage + cost from the
        Responses-API on-the-wire shape. Without including it here, the
        outer dispatch filters Responses calls out before reaching the
        handler — the inner branch is then unreachable and Responses
        calls land in `LiteLLM_SpendLogs` with zero tokens / zero spend.

        The embeddings route is included for the same reason: pass-through
        embeddings bill upstream against our credentials, so omitting them
        from this allow-list records $0 against the calling virtual key.
        """
        from .llm_provider_handlers.openai_passthrough_logging_handler import (
            OpenAIPassthroughLoggingHandler,
        )

        return (
            OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(url_route)
            or OpenAIPassthroughLoggingHandler.is_openai_image_generation_route(url_route)
            or OpenAIPassthroughLoggingHandler.is_openai_image_editing_route(url_route)
            or OpenAIPassthroughLoggingHandler.is_openai_responses_route(url_route)
            or OpenAIPassthroughLoggingHandler.is_openai_embeddings_route(url_route)
            or OpenAIPassthroughLoggingHandler.is_openai_agents_route(url_route)
        )

    def _dispatch_openai_passthrough(
        self,
        httpx_response: httpx.Response,
        response_body: dict | list[dict[str, object]] | None,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: dict,
        custom_llm_provider: str | None,
        kwargs: dict,
        background_response_settled_before: bool = False,
    ) -> PassThroughEndpointLoggingTypedDict:
        """Three distinct fates on a recognised OpenAI route, and collapsing
        any two of them has already caused a money bug in each direction:
          1. supported billable endpoint -> the OpenAI handler;
          2. a Responses ITEM route that is not settling a background call ->
             priced by NOBODY. Its body echoes the original usage block, and
             `POST .../{id}/cancel` slips past the generic pricer's method
             gate — falling through re-billed the full generation on every
             poll. A poll or cancel that DOES settle a background call is the
             one item-route case that must still reach the OpenAI handler
             below, exactly once, the moment it finishes;
          3. anything else -> the generic pricer. Blanket-unpricing this
             bucket instead regressed billable provider-less POSTs
             (`/v1/completions` on an OpenAI host) from priced to $0.
        """
        from .llm_provider_handlers.openai_passthrough_logging_handler import (
            OpenAIPassthroughLoggingHandler,
        )

        is_unbilled_responses_item_route: Final = (
            OpenAIPassthroughLoggingHandler.is_openai_responses_item_route(url_route, custom_llm_provider)
            and OpenAIPassthroughLoggingHandler.openai_responses_call_billing(
                httpx_response.request.method,
                url_route,
                response_body if isinstance(response_body, dict) else {},
                background_response_settled_before,
                custom_llm_provider,
            )
            == "unbilled"
        )
        if is_unbilled_responses_item_route or OpenAIPassthroughLoggingHandler.is_openai_decisions_item_route(
            url_route
        ):
            # Fate 2: priced by nobody, but still named -- an unbilled item
            # route's spend row otherwise records "unknown" instead of the
            # model that served it.
            OpenAIPassthroughLoggingHandler.label_unbilled_call(
                logging_obj,
                url_route,
                request_body=request_body,
                response_body=response_body if isinstance(response_body, dict) else {},
            )
            return {"result": None, "kwargs": kwargs}

        openai_passthrough_logging_handler_result: Final = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=httpx_response,
            response_body=response_body if isinstance(response_body, dict) else {},
            logging_obj=logging_obj,
            url_route=url_route,
            result=result,
            start_time=start_time,
            end_time=end_time,
            cache_hit=cache_hit,
            request_body=request_body,
            custom_llm_provider=custom_llm_provider,
            background_response_settled_before=background_response_settled_before,
            **kwargs,
        )
        if openai_passthrough_logging_handler_result["result"] is not None:
            # Fate 1.
            return openai_passthrough_logging_handler_result

        # Fate 3: the OpenAI handler doesn't recognise this route (e.g.
        # legacy /v1/completions) — let the generic pricer try.
        return {
            "result": None,
            "kwargs": self._price_generic_passthrough(
                response_body=response_body if isinstance(response_body, dict) else None,
                request_body=request_body,
                logging_obj=logging_obj,
                url_route=url_route,
                custom_llm_provider=custom_llm_provider,
                kwargs=openai_passthrough_logging_handler_result["kwargs"],
            ),
        }

    def _resolve_generic_model(
        self,
        response_body: dict | None,
        request_body: dict,
        logging_obj: LiteLLMLoggingObj,
    ) -> str | None:
        """Find the model name for an otherwise-unrecognised pass-through call."""
        candidates = [
            request_body.get("model") if isinstance(request_body, dict) else None,
            response_body.get("model") if isinstance(response_body, dict) else None,
            logging_obj.model_call_details.get("model"),
        ]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None

    def _price_generic_passthrough(
        self,
        response_body: dict | None,
        request_body: dict,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        custom_llm_provider: str | None,
        kwargs: dict,
    ) -> dict:
        """Price a pass-through call that no provider handler recognised.

        Cost tracking used to be a strict allow-list: any route without a
        bespoke handler recorded $0 while still billing our upstream account,
        which breaks per-key budgets and corrupts invoice reconciliation. This
        makes pricing the default — a recognisable usage block plus a model
        that resolves to a real price entry is enough.

        Cost is published via `logging_obj.model_call_details["response_cost"]`,
        the contract every provider handler uses (see
        `base_passthrough_logging_handler._create_response_logging_payload`).
        It must be set here, synchronously, and not from a later callback:
        a callback races the spend-log DB writer.

        Silent no-op when the usage shape is unrecognised or the model is not
        priced — leaving a call unpriced is recoverable, inventing a number is
        not.
        """
        try:
            payload = logging_obj.model_call_details.get("passthrough_logging_payload") or {}

            # Only POST creates work. A GET/DELETE on any surface is object
            # management whose response may ECHO the original usage block —
            # pricing it re-bills a generation the caller already paid for.
            # Unknown method (older payloads, test doubles) keeps pricing.
            request_method = payload.get("request_method")
            if isinstance(request_method, str) and request_method.upper() != "POST":
                verbose_proxy_logger.debug(
                    "Generic passthrough pricing skipped for %s %s: non-POST object management",
                    request_method,
                    url_route,
                )
                return kwargs

            body = response_body if isinstance(response_body, dict) else payload.get("response_body")

            usage = extract_generic_usage(body)
            if usage is None:
                return kwargs
            prompt_tokens, completion_tokens, cache_read_tokens, cache_creation_tokens = usage

            model = self._resolve_generic_model(
                response_body=body,
                request_body=request_body,
                logging_obj=logging_obj,
            )
            if not model:
                verbose_proxy_logger.debug(
                    "Generic passthrough usage found for %s but no model to price it with", url_route
                )
                return kwargs

            rates = _resolve_generic_price(model=model, custom_llm_provider=custom_llm_provider)
            if rates is None:
                verbose_proxy_logger.debug(
                    "Generic passthrough model %s has no per-token price entry; leaving unpriced", model
                )
                return kwargs
            input_rate, output_rate, cache_read_rate, cache_creation_rate = rates

            # A cache component without an explicit discounted rate must leave
            # the call unpriced, not be flattened to the input rate: DeepSeek
            # bills cache hits at ~10% of input, so flat pricing confidently
            # overcharges — harder to detect than the $0 row this module exists
            # to prevent.
            if cache_read_tokens and cache_read_rate is None:
                verbose_proxy_logger.debug(
                    "Generic passthrough model %s reports %d cached tokens but has no "
                    "cache_read_input_token_cost; leaving unpriced rather than overcharging",
                    model,
                    cache_read_tokens,
                )
                return kwargs
            if cache_creation_tokens and cache_creation_rate is None:
                verbose_proxy_logger.debug(
                    "Generic passthrough model %s reports %d cache-creation tokens but has no "
                    "cache_creation_input_token_cost; leaving unpriced rather than mispricing",
                    model,
                    cache_creation_tokens,
                )
                return kwargs

            uncached_prompt_tokens = max(prompt_tokens - cache_read_tokens - cache_creation_tokens, 0)
            response_cost = (
                (uncached_prompt_tokens * input_rate)
                + (cache_read_tokens * (cache_read_rate or 0))
                + (cache_creation_tokens * (cache_creation_rate or 0))
                + (completion_tokens * output_rate)
            )

            kwargs["response_cost"] = response_cost
            kwargs["model"] = model
            if custom_llm_provider:
                kwargs["custom_llm_provider"] = custom_llm_provider
            # The pass-through spend path reads cost from model_call_details.
            logging_obj.model_call_details["response_cost"] = response_cost
            logging_obj.model_call_details.setdefault("model", model)
            if custom_llm_provider:
                logging_obj.model_call_details.setdefault("custom_llm_provider", custom_llm_provider)

            verbose_proxy_logger.debug(
                "Priced generic passthrough %s (model=%s) at %s", url_route, model, response_cost
            )
        except Exception as e:  # noqa: BLE001  # cost tracking must never break the response path
            # Cost tracking must never break the response path.
            verbose_proxy_logger.exception("Error pricing generic passthrough request: %s", e)
        return kwargs

    def _set_cost_per_request(
        self,
        logging_obj: LiteLLMLoggingObj,
        passthrough_logging_payload: PassthroughStandardLoggingPayload,
        kwargs: dict[str, object],
    ) -> dict[str, object]:
        """
        Helper function to set the cost per request in the logging object

        Only set the cost per request if it's set in the passthrough logging payload.
        If it's not set, don't set it in the logging object.

        An upstream that prices its own requests always wins: ``cost_per_request``
        is a flat per-request estimate for targets LiteLLM cannot price, and it
        defaults to 0.0 on every config-defined endpoint, so honoring it here
        would zero out the real cost the upstream reported. That holds even when
        the reported value was unusable, where the contract records 0 rather
        than billing an estimate the upstream just contradicted.
        """
        #########################################################
        # Check if cost per request is set
        #########################################################
        if has_upstream_reported_usage(logging_obj):
            return kwargs

        if passthrough_logging_payload.get("cost_per_request") is not None:
            kwargs["response_cost"] = passthrough_logging_payload.get("cost_per_request")
            logging_obj.model_call_details["response_cost"] = passthrough_logging_payload.get("cost_per_request")

        return kwargs
