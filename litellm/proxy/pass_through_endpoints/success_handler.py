import json
from datetime import datetime
from typing import Any, Union
from urllib.parse import urlparse

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._types import PassThroughEndpointLoggingResultValues
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)
from litellm.types.utils import StandardPassThroughResponseObject

from .llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)
from .llm_provider_handlers.assembly_passthrough_logging_handler import (
    AssemblyAIPassthroughLoggingHandler,
)
from .llm_provider_handlers.cohere_passthrough_logging_handler import (
    CoherePassthroughLoggingHandler,
)
from .llm_provider_handlers.cursor_passthrough_logging_handler import (
    CursorPassthroughLoggingHandler,
)
from .llm_provider_handlers.gemini_passthrough_logging_handler import (
    GeminiPassthroughLoggingHandler,
)
from .llm_provider_handlers.vertex_passthrough_logging_handler import (
    VertexPassthroughLoggingHandler,
)

cohere_passthrough_logging_handler = CoherePassthroughLoggingHandler()

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
    """Best-effort (prompt_tokens, completion_tokens) from an unknown provider.

    Cost tracking for pass-through is otherwise an allow-list: a provider
    without a bespoke handler records $0 while its request is still billed
    against our upstream credentials. This makes a recognisable usage block
    sufficient to be priced, so pricing is the default for future providers
    rather than the exception.

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
        if prompt_tokens == 0 and completion_tokens == 0:
            # A usage block of all zeros carries no billable signal; keep
            # probing in case another shape is populated.
            continue
        return prompt_tokens, completion_tokens
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

    for provider in (custom_llm_provider, None):
        try:
            model_info = get_model_info(model=model, custom_llm_provider=provider)
        except Exception:  # noqa: BLE001  # price lookup is best-effort; try the next provider
            continue
        input_rate = model_info.get("input_cost_per_token") or 0
        output_rate = model_info.get("output_cost_per_token") or 0
        if input_rate or output_rate:
            return input_rate, output_rate
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


class PassThroughEndpointLogging:
    def __init__(self):
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
        self.TRACKED_COHERE_ROUTES = ["/v2/chat", "/v1/embed"]
        self.assemblyai_passthrough_logging_handler = AssemblyAIPassthroughLoggingHandler()

        # Langfuse
        self.TRACKED_LANGFUSE_ROUTES = ["/langfuse/"]

        # Gemini
        self.TRACKED_GEMINI_ROUTES = [
            "generateContent",
            "streamGenerateContent",
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

    async def _handle_logging(
        self,
        logging_obj: LiteLLMLoggingObj,
        standard_logging_response_object: Union[
            StandardPassThroughResponseObject,
            PassThroughEndpointLoggingResultValues,
            dict,
        ],
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        **kwargs,
    ):
        """Log pass-through success via the shared async dispatch path."""
        # Always reached from pass_through_async_success_handler, which runs in
        # an async context. call_type is "pass_through_endpoint" here, so the
        # passthrough guard in dispatch_success_handlers already forces the
        # async handler to run; pass prefer_async_handlers explicitly to match
        # the streaming sibling (_route_streaming_logging_to_handler) and keep
        # async-only loggers (e.g. the proxy spend logger) firing regardless of
        # how the call-type classification evolves.
        await logging_obj.dispatch_success_handlers(
            result=(json.dumps(result) if isinstance(result, dict) else standard_logging_response_object),
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
            prefer_async_handlers=True,
            **kwargs,
        )

    def normalize_llm_passthrough_logging_payload(
        self,
        httpx_response: httpx.Response,
        response_body: dict | None,
        request_body: dict,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        custom_llm_provider: str | None = None,
        **kwargs,
    ):
        return_dict = {
            "standard_logging_response_object": None,
            "kwargs": kwargs,
        }
        standard_logging_response_object: Any | None = None

        if self.is_gemini_route(url_route, custom_llm_provider):
            gemini_passthrough_logging_handler_result = GeminiPassthroughLoggingHandler.gemini_passthrough_handler(
                httpx_response=httpx_response,
                response_body=response_body or {},
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
            anthropic_passthrough_logging_handler_result = (
                AnthropicPassthroughLoggingHandler.anthropic_passthrough_handler(
                    httpx_response=httpx_response,
                    response_body=response_body or {},
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
                response_body=response_body or {},
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
        elif self.is_openai_route(url_route, custom_llm_provider) and self._is_supported_openai_endpoint(
            url_route, custom_llm_provider
        ):
            from .llm_provider_handlers.openai_passthrough_logging_handler import (
                OpenAIPassthroughLoggingHandler,
            )

            openai_passthrough_logging_handler_result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
                httpx_response=httpx_response,
                response_body=response_body or {},
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                request_body=request_body,
                custom_llm_provider=custom_llm_provider,
                **kwargs,
            )
            standard_logging_response_object = openai_passthrough_logging_handler_result["result"]
            kwargs = openai_passthrough_logging_handler_result["kwargs"]

        elif self.is_cursor_route(url_route, custom_llm_provider):
            cursor_passthrough_logging_handler_result = CursorPassthroughLoggingHandler.cursor_passthrough_handler(
                httpx_response=httpx_response,
                response_body=response_body or {},
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
        elif self.is_vertex_ai_live_route(url_route):
            from .llm_provider_handlers.vertex_ai_live_passthrough_logging_handler import (
                VertexAILivePassthroughLoggingHandler,
            )

            vertex_ai_live_handler = VertexAILivePassthroughLoggingHandler()

            # For WebSocket responses, response_body should be a list of messages
            websocket_messages: list[dict[str, Any]] = response_body if isinstance(response_body, list) else []

            vertex_ai_live_handler_result = vertex_ai_live_handler.vertex_ai_live_passthrough_handler(
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
        else:
            # No bespoke handler matched. Rather than record $0 for a request we
            # are still billed for, price it from whatever recognisable usage
            # block the upstream returned. See `_price_generic_passthrough`.
            kwargs = self._price_generic_passthrough(
                response_body=response_body,
                request_body=request_body,
                logging_obj=logging_obj,
                url_route=url_route,
                custom_llm_provider=custom_llm_provider,
                kwargs=kwargs,
            )
        return_dict["standard_logging_response_object"] = standard_logging_response_object

        return_dict["kwargs"] = kwargs
        return return_dict

    async def pass_through_async_success_handler(
        self,
        httpx_response: httpx.Response,
        response_body: dict | None,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: dict,
        passthrough_logging_payload: PassthroughStandardLoggingPayload,
        custom_llm_provider: str | None = None,
        **kwargs,
    ):
        standard_logging_response_object: PassThroughEndpointLoggingResultValues | None = None
        logging_obj.model_call_details["passthrough_logging_payload"] = passthrough_logging_payload
        if self.is_assemblyai_route(url_route):
            if AssemblyAIPassthroughLoggingHandler._should_log_request(httpx_response.request.method) is not True:
                return
            self.assemblyai_passthrough_logging_handler.assemblyai_passthrough_logging_handler(
                httpx_response=httpx_response,
                response_body=response_body or {},
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
        else:
            normalized_llm_passthrough_logging_payload = self.normalize_llm_passthrough_logging_payload(
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
                **kwargs,
            )
            standard_logging_response_object = normalized_llm_passthrough_logging_payload[
                "standard_logging_response_object"
            ]
            kwargs = normalized_llm_passthrough_logging_payload["kwargs"]
        if standard_logging_response_object is None:
            standard_logging_response_object = StandardPassThroughResponseObject(
                response=_safe_response_text(httpx_response)
            )

        kwargs = self._set_cost_per_request(
            logging_obj=logging_obj,
            passthrough_logging_payload=passthrough_logging_payload,
            kwargs=kwargs,
        )

        await self._handle_logging(
            logging_obj=logging_obj,
            standard_logging_response_object=standard_logging_response_object,
            result=result,
            start_time=start_time,
            end_time=end_time,
            cache_hit=cache_hit,
            standard_pass_through_logging_payload=passthrough_logging_payload,
            **kwargs,
        )

    def is_vertex_route(self, url_route: str) -> bool:
        if any(f":{method}" in url_route for method in self.TRACKED_VERTEX_METHOD_ROUTES):
            return True
        return any(resource in url_route for resource in self.TRACKED_VERTEX_RESOURCE_ROUTES)

    def is_anthropic_route(self, url_route: str):
        for route in self.TRACKED_ANTHROPIC_ROUTES:
            if route in url_route:
                return True
        return False

    def is_cohere_route(self, url_route: str):
        for route in self.TRACKED_COHERE_ROUTES:
            if route in url_route:
                return True

    def is_assemblyai_route(self, url_route: str):
        parsed_url = urlparse(url_route)
        if parsed_url.hostname == "api.assemblyai.com":
            return True
        elif "/transcript" in parsed_url.path:
            return True
        return False

    def is_langfuse_route(self, url_route: str):
        parsed_url = urlparse(url_route)
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

    def is_cursor_route(self, url_route: str, custom_llm_provider: str | None = None):
        """Check if the URL route is a Cursor Cloud Agents API route."""
        if custom_llm_provider == "cursor":
            return True
        parsed_url = urlparse(url_route)
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
        from .common_utils import is_openai_wire_compatible_route

        return is_openai_wire_compatible_route(url_route, custom_llm_provider)

    def is_gemini_route(self, url_route: str, custom_llm_provider: str | None = None):
        """Check if the URL route is a Gemini API route."""
        for route in self.TRACKED_GEMINI_ROUTES:
            if route in url_route and custom_llm_provider == "gemini":
                return True
        return False

    def _is_supported_openai_endpoint(self, url_route: str, custom_llm_provider: str | None = None) -> bool:
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
            OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(url_route, custom_llm_provider)
            or OpenAIPassthroughLoggingHandler.is_openai_image_generation_route(url_route, custom_llm_provider)
            or OpenAIPassthroughLoggingHandler.is_openai_image_editing_route(url_route, custom_llm_provider)
            or OpenAIPassthroughLoggingHandler.is_openai_responses_route(url_route, custom_llm_provider)
            or OpenAIPassthroughLoggingHandler.is_openai_embeddings_route(url_route, custom_llm_provider)
        )

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
            body = response_body if isinstance(response_body, dict) else payload.get("response_body")

            usage = extract_generic_usage(body)
            if usage is None:
                return kwargs
            prompt_tokens, completion_tokens = usage

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
            input_rate, output_rate = rates

            response_cost = (prompt_tokens * input_rate) + (completion_tokens * output_rate)

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
        kwargs: dict,
    ):
        """
        Helper function to set the cost per request in the logging object

        Only set the cost per request if it's set in the passthrough logging payload.
        If it's not set, don't set it in the logging object.
        """
        #########################################################
        # Check if cost per request is set
        #########################################################
        if passthrough_logging_payload.get("cost_per_request") is not None:
            kwargs["response_cost"] = passthrough_logging_payload.get("cost_per_request")
            logging_obj.model_call_details["response_cost"] = passthrough_logging_payload.get("cost_per_request")

        return kwargs
