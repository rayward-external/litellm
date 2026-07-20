"""
OpenAI Passthrough Logging Handler

Handles cost tracking and logging for OpenAI passthrough endpoints, specifically /chat/completions.
"""

from datetime import datetime
from typing import List, Optional, Tuple, Union
from urllib.parse import urlparse

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)
from litellm.llms.openai.openai import OpenAIConfig
from litellm.llms.openai.openai import OpenAIConfig as OpenAIConfigType
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.proxy.pass_through_endpoints.common_utils import (
    AZURE_OPENAI_HOSTNAMES,
    AZURE_OPENAI_PATH_MARKERS,
    OPENAI_HOSTNAMES,
    hostname_matches,
    is_openai_compatible_url,
)
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.base_passthrough_logging_handler import (
    BasePassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    EndpointType,
    PassthroughStandardLoggingPayload,
)
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import (
    EmbeddingResponse,
    ImageResponse,
    LlmProviders,
    PassthroughCallTypes,
    Usage,
)
from litellm.utils import ModelResponse, TextCompletionResponse

# Hostname/URL classification for OpenAI-compatible APIs lives in
# `pass_through_endpoints.common_utils` so the streaming path
# (`HttpPassThroughEndpointHelpers.get_endpoint_type`) and this logging handler
# share one implementation. Re-exported here under the private names this module
# has always used.
_OPENAI_HOSTNAMES = OPENAI_HOSTNAMES
_AZURE_OPENAI_HOSTNAMES = AZURE_OPENAI_HOSTNAMES
_AZURE_OPENAI_PATH_MARKERS = AZURE_OPENAI_PATH_MARKERS
_hostname_matches = hostname_matches
_is_openai_compatible_url = is_openai_compatible_url

# Classic Azure OpenAI routes are shaped
# `/openai/deployments/{deployment-id}/chat/completions?api-version=...` — they
# carry no `/v1/` segment, so a `/v1/chat/completions` check alone misses them
# and their cost is silently recorded as $0.
_AZURE_DEPLOYMENTS_PATH_MARKER = "/openai/deployments/"


def _is_chat_completions_path(path: str) -> bool:
    """True for both OpenAI-v1 and classic Azure-deployment chat paths.

    OpenAI-v1: `/v1/chat/completions` (also LiteLLM's `/openai/v1/...` surface).
    Classic Azure: `/openai/deployments/{deployment-id}/chat/completions`.
    """
    if "/v1/chat/completions" in path:
        return True
    return _AZURE_DEPLOYMENTS_PATH_MARKER in path and path.rstrip("/").endswith("/chat/completions")


def _is_openai_compatible_host(hostname: Optional[str]) -> bool:
    """True if the hostname is OpenAI proper or one of the Azure OpenAI domains.

    Hostname-only check, kept for the route-level helpers that additionally
    require a specific OpenAI path (e.g. `/v1/chat/completions`). When only the
    hostname would otherwise gate dispatch, use `_is_openai_compatible_url` so
    non-OpenAI Azure Cognitive Services on the shared domains are excluded.
    """
    if not hostname:
        return False
    return _hostname_matches(hostname, _OPENAI_HOSTNAMES) or _hostname_matches(hostname, _AZURE_OPENAI_HOSTNAMES)


class OpenAIPassthroughLoggingHandler(BasePassthroughLoggingHandler):
    """
    OpenAI-specific passthrough logging handler that provides cost tracking for /chat/completions endpoints.
    """

    @property
    def llm_provider_name(self) -> LlmProviders:
        return LlmProviders.OPENAI

    def get_provider_config(self, model: str) -> OpenAIConfigType:
        """Get OpenAI provider configuration for the given model."""
        return OpenAIConfig()

    @staticmethod
    def is_openai_chat_completions_route(url_route: str) -> bool:
        """Check if the URL route is an OpenAI chat completions endpoint.

        Accepts both the OpenAI-v1 shape and the classic Azure OpenAI
        deployment shape — see `_is_chat_completions_path`.
        """
        if not url_route:
            return False
        parsed_url = urlparse(url_route)
        return _is_openai_compatible_host(parsed_url.hostname) and _is_chat_completions_path(parsed_url.path)

    @staticmethod
    def is_openai_image_generation_route(url_route: str) -> bool:
        """Check if the URL route is an OpenAI image generation endpoint."""
        if not url_route:
            return False
        parsed_url = urlparse(url_route)
        return _is_openai_compatible_host(parsed_url.hostname) and "/v1/images/generations" in parsed_url.path

    @staticmethod
    def is_openai_image_editing_route(url_route: str) -> bool:
        """Check if the URL route is an OpenAI image editing endpoint."""
        if not url_route:
            return False
        parsed_url = urlparse(url_route)
        return _is_openai_compatible_host(parsed_url.hostname) and "/v1/images/edits" in parsed_url.path

    @staticmethod
    def is_openai_responses_route(url_route: str) -> bool:
        """Check if the URL route is an OpenAI responses API endpoint."""
        if not url_route:
            return False
        parsed_url = urlparse(url_route)
        return _is_openai_compatible_host(parsed_url.hostname) and (
            "/v1/responses" in parsed_url.path or "/responses" in parsed_url.path
        )

    @staticmethod
    def is_openai_embeddings_route(url_route: str) -> bool:
        """Check if the URL route is an OpenAI embeddings endpoint.

        Matches both the OpenAI-v1 surface (`/v1/embeddings`, which is also
        what the Azure `/openai/v1/` surface exposes) and the classic Azure
        deployment path (`/openai/deployments/{deployment}/embeddings`), which
        has no `/v1/` segment — hence the bare `/embeddings` containment test.
        """
        if not url_route:
            return False
        parsed_url = urlparse(url_route)
        return _is_openai_compatible_host(parsed_url.hostname) and "/embeddings" in parsed_url.path

    def _get_user_from_metadata(
        self,
        passthrough_logging_payload: PassthroughStandardLoggingPayload,
    ) -> Optional[str]:
        """Extract user information from passthrough logging payload."""
        request_body = passthrough_logging_payload.get("request_body")
        if request_body:
            return request_body.get("user")
        return None

    @staticmethod
    def _calculate_image_generation_cost(
        model: str,
        response_body: dict,
        request_body: dict,
    ) -> float:
        """Calculate cost for OpenAI image generation."""
        try:
            # Extract parameters from request
            n = request_body.get("n", 1)
            try:
                n = int(n)
            except Exception:
                n = 1
            size = request_body.get("size", "1024x1024")
            quality = request_body.get("quality", None)

            # Use LiteLLM's default image cost calculator
            from litellm.cost_calculator import default_image_cost_calculator

            cost = default_image_cost_calculator(
                model=model,
                custom_llm_provider="openai",
                quality=quality,
                n=n,
                size=size,
                optional_params=request_body,
            )

            return cost
        except Exception as e:
            verbose_proxy_logger.warning(f"Error calculating image generation cost: {str(e)}")
            return 0.0

    @staticmethod
    def _calculate_image_editing_cost(
        model: str,
        response_body: dict,
        request_body: dict,
    ) -> float:
        """Calculate cost for OpenAI image editing."""
        try:
            # Extract parameters from request
            n = request_body.get("n", 1)
            # Image edit typically uses multipart/form-data (because of files), so all fields arrive as strings (e.g., n = "1").
            try:
                n = int(n)
            except Exception:
                n = 1
            size = request_body.get("size", "1024x1024")

            # Use LiteLLM's default image cost calculator
            from litellm.cost_calculator import default_image_cost_calculator

            cost = default_image_cost_calculator(
                model=model,
                custom_llm_provider="openai",
                quality=None,  # Image editing doesn't have quality parameter
                n=n,
                size=size,
                optional_params=request_body,
            )

            return cost
        except Exception as e:
            verbose_proxy_logger.warning(f"Error calculating image editing cost: {str(e)}")
            return 0.0

    @staticmethod
    def _build_responses_api_response_and_cost(
        model: str,
        httpx_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str,
    ) -> Tuple[ResponsesAPIResponse, float]:
        """Transform a Responses API raw response into a ResponsesAPIResponse
        and compute its cost.

        The Responses API has a different on-the-wire shape from chat
        completions (`output: [...]` instead of `choices: [...]`), so the
        chat-completions `transform_response` raises KeyError 'choices' on
        a Responses payload. Use the dedicated Responses-API transformer
        (`OpenAIResponsesAPIConfig.transform_response_api_response`) here.

        Returns (litellm_model_response, response_cost) — symmetric with the
        chat-completions branch which produces the same two values inline,
        and analogous to the image branches' `_calculate_image_*_cost` helpers
        (which return cost only because the image-response object is trivial
        to build inline; the Responses payload needs a real transformer).
        """
        responses_config = OpenAIResponsesAPIConfig()
        litellm_model_response = responses_config.transform_response_api_response(
            model=model,
            raw_response=httpx_response,
            logging_obj=logging_obj,
        )
        response_cost = litellm.completion_cost(
            completion_response=litellm_model_response,
            model=model,
            custom_llm_provider=custom_llm_provider,
            call_type="responses",
        )
        return litellm_model_response, response_cost

    @staticmethod
    def _build_embeddings_response_and_cost(
        model: str,
        response_body: dict,
        custom_llm_provider: str,
    ) -> Tuple[EmbeddingResponse, float]:
        """Build an EmbeddingResponse from an embeddings payload and cost it.

        Embeddings responses are plain JSON (`data: [...]` plus a standard
        `usage` object) so — like the image branches — the response object is
        cheap to build inline and needs no provider transformer. The usage
        object only carries `prompt_tokens` / `total_tokens`; there are no
        completion tokens, so `completion_tokens` is pinned to 0 rather than
        left unset, which keeps `completion_cost` from inferring a value.

        Returns (litellm_model_response, response_cost), symmetric with
        `_build_responses_api_response_and_cost`.
        """
        usage = response_body.get("usage", {}) or {}
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        litellm_model_response = EmbeddingResponse(
            model=model,
            data=response_body.get("data", []),
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                total_tokens=usage.get("total_tokens", prompt_tokens) or prompt_tokens,
            ),
        )
        response_cost = litellm.completion_cost(
            completion_response=litellm_model_response,
            model=model,
            custom_llm_provider=custom_llm_provider,
            call_type="embedding",
        )
        return litellm_model_response, response_cost

    @staticmethod
    def openai_passthrough_handler(
        httpx_response: httpx.Response,
        response_body: dict,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: dict,
        **kwargs,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Handle OpenAI passthrough logging with cost tracking for chat completions, image generation, image editing, embeddings, and responses API.
        """
        # Check if this is a supported endpoint for cost tracking
        is_chat_completions = OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(url_route)
        is_image_generation = OpenAIPassthroughLoggingHandler.is_openai_image_generation_route(url_route)
        is_image_editing = OpenAIPassthroughLoggingHandler.is_openai_image_editing_route(url_route)
        is_responses = OpenAIPassthroughLoggingHandler.is_openai_responses_route(url_route)
        is_embeddings = OpenAIPassthroughLoggingHandler.is_openai_embeddings_route(url_route)

        if not (is_chat_completions or is_image_generation or is_image_editing or is_responses or is_embeddings):
            # For unsupported endpoints, return None to let the system fall back to generic behavior
            return {
                "result": None,
                "kwargs": kwargs,
            }

        # Extract model from request or response
        model = request_body.get("model", response_body.get("model", ""))
        if not model:
            verbose_proxy_logger.warning("No model found in request or response for OpenAI passthrough cost tracking")
            base_handler = OpenAIPassthroughLoggingHandler()
            return base_handler.passthrough_chat_handler(
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

        try:
            response_cost = 0.0
            litellm_model_response: Optional[
                Union[
                    ModelResponse,
                    TextCompletionResponse,
                    ImageResponse,
                    ResponsesAPIResponse,
                    EmbeddingResponse,
                ]
            ] = None
            handler_instance = OpenAIPassthroughLoggingHandler()

            custom_llm_provider = kwargs.get("custom_llm_provider", "openai")

            if is_chat_completions:
                # Handle chat completions with existing logic
                provider_config = handler_instance.get_provider_config(model=model)
                # Preserve existing litellm_params to maintain metadata tags
                existing_litellm_params = kwargs.get("litellm_params", {}) or {}
                litellm_model_response = provider_config.transform_response(
                    raw_response=httpx_response,
                    model_response=litellm.ModelResponse(),
                    model=model,
                    messages=request_body.get("messages", []),
                    logging_obj=logging_obj,
                    optional_params=request_body.get("optional_params", {}),
                    api_key="",
                    request_data=request_body,
                    encoding=litellm.encoding,
                    json_mode=request_body.get("response_format", {}).get("type") == "json_object",
                    litellm_params=existing_litellm_params,
                )

                # Calculate cost using LiteLLM's cost calculator
                response_cost = litellm.completion_cost(
                    completion_response=litellm_model_response,
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                )
            elif is_image_generation:
                # Handle image generation cost calculation
                response_cost = OpenAIPassthroughLoggingHandler._calculate_image_generation_cost(
                    model=model,
                    response_body=response_body,
                    request_body=request_body,
                )
                # Mark call type for downstream image-aware logic/metrics
                try:
                    logging_obj.call_type = PassthroughCallTypes.passthrough_image_generation.value
                except Exception:
                    pass
                # Create a simple response object for logging
                litellm_model_response = ImageResponse(
                    data=response_body.get("data", []),
                    model=model,
                )
                # Set the calculated cost in _hidden_params to prevent recalculation
                if not hasattr(litellm_model_response, "_hidden_params"):
                    litellm_model_response._hidden_params = {}
                litellm_model_response._hidden_params["response_cost"] = response_cost
            elif is_image_editing:
                # Handle image editing cost calculation
                response_cost = OpenAIPassthroughLoggingHandler._calculate_image_editing_cost(
                    model=model,
                    response_body=response_body,
                    request_body=request_body,
                )
                # Mark call type for downstream image-aware logic/metrics
                try:
                    logging_obj.call_type = PassthroughCallTypes.passthrough_image_generation.value
                except Exception:
                    pass
                # Create a simple response object for logging
                litellm_model_response = ImageResponse(
                    data=response_body.get("data", []),
                    model=model,
                )
                # Set the calculated cost in _hidden_params to prevent recalculation
                if not hasattr(litellm_model_response, "_hidden_params"):
                    litellm_model_response._hidden_params = {}
                litellm_model_response._hidden_params["response_cost"] = response_cost
            elif is_responses:
                # Responses-API cost tracking — see
                # `_build_responses_api_response_and_cost` for why this needs
                # a dedicated transformer (the chat-completions transform
                # crashes on the Responses payload shape).
                (
                    litellm_model_response,
                    response_cost,
                ) = OpenAIPassthroughLoggingHandler._build_responses_api_response_and_cost(
                    model=model,
                    httpx_response=httpx_response,
                    logging_obj=logging_obj,
                    custom_llm_provider=custom_llm_provider,
                )
            elif is_embeddings:
                # Embeddings cost tracking — the response is plain JSON with a
                # standard `usage` object, so build the response object inline
                # (as the image branches do) and price it with the embedding
                # call type.
                (
                    litellm_model_response,
                    response_cost,
                ) = OpenAIPassthroughLoggingHandler._build_embeddings_response_and_cost(
                    model=model,
                    response_body=response_body,
                    custom_llm_provider=custom_llm_provider,
                )
                # Set the calculated cost in _hidden_params to prevent recalculation
                if not hasattr(litellm_model_response, "_hidden_params"):
                    litellm_model_response._hidden_params = {}
                litellm_model_response._hidden_params["response_cost"] = response_cost

            # Update kwargs with cost information
            kwargs["response_cost"] = response_cost
            kwargs["model"] = model
            kwargs["custom_llm_provider"] = custom_llm_provider

            # Extract user information for tracking
            passthrough_logging_payload: Optional[PassthroughStandardLoggingPayload] = kwargs.get(
                "passthrough_logging_payload"
            )
            if passthrough_logging_payload:
                user = handler_instance._get_user_from_metadata(
                    passthrough_logging_payload=passthrough_logging_payload,
                )
                if user:
                    kwargs["litellm_params"].setdefault("proxy_server_request", {}).setdefault("body", {})["user"] = (
                        user
                    )

            # Create standard logging object
            if litellm_model_response is not None:
                get_standard_logging_object_payload(
                    kwargs=kwargs,
                    init_response_obj=litellm_model_response,
                    start_time=start_time,
                    end_time=end_time,
                    logging_obj=logging_obj,
                    status="success",
                )

            # Update logging object with cost information
            logging_obj.model_call_details["model"] = model
            logging_obj.model_call_details["custom_llm_provider"] = custom_llm_provider
            logging_obj.model_call_details["response_cost"] = response_cost

            endpoint_type = (
                "chat_completions"
                if is_chat_completions
                else "image_generation"
                if is_image_generation
                else "image_editing"
                if is_image_editing
                else "embeddings"
                if is_embeddings
                else "responses"
            )
            verbose_proxy_logger.debug(
                f"OpenAI passthrough cost tracking - Endpoint: {endpoint_type}, Model: {model}, Cost: ${response_cost:.6f}"
            )

            return {
                "result": litellm_model_response,
                "kwargs": kwargs,
            }

        except Exception as e:
            verbose_proxy_logger.error(f"Error in OpenAI passthrough cost tracking: {str(e)}")
            # Fall back to base handler without cost tracking
            base_handler = OpenAIPassthroughLoggingHandler()
            return base_handler.passthrough_chat_handler(
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

    def _build_complete_streaming_response(
        self,
        all_chunks: list,
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
    ) -> Optional[Union[ModelResponse, TextCompletionResponse]]:
        """
        Builds complete response from raw chunks for OpenAI streaming responses.

        - Converts str chunks to generic chunks
        - Converts generic chunks to litellm chunks (OpenAI format)
        - Builds complete response from litellm chunks
        """
        try:
            # OpenAI's response iterator to parse chunks
            from litellm.llms.openai.openai import OpenAIChatCompletionResponseIterator

            openai_iterator = OpenAIChatCompletionResponseIterator(
                streaming_response=None,
                sync_stream=False,
            )

            all_openai_chunks = []
            for chunk_str in all_chunks:
                try:
                    # Parse the string chunk using the base iterator's string parser
                    from litellm.llms.base_llm.base_model_iterator import (
                        BaseModelResponseIterator,
                    )

                    # Convert string chunk to dict
                    stripped_json_chunk = BaseModelResponseIterator._string_to_dict_parser(str_line=chunk_str)

                    if stripped_json_chunk:
                        # Parse the chunk using OpenAI's chunk parser
                        transformed_chunk = openai_iterator.chunk_parser(chunk=stripped_json_chunk)
                        if transformed_chunk is not None:
                            all_openai_chunks.append(transformed_chunk)

                except (StopIteration, StopAsyncIteration, Exception) as e:
                    verbose_proxy_logger.debug(f"Error parsing streaming chunk: {e}")
                    continue

            if not all_openai_chunks:
                verbose_proxy_logger.warning("No valid chunks found in streaming response")
                return None

            # Build complete response from chunks
            complete_streaming_response = litellm.stream_chunk_builder(chunks=all_openai_chunks)

            return complete_streaming_response

        except Exception as e:
            verbose_proxy_logger.error(f"Error building complete streaming response: {str(e)}")
            return None

    @staticmethod
    def _extract_responses_api_completed_response(
        all_chunks: List[str],
    ) -> Optional[dict]:
        """Return the final `response` object of a Responses-API event stream.

        A streamed Responses call does not emit chat-completion chunks — it
        emits typed events (`response.created`, `response.output_text.delta`,
        ..., `response.completed`). `OpenAIChatCompletionResponseIterator`
        understands only the chat shape, so these streams reassemble to
        nothing and the request is billed at $0. Only the terminal
        `response.completed` event carries the finished `response` object with
        `usage.input_tokens` / `usage.output_tokens`, so that is the one we
        cost from.

        Detection is on the JSON payload's own `type` field, not the SSE
        `event:` line: `_convert_raw_bytes_to_str_lines` splits the stream on
        newlines, so `event: response.completed` and its `data: {...}` arrive
        as separate entries. Scanned in reverse because the completed event is
        the last one on the wire.

        Returns None when these chunks are not a Responses event stream, which
        keeps the chat-completions path unchanged.
        """
        from litellm.llms.base_llm.base_model_iterator import (
            BaseModelResponseIterator,
        )

        for chunk_str in reversed(all_chunks):
            try:
                parsed_chunk = BaseModelResponseIterator._string_to_dict_parser(str_line=chunk_str)
            except Exception as e:
                verbose_proxy_logger.debug(f"Error parsing streaming chunk as Responses event: {e}")
                continue
            if not isinstance(parsed_chunk, dict):
                continue
            if parsed_chunk.get("type") == "response.completed":
                completed_response = parsed_chunk.get("response")
                if isinstance(completed_response, dict):
                    return completed_response
        return None

    @staticmethod
    def _handle_logging_openai_collected_chunks(
        litellm_logging_obj: LiteLLMLoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        start_time: datetime,
        all_chunks: List[str],
        end_time: datetime,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Handle logging for collected OpenAI streaming chunks with cost tracking.
        """
        try:
            # Extract model from request body
            model = request_body.get("model", "gpt-4o")

            handler = OpenAIPassthroughLoggingHandler()
            handler_instance = handler
            custom_llm_provider = litellm_logging_obj.model_call_details.get("custom_llm_provider", "openai")

            complete_response: Optional[
                Union[ModelResponse, TextCompletionResponse, ResponsesAPIResponse]
            ] = None
            responses_api_completed_response = (
                OpenAIPassthroughLoggingHandler._extract_responses_api_completed_response(all_chunks=all_chunks)
            )

            if responses_api_completed_response is not None:
                # Streamed Responses API call — reuse the non-streaming
                # Responses costing path rather than duplicating cost math.
                # The transformer takes an httpx.Response, so wrap the
                # completed event's `response` object (which is exactly the
                # non-streaming response body) in one.
                #
                # Prefer the model echoed by the completed event when the
                # request body carried none — on Azure the deployment lives in
                # the URL, and costing against the "gpt-4o" default would be
                # wrong.
                model = request_body.get("model") or responses_api_completed_response.get("model") or model
                (
                    complete_response,
                    response_cost,
                ) = OpenAIPassthroughLoggingHandler._build_responses_api_response_and_cost(
                    model=model,
                    httpx_response=httpx.Response(
                        status_code=200,
                        json=responses_api_completed_response,
                    ),
                    logging_obj=litellm_logging_obj,
                    custom_llm_provider=custom_llm_provider,
                )
            else:
                # Build complete response from chunks using our streaming handler
                complete_response = handler._build_complete_streaming_response(
                    all_chunks=all_chunks,
                    litellm_logging_obj=litellm_logging_obj,
                    model=model,
                )

                if complete_response is None:
                    verbose_proxy_logger.warning("Failed to build complete response from OpenAI streaming chunks")
                    return {
                        "result": None,
                        "kwargs": {},
                    }

                # Calculate cost using LiteLLM's cost calculator
                response_cost = litellm.completion_cost(
                    completion_response=complete_response,
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                )

            # Preserve existing litellm_params to maintain metadata tags
            existing_litellm_params = litellm_logging_obj.model_call_details.get("litellm_params", {}) or {}

            # Prepare kwargs for logging
            kwargs = {
                "response_cost": response_cost,
                "model": model,
                "custom_llm_provider": custom_llm_provider,
                "litellm_params": existing_litellm_params.copy(),
            }

            # Extract user information for tracking
            passthrough_logging_payload: Optional[PassthroughStandardLoggingPayload] = (
                litellm_logging_obj.model_call_details.get("passthrough_logging_payload")
            )
            if passthrough_logging_payload:
                user = handler_instance._get_user_from_metadata(
                    passthrough_logging_payload=passthrough_logging_payload,
                )
                if user:
                    kwargs["litellm_params"].setdefault("proxy_server_request", {}).setdefault("body", {})["user"] = (
                        user
                    )

            # Create standard logging object
            get_standard_logging_object_payload(
                kwargs=kwargs,
                init_response_obj=complete_response,
                start_time=start_time,
                end_time=end_time,
                logging_obj=litellm_logging_obj,
                status="success",
            )

            # Update logging object with cost information
            litellm_logging_obj.model_call_details["model"] = model
            litellm_logging_obj.model_call_details["custom_llm_provider"] = custom_llm_provider
            litellm_logging_obj.model_call_details["response_cost"] = response_cost

            verbose_proxy_logger.debug(
                f"OpenAI streaming passthrough cost tracking - Model: {model}, Cost: ${response_cost:.6f}"
            )

            return {
                "result": complete_response,
                "kwargs": kwargs,
            }

        except Exception as e:
            verbose_proxy_logger.error(f"Error in OpenAI streaming passthrough cost tracking: {str(e)}")
            return {
                "result": None,
                "kwargs": {},
            }
