from collections.abc import Sequence
from datetime import datetime
from typing import Final

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.gemini.videos.transformation import GeminiVideoConfig
from litellm.llms.vertex_ai.common_utils import get_vertex_model_id_from_url
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
    GeminiSseChunkParser,
    VertexPassthroughLoggingHandler,
)
from litellm.types.utils import (
    ModelResponse,
    TextCompletionResponse,
)


class GeminiPassthroughLoggingHandler:
    @staticmethod
    def gemini_passthrough_handler(
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
        if VertexPassthroughLoggingHandler.is_interactions_route(url_route):
            return VertexPassthroughLoggingHandler.interactions_passthrough_handler(
                httpx_response=httpx_response,
                request_body=request_body,
                logging_obj=logging_obj,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
                custom_llm_provider="gemini",
                vertex_location=None,
            )
        if "predictLongRunning" in url_route:
            model = GeminiPassthroughLoggingHandler.extract_model_from_url(url_route)

            gemini_video_config = GeminiVideoConfig()
            litellm_video_response = gemini_video_config.transform_video_create_response(
                model=model,
                raw_response=httpx_response,
                logging_obj=logging_obj,
                custom_llm_provider="gemini",
                request_data=request_body,
            )
            logging_obj.model = model
            logging_obj.model_call_details["model"] = model
            logging_obj.model_call_details["custom_llm_provider"] = "gemini"
            logging_obj.custom_llm_provider = "gemini"

            response_cost = litellm.completion_cost(
                completion_response=litellm_video_response,
                model=model,
                custom_llm_provider="gemini",
                call_type="create_video",
            )

            # Set response_cost in _hidden_params to prevent recalculation
            if not hasattr(litellm_video_response, "_hidden_params"):
                litellm_video_response.hidden_params = {}
            litellm_video_response.hidden_params["response_cost"] = response_cost

            kwargs["response_cost"] = response_cost
            kwargs["model"] = model
            kwargs["custom_llm_provider"] = "gemini"
            logging_obj.model_call_details["response_cost"] = response_cost
            return {
                "result": litellm_video_response,
                "kwargs": kwargs,
            }

        if "generateContent" in url_route:
            model = GeminiPassthroughLoggingHandler.extract_model_from_url(url_route)

            # Use Gemini config for transformation
            instance_of_gemini_llm = litellm.GoogleAIStudioGeminiConfig()
            litellm_model_response: ModelResponse = instance_of_gemini_llm.transform_response(
                model=model,
                messages=[{"role": "user", "content": "no-message-pass-through-endpoint"}],
                raw_response=httpx_response,
                model_response=litellm.ModelResponse(),
                logging_obj=logging_obj,
                optional_params={},
                litellm_params={},
                api_key="",
                request_data={},
                encoding=getattr(litellm, "encoding", None),
            )
            kwargs = GeminiPassthroughLoggingHandler._create_gemini_response_logging_payload_for_generate_content(
                litellm_model_response=litellm_model_response,
                model=model,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
                logging_obj=logging_obj,
                custom_llm_provider="gemini",
            )

            return {
                "result": litellm_model_response,
                "kwargs": kwargs,
            }
        else:
            return {
                "result": None,
                "kwargs": kwargs,
            }

    # A `_handle_logging_gemini_collected_chunks` used to live here with no call
    # site: streamed Gemini passthrough already gets costed via
    # VertexPassthroughLoggingHandler (HttpPassThroughEndpointHelpers.get_endpoint_type
    # classifies every generateContent/streamGenerateContent URL, Gemini's AI
    # Studio host included, as EndpointType.VERTEX_AI, and
    # VertexPassthroughLoggingHandler._get_custom_llm_provider_from_url resolves
    # generativelanguage.googleapis.com to "gemini"), so a duplicate
    # EndpointType.GEMINI dispatch branch is unreachable code that looks like
    # coverage without providing any. See streaming_handler.py's dispatch and
    # get_endpoint_type in pass_through_endpoints.py, which never returns
    # EndpointType.GEMINI.

    @staticmethod
    def _build_complete_streaming_response(
        all_chunks: Sequence[str],
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
        url_route: str,
    ) -> ModelResponse | TextCompletionResponse | None:
        if "generateContent" in url_route or "streamGenerateContent" in url_route:
            gemini_iterator: Final = GeminiSseChunkParser(
                streaming_response=None,
                sync_stream=False,
                logging_obj=litellm_logging_obj,
            )
            parsed_chunks: Final = [gemini_iterator.parse_sse_chunk(chunk) for chunk in all_chunks]
        else:
            return None

        if len(parsed_chunks) == 0:
            return None

        all_openai_chunks: Final = [parsed_chunk for parsed_chunk in parsed_chunks if parsed_chunk is not None]

        complete_streaming_response: Final = litellm.stream_chunk_builder(chunks=all_openai_chunks)

        return complete_streaming_response

    @staticmethod
    def extract_model_from_url(url: str) -> str:
        return get_vertex_model_id_from_url(url) or "unknown"

    @staticmethod
    def _create_gemini_response_logging_payload_for_generate_content(
        litellm_model_response: ModelResponse | TextCompletionResponse,
        model: str,
        kwargs: dict,
        start_time: datetime,
        end_time: datetime,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str,
    ):
        """
        Create the standard logging object for Gemini passthrough generateContent (streaming and non-streaming)
        """

        response_cost = litellm.completion_cost(
            completion_response=litellm_model_response,
            model=model,
            custom_llm_provider="gemini",
        )

        kwargs["response_cost"] = response_cost
        kwargs["model"] = model
        kwargs["custom_llm_provider"] = custom_llm_provider

        # pretty print standard logging object
        verbose_proxy_logger.debug("kwargs= %s", kwargs)

        # set litellm_call_id to logging response object
        litellm_model_response.id = logging_obj.litellm_call_id
        logging_obj.model = litellm_model_response.model or model
        logging_obj.model_call_details["model"] = logging_obj.model
        logging_obj.model_call_details["custom_llm_provider"] = custom_llm_provider
        return kwargs
