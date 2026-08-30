# What is this?
## Common Utility file for Logging handler
# Logging function -> log the exact model details + what's being sent | Non-Blocking
import copy
import datetime
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime as dt_object
from functools import lru_cache
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Any, Final, Literal, Optional, Union, cast

from httpx import Response
from pydantic import BaseModel

import litellm
from litellm import (
    _custom_logger_compatible_callbacks_literal,
    json_logs,
    log_raw_request_response,
    turn_off_message_logging,
)
from litellm._logging import (
    _is_debugging_on,
    _redact_string,
    session_id_var,
    set_session_id,
    set_trace_id,
    trace_id_var,
    verbose_logger,
)
from litellm._uuid import uuid
from litellm.batches.batch_utils import _handle_completed_batch
from litellm.caching.caching import DualCache, InMemoryCache
from litellm.caching.caching_handler import LLMCachingHandler
from litellm.constants import (
    DEFAULT_MOCK_RESPONSE_COMPLETION_TOKEN_COUNT,
    DEFAULT_MOCK_RESPONSE_PROMPT_TOKEN_COUNT,
    SENTRY_DENYLIST,
    SENTRY_PII_DENYLIST,
)
from litellm.cost_calculator import (
    RealtimeAPITokenUsageProcessor,
    _select_model_name_for_cost_calc,
)
from litellm.exceptions import (
    BudgetExceededError,
    validate_rate_limit_category,
    validate_rate_limit_type,
)
from litellm.integrations.agentops import AgentOps
from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook
from litellm.integrations.arize.arize import ArizeLogger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.deepeval.deepeval import DeepEvalLogger
from litellm.integrations.mlflow import MlflowLogger
from litellm.integrations.sqs import SQSLogger
from litellm.litellm_core_utils.core_helpers import is_expected_client_error, reconstruct_model_name
from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
from litellm.litellm_core_utils.internal_call_metadata import (
    MODEL_ACCESS_GROUP_METADATA_KEY,
    is_unbilled_non_inference_call,
)
from litellm.litellm_core_utils.llm_cost_calc.guardrail_cost import (
    cost_breakdown_with_guardrail,
    guardrail_information_cost,
)
from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import (
    StandardBuiltInToolCostTracking,
)
from litellm.litellm_core_utils.llm_cost_calc.usage_object_transformation import (
    InteractionsUsageObjectTransformation,
)
from litellm.litellm_core_utils.logging_utils import truncate_base64_in_messages
from litellm.litellm_core_utils.model_param_helper import ModelParamHelper
from litellm.litellm_core_utils.redact_messages import (
    redact_message_input_output_from_custom_logger,
    redact_message_input_output_from_logging,
    redact_streaming_responses_for_custom_logger,
)
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.responses.utils import ResponseAPILoggingUtils
from litellm.types.agents import LiteLLMSendMessageResponse
from litellm.types.containers.main import ContainerObject
from litellm.types.interactions import (
    InteractionsAPIResponse,
    InteractionsAPIStreamingResponse,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    Batch,
    FineTuningJob,
    HttpxBinaryResponseContent,
    OpenAIFileObject,
    OpenAIModerationResponse,
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponsesAPIResponse,
)
from litellm.types.mcp import MCPPostCallResponseObject
from litellm.types.prompts.init_prompts import PromptSpec
from litellm.types.rerank import RerankResponse
from litellm.types.utils import (
    CachingDetails,
    CallTypes,
    CostBreakdown,
    CostResponseTypes,
    CustomPricingLiteLLMParams,
    DynamicPromptManagementParamLiteral,
    EmbeddingResponse,
    GuardrailStatus,
    ImageResponse,
    LiteLLMBatch,
    LiteLLMLoggingBaseClass,
    LiteLLMRealtimeStreamLoggingObject,
    ModelInfo,
    ModelResponse,
    ModelResponseStream,
    RawRequestTypedDict,
    StandardBuiltInToolsParams,
    StandardCallbackDynamicParams,
    StandardLoggingAdditionalHeaders,
    StandardLoggingHiddenParams,
    StandardLoggingMCPToolCall,
    StandardLoggingMetadata,
    StandardLoggingModelCostFailureDebugInformation,
    StandardLoggingModelInformation,
    StandardLoggingPayload,
    StandardLoggingPayloadErrorInformation,
    StandardLoggingPayloadStatus,
    StandardLoggingPayloadStatusFields,
    StandardLoggingPromptManagementMetadata,
    StandardLoggingVectorStoreRequest,
    TextCompletionResponse,
    TranscriptionResponse,
    Usage,
)
from litellm.types.videos.main import VideoObject
from litellm.utils import _get_base_model_from_metadata, executor, print_verbose

from ..integrations.argilla import ArgillaLogger
from ..integrations.arize.arize_phoenix import ArizePhoenixLogger
from ..integrations.athina import AthinaLogger
from ..integrations.azure_sentinel.azure_sentinel import AzureSentinelLogger
from ..integrations.azure_storage.azure_storage import AzureBlobStorageLogger
from ..integrations.custom_prompt_management import CustomPromptManagement
from ..integrations.datadog.datadog import DataDogLogger
from ..integrations.datadog.datadog_llm_obs import DataDogLLMObsLogger
from ..integrations.datadog.datadog_metrics import DatadogMetricsLogger
from ..integrations.dotprompt import DotpromptManager
from ..integrations.dynamodb import DyanmoDBLogger
from ..integrations.galileo import GalileoObserve
from ..integrations.gcs_bucket.gcs_bucket import GCSBucketLogger
from ..integrations.gcs_pubsub.pub_sub import GcsPubSubLogger
from ..integrations.greenscale import GreenscaleLogger
from ..integrations.helicone import HeliconeLogger
from ..integrations.humanloop import HumanloopLogger
from ..integrations.lago import LagoLogger
from ..integrations.langfuse.langfuse import LangFuseLogger
from ..integrations.langfuse.langfuse_handler import LangFuseHandler
from ..integrations.langfuse.langfuse_prompt_management import LangfusePromptManagement
from ..integrations.langsmith import LangsmithLogger
from ..integrations.litellm_agent import LiteLLMAgentModelResolver
from ..integrations.literal_ai import LiteralAILogger
from ..integrations.logfire_logger import LogfireLevel, LogfireLogger
from ..integrations.lunary import LunaryLogger
from ..integrations.newrelic import NewRelicLogger
from ..integrations.openmeter import OpenMeterLogger
from ..integrations.opik.opik import OpikLogger
from ..integrations.posthog import PostHogLogger
from ..integrations.prompt_layer import PromptLayerLogger
from ..integrations.s3 import S3Logger
from ..integrations.s3_v2 import S3Logger as S3V2Logger
from ..integrations.supabase import Supabase
from ..integrations.traceloop import TraceloopLogger
from .exception_mapping_utils import _get_response_headers
from .initialize_dynamic_callback_params import (
    get_trusted_callback_params,
)
from .initialize_dynamic_callback_params import (
    initialize_standard_callback_dynamic_params as _initialize_standard_callback_dynamic_params,
)
from .specialty_caches.dynamic_logging_cache import DynamicLoggingCache

if TYPE_CHECKING:
    from mcp.types import EmbeddedResource, ImageContent, TextContent

    from litellm.integrations.otel.logger import OpenTelemetryV2
    from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig
try:
    from litellm_enterprise.enterprise_callbacks.callback_controls import (
        EnterpriseCallbackControls,
    )
    from litellm_enterprise.enterprise_callbacks.pagerduty.pagerduty import (
        PagerDutyAlerting,
    )
    from litellm_enterprise.enterprise_callbacks.send_emails.resend_email import (
        ResendEmailLogger,
    )
    from litellm_enterprise.enterprise_callbacks.send_emails.sendgrid_email import (
        SendGridEmailLogger,
    )
    from litellm_enterprise.enterprise_callbacks.send_emails.smtp_email import (
        SMTPEmailLogger,
    )
    from litellm_enterprise.litellm_core_utils.litellm_logging import (
        StandardLoggingPayloadSetup as EnterpriseStandardLoggingPayloadSetup,
    )

    from litellm.integrations.generic_api.generic_api_callback import GenericAPILogger

    EnterpriseStandardLoggingPayloadSetupVAR: type[EnterpriseStandardLoggingPayloadSetup] | None = (
        EnterpriseStandardLoggingPayloadSetup
    )
except Exception as e:
    verbose_logger.debug("[Non-Blocking] Unable to import GenericAPILogger - LiteLLM Enterprise Feature - %s", e)
    GenericAPILogger = CustomLogger
    ResendEmailLogger = CustomLogger
    SendGridEmailLogger = CustomLogger
    SMTPEmailLogger = CustomLogger
    PagerDutyAlerting = CustomLogger
    EnterpriseCallbackControls = None
    EnterpriseStandardLoggingPayloadSetupVAR = None
if TYPE_CHECKING:
    from litellm.integrations.generic_api.generic_api_callback import (
        GenericAPILogger as _GenericAPILoggerCls,
    )

    _GENERIC_API_LOGGER_CLS: Final = _GenericAPILoggerCls
    _RESEND_EMAIL_LOGGER_FACTORY: Final = CustomLogger
    _SENDGRID_EMAIL_LOGGER_FACTORY: Final = CustomLogger
    _SMTP_EMAIL_LOGGER_FACTORY: Final = CustomLogger
    _PAGERDUTY_ALERTING_FACTORY: Final = CustomLogger
else:
    _GENERIC_API_LOGGER_CLS: Final = GenericAPILogger
    _RESEND_EMAIL_LOGGER_FACTORY: Final = ResendEmailLogger
    _SENDGRID_EMAIL_LOGGER_FACTORY: Final = SendGridEmailLogger
    _SMTP_EMAIL_LOGGER_FACTORY: Final = SMTPEmailLogger
    _PAGERDUTY_ALERTING_FACTORY: Final = PagerDutyAlerting
_in_memory_loggers: Final[list[CustomLogger]] = []

_STANDARD_LOGGING_METADATA_KEYS: Final[frozenset[str]] = frozenset(StandardLoggingMetadata.__annotations__.keys())

### GLOBAL VARIABLES ###

# Cache custom pricing keys as frozenset for O(1) lookups instead of looping through 49 keys
_CUSTOM_PRICING_KEYS: Final[frozenset[str]] = frozenset(CustomPricingLiteLLMParams.model_fields.keys())

sentry_sdk_instance = None
capture_exception = None
add_breadcrumb = None
slack_app = None
alerts_channel = None
heliconeLogger = None
athinaLogger = None
promptLayerLogger = None
logfireLogger = None
weightsBiasesLogger = None
customLogger = None
langFuseLogger = None
openMeterLogger = None
lagoLogger: Final = None
dataDogLogger = None
prometheusLogger = None
dynamoLogger = None
s3Logger = None
greenscaleLogger = None
lunaryLogger = None
supabaseClient = None
deepevalLogger = None
callback_list: Final[list[str] | None] = []
user_logger_fn: Final = None
additional_details: Final[dict[str, str] | None] = {}
local_cache: Final[dict[str, str] | None] = {}
last_fetched_at: Final = None
last_fetched_at_keys: Final = None


####
class ServiceTraceIDCache:
    def __init__(self) -> None:
        self.cache = InMemoryCache()

    def get_cache(self, litellm_call_id: str, service_name: str) -> str | None:
        key_name: Final = f"{service_name}:{litellm_call_id}"
        response: Final = self.cache.get_cache(key=key_name)
        return response

    def set_cache(self, litellm_call_id: str, service_name: str, trace_id: str) -> None:
        key_name: Final = f"{service_name}:{litellm_call_id}"
        self.cache.set_cache(key=key_name, value=trace_id)


in_memory_trace_id_cache: Final = ServiceTraceIDCache()
in_memory_dynamic_logger_cache: Final = DynamicLoggingCache()

# Cached lazy import for PrometheusLogger
# Module-level cache to avoid repeated imports while preserving memory benefits
_PrometheusLogger = None


def _get_cached_prometheus_logger():
    """
    Get cached PrometheusLogger class.
    Lazy imports on first call to avoid loading prometheus.py and utils.py at import time (60MB saved).
    Subsequent calls use cached class for better performance.
    """
    global _PrometheusLogger
    if _PrometheusLogger is None:
        from litellm.integrations.prometheus import PrometheusLogger

        _PrometheusLogger = PrometheusLogger
    return _PrometheusLogger


_DEPLOYMENT_PRICING_KEYS: Final = (
    "input_cost_per_token",
    "output_cost_per_token",
    "input_cost_per_token_batches",
    "output_cost_per_token_batches",
)


def deployment_pricing_model_info(model_id: str | None, deployment_model: str | None) -> ModelInfo | None:
    """Pricing the router registered under this deployment's model_info.id.

    Returns None when the deployment declares no pricing of its own, so the
    caller falls back to the global cost map. The raw registration is what
    decides that: the router registers an entry for every deployment, and
    get_model_info fills absent costs with 0, so asking it directly cannot
    tell "configured as free" apart from "no pricing configured". A deployment
    may declare only one side of its pricing, so the side it leaves out keeps
    the model's published rates instead of billing as zero. Ownership is per
    token direction: declaring either rate for a direction takes that whole
    direction, so a published batch rate can never displace a standard rate
    the deployment configured itself.
    """
    if model_id is None:
        return None
    registered: Final = litellm.model_cost.get(model_id)
    if not isinstance(registered, dict) or not any(registered.get(key) is not None for key in _DEPLOYMENT_PRICING_KEYS):
        return None
    try:
        merged: Final = litellm.get_model_info(model=model_id).copy()
    except Exception:  # noqa: BLE001  # get_model_info raises for ids it cannot resolve a provider for
        return None
    published: Final = _published_pricing(deployment_model)
    if published is None:
        return merged
    declares_input: Final = (
        registered.get("input_cost_per_token") is not None or registered.get("input_cost_per_token_batches") is not None
    )
    declares_output: Final = (
        registered.get("output_cost_per_token") is not None
        or registered.get("output_cost_per_token_batches") is not None
    )
    if not declares_input:
        merged["input_cost_per_token"] = published.get("input_cost_per_token")
        merged["input_cost_per_token_batches"] = published.get("input_cost_per_token_batches")
    if not declares_output:
        merged["output_cost_per_token"] = published.get("output_cost_per_token")
        merged["output_cost_per_token_batches"] = published.get("output_cost_per_token_batches")
    return merged


def _published_pricing(deployment_model: str | None) -> ModelInfo | None:
    """The cost map's own entry for the deployment's model, when it resolves."""
    if deployment_model is None:
        return None
    try:
        return litellm.get_model_info(model=deployment_model)
    except Exception:  # noqa: BLE001  # no published entry to layer the declared rates over
        return None


def _resolve_vertex_location_for_cost(
    custom_llm_provider: str | None,
    litellm_params: Mapping[str, object] | None,
    optional_params: Mapping[str, object] | None,
    model: str,
) -> str | None:
    """
    The Vertex AI location a request was served from, resolved the same way
    dispatch resolves it, so regional deployments price with the
    regional-endpoint uplift. None for non-Vertex providers.

    Chat dispatch reads the location from request kwargs, which reach this
    logging object through optional_params: on the proxy the logging object is
    created before the router picks a deployment, so the deployment's location
    never lands in litellm_params.
    """
    if custom_llm_provider is None or not custom_llm_provider.startswith("vertex_ai"):
        return None
    from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

    empty: Final[Mapping[str, object]] = MappingProxyType({})
    configured_location: Final = (
        VertexBase.explicit_vertex_ai_location(optional_params or empty)
        or VertexBase.explicit_vertex_ai_location(litellm_params or empty)
        or VertexBase.safe_get_vertex_ai_location(empty)
    )
    return VertexBase.get_vertex_region(configured_location, model)


class Logging(LiteLLMLoggingBaseClass):
    global \
        supabaseClient, \
        promptLayerLogger, \
        weightsBiasesLogger, \
        logfireLogger, \
        capture_exception, \
        add_breadcrumb, \
        lunaryLogger, \
        logfireLogger, \
        prometheusLogger, \
        slack_app
    custom_pricing: bool = False
    stream_options = None
    litellm_request_debug: bool = False

    def __init__(
        self,
        model: str,
        messages,
        stream,
        call_type,
        start_time,
        litellm_call_id: str,
        function_id: str,
        litellm_trace_id: str | None = None,
        dynamic_input_callbacks: list[str | Callable | CustomLogger] | None = None,
        dynamic_success_callbacks: list[str | Callable | CustomLogger] | None = None,
        dynamic_async_success_callbacks: list[str | Callable | CustomLogger] | None = None,
        dynamic_failure_callbacks: list[str | Callable | CustomLogger] | None = None,
        dynamic_async_failure_callbacks: list[str | Callable | CustomLogger] | None = None,
        applied_guardrails: list[str] | None = None,
        kwargs: dict | None = None,
        log_raw_request_response: bool = False,
        supports_correlation_logging: bool = True,
    ):
        _input: Final[str | None] = messages  # save original value of messages
        if messages is not None:
            if isinstance(messages, str):
                messages = [
                    {"role": "user", "content": messages}
                ]  # convert text completion input to the chat completion format
            elif isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], str):
                new_messages: Final = []
                for m in messages:
                    new_messages.append({"role": "user", "content": m})
                messages = new_messages

        self.model = model
        # Shallow copy of the outer list only (inner message dicts are shared).
        # Safe because the logging layer does not mutate individual message dicts.
        _copy_start: Final = time.time()
        self.messages = copy.copy(messages) if messages is not None else None
        self.message_copy_duration_ms: float = (time.time() - _copy_start) * 1000
        self.callback_duration_ms: float = 0.0
        self.stream = stream
        self.start_time = start_time  # log the call start time
        self.call_type = call_type
        self.litellm_call_id = litellm_call_id
        self.litellm_trace_id: str = litellm_trace_id if litellm_trace_id else str(uuid.uuid4())

        # Capture the pre-call *value* (not a contextvars.Token) so restoration works
        # even if this attempt's own logging ends up dispatched onto a different
        # asyncio Task/context (e.g. via asyncio.create_task or the logging worker) -
        # a Token can only be reset in the exact Context where it was created.
        self._pre_call_trace_id: str = trace_id_var.get()
        self._pre_call_session_id: str = session_id_var.get()
        _sid: Final = kwargs.get("litellm_session_id") if kwargs else None
        self.litellm_session_id: str = str(_sid) if _sid else ""
        # supports_correlation_logging is False for calls originating from the
        # sync client entry point (wrapper() in utils.py): a plain OS thread
        # has no per-call context isolation the way an asyncio Task does, and
        # a thread pool's worker threads are recycled across unrelated
        # requests, so stamping trace_id/session_id there risks one request's
        # ids leaking into a different, later request on the same thread. Sync
        # support is deferred to a follow-up PR with its own safe-restore
        # mechanism; async calls (the proxy's only call path) are unaffected.
        if supports_correlation_logging:
            set_trace_id(self.litellm_trace_id)
            set_session_id(self.litellm_session_id)
        # set_trace_id()/set_session_id() sanitize (strip control chars, bound
        # length) before storing, so the contextvar's actual value can differ
        # from self.litellm_trace_id/litellm_session_id. Capture what was
        # really stored - _restore_correlation_context_if_unclaimed() must
        # compare against this, not the raw ids, or a caller-supplied id
        # containing control characters/oversized input would never match
        # and cleanup would be skipped forever.
        self._own_trace_id: str = trace_id_var.get()
        self._own_session_id: str = session_id_var.get()

        self.function_id = function_id
        self.streaming_chunks: list[Any] = []  # for generating complete stream response
        self.sync_streaming_chunks: list[Any] = []  # for generating complete stream response
        self.log_raw_request_response = log_raw_request_response

        # Initialize dynamic callbacks
        self.dynamic_input_callbacks: list[str | Callable | CustomLogger] | None = dynamic_input_callbacks
        self.dynamic_success_callbacks: list[str | Callable | CustomLogger] | None = dynamic_success_callbacks
        self.dynamic_async_success_callbacks: list[str | Callable | CustomLogger] | None = (
            dynamic_async_success_callbacks
        )
        self.dynamic_failure_callbacks: list[str | Callable | CustomLogger] | None = dynamic_failure_callbacks
        self.dynamic_async_failure_callbacks: list[str | Callable | CustomLogger] | None = (
            dynamic_async_failure_callbacks
        )

        ## DYNAMIC LANGFUSE / GCS / logging callback KEYS ##
        self.standard_callback_dynamic_params: StandardCallbackDynamicParams = (
            self.initialize_standard_callback_dynamic_params(kwargs)
        )
        self._trusted_callback_vars: tuple[tuple[str, str], ...] = get_trusted_callback_params(kwargs)

        # Process dynamic callbacks (after standard_callback_dynamic_params is initialized,
        # so team-scoped credentials are available for callback initialization)
        self.process_dynamic_callbacks()
        self.standard_built_in_tools_params: StandardBuiltInToolsParams = (
            self.initialize_standard_built_in_tools_params(kwargs)
        )
        ## TIME TO FIRST TOKEN LOGGING ##
        self.completion_start_time: datetime.datetime | None = None
        self._llm_caching_handler: LLMCachingHandler | None = None

        # INITIAL LITELLM_PARAMS
        litellm_params = {}
        if kwargs is not None:
            litellm_params = get_litellm_params(**kwargs)
            litellm_params = scrub_sensitive_keys_in_metadata(litellm_params)

        self.litellm_params = litellm_params

        # Initialize cost breakdown field
        self.cost_breakdown: CostBreakdown | None = None

        # Init Caching related details
        self.caching_details: CachingDetails | None = None
        # Timing for results that cannot carry ``_hidden_params`` (plain-dict /v1/messages
        # responses and the bridge stream wrappers); see ``update_response_metadata``.
        self.response_timing_metrics: Mapping[str, float] = {}  # mutable-ok: kept deep-copyable

        # Passthrough endpoint guardrails config for field targeting
        self.passthrough_guardrails_config: dict[str, Any] | None = None

        self.model_call_details: dict[str, Any] = {
            "litellm_trace_id": self.litellm_trace_id,
            "litellm_call_id": litellm_call_id,
            "input": _input,
            "litellm_params": litellm_params,
            "applied_guardrails": applied_guardrails,
            "model": model,
        }

        # Set by proxy request handlers to defer spend-log fire until after
        # post_call guardrails have run; the @client decorator then stores the
        # enqueue closure here instead of firing it immediately.
        self._defer_async_logging: bool = False
        self._enqueue_deferred_logging: Callable[[], None] | None = None

    def set_response_timing_metrics(self, timing_metrics: Mapping[str, float]) -> None:
        """Keep ``_response_ms`` / ``litellm_overhead_time_ms`` for a result that has no ``_hidden_params``."""
        self.response_timing_metrics = dict(timing_metrics)  # mutable-ok: kept deep-copyable

    def process_dynamic_callbacks(self):
        """
        Initializes CustomLogger compatible callbacks in self.dynamic_* callbacks

        If a callback is in litellm._known_custom_logger_compatible_callbacks, it needs to be intialized and added to the respective dynamic_* callback list.
        """
        # Process input callbacks
        self.dynamic_input_callbacks = self._process_dynamic_callback_list(
            self.dynamic_input_callbacks, dynamic_callbacks_type="input"
        )

        # Process failure callbacks
        self.dynamic_failure_callbacks = self._process_dynamic_callback_list(
            self.dynamic_failure_callbacks, dynamic_callbacks_type="failure"
        )

        # Process async failure callbacks
        self.dynamic_async_failure_callbacks = self._process_dynamic_callback_list(
            self.dynamic_async_failure_callbacks, dynamic_callbacks_type="async_failure"
        )

        # Process success callbacks
        self.dynamic_success_callbacks = self._process_dynamic_callback_list(
            self.dynamic_success_callbacks, dynamic_callbacks_type="success"
        )

        # Process async success callbacks
        self.dynamic_async_success_callbacks = self._process_dynamic_callback_list(
            self.dynamic_async_success_callbacks, dynamic_callbacks_type="async_success"
        )

    def _process_dynamic_callback_list(
        self,
        callback_list: list[str | Callable | CustomLogger] | None,
        dynamic_callbacks_type: Literal["input", "success", "failure", "async_success", "async_failure"],
    ) -> list[str | Callable | CustomLogger] | None:
        """
        Helper function to initialize CustomLogger compatible callbacks in self.dynamic_* callbacks

        - If a callback is in litellm._known_custom_logger_compatible_callbacks,
        replace the string with the initialized callback class.
        - If dynamic callback is a "success" callback that is a known_custom_logger_compatible_callbacks then add it to dynamic_async_success_callbacks
        - If dynamic callback is a "failure" callback that is a known_custom_logger_compatible_callbacks then add it to dynamic_failure_callbacks
        """
        if callback_list is None:
            return None

        processed_list: Final[list[str | Callable | CustomLogger]] = []
        for callback in callback_list:
            if isinstance(callback, str) and callback in litellm._known_custom_logger_compatible_callbacks:
                for callback_instance in self._resolve_dynamic_callback_string(callback):
                    processed_list.append(callback_instance)

                    # If processing dynamic_success_callbacks, add to dynamic_async_success_callbacks
                    if dynamic_callbacks_type == "success":
                        if self.dynamic_async_success_callbacks is None:
                            self.dynamic_async_success_callbacks = []
                        self.dynamic_async_success_callbacks.append(callback_instance)
                    elif dynamic_callbacks_type == "failure":
                        if self.dynamic_async_failure_callbacks is None:
                            self.dynamic_async_failure_callbacks = []
                        self.dynamic_async_failure_callbacks.append(callback_instance)
            else:
                processed_list.append(callback)
        return processed_list

    def _resolve_dynamic_callback_string(self, callback: str) -> "tuple[CustomLogger, ...]":
        """
        Resolve a known callback name to the logger instance(s) it dispatches to.

        For callbacks that support team-scoped credentials (datadog, newrelic),
        only the proxy-stamped team/key callback vars are passed as
        custom_logger_init_args: dd_*/newrelic_* params are blocked from
        standard_callback_dynamic_params (request-level security), so the
        trusted-vars channel is the only way credentials reach a per-team logger.
        """
        _trusted_var_prefix: Final = "dd_" if callback == "datadog" else "newrelic_" if callback == "newrelic" else None
        _custom_logger_init_args: Final[dict | None] = (
            {k: v for k, v in self._trusted_callback_vars if k.startswith(_trusted_var_prefix)}
            if _trusted_var_prefix is not None
            else None
        )

        callback_class: Final = _init_custom_logger_compatible_class(
            callback,
            internal_usage_cache=None,
            llm_router=None,
            custom_logger_init_args=_custom_logger_init_args,
        )
        if callback_class is None:
            return ()

        # With team creds, "newrelic" resolves to the per-team METRICS logger;
        # resolve the name again without creds so the trace logger (OTel v2 /
        # legacy agent) keeps receiving this request.
        _newrelic_trace_class: Final = (
            _init_custom_logger_compatible_class(callback, internal_usage_cache=None, llm_router=None)
            if callback == "newrelic" and _custom_logger_init_args and _custom_logger_init_args.get("newrelic_api_key")
            else None
        )
        if _newrelic_trace_class is not None and _newrelic_trace_class is not callback_class:
            return (callback_class, _newrelic_trace_class)
        return (callback_class,)

    def initialize_standard_callback_dynamic_params(self, kwargs: dict | None = None) -> StandardCallbackDynamicParams:
        """
        Initialize the standard callback dynamic params from the kwargs

        checks if langfuse_secret_key, gcs_bucket_name in kwargs and sets the corresponding attributes in StandardCallbackDynamicParams
        """

        return _initialize_standard_callback_dynamic_params(kwargs)

    def initialize_standard_built_in_tools_params(self, kwargs: dict | None = None) -> StandardBuiltInToolsParams:
        """
        Initialize the standard built-in tools params from the kwargs

        checks if web_search_options in kwargs or tools and sets the corresponding attribute in StandardBuiltInToolsParams
        """
        return StandardBuiltInToolsParams(
            web_search_options=StandardBuiltInToolCostTracking._get_web_search_options(kwargs or {}),
            file_search=StandardBuiltInToolCostTracking._get_file_search_tool_call(kwargs or {}),
        )

    def get_router_model_id(self) -> str | None:
        """Extract the router deployment model_id from litellm_params.

        Checks both litellm_metadata and metadata for model_info.id.
        Used by cost calculators to look up custom pricing registered
        under the deployment's model_info.id in litellm.model_cost.
        """
        if not hasattr(self, "litellm_params"):
            return None
        for key in ("litellm_metadata", "metadata"):
            meta = self.litellm_params.get(key, {}) or {}
            info = meta.get("model_info", {}) or {}
            model_id = info.get("id")
            if model_id is not None:
                return model_id
        return None

    def get_deployment_model_for_cost(self) -> str | None:
        """The provider-qualified model to price against.

        On a batch retrieve both self.model and litellm_params["model"] can be
        unset, and self.model can otherwise carry the router's model_group alias,
        which no cost map resolves. model_call_details holds the deployment's own
        provider-qualified model, so it is preferred.
        """
        candidates: Final = (
            (self.model_call_details or {}).get("model") if hasattr(self, "model_call_details") else None,
            self.litellm_params.get("model") if hasattr(self, "litellm_params") else None,
            self.model,
        )
        return next((candidate for candidate in candidates if isinstance(candidate, str) and candidate), None)

    def get_router_deployment_model_info(self) -> ModelInfo | None:
        """See deployment_pricing_model_info; None means fall back to the global cost map."""
        return deployment_pricing_model_info(
            model_id=self.get_router_model_id(),
            deployment_model=self.get_deployment_model_for_cost(),
        )

    def update_environment_variables(
        self,
        litellm_params: dict,
        optional_params: dict,
        model: str | None = None,
        user: str | None = None,
        **additional_params,
    ):
        self.optional_params = optional_params
        if model is not None:
            self.model = model
        self.user = user
        self.litellm_params = {
            **self.litellm_params,
            **scrub_sensitive_keys_in_metadata(litellm_params),
        }
        self.litellm_request_debug = litellm_params.get("litellm_request_debug", False)
        self.logger_fn = litellm_params.get("logger_fn", None)
        if _is_debugging_on() or self.litellm_request_debug:
            verbose_logger.debug("self.optional_params: %s", self.optional_params)

        self.model_call_details.update(
            {
                "model": self.model,
                "messages": self.messages,
                "optional_params": self.optional_params,
                "litellm_params": self.litellm_params,
                "start_time": self.start_time,
                "stream": self.stream,
                "user": user,
                "call_type": str(self.call_type),
                "litellm_call_id": self.litellm_call_id,
                "completion_start_time": self.completion_start_time,
                "standard_callback_dynamic_params": self.standard_callback_dynamic_params,
                **self.optional_params,
                **additional_params,
            }
        )

        ## check if stream options is set ##  - used by CustomStreamWrapper for easy instrumentation
        if "stream_options" in additional_params:
            self.stream_options = additional_params["stream_options"]
        ## check if custom pricing set ##
        if any(litellm_params.get(key) is not None for key in _CUSTOM_PRICING_KEYS & litellm_params.keys()):
            self.custom_pricing = True

        if "custom_llm_provider" in self.model_call_details:
            self.custom_llm_provider = self.model_call_details["custom_llm_provider"]

    def update_from_kwargs(
        self,
        kwargs: dict,
        litellm_params: dict | None = None,
        optional_params: dict | None = None,
        model: str | None = None,
        user: str | None = None,
        **additional_params,
    ):
        """
        Convenience wrapper around update_environment_variables that
        automatically extracts metadata/litellm_metadata from kwargs,
        so callers don't need to manually plumb them into litellm_params.
        """
        base_litellm_params: Final[dict[str, Any]] = {}

        if isinstance(kwargs.get("metadata"), dict):
            base_litellm_params["metadata"] = kwargs["metadata"].copy()
        if "litellm_metadata" in kwargs and isinstance(kwargs["litellm_metadata"], dict):
            base_litellm_params["litellm_metadata"] = kwargs["litellm_metadata"]
            if "metadata" not in base_litellm_params:
                base_litellm_params["metadata"] = kwargs["litellm_metadata"].copy()

        if litellm_params:
            # Merge metadata carefully — don't overwrite the merged metadata
            # from kwargs/litellm_metadata with the caller's litellm_params metadata.
            # e.g. anthropic_messages passes Anthropic's native metadata ({user_id: ...})
            # in litellm_params, which would overwrite proxy key-auth fields.
            lp_metadata: Final = litellm_params.pop("metadata", None)
            base_litellm_params.update(litellm_params)
            if lp_metadata and isinstance(lp_metadata, dict):
                base_litellm_params.setdefault("metadata", {})
                for k, v in lp_metadata.items():
                    if k not in base_litellm_params["metadata"]:
                        base_litellm_params["metadata"][k] = v

        self.update_environment_variables(
            litellm_params=base_litellm_params,
            optional_params=optional_params or {},
            model=model,
            user=user,
            **additional_params,
        )

    def update_messages(self, messages: list[AllMessageValues]):
        """
        Update the logged value of the messages in the model_call_details

        Allows pre-call hooks to update the messages before the call is made
        """
        self.messages = messages
        self.model_call_details["messages"] = messages

    def should_run_prompt_management_hooks(
        self,
        non_default_params: dict,
        prompt_id: str | None = None,
        tools: list[dict] | None = None,
    ) -> bool:
        """
        Return True if prompt management hooks should be run
        """
        if prompt_id:
            return True

        # Check if model uses litellm_agent prefix (model replacement without prompt_id)
        model: Final = non_default_params.get("model", "")
        if isinstance(model, str) and model.startswith("litellm_agent/"):
            return True

        if self._should_run_prompt_management_hooks_without_prompt_id(
            non_default_params=non_default_params,
            tools=tools,
        ):
            return True

        return False

    def _should_run_prompt_management_hooks_without_prompt_id(
        self,
        non_default_params: dict,
        tools: list[dict] | None = None,
    ) -> bool:
        """
        Certain prompt management hooks don't need a `prompt_id` to be passed in, they are triggered by dynamic params

        eg. AnthropicCacheControlHook and BedrockKnowledgeBaseHook both don't require a `prompt_id` to be passed in, they are triggered by dynamic params
        """
        for param in DynamicPromptManagementParamLiteral.list_all_params():
            if non_default_params.get(param):
                return True

        #############################################################################
        # Check if Vector Store / Knowledge Base hooks should be applied to the prompt
        #############################################################################
        if litellm.vector_store_registry is not None:
            if litellm.vector_store_registry.get_vector_store_to_run(
                non_default_params=non_default_params, tools=tools
            ):
                return True
        return False

    def get_chat_completion_prompt(
        self,
        model: str,
        messages: list[AllMessageValues],
        non_default_params: dict,
        prompt_variables: dict | None,
        prompt_id: str | None = None,
        prompt_spec: PromptSpec | None = None,
        prompt_management_logger: CustomLogger | None = None,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
        request_kwargs: dict[str, object] | None = None,  # mutable-ok: marker stamped into live request kwargs
    ) -> tuple[str, list[AllMessageValues], dict]:
        from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook

        custom_logger: Final = prompt_management_logger or self.get_custom_logger_for_prompt_management(
            model=model,
            non_default_params=non_default_params,
            prompt_id=prompt_id,
            prompt_spec=prompt_spec,
            dynamic_callback_params=self.standard_callback_dynamic_params,
        )

        if custom_logger:
            breakpoints_before: Final = AnthropicCacheControlHook.count_request_cache_breakpoints(messages)
            (
                model,
                messages,
                non_default_params,
            ) = custom_logger.get_chat_completion_prompt(
                model=model,
                messages=messages,
                non_default_params=non_default_params or {},
                prompt_id=prompt_id,
                prompt_spec=prompt_spec,
                prompt_variables=prompt_variables,
                dynamic_callback_params=self.standard_callback_dynamic_params,
                prompt_label=prompt_label,
                prompt_version=prompt_version,
            )
            if request_kwargs is not None:
                AnthropicCacheControlHook.record_gateway_injection(
                    request_kwargs,
                    AnthropicCacheControlHook.count_request_cache_breakpoints(messages) - breakpoints_before,
                )
        self.messages = messages
        return model, messages, non_default_params

    async def async_get_chat_completion_prompt(
        self,
        model: str,
        messages: list[AllMessageValues],
        non_default_params: dict,
        prompt_variables: dict | None,
        prompt_id: str | None = None,
        prompt_spec: PromptSpec | None = None,
        prompt_management_logger: CustomLogger | None = None,
        tools: list[dict] | None = None,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
        request_kwargs: dict[str, object] | None = None,  # mutable-ok: marker stamped into live request kwargs
    ) -> tuple[str, list[AllMessageValues], dict]:
        from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook

        custom_logger: Final = prompt_management_logger or self.get_custom_logger_for_prompt_management(
            model=model,
            tools=tools,
            non_default_params=non_default_params,
            prompt_id=prompt_id,
            prompt_spec=prompt_spec,
            dynamic_callback_params=self.standard_callback_dynamic_params,
        )

        if custom_logger:
            breakpoints_before: Final = AnthropicCacheControlHook.count_request_cache_breakpoints(messages)
            (
                model,
                messages,
                non_default_params,
            ) = await custom_logger.async_get_chat_completion_prompt(
                model=model,
                messages=messages,
                non_default_params=non_default_params or {},
                prompt_id=prompt_id,
                prompt_spec=prompt_spec,
                prompt_variables=prompt_variables,
                dynamic_callback_params=self.standard_callback_dynamic_params,
                litellm_logging_obj=self,
                tools=tools,
                prompt_label=prompt_label,
                prompt_version=prompt_version,
            )
            if request_kwargs is not None:
                AnthropicCacheControlHook.record_gateway_injection(
                    request_kwargs,
                    AnthropicCacheControlHook.count_request_cache_breakpoints(messages) - breakpoints_before,
                )
        self.messages = messages
        return model, messages, non_default_params

    def _auto_detect_prompt_management_logger(
        self,
        prompt_id: str,
        prompt_spec: PromptSpec | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> CustomLogger | None:
        """
        Auto-detect which prompt management system owns the given prompt_id.

        This allows  a user to just pass prompt_id in the completion call and it will be auto-detected which system owns this prompt.

        Args:
            prompt_id: The prompt ID to check
            dynamic_callback_params: Dynamic callback parameters for should_run_prompt_management checks

        Returns:
            A CustomLogger instance if a matching prompt management system is found, None otherwise
        """
        prompt_management_loggers: Final = litellm.logging_callback_manager.get_custom_loggers_for_type(
            callback_type=CustomPromptManagement
        )

        for logger in prompt_management_loggers:
            if isinstance(logger, CustomPromptManagement):
                try:
                    if logger.should_run_prompt_management(
                        prompt_id=prompt_id,
                        prompt_spec=prompt_spec,
                        dynamic_callback_params=dynamic_callback_params,
                    ):
                        self.model_call_details["prompt_integration"] = logger.__class__.__name__
                        return logger
                except Exception:
                    # If check fails, continue to next logger
                    continue

        return None

    @staticmethod
    def _prompt_manager_runs_without_prompt_id(
        logger: CustomLogger,
        prompt_spec: PromptSpec | None,
        dynamic_callback_params: StandardCallbackDynamicParams | None,
    ) -> bool:
        if not isinstance(logger, CustomPromptManagement):
            return False
        try:
            return logger.should_run_prompt_management(
                prompt_id=None,
                prompt_spec=prompt_spec,
                dynamic_callback_params=dynamic_callback_params or StandardCallbackDynamicParams(),
            )
        except Exception:
            return False

    def get_custom_logger_for_prompt_management(
        self,
        model: str,
        non_default_params: dict,
        tools: list[dict] | None = None,
        prompt_id: str | None = None,
        prompt_spec: PromptSpec | None = None,
        dynamic_callback_params: StandardCallbackDynamicParams | None = None,
    ) -> CustomLogger | None:
        """
        Get a custom logger for prompt management based on model name or available callbacks.

        Args:
            model: The model name to check for prompt management integration
            non_default_params: Non-default parameters passed to the completion call
            tools: Optional tools passed to the completion call
            prompt_id: Optional prompt ID to auto-detect which system owns this prompt
            dynamic_callback_params: Dynamic callback parameters for should_run_prompt_management checks

        Returns:
            A CustomLogger instance if one is found, None otherwise
        """
        # First check if model starts with a known custom logger compatible callback
        # This takes precedence for backward compatibility
        for callback_name in litellm._known_custom_logger_compatible_callbacks:
            if model.startswith(callback_name):
                custom_logger = _init_custom_logger_compatible_class(
                    logging_integration=callback_name,
                    internal_usage_cache=None,
                    llm_router=None,
                )
                if custom_logger is not None:
                    self.model_call_details["prompt_integration"] = model.split("/")[0]
                    return custom_logger

        # If prompt_id is provided, try to auto-detect which system has this prompt
        if prompt_id and dynamic_callback_params is not None:
            auto_detected_logger: Final = self._auto_detect_prompt_management_logger(
                prompt_id=prompt_id,
                prompt_spec=prompt_spec,
                dynamic_callback_params=dynamic_callback_params,
            )
            if auto_detected_logger is not None:
                return auto_detected_logger

        # Then check for any registered CustomPromptManagement loggers (fallback)
        prompt_management_loggers: Final = litellm.logging_callback_manager.get_custom_loggers_for_type(
            callback_type=CustomPromptManagement
        )

        for logger in prompt_management_loggers:
            if prompt_id is None and not self._prompt_manager_runs_without_prompt_id(
                logger=logger,
                prompt_spec=prompt_spec,
                dynamic_callback_params=dynamic_callback_params,
            ):
                continue
            self.model_call_details["prompt_integration"] = logger.__class__.__name__
            return logger

        if (
            anthropic_cache_control_logger
            := AnthropicCacheControlHook.get_custom_logger_for_anthropic_cache_control_hook(non_default_params)
        ):
            self.model_call_details["prompt_integration"] = anthropic_cache_control_logger.__class__.__name__
            return anthropic_cache_control_logger

        #########################################################
        # Vector Store / Knowledge Base hooks
        #########################################################
        if litellm.vector_store_registry is not None:
            vector_store_custom_logger: Final = _init_custom_logger_compatible_class(
                logging_integration="vector_store_pre_call_hook",
                internal_usage_cache=None,
                llm_router=None,
            )
            self.model_call_details["prompt_integration"] = vector_store_custom_logger.__class__.__name__
            # Add to global callbacks so post-call hooks are invoked
            if vector_store_custom_logger and vector_store_custom_logger not in litellm.callbacks:
                litellm.logging_callback_manager.add_litellm_callback(vector_store_custom_logger)
            return vector_store_custom_logger

        return None

    def get_custom_logger_for_anthropic_cache_control_hook(self, non_default_params: dict) -> CustomLogger | None:
        if non_default_params.get("cache_control_injection_points", None):
            custom_logger: Final = _init_custom_logger_compatible_class(
                logging_integration="anthropic_cache_control_hook",
                internal_usage_cache=None,
                llm_router=None,
            )
            return custom_logger
        return None

    def _get_raw_request_body(self, data: dict | str | None) -> dict:
        if data is None:
            return {"error": "Received empty dictionary for raw request body"}
        if isinstance(data, str):
            try:
                return json.loads(data)
            except Exception:
                return {"error": f"Unable to parse raw request body. Got - {data}"}
        return data

    def _get_masked_api_base(self, api_base: str) -> str:
        if "key=" in api_base:
            # Find the position of "key=" in the string
            key_index: Final = api_base.find("key=") + 4
            # Mask the last 5 characters after "key="
            masked_api_base = api_base[:key_index] + "*" * 5 + api_base[-4:]
        else:
            masked_api_base = api_base
        return str(masked_api_base)

    def _pre_call(self, input, api_key, model=None, additional_args={}):
        """
        Common helper function across the sync + async pre-call function
        """

        self.model_call_details["input"] = input
        self.model_call_details["api_key"] = api_key
        self.model_call_details["additional_args"] = additional_args
        self.model_call_details["log_event_type"] = "pre_api_call"
        if model:  # if model name was changes pre-call, overwrite the initial model call name with the new one
            self.model_call_details["model"] = model
        self.model_call_details["litellm_params"]["api_base"] = self._get_masked_api_base(
            additional_args.get("api_base", "")
        )

    def pre_call(self, input, api_key, model=None, additional_args={}):
        # Log the exact input to the LLM API
        try:
            self._pre_call(
                input=input,
                api_key=api_key,
                model=model,
                additional_args=additional_args,
            )

            # User Logging -> if you pass in a custom logging function
            self._print_llm_call_debugging_log(
                api_base=additional_args.get("api_base", ""),
                headers=additional_args.get("headers", {}),
                additional_args=additional_args,
            )
            # log raw request to provider (like LangFuse) -- if opted in.
            if self.log_raw_request_response is True or log_raw_request_response is True:
                _litellm_params: Final = self.model_call_details.get("litellm_params", {})
                _metadata: Final = _litellm_params.get("metadata", {}) or {}
                try:
                    # [Non-blocking Extra Debug Information in metadata]
                    if turn_off_message_logging is True:
                        _metadata["raw_request"] = "redacted by litellm. \
                            'litellm.turn_off_message_logging=True'"
                    else:
                        curl_command: Final = self._get_request_curl_command(
                            api_base=additional_args.get("api_base", ""),
                            headers=additional_args.get("headers", {}),
                            additional_args=additional_args,
                            data=additional_args.get("complete_input_dict", {}),
                        )

                        _metadata["raw_request"] = _redact_string(str(curl_command))
                        # split up, so it's easier to parse in the UI
                        self.model_call_details["raw_request_typed_dict"] = RawRequestTypedDict(
                            raw_request_api_base=self._get_masked_api_base(str(additional_args.get("api_base") or "")),
                            raw_request_body=self._get_raw_request_body(additional_args.get("complete_input_dict", {})),
                            # NOTE: setting ignore_sensitive_headers to True will cause
                            # the Authorization header to be leaked when calls to the health
                            # endpoint are made and fail.
                            raw_request_headers=self._get_masked_headers(
                                additional_args.get("headers", {}) or {},
                            ),
                            error=None,
                        )
                except Exception as e:
                    self.model_call_details["raw_request_typed_dict"] = RawRequestTypedDict(
                        error=str(e),
                    )
                    _metadata["raw_request"] = _redact_string(
                        f"Unable to Log \
                        raw request: {e}"
                    )
            if getattr(self, "logger_fn", None) and callable(self.logger_fn):
                try:
                    self.logger_fn(
                        self.model_call_details
                    )  # Expectation: any logger function passed in by the user should accept a dict object
                except Exception as e:
                    verbose_logger.exception(
                        "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging %s", e
                    )

            self.model_call_details["api_call_start_time"] = datetime.datetime.now()
            # Set-once first provider-handoff instant. api_call_start_time
            # is overwritten on every retry, so it can't measure one-time
            # preprocessing; pinning the first attempt excludes retry loops
            # + backoff. Logging object only — must NOT go into
            # litellm_params["metadata"] (caller request metadata, typed
            # Dict[str, str], echoed downstream; a datetime breaks it).
            if self.model_call_details.get("first_api_call_start_time") is None:
                self.model_call_details["first_api_call_start_time"] = self.model_call_details["api_call_start_time"]
            # Input Integration Logging -> If you want to log the fact that an attempt to call the model was made
            callbacks: Final = litellm.input_callback + (self.dynamic_input_callbacks or [])
            for callback in callbacks:
                try:
                    if callback == "supabase" and supabaseClient is not None:
                        verbose_logger.debug("reaches supabase for logging!")
                        model = self.model_call_details["model"]
                        messages = self.model_call_details["input"]
                        verbose_logger.debug("supabaseClient: %s", supabaseClient)
                        supabaseClient.input_log_event(
                            model=model,
                            messages=messages,
                            end_user=self.model_call_details.get("user", "default"),
                            litellm_call_id=self.litellm_params["litellm_call_id"],
                            print_verbose=print_verbose,
                        )
                    elif callback == "sentry" and add_breadcrumb:
                        try:
                            details_to_log = copy.deepcopy(self.model_call_details)
                        except Exception:
                            details_to_log = self.model_call_details
                        if litellm.turn_off_message_logging:
                            # make a copy of the _model_Call_details and log it
                            details_to_log.pop("messages", None)
                            details_to_log.pop("input", None)
                            details_to_log.pop("prompt", None)

                        add_breadcrumb(
                            category="litellm.llm_call",
                            message=f"Model Call Details pre-call: {details_to_log}",
                            level="info",
                        )

                    elif isinstance(callback, CustomLogger):  # custom logger class
                        callback.log_pre_api_call(
                            model=self.model,
                            messages=self.messages,
                            kwargs=self.model_call_details,
                        )
                    elif callable(callback) and customLogger is not None:  # custom logger functions
                        customLogger.log_input_event(
                            model=self.model,
                            messages=self.messages,
                            kwargs=self.model_call_details,
                            print_verbose=print_verbose,
                            callback_func=callback,
                        )
                except Exception as e:
                    verbose_logger.exception("litellm.Logging.pre_call(): Exception occured - %s", e)
                    verbose_logger.debug(
                        "LiteLLM.Logging: is sentry capture exception initialized %s", capture_exception
                    )
                    if capture_exception:  # log this error to sentry for debugging
                        capture_exception(e)
        except Exception as e:
            verbose_logger.exception("LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging %s", e)
            verbose_logger.error("LiteLLM.Logging: is sentry capture exception initialized %s", capture_exception)
            if capture_exception:  # log this error to sentry for debugging
                capture_exception(e)

    def _print_llm_call_debugging_log(
        self,
        api_base: str,
        headers: dict,
        additional_args: dict,
    ):
        """
        Internal debugging helper function

        Prints the RAW curl command sent from LiteLLM
        """
        if _is_debugging_on() or self.litellm_request_debug:
            if json_logs:
                masked_headers: Final = self._get_masked_headers(headers)
                masked_api_base: Final = self._get_masked_api_base(str(api_base or ""))
                if self.litellm_request_debug:
                    verbose_logger.warning(  # .warning ensures this shows up in all environments
                        "POST Request Sent from LiteLLM",
                        extra={"api_base": {masked_api_base}, **masked_headers},
                    )
                else:
                    verbose_logger.debug(
                        "POST Request Sent from LiteLLM",
                        extra={"api_base": {masked_api_base}, **masked_headers},
                    )
            else:
                headers = additional_args.get("headers", {})
                if headers is None:
                    headers = {}
                data: Final = additional_args.get("complete_input_dict", {})
                api_base = str(additional_args.get("api_base", ""))
                curl_command: Final = self._get_request_curl_command(
                    api_base=api_base,
                    headers=headers,
                    additional_args=additional_args,
                    data=data,
                )
                if self.litellm_request_debug:
                    verbose_logger.warning(
                        "\x1b[92m%s\x1b[0m\n", curl_command
                    )  # .warning ensures this shows up in all environments
                else:
                    verbose_logger.debug("\x1b[92m%s\x1b[0m\n", curl_command)

    def _get_request_body(self, data: dict) -> str:
        return str(data)

    def _get_request_curl_command(self, api_base: str, headers: dict | None, additional_args: dict, data: dict) -> str:
        masked_api_base: Final = self._get_masked_api_base(api_base)
        if headers is None:
            headers = {}
        curl_command = "\n\nPOST Request Sent from LiteLLM:\n"
        curl_command += "curl -X POST \\\n"
        curl_command += f"{masked_api_base} \\\n"
        masked_headers: Final = self._get_masked_headers(headers)
        formatted_headers: Final = " ".join([f"-H '{k}: {v}'" for k, v in masked_headers.items()])
        curl_command += f"{formatted_headers} \\\n" if formatted_headers.strip() != "" else ""
        curl_command += f"-d '{self._get_request_body(data)}'\n"
        if additional_args.get("request_str", None) is not None:
            # print the sagemaker / bedrock client request
            curl_command = "\nRequest Sent from LiteLLM:\n"
            request_str: Final = additional_args.get("request_str", "")
            curl_command += request_str
        return curl_command

    def _get_masked_headers(self, headers: dict, ignore_sensitive_headers: bool = False) -> dict:
        """
        Internal debugging helper function

        Masks the headers of the request sent from LiteLLM
        """
        return _get_masked_values(headers, ignore_sensitive_values=ignore_sensitive_headers)

    def post_call(self, original_response, input=None, api_key=None, additional_args={}):
        # Log the exact result from the LLM API, for streaming - log the type of response received
        if isinstance(original_response, dict):
            original_response = json.dumps(original_response, default=str)
        try:
            self.model_call_details["input"] = input
            self.model_call_details["api_key"] = api_key
            self.model_call_details["original_response"] = original_response
            self.model_call_details["additional_args"] = additional_args
            self.model_call_details["log_event_type"] = "post_api_call"

            attr: Literal["warning", "debug"]
            if self.litellm_request_debug:
                attr = "warning"
            else:
                attr = "debug"

            if json_logs:
                callattr = getattr(verbose_logger, attr)
                callattr(
                    "RAW RESPONSE:\n{}\n\n".format(
                        self.model_call_details.get("original_response", self.model_call_details)
                    ),
                )
            else:
                callattr = getattr(verbose_logger, attr)
                callattr(
                    "RAW RESPONSE:\n{}\n\n".format(
                        self.model_call_details.get("original_response", self.model_call_details)
                    )
                )
            if getattr(self, "logger_fn", None) and callable(self.logger_fn):
                try:
                    self.logger_fn(
                        self.model_call_details
                    )  # Expectation: any logger function passed in by the user should accept a dict object
                except Exception as e:
                    verbose_logger.exception(
                        "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging %s", e
                    )
            original_response = redact_message_input_output_from_logging(
                model_call_details=(self.model_call_details if hasattr(self, "model_call_details") else {}),
                result=original_response,
            )
            # Input Integration Logging -> If you want to log the fact that an attempt to call the model was made

            callbacks: Final = litellm.input_callback + (self.dynamic_input_callbacks or [])
            for callback in callbacks:
                try:
                    if callback == "sentry" and add_breadcrumb:
                        verbose_logger.debug("reaches sentry breadcrumbing")
                        try:
                            details_to_log = copy.deepcopy(self.model_call_details)
                        except Exception:
                            details_to_log = self.model_call_details
                        if litellm.turn_off_message_logging:
                            # make a copy of the _model_Call_details and log it
                            details_to_log.pop("messages", None)
                            details_to_log.pop("input", None)
                            details_to_log.pop("prompt", None)

                        add_breadcrumb(
                            category="litellm.llm_call",
                            message=f"Model Call Details post-call: {details_to_log}",
                            level="info",
                        )
                    elif isinstance(callback, CustomLogger):  # custom logger class
                        callback.log_post_api_call(
                            kwargs=self.model_call_details,
                            response_obj=None,
                            start_time=self.start_time,
                            end_time=None,
                        )
                except Exception as e:
                    verbose_logger.exception(
                        "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while post-call logging with integrations %s",
                        e,
                    )
                    verbose_logger.debug(
                        "LiteLLM.Logging: is sentry capture exception initialized %s", capture_exception
                    )
                    if capture_exception:  # log this error to sentry for debugging
                        capture_exception(e)
        except Exception as e:
            verbose_logger.exception("LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging %s", e)

    async def async_post_mcp_tool_call_hook(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ):
        """
        Post MCP Tool Call Hook

        Use this to modify the MCP tool call response before it is returned to the user.
        """
        from litellm.types.llms.base import HiddenParams
        from litellm.types.mcp import MCPPostCallResponseObject

        callbacks: Final = self.get_combined_callback_list(
            dynamic_success_callbacks=self.dynamic_success_callbacks,
            global_callbacks=litellm.success_callback,
        )
        post_mcp_tool_call_response_obj: Final[MCPPostCallResponseObject] = MCPPostCallResponseObject(
            mcp_tool_call_response=response_obj, hidden_params=HiddenParams()
        )
        for callback in callbacks:
            try:
                if isinstance(callback, CustomLogger):
                    response: MCPPostCallResponseObject | None = await callback.async_post_mcp_tool_call_hook(
                        kwargs=kwargs,
                        response_obj=post_mcp_tool_call_response_obj,
                        start_time=start_time,
                        end_time=end_time,
                    )
                    ######################################################################
                    # if any of the callbacks modify the response, use the modified response
                    # current implementation returns the first modified response
                    ######################################################################
                    if response is not None:
                        response_obj = self._parse_post_mcp_call_hook_response(response=response)
            except Exception as e:
                verbose_logger.exception("LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging %s", e)
        return response_obj

    def _parse_post_mcp_call_hook_response(
        self, response: MCPPostCallResponseObject | None
    ) -> "Sequence[TextContent | ImageContent | EmbeddedResource] | None":
        """
        Parse the response from the post_mcp_tool_call_hook

        1. Unpack the mcp_tool_call_response
        2. save the updated response_cost to the model_call_details
        """
        if response is None:
            return None
        self.model_call_details["response_cost"] = response.hidden_params.response_cost
        return response.mcp_tool_call_response

    def get_response_ms(self) -> float:
        return (
            self.model_call_details.get("end_time", datetime.datetime.now())
            - self.model_call_details.get("start_time", datetime.datetime.now())
        ).total_seconds() * 1000

    def set_cost_breakdown(
        self,
        input_cost: float,
        output_cost: float,
        total_cost: float,
        cost_for_built_in_tools_cost_usd_dollar: float,
        additional_costs: dict | None = None,
        original_cost: float | None = None,
        discount_percent: float | None = None,
        discount_amount: float | None = None,
        margin_percent: float | None = None,
        margin_fixed_amount: float | None = None,
        margin_total_amount: float | None = None,
        cache_read_cost: float | None = None,
        cache_creation_cost: float | None = None,
        reasoning_cost: float | None = None,
        service_tier: str | None = None,
        data_residency: str | None = None,
        vertex_location: str | None = None,
    ) -> None:
        """
        Helper method to store cost breakdown in the logging object.

        Args:
            input_cost: Cost of input/prompt tokens
            output_cost: Cost of output/completion tokens
            cost_for_built_in_tools_cost_usd_dollar: Cost of built-in tools
            total_cost: Total cost of request
            additional_costs: Free-form additional costs dict (e.g., {"azure_model_router_flat_cost": 0.00014})
            original_cost: Cost before discount
            discount_percent: Discount percentage (0.05 = 5%)
            discount_amount: Discount amount in USD
            margin_percent: Margin percentage applied (0.10 = 10%)
            margin_fixed_amount: Fixed margin amount in USD
            margin_total_amount: Total margin added in USD
            service_tier: Tier the costs above were priced on, already resolved
            data_residency: Region uplift the costs above were priced on, already resolved
            vertex_location: Vertex AI location the costs above were priced on, already resolved
        """

        self.cost_breakdown = CostBreakdown(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            tool_usage_cost=cost_for_built_in_tools_cost_usd_dollar,
            service_tier=service_tier,
            data_residency=data_residency,
            vertex_location=vertex_location,
        )
        if cache_read_cost is not None and cache_read_cost > 0:
            self.cost_breakdown["cache_read_cost"] = cache_read_cost
        if cache_creation_cost is not None and cache_creation_cost > 0:
            self.cost_breakdown["cache_creation_cost"] = cache_creation_cost
        if reasoning_cost is not None and reasoning_cost > 0:
            self.cost_breakdown["reasoning_cost"] = reasoning_cost

        # Store additional costs if provided (free-form dict for extensibility)
        if additional_costs and isinstance(additional_costs, dict) and len(additional_costs) > 0:
            self.cost_breakdown["additional_costs"] = additional_costs

        # Store discount information if provided
        if original_cost is not None:
            self.cost_breakdown["original_cost"] = original_cost
        if discount_percent is not None:
            self.cost_breakdown["discount_percent"] = discount_percent
        if discount_amount is not None:
            self.cost_breakdown["discount_amount"] = discount_amount

        # Store margin information if provided
        if margin_percent is not None:
            self.cost_breakdown["margin_percent"] = margin_percent
        if margin_fixed_amount is not None:
            self.cost_breakdown["margin_fixed_amount"] = margin_fixed_amount
        if margin_total_amount is not None:
            self.cost_breakdown["margin_total_amount"] = margin_total_amount

    def _response_cost_calculator(
        self,
        result: Union[
            ModelResponse,
            ModelResponseStream,
            EmbeddingResponse,
            ImageResponse,
            TranscriptionResponse,
            TextCompletionResponse,
            HttpxBinaryResponseContent,
            RerankResponse,
            Batch,
            FineTuningJob,
            ResponsesAPIResponse,
            ResponseCompletedEvent,
            OpenAIFileObject,
            LiteLLMRealtimeStreamLoggingObject,
            OpenAIModerationResponse,
            "SearchResponse",
            dict,
            list,
        ],
        cache_hit: bool | None = None,
        litellm_model_name: str | None = None,
        router_model_id: str | None = None,
    ) -> float | None:
        """
        Calculate response cost using result + logging object variables.

        used for consistent cost calculation across response headers + logging integrations.
        """

        if cache_hit is None:
            cache_hit = self.model_call_details.get("cache_hit", False)

        if cache_hit is True:
            return 0.0

        if is_unbilled_non_inference_call(
            self.call_type, StandardLoggingPayloadSetup.merge_litellm_metadata(self.litellm_params), result
