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
