"""A provider's "the account behind this key has no credit left" maps to CreditBalanceExhaustedError, a 429
the router retries on another deployment; any other 400 stays the caller's BadRequestError."""

import json
from typing import Final

import openai
import pytest

import litellm
from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.common_utils import OpenAIError

ANTHROPIC_CREDIT_TEXT: Final = (
    "Your credit balance is too low to access the Anthropic API. "
    "Please go to Plans & Billing to upgrade or purchase credits."
)


def _anthropic_body(error_type: str, message: str) -> str:
    return json.dumps({"type": "error", "error": {"type": error_type, "message": message}})


def _openai_body(error_type: str, code: str) -> str:
    return json.dumps({"error": {"message": "quota reached", "type": error_type, "param": None, "code": code}})


@pytest.fixture(autouse=True)
def quiet_exception_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "suppress_debug_info", True)


def _mapped(original: BaseLLMException, provider: str) -> openai.APIError:
    with pytest.raises(openai.APIError) as raised:
        exception_type(model="test-model", original_exception=original, custom_llm_provider=provider)
    return raised.value


@pytest.mark.parametrize(
    ("original", "provider"),
    [
        (
            AnthropicError(status_code=400, message=_anthropic_body("invalid_request_error", ANTHROPIC_CREDIT_TEXT)),
            "anthropic",
        ),
        (AnthropicError(status_code=402, message=_anthropic_body("billing_error", "Billing issue")), "anthropic"),
        (
            OpenAIError(status_code=429, message=_openai_body("insufficient_quota", "credit_balance_exhausted")),
            "openai",
        ),
        (OpenAIError(status_code=429, message=_openai_body("insufficient_quota", "insufficient_quota")), "openai"),
    ],
    ids=["anthropic_400_credit_text", "anthropic_402_billing_error", "openai_429_credit_code", "openai_429_quota"],
)
def test_an_exhausted_credit_balance_is_a_rate_limit_the_router_retries_elsewhere(
    original: BaseLLMException, provider: str
) -> None:
    mapped: Final = _mapped(original, provider)

    assert (type(mapped), mapped.status_code, mapped.llm_provider) == (
        litellm.CreditBalanceExhaustedError,
        429,
        provider,
    ), str(mapped)
    assert isinstance(mapped, litellm.RateLimitError)


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        (
            AnthropicError(
                status_code=400, message=_anthropic_body("invalid_request_error", "max_tokens: Field required")
            ),
            litellm.BadRequestError,
        ),
        (
            AnthropicError(status_code=402, message=_anthropic_body("invalid_request_error", "Payment required")),
            litellm.PaymentRequiredError,
        ),
        (
            AnthropicError(status_code=500, message=_anthropic_body("api_error", ANTHROPIC_CREDIT_TEXT)),
            litellm.InternalServerError,
        ),
    ],
    ids=["generic_400", "402_without_a_billing_signal", "credit_text_on_a_server_error"],
)
def test_a_failure_without_the_credit_signal_keeps_its_own_class(
    original: BaseLLMException, expected: type[Exception]
) -> None:
    mapped: Final = _mapped(original, "anthropic")

    assert type(mapped) is expected, str(mapped)
