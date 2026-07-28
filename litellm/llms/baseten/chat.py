from typing import Optional
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class BasetenConfig(OpenAIGPTConfig):
    """
    Reference: https://inference.baseten.co/v1

    Below are the parameters:
    """

    max_tokens: Optional[int] = None
    response_format: Optional[dict] = None
    seed: Optional[int] = None
    stream: Optional[bool] = None
    top_p: Optional[int] = None
    tool_choice: Optional[str] = None
    tools: Optional[list] = None
    user: Optional[str] = None
    presence_penalty: Optional[int] = None
    frequency_penalty: Optional[int] = None
    stream_options: Optional[dict] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
        seed: Optional[int] = None,
        stop: Optional[list] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[int] = None,
        tool_choice: Optional[str] = None,
        tools: Optional[list] = None,
        user: Optional[str] = None,
        presence_penalty: Optional[int] = None,
        frequency_penalty: Optional[int] = None,
        stream_options: Optional[dict] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the supported OpenAI params for the given model.

        ``reasoning_effort`` is declared unconditionally, like every other
        parameter here: Baseten hosts arbitrary, user-supplied deployments
        (there is no enumerable Baseten catalog in litellm's cost map the way
        there is for a curated provider), so litellm cannot know in advance
        which deployed model supports it -- the deployment itself is the only
        real source of truth on capability, exactly as already true for
        ``temperature``, ``tools``, etc. above. map_openai_params below is a
        pure allow-list passthrough with no value translation, so this is the
        whole fix -- previously reasoning_effort was silently stripped by
        drop_params before ever reaching a deployment that may well accept it.

        Deliberately a passthrough, NOT a translation into Baseten's native
        ``thinking`` switch. An earlier version of this file resolved
        reasoning_effort into ``thinking={"type": "enabled"|"disabled"}`` and
        dropped the caller's value; that was removed on purpose. Measured
        directly against ``https://inference.baseten.co/v1/chat/completions``
        with a real key on 2026-07-28, the deployments served there have real,
        per-model reasoning_effort vocabularies of their own:

        * ``moonshotai/Kimi-K3`` accepts none/minimal/low/high/max (a genuine
          graded ladder) and 400s an unknown value.
        * ``zai-org/GLM-5.2`` accepts none/high/max and 400s minimal/low.

        Collapsing that to a binary on/off destroyed Kimi-K3's ladder, so the
        caller's literal value goes to the provider and the provider's own
        vocabulary governs. GLM callers therefore get a real upstream 400 on
        ``minimal``/``low``. That is the intended, accepted behaviour -- it is
        the provider's answer, faithfully delivered. Do NOT add a fallback,
        remap, or special case to paper over it, and do not reintroduce the
        ``thinking`` translation.
        """
        return [
            "max_tokens",
            "max_completion_tokens",
            "response_format",
            "seed",
            "stop",
            "stream",
            "temperature",
            "top_p",
            "tool_choice",
            "tools",
            "user",
            "presence_penalty",
            "frequency_penalty",
            "stream_options",
            "reasoning_effort",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                optional_params[param] = value
        return optional_params

    def _get_openai_compatible_provider_info(self, api_base: str, api_key: str) -> tuple:
        """
        Get the OpenAI compatible provider info for Baseten
        """
        # Default to Model API
        default_api_base = "https://inference.baseten.co/v1"
        default_api_key = api_key or "BASETEN_API_KEY"

        return default_api_base, default_api_key

    @staticmethod
    def is_dedicated_deployment(model: str) -> bool:
        """
        Check if the model is a dedicated deployment (8-digit alphanumeric code)
        """
        # Remove 'baseten/' prefix if present
        model_id = model.replace("baseten/", "")

        # Check if it's an 8-digit alphanumeric code
        import re

        return bool(re.match(r"^[a-zA-Z0-9]{8}$", model_id))

    @staticmethod
    def get_api_base_for_model(model: str) -> str:
        """
        Get the appropriate API base URL for the given model
        """
        if BasetenConfig.is_dedicated_deployment(model):
            # Extract the model ID (remove 'baseten/' prefix if present)
            model_id = model.replace("baseten/", "")
            return f"https://model-{model_id}.api.baseten.co/environments/production/sync/v1"
        else:
            # Use Model API
            return "https://inference.baseten.co/v1"
