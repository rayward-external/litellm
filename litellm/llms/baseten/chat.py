from collections.abc import Mapping
from typing import Optional

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

# Value emitted for Baseten's native `thinking` switch when reasoning is on / off.
_THINKING_ENABLED = "enabled"
_THINKING_DISABLED = "disabled"

# litellm's standard OpenAI-style sentinel for "reasoning off"
# (litellm.types.llms.openai.REASONING_EFFORT). Baseten deployments have no
# corresponding effort level, so it becomes the disabled switch below.
_REASONING_EFFORT_OFF = "none"


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

        ``reasoning_effort`` and ``thinking`` are declared unconditionally, like
        every other parameter here: Baseten hosts arbitrary, user-supplied
        deployments (there is no enumerable Baseten catalog in litellm's cost map
        the way there is for a curated provider), so litellm cannot know in
        advance which deployed model supports them -- the deployment itself is
        the only real source of truth on capability, exactly as already true for
        ``temperature``, ``tools``, etc. above. Declaring them is what stops
        litellm's own drop_params logic stripping the caller's reasoning intent
        before map_openai_params ever sees it; map_openai_params then resolves
        that intent into the switch the deployment actually understands.
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
            "thinking",
        ]

    @staticmethod
    def _thinking_switch(reasoning_effort: object) -> Mapping[str, str]:
        """
        Translate an OpenAI-style ``reasoning_effort`` into Baseten's native
        ``thinking`` switch.

        Why this is a translation and not a passthrough -- measured directly
        against ``https://inference.baseten.co/v1/chat/completions`` on the
        reasoning-capable deployments served there:

        * ``thinking={"type": "enabled"}`` reliably produces reasoning tokens and
          ``thinking={"type": "disabled"}`` reliably suppresses them. It is the
          one knob that behaves consistently across those deployments, i.e. the
          provider's native control.
        * ``reasoning_effort`` is *not* a usable substitute. Some deployments
          ignore it outright (zero reasoning tokens for every value, no error),
          and others accept only a narrow subset of values and reject the rest
          with an upstream HTTP 400 of the form ``reasoning_effort must be one
          of ...``. Forwarding the OpenAI-style string verbatim therefore either
          silently does nothing or breaks the request -- which is exactly why
          ``map_openai_params`` never copies the raw key into the outbound body.
        * The underlying control is **binary**: reasoning on or reasoning off.
          There is no graded budget to express, so there is deliberately no
          effort-to-budget_tokens ladder here of the kind Anthropic-style
          providers need. Do not "simplify" this back into a passthrough, and do
          not invent intermediate levels.

        Precedent for the shape: ``DeepSeekChatConfig.map_openai_params`` does the
        same ``reasoning_effort`` -> ``{"type": enabled|disabled}`` translation for
        the same reason (an OpenAI-compatible endpoint whose reasoning switch is
        binary rather than graded).
        """
        switch = _THINKING_DISABLED if reasoning_effort == _REASONING_EFFORT_OFF else _THINKING_ENABLED
        return {"type": switch}  # mutable-ok: goes straight into the JSON request body as a plain object

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model=model)
        effort_switch: Mapping[str, str] | None = None
        caller_thinking: Mapping[str, str] | None = None
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param == "thinking":
                if isinstance(value, Mapping):
                    # The caller named the provider's native switch explicitly, so
                    # they were specific about what they want -- honour it verbatim
                    # and let it win over anything reasoning_effort would imply.
                    caller_thinking = value
            elif param == "reasoning_effort":
                # Deliberately never copied through as-is: see _thinking_switch for
                # why sending the raw OpenAI-style value to Baseten is harmful.
                if value is not None:
                    effort_switch = self._thinking_switch(value)
            elif param in supported_openai_params:
                optional_params[param] = value

        # `thinking` MUST go in extra_body, never at the top level of
        # optional_params. Baseten is dispatched through the OpenAI *SDK*
        # (main.py's `custom_llm_provider == "baseten"` branch reaches
        # _complete_custom_openai -> openai_chat_completions.completion), and
        # OpenAIGPTConfig.transform_request splats optional_params straight into
        # the kwargs of `client.chat.completions.create(**data)`. That method has
        # an explicit signature with no `thinking` parameter and no **kwargs, so a
        # top-level key raises TypeError BEFORE any HTTP call and litellm re-wraps
        # it as a 500 -- turning "reasoning silently ignored" into a hard outage on
        # exactly the requests this translation exists to fix. extra_body is the
        # SDK's supported escape hatch for provider-specific body fields, and
        # add_provider_specific_params_to_optional_params preserves what is already
        # there. Do NOT "simplify" this back to optional_params["thinking"].
        #
        # This is also why DeepSeekChatConfig can write the key top-level and we
        # cannot: deepseek dispatches via base_llm_http_handler, which POSTs the
        # dict as raw JSON and tolerates any body key. Baseten does not.
        thinking = caller_thinking if caller_thinking is not None else effort_switch
        if thinking is not None:
            # Seeds the caller's own extra_body when absent, then mutates it in
            # place below; merged downstream by
            # add_provider_specific_params_to_optional_params.
            extra_body = optional_params.setdefault("extra_body", {})  # mutable-ok: seeded then filled in place
            extra_body["thinking"] = thinking
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
