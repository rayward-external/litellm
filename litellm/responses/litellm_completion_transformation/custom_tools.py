"""
Utilities for handling OpenAI Responses API 'custom' tools (freeform/grammar tools)
when bridging to Chat Completions providers.

Custom tools are defined with ``type: "custom"`` and a grammar/format specification.
Since most Chat Completions providers only support standard ``function`` tools,
the bridge converts them to ``function`` tools with a single ``content`` string
parameter. When the model responds with a ``function_call`` for such a tool, this
module converts it back to the ``custom_tool_call`` format expected by clients like
Codex CLI.

The forward direction (custom -> function) and reverse direction (function_call ->
custom_tool_call) are both handled here so future custom tool types can be added by
extending this module without touching the streaming iterator or transformation
logic.
"""

import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm.types.llms.openai import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)
from litellm.types.responses.main import (
    OutputWebSearchCall,
    OutputWebSearchCallAction,
    OutputWebSearchCallOpenPageAction,
    OutputWebSearchCallSearchAction,
    WebSearchCallStatus,
)

_MAX_ARGUMENTS_LEN: Final = 1_000_000

TOOL_CALL_ITEM_ID_PREFIX_BY_TYPE: Final = MappingProxyType(
    {"function_call": "fc", "custom_tool_call": "ctc", "web_search_call": "ws"}
)

# Anthropic stamps every tool call its OWN fleet executed with this prefix
# (``server_tool_use`` blocks), as opposed to the ``toolu_`` calls it hands to
# the client. The distinction is invisible in the Chat Completions shape both
# collapse into, so the id prefix is the only signal the Responses bridge has.
SERVER_EXECUTED_TOOL_CALL_ID_PREFIX: Final = "srvtoolu_"

# Server-executed tool names that map onto OpenAI's ``web_search_call`` output
# item. ``web_fetch`` belongs here because OpenAI has no fetch-shaped item: a
# page fetch is a ``web_search_call`` with an ``open_page`` action. Code
# execution has its own item type and is converted elsewhere (see
# ``_extract_tool_result_output_items``), so it is deliberately absent here.
SERVER_EXECUTED_WEB_SEARCH_TOOL_NAMES: Final = frozenset({"web_search", "web_fetch"})

# The one whose Anthropic input is ``{"url": ...}`` rather than ``{"query": ...}``.
SERVER_EXECUTED_WEB_FETCH_TOOL_NAME: Final = "web_fetch"


def openai_shaped_tool_call_item_id(item_type: str, tool_id: str) -> str:
    prefix: Final = TOOL_CALL_ITEM_ID_PREFIX_BY_TYPE.get(item_type)
    if prefix is None or not tool_id or tool_id.startswith(prefix):
        return tool_id
    return f"{prefix}_{tool_id}"


def extract_custom_tool_names(tools: Sequence[object] | None) -> set[str]:
    """Extract names of tools originally defined as ``type: "custom"``."""
    if not tools:
        return set()
    names: Final[set[str]] = set()
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") == "custom" and "name" in tool:
            names.add(tool["name"])
    return names


def is_custom_tool_call(tool_name: str, custom_tool_names: set[str]) -> bool:
    """Check if a tool call name corresponds to a custom tool."""
    return tool_name in custom_tool_names


def serialize_tool_call_arguments(raw_arguments: object, default: str = "") -> str:
    """Render tool call arguments as the JSON string tool-call schemas require.

    Arguments normally arrive already JSON-encoded, but clients and providers
    also send the decoded object. ``str()`` on a dict yields a Python repr with
    single quotes, which every downstream JSON parser rejects with errors like
    "Expecting ',' delimiter".
    """
    if isinstance(raw_arguments, str):
        return raw_arguments or default
    if raw_arguments is None:
        return default
    return json.dumps(raw_arguments, default=str)


def unwrap_custom_tool_arguments(arguments: str) -> str:
    """Extract the raw content string from JSON-wrapped arguments.

    The bridge converts custom tools to function tools with schema
    ``{"properties": {"content": {"type": "string"}}}``, so the model returns
    arguments like ``{"content": "*** Begin Patch\\n..."}``. This function
    extracts just the content string. If the arguments are not valid JSON or do
    not contain a ``content`` key, the original string is returned unchanged.
    """
    if not arguments:
        return ""
    if len(arguments) > _MAX_ARGUMENTS_LEN:
        return arguments
    try:
        parsed: Final = json.loads(arguments)
        if isinstance(parsed, dict) and "content" in parsed:
            return str(parsed["content"])
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return arguments


def build_tool_call_item_kwargs(
    call_id: str,
    name: str,
    arguments_or_input: str,
    status: str,
    custom_tool_names: set[str],
) -> dict[str, str]:
    """Build kwargs for an output item dict that is either a ``function_call``
    or a ``custom_tool_call`` depending on whether *name* is in
    *custom_tool_names*.

    For custom tools the ``arguments`` JSON is unwrapped into the ``input``
    field. For regular function tools the raw ``arguments`` string is kept.

    This centralises the branching logic so the streaming iterator and the
    non-streaming transformation share a single code path.
    """
    custom: Final = is_custom_tool_call(name, custom_tool_names)
    item_type: Final = "custom_tool_call" if custom else "function_call"
    kwargs: Final[dict[str, str]] = {
        "type": item_type,
        "id": openai_shaped_tool_call_item_id(item_type, call_id),
        "call_id": call_id,
        "name": name,
        "status": status,
    }
    if custom:
        if status == "completed":
            kwargs["input"] = unwrap_custom_tool_arguments(arguments_or_input)
        else:
            kwargs["input"] = ""
    else:
        kwargs["arguments"] = arguments_or_input
    return kwargs


def is_server_executed_web_search_call(call_id: str, tool_name: str) -> bool:
    """Was this tool call a web search the provider already ran itself?

    Such a call is not actionable by the client: the provider ran the search and
    returned the results in the same response. Surfacing it as a ``function_call``
    makes the client answer it (codex answers ``unsupported call: web_search``),
    and on the next turn that answer replays as a ``tool_result`` whose paired
    ``tool_use`` does not exist, which Anthropic rejects with
    "unexpected `tool_use_id` found in `tool_result` blocks".
    """
    return (
        call_id.startswith(SERVER_EXECUTED_TOOL_CALL_ID_PREFIX) and tool_name in SERVER_EXECUTED_WEB_SEARCH_TOOL_NAMES
    )


def extract_string_tool_argument(arguments: str, key: str) -> str | None:
    """Read one non-empty string field out of a tool call's JSON arguments."""
    if not arguments:
        return None
    try:
        parsed: Final = json.loads(arguments)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if isinstance(parsed, dict):
        value: Final = parsed.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def build_web_search_call_action(tool_name: str, arguments: str) -> OutputWebSearchCallAction | None:
    """Describe what a provider-executed web tool actually did.

    The two Anthropic server tools read from different keys -- ``web_search``
    takes ``{"query": ...}`` and ``web_fetch`` takes ``{"url": ...}`` (both
    measured against api.anthropic.com on 2026-09-02) -- and OpenAI keeps them
    distinct too: a fetch is a ``web_search_call`` whose action is
    ``open_page`` (``openai.types.responses.response_function_web_search``).
    Reading a fetch with the search key would drop the URL and report a query
    the model never ran, so the branch is on the tool name, not the arguments.

    Returns None only when the arguments carry neither key, leaving the item
    without an action rather than describing work that did not happen.
    """
    if tool_name == SERVER_EXECUTED_WEB_FETCH_TOOL_NAME:
        url: Final = extract_string_tool_argument(arguments, "url")
        return OutputWebSearchCallOpenPageAction(type="open_page", url=url) if url is not None else None
    query: Final = extract_string_tool_argument(arguments, "query")
    return OutputWebSearchCallSearchAction(type="search", query=query) if query is not None else None


def build_web_search_call_item(
    call_id: str,
    tool_name: str,
    arguments: str,
    status: WebSearchCallStatus,
) -> OutputWebSearchCall:
    """Build the ``web_search_call`` output item for a provider-executed call.

    Shared by the streaming iterator and the non-streaming transformation so
    both surfaces describe the same call the same way.
    """
    return OutputWebSearchCall(
        type="web_search_call",
        id=openai_shaped_tool_call_item_id("web_search_call", call_id),
        status=status,
        action=build_web_search_call_action(tool_name, arguments),
    )


class _CustomToolFormat(BaseModel):
    syntax: str = ""
    definition: str = ""


_ALLOWED_CALLERS_ADAPTER: Final = TypeAdapter(list[str] | None)


def validated_allowed_callers(value: object) -> list[str] | None:
    try:
        return _ALLOWED_CALLERS_ADAPTER.validate_python(value, strict=True)
    except ValidationError as exc:
        raise ValueError("allowed_callers must be a list of strings") from exc


def _grammar_suffix(fmt: object) -> str:
    try:
        parsed: Final = _CustomToolFormat.model_validate(fmt)
    except ValidationError:
        return ""
    if not parsed.definition:
        return ""
    return f"\n\nFormat:\n```{parsed.syntax}\n{parsed.definition}\n```"


def convert_custom_tool_to_function_tool(tool: Mapping[str, object]) -> ChatCompletionToolParam | None:
    """Convert a Responses API ``custom`` tool to a Chat Completions ``function``
    tool.

    The grammar definition is embedded in the description so the model can
    produce correctly-formatted output. Returns ``None`` if the tool is not a
    custom tool. Raises ``ValueError`` if ``allowed_callers`` is not a list of
    strings.
    """
    if tool.get("type") != "custom":
        return None
    raw_name: Final = tool.get("name")
    name: Final = raw_name if isinstance(raw_name, str) else ""
    raw_description: Final = tool.get("description")
    description = (raw_description if isinstance(raw_description, str) else "") + _grammar_suffix(tool.get("format"))
    allowed_callers: Final = validated_allowed_callers(tool.get("allowed_callers"))
    function_chunk: Final = ChatCompletionToolParamFunctionChunk(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": f"The {name} content following the specified format",
                }
            },
            "required": ["content"],
        },
    )
    if allowed_callers is None:
        return ChatCompletionToolParam(type="function", function=function_chunk)
    return ChatCompletionToolParam(type="function", function=function_chunk, allowed_callers=allowed_callers)
