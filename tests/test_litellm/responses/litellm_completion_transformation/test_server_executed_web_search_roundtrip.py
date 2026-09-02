"""
Round-trip tests for Anthropic server-executed tool calls (``srvtoolu_`` ids)
across the Responses API bridge.

Anthropic runs ``web_search`` on its own fleet: the response carries a
``server_tool_use`` block plus the ``web_search_tool_result`` it produced, and
the model answers in the same turn (``stop_reason: end_turn``, measured against
api.anthropic.com on 2026-09-02). Nothing there is a call the client can run.

Surfacing that block as a Responses ``function_call`` made codex answer it
("unsupported call: web_search"), and on the next turn the answer replayed as a
``tool_result`` whose ``tool_use`` did not exist, which Anthropic rejects with:

    messages.60.content.0: unexpected `tool_use_id` found in `tool_result`
    blocks: srvtoolu_... Each `tool_result` block must have a corresponding
    `tool_use` block in the previous message.

The tests below pin both halves: the emission never asks the client to run a
provider-executed search, and a replayed history never leaves a ``tool_result``
without its ``tool_use``.
"""

from unittest.mock import AsyncMock

import pytest

from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.openai import ResponsesAPIStreamEvents
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Delta,
    Function,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)

SERVER_SEARCH_ID = "srvtoolu_01NQtgb8rWnWji3GxBsBUxaq"
CLIENT_TOOL_ID = "toolu_012UNSopfSsL8rmKbtEyHfSB"


def _chat_completion_with_server_search() -> ModelResponse:
    """What the Anthropic transformer produces for a turn whose content is
    [server_tool_use, web_search_tool_result, text, tool_use]."""
    return ModelResponse(
        id="chatcmpl-1",
        created=1,
        model="claude-sonnet-5",
        object="chat.completion",
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    role="assistant",
                    content="Here is what I found.",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id=SERVER_SEARCH_ID,
                            type="function",
                            function=Function(
                                name="web_search",
                                arguments='{"query": "capital of France"}',
                            ),
                        ),
                        ChatCompletionMessageToolCall(
                            id=CLIENT_TOOL_ID,
                            type="function",
                            function=Function(name="exec_command", arguments='{"cmd": "ls"}'),
                        ),
                    ],
                ),
            )
        ],
    )


def _output_items(response) -> list:
    return list(response.output or [])


def _assert_every_tool_result_has_its_tool_use(anthropic_messages: list[dict]) -> None:
    """The invariant Anthropic enforces, asserted directly.

    A ``tool_result`` must be answered by a ``tool_use`` in the message before
    it. A ``server_tool_use`` does NOT satisfy it — that mismatch is exactly
    what produced the 400 in rayward-internal/llm-gateway-infra#682.
    """
    for index, message in enumerate(anthropic_messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        result_ids = {
            block["tool_use_id"] for block in content if isinstance(block, dict) and block.get("type") == "tool_result"
        }
        if not result_ids:
            continue
        previous = anthropic_messages[index - 1] if index > 0 else {}
        previous_content = previous.get("content")
        use_ids = {
            block["id"]
            for block in (previous_content if isinstance(previous_content, list) else [])
            if isinstance(block, dict) and block.get("type") == "tool_use"
        }
        orphans = result_ids - use_ids
        assert not orphans, (
            f"messages.{index} carries tool_result blocks with no matching tool_use "
            f"in messages.{index - 1}: {sorted(orphans)}"
        )


class TestServerExecutedWebSearchEmission:
    def test_server_search_becomes_a_web_search_call_not_a_function_call(self):
        response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            request_input="what is the capital of France?",
            responses_api_request={},
            chat_completion_response=_chat_completion_with_server_search(),
        )

        items = _output_items(response)
        by_type = {item.type: item for item in items}

        assert "web_search_call" in by_type
        search_item = by_type["web_search_call"]
        assert search_item.id == f"ws_{SERVER_SEARCH_ID}"
        assert search_item.status == "completed"
        assert search_item.action is not None
        assert search_item.action.query == "capital of France"
        # No call_id: there is nothing for the client to answer.
        assert getattr(search_item, "call_id", None) is None

        # The real client tool call is untouched.
        assert by_type["function_call"].call_id == CLIENT_TOOL_ID
        assert by_type["function_call"].name == "exec_command"

        call_ids = {getattr(item, "call_id", None) for item in items}
        assert SERVER_SEARCH_ID not in call_ids

    def test_client_tool_named_web_search_is_still_a_function_call(self):
        """The ``srvtoolu_`` prefix is what marks a call as provider-executed.
        A client tool that happens to be named ``web_search`` must still be
        handed to the client."""
        chat_completion_response = ModelResponse(
            id="chatcmpl-2",
            created=1,
            model="claude-sonnet-5",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="toolu_01ClientSearch",
                                type="function",
                                function=Function(name="web_search", arguments='{"query": "x"}'),
                            )
                        ],
                    ),
                )
            ],
        )

        response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            request_input="search",
            responses_api_request={},
            chat_completion_response=chat_completion_response,
        )

        items = _output_items(response)
        assert [item.type for item in items] == ["function_call"]
        assert items[0].call_id == "toolu_01ClientSearch"


class TestServerExecutedWebSearchStreaming:
    def _iterator(self) -> LiteLLMCompletionStreamingIterator:
        return LiteLLMCompletionStreamingIterator(
            model="claude-sonnet-5",
            litellm_custom_stream_wrapper=AsyncMock(),
            request_input="what is the capital of France?",
            responses_api_request={},
        )

    def test_streamed_server_search_emits_no_function_call_events(self):
        iterator = self._iterator()
        chunk = ModelResponseStream(
            id="chunk-1",
            created=1,
            model="claude-sonnet-5",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        role="assistant",
                        content="",
                        tool_calls=[
                            {
                                "id": SERVER_SEARCH_ID,
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query": "capital of France"}',
                                },
                                "index": 0,
                            }
                        ],
                    ),
                )
            ],
        )

        assert iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk) is None
        assert iterator._pending_tool_events == []

    def test_final_events_report_the_search_as_a_web_search_call_item(self):
        iterator = self._iterator()
        iterator._queue_final_tool_call_done_events(_chat_completion_with_server_search())

        events = iterator._pending_tool_events
        types = [event.type for event in events]
        assert ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA not in types[:2]

        added, done = events[0], events[1]
        assert added.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
        assert added.item.type == "web_search_call"
        assert added.item.id == f"ws_{SERVER_SEARCH_ID}"
        assert added.item.status == "in_progress"
        assert done.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE
        assert done.item.type == "web_search_call"
        assert done.item.status == "completed"
        assert done.item.action is not None
        assert done.item.action.query == "capital of France"
        assert getattr(done.item, "call_id", None) is None

        # The genuine client tool call still streams as a function call.
        function_call_items = [
            event.item
            for event in events
            if event.type
            in (
                ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
            )
            and event.item.type == "function_call"
        ]
        assert {item.call_id for item in function_call_items} == {CLIENT_TOOL_ID}


class TestServerExecutedWebSearchReplay:
    def _anthropic_messages(self, responses_input: list) -> list[dict]:
        messages = LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
            responses_input
        )
        request = AnthropicConfig().transform_request(
            model="claude-sonnet-5",
            messages=[message if isinstance(message, dict) else message.model_dump() for message in messages],
            optional_params={"max_tokens": 64},
            litellm_params={},
            headers={},
        )
        return request["messages"]

    def test_echoed_web_search_call_item_produces_no_tool_result(self):
        """The new output item, replayed by the client, must add nothing to the
        Anthropic history — no orphan ``tool_result``, no fabricated
        ``tool_use``."""
        responses_input = [
            {"role": "user", "content": [{"type": "input_text", "text": "capital of France?"}]},
            {
                "type": "web_search_call",
                "id": f"ws_{SERVER_SEARCH_ID}",
                "status": "completed",
                "action": {"type": "search", "query": "capital of France"},
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Paris."}],
            },
            {"role": "user", "content": [{"type": "input_text", "text": "thanks"}]},
        ]

        anthropic_messages = self._anthropic_messages(responses_input)

        _assert_every_tool_result_has_its_tool_use(anthropic_messages)
        blocks = [
            block
            for message in anthropic_messages
            if isinstance(message.get("content"), list)
            for block in message["content"]
            if isinstance(block, dict)
        ]
        assert not [block for block in blocks if block.get("type") == "tool_result"]
        assert not [block for block in blocks if block.get("type") == "server_tool_use"]

    def test_legacy_recorded_server_search_pair_keeps_its_tool_use(self):
        """Histories recorded before this fix still carry the ``function_call`` /
        ``function_call_output`` pair codex wrote. Replaying them must not
        orphan the ``tool_result`` — this is the shape that 400'd in #682."""
        responses_input = [
            {"role": "user", "content": [{"type": "input_text", "text": "capital of France?"}]},
            {
                "type": "function_call",
                "id": f"fc_{SERVER_SEARCH_ID}",
                "call_id": SERVER_SEARCH_ID,
                "name": "web_search",
                "arguments": '{"query": "capital of France"}',
                "status": "completed",
            },
            {
                "type": "function_call",
                "id": f"fc_{CLIENT_TOOL_ID}",
                "call_id": CLIENT_TOOL_ID,
                "name": "exec_command",
                "arguments": '{"cmd": "ls"}',
                "status": "completed",
            },
            {
                "type": "function_call_output",
                "call_id": SERVER_SEARCH_ID,
                "output": "unsupported call: web_search",
            },
            {"type": "function_call_output", "call_id": CLIENT_TOOL_ID, "output": "a.txt"},
        ]

        anthropic_messages = self._anthropic_messages(responses_input)

        _assert_every_tool_result_has_its_tool_use(anthropic_messages)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
