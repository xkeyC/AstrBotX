"""A run keeps the prefix it sent and routes its requests to one prompt cache.

Every step of a run resends the previous request plus what that step added, so
the provider can reuse its cached prefix. Compaction may only rewrite that
prefix at the start of a run, or later when the window is actually full.
"""

import copy
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletionChunk

from astrbot.core.agent.context.config import ContextConfig
from astrbot.core.agent.context.manager import ContextManager
from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.message import Message
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial

WINDOW = 1000


class _ToolExecutor:
    @classmethod
    def execute(cls, tool, run_context, **tool_args):
        async def generator():
            from mcp.types import CallToolResult, TextContent

            yield CallToolResult(content=[TextContent(type="text", text="found")])

        return generator()


def _chunk(delta: dict | None = None, finish=None, usage=None):
    return ChatCompletionChunk.model_validate(
        {
            "id": "chunk",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-test",
            "choices": []
            if usage
            else [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
            "usage": usage,
        }
    )


def _provider(reported_usage: list[int], tool_steps: int, **config):
    """Build an OpenAI provider whose SDK call records every request."""
    provider = ProviderOpenAIOfficial(
        {
            "id": "openai-test",
            "provider": "openai",
            "type": "openai_chat_completion",
            "key": ["k"],
            "model": "gpt-test",
            "modalities": ["text", "tool_use"],
            "max_context_tokens": WINDOW,
            **config,
        },
        {},
    )
    requests: list[dict] = []

    async def create(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        call = len(requests)
        if call <= tool_steps:
            chunks = [
                _chunk(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": f"call_{call}",
                                "type": "function",
                                "function": {"name": "search", "arguments": "{}"},
                            }
                        ],
                    }
                ),
                _chunk(finish="tool_calls"),
            ]
        else:
            chunks = [
                _chunk({"role": "assistant", "content": "answer"}),
                _chunk(finish="stop"),
            ]
        total = reported_usage[min(call, len(reported_usage)) - 1]
        chunks.append(
            _chunk(
                usage={
                    "prompt_tokens": total - 5,
                    "completion_tokens": 5,
                    "total_tokens": total,
                }
            )
        )

        async def stream():
            for item in chunks:
                yield item

        return stream()

    provider.client.chat.completions.create = create
    return provider, requests


def _history(turns: int) -> list[dict]:
    history = []
    for index in range(turns):
        history += [
            {"role": "user", "content": f"question {index}"},
            {"role": "assistant", "content": f"answer {index}"},
        ]
    return history


async def _run(provider, starting_usage: int = 0) -> None:
    request = ProviderRequest(
        prompt="Search, then answer.",
        session_id="aiocqhttp:GroupMessage:12345",
        contexts=_history(4),
        func_tool=ToolSet(
            [
                FunctionTool(
                    name="search",
                    description="Search the web",
                    parameters={"type": "object", "properties": {}},
                )
            ]
        ),
    )
    request.conversation = SimpleNamespace(cid="conv-1", token_usage=starting_usage)
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=request,
        run_context=ContextWrapper(context=None),
        tool_executor=_ToolExecutor(),
        agent_hooks=BaseAgentRunHooks(),
        streaming=False,
    )
    async for _ in runner.step_until_done(10):
        pass


def _append_only(requests: list[dict]) -> list[bool]:
    return [
        later["messages"][: len(earlier["messages"])] == earlier["messages"]
        for earlier, later in zip(requests, requests[1:])
    ]


@pytest.mark.asyncio
async def test_steps_above_the_soft_threshold_keep_the_prefix():
    # 90% of the window: the soft threshold (82%) is crossed after call 1.
    provider, requests = _provider([900], tool_steps=3)

    await _run(provider)

    assert len(requests) == 4
    assert _append_only(requests) == [True, True, True]


@pytest.mark.asyncio
async def test_a_full_window_still_compacts_within_the_run():
    # Call 2 fills the window; the compacted call 3 reports far less again.
    provider, requests = _provider([900, 1100, 300], tool_steps=3)

    await _run(provider)

    assert _append_only(requests) == [True, False, True]
    assert requests[1]["messages"][0]["content"] == "question 0"
    assert requests[2]["messages"][0]["content"] == "question 1"


@pytest.mark.asyncio
async def test_the_soft_threshold_still_applies_when_a_run_starts():
    provider, requests = _provider([100], tool_steps=0)

    await _run(provider, starting_usage=900)

    # Four stored turns plus the prompt; one turn is dropped before sending.
    assert len(requests[0]["messages"]) < 9


@pytest.mark.asyncio
async def test_hard_limit_mode_ignores_the_soft_threshold():
    manager = ContextManager(ContextConfig(max_context_tokens=WINDOW))
    messages = [
        Message(role="user", content="q"),
        Message(role="assistant", content="a"),
    ]

    kept = await manager.process(messages, 900, hard_limit_only=True)
    compacted = await manager.process(messages, 900)

    assert kept == messages
    assert compacted != messages


def test_messages_after_the_reported_request_are_estimated_on_top():
    manager = ContextManager(ContextConfig(max_context_tokens=WINDOW))
    tool_result = "x" * 1000
    messages = [
        Message(role="user", content="q"),
        Message(role="assistant", content="a"),
        Message(role="user", content=tool_result),
    ]

    counted = manager._count_tokens(messages, 900)

    assert counted == 900 + manager.token_counter.count_tokens([messages[-1]])


@pytest.mark.asyncio
async def test_each_run_routes_its_requests_by_its_own_short_key():
    provider, requests = _provider([100], tool_steps=2)

    await _run(provider)
    await _run(provider)

    # The first run makes three requests; the second answers at once.
    first_run = {request.get("prompt_cache_key") for request in requests[:3]}
    second_run = requests[3].get("prompt_cache_key")
    assert len(first_run) == 1
    key = first_run.pop()
    assert key and len(key) == 8
    assert second_run and second_run != key


@pytest.mark.asyncio
async def test_prompt_cache_key_can_be_disabled():
    provider, requests = _provider([100], tool_steps=0, enable_prompt_cache_key=False)

    await _run(provider)

    assert "prompt_cache_key" not in requests[0]


@pytest.mark.asyncio
async def test_prompt_cache_key_is_off_by_default_for_other_vendors():
    provider, requests = _provider([100], tool_steps=0, provider="deepseek")

    await _run(provider)

    assert "prompt_cache_key" not in requests[0]
