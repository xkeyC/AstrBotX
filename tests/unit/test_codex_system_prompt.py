"""The Codex thread's base instructions: the operator's prompt or the default
one, plus the code-mode tools guide, which a custom prompt does not drop."""

from astrbot.core.agent.runners.codex.codex_agent_runner import system_prompt
from astrbot.core.agent.runners.codex.constants import (
    CODE_MODE_TOOLS_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    RUN_CODE_TOOLS_PROMPT,
)


def test_code_mode_appends_the_tools_guide_to_a_custom_prompt():
    prompt = system_prompt({"base_instructions": "# How you work\nBe brief."})
    assert prompt.startswith("# How you work\nBe brief.")
    assert prompt.endswith(CODE_MODE_TOOLS_PROMPT)
    assert DEFAULT_SYSTEM_PROMPT.strip() not in prompt


def test_the_default_prompt_gets_the_guide_too():
    prompt = system_prompt({})
    assert prompt.startswith(DEFAULT_SYSTEM_PROMPT.rstrip())
    assert "# Orchestrating tools in code" in prompt


def test_direct_tool_mode_has_no_code_guide():
    prompt = system_prompt({"tool_mode": "direct"})
    assert "# Orchestrating tools in code" not in prompt
    assert "tools.exec_command" not in prompt


def test_the_guide_covers_orchestration_and_every_kind_of_running_work():
    guide = " ".join(CODE_MODE_TOOLS_PROMPT.split())
    assert "Promise.allSettled" in guide and "store(" in guide and "notify(" in guide
    # Each id goes back to the tool that gave it.
    assert 'wait({cell_id: "N"})' in guide
    assert (
        'write_stdin({session_id: "S"})' in guide and "`wait` does not take S" in guide
    )
    assert "tools.astrbot_tool_task" in guide
    assert "never start the same work again while it runs" in guide


def test_an_open_model_provider_gets_the_typescript_guide():
    providers = [
        {"id": "deep seek", "base_url": "https://x/v1", "open_model": True},
        {"id": "openai-like", "base_url": "https://y/v1"},
    ]
    prompt = system_prompt(
        {"model_provider": "deep_seek", "model_providers": providers}
    )
    assert prompt.endswith(RUN_CODE_TOOLS_PROMPT)
    assert "```ts" in prompt and "`run_code`" in prompt and "```js" not in prompt
    # Annotations only TypeScript takes.
    assert "const page: string" in prompt

    prompt = system_prompt(
        {"model_provider": "openai-like", "model_providers": providers}
    )
    assert prompt.endswith(CODE_MODE_TOOLS_PROMPT)
    assert "```js" in prompt and "```ts" not in prompt and "run_code" not in prompt
    # Codex's own provider (none set) takes exec too.
    assert system_prompt({"model_providers": providers}).endswith(
        CODE_MODE_TOOLS_PROMPT
    )


def test_both_guides_share_everything_but_the_tool_and_example():
    assert "CODE_TOOL" not in CODE_MODE_TOOLS_PROMPT + RUN_CODE_TOOLS_PROMPT
    for guide in (CODE_MODE_TOOLS_PROMPT, RUN_CODE_TOOLS_PROMPT):
        assert "# Work that is still running" in guide
        assert 'wait({cell_id: "N"})' in guide
