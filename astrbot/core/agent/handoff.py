from typing import Generic

from .agent import Agent
from .run_context import TContext
from .tool import FunctionTool


class HandoffTool(FunctionTool, Generic[TContext]):
    """Handoff tool for delegating tasks to another agent."""

    def __init__(
        self,
        agent: Agent[TContext],
        parameters: dict | None = None,
        tool_description: str | None = None,
        **kwargs,
    ) -> None:
        # `tool_description` is the public description shown to the main LLM.
        # Keep a separate kwarg to avoid conflicting with FunctionTool's `description`.
        description = tool_description or self.default_description(agent.name)
        super().__init__(
            name=f"transfer_to_{agent.name}",
            parameters=parameters or self.default_parameters(),
            description=description,
            **kwargs,
        )

        # Note: Must assign after super().__init__() to prevent parent class from overriding this attribute
        self.agent = agent

    def default_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The input to be handed off to another agent. This should be a clear and concise request or task.",
                },
                "image_urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: An array of image sources (public HTTP URLs or local file paths) used as references in multimodal tasks such as video generation.",
                },
                "background_task": {
                    "type": "boolean",
                    "description": (
                        "Defaults to false. "
                        "Set to true if the task may take noticeable time, involves external tools, or the user does not need to wait. "
                        "Use false only for quick, immediate tasks."
                    ),
                },
            },
        }

    def default_description(self, agent_name: str | None) -> str:
        agent_name = agent_name or "another"
        return f"Delegate tasks to {agent_name} agent to handle the request."
