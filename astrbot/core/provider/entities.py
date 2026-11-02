from __future__ import annotations

import enum
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from anthropic.types import Message as AnthropicMessage
from deprecated import deprecated
from google.genai.types import GenerateContentResponse
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.responses import Response

import astrbot.core.message.components as Comp
from astrbot import logger
from astrbot.core.agent.context.persistent_context import (
    MESSAGE_META_UNIT,
    render_unit,
)
from astrbot.core.agent.message import (
    AssistantMessageSegment,
    ContentPart,
    TextPart,
    ToolCall,
    ToolCallMessageSegment,
    is_checkpoint_message,
)
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.db.po import Conversation
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.utils.media_utils import (
    MediaResolver,
    is_recoverable_image_error,
    resolve_image_ref_to_base64_data,
)


class ProviderType(enum.Enum):
    CHAT_COMPLETION = "chat_completion"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    DECISION = "decision"


@dataclass(frozen=True, slots=True)
class DecisionChoiceQuestion:
    """A decision question with a fixed set of choices.

    Attributes:
        instructions: The question or instruction evaluated by the model.
        criteria: Mapping of option IDs to descriptions. At least two entries are required.
    """

    instructions: str
    criteria: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class DecisionBooleanQuestion:
    """A decision question that asks for the probability of a statement being true.

    Attributes:
        instructions: The statement or question evaluated by the model.
    """

    instructions: str


@dataclass(frozen=True, slots=True)
class DecisionScoreQuestion:
    """A decision question scored against ordered levels.

    Attributes:
        instructions: The question or instruction evaluated by the model.
        criteria: Two to ten level descriptions, ordered from lowest to highest.
    """

    instructions: str
    criteria: list[str]


DecisionQuestion: TypeAlias = (
    DecisionChoiceQuestion | DecisionBooleanQuestion | DecisionScoreQuestion
)
DecisionState: TypeAlias = str | list[Any] | dict[str, Any]


@dataclass(frozen=True, slots=True)
class DecisionChoiceAnswer:
    """The normalized answer for a choice question.

    Attributes:
        choice: Selected option ID.
        probability: Probability of the selected option.
        probabilities: Full probability distribution over option IDs.
        confidence: Model confidence in the selected option.
    """

    choice: str
    probability: float
    probabilities: Mapping[str, float]
    confidence: float


@dataclass(frozen=True, slots=True)
class DecisionBooleanAnswer:
    """The normalized answer for a boolean question.

    Attributes:
        probability: Probability of the statement being true.
    """

    probability: float


@dataclass(frozen=True, slots=True)
class DecisionScoreAnswer:
    """The normalized answer for a score question.

    Attributes:
        score: Probability-weighted level number, possibly fractional.
        legend: Mapping of string level numbers to descriptions.
        probabilities: Full probability distribution over string level numbers.
        confidence: Model confidence in the score.
    """

    score: float
    legend: Mapping[str, str]
    probabilities: Mapping[str, float]
    confidence: float


DecisionAnswer: TypeAlias = (
    DecisionChoiceAnswer | DecisionBooleanAnswer | DecisionScoreAnswer
)


class DecisionError(RuntimeError):
    """Raised when a decision provider is unavailable or returns an invalid answer."""


@dataclass
class ProviderMeta:
    """The basic metadata of a provider instance."""

    id: str
    """the unique id of the provider instance that user configured"""
    model: str | None
    """the model name of the provider instance currently used"""
    type: str
    """the name of the provider adapter, such as openai, ollama"""
    provider_type: ProviderType = ProviderType.CHAT_COMPLETION
    """the capability type of the provider adapter"""


@dataclass
class ProviderMetaData(ProviderMeta):
    """The metadata of a provider adapter for registration."""

    desc: str = ""
    """the short description of the provider adapter"""
    cls_type: Any = None
    """the class type of the provider adapter"""
    default_config_tmpl: dict | None = None
    """the default configuration template of the provider adapter"""
    provider_display_name: str | None = None
    """the display name of the provider shown in the WebUI configuration page; if empty, the type is used"""


@dataclass
class ToolCallsResult:
    """工具调用结果"""

    tool_calls_info: AssistantMessageSegment
    """函数调用的信息"""
    tool_calls_result: list[ToolCallMessageSegment]
    """函数调用的结果"""

    def to_openai_messages(self) -> list[dict]:
        ret = [
            self.tool_calls_info.model_dump(),
            *[item.model_dump() for item in self.tool_calls_result],
        ]
        return ret

    def to_openai_messages_model(
        self,
    ) -> list[AssistantMessageSegment | ToolCallMessageSegment]:
        return [
            self.tool_calls_info,
            *self.tool_calls_result,
        ]


@dataclass
class ProviderRequest:
    prompt: str | None = None
    """提示词"""
    session_id: str | None = ""
    """会话 ID"""
    image_urls: list[str] = field(default_factory=list)
    """图片 URL 列表"""
    audio_urls: list[str] = field(default_factory=list)
    """音频 URL 列表，也支持本地路径"""
    dynamic_user_context_parts: list[ContentPart] = field(default_factory=list)
    """临时上下文：只随本次请求发送，放在用户原始输入之前，不持久化。"""
    persistent_user_context_parts: list[ContentPart] = field(default_factory=list)
    """消息单元：描述本条消息（发言人、时间、之前的群消息），随本轮持久化。"""
    context_anchors: dict[str, str] = field(default_factory=dict)
    """锚点：常驻指令，首次出现或内容变化时才发送并持久化。"""
    context_anchors_complete: bool = False
    """为 True 时 context_anchors 是该会话完整的常驻指令，历史里有而本次未声明的锚点会被撤销。
    主 Agent 构建的请求会设置；插件等临时请求保持 False，只补发不撤销，历史锚点继续生效。"""
    extra_user_content_parts: list[ContentPart] = field(default_factory=list)
    """用户输入后的附件、引用等额外内容块。"""
    func_tool: ToolSet | None = None
    """可用的函数工具"""
    contexts: list[dict] = field(default_factory=list)
    """
    OpenAI 格式上下文列表。
    参考 https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages
    """
    system_prompt: str = ""
    """系统提示词"""
    conversation: Conversation | None = None
    """关联的对话对象"""
    tool_calls_result: list[ToolCallsResult] | ToolCallsResult | None = None
    """附加的上次请求后工具调用的结果。参考: https://platform.openai.com/docs/guides/function-calling#handling-function-calls"""
    model: str | None = None
    """模型名称，为 None 时使用提供商的默认模型"""

    def __repr__(self) -> str:
        return (
            f"ProviderRequest(prompt={self.prompt}, session_id={self.session_id}, "
            f"image_count={len(self.image_urls or [])}, "
            f"audio_count={len(self.audio_urls or [])}, "
            f"func_tool={self.func_tool}, "
            f"contexts={self._print_friendly_context()}, "
            f"system_prompt={self.system_prompt}, "
            f"conversation_id={self.conversation.cid if self.conversation else 'N/A'}, "
        )

    def __str__(self) -> str:
        return self.__repr__()

    def append_tool_calls_result(self, tool_calls_result: ToolCallsResult) -> None:
        """添加工具调用结果到请求中"""
        if not self.tool_calls_result:
            self.tool_calls_result = []
        if isinstance(self.tool_calls_result, ToolCallsResult):
            self.tool_calls_result = [self.tool_calls_result]
        self.tool_calls_result.append(tool_calls_result)

    def _print_friendly_context(self):
        """打印友好的消息上下文。将多模态内容折叠为简短标记。"""
        if not self.contexts:
            return (
                f"prompt: {self.prompt}, image_count: {len(self.image_urls or [])}, "
                f"audio_count: {len(self.audio_urls or [])}"
            )

        result_parts = []

        for ctx in self.contexts:
            if is_checkpoint_message(ctx):
                continue
            role = ctx.get("role", "unknown")
            content = ctx.get("content", "")

            if isinstance(content, str):
                result_parts.append(f"{role}: {content}")
            elif isinstance(content, list):
                msg_parts = []
                image_count = 0
                audio_count = 0

                for item in content:
                    item_type = item.get("type", "")

                    if item_type == "text":
                        msg_parts.append(item.get("text", ""))
                    elif item_type == "image_url":
                        image_count += 1
                    elif item_type == "audio_url":
                        audio_count += 1

                if image_count > 0:
                    if msg_parts:
                        msg_parts.append(f"[+{image_count} images]")
                    else:
                        msg_parts.append(f"[{image_count} images]")
                if audio_count > 0:
                    if msg_parts:
                        msg_parts.append(f"[+{audio_count} audios]")
                    else:
                        msg_parts.append(f"[{audio_count} audios]")

                result_parts.append(f"{role}: {''.join(msg_parts)}")

        return "\n".join(result_parts)

    def add_temporary_context(self, name: str, content: str) -> None:
        """Send context with this request only; it is never stored.

        Use it for one-shot loads such as retrieval results or file contents.

        Args:
            name: Stable machine-readable context category.
            content: Context or instructions to expose to the model.
        """
        content = content.strip()
        if not content:
            return
        wrapped = (
            f'<request_context name="{name}">\n'
            "The following is trusted request-scoped application context. Follow it "
            "unless it conflicts with the root system message.\n"
            f"{content}\n"
            "</request_context>"
        )
        if any(
            isinstance(part, TextPart) and part.text == wrapped
            for part in self.dynamic_user_context_parts
        ):
            return
        self.dynamic_user_context_parts.append(TextPart(text=wrapped).mark_as_temp())

    def set_context_anchor(self, name: str, content: str) -> None:
        """Declare standing instructions for the conversation.

        An anchor is stored the first time it is sent and sent again only when
        its content changes or compaction dropped it, so unchanged instructions
        stay in the cached history prefix. Anchors a request stops declaring are
        revoked.

        Args:
            name: Stable machine-readable anchor name.
            content: Instructions the anchor carries; empty removes the anchor.
        """
        content = content.strip()
        if content:
            self.context_anchors[name] = content
        else:
            self.context_anchors.pop(name, None)

    def add_persistent_context(
        self,
        name: str,
        content: str,
        unit_id: str | None = None,
    ) -> None:
        """Store context with this turn, such as who sent the message.

        Args:
            name: Stable machine-readable unit category.
            content: Facts describing the current message.
            unit_id: Optional stable id. A unit whose id the conversation
                already stores is not stored again.
        """
        content = content.strip()
        if not content:
            return
        text = render_unit(name, content, unit_id)
        if any(
            isinstance(part, TextPart) and part.text == text
            for part in self.persistent_user_context_parts
        ):
            return
        part = TextPart(text=text)
        # Units about earlier messages (such as the group messages that
        # preceded this one) stay in front of the metadata of the message
        # being answered, which sits right before that message.
        meta_prefix = f'<context_unit name="{MESSAGE_META_UNIT}"'
        meta_index = next(
            (
                index
                for index, existing in enumerate(self.persistent_user_context_parts)
                if isinstance(existing, TextPart)
                and existing.text.startswith(meta_prefix)
            ),
            None,
        )
        if name != MESSAGE_META_UNIT and meta_index is not None:
            self.persistent_user_context_parts.insert(meta_index, part)
        else:
            self.persistent_user_context_parts.append(part)

    def assemble_dynamic_context(self) -> dict | None:
        """Wrap the request-scoped context into its own message.

        The message is delivered right before the user's own message, after the
        immutable history: cache checkpoints of the history stay reusable, and
        the user's message keeps only what the user actually sent.

        Returns:
            A user message carrying the request-scoped context blocks, or None
            when this request has no dynamic context.
        """
        if not self.dynamic_user_context_parts:
            return None

        content_blocks = []
        for part in self.dynamic_user_context_parts:
            dynamic_part = part.model_dump_for_context()
            dynamic_part["_no_save"] = True
            content_blocks.append(dynamic_part)
        return {"role": "user", "content": content_blocks}

    async def assemble_context(self) -> dict:
        """Serialize current media references without transforming image content.

        Returns:
            A user message containing encoded media, text or a failure placeholder.
        """
        # 构建内容块列表
        content_blocks = []
        image_capture_failed = False

        # 1. 用户原始发言
        if self.prompt and self.prompt.strip():
            content_blocks.append({"type": "text", "text": self.prompt})
        elif self.image_urls:
            # 如果没有文本但有图片，添加占位文本
            content_blocks.append({"type": "text", "text": "[图片]"})
        elif self.audio_urls:
            # 如果没有文本但有音频，添加占位文本
            content_blocks.append({"type": "text", "text": "[音频]"})

        # 2. 额外的内容块（附件、引用等）
        if self.extra_user_content_parts:
            for part in self.extra_user_content_parts:
                dumped = (
                    part if isinstance(part, dict) else part.model_dump_for_context()
                )
                # Capture bytes before event cleanup. Extra image paths must also
                # reach providers and persisted history as portable data URIs.
                if isinstance(dumped, dict) and dumped.get("type") == "image_url":
                    image_url = dumped.get("image_url")
                    url = image_url.get("url") if isinstance(image_url, dict) else None
                    if isinstance(url, str) and url:
                        try:
                            resolved = await resolve_image_ref_to_base64_data(url)
                        except Exception as exc:
                            if not is_recoverable_image_error(exc):
                                raise
                            logger.warning(
                                "Image source capture failed; skipping image (%s).",
                                type(exc).__name__,
                            )
                            image_capture_failed = True
                            continue
                        if resolved is None:
                            logger.warning(
                                "Image source capture returned no data; skipping image."
                            )
                            image_capture_failed = True
                            continue
                        dumped = {
                            **dumped,
                            "image_url": {
                                **image_url,
                                "url": resolved.to_data_url(),
                            },
                        }
                content_blocks.append(dumped)

        # 3. Read image references without resizing or transcoding.
        if self.image_urls:
            for image_url in self.image_urls:
                try:
                    image_data = await resolve_image_ref_to_base64_data(image_url)
                except Exception as exc:
                    if not is_recoverable_image_error(exc):
                        raise
                    logger.warning(
                        "Image source capture failed; skipping image (%s).",
                        type(exc).__name__,
                    )
                    image_capture_failed = True
                    continue
                if image_data is None:
                    logger.warning(
                        "Image source capture returned no data; skipping image."
                    )
                    image_capture_failed = True
                    continue
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data.to_data_url()},
                    },
                )

        # 4. 音频内容
        if self.audio_urls:
            for audio_url in self.audio_urls:
                try:
                    audio_data = await MediaResolver(
                        audio_url,
                        media_type="audio",
                        default_suffix=".wav",
                    ).to_base64_data(
                        strict=True,
                        target_format="wav",
                    )
                except Exception as exc:
                    logger.warning("音频预处理失败，将忽略。错误: %s", exc)
                    continue
                if not audio_data:
                    logger.warning("音频预处理结果为空，将忽略。")
                    continue
                content_blocks.append(
                    {
                        "type": "audio_url",
                        "audio_url": {"url": audio_data.to_data_url()},
                    },
                )

        if image_capture_failed and not any(
            block.get("type") != "text" or block.get("text", "").strip()
            for block in content_blocks
        ):
            content_blocks = [{"type": "text", "text": "[Image unavailable]"}]

        # 只有当只有一个来自 prompt 的文本块且没有额外内容块时，才降级为简单格式以保持向后兼容
        if (
            len(content_blocks) == 1
            and content_blocks[0]["type"] == "text"
            and not self.extra_user_content_parts
            and not self.image_urls
            and not self.audio_urls
        ):
            return {"role": "user", "content": content_blocks[0]["text"]}

        # 否则返回多模态格式
        return {"role": "user", "content": content_blocks}


@dataclass
class TokenUsage:
    input_other: int = 0
    """The number of input tokens, excluding cached tokens."""
    input_cached: int = 0
    """The number of input cached tokens."""
    output: int = 0
    """The number of output tokens."""

    @property
    def total(self) -> int:
        return self.input_other + self.input_cached + self.output

    @property
    def input(self) -> int:
        return self.input_other + self.input_cached

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_other=self.input_other + other.input_other,
            input_cached=self.input_cached + other.input_cached,
            output=self.output + other.output,
        )

    def __sub__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_other=self.input_other - other.input_other,
            input_cached=self.input_cached - other.input_cached,
            output=self.output - other.output,
        )


@dataclass
class LLMResponse:
    role: str
    """The role of the message, e.g., assistant, tool, err"""
    result_chain: MessageChain | None = None
    """A chain of message components representing the text completion from LLM."""
    reply_segments: list[MessageChain] = field(default_factory=list)
    """Independent replies of one response, e.g. the text written before and
    after a server-side tool call. Empty when the response is a single reply;
    `result_chain` always keeps the merged content."""
    tools_call_args: list[dict[str, Any]] = field(default_factory=list)
    """Tool call arguments."""
    tools_call_name: list[str] = field(default_factory=list)
    """Tool call names."""
    tools_call_ids: list[str] = field(default_factory=list)
    """Tool call IDs."""
    tools_call_extra_content: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Tool call extra content. tool_call_id -> extra_content dict"""
    internal_tools: dict[str, FunctionTool] = field(default_factory=dict)
    """Provider-owned client tools scoped to this response by tool call ID."""
    reasoning_content: str | None = None
    """The reasoning content extracted from the LLM, if any."""
    reasoning_signature: str | None = None
    """The signature of the reasoning content, if any."""

    raw_completion: (
        ChatCompletion | Response | GenerateContentResponse | AnthropicMessage | None
    ) = None
    """The raw completion response from the LLM provider."""

    _completion_text: str = ""
    """The plain text of the completion."""

    is_chunk: bool = False
    """Indicates if the response is a chunked response."""

    id: str | None = None
    """The ID of the response. For chunked responses, it's the ID of the chunk; for non-chunked responses, it's the ID of the response."""
    usage: TokenUsage | None = None
    """The usage of the response. For chunked responses, it's the usage of the chunk; for non-chunked responses, it's the usage of the response."""

    def __init__(
        self,
        role: str,
        completion_text: str | None = None,
        result_chain: MessageChain | None = None,
        tools_call_args: list[dict[str, Any]] | None = None,
        tools_call_name: list[str] | None = None,
        tools_call_ids: list[str] | None = None,
        tools_call_extra_content: dict[str, dict[str, Any]] | None = None,
        internal_tools: dict[str, FunctionTool] | None = None,
        reasoning_content: str | None = None,
        reasoning_signature: str | None = None,
        raw_completion: ChatCompletion
        | Response
        | GenerateContentResponse
        | AnthropicMessage
        | None = None,
        is_chunk: bool = False,
        id: str | None = None,
        usage: TokenUsage | None = None,
    ) -> None:
        """初始化 LLMResponse

        Args:
            role (str): 角色, assistant, tool, err
            completion_text (str, optional): 返回的结果文本，已经过时，推荐使用 result_chain. Defaults to "".
            result_chain (MessageChain, optional): 返回的消息链. Defaults to None.
            tools_call_args (List[Dict[str, any]], optional): 工具调用参数. Defaults to None.
            tools_call_name (List[str], optional): 工具调用名称. Defaults to None.
            raw_completion (ChatCompletion, optional): 原始响应, OpenAI 格式. Defaults to None.

        """
        if tools_call_args is None:
            tools_call_args = []
        if tools_call_name is None:
            tools_call_name = []
        if tools_call_ids is None:
            tools_call_ids = []
        if tools_call_extra_content is None:
            tools_call_extra_content = {}
        if internal_tools is None:
            internal_tools = {}

        self.role = role
        self.completion_text = completion_text
        self.result_chain = result_chain
        self.reply_segments = []
        self.tools_call_args = tools_call_args
        self.tools_call_name = tools_call_name
        self.tools_call_ids = tools_call_ids
        self.tools_call_extra_content = tools_call_extra_content
        self.internal_tools = internal_tools
        self.reasoning_content = reasoning_content
        self.reasoning_signature = reasoning_signature
        self.raw_completion = raw_completion
        self.is_chunk = is_chunk

        if id is not None:
            self.id = id
        if usage is not None:
            self.usage = usage

    @property
    def completion_text(self):
        if self.result_chain:
            return self.result_chain.get_plain_text()
        return self._completion_text

    @completion_text.setter
    def completion_text(self, value) -> None:
        if self.result_chain:
            self.result_chain.chain = [
                comp
                for comp in self.result_chain.chain
                if not isinstance(comp, Comp.Plain)
            ]  # 清空 Plain 组件
            self.result_chain.chain.insert(0, Comp.Plain(value))
        else:
            self._completion_text = value

    @deprecated(reason="Use to_openai_tool_calls_model instead.")
    def to_openai_tool_calls(self) -> list[dict]:
        """Convert to OpenAI tool calls format. Deprecated, use to_openai_tool_calls_model instead."""
        ret = []
        for idx, tool_call_arg in enumerate(self.tools_call_args):
            payload = {
                "id": self.tools_call_ids[idx],
                "function": {
                    "name": self.tools_call_name[idx],
                    "arguments": json.dumps(tool_call_arg),
                },
                "type": "function",
            }
            if self.tools_call_extra_content.get(self.tools_call_ids[idx]):
                payload["extra_content"] = self.tools_call_extra_content[
                    self.tools_call_ids[idx]
                ]
            ret.append(payload)
        return ret

    def to_openai_tool_calls_model(self) -> list[ToolCall]:
        """The same as to_openai_tool_calls but return pydantic model."""
        ret = []
        for idx, tool_call_arg in enumerate(self.tools_call_args):
            ret.append(
                ToolCall(
                    id=self.tools_call_ids[idx],
                    function=ToolCall.FunctionBody(
                        name=self.tools_call_name[idx],
                        arguments=json.dumps(tool_call_arg),
                    ),
                    # the extra_content will not serialize if it's None when calling ToolCall.model_dump()
                    extra_content=self.tools_call_extra_content.get(
                        self.tools_call_ids[idx]
                    ),
                ),
            )
        return ret

    @deprecated(reason="Use to_openai_tool_calls_model instead.")
    def to_openai_to_calls_model(self) -> list[ToolCall]:
        """Deprecated alias of to_openai_tool_calls_model (legacy misspelled name)."""
        return self.to_openai_tool_calls_model()


@dataclass
class RerankResult:
    index: int
    """在候选列表中的索引位置"""
    relevance_score: float
    """相关性分数"""
