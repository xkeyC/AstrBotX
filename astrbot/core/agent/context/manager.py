from astrbot import logger

from ..message import Message
from .compressor import LLMSummaryCompressor, TruncateByTurnsCompressor
from .config import ContextConfig
from .round_utils import split_into_rounds
from .token_counter import EstimateTokenCounter
from .truncator import ContextTruncator


class ContextManager:
    """Context compression manager."""

    def __init__(
        self,
        config: ContextConfig,
    ) -> None:
        """Initialize the context manager.

        There are two strategies to handle context limit reached:
        1. Truncate by turns: remove older messages by turns.
        2. LLM-based compression: use LLM to summarize old messages.

        Args:
            config: The context configuration.
        """
        self.config = config

        self.token_counter = config.custom_token_counter or EstimateTokenCounter()
        self.truncator = ContextTruncator()

        if config.custom_compressor:
            self.compressor = config.custom_compressor
        elif config.llm_compress_provider:
            self.compressor = LLMSummaryCompressor(
                provider=config.llm_compress_provider,
                keep_recent_ratio=config.llm_compress_keep_recent_ratio,
                instruction_text=config.llm_compress_instruction,
                token_counter=self.token_counter,
                tools=config.llm_compress_tools,
            )
        else:
            self.compressor = TruncateByTurnsCompressor(
                truncate_turns=config.truncate_turns
            )

    async def process(
        self,
        messages: list[Message],
        trusted_token_usage: int = 0,
        *,
        hard_limit_only: bool = False,
    ) -> list[Message]:
        """Process the messages.

        Args:
            messages: The original message list.
            trusted_token_usage: Total tokens the provider reported for the
                request that produced the latest assistant message, if known.
            hard_limit_only: Compress only once the context no longer fits
                the window. The later steps of a run set it, so the prefix the
                run has already sent stays unchanged, and cached, until the
                window is actually full.

        Returns:
            The processed message list.
        """
        try:
            result = messages

            # 1. 基于轮次的截断 (Enforce max turns)
            if self.config.enforce_max_turns != -1:
                result = self.truncator.truncate_by_turns(
                    result,
                    keep_most_recent_turns=self.config.enforce_max_turns,
                    drop_turns=self.config.truncate_turns,
                )

            # 2. 基于 token 的压缩
            if self.config.max_context_tokens > 0:
                total_tokens = self._count_tokens(result, trusted_token_usage)

                if hard_limit_only:
                    needs_compression = total_tokens > self.config.max_context_tokens
                else:
                    needs_compression = self.compressor.should_compress(
                        result, total_tokens, self.config.max_context_tokens
                    )
                if needs_compression:
                    result = await self._run_compression(result, total_tokens)

            return result
        except Exception as e:
            logger.error(f"Error during context processing: {e}", exc_info=True)
            return messages

    def _count_tokens(self, messages: list[Message], trusted_token_usage: int) -> int:
        """Count tokens, estimating what the reported usage does not cover.

        Reported usage belongs to the request that produced the latest
        assistant message, so tool results and prompts added after it are
        estimated on top of it.

        Args:
            messages: The message list.
            trusted_token_usage: Total tokens the provider reported, or 0.

        Returns:
            The token count of the message list.
        """
        total = self.token_counter.count_tokens(messages, trusted_token_usage)
        if trusted_token_usage <= 0:
            return total
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role == "assistant":
                pending = messages[index + 1 :]
                if pending:
                    total += self.token_counter.count_tokens(pending)
                break
        return total

    async def _run_compression(
        self, messages: list[Message], prev_tokens: int
    ) -> list[Message]:
        """
        Compress/truncate the messages.

        If the context still exceeds the limit after compression (for example
        because the summary request failed), whole rounds are dropped, oldest
        first, until it fits. The latest round is always kept, even when it
        alone exceeds the limit.

        Args:
            messages: The original message list.
            prev_tokens: The token count before compression.

        Returns:
            The compressed/truncated message list.
        """
        logger.debug("Compress triggered, starting compression...")

        messages = await self.compressor(messages)

        # double check
        tokens_after_summary = self.token_counter.count_tokens(messages)

        # calculate compress rate
        compress_rate = (tokens_after_summary / self.config.max_context_tokens) * 100
        logger.info(
            f"Compress completed."
            f" {prev_tokens} -> {tokens_after_summary} tokens,"
            f" compression rate: {compress_rate:.2f}%.",
        )

        # last check
        if not self.compressor.should_compress(
            messages, tokens_after_summary, self.config.max_context_tokens
        ):
            return messages

        logger.info(
            "Context still exceeds max tokens after compression, dropping the oldest rounds..."
        )
        # Drop whole rounds, oldest first, so that tool calls stay with their
        # results, and recount after each one. The latest round is the current
        # request and is always kept.
        first_non_system = next(
            (i for i, msg in enumerate(messages) if msg.role != "system"),
            len(messages),
        )
        system_messages = messages[:first_non_system]
        rounds = [
            [seg for seg in rnd if isinstance(seg, Message)]
            for rnd in split_into_rounds(messages[first_non_system:])
        ]
        tokens = tokens_after_summary
        dropped_rounds = 0
        while len(rounds) > 1 and self.compressor.should_compress(
            messages, tokens, self.config.max_context_tokens
        ):
            rounds.pop(0)
            dropped_rounds += 1
            messages = system_messages + [msg for rnd in rounds for msg in rnd]
            tokens = self.token_counter.count_tokens(messages)
        logger.info(f"Dropped {dropped_rounds} oldest round(s).")

        if self.compressor.should_compress(
            messages, tokens, self.config.max_context_tokens
        ):
            logger.warning(
                "Context still exceeds max tokens with only the latest round left; "
                "sending it as is."
            )
        return messages
