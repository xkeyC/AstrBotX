# Context Compression

The longer a conversation runs, the more history and tool results are sent to the model. They take up the model's context window and add to the tokens each request uses.

In this build Codex is the only Agent runner. Codex maintains each conversation's thread context itself and compacts it automatically when the context approaches the window limit. The former built-in Agent's **Context Management** settings (maximum turns before compression, truncating by turns or compressing with an LLM, the compression prompt, the recent context ratio, the compression model and the fallback context window) have been removed; their values in old configuration files are dropped automatically.

## Tuning Codex's compaction

- On the **Codex** page of the WebUI, choose the compaction mode of a custom model provider (`auto`, `local` or `remote`) and optionally set a model's context window (`context_window`) and automatic compaction threshold (`auto_compact_token_limit`).
- An idle realtime voice session is compacted early once its prompt passes a share of the context window (`agent_runner.config.realtime_voice.idle_compact_percent`, 70% by default; 0 disables it).
- Send `/new` to start a new conversation from an empty context.

> [!NOTE]
> Compaction cannot keep every detail. Keep long-term material in a [knowledge base](./knowledge-base.md) or in files, or enable Codex memories.
