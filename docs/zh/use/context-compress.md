# 上下文压缩

对话越长，发送给模型的历史消息、工具调用结果等内容就越多，它们会占用模型的上下文窗口，也会增加请求的 Token 用量。

本构建中 Codex 是唯一的 Agent 执行器。Codex 自己维护每个会话的线程上下文，并在上下文接近窗口上限时自动压缩（compaction）。旧版内置 Agent 的「上下文管理策略」设置（压缩前最多保留对话轮数、按对话轮数截断或由 LLM 压缩、压缩提示词、保留最近上下文比例、压缩模型、上下文窗口兜底值）已经删除，旧配置文件中的这些值会被自动丢弃。

## 调整 Codex 的压缩

- 在 WebUI 的「Codex」页面为自定义模型提供商选择压缩方式（`auto`、`local`、`remote`），并可为模型填写上下文窗口（`context_window`）和自动压缩阈值（`auto_compact_token_limit`）。
- 实时语音会话空闲时，如果提示词超过上下文窗口的一定比例（`agent_runner.config.realtime_voice.idle_compact_percent`，默认 70%，设为 0 则不压缩），会提前压缩。
- 发送 `/new` 开始新的对话，可以从空白上下文重新开始。

> [!NOTE]
> 压缩不能保证保留每个细节。需要长期保存的资料应放入[知识库](./knowledge-base.md)或文件，或启用 Codex 记忆。
