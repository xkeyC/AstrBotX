# Agent 执行方式 {#agent-执行器}

Agent 执行器是 AstrBot 中用于执行 Agent 的组件。

本构建只有一种 Agent 执行器：**Codex**。AstrBot 在进程内驱动 Codex 运行 Agent 循环，插件工具以 Codex 动态工具的形式提供给模型；人格、知识库、执行环境和主动型 Agent 等能力仍由 AstrBot 提供。

旧版本的内置 Agent 执行器以及 Dify、Coze、阿里云百炼应用、DeerFlow 执行器已经移除。使用它们的旧配置文件会在启动时自动迁移到 Codex 的默认配置（迁移前的配置备份为同目录下的 `*.pre-codex.json`），内置执行器中设置的默认人格会保留。

## 为什么需要抽象出 Agent 执行器

在早期版本中，Dify、Coze、阿里云百炼应用这类「自带 Agent 能力」的平台，是作为普通 Chat Provider 集成进 AstrBot 的。实践下来会发现，它们和传统「只负责补全文本」的 Chat Provider 有本质差异，强行放在同一层会带来很多设计和使用上的冲突。因此，从 v4.7.0 起，我们将它们抽象为独立的 Agent 执行器（Agent Runner）。

从架构上看，可以理解为：

- Chat Provider 负责「说话」；
- Agent 执行器负责「思考 + 做事」。

Agent 执行器会调用 Chat Provider 的接口，并根据 Chat Provider 的回复，进行多轮「感知 → 规划 → 执行动作 → 观察结果 → 再规划」的循环。

Chat Provider 本质上是一个 `单轮补全接口`，输入 prompt + 历史对话 + 工具列表，输出模型回复（文本、工具调用指令等）。

而 Agent Runner 通常是一个 `循环（Loop）`，接收用户意图、上下文与环境状态，基于策略 / 模型做出规划（Plan），选择并调用工具（Act），从环境中读取结果（Observe），再次理解结果、更新内部状态，决定下一步动作，重复上述过程，直到任务完成或超时。

![image](https://files.astrbot.app/docs/source/images/use/agent-runner/agent-arch.svg)

Dify、Coze、百炼应用、DeerFlow 等平台已经内置了这个循环，如果把它们当成普通 Chat Provider，会和 AstrBot 的内置 Agent 执行器功能冲突。

## 使用

1. 打开 WebUI 左侧「配置文件」，选择要修改的配置文件，进入「AI」，确保「启用 AI」已打开。
2. 在「Codex 设置」中配置工具模式、代理、记忆、超时等运行参数（执行环境由「使用电脑能力」中的运行环境决定）；在「人格」「知识库」「使用电脑能力」「主动型能力」中配置 AstrBot 提供的能力。
3. 模型、模型提供商、推理强度、系统提示词和附加开发者指令在 WebUI 的「Codex」页面设置（写入默认配置文件），「配置文件」页面不再重复显示。
4. 点击右下角「保存配置」使更改生效。

「Codex 代理」填写后，Codex 的所有网络请求都走它，优先于「系统设置」页面的「HTTP 代理」；留空时 Codex 使用 HTTP 代理写入的 `http_proxy` 等环境变量，都没有时直连。

> [!NOTE]
> 以下旧的内置 Agent 设置已删除，它们对 Codex 不起作用：对话模型、回退对话模型、异常重试次数、图片转述模型（Codex 直接读取图片）、上下文管理策略（Codex 自行管理上下文，见[上下文压缩](./context-compress.md)）、工具调用轮数上限、工具调用超时与工具调用模式（Codex 的超时在「Codex 设置」中）、按模型能力清理历史上下文、输出工具调用信息和结果、合并 Agent 中间消息、输入图片最大边长、引用图片回退解析上限，以及「人格」中的「健康模式」（Codex 使用「Codex 设置」中的「注入健康模式提示词」）。「图片转述提示词」移到了「群聊上下文感知」，只用于群聊记录中的图片。
