# Agent Execution Mode {#agent-runner}

The Agent Runner is a component in AstrBot used to execute Agents.

This build has a single Agent Runner: **Codex**. AstrBot drives Codex in-process to run the agent loop and exposes plugin tools to the model as Codex dynamic tools; personas, knowledge bases, the execution environment and the proactive agent are still provided by AstrBot.

The former built-in Agent Runner and the Dify, Coze, Alibaba Cloud Bailian Application and DeerFlow runners have been removed. Configuration files that used them are migrated to Codex's defaults on startup (the previous configuration is backed up next to it as `*.pre-codex.json`); the default persona set for the built-in runner is kept.

## Why Abstract the Agent Runner

In earlier versions, platforms with "built-in Agent capabilities" like Dify, Coze, and Alibaba Cloud Bailian Application were integrated into AstrBot as regular Chat Providers. In practice, we found that they are fundamentally different from traditional Chat Providers that "only handle text completion". Forcing them into the same layer caused many design and usage conflicts. Therefore, starting from v4.7.0, we abstracted them into independent Agent Runners.

From an architectural perspective, you can understand it as:

- Chat Provider is responsible for "talking";
- Agent Runner is responsible for "thinking + doing".

The Agent Runner calls the Chat Provider's interface and, based on the Chat Provider's response, performs multi-turn "perceive → plan → execute action → observe result → re-plan" loops.

A Chat Provider is essentially a `single-turn completion interface`, taking prompt + conversation history + tool list as input and outputting model responses (text, tool call instructions, etc.).

An Agent Runner is typically a `loop` that receives user intent, context, and environment state, makes plans based on strategy/model (Plan), selects and invokes tools (Act), reads results from the environment (Observe), understands the results again, updates internal state, decides the next action, and repeats this process until the task is completed or times out.

![image](https://files.astrbot.app/docs/source/images/use/agent-runner/agent-arch.svg)

Platforms like Dify, Coze, Bailian Application, and DeerFlow have this loop built-in. If you treat them as regular Chat Providers, it will conflict with AstrBot's built-in Agent Runner functionality.

## Usage

1. Open **Configuration** in the WebUI sidebar, select the configuration file to edit, open **AI**, and make sure **Enable AI** is on.
2. Under **Codex Settings**, configure the tool mode, proxy, memories, timeouts and other runtime options (the execution environment is the runtime chosen under **Agent Computer Use**); configure the capabilities AstrBot provides under **Persona**, **Knowledge Base**, **Agent Computer Use** and **Proactive Agent**.
3. The model, model provider, reasoning effort, system prompt and extra developer instructions are set on the **Codex** page of the WebUI (written to the default configuration file); the **Configuration** page no longer repeats them.
4. Click **Save** in the lower-right corner to apply the changes.

When **Codex Proxy** is set, all of Codex's traffic goes through it, taking precedence over the **HTTP Proxy** on the **System Settings** page; when it is empty, Codex uses the `http_proxy`-style environment variables that the HTTP proxy is written to, and connects directly if none is set.

> [!NOTE]
> The following settings of the former built-in Agent have been removed; they had no effect on Codex: chat model, fallback chat models, retries on error, image caption model (Codex reads images itself), context management (Codex manages its own context, see [Context Compression](./context-compress.md)), maximum tool call steps, tool call timeout and tool schema mode (Codex's timeouts are under **Codex Settings**), sanitizing history by model capability, showing tool calls and their results, merging intermediate agent messages, input image maximum edge length, the quoted image fallback limit, and **Safety Mode** under **Persona** (Codex uses **Inject Safety Prompt** under **Codex Settings**). **Image Caption Prompt** moved to **Group Chat Context Awareness** and only applies to images in group chat records.
