---
outline: deep
---

# AstrBot 配置文件

## data/cmd_config.json

AstrBot 的配置文件是一个 JSON 格式的文件。AstrBot 会在启动时读取这个文件，并根据文件中的配置来初始化 AstrBot，其路径位于 `data/cmd_config.json`。

> 在 AstrBot v4.0.0 版本及之后，我们引入了[多配置文件](https://blog.astrbot.app/posts/what-is-changed-in-4.0.0/#%E5%A4%9A%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6)的概念。`data/cmd_config.json` 作为默认配置文件 `default`。其他您在 WebUI 新建的配置文件会存储在 `data/config/` 目录下，以 `abconf_` 开头。

WebUI 中，按机器人或会话使用的行为配置（含文本转图像）在 `配置文件` 页面管理；全局运行、日志、网络和 WebUI 安全等配置在 `系统设置` 页面管理。模型连接和机器人连接分别在 `模型提供商` 和 `机器人` 页面管理。

AstrBot 默认配置如下：

```jsonc
{
    "config_version": 2,
    "platform_settings": {
        "unique_session": False,
        "rate_limit": {
            "time": 60,
            "count": 30,
            "strategy": "stall",  # stall, discard
        },
        "reply_prefix": "",
        "forward_threshold": 1500,
        "enable_id_white_list": True,
        "id_whitelist": [],
        "id_whitelist_log": True,
        "wl_ignore_admin_on_group": True,
        "wl_ignore_admin_on_friend": True,
        "reply_with_mention": False,
        "reply_with_quote": False,
        "path_mapping": [],
        "segmented_reply": {
            "enable": False,
            "only_llm_result": True,
            "interval_method": "random",
            "interval": "1.5,3.5",
            "log_base": 2.6,
            "words_count_threshold": 150,
            "regex": ".*?[。？！~…]+|.+$",
            "content_cleanup_rule": "",
        },
        "no_permission_reply": True,
        "empty_mention_waiting": True,
        "empty_mention_waiting_need_reply": True,
        "friend_message_needs_wake_prefix": False,
        "ignore_bot_self_message": False,
        "ignore_at_all": False,
    },
    "provider": [],
    "provider_settings": {
        "enable": True,
        "image_caption_prompt": "Please describe the image using Chinese.",
        "wake_prefix": "",
        "display_reasoning_text": False,
        "identifier": False,
        "group_name_display": False,
        "datetime_system_prompt": True,
        "prompt_prefix": "{{prompt}}",
        "streaming_response": False,
        "quoted_message_parser": {
            "max_component_chain_depth": 4,
            "max_forward_node_depth": 6,
            "max_forward_fetch": 32,
            "warn_on_action_failure": False,
        },
        "unsupported_streaming_strategy": "realtime_segmenting",
        # ……以及 Computer Use、沙箱和主动型 Agent 的设置
    },
    "agent_runner": {
        "runner_type": "codex",
        "config": {
            "persona_id": "default",
            "safety_mode": False,
            "proxy": "",
            # ……其余 Codex 设置
        },
    },
    "provider_stt_settings": {
        "enable": False,
        "provider_id": "",
    },
    "provider_tts_settings": {
        "enable": False,
        "provider_id": "",
        "dual_output": False,
        "use_file_service": False,
    },
    "provider_ltm_settings": {
        "group_icl_enable": False,
        "group_message_max_cnt": 1000,
        "image_caption": False,
        "active_reply": {
            "enable": False,
            "method": "possibility_reply",
            "possibility_reply": 0.1,
            "whitelist": [],
        },
    },
    "content_safety": {
        "also_use_in_response": False,
        "internal_keywords": {"enable": True, "extra_keywords": []},
        "baidu_aip": {"enable": False, "app_id": "", "api_key": "", "secret_key": ""},
    },
    "admins_id": ["astrbot"],
    "t2i": False,
    "t2i_word_threshold": 150,
    "t2i_strategy": "remote",
    "t2i_endpoint": "",
    "t2i_use_file_service": False,
    "t2i_active_template": "base",
    "http_proxy": "",
    "no_proxy": ["localhost", "127.0.0.1", "::1"],
    "local_infra": {
        "enable": False,
        "base_url": "http://127.0.0.1:17890",
        "token": "",
        "timeout": 120,
        "services": {
            "stt": {"enable": True, "model": ""},
            "tts": {"enable": True, "model": "", "reference_audio": "", "reference_text": ""},
            "embedding": {"enable": True, "model": ""},
            "rerank": {"enable": True, "model": ""},
            "moderation": {"enable": True, "blocked_reply": "这条内容无法处理。", "threshold": 0.9, "nsfw_threshold": 0.5, "nsfw_labels": ["low", "medium", "high"], "categories": ["violent", "illegal", "sexual", "pii", "self-harm", "unethical", "political", "copyright", "jailbreak"]},
            "realtime_voice": {"enable": True}
        }
    },
    "dashboard": {
        "enable": True,
        "username": "astrbot",
        "password": "<your_password_md5>",
        "jwt_secret": "",
        "host": "0.0.0.0",
        "port": 6185,
    },
    "platform": [],
    "platform_specific": {
        # 平台特异配置：按平台分类，平台下按功能分组
        "lark": {
            "pre_ack_emoji": {"enable": False, "emojis": ["Typing"]},
        },
        "telegram": {
            "pre_ack_emoji": {"enable": False, "emojis": ["✍️"]},
        },
        "discord": {
            "pre_ack_emoji": {"enable": False, "emojis": ["🤔"]},
        },
    },
    "wake_prefix": ["/"],
    "log_level": "INFO",
    "trace_enable": False,
    "pip_install_arg": "",
    "pypi_index_url": "https://mirrors.aliyun.com/pypi/simple/",
    "persona": [],  # deprecated
    "timezone": "Asia/Shanghai",
    "callback_api_base": "",
    "plugin_set": ["*"],  # "*" 表示使用所有可用的插件, 空列表表示不使用任何插件
}
```

## 字段详解

### `config_version`

配置文件版本，请勿修改。

### `platform_settings`

消息平台适配器的通用设置。

#### `platform_settings.unique_session`

是否启用「隔离对话」。默认为 `false`。启用后，支持隔离的渠道会为每位群成员使用独立上下文；不支持隔离的渠道仍使用原有上下文。`/new` 和 `/reset` 的权限由[指令管理设置](../use/command.md)决定，默认跟随对话是否隔离。

#### `platform_settings.rate_limit`

当消息速率超过限制时的处理策略。`time` 为时间窗口，`count` 为消息数量，`strategy` 为限制策略。`stall` 为等待，`discard` 为丢弃。

#### `platform_settings.reply_prefix`

回复消息时的固定前缀字符串。默认为空。

#### `platform_settings.forward_threshold`

> 目前仅 QQ 平台适配器适用。

消息转发阈值。当回复内容超过一定字数后，机器人会将消息折叠成 QQ 群聊的 “转发消息”，以防止刷屏。

在 WebUI 中，选择机器人使用的配置文件，进入 `访问控制 → 白名单` 管理以下白名单设置。

#### `platform_settings.enable_id_white_list`

是否启用 ID 白名单。默认为 `true`。启用后，只有在白名单中的 ID 发来的消息才会被处理。

#### `platform_settings.id_whitelist`

ID 白名单。填写后，将只处理所填写的 ID 发来的消息事件。为空时表示不启用白名单过滤。可以使用 `/sid` 指令获取在某个平台上的会话 ID。

也可在 AstrBot 日志内获取会话 ID，当一条消息没通过白名单时，会输出 INFO 级别的日志，格式类似 `aiocqhttp:GroupMessage:547540978`

#### `platform_settings.id_whitelist_log`

是否打印未通过 ID 白名单的消息日志。默认为 `true`。

#### `platform_settings.wl_ignore_admin_on_group` & `platform_settings.wl_ignore_admin_on_friend`

- `wl_ignore_admin_on_group`: 是否管理员发送的群组消息无视 ID 白名单。默认为 `true`。

- `wl_ignore_admin_on_friend`: 是否管理员发送的私聊消息无视 ID 白名单。默认为 `true`。

#### `platform_settings.reply_with_mention`

是否在回复消息时 @ 提到用户。默认为 `false`。

#### `platform_settings.reply_with_quote`

是否在回复消息时引用用户的消息。默认为 `false`。

#### `platform_settings.path_mapping`

*该配置项已经在 v4.0.0 版本之后被废弃。*

路径映射列表。用于将消息中的文件路径进行替换。每个映射项包含 `from` 和 `to` 两个字段，表示将消息中的 `from` 路径替换为 `to` 路径。

#### `platform_settings.segmented_reply`

分段回复设置。

- `enable`: 是否启用分段回复。默认为 `false`。
- `only_llm_result`: 是否仅对 LLM 生成的回复进行分段。默认为 `true`。
- `interval_method`: 分段间隔方法。可选值为 `random` 和 `log`。默认为 `random`。
- `interval`: 分段间隔时间。对于 `random` 方法，填写两个逗号分隔的数字，表示最小和最大间隔时间（单位：秒）。对于 `log` 方法，填写一个数字，表示对数基底。默认为 `"1.5,3.5"`。
- `log_base`: 对数基底，仅在 `interval_method` 为 `log` 时适用。默认为 `2.6`。
- `words_count_threshold`: 分段回复的字数上限。只有字数小于此值的消息才会被分段，超过此值的长消息将直接发送（不分段）。默认为 `150`。
- `regex`: 用于分隔一段消息。默认情况下会根据句号、问号等标点符号分隔。`re.findall(r'<regex>', text)`。默认值为 `".*?[。？！~…]+|.+$"`。
- `content_cleanup_rule`: 移除分段后的内容中的指定的内容。支持正则表达式。如填写 `[。？！]` 将移除所有的句号、问号、感叹号。`re.sub(r'<regex>', '', text)`。

#### `platform_settings.no_permission_reply`

是否在用户没有权限时回复无权限的提示消息。默认为 `true`。

#### `platform_settings.empty_mention_waiting`

是否启用空 @ 等待机制。默认为 `true`。启用后，当用户发送一条仅包含 @ 机器人的消息时，机器人会等待用户在 60 秒内发送下一条消息，并将两条消息合并后进行处理。这在某些平台不支持 @ 和语音/图片等消息同时发送时特别有用。

#### `platform_settings.empty_mention_waiting_need_reply`

在上面一个配置项(`empty_mention_waiting`)中，如果启用了触发等待，启用此项后，机器人会立即使用 LLM 生成一条回复。否则，将不回复而只是等待。默认为 `true`。

#### `platform_settings.friend_message_needs_wake_prefix`

是否在消息平台的私聊消息中需要唤醒前缀。默认为 `false`。启用后，在私聊消息中，用户需要使用唤醒前缀才能触发机器人的响应。

#### `platform_settings.ignore_bot_self_message`

是否忽略机器人自己发送的消息。默认为 `false`。启用后，机器人将不会处理自己发送的消息，在某些平台可以防止死循环。

#### `platform_settings.ignore_at_all`

是否忽略 @ 全体成员的消息。默认为 `false`。启用后，机器人将不会响应包含 @ 全体成员的消息。

### `provider`

> 此配置项仅在 `data/cmd_config.json` 中生效，AstrBot 不会读取 `data/config/` 目录下的配置文件中的此项。

已配置的模型服务提供商的配置列表。

### `provider_settings`

大语言模型提供商的通用设置。

#### `provider_settings.enable`

是否启用大语言模型聊天。默认为 `true`。

#### `provider_settings.image_caption_prompt`

群聊上下文感知的图片转述提示词，在 WebUI 的 `群聊上下文感知` 中与 `群聊图片转述模型`（`provider_ltm_settings.image_caption_provider_id`）一起设置。默认为 `"Please describe the image using Chinese."`。

#### `provider_settings.wake_prefix`

使用 LLM 聊天额外的触发条件。如填写 `chat`，则需要发送消息时要以 `/chat` 才能触发 LLM 聊天。其中 `/` 是机器人的唤醒前缀。是一个防止滥用的手段。

#### `provider_settings.display_reasoning_text`

是否在回复前附上模型的思考内容（Codex 的推理摘要）。默认为 `false`。

#### `provider_settings.identifier`

是否在 Prompt 前加上群成员的名字以让模型更好地了解群聊状态。默认为 `false`。启用将略微增加 token 开销。

#### `provider_settings.group_name_display`

是否在提示模型了解所在群的名称。默认为 `false`。此配置项目前仅在 QQ 平台适配器中生效。

#### `provider_settings.datetime_system_prompt`

是否在系统提示词中加上当前机器的日期时间。默认为 `true`。

#### `provider_settings.prompt_prefix`

用户提示词。可使用 `{{prompt}}` 作为用户输入的占位符。如果不输入占位符则代表添加在用户输入的前面。

#### `provider_settings.streaming_response`

是否启用流式响应。默认为 `false`。启用后，模型的回复会实时类似打字机的效果发送给用户。此配置项仅在 WebChat、Telegram、飞书平台生效。

#### `provider_settings.quoted_message_parser`

解析引用消息和合并转发消息时的限制：`max_component_chain_depth` 为 Reply 组件链的最大递归深度，`max_forward_node_depth` 为合并转发节点的最大递归深度，`max_forward_fetch` 为递归调用 `get_forward_msg` 的最大次数，`warn_on_action_failure` 为全部拉取失败时是否输出 warning 日志。

> [!NOTE]
> 以下设置只对已移除的内置 Agent 执行器生效，现已删除。`provider_settings` 和顶层的旧值在加载时自动丢弃；`agent_runner.config` 下的旧值在保存配置时丢弃（运行时已忽略）：
>
> - `provider_settings` 下的 `default_image_caption_provider_id`（Codex 直接读取图片）、`image_compress_options`、`show_tool_use_status`、`show_tool_call_result`、`buffer_intermediate_messages`、`max_quoted_fallback_images`、`file_extract`、`provider_pool`、`persona_pool`；
> - `agent_runner.config` 下旧的 `model` 对象（对话模型提供商、回退模型、重试次数；Codex 的 `model` 是模型名字符串）、`persona`（默认人格改为 `agent_runner.config.persona_id`，健康模式为 `agent_runner.config.safety_mode`）、`compression`（上下文压缩，由 Codex 自行管理上下文）和 `misc`（工具调用轮数、超时、工具调用模式等）；
> - 顶层的 `default_kb_collection`（知识库请使用 `kb_names`）。

#### `provider_stt_settings`

语音转文本服务提供商的通用设置。

#### `provider_stt_settings.enable`

是否启用语音转文本服务。默认为 `false`。

#### `provider_stt_settings.provider_id`

语音转文本服务提供商 ID。必须是 `provider` 列表中已配置的 STT 提供商 ID。

#### `provider_tts_settings`

文本转语音服务提供商的通用设置。

#### `provider_tts_settings.enable`

是否启用文本转语音服务。默认为 `false`。

#### `provider_tts_settings.provider_id`

文本转语音服务提供商 ID。必须是 `provider` 列表中已配置的 TTS 提供商 ID。

#### `provider_tts_settings.dual_output`

是否启用双输出。默认为 `false`。启用后，机器人会同时发送文本和语音消息。

#### `provider_tts_settings.use_file_service`

是否启用文件服务。默认为 `false`。启用后，机器人会将输出的语音文件以 HTTP 文件外链的形式提供给消息平台。此配置项依赖于 `callback_api_base` 的配置。

#### `provider_ltm_settings`

群聊上下文感知服务提供商的通用设置。

#### `provider_ltm_settings.group_icl_enable`

是否将群聊记录注入模型上下文。默认为 `false`。启用后，机器人会暂存群聊中的对话内容，并在下一次回复时注入模型上下文。

上下文的内容会被放在对话的系统提示词中。

#### `provider_ltm_settings.group_message_max_cnt`

注入上下文所保留的最大群聊消息数量。默认为 `1000`。超过此数量的消息将被丢弃。仅在群聊记录注入上下文开启时生效。

#### `provider_ltm_settings.image_caption`

是否自动使用群聊图片转述模型生成图片描述并注入上下文。默认为 `false`。仅在群聊记录注入上下文开启时生效。请谨慎使用，因为这可能会增加大量的 API 调用和 token 开销。

#### `provider_ltm_settings.active_reply`

- `enable`: 是否启用主动回复。默认为 `false`。
- `method`: 主动回复的方法。可选值为 `possibility_reply`。
- `possibility_reply`: 主动回复的概率。默认为 `0.1`。仅在 `method` 为 `possibility_reply` 时适用。
- `whitelist`: 主动回复的 ID 白名单。仅在此列表中的 ID 才会触发主动回复。为空时表示不启用白名单过滤。可以使用 `/sid` 指令获取在某个平台上的会话 ID。

### `agent_runner`

Agent 执行器。Codex 是唯一的执行器（`runner_type` 固定为 `codex`），在 WebUI 的 `配置文件` → `AI` 中设置。常用字段：

- `config.persona_id`：对话没有选定人格时使用的默认人格，默认为 `default`。会话管理中设置的人格和权限规则绑定的人格优先。
- `config.safety_mode`：是否注入健康模式提示词，默认为 `false`。
- `config.proxy`：Codex 的代理，支持 `socks5://`、`socks5h://`、`http://`。填写后 Codex 的所有网络请求都走它，优先于 [`http_proxy`](#http_proxy)；留空时 Codex 使用 `http_proxy` 等代理环境变量（`http_proxy` 配置项也写入这些变量），都没有时直连。
- `config.model`、`config.model_provider`、`config.reasoning_effort`、`config.base_instructions`、`config.developer_instructions`：模型、模型提供商、推理强度和提示词，在 WebUI 的 `Codex` 页面设置。

### `content_safety`

内容安全设置。

#### `content_safety.also_use_in_response`

是否在 LLM 回复中也进行内容安全检查。默认为 `false`。启用后，机器人生成的回复也会经过内容安全检查，以防止生成不当内容。

#### `content_safety.internal_keywords`

内部关键词检测设置。

- `enable`: 是否启用内部关键词检测。默认为 `true`。
- `extra_keywords`: 额外的关键词列表，支持正则表达式。默认为空。

#### `content_safety.baidu_aip`

百度 AI 内容审核设置。

- `enable`: 是否启用百度 AI 内容审核。默认为 `false`。
- `app_id`: 百度 AI 内容审核的 App ID。
- `api_key`: 百度 AI 内容审核的 API Key。
- `secret_key`: 百度 AI 内容审核的 Secret Key。

> [!TIP]
> 如果要启用百度 AI 内容审核，请先 `pip install baidu-aip`。

### `admins_id`

管理员 ID 列表。此外，还可以使用 `/op`, `/deop` 指令来添加或删除管理员。

在 WebUI 的 `配置文件` 中选择机器人使用的配置文件，再进入 `访问控制 → 管理员 → 管理员 ID` 设置。给机器人发送 `/sid` 指令以获取 `UID`。

### `t2i`

是否启用文本转图像功能。默认为 `false`。启用后，当用户发送的消息超过一定字数时，机器人会将消息渲染成图片发送给用户，以提高可读性并防止刷屏。支持 Markdown 渲染。

### `t2i_word_threshold`

文本转图像的字数阈值。默认为 `150`。当用户发送的消息超过此字数时，机器人会将消息渲染成图片发送给用户。

### `t2i_strategy`

文本转图像的渲染策略。可选值为 `local` 和 `remote`。默认为 `remote`。

- `local`: 使用 AstrBot 本地的文本转图像服务进行渲染。效果较差，但不依赖外部服务。
- `remote`: 使用远程的文本转图像服务进行渲染。默认使用 AstrBot 官方提供的服务，效果较好。

### `t2i_endpoint`

AstrBot API 的地址。用于渲染 Markdown 图片。当 `t2i_strategy` 为 `remote` 时生效。默认为空，表示使用 AstrBot 官方提供的服务。

### `t2i_use_file_service`

是否启用文件服务。默认为 `false`。启用后，机器人会将渲染的图片以 HTTP 文件外链的形式提供给消息平台。此配置项依赖于 `callback_api_base` 的配置。

### `http_proxy`

HTTP 代理。如 `http://localhost:7890`。Docker 部署时请填写 AstrBot 容器能访问到的地址，见 [Docker 部署](/deploy/astrbot/docker.md)。

AstrBot 以环境变量的方式设置这个代理，Codex 也会使用它；如果填写了 Codex 代理（`agent_runner.config.proxy`），Codex 只用后者。

### `no_proxy`

不使用代理的地址列表。如 `["localhost", "127.0.0.1"]`。

### `local_infra`

本地多模态服务（[local-multimodal-infra](https://github.com/xkeyC/local-multimodal-infra)），在 WebUI `模型提供商` → `本地多模态服务` 中设置。

- `enable` / `base_url` / `token` / `timeout`：是否启用、服务的 HTTP 地址（不带 `/v1`）、推理令牌（`Authorization: Bearer`）、单个请求超时秒数。
- `services`：AstrBot 使用的各项服务，每项可单独关闭；`model` 为空时自动选用服务器上第一个已启用的对应模型。
  - `stt` / `tts` / `embedding` / `rerank`：生成提供商 `local_infra/stt`、`local_infra/tts`、`local_infra/embedding`、`local_infra/rerank`。`tts.reference_audio` / `tts.reference_text` 为克隆的参考音频（相对数据目录）及其文本。
  - `moderation`：[内容审核](/use/content-moderation)，`blocked_reply` 拦截提示语，`threshold` 文字阈值（`1 - 安全概率` 高于此值即违规，默认 `0.9`），`nsfw_threshold` 图片阈值（默认 `0.5`），`nsfw_labels` 算作违规的图片档位，`categories` 拦截的文字类别（为空等于全部）。
  - `realtime_voice`：实时语音选择本地后端时连接 `ws://<地址>/v1/realtime`。

旧版的 `content_moderation_*` 与实时语音的 `infra_url` / `infra_token` 会在启动时自动迁移到这里。

### `dashboard`

AstrBot WebUI 配置。

请不要随意修改 `password` 的值。它是随机初始密码经过 `md5` 编码后的值。首次启动时请在日志中获取初始密码，并在控制面板中修改。

- `enable`: 是否启用 AstrBot WebUI。默认为 `true`。
- `username`: AstrBot WebUI 的用户名。
- `password`: AstrBot WebUI 的密码。首次启动会随机生成初始密码（已在日志中打印），这里保存的是该密码的 `md5` 值。请勿直接修改，除非您知道自己在做什么。
- `jwt_secret`: JWT 的密钥。AstrBot 会在初始化时随机生成。请勿修改，除非您知道自己在做什么。
- `host`: AstrBot WebUI 监听的地址。默认为 `0.0.0.0`。
- `port`: AstrBot WebUI 监听的端口。默认为 `6185`。

### `platform`

> 此配置项仅在 `data/cmd_config.json` 中生效，AstrBot 不会读取 `data/config/` 目录下的配置文件中的此项。

已配置的 AstrBot 消息平台适配器的配置列表。

每个平台的 `content_moderation` 设置该平台的内容审核：`enabled`（默认）、`text_only`、`disabled`。

### `platform_specific`

平台特异配置。按平台分类，平台下按功能分组。

#### `platform_specific.<platform>.pre_ack_emoji`

启用后，当请求 LLM 前，AstrBot 会先发送一个预回复的表情以告知用户正在处理请求。此功能目前仅在飞书平台适配器和 Telegram 中生效。

##### lark (飞书)

- `enable`: 是否启用飞书消息预回复表情。默认为 `false`。
- `emojis`: 预回复的表情列表。默认为 `["Typing"]`。表情枚举名参考：[表情文案说明](https://open.feishu.cn/document/server-docs/im-v1/message-reaction/emojis-introduce)

##### telegram

- `enable`: 是否启用 Telegram 消息预回复表情。默认为 `false`。
- `emojis`: 预回复的表情列表。默认为 `["✍️"]`。Telegram 仅支持固定反应集合，参考：[reactions.txt](https://gist.github.com/Soulter/3f22c8e5f9c7e152e967e8bc28c97fc9)

##### discord

- `enable`: 是否启用 Discord 消息预回复表情。默认为 `false`。
- `emojis`: 预回复的表情列表。默认为 `["🤔"]`。Discord反应支持参考：[Discord Reaction FAQ](https://support.discord.com/hc/en-us/articles/12102061808663-Reactions-and-Super-Reactions-FAQ)

### `wake_prefix`

唤醒前缀。默认为 `/`。当消息以 `/` 开头时，AstrBot 会被唤醒。

> [!TIP]
> 如果唤醒的会话不在 ID 白名单中，AstrBot 将不会响应。

### `log_level`

日志级别。默认为 `INFO`。可以设置为 `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`。

### `trace_enable`

是否启用追踪记录。默认为 `false`。启用后，AstrBot 会记录运行追踪信息，可以在管理面板的 `数据与日志 → 追踪` 页面查看。

### `pip_install_arg`

`pip install` 的参数。如 `-i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`。

### `pypi_index_url`

PyPI 镜像源地址。默认为 `https://mirrors.aliyun.com/pypi/simple/`。

### `persona`

*此配置项已经在 v4.0.0 版本之后被废弃。请使用 WebUI 来配置人格。*

已配置的人格列表。每个人格包含 `id`, `name`, `description`, `system_prompt` 四个字段。

该列表只保留给从 v3 升级时的数据迁移读取。

### `timezone`

时区设置。请填写 IANA 时区名称, 如 Asia/Shanghai, 为空时使用系统默认时区。所有时区请查看: [IANA Time Zone Database](https://data.iana.org/time-zones/tzdb-2021a/zone1970.tab)。

### `callback_api_base`

AstrBot API 的基础地址。用于文件服务和插件回调等功能。如 `http://example.com:6185`。默认为空，表示不启用文件服务和插件回调功能。

### `kb_tavily_key`

知识库「从 URL 导入」使用的 Tavily API Key 列表（Tavily Extract 提取网页正文），只读取默认配置文件中的值。联网搜索由 Codex 自己完成（Codex 设置中的「启用 Codex 联网搜索」），不需要这个 Key。

### `plugin_set`

已启用的插件列表。`*` 表示启用所有可用的插件。默认为 `["*"]`。
