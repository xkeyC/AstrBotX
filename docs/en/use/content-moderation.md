# Content Moderation

When enabled, AstrBot checks what it hands to the cloud model (Codex) with a service compatible with OpenAI's moderation API (`POST /v1/moderations`), such as the local moderation service of [local-multimodal-infra](https://github.com/xkeyC/local-multimodal-infra) (the Qwen3Guard text model, an NSFW image classifier, and text recognition in images). Flagged content is not sent to the model, which keeps the model provider from warning or banning the account.

## Turning it on

1. In `Settings` → `Security` → `Content Moderation`, fill in **Content Moderation Service URL**, e.g. `http://127.0.0.1:17890`. Empty means no moderation. If the service requires a token, also fill in **Content Moderation Service Token** (sent as `Authorization: Bearer <token>`).
2. In `Platforms`, edit each bot and set **Content Moderation**:
   - `On`: text and images (default);
   - `Text only`: text is checked, images are not;
   - `Off`: nothing is checked.

## What is checked

| Content | When flagged |
| --- | --- |
| The user's message: text, images, images sent as files, the quoted message | The whole message is not given to the model; the bot replies with the Reply When Blocked setting (default "这条内容无法处理。") |
| Group chat context (earlier group messages) | Only the flagged messages are left out; the rest still goes |
| Group messages read by the `get_group_message_history` tool | Same: only the flagged messages are left out |
| Tool results: file reads, image reads, web / MCP / plugin tools, sandbox output, results of background tasks and commands | Only the flagged text parts or images are removed, and the model is told what was removed and why |
| A plugin's own model calls, group image captions | The call fails (the plugin gets an error result); no caption is made |

Context AstrBot adds itself (sender metadata, retrieved knowledge) is not the user's message and is not checked. Logs record only the categories, never the content.

### Quoted messages

A quoted message counts as part of the user's message and is checked and judged together with what the user wrote:

- the quoted text (the `<Quoted Message>` block) is checked as text;
- images in the quote are checked like the user's own images (not on `Text only` platforms);
- if the quoted content is flagged, the whole message is blocked, even when the user's own words are fine.

In the group chat context, the quote summary inside each earlier message (`[Quote(...)]`, at most 200 characters) is checked with that message; if flagged, only that message is left out.

## Rules and settings

The service gives text the probabilities of safe / controversial / unsafe, and images the NSFW probability of each level. These settings are in `Settings` → `Security` → `Content Moderation`, are sent with every check and apply at once:

| Setting | Default | Meaning |
| --- | --- | --- |
| Text Moderation Threshold | `0.9` | Text is flagged when its unsafe probability (`1 - p(safe)`, controversial included) is above it. Lower is stricter: `0.5` catches ~97% but flags ~10% of harmless text |
| Image Moderation Threshold | `0.5` | An image is flagged when the probabilities of the selected levels add up to more than it |
| Image Categories That Count | Suggestive, Medium, Explicit | Which levels count. Untick Suggestive to stop flagging swimwear or lingerie and flag only explicit images |
| Text Categories to Block | All | Violent, non-violent illegal acts, sexual, privacy, self-harm, unethical, politically sensitive, copyright, jailbreak. Text is blocked only when judged unsafe for a selected category; text judged unsafe without a category is still blocked |

Text read in images follows the text rules (threshold and categories); an image flagged as NSFW is not read.

## When the service is unavailable

- Text checks time out after 5 s, checks with images after 15 s.
- Service unavailable (timeout, 5xx): user messages with images are refused, text-only ones go through and are logged; images in tool results are withheld, text goes through.
- Input rejected (400, e.g. an unsupported image type): a user message counts as flagged; in a tool result, the rejected part is withheld.

## Long text

The service checks one request at a time, and a check costs about its tokens. So that one long text does not hold up other checks, AstrBot checks text in pieces of at most 16 KB (each overlapping the previous by 512 bytes); a text is flagged when any piece is. At most 64 KB of a tool result's text is checked; the rest is cut off and the model is told.
