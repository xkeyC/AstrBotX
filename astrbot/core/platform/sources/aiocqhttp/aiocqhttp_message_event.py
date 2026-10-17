import asyncio
import os
import pathlib
import re
from collections.abc import AsyncGenerator

from aiocqhttp import CQHttp, Event

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.message_components import (
    At,
    BaseMessageComponent,
    File,
    Image,
    Node,
    Nodes,
    Plain,
    Record,
    Video,
)
from astrbot.api.platform import Group, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.utils.media_utils import file_uri_to_path, is_file_uri

# 协议端报告"文件不存在"的几种说法。命中这些说明它读不到本机路径，
# 而不是发送本身失败。
_MISSING_FILE_HINTS = (
    "enoent",
    "no such file",
    "文件不存在",
    "file not found",
)


def _local_file_path(value: object) -> str | None:
    """把消息段里的 file 字段还原成本机真实存在的路径。

    Args:
        value: file 字段的值，可能是路径、file: URI 或 http(s) 链接。

    Returns:
        本机存在该文件时返回绝对路径，否则返回 None。
    """
    if not isinstance(value, str) or not value:
        return None
    path = file_uri_to_path(value) if is_file_uri(value) else value
    if "://" in path:
        return None
    try:
        return os.path.abspath(path) if os.path.isfile(path) else None
    except OSError:
        return None


def _is_missing_file_error(exc: BaseException) -> bool:
    """判断协议端的报错是不是"它那边看不到这个文件"。"""
    text = f"{exc}".lower()
    return any(hint in text for hint in _MISSING_FILE_HINTS)


#: Seconds a pre-reply reaction may take before the reply goes on without it.
REACTION_TIMEOUT_S = 5


def qq_emoji_id(emoji: str) -> str:
    """The ``emoji_id`` of ``set_msg_emoji_like`` for an emoji setting.

    Digits are a QQ face id; anything else must be one Unicode emoji (an
    optional variation selector U+FE0F aside), sent as the decimal code
    point. NapCat reads ids longer than 3 digits as Unicode emoji, so code
    points below 1000 (``#``, ``©``) and sequences (keycaps, flags, ZWJ
    families, skin tones) cannot be sent and are refused.

    Raises:
        ValueError: The setting is neither a face id nor one sendable emoji.
    """
    emoji = emoji.strip()
    if emoji.isascii() and emoji.isdigit():
        return emoji
    code_points = [ord(char) for char in emoji if char != "️"]
    if len(code_points) != 1 or code_points[0] < 1000:
        raise ValueError(
            f"{emoji!r} is neither a QQ face id nor a single Unicode emoji"
        )
    return str(code_points[0])


class AiocqhttpMessageEvent(AstrMessageEvent):
    def __init__(
        self,
        message_str,
        message_obj,
        platform_meta,
        session_id,
        bot: CQHttp,
    ) -> None:
        super().__init__(message_str, message_obj, platform_meta, session_id)
        self.bot = bot

    async def react(self, emoji: str) -> None:
        """Reacts to this message with an emoji (NapCatQQ ``set_msg_emoji_like``).

        Args:
            emoji: A QQ face id (``"76"``, ``"124"``) or one Unicode emoji,
                sent as its code point. QQ shows reactions in groups only, so
                private messages get none.
        """
        if self.get_message_type() != MessageType.GROUP_MESSAGE:
            return
        # A stalled protocol end must not hold up the reply (the OneBot API
        # timeout is minutes).
        await asyncio.wait_for(
            self.bot.call_action(
                "set_msg_emoji_like",
                message_id=self.message_obj.message_id,
                emoji_id=qq_emoji_id(emoji),
                set=True,
            ),
            REACTION_TIMEOUT_S,
        )

    @staticmethod
    async def _from_segment_to_dict(segment: BaseMessageComponent) -> dict:
        """修复部分字段"""
        if isinstance(segment, Image | Record):
            # For Image and Record segments, we convert them to base64
            bs64 = await segment.convert_to_base64()
            return {
                "type": segment.type.lower(),
                "data": {
                    "file": f"base64://{bs64}",
                },
            }
        if isinstance(segment, File):
            # For File segments, we need to handle the file differently
            d = await segment.to_dict()
            file_val = d.get("data", {}).get("file", "")
            if file_val:
                try:
                    # 使用 pathlib 处理路径，能更好地处理 Windows/Linux 差异
                    path_obj = pathlib.Path(file_val)
                    # 如果是绝对路径且不包含协议头 (://)，则转换为标准的 file: URI
                    if path_obj.is_absolute() and "://" not in file_val:
                        d["data"]["file"] = path_obj.as_uri()
                except Exception:
                    # 如果不是合法路径（例如已经是特定的特殊字符串），则跳过转换
                    pass
            return d
        if isinstance(segment, Video):
            d = await segment.to_dict()
            return d
        # For other segments, we simply convert them to a dict by calling toDict
        return segment.toDict()

    @staticmethod
    async def _parse_onebot_json(message_chain: MessageChain):
        """解析成 OneBot json 格式"""
        ret = []
        for segment in message_chain.chain:
            if isinstance(segment, At):
                # At 组件后插入一个空格，避免与后续文本粘连
                d = await AiocqhttpMessageEvent._from_segment_to_dict(segment)
                ret.append(d)
                ret.append({"type": "text", "data": {"text": " "}})
            elif isinstance(segment, Plain):
                if not segment.text.strip():
                    continue
                d = await AiocqhttpMessageEvent._from_segment_to_dict(segment)
                ret.append(d)
            else:
                d = await AiocqhttpMessageEvent._from_segment_to_dict(segment)
                ret.append(d)
        return ret

    @classmethod
    async def _dispatch_send(
        cls,
        bot: CQHttp,
        event: Event | None,
        is_group: bool,
        session_id: str | None,
        messages: list[dict],
    ) -> None:
        # session_id 必须是纯数字字符串
        session_id_int = (
            int(session_id) if session_id and session_id.isdigit() else None
        )
        routing_params = {}
        if isinstance(event, Event) and event.get("self_id"):
            routing_params["self_id"] = event["self_id"]

        if is_group and isinstance(session_id_int, int):
            await bot.send_group_msg(
                group_id=session_id_int,
                message=messages,
                **routing_params,
            )
        elif not is_group and isinstance(session_id_int, int):
            await bot.send_private_msg(
                user_id=session_id_int,
                message=messages,
                **routing_params,
            )
        elif isinstance(event, Event):  # 最后兜底
            await bot.send(event=event, message=messages)
        else:
            raise ValueError(
                f"无法发送消息：缺少有效的数字 session_id({session_id}) 或 event({event})",
            )

    @classmethod
    async def send_message(
        cls,
        bot: CQHttp,
        message_chain: MessageChain,
        event: Event | None = None,
        is_group: bool = False,
        session_id: str | None = None,
    ) -> None:
        """发送消息至 QQ 协议端（aiocqhttp）。

        Args:
            bot (CQHttp): aiocqhttp 机器人实例
            message_chain (MessageChain): 要发送的消息链
            event (Event | None, optional): aiocqhttp 事件对象.
            is_group (bool, optional): 是否为群消息.
            session_id (str | None, optional): 会话 ID（群号或 QQ 号

        """
        # 转发消息、文件消息不能和普通消息混在一起发送
        send_one_by_one = any(
            isinstance(seg, Node | Nodes | File) for seg in message_chain.chain
        )
        if not send_one_by_one:
            ret = await cls._parse_onebot_json(message_chain)
            if not ret:
                return
            await cls._dispatch_send(bot, event, is_group, session_id, ret)
            return
        for seg in message_chain.chain:
            if isinstance(seg, Node | Nodes):
                # 合并转发消息
                if isinstance(seg, Node):
                    nodes = Nodes([seg])
                    seg = nodes

                payload = await seg.to_dict()

                if is_group:
                    payload["group_id"] = session_id
                    if isinstance(event, Event) and event.get("self_id"):
                        payload["self_id"] = event["self_id"]
                    await bot.call_action("send_group_forward_msg", **payload)
                else:
                    payload["user_id"] = session_id
                    if isinstance(event, Event) and event.get("self_id"):
                        payload["self_id"] = event["self_id"]
                    await bot.call_action("send_private_forward_msg", **payload)
            elif isinstance(seg, File):
                await cls._send_file_segment(bot, event, is_group, session_id, seg)
            else:
                messages = await cls._parse_onebot_json(MessageChain([seg]))
                if not messages:
                    continue
                await cls._dispatch_send(bot, event, is_group, session_id, messages)
                await asyncio.sleep(0.5)

    @classmethod
    async def _send_file_segment(
        cls,
        bot: CQHttp,
        event: Event | None,
        is_group: bool,
        session_id: str | None,
        seg: File,
    ) -> None:
        """发送文件段，并在协议端读不到本机路径时给出可执行的替代方案。

        协议端（NapCat、Lagrange 等）常与 AstrBot 跑在不同的容器或主机上，
        此时本机路径在它那边并不存在，发送会以 ENOENT 失败。这里不做任何
        重发：失败信息会顺着工具返回值回到 agent，所以直接告诉它改用平台
        自己的上传工具。发送超时之类的错误原样抛出，因为消息可能已经送达，
        重试只会变成重复消息。
        """
        d = await cls._from_segment_to_dict(seg)
        try:
            await cls._dispatch_send(bot, event, is_group, session_id, [d])
        except Exception as e:
            local_path = _local_file_path(d.get("data", {}).get("file", ""))
            if not local_path or not _is_missing_file_error(e):
                raise
            name = os.path.basename(local_path)
            logger.warning(
                f"协议端读不到本机文件 {local_path}，"
                "说明它与 AstrBot 不在同一个文件系统上；"
                "请改用平台自带的上传工具，或配置 callback_api_base 走文件服务。"
            )
            raise RuntimeError(
                f"{name} could not be sent: the protocol side cannot read local "
                "paths, so it and AstrBot are not sharing a filesystem. Do not "
                "retry this send. Look for the platform's own file-upload tool "
                "-- search the tool registry under the platform prefix (`qq_` "
                "for QQ) for a file or group-file upload capability -- and use "
                "that instead. If there is none, tell the user the file could "
                "not be sent and that an administrator can set callback_api_base "
                "so files are served over HTTP."
            ) from e

    async def send(self, message: MessageChain) -> None:
        """发送消息"""
        event = getattr(self.message_obj, "raw_message", None)

        is_group = bool(self.get_group_id())
        session_id = self.get_group_id() if is_group else self.get_sender_id()

        await self.send_message(
            bot=self.bot,
            message_chain=message,
            event=event,  # 不强制要求一定是 Event
            is_group=is_group,
            session_id=session_id,
        )
        await super().send(message)

    async def send_streaming(
        self,
        generator: AsyncGenerator,
        use_fallback: bool = False,
    ):
        if not use_fallback:
            buffer = None
            async for chain in generator:
                if not buffer:
                    buffer = chain
                else:
                    buffer.chain.extend(chain.chain)
            if not buffer:
                return None
            buffer.squash_plain()
            await self.send(buffer)
            return await super().send_streaming(generator, use_fallback)

        buffer = ""
        pattern = re.compile(r"[^。？！~…]+[。？！~…]+")

        async for chain in generator:
            if isinstance(chain, MessageChain):
                for comp in chain.chain:
                    if isinstance(comp, Plain):
                        buffer += comp.text
                        if any(p in buffer for p in "。？！~…"):
                            buffer = await self.process_buffer(buffer, pattern)
                    else:
                        await self.send(MessageChain(chain=[comp]))
                        await asyncio.sleep(1.5)  # 限速

        buffer = buffer.strip()
        if buffer:
            await self.send(MessageChain([Plain(buffer)]))
        return await super().send_streaming(generator, use_fallback)

    async def get_group(self, group_id=None, **kwargs):
        """Get OneBot group details while preserving inbound data on failures.

        Args:
            group_id: Optional OneBot group identifier.
            **kwargs: Reserved compatibility arguments.

        Returns:
            Enriched group information, or a basic group when an API is unavailable.
        """
        resolved_group_id = group_id or self.get_group_id()
        if not resolved_group_id:
            return None
        resolved_group_id = str(resolved_group_id)
        api_group_id = (
            int(resolved_group_id) if resolved_group_id.isdigit() else resolved_group_id
        )

        current_group = self.message_obj.group
        group = (
            current_group
            if current_group and current_group.group_id == resolved_group_id
            else Group(group_id=resolved_group_id)
        )

        routing_params = {}
        if getattr(self.message_obj, "self_id", None):
            routing_params["self_id"] = self.message_obj.self_id

        try:
            info = await self.bot.call_action(
                "get_group_info",
                group_id=api_group_id,
                **routing_params,
            )
            if isinstance(info, dict):
                group.group_name = info.get("group_name") or group.group_name
                member_count = info.get("member_count")
                if member_count is not None:
                    try:
                        group.member_count = int(member_count)
                    except (TypeError, ValueError):
                        logger.warning(
                            "[aiocqhttp] Invalid member_count for group %s",
                            resolved_group_id,
                        )
        except Exception as exc:
            logger.warning(
                "[aiocqhttp] Failed to get group information for %s: %s",
                resolved_group_id,
                exc,
            )

        try:
            members = await self.bot.call_action(
                "get_group_member_list",
                group_id=api_group_id,
                **routing_params,
            )
        except Exception as exc:
            logger.warning(
                "[aiocqhttp] Failed to get members for group %s: %s",
                resolved_group_id,
                exc,
            )
            return group
        if not isinstance(members, list):
            return group

        owner_id = None
        admin_ids: list[str] = []
        for member in members:
            if not isinstance(member, dict) or member.get("user_id") is None:
                continue
            if member.get("role") == "owner":
                owner_id = str(member["user_id"])
            if member.get("role") == "admin":
                admin_ids.append(str(member["user_id"]))

        group.group_admins = admin_ids
        group.group_owner = owner_id
        group.members = [
            MessageMember(
                user_id=str(member["user_id"]),
                nickname=member.get("nickname") or member.get("card"),
            )
            for member in members
            if isinstance(member, dict) and member.get("user_id") is not None
        ]
        if group.member_count is None:
            group.member_count = len(group.members)
        return group
