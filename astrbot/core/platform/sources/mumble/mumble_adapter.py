"""Mumble platform adapter: text chat plus full-duplex voice.

The whole server is one group conversation, whichever channel the bot is in,
so an admin can move the bot around freely. Private text messages are
private conversations. Channel text wakes the bot with its platform wake
prefix; private messages and whispers always reach it.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import datetime
import time
from collections import deque
from pathlib import Path
from typing import Any, cast

from astrbot.api import logger
from astrbot.api.event import MessageChain
from astrbot.api.message_components import At, Image, Plain
from astrbot.api.platform import (
    AstrBotMessage,
    Group,
    MessageMember,
    MessageType,
    Platform,
    PlatformMetadata,
)
from astrbot.core import astrbot_config
from astrbot.core.platform.astr_message_event import MessageSesion
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from ...register import register_platform_adapter
from .audio import SpeechDetector
from .client import MumbleClient, TextMessage, User, VoicePacket
from .messages import AudioContext, AudioTarget
from .mumble_event import (
    DEFAULT_IMAGE_MESSAGE_LENGTH,
    DEFAULT_MESSAGE_LENGTH,
    UNLIMITED_LENGTH,
    MumbleMessageEvent,
    chain_to_html,
)
from .text import html_to_text

SERVER_SESSION = "server"
MUTE_COMMANDS = {"闭麦", "mute"}
UNMUTE_COMMANDS = {"开麦", "unmute"}
# A muted bot drops its voice sessions after this long.
MUTE_DISCONNECT_SECONDS = 60.0
MAX_WHISPER_SESSIONS = 3
WATCHDOG_INTERVAL = 5.0
# Voice kept while in standby and replayed into a resumed session, so the
# words that woke it are not lost.
PREROLL_SECONDS = 2.0
# A voice session that failed to start is not retried for this long.
START_RETRY_SECONDS = 30.0
VOICE_BITRATE = 64000
# Servers rate-limit text messages (default burst 5, then 1 per second) and
# silently drop the excess, so long replies are paced after the burst.
TEXT_BURST = 4
TEXT_INTERVAL = 1.1


def user_key(user: User) -> str:
    """Stable id of a Mumble user.

    The certificate hash when there is one (official clients always have
    one), else the registered user id. A user with neither is known only by
    a name anyone can take once they leave, so their id is scoped to the
    current session and no conversation carries over to an impostor.
    """
    if user.hash:
        return user.hash
    if user.user_id is not None:
        return f"uid:{user.user_id}"
    return f"session:{user.session}:{user.name}"


def ensure_certificate(directory: Path, name: str) -> tuple[str, str]:
    """A self-signed client certificate, created on first use.

    A stable certificate lets the server recognise the bot across reconnects,
    so an admin can register it and grant it permissions.

    Args:
        directory: Where the certificate and key are kept.
        name: Common name for a new certificate.

    Returns:
        Paths of the certificate and the private key.
    """
    cert_path, key_path = directory / "client.crt", directory / "client.key"
    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    directory.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name or "AstrBot")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365 * 20))
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return str(cert_path), str(key_path)


@register_platform_adapter(
    "mumble",
    "Mumble 平台适配器（文字 + 实时语音）",
    support_streaming_message=False,
)
class MumblePlatformAdapter(Platform):
    def __init__(
        self,
        platform_config: dict,
        platform_settings: dict,
        event_queue: asyncio.Queue,
    ) -> None:
        super().__init__(platform_config, event_queue)
        self.settings = platform_settings
        cfg = platform_config
        self.host = str(cfg.get("mumble_host", "")).strip()
        if not self.host:
            raise ValueError("Mumble 服务器地址是必需的")
        self.username = str(cfg.get("mumble_username") or "AstrBot").strip()
        self.home_channel = str(cfg.get("mumble_channel") or "").strip()
        self.text_wake_prefix = str(cfg.get("mumble_text_wake_prefix") or "")
        self.reconnect_delay = float(cfg.get("mumble_reconnect_delay", 5.0))
        self.voice_enabled = bool(cfg.get("mumble_voice_enabled", True))
        self.voice_idle_timeout = float(cfg.get("mumble_voice_idle_timeout", 300))

        platform_id = cast(str, cfg.get("id", "mumble"))
        certfile = str(cfg.get("mumble_certfile") or "").strip()
        keyfile = str(cfg.get("mumble_keyfile") or "").strip() or None
        if not certfile:
            directory = Path(get_astrbot_data_path()) / "mumble" / platform_id
            certfile, keyfile = ensure_certificate(directory, self.username)
        self.client = MumbleClient(
            self.host,
            int(cfg.get("mumble_port") or 64738),
            username=self.username,
            password=str(cfg.get("mumble_password") or ""),
            certfile=certfile,
            keyfile=keyfile,
        )
        self.client.on_text = self._on_text
        self.client.on_voice = self._on_voice
        self.client.on_user_changed = self._on_user_changed
        self.client.on_user_removed = self._on_user_removed
        self.client.on_disconnected = self._on_disconnected
        self.metadata = PlatformMetadata(
            name="mumble",
            description="Mumble 平台适配器（文字 + 实时语音）",
            id=platform_id,
            support_streaming_message=False,
        )

        from astrbot.core.voice.session import VoiceOptions

        aliases = cfg.get("mumble_voice_aliases") or []
        self.voice_options = VoiceOptions(
            name=str(cfg.get("mumble_voice_name") or self.username),
            aliases=[str(a) for a in aliases if str(a).strip()],
            extra_prompt=str(cfg.get("mumble_voice_prompt") or ""),
        )
        self.voice_sessions: dict[str, Any] = {}  # key -> VoiceSession
        self.whisper_targets: dict[str, int] = {}  # user key -> voice target id
        self.muted = False
        self.muted_at = 0.0
        # Standby: no voice session; local VAD decides when to resume one.
        self._detector = SpeechDetector()
        self._preroll: dict[str, deque[tuple[float, int, bytes, bool]]] = {}
        self._retry_at: dict[str, float] = {}
        self._tasks: set[asyncio.Task] = set()
        # Bumped on every connection; ids from an older one are stale.
        self._generation = 0
        self._running = True
        self._disconnected = asyncio.Event()

    def meta(self) -> PlatformMetadata:
        return self.metadata

    # -- lifecycle --------------------------------------------------------

    async def run(self) -> None:
        watchdog = asyncio.create_task(self._watchdog(), name="mumble-voice-watchdog")
        try:
            while self._running:
                try:
                    await self._connect_once()
                    await self._disconnected.wait()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if not self._running:
                        break
                    logger.warning(
                        "Mumble connection to %s failed: %s. Retrying in %.1fs.",
                        self.host,
                        exc,
                        self.reconnect_delay,
                    )
                await self._close_voice("disconnected")
                if self._running:
                    await asyncio.sleep(self.reconnect_delay)
        finally:
            watchdog.cancel()

    async def _connect_once(self) -> None:
        self._disconnected.clear()
        self._generation += 1
        await self.client.connect()
        me = self.client.me
        logger.info(
            "Mumble connected to %s as %s (session %s, server %s)",
            self.host,
            self.username,
            self.client.session,
            self.client.server_release,
        )
        if self.home_channel and me is not None:
            channel = self.client.find_channel(self.home_channel)
            if channel is None:
                logger.warning("Mumble channel %r not found", self.home_channel)
            elif channel.channel_id != me.channel_id:
                self.client.join_channel(channel.channel_id)
        if self.muted:
            self.client.set_self_state(self_mute=True, self_deaf=True)
        # Sessions ids from before the reconnect may now belong to others.
        self.whisper_targets.clear()
        self._detector.clear()
        self._preroll.clear()
        self._retry_at.clear()

    def _on_disconnected(self, error: Exception | None) -> None:
        logger.warning("Mumble disconnected: %s", error or "connection closed")
        self._disconnected.set()

    async def terminate(self) -> None:
        self._running = False
        await self._close_voice("terminated")
        await self.client.close()
        self._disconnected.set()

    def get_client(self) -> MumbleClient:
        return self.client

    # -- text -------------------------------------------------------------

    def _on_text(self, message: TextMessage) -> None:
        sender = self.client.users.get(message.actor or -1)
        if sender is None or sender.session == self.client.session:
            return
        parsed = html_to_text(message.message)
        text = parsed.text
        private = message.is_private
        woken = False
        if (
            not private
            and self.text_wake_prefix
            and text.startswith(self.text_wake_prefix)
        ):
            text = text[len(self.text_wake_prefix) :].lstrip()
            woken = True

        command = text.strip()
        for prefix in astrbot_config.get("wake_prefix", []):
            if prefix and command.startswith(prefix):
                command = command[len(prefix) :].strip()
                break
        if (private or woken or command != text.strip()) and command.lower() in (
            MUTE_COMMANDS | UNMUTE_COMMANDS
        ):
            self._spawn(
                self._set_muted(command.lower() in MUTE_COMMANDS, sender, private)
            )
            return

        components: list[Any] = []
        if woken:
            components.append(At(qq=str(self.client.session), name=self.username))
        if text:
            components.append(Plain(text))
        for image in parsed.images:
            components.append(Image.fromBase64(base64.b64encode(image.data).decode()))
        if not components or (woken and len(components) == 1):
            return

        abm = AstrBotMessage()
        abm.self_id = str(self.client.session)
        abm.sender = MessageMember(user_id=user_key(sender), nickname=sender.name)
        abm.message = components
        abm.message_str = text
        abm.raw_message = message
        abm.message_id = f"{sender.session}-{time.time_ns()}"
        if private:
            abm.type = MessageType.FRIEND_MESSAGE
            abm.session_id = user_key(sender)
        else:
            abm.type = MessageType.GROUP_MESSAGE
            abm.session_id = SERVER_SESSION
            abm.group = Group(group_id=SERVER_SESSION, group_name=self._server_name())
        self.commit_event(
            MumbleMessageEvent(
                message_str=abm.message_str,
                message_obj=abm,
                platform_meta=self.meta(),
                session_id=abm.session_id,
                adapter=self,
            )
        )

    def _server_name(self) -> str:
        root = self.client.channels.get(0)
        return root.name if root and root.name else self.host

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def send_chain(
        self,
        session_id: str,
        chain: MessageChain,
        origin: TextMessage | None = None,
    ) -> None:
        """Sends a chain to the server conversation or a user's private one.

        Args:
            session_id: ``server`` or a user key.
            chain: What to send.
            origin: The message being answered. A channel reply goes where it
                was sent (and to the sender's channel), not just to the
                channel the bot is in.
        """
        if not self.client.connected:
            logger.warning("Mumble: not connected, dropping outgoing message")
            return
        config = self.client.server_config
        # 0 means no limit on the server.
        message_length = config.get("message_length", DEFAULT_MESSAGE_LENGTH)
        image_length = config.get("image_message_length", DEFAULT_IMAGE_MESSAGE_LENGTH)
        messages = await chain_to_html(
            chain,
            int(message_length) or UNLIMITED_LENGTH,
            int(image_length) or UNLIMITED_LENGTH,
        )
        if session_id == SERVER_SESSION:
            # The server drops the whole message if any target channel is
            # gone or not writable, so reply where the message was sent (the
            # sender chose it) while those channels exist, else to the bot's.
            existing = self.client.channels
            channels = [
                c for c in (origin.channel_ids if origin else []) if c in existing
            ]
            trees = [c for c in (origin.tree_ids if origin else []) if c in existing]
            if not channels and not trees:
                me = self.client.me
                channels = [me.channel_id if me else 0]
            target: dict[str, Any] = {"channel_ids": channels}
            if trees:
                target["tree_ids"] = trees
        else:
            user = next(
                (u for u in self.client.users.values() if user_key(u) == session_id),
                None,
            )
            if user is None:
                logger.warning(
                    "Mumble: user %s is offline, message dropped", session_id
                )
                return
            target = {"sessions": [user.session]}
        generation = self._generation
        for index, html_message in enumerate(messages):
            if index >= TEXT_BURST:
                await asyncio.sleep(TEXT_INTERVAL)
                # After a reconnect the target ids may belong to someone else.
                if not self.client.connected or self._generation != generation:
                    return
            self.client.send_text(html_message, **target)
            await self.client.drain()

    async def send_by_session(
        self, session: MessageSesion, message_chain: MessageChain
    ) -> None:
        await self.send_chain(session.session_id, message_chain)
        await super().send_by_session(session, message_chain)

    async def _set_muted(self, muted: bool, sender: User, private: bool) -> None:
        self.muted = muted
        self.muted_at = time.monotonic()
        for session in self.voice_sessions.values():
            if session.closing:
                continue  # stays muted until released
            session.media.outbound.muted = muted
            if muted:
                session.media.mixer.clear()
        self._detector.clear()
        self._preroll.clear()
        with contextlib.suppress(Exception):
            self.client.set_self_state(self_mute=muted, self_deaf=muted)
        note = (
            "已闭麦，1 分钟后断开语音会话；发送“开麦”恢复。"
            if muted
            else "已开麦，有人说话时我会重新接入语音。"
        )
        logger.info("Mumble: %s by %s", "muted" if muted else "unmuted", sender.name)
        await self.send_chain(
            user_key(sender) if private else SERVER_SESSION,
            MessageChain([Plain(note)]),
        )

    # -- voice ------------------------------------------------------------

    def _on_voice(self, packet: VoicePacket) -> None:
        if not self.voice_enabled or self.muted:
            return
        user = self.client.users.get(packet.sender_session)
        if user is None or user.session == self.client.session:
            return
        if packet.context == AudioContext.WHISPER:
            key = f"whisper:{user_key(user)}"
        elif packet.context in (AudioContext.NORMAL, AudioContext.SHOUT):
            key = SERVER_SESSION
        else:
            return  # channel listeners: not our conversation
        session = self.voice_sessions.get(key)
        if session is not None:
            session.media.mixer.feed(
                user.session, packet.opus_data, packet.is_terminator
            )
            return
        # Standby: keep a short pre-roll and wait for real speech.
        now = time.monotonic()
        preroll = self._preroll.setdefault(key, deque())
        preroll.append((now, user.session, packet.opus_data, packet.is_terminator))
        while preroll and now - preroll[0][0] > PREROLL_SECONDS:
            preroll.popleft()
        if now < self._retry_at.get(key, 0.0):
            return
        if not self._detector.feed(
            user.session, packet.opus_data, packet.is_terminator
        ):
            return
        session = self._start_voice(key, user)
        if session is None:
            return
        for _, speaker, data, terminator in self._preroll.pop(key, ()):
            session.media.mixer.feed(speaker, data, terminator)

    def _start_voice(self, key: str, user: User):
        from astrbot.core.voice.chat import VoiceChat
        from astrbot.core.voice.session import new_voice_session, realtime_voice_config

        from .audio import MumbleMedia
        from .voice import channel_prompt, whisper_prompt

        if key == SERVER_SESSION:
            # The voice server passes on only what calls the bot (or all, to
            # one other person): local_infra.
            gated = realtime_voice_config()["backend"] == "local_infra"
            prompt = channel_prompt(self.voice_options, gated=gated)

            def send(frame: bytes, terminator: bool) -> None:
                self._send_voice(frame, AudioTarget.NORMAL, terminator)

        else:
            whispers = [k for k in self.voice_sessions if k != SERVER_SESSION]
            if len(whispers) >= MAX_WHISPER_SESSIONS:
                return None
            prompt = whisper_prompt(self.voice_options, user.name)
            target = self._whisper_target(user)
            if target is None:
                return None

            def send(frame: bytes, terminator: bool) -> None:
                self._send_voice(frame, target, terminator)

        # The realtime backend is a setting of the Codex runner.
        session = new_voice_session(
            key=key,
            scope_id=f"{self.meta().id}:voice:{key}",
            prompt=prompt,
            options=self.voice_options,
            media=MumbleMedia(send, self._voice_bitrate()),
            on_closed=self._voice_closed,
            label="Mumble",
            thread_key="mumble_voice_thread",
            # The chat that runs what is asked by voice: the server group (as
            # the fixed voice user, a member: speakers cannot be told apart),
            # or the whisperer's private chat (as the whisperer).
            chat=(
                VoiceChat(
                    umo=f"{self.meta().id}:{MessageType.GROUP_MESSAGE.value}:{SERVER_SESSION}",
                    private=False,
                    via="Mumble voice channel",
                )
                if key == SERVER_SESSION
                else VoiceChat(
                    umo=f"{self.meta().id}:{MessageType.FRIEND_MESSAGE.value}:{user_key(user)}",
                    private=True,
                    sender_id=user_key(user),
                    sender_name=user.name,
                    via="Mumble whisper",
                )
            ),
        )
        self.voice_sessions[key] = session
        if key == SERVER_SESSION:
            self._spawn(session.set_people(self._channel_others()))

        def failed(exc: Exception) -> None:
            logger.error("Mumble voice session %s failed to start: %s", key, exc)
            self._retry_at[key] = time.monotonic() + START_RETRY_SECONDS

        session.launch(failed)
        return session

    def _voice_bitrate(self) -> int:
        """Opus bitrate for the bot's voice: 64 kbit/s unless the server's
        per-user bandwidth (which counts the TCP tunnel's overhead, about
        20 kbit/s at 50 packets a second) allows less; never below 32."""
        limit = self.client.max_bandwidth
        if not limit:
            return VOICE_BITRATE
        return max(32000, min(VOICE_BITRATE, limit - 20000))

    def _whisper_target(self, user: User) -> int | None:
        """The voice target (1-30) aimed at ``user``, or None if all are used."""
        key = user_key(user)
        target = self.whisper_targets.get(key)
        if target is None:
            used = set(self.whisper_targets.values())
            target = next((i for i in range(1, 31) if i not in used), None)
            if target is None:
                return None
            self.whisper_targets[key] = target
        self.client.set_voice_target(target, sessions=[user.session])
        return target

    def _send_voice(self, frame: bytes, target: int, terminator: bool) -> None:
        if self.client.connected:
            with contextlib.suppress(Exception):
                self.client.send_audio(frame, target=target, is_terminator=terminator)

    def _voice_closed(self, session) -> None:
        if self.voice_sessions.get(session.key) is session:
            del self.voice_sessions[session.key]
        if session.key.startswith("whisper:"):
            self.whisper_targets.pop(session.key.removeprefix("whisper:"), None)

    def _channel_others(self) -> int | None:
        """How many others are in the bot's channel (None: not known yet)."""
        me = self.client.me
        if me is None:
            return None
        return sum(
            1
            for user in self.client.users.values()
            if user.channel_id == me.channel_id and user.session != me.session
        )

    def _people_changed(self) -> None:
        """Someone came, went or moved (the bot too): the channel's voice
        session hears accordingly (everything with one other person, else
        only what calls the bot by name)."""
        session = self.voice_sessions.get(SERVER_SESSION)
        if session is not None:
            self._spawn(session.set_people(self._channel_others()))

    def _on_user_removed(self, user: User, _message: dict) -> None:
        """Drops a departed speaker's audio state; their session id is reused."""
        self._people_changed()
        self._detector.forget(user.session)
        self._preroll.pop(f"whisper:{user_key(user)}", None)
        channel = self._preroll.get(SERVER_SESSION)
        if channel:
            self._preroll[SERVER_SESSION] = deque(
                item for item in channel if item[1] != user.session
            )
        for session in self.voice_sessions.values():
            session.media.mixer.forget(user.session)
        # Their whisper session would keep answering a session id the server
        # may hand to someone else.
        whisper = self.voice_sessions.get(f"whisper:{user_key(user)}")
        if whisper is not None:
            self._spawn(whisper.close("whisper partner left"))

    def _on_user_changed(self, user: User, changed: set[str]) -> None:
        # Someone new (their name comes first), or someone (the bot too)
        # changed channels.
        if changed & {"channel_id", "name"}:
            self._people_changed()
        # A whisper partner who reconnected has a new session: re-aim the target.
        target = self.whisper_targets.get(user_key(user))
        if target is not None and "name" in changed and self.client.connected:
            with contextlib.suppress(Exception):
                self.client.set_voice_target(target, sessions=[user.session])

    async def _close_voice(self, reason: str) -> None:
        for session in list(self.voice_sessions.values()):
            await session.close(reason)

    async def _watchdog(self) -> None:
        while True:
            await asyncio.sleep(WATCHDOG_INTERVAL)
            now = time.monotonic()
            if self.muted and now - self.muted_at >= MUTE_DISCONNECT_SECONDS:
                await self._close_voice("muted")
                continue
            for session in list(self.voice_sessions.values()):
                # A session still starting has its own timeouts.
                if (
                    session.ready
                    and now - session.last_activity >= self.voice_idle_timeout
                ):
                    await session.close("standby: no speech recognised")
