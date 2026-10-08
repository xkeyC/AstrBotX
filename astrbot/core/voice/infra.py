"""Voice through a local-multimodal-infra server, with Codex doing the talking.

With ``realtime_voice.backend = "local_infra"`` Codex connects the voice
thread to the server's ``/v1/realtime`` in audio mode (Codex's
``[realtime] backend = "local_multimodal_infra"``): the server listens and
speaks (VAD, ASR, barge-in, TTS), and every utterance that wants a reply is a
turn of the voice thread, on the model chosen for it (``text_*``). This
session carries the audio both ways through Codex and runs what the voice
model hands off (its one tool, ``backend_task``) as a turn of the paired
chat, like ``VoiceSession`` does; the answer goes back to the voice thread,
which tells it.

The voice thread is kept per conversation key, like the realtime one: its
history (compacted while idle) carries over from call to call.
"""

from __future__ import annotations

import asyncio
import base64
import bisect
import fractions
import inspect
import json
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import av
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError

from astrbot import logger
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from . import session as voice
from .chat import TASK_BODY
from .session import (
    DONE_SPEECH,
    FAILED_SPEECH,
    VOICE_THREAD_CONFIG,
    VoiceSession,
    realtime_voice_config,
    time_prompt,
)

IN_RATE = 16000  # what Codex hands the server: 16-bit mono PCM
# A session loads every model on the server before it starts (IndexTTS takes
# tens of seconds the first time).
START_TIMEOUT = 320.0  # a little over Codex's own, whose error says why
# The reference voice goes in one WebSocket message (the server takes 16 MB).
MAX_REF_AUDIO_BYTES = 8 * 1024 * 1024
# With its transcript, Qwen3-TTS takes a reference of at most this long
# (in-context cloning; the server's max_reference_seconds default).
MAX_REF_SECONDS_WITH_TEXT = 15
# Emotions the server's TTS (IndexTTS-2.5) speaks with; "none" keeps the
# reference voice's own.
EMOTIONS = (
    "calm",
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "none",
)
# After the server's speech stops, this much silence (20 ms frames) follows,
# so the platform's player starts and ends even a very short reply. Speech has
# stopped when nothing arrived for TAIL_GAP: well over the server's 20 ms
# frame period and its jitter, well under its 300 ms lead.
TAIL_FRAMES = 15
# A platform tool's longest run: the voice turn waits for it.
VOICE_TOOL_TIMEOUT = 30.0
TAIL_GAP = 0.2
# Where the audio handed to the server came from, kept this long (in frames
# as the platform's media serves them, 20 ms): a speaker label for older
# audio is dropped (the server keeps about a minute of them too).
POSITION_FRAMES = 3000
# Speaker labels go to the server at most this often (the platform sends
# one segment's growing span every ~250 ms; only the newest counts).
LABEL_INTERVAL = 0.2
# A label passes on at most this many candidates (who it may be, and how
# likely).
LABEL_CANDIDATES = 3
# on_wake is called at most this often (seconds): a call's wake word and
# its transcript, or the next utterance's wake heard before this one's
# transcript, are one turn toward the speaker.
WAKE_DEBOUNCE = 3.0
# A wake word heard this recently (seconds, and not yet matched by a called
# transcript) is the one a called transcript was spotted by: that call was
# already turned to. A called transcript without one was called by name in
# its text (the spotter missed it): it turns.
WAKE_COVERS = 30.0

# How the voice thread's model takes part, ahead of the platform's prompt
# (whose "delegate to the backend" and "stay silent" these define).
INFRA_INSTRUCTIONS = """You are the voice in a live voice conversation. What people say reaches you as transcripts (speech recognition may mishear words); everything you write is spoken aloud by speech synthesis.

- Speak naturally and briefly, usually one to three short sentences, in the speaker's language. Never use Markdown, lists, links, code or emoji.
- To stay silent, call stay_silent and write nothing.
- To delegate to the backend, call backend_task with the whole task in one sentence, then say in a few words that you are on it. The result comes back later as a message; tell it then, briefly and in your own words.
- Text in parentheses comes from the system, not from anyone speaking."""
START_INSTRUCTIONS = "A voice conversation has started."
# Codex reads {now} as the time of the hang-up.
END_INSTRUCTIONS = "The voice conversation ended at {now}; nothing said now is heard."
# A handed-off task's answer, for the voice thread to tell.
RESULT_PROMPT = """(The backend finished "{task}": {answer}
Tell the listener briefly, in your own words.)"""
# Ends the turn without a word (instead of a "<silence>" reply, which Codex
# still takes as silence).
STAY_SILENT_TOOL = {
    "type": "function",
    "name": "stay_silent",
    "description": "Says nothing: call it instead of replying whenever you stay silent.",
    "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
}
BACKEND_TASK_TOOL = {
    "type": "function",
    "name": "backend_task",
    "description": (
        "Hands a task to the backend: anything needing current information "
        "(time, date, weather, news, prices), searching or looking things up, "
        "or doing something (messages, reminders, devices, remembering). The "
        "result comes back later as a message."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The whole task in one sentence, with every detail it needs.",
            }
        },
        "required": ["task"],
        "additionalProperties": False,
    },
}


def infra_settings_error(settings: dict) -> str | None:
    """What is wrong with the ``local_infra`` realtime voice settings.

    Args:
        settings: The runner's ``realtime_voice``.

    Returns:
        The problem, or None when they are usable.
    """
    parts = urlsplit(str(settings["infra_url"]))
    if parts.scheme not in ("ws", "wss") or not parts.netloc:
        return (
            f"voice server URL must be ws:// or wss://, got {settings['infra_url']!r}"
        )
    if (emotion := str(settings["emotion"])) and emotion not in EMOTIONS:
        return f"voice emotion must be one of {', '.join(EMOTIONS)}, got {emotion!r}"
    if not 0 <= float(settings["emotion_strength"]) <= 1:
        return (
            f"voice emotion strength must be 0 to 1, got {settings['emotion_strength']}"
        )
    if not 0 <= int(settings["idle_compact_percent"]) <= 100:
        return f"idle compaction must be 0 to 100 percent, got {settings['idle_compact_percent']}"
    if path := ref_audio_path(settings):
        if not path.is_file() or path.suffix.lower() != ".wav":
            return f"reference audio must be an existing .wav file: {path}"
        if path.stat().st_size > MAX_REF_AUDIO_BYTES:
            return f"reference audio is over {MAX_REF_AUDIO_BYTES // 2**20} MB: {path}"
        seconds = wav_seconds(path)
        if (
            str(settings["ref_text"]).strip()
            and seconds is not None
            and seconds > MAX_REF_SECONDS_WITH_TEXT
        ):
            return (
                f"with its transcript the reference audio may be at most "
                f"{MAX_REF_SECONDS_WITH_TEXT} s, {path} is {seconds:.1f} s"
            )
    return None


def wav_seconds(path: Path) -> float | None:
    """How long a WAV file plays (None when ``wave`` cannot read it)."""
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    except (wave.Error, EOFError, ZeroDivisionError):
        return None


def ref_audio_path(settings: dict) -> Path | None:
    """The reference voice file (relative paths are in the data directory)."""
    if not (ref := str(settings["ref_audio"]).strip()):
        return None
    path = Path(ref)
    return path if path.is_absolute() else Path(get_astrbot_data_path()) / path


class SpeechTrack(MediaStreamTrack):
    """The server's speech (already at real-time pace) as a track."""

    kind = "audio"

    def __init__(self, rate: int = 24000) -> None:
        """Creates an empty track.

        Args:
            rate: Sample rate of the server's 16-bit mono PCM.
        """
        super().__init__()
        self.rate = rate
        self._queue: asyncio.Queue[av.AudioFrame | None] = asyncio.Queue()
        self._pts = 0
        self._tail = 0
        self._odd = b""  # a byte of a sample split between two messages
        self._ended = False

    def _frame(self, pcm: bytes) -> av.AudioFrame:
        frame = av.AudioFrame(format="s16", layout="mono", samples=len(pcm) // 2)
        frame.planes[0].update(pcm)
        frame.sample_rate = self.rate
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, self.rate)
        self._pts += frame.samples
        return frame

    def put(self, pcm: bytes) -> None:
        """Queues server speech.

        Args:
            pcm: 16-bit mono PCM; a trailing odd byte goes before the next.
        """
        pcm = self._odd + pcm
        cut = len(pcm) - len(pcm) % 2
        pcm, self._odd = pcm[:cut], pcm[cut:]
        if pcm:
            self._queue.put_nowait(self._frame(pcm))
            self._tail = TAIL_FRAMES

    def clear(self) -> None:
        """Drops what the player has not taken yet."""
        while not self._queue.empty():
            self._queue.get_nowait()
        self._tail = 0

    def end(self) -> None:
        self._ended = True
        self._queue.put_nowait(None)

    async def recv(self) -> av.AudioFrame:
        """The next frame of speech, or of the silence after it."""
        if self._ended:
            raise MediaStreamError
        while True:
            if self._tail > 0 and self._queue.empty():
                # The first silent frame only once speech has clearly
                # stopped, then one per frame period.
                wait = TAIL_GAP if self._tail == TAIL_FRAMES else 0.02
                try:
                    frame = await asyncio.wait_for(self._queue.get(), wait)
                except asyncio.TimeoutError:
                    self._tail = max(self._tail - 1, 0)
                    return self._frame(bytes(2 * (self.rate // 50)))
            else:
                frame = await self._queue.get()
            if frame is None:
                raise MediaStreamError
            return frame


def tool_content(result) -> list[dict]:
    """A platform tool's result as Codex content items: a text, or well-formed
    ``inputText`` / ``inputImage`` items as they are; anything else (a
    ``run`` returning nothing or a mapping) as text, so Codex always gets an
    answer it can read (a malformed one would leave the voice turn waiting).
    """
    if isinstance(result, str):
        return [{"type": "inputText", "text": result}]
    if (
        isinstance(result, list)
        and result
        and all(
            isinstance(item, dict)
            and (
                (item.get("type") == "inputText" and isinstance(item.get("text"), str))
                or (
                    item.get("type") == "inputImage"
                    and isinstance(item.get("imageUrl"), str)
                )
            )
            for item in result
        )
    ):
        return [
            {key: item[key] for key in ("type", "text", "imageUrl") if key in item}
            for item in result
        ]
    return [{"type": "inputText", "text": "Done." if result is None else str(result)}]


@dataclass
class SpeakerLabel:
    """A platform's guess of who spoke a stretch of its audio, and what of
    it the server has been told."""

    name: str | None
    start: int
    end: int
    final: bool
    # How likely each is, likeliest first: (name, or None for someone not
    # recognised; p). Empty: the platform gave none.
    candidates: tuple[tuple[str | None, float], ...] = ()
    # The server's sample index of ``start``, fixed once known (the server
    # keeps one span per start).
    given_start: int | None = None
    # What the server was last told: (name, start, end, final,
    # candidates), server samples.
    given: tuple | None = None


class InfraVoiceSession(VoiceSession):
    """A voice conversation on a local-multimodal-infra server, the voice
    thread's model doing the talking."""

    def __init__(self, *args, **kwargs) -> None:
        """Creates the session (it starts with ``launch``).

        Args:
            *args, **kwargs: As for ``VoiceSession``. The ``thread_key`` gets
                an ``_infra`` suffix: this thread talks itself, unlike the
                realtime one.
        """
        super().__init__(*args, **kwargs)
        # A platform tool named as a built-in one is left out.
        reserved = {BACKEND_TASK_TOOL["name"], STAY_SILENT_TOOL["name"]}
        if dropped := [
            t.spec.get("name") for t in self.tools if t.spec.get("name") in reserved
        ]:
            logger.warning(
                "%s voice %s: platform tools %s left out (built-in names)",
                self.label,
                self.key,
                ", ".join(dropped),
            )
            self.tools = [t for t in self.tools if t.spec.get("name") not in reserved]
        self.thread_key = f"{self.thread_key}_infra"
        self._track = SpeechTrack()
        # The last utterance heard (what a handed-off task was asked with),
        # and who the server says said it (a guess: see label_speaker).
        self._heard = ""
        self._speaker: str | None = None
        # Where the audio handed to the server came from, frame by frame:
        # (platform position, its end, server sample index, its end), the
        # platform's positions growing (see _send).
        self._positions: deque[tuple[int, int, int, int]] = deque(
            maxlen=POSITION_FRAMES
        )
        # The platform's speaker labels not fully given yet, by start.
        self._labels: dict[int, SpeakerLabel] = {}
        self._labels_changed = asyncio.Event()
        # When on_wake was last called (monotonic; see WAKE_DEBOUNCE).
        self._woke_at: float | None = None
        # When a wake word was last heard that no called transcript has
        # matched yet (see WAKE_COVERS).
        self._wake_heard_at: float | None = None
        # Whether on_wake takes ``speaker=`` (an older platform's does not),
        # for the callable it was checked for.
        self._wake_takes_speaker: tuple[object, bool] | None = None

    def _thread_params(self) -> dict:
        """The voice thread: the voice instructions and persona, the
        ``backend_task`` tool and the platform's tools, the chosen model, and
        the server (Codex connects to it when the conversation starts)."""
        settings = realtime_voice_config()
        if problem := infra_settings_error(settings):
            raise ValueError(problem)
        session: dict = {
            "name": self.options.name,
            "aliases": self.options.aliases,
            "group": not self.chat.private,
        }
        if not self.chat.private:
            # Only what calls the bot by name (see set_people).
            session["wake"] = self._wake
        if emotion := str(settings["emotion"]):
            session["tts_emotion"] = emotion
            session["tts_emotion_strength"] = float(settings["emotion_strength"])
        session["tts_stream_text"] = bool(settings["stream_text"])
        config = {
            **{
                key: value
                for key, value in VOICE_THREAD_CONFIG.items()
                if key != "realtime.host_routes_handoffs"
            },
            # The voice thread's tools are plain functions.
            "model_tool_mode": "direct",
            "features.shell_tool": False,
            "web_search": "disabled",
            "realtime.backend": "local_multimodal_infra",
            "realtime.local_infra.url": str(settings["infra_url"]),
            "realtime.local_infra.session": session,
            "realtime.local_infra.idle_compact_percent": int(
                settings["idle_compact_percent"]
            ),
        }
        # Its turns are its own: the chat's memory tools and rights too.
        config.update(self.chat.memory_config())
        if token := str(settings["infra_token"]).strip():
            config["realtime.local_infra.token"] = token
        if path := ref_audio_path(settings):
            config["realtime.local_infra.ref_audio_path"] = str(path)
            if ref_text := str(settings["ref_text"]).strip():
                config["realtime.local_infra.ref_text"] = ref_text
        provider = str(settings["text_model_provider"]).strip()
        model = str(settings["text_model"]).strip()
        if (
            provider
            and not model
            # "" is the built-in OpenAI provider.
            and provider != (voice._runner_config()["model_provider"] or "openai")
        ):
            # The runner's model is another provider's.
            raise ValueError(
                f"choose the voice text model of provider {provider!r} (realtime voice settings)"
            )
        if provider:
            config["model_provider"] = provider
        if model:
            config["model"] = model
        if effort := str(settings["text_reasoning_effort"]).strip():
            config["model_reasoning_effort"] = effort
        return {
            # Kept the same from call to call: the thread's prompt prefix
            # stays cached (the time goes with each call's start).
            "base_instructions": f"{INFRA_INSTRUCTIONS}\n\n{self.prompt}",
            "dynamic_tools": [
                BACKEND_TASK_TOOL,
                STAY_SILENT_TOOL,
                *(tool.spec for tool in self.tools),
            ],
            "no_environment": True,
            "config": config,
        }

    def _tool_handler(self):
        return self._backend_task

    async def _backend_task(self, msg: dict) -> dict:
        """Runs a ``backend_task`` call as a turn of the paired chat; the
        voice thread gets its answer later (``RESULT_PROMPT``). A platform
        tool (``VoiceTool``) is done at once and answered with its result;
        ``stay_silent`` and a platform tool that ends the turn end it (no
        model round follows), the latter saying its ``say`` argument."""
        arguments = msg.get("arguments")
        if msg.get("tool") == "stay_silent":
            return {
                "contentItems": [{"type": "inputText", "text": "Silent."}],
                "success": True,
                "endTurn": True,
            }
        if msg.get("tool") != "backend_task":
            tool = next(
                (t for t in self.tools if t.spec["name"] == msg.get("tool")),
                None,
            )
            if tool is None:
                return {
                    "contentItems": [{"type": "inputText", "text": "Unknown tool."}],
                    "success": False,
                }
            logger.info(
                "%s voice %s: tool %s %s",
                self.label,
                self.key,
                msg.get("tool"),
                arguments,
            )
            try:
                result = await asyncio.wait_for(
                    tool.run(arguments if isinstance(arguments, dict) else {}),
                    VOICE_TOOL_TIMEOUT,
                )
                success = True
            except asyncio.TimeoutError:
                logger.warning("%s voice %s: tool timed out", self.label, self.key)
                result, success = "Failed: it took too long.", False
            except Exception as exc:  # noqa: BLE001 - told to the model
                logger.warning(
                    "%s voice %s: tool failed: %s", self.label, self.key, exc
                )
                result, success = f"Failed: {exc}", False
            response: dict = {
                "contentItems": tool_content(result),
                "success": success,
            }
            # The call's end_turn argument overrides the tool's default when
            # the tool offers it (a tool that does not, the model must see the
            # result of); a failed action is the model's to tell.
            offers_end_turn = "end_turn" in (
                tool.spec.get("inputSchema", {}).get("properties") or {}
            )
            ends_turn = (
                arguments.get("end_turn", tool.ends_turn)
                if offers_end_turn and isinstance(arguments, dict)
                else tool.ends_turn
            )
            if ends_turn is True and success:
                response["endTurn"] = True
                say = arguments.get("say") if isinstance(arguments, dict) else None
                if isinstance(say, str) and say.strip():
                    response["speak"] = say.strip()
            return response
        task = str(
            (arguments.get("task") if isinstance(arguments, dict) else None)
            or self._heard
        ).strip()
        logger.info("%s voice %s: task %r", self.label, self.key, task)

        async def tell(answer: str | None) -> None:
            if answer is None:
                answer = FAILED_SPEECH
            await self._speak(
                RESULT_PROMPT.format(task=task, answer=answer or DONE_SPEECH)
            )

        busy = self._ask(
            TASK_BODY.format(heard=self._heard or task, task=task),
            tell,
            speaker=self._speaker,
        )
        text = "Handed to the backend; the result comes later as a message."
        if busy:
            text += " It is still busy with an earlier request: this one is next."
        return {"contentItems": [{"type": "inputText", "text": text}], "success": True}

    def label_speaker(
        self,
        name: str | None,
        start: int,
        end: int,
        final: bool,
        candidates: list[dict] | None = None,
        **info,
    ) -> None:
        """Who the platform guesses spoke a stretch of its audio, for the
        voice server to name in what it hears (``"name: text"``, or with
        the likelihoods when unsure: ``"[Ann 62% / Bob 30%]: text"``).

        The span is on the platform's clock (``PcmMedia.feed``'s ``pos``);
        it reaches the server on the server's (its 16 kHz samples since the
        session's audio began) once that audio was handed to it: a label for
        audio still queued waits for it, one for audio that never reached
        the server (dropped, or before this session) is dropped. A later
        label with the same ``start`` replaces the earlier one.

        Args:
            name: Their name; None when not recognised.
            start: The span's first sample, platform clock.
            end: Its end (exclusive).
            final: The span is complete (it is not sent again).
            candidates: How likely each one is, likeliest first:
                ``[{"name": str | None, "p": float}]`` (None: someone not
                recognised); passed on (at most LABEL_CANDIDATES).
            **info: More of the platform's guess (bearing, confidence),
                not passed on.
        """
        if self._closed:
            return
        start, end = int(start), int(end)
        ranked: list[tuple[str | None, float]] = []
        for c in candidates or []:
            try:
                who = str(c.get("name") or "").strip() or None
                p = round(min(max(float(c["p"]), 0.0), 1.0), 3)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue  # one malformed entry left out
            ranked.append((who, p))
        ranked = ranked[:LABEL_CANDIDATES]
        label = self._labels.get(start)
        if label is None:
            label = self._labels[start] = SpeakerLabel(
                name, start, end, bool(final), tuple(ranked)
            )
        else:
            label.name, label.end, label.final = name, end, bool(final)
            label.candidates = tuple(ranked)
        self._labels_changed.set()

    def _server_sample(self, pos: int) -> int:
        """The server's sample index of the platform's position ``pos`` (at
        least the first one handed over); audio dropped on the way takes
        the index of what followed it."""
        positions = self._positions
        i = bisect.bisect_right(positions, (pos, float("inf"))) - 1
        if i < 0:
            return positions[0][2]
        start, end, sample, sample_end = positions[i]
        if pos < end:
            return sample + round((pos - start) * (sample_end - sample) / (end - start))
        return positions[i + 1][2] if i + 1 < len(positions) else sample_end

    def _labels_to_give(self) -> list[tuple[int, SpeakerLabel, tuple, tuple]]:
        """The speaker labels the server has not been told as they are now:
        (key, label, what to tell, the label as it was then) each; labels
        done with are forgotten. A label counts as told (and a final one is
        forgotten) only once telling succeeded (``_label_given``)."""
        if not self._positions:
            return []
        first, last = self._positions[0][0], self._positions[-1][1]
        updates = []
        for key, label in list(self._labels.items()):
            if label.end <= first:
                # Older than what is kept, or it never reached the server.
                del self._labels[key]
                continue
            if label.start >= last:
                continue  # not handed over yet
            if label.given_start is None:
                label.given_start = self._server_sample(label.start)
            complete = label.end <= last
            given = (
                label.name,
                label.given_start,
                self._server_sample(min(label.end, last)),
                label.final and complete,
                label.candidates,
            )
            if given[2] <= given[1] or given == label.given:
                # All of it dropped on the way, or nothing new.
                if given[3]:
                    del self._labels[key]
                continue
            updates.append(
                (
                    key,
                    label,
                    given,
                    (label.name, label.end, label.final, label.candidates),
                )
            )
        return updates

    def _label_given(
        self, key: int, label: SpeakerLabel, given: tuple, was: tuple
    ) -> None:
        """The server was told ``given`` of ``label``: a final one is done
        with, unless the platform changed it meanwhile."""
        label.given = given
        if (
            given[3]
            and self._labels.get(key) is label
            and (label.name, label.end, label.final, label.candidates) == was
        ):
            del self._labels[key]

    async def _give_labels(self) -> None:
        """Gives the server the platform's speaker labels as they change and
        as their audio reaches it, at most every LABEL_INTERVAL; one not
        given (the call failed) is tried again the next time."""
        while True:
            await self._labels_changed.wait()
            self._labels_changed.clear()
            for key, label, given, was in self._labels_to_give():
                name, start, end, final, candidates = given
                update = {
                    "speaker": {
                        "name": name,
                        "start": start,
                        "end": end,
                        "final": final,
                    }
                }
                if candidates:
                    update["speaker"]["candidates"] = [
                        {"name": who, "p": p} for who, p in candidates
                    ]
                try:
                    await self._engine.rt.realtime_append_text(
                        self._thread_id, json.dumps(update), "voice_session"
                    )
                except Exception as exc:  # noqa: BLE001 - the conversation goes on
                    logger.warning(
                        "%s voice %s: speaker label not given: %s",
                        self.label,
                        self.key,
                        exc,
                    )
                    self._labels_changed.set()  # the next round tries again
                    break
                self._label_given(key, label, given, was)
            await asyncio.sleep(LABEL_INTERVAL)

    def _called(self, source: str, speaker: str | None = None) -> None:
        """The bot was called by name (``source``: ``wake`` or
        ``transcript``, whose ``speaker`` is the server's guess of who said
        it): ``on_wake``, at most once per WAKE_DEBOUNCE. What it does
        (turning toward the speaker) never gets in the conversation's way."""
        if self._on_wake is None or self._closed:
            return
        now = time.monotonic()
        if self._woke_at is not None and now - self._woke_at < WAKE_DEBOUNCE:
            return
        self._woke_at = now
        logger.debug(
            "%s voice %s: called (%s, %s)", self.label, self.key, source, speaker
        )
        kwargs: dict = {"source": source}
        if self._wake_takes(self._on_wake):
            kwargs["speaker"] = speaker
        try:
            result = self._on_wake(**kwargs)
        except Exception as exc:  # noqa: BLE001 - the conversation goes on
            logger.warning("%s voice %s: on_wake failed: %s", self.label, self.key, exc)
            return
        if inspect.isawaitable(result):

            async def wait() -> None:
                try:
                    await result
                except Exception as exc:  # noqa: BLE001 - the conversation goes on
                    logger.warning(
                        "%s voice %s: on_wake failed: %s", self.label, self.key, exc
                    )

            self._spawn(wait(), "wake")

    def _wake_takes(self, on_wake) -> bool:
        """Whether ``on_wake`` takes ``speaker=`` (a platform made before it
        was passed does not), checked once per callable."""
        if (
            self._wake_takes_speaker is None
            or self._wake_takes_speaker[0] is not on_wake
        ):
            try:
                params = inspect.signature(on_wake).parameters.values()
                takes = any(
                    p.name == "speaker" or p.kind is inspect.Parameter.VAR_KEYWORD
                    for p in params
                )
            except (TypeError, ValueError):
                takes = False
            self._wake_takes_speaker = (on_wake, takes)
        return self._wake_takes_speaker[1]

    async def _give_wake(self) -> None:
        """Tells the voice server whether to pass on only what calls the bot
        by name (one sender at a time: the newest setting is what it ends
        with)."""
        async with self._wake_lock:
            while (
                self._wake != self._wake_given
                and self._engine is not None
                and self._thread_id is not None
                and not self._closed
            ):
                wake = self._wake
                try:
                    await self._engine.rt.realtime_append_text(
                        self._thread_id, json.dumps({"wake": wake}), "voice_session"
                    )
                except Exception as exc:  # noqa: BLE001 - the conversation goes on
                    logger.warning(
                        "%s voice %s: wake setting not given: %s",
                        self.label,
                        self.key,
                        exc,
                    )
                    return
                self._wake_given = wake
                logger.info(
                    "%s voice %s: %s",
                    self.label,
                    self.key,
                    "only what calls it by name" if wake else "hears everything",
                )

    async def _connect(self) -> None:
        """Has Codex start the conversation on the server; the server has
        loaded its models when it is up."""
        engine, events = self._engine, self._events_queue
        started: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._spawn(self._events(events, started), "events")
        self._realtime_requested = True
        self._context_given = self._context
        # A group's setting is given again once the server is up: the
        # thread's session settings may not be it (people came or went
        # while the session started; a thread still loaded keeps the
        # settings it was loaded with). One to one there is none.
        self._wake_given = self._wake if self.chat.private else None
        await engine.rt.realtime_start(
            self._thread_id,
            json.dumps(
                {
                    "transport": {"type": "websocket"},
                    # The platform's context so far (set_context) goes with the start.
                    "realtime_start_instructions": "\n\n".join(
                        part
                        for part in (START_INSTRUCTIONS, time_prompt(), self._context)
                        if part
                    ),
                    "realtime_end_instructions": END_INSTRUCTIONS,
                }
            ),
        )
        await self._wait_open(started, START_TIMEOUT)
        self._phase("voice server session started")
        self._spawn(self.media.play(self._track), "outbound")
        self._spawn(self._send(), "send")
        self._spawn(self._give_labels(), "speakers")
        self.media.start()
        self.started_at = time.monotonic()
        self.ready = True
        await self._give_context()
        await self._give_wake()
        logger.info(
            "%s voice session %s started on the voice server in %.1fs (thread %s)",
            self.label,
            self.key,
            self.started_at - self.created_at,
            self._thread_id,
        )

    async def _send(self) -> None:
        """Hands the platform's audio to Codex as 16 kHz mono PCM, keeping
        where each frame came from (the media's ``track.pos``, if it tells)
        and where it lands on the server's clock, for speaker labels."""
        resampler = av.AudioResampler(format="s16", layout="mono", rate=IN_RATE)
        # Seconds handed over: the server counts its samples from its first
        # (it gets all of them, after it started).
        #
        # Known limitation: Codex queues the audio for the server in a
        # bounded queue (256 frames, about 5 s) and drops a frame when it is
        # full (``realtime_conversation.rs`` ``audio_in``: ``try_send``),
        # while the call here still succeeds. After such a stall ``sent``
        # runs ahead of the server's clock by what was dropped, and speaker
        # labels land that much late until the session restarts. Codex logs
        # it as the warning "dropping input audio frame due to full queue":
        # that line in Codex's log is how to tell.
        sent = 0.0
        try:
            while True:
                frame = await self.media.track.recv()
                pos = getattr(self.media.track, "pos", None)
                seconds = frame.samples / frame.sample_rate
                if pos is not None:
                    if self._positions and pos < self._positions[-1][1]:
                        # The platform's clock went back (a new one): what
                        # was kept is on the old one. Labels from here on
                        # that were never given on the old clock are the new
                        # one's (they may come before their audio is sent).
                        self._positions.clear()
                        self._labels = {
                            key: label
                            for key, label in self._labels.items()
                            if label.start >= pos and label.given_start is None
                        }
                    self._positions.append(
                        (
                            pos,
                            pos + frame.samples,
                            round(sent * IN_RATE),
                            round((sent + seconds) * IN_RATE),
                        )
                    )
                    if self._labels:
                        self._labels_changed.set()  # more of them may go now
                sent += seconds
                for out in resampler.resample(frame):
                    pcm = bytes(out.planes[0])[: out.samples * 2]
                    await self._engine.rt.realtime_append_audio(
                        self._thread_id,
                        json.dumps(
                            {
                                "data": base64.b64encode(pcm).decode(),
                                "sample_rate": IN_RATE,
                                "num_channels": 1,
                                "samples_per_channel": out.samples,
                            }
                        ),
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported, the session ends
            logger.warning("%s voice %s: sending failed: %s", self.label, self.key, exc)
            self._request_close(f"voice send failed: {exc}")

    async def _events(self, events: asyncio.Queue, started: asyncio.Future) -> None:
        """Plays the server's speech and follows the conversation until it
        ends."""
        while True:
            msg = await events.get()
            try:
                if self._record is not None:
                    self._record.event(msg)
                kind = msg.get("type")
                if kind in ("realtime_conversation_closed", "_pump_closed"):
                    reason = msg.get("reason") or msg.get("message") or "closed"
                    self._conversation_ended(started, f"voice closed: {reason}")
                    return
                self._event(msg, started)
            except Exception as exc:  # noqa: BLE001 - one bad event
                logger.warning(
                    "%s voice %s: bad event skipped: %s", self.label, self.key, exc
                )

    def _event(self, msg: dict, started: asyncio.Future) -> None:
        """One event of the conversation (other than its end)."""
        kind = msg.get("type")
        if kind == "realtime_conversation_started":
            if not started.done():
                started.set_result(None)
        elif kind == "realtime_conversation_realtime":
            payload = msg.get("payload")
            if not isinstance(payload, dict):
                return
            if audio := payload.get("AudioOut"):
                self._track.rate = int(audio.get("sample_rate") or 24000)
                self._track.put(base64.b64decode(audio.get("data") or ""))
            elif "ResponseCancelled" in payload:
                # Talked over: what is buffered goes.
                self._track.clear()
                self.media.flush()
            elif wake := payload.get("InputWake"):
                logger.debug(
                    "%s voice %s: wake word %r (%.2f)",
                    self.label,
                    self.key,
                    wake.get("word"),
                    float(wake.get("score") or 0),
                )
                self._wake_heard_at = time.monotonic()
                self._called("wake")
            elif done := payload.get("InputTranscriptDone"):
                self.last_transcript_at = time.monotonic()
                self._heard = str(done.get("text") or "")
                self._speaker = done.get("speaker") or None
                logger.debug("%s voice %s heard: %s", self.label, self.key, self._heard)
                # ``called``: a wake word in it, or its text names the bot
                # (the spotter missed it), or it goes on from a call. A wake
                # word heard since the last called transcript was this one's
                # (or, heard before this transcript came, the next one's: a
                # turn to the newer call): the wake path turned already.
                now = time.monotonic()
                wake_heard = (
                    self._wake_heard_at is not None
                    and now - self._wake_heard_at < WAKE_COVERS
                )
                called = bool(done.get("called"))
                if called:
                    self._wake_heard_at = None
                # While the server passes on only what calls the bot, what
                # wants a reply and no wake word turned to was called by its
                # transcript (or goes on with the exchange).
                if (
                    done.get("respond")
                    and not (called and wake_heard)
                    and not self.chat.private
                    and self._wake
                ):
                    self._called("transcript", self._speaker)
            elif "InputTranscriptDelta" in payload:
                self.last_transcript_at = time.monotonic()
            elif "OutputTranscriptDelta" in payload:
                self.last_answer_at = time.monotonic()
            elif "Error" in payload:
                logger.warning(
                    "%s voice %s: %s", self.label, self.key, payload["Error"]
                )
                if not started.done():
                    started.set_exception(RuntimeError(str(payload["Error"])))
        elif kind == "error":
            logger.warning(
                "%s voice %s: Codex error: %s", self.label, self.key, msg.get("message")
            )

    async def _speak(self, text: str) -> None:
        """Gives the voice thread ``text`` (a task's answer) to tell at the
        next quiet moment."""
        if self._engine is None or self._thread_id is None or self._closed:
            return
        try:
            await self._engine.rt.realtime_append_text(
                self._thread_id, text, "developer"
            )
        except Exception as exc:  # noqa: BLE001 - the conversation goes on
            logger.warning(
                "%s voice %s: answer not told: %s", self.label, self.key, exc
            )

    async def _release_transport(self) -> None:
        """Stops the conversation (Codex ends the server session) and the
        speech track."""
        await super()._release_transport()
        self._track.end()
