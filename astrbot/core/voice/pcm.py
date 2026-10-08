"""Raw PCM audio for a voice session, e.g. from a phone bridge.

Audio is 16-bit little-endian mono at 48 kHz both ways, the rate WebRTC
(Opus) runs at, so nothing is resampled on the way in.
"""

from __future__ import annotations

import asyncio
import fractions
import time
from collections import deque
from collections.abc import Callable
from typing import Protocol

import av
import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError

from astrbot import logger

SAMPLE_RATE = 48000
FRAME_SAMPLES = 960  # 20 ms
FRAME_BYTES = FRAME_SAMPLES * 2
# Frames buffered before inbound audio is handed out, and again after it ran
# dry, so network jitter is not heard as gaps between half-empty frames.
JITTER_FRAMES = 2  # 40 ms
# Inbound audio beyond this is dropped, oldest first. It is large so that
# what is said while the session is still connecting is kept; it drains
# during the pauses (see CATCH_UP_FRAMES).
MAX_BACKLOG_BYTES = SAMPLE_RATE * 2 * 10
# Inbound audio is served at real-time pace, so a backlog never drains by
# itself when the platform sends silence all along (a game's audio): each
# pause beyond its first KEEP_SILENT_FRAMES is dropped while more than
# CATCH_UP_FRAMES are queued. The kept second still ends what was said
# before it (the far side's VAD waits ~0.6 s), so utterances stay apart.
CATCH_UP_FRAMES = 10  # 200 ms
KEEP_SILENT_FRAMES = 50  # 1 s
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE
# With a playout buffer, a stretch of speech starts once this much is queued
# or has been waiting this long: WebRTC hands over a realtime model's audio
# in bursts (measured gaps p95 ~110 ms, p99 ~180 ms), and the far side plays
# what arrives, so each gap would be heard as a dropout.
PREBUFFER_FRAMES = 10  # 200 ms
# Sent at once when a stretch starts, to fill the far side's small buffer.
LEAD_FRAMES = 3  # 60 ms
# Chunks quieter than this (int16 RMS) are silence (as in Mumble's outbound).
SILENCE_RMS = 120


def is_silent(pcm: bytes | bytearray) -> bool:
    """Whether 16-bit PCM is quieter than SILENCE_RMS."""
    samples = np.frombuffer(bytes(pcm), dtype=np.int16).astype(np.float32)
    return not samples.size or float(np.sqrt(np.mean(samples * samples))) < SILENCE_RMS


class FrameSource(Protocol):
    def pull(self) -> bytes | None:
        """One 20 ms frame of 16-bit mono PCM, or None for silence."""


class FrameTrack(MediaStreamTrack):
    """Serves a source's 20 ms frames as an aiortc track at real-time pace;
    silence when the source has nothing.

    ``pos`` is where the frame ``recv`` returned last came from: the
    platform's sample index of its first sample (the source's
    ``pulled_pos``), None for silence or when the platform gave none.
    """

    kind = "audio"

    def __init__(self, source: FrameSource) -> None:
        super().__init__()
        self.source = source
        self.pos: int | None = None
        self._pts = 0
        self._start: float | None = None
        self._silence = bytes(FRAME_BYTES)

    async def recv(self) -> av.AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError
        if self._start is None:
            self._start = time.monotonic()
        wait = self._start + self._pts / SAMPLE_RATE - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        elif wait < -1.0:
            # The loop stalled; resync instead of bursting to catch up.
            self._start = time.monotonic() - self._pts / SAMPLE_RATE
        data = self.source.pull()
        self.pos = getattr(self.source, "pulled_pos", None) if data else None
        data = data or self._silence
        frame = av.AudioFrame(format="s16", layout="mono", samples=FRAME_SAMPLES)
        frame.planes[0].update(data)
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, SAMPLE_RATE)
        self._pts += FRAME_SAMPLES
        return frame


class PcmMedia:
    """A voice session's audio as raw PCM (``astrbot.core.voice.VoiceMedia``).

    The platform calls ``feed`` with what the other side says, in chunks of
    any size, and gets the bot's voice through ``send``: as it arrives, or,
    with a playout buffer, at real-time pace so ``flush`` can still drop it.

    Inbound audio may come with its position on the platform's own clock
    (``feed``'s ``pos``): it travels with the bytes through what is dropped
    and trimmed, so each frame served says where it came from
    (``pulled_pos``, the track's ``pos``).
    """

    def __init__(
        self,
        send: Callable[[bytes], None],
        buffer_seconds: float = 0,
        trim_silence: bool = False,
        prebuffer_frames: int = PREBUFFER_FRAMES,
    ) -> None:
        """
        Args:
            send: Receives the bot's voice, 16-bit mono PCM at 48 kHz, in
                20 ms chunks.
            buffer_seconds: Playout buffer (upper bound of the speech queued
                ahead); 0 passes the audio on as it arrives.
            trim_silence: Drop silent chunks while more than the prebuffer is
                queued. For a realtime peer, which sends silence all along:
                otherwise the backlog of one stall stays as added latency
                until the call ends. Not for a model that delivers speech
                ahead of time (its pauses would be cut out).
            prebuffer_frames: 20 ms frames buffered before a stretch of speech
                starts. A peer sending in bursts needs the default; one that
                sends at real-time pace (the local voice server) needs little.
        """
        self._send = send
        self._queue: deque[bytes] | None = (
            deque(
                maxlen=max(
                    int(buffer_seconds / FRAME_SECONDS),
                    prebuffer_frames + LEAD_FRAMES,
                )
            )
            if buffer_seconds > 0
            else None
        )
        self._trim_silence = trim_silence
        self._prebuffer = max(prebuffer_frames, 1)
        # Frames sent at once when a stretch starts: fewer than are buffered,
        # or the queue runs dry at once and playback stutters.
        self._lead = min(LEAD_FRAMES, self._prebuffer - 1)
        self._buffer = bytearray()
        # Where the queued inbound bytes came from, in order: runs of
        # [platform position of their first sample (None: not given), bytes].
        self._runs: deque[list] = deque()
        # The position of the frame ``pull`` returned last (None: none).
        self.pulled_pos: int | None = None
        # Silent frames served in a row (inbound).
        self._silent_run = 0
        self._playing = False
        # Until the model listens, inbound audio is kept, not handed out.
        self.holding = True
        self.muted = False
        self.track = FrameTrack(self)

    def feed(self, pcm: bytes, pos: int | None = None) -> None:
        """Queues inbound audio for the model.

        Args:
            pcm: 16-bit mono PCM at 48 kHz.
            pos: The platform's sample index of its first sample (its own
                clock, e.g. samples received since it connected), if it
                keeps one.
        """
        if self.muted or not pcm:
            return
        last = self._runs[-1] if self._runs else None
        if last is not None and (
            # Goes on from the last run: one run.
            (pos is None and last[0] is None)
            or (
                pos is not None
                and last[0] is not None
                and last[0] + last[1] // 2 == pos
            )
        ):
            last[1] += len(pcm)
        else:
            self._runs.append([pos, len(pcm)])
        self._buffer += pcm
        if len(self._buffer) > MAX_BACKLOG_BYTES:
            # Keep whole samples: drop an even number of bytes.
            self._take((len(self._buffer) - MAX_BACKLOG_BYTES) & ~1)

    def _take(self, size: int) -> int | None:
        """Drops the first ``size`` inbound bytes.

        Returns:
            The platform position of the first of them (None: not given).
        """
        del self._buffer[:size]
        pos = self._runs[0][0] if self._runs else None
        while size > 0 and self._runs:
            run = self._runs[0]
            if run[1] <= size:
                size -= run[1]
                self._runs.popleft()
            else:
                run[1] -= size
                if run[0] is not None:
                    run[0] += size // 2
                size = 0
        return pos

    def pull(self) -> bytes | None:
        self.pulled_pos = None
        if self.holding:
            return None
        if not self._playing:
            if len(self._buffer) < JITTER_FRAMES * FRAME_BYTES:
                return None
            self._playing = True
        elif len(self._buffer) < FRAME_BYTES:
            self._playing = False  # ran dry: buffer up again
            return None
        while (
            self._silent_run >= KEEP_SILENT_FRAMES
            and len(self._buffer) > (CATCH_UP_FRAMES + 1) * FRAME_BYTES
            and is_silent(self._buffer[:FRAME_BYTES])
        ):
            self._take(FRAME_BYTES)  # a backlog drains in the pauses
        frame = bytes(self._buffer[:FRAME_BYTES])
        self.pulled_pos = self._take(FRAME_BYTES)
        self._silent_run = self._silent_run + 1 if is_silent(frame) else 0
        return frame

    async def play(self, track: MediaStreamTrack) -> None:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        buffer = bytearray()
        ended = asyncio.Event()
        pacer = (
            asyncio.create_task(self._pace(ended)) if self._queue is not None else None
        )
        try:
            while True:
                try:
                    frame = await track.recv()
                except MediaStreamError:
                    break
                for resampled in resampler.resample(frame):
                    buffer += bytes(resampled.planes[0])[: resampled.samples * 2]
                while len(buffer) >= FRAME_BYTES:
                    chunk = bytes(buffer[:FRAME_BYTES])
                    del buffer[:FRAME_BYTES]
                    if self.muted:
                        continue
                    if self._queue is None:
                        self._send(chunk)
                        continue
                    if (
                        self._trim_silence
                        and len(self._queue) > max(self._prebuffer, PREBUFFER_FRAMES)
                        and is_silent(chunk)
                    ):
                        continue  # a backlog drains in the pauses
                    self._queue.append(chunk)  # oldest dropped when full
            ended.set()
            if pacer is not None:
                await pacer  # plays out what is queued
        finally:
            if pacer is not None:
                pacer.cancel()

    async def _pace(self, ended: asyncio.Event) -> None:
        """Sends queued speech at real-time pace, one 20 ms chunk at a time,
        until the queue is empty after ``ended`` is set.

        Each stretch of speech (and each restart after the queue ran dry)
        first buffers up its prebuffer frames, or waits that long.
        """
        loop = asyncio.get_running_loop()
        queue = self._queue
        assert queue is not None
        next_at = 0.0
        playing = False
        waiting_since: float | None = None
        while True:
            if not playing:
                now = loop.time()
                if not queue:
                    if ended.is_set() or self.muted:
                        return
                    waiting_since = None
                    await asyncio.sleep(FRAME_SECONDS)
                    continue
                if waiting_since is None:
                    waiting_since = now
                if (
                    len(queue) < self._prebuffer
                    and now - waiting_since < self._prebuffer * FRAME_SECONDS
                    and not ended.is_set()
                ):
                    await asyncio.sleep(FRAME_SECONDS)
                    continue
                playing, waiting_since = True, None
                next_at = now - self._lead * FRAME_SECONDS
            wait = next_at - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
            elif wait < -1.0:
                next_at = loop.time()  # the loop stalled: resync, don't burst
            next_at += FRAME_SECONDS
            if not queue:
                # Nothing by the time the next chunk is due: buffer up again.
                playing = False
                continue
            try:
                self._send(queue.popleft())
            except Exception:  # noqa: BLE001 - one lost chunk, playing goes on
                logger.exception("Voice: sending the bot's audio failed")

    def start(self) -> None:
        self.holding = False

    def stop(self) -> None:
        self.muted = True
        self._buffer.clear()
        self._runs.clear()
        self.flush()

    def flush(self) -> None:
        """Drops the bot's speech not sent yet (only a playout buffer has any)."""
        if self._queue is not None:
            self._queue.clear()
