import asyncio
import fractions

import av
import numpy as np
import pytest
from aiortc.mediastreams import MediaStreamError

from astrbot.core.voice import pcm
from astrbot.core.voice.pcm import FRAME_BYTES, PcmMedia


def test_inbound_is_held_until_started_then_jitter_buffered():
    media = PcmMedia(lambda chunk: None)
    media.feed(b"\x01\x00" * (pcm.FRAME_SAMPLES * 3))
    assert media.pull() is None  # holding
    media.start()
    assert media.pull() == b"\x01\x00" * pcm.FRAME_SAMPLES
    assert media.pull() is not None
    assert media.pull() is not None
    assert media.pull() is None  # ran dry
    media.feed(bytes(FRAME_BYTES))
    assert media.pull() is None  # buffers two frames before resuming
    media.feed(bytes(FRAME_BYTES))
    assert media.pull() is not None


def test_inbound_backlog_is_bounded_to_whole_samples(monkeypatch):
    monkeypatch.setattr(pcm, "MAX_BACKLOG_BYTES", FRAME_BYTES * 2)
    media = PcmMedia(lambda chunk: None)
    media.feed(b"\x01\x02" * (pcm.FRAME_SAMPLES * 5))
    media.feed(b"\x01\x02" * 7)
    assert len(media._buffer) == FRAME_BYTES * 2
    assert media._buffer[:2] == b"\x01\x02"  # still sample aligned


def test_an_inbound_backlog_drains_in_the_pauses():
    loud = (np.full(pcm.FRAME_SAMPLES, 3000, dtype=np.int16)).tobytes()
    quiet = bytes(FRAME_BYTES)
    media = PcmMedia(lambda chunk: None)
    # 5 s held while connecting: speech, a 3 s pause, speech; the platform
    # goes on sending silence meanwhile.
    for frame in [loud] * 50 + [quiet] * 150 + [loud] * 50:
        media.feed(frame)
    media.start()
    served = [media.pull() for _ in range(170)]
    assert served[:50] == [loud] * 50
    # The pause keeps its first second (the far side's VAD ends the
    # utterance), the rest is dropped while there is a backlog.
    assert served[50 : 50 + pcm.KEEP_SILENT_FRAMES] == [quiet] * pcm.KEEP_SILENT_FRAMES
    assert served[50 + pcm.KEEP_SILENT_FRAMES : 150] == [loud] * 50
    assert all(frame is None for frame in served[150:])  # caught up
    # Fed as fast as it is served (no backlog), a long pause passes whole.
    served = 0
    for _ in range(120):
        media.feed(quiet)
        served += media.pull() == quiet
    assert served == 119  # the first waits for the jitter buffer
    assert len(media._buffer) == FRAME_BYTES


def test_stop_drops_inbound_and_outbound():
    sent = []
    media = PcmMedia(sent.append)
    media.start()
    media.feed(bytes(FRAME_BYTES * 3))
    media.stop()
    assert media.pull() is None
    media.feed(bytes(FRAME_BYTES * 3))
    assert media.pull() is None


class ToneTrack:
    """24 kHz stereo frames, as a peer might send, then the end."""

    def __init__(self, frames: int) -> None:
        self.left = frames
        self.pts = 0

    async def recv(self):
        if not self.left:
            raise MediaStreamError
        self.left -= 1
        samples = (np.ones((1, 480 * 2)) * 1000).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="stereo")
        frame.sample_rate = 24000
        frame.pts = self.pts
        frame.time_base = fractions.Fraction(1, 24000)
        self.pts += 480
        return frame


@pytest.mark.asyncio
async def test_play_resamples_to_48k_mono_20ms_chunks():
    sent = []
    media = PcmMedia(sent.append)
    await media.play(ToneTrack(10))  # 200 ms
    assert sent and all(len(chunk) == FRAME_BYTES for chunk in sent)
    assert 8 <= len(sent) <= 10  # the resampler may hold a few samples back
    media.stop()
    sent.clear()
    await media.play(ToneTrack(5))
    assert sent == []


@pytest.mark.asyncio
async def test_playout_buffer_paces_speech_and_flush_drops_it(monkeypatch):
    monkeypatch.setattr(pcm, "FRAME_SECONDS", 0.005)  # a fast clock
    sent = []
    media = PcmMedia(sent.append, buffer_seconds=10)
    task = asyncio.create_task(media.play(ToneTrack(50)))  # 1 s, delivered at once
    for _ in range(200):
        if sent:
            break
        await asyncio.sleep(0.005)
    # Paced: only the lead and a few frames went out right away.
    assert 0 < len(sent) < 40
    media.flush()
    await asyncio.wait_for(task, 5)
    assert len(sent) < 45  # the flushed rest was never sent


class TrickleTrack(ToneTrack):
    """Frames with pauses between them, as WebRTC delivers a model's speech."""

    def __init__(self, frames: int, pause: float) -> None:
        super().__init__(frames)
        self.pause = pause

    async def recv(self):
        await asyncio.sleep(self.pause)
        return await super().recv()


@pytest.mark.asyncio
async def test_playout_buffer_waits_to_prebuffer_before_speaking(monkeypatch):
    monkeypatch.setattr(pcm, "FRAME_SECONDS", 0.01)
    sent = []
    media = PcmMedia(sent.append, buffer_seconds=3)
    # Three frames (60 ms of speech) spread over 60 ms: under the prebuffer.
    task = asyncio.create_task(media.play(TrickleTrack(3, 0.02)))
    await asyncio.sleep(0.07)
    assert sent == []  # still buffering up
    await asyncio.wait_for(task, 5)
    assert len(sent) >= 2  # played out once the track ended


@pytest.mark.asyncio
async def test_a_small_prebuffer_keeps_its_pace(monkeypatch):
    monkeypatch.setattr(pcm, "FRAME_SECONDS", 0.01)
    media = PcmMedia(lambda chunk: None, buffer_seconds=3, prebuffer_frames=3)
    # The start's lead stays under what is buffered: the queue never runs dry
    # at once.
    assert media._lead == 2
    assert PcmMedia(lambda chunk: None, buffer_seconds=3, prebuffer_frames=1)._lead == 0
    assert PcmMedia(lambda chunk: None, buffer_seconds=3)._lead == pcm.LEAD_FRAMES


def test_inbound_positions_travel_with_the_audio(monkeypatch):
    monkeypatch.setattr(pcm, "MAX_BACKLOG_BYTES", FRAME_BYTES * 4)
    loud = (np.full(pcm.FRAME_SAMPLES, 3000, dtype=np.int16)).tobytes()
    media = PcmMedia(lambda chunk: None)
    # Six frames while held: the backlog keeps the last four.
    for i in range(6):
        media.feed(loud, pos=10_000 + i * pcm.FRAME_SAMPLES)
    media.start()
    track = pcm.FrameTrack(media)
    assert media.pull() is not None
    assert media.pulled_pos == 10_000 + 2 * pcm.FRAME_SAMPLES
    # Half frames, and a jump on the platform's clock (it lost audio); the
    # backlog drops the oldest frame again.
    media.feed(loud[: FRAME_BYTES // 2], pos=20_000)
    media.feed(loud[FRAME_BYTES // 2 :], pos=20_000 + pcm.FRAME_SAMPLES // 2)
    media.feed(loud, pos=30_000)
    served = []
    for _ in range(4):
        assert media.pull() is not None
        served.append(media.pulled_pos)
    assert served == [
        10_000 + 4 * pcm.FRAME_SAMPLES,
        10_000 + 5 * pcm.FRAME_SAMPLES,
        20_000,
        30_000,
    ]
    assert media.pull() is None and media.pulled_pos is None
    # Audio without a position has none.
    media.feed(loud)
    media.feed(loud)
    assert media.pull() is not None and media.pulled_pos is None
    assert track.pos is None


def test_trimmed_silence_takes_its_positions_along():
    loud = (np.full(pcm.FRAME_SAMPLES, 3000, dtype=np.int16)).tobytes()
    quiet = bytes(FRAME_BYTES)
    media = PcmMedia(lambda chunk: None)
    frames = [loud] * 5 + [quiet] * 100 + [loud] * 5
    for i, frame in enumerate(frames):
        media.feed(frame, pos=i * pcm.FRAME_SAMPLES)
    media.start()
    served = {}
    while (frame := media.pull()) is not None:
        served[media.pulled_pos // pcm.FRAME_SAMPLES] = frame
    # The pause beyond its first second went; what follows it says where
    # it came from.
    assert sorted(served)[: 5 + pcm.KEEP_SILENT_FRAMES] == list(
        range(5 + pcm.KEEP_SILENT_FRAMES)
    )
    assert sorted(served)[-5:] == list(range(105, 110))
    assert all(served[i] == loud for i in range(105, 110))


@pytest.mark.asyncio
async def test_the_track_tells_where_each_frame_came_from():
    media = PcmMedia(lambda chunk: None)
    media.feed(bytes(FRAME_BYTES * 2), pos=480)
    media.start()
    track = media.track
    await track.recv()
    assert track.pos == 480
    await track.recv()
    assert track.pos == 480 + pcm.FRAME_SAMPLES
    await track.recv()  # ran dry: silence
    assert track.pos is None
    track.stop()
