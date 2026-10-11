"""Microphone capture and speaker playback via sounddevice."""

import queue
import threading
from collections import deque

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 512  # 32 ms at 16 kHz — matches Silero VAD's expected chunk size


class Microphone:
    """Continuously captures 16 kHz mono int16 audio into a queue of chunks."""

    def __init__(self) -> None:
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SAMPLES,
            callback=self._on_audio,
        )

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if status:
            # Over/underruns mean dropped samples — surface for latency diagnosis.
            print(f"[audio] {status}", flush=True)
        self._queue.put(indata[:, 0].copy())

    def __enter__(self) -> "Microphone":
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stream.stop()
        self._stream.close()

    def read_chunk(self, timeout: float | None = None) -> np.ndarray | None:
        """Next 512-sample int16 chunk; None if none arrived within `timeout` seconds."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        """Discard any audio buffered while we were busy speaking."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


PLAYBACK_RATE = 22_050  # Piper's voices and the filler clips use this rate
# Big blocks: the callback is Python and needs the GIL, which Piper and
# Whisper hold for a while; small blocks would underrun and click.
PLAYBACK_BLOCK = 2048  # ~93 ms


def _to_rate(audio: np.ndarray, rate: int, target: int) -> np.ndarray:
    """int16 mono audio at `target` Hz (linear resampling; fine for speech)."""
    if np.issubdtype(audio.dtype, np.floating):
        audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    elif audio.dtype != np.int16:
        raise ValueError(f"expected int16 or float audio, got {audio.dtype}")
    if rate == target or not audio.size:
        return audio
    n = int(len(audio) * target / rate)
    x = np.linspace(0, len(audio) - 1, n)
    return np.interp(x, np.arange(len(audio)), audio).astype(np.int16)


class Player:
    """One long-lived output stream, fed from a queue of clips.

    Enqueueing never blocks, so TTS synthesis of sentence N+1 overlaps with
    playback of sentence N, and sentences play back to back without the gap
    of opening a new stream each time. Only PortAudio's own callback thread
    touches the audio; `stop()` just empties the buffer under a lock. (Calling
    sd.stop() from one thread while another sat in sd.wait() raced inside
    PortAudio and could crash the process.)
    """

    def __init__(self, sample_rate: int = PLAYBACK_RATE) -> None:
        self._rate = sample_rate
        self._lock = threading.Lock()
        self._clips: deque[np.ndarray] = deque()
        self._current = np.zeros(0, dtype=np.int16)
        self._pos = 0
        self._idle = threading.Event()
        self._idle.set()
        self._silent_frames = 0  # frames of silence written since the last sample
        self.level = 0.0  # loudness of what's playing now, 0..1 (for a visualizer)
        self.underruns = 0
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=PLAYBACK_BLOCK,
            callback=self._fill,
            # If the device goes away the stream ends; never leave waiters hanging.
            finished_callback=self._idle.set,
        )
        # Audio written now is heard this much later. Idle is reported only
        # once the last sample has actually left the speakers, so the mic
        # drain that follows doesn't miss our own echo.
        latency = self._stream.latency
        self._latency_frames = int(latency * sample_rate) if isinstance(latency, (int, float)) else 0
        self._stream.start()

    def _fill(self, outdata, frames, time_info, status) -> None:
        if status and status.output_underflow:
            self.underruns += 1  # a click or gap; reported when the session ends
        out = outdata[:, 0]
        written = 0
        try:
            with self._lock:
                while written < frames:
                    if self._pos >= len(self._current):
                        if not self._clips:
                            break
                        self._current, self._pos = self._clips.popleft(), 0
                        continue
                    n = min(frames - written, len(self._current) - self._pos)
                    out[written : written + n] = self._current[self._pos : self._pos + n]
                    self._pos += n
                    written += n
                if written:
                    self._silent_frames = frames - written
                elif not self._idle.is_set():
                    self._silent_frames += frames
                    if self._silent_frames >= self._latency_frames + frames:
                        self._idle.set()  # the last sample has been heard
        except Exception:  # noqa: BLE001 — an exception here would kill the stream silently
            written = 0
        out[written:] = 0
        played = out[:written].astype(np.float32)
        self.level = float(np.sqrt(np.mean(played**2))) / 32768 if written else 0.0

    def enqueue(self, audio: np.ndarray, sample_rate: int) -> None:
        if not audio.size:
            return
        clip = _to_rate(audio, sample_rate, self._rate)
        with self._lock:
            self._clips.append(clip)
            if self._stream.active:
                self._idle.clear()

    def stop(self) -> None:
        """Drop everything queued and cut the current clip immediately."""
        with self._lock:
            self._clips.clear()
            self._current = np.zeros(0, dtype=np.int16)
            self._pos = 0
            self._idle.set()

    @property
    def busy(self) -> bool:
        """True while any clip is queued or playing."""
        return not self._idle.is_set()

    def wait_until_done(self, timeout: float | None = None) -> bool:
        """Block until all queued audio has finished playing. False on timeout."""
        return self._idle.wait(timeout)

    def close(self) -> None:
        self.stop()
        self._stream.stop()
        self._stream.close()
