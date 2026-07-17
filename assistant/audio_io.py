"""Microphone capture and speaker playback via sounddevice."""

import queue
import threading

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

    def read_chunk(self) -> np.ndarray:
        """Blocking read of the next 512-sample int16 chunk."""
        return self._queue.get()

    def drain(self) -> None:
        """Discard any audio buffered while we were busy speaking."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


def play(audio: np.ndarray, sample_rate: int) -> None:
    """Play int16 or float32 mono audio, blocking until done."""
    sd.play(audio, samplerate=sample_rate)
    sd.wait()


class Player:
    """Dedicated playback thread fed by a queue of (audio, sample_rate) clips.

    Enqueueing never blocks, so TTS synthesis of sentence N+1 overlaps with
    playback of sentence N. `stop()` clears pending clips and cuts the current
    one (used for barge-in later).
    """

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._pending = 0  # clips enqueued but not finished playing
        self._pending_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            audio, sample_rate = self._queue.get()
            try:
                if audio.size:
                    sd.play(audio, samplerate=sample_rate)
                    sd.wait()
            finally:
                self._queue.task_done()
                with self._pending_lock:
                    self._pending -= 1

    def enqueue(self, audio: np.ndarray, sample_rate: int) -> None:
        if audio.size:
            with self._pending_lock:
                self._pending += 1
            self._queue.put((audio, sample_rate))

    def stop(self) -> None:
        """Drop everything queued and stop the current clip immediately."""
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                with self._pending_lock:
                    self._pending -= 1
            except queue.Empty:
                break
        sd.stop()

    @property
    def busy(self) -> bool:
        """True while any clip is queued or playing."""
        with self._pending_lock:
            return self._pending > 0

    def wait_until_done(self) -> None:
        """Block until all queued audio has finished playing."""
        self._queue.join()
