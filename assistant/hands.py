"""The webcam and MediaPipe hand tracking, on their own thread.

Frames are mirrored (so moving your hand right moves it right), turned by
MediaPipe's gesture recognizer into the 21 points of one hand plus a gesture
label ("Open_Palm", "Victory", "Thumb_Down", ...), and handed to a callback.
The recognizer is the hand landmarker with a classifier trained on real
hands on top, at the same cost (~17 ms a frame). The frames themselves
stay in memory and are never saved or sent anywhere. The camera is only
open while hand control is on.

OpenCV and MediaPipe are optional (requirements-hands.txt) and imported only
when the camera starts, so the rest of the assistant runs without them.
"""

import importlib.util
import math
import os
import threading
import time
from pathlib import Path
from typing import Callable

from .gestures import Points

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gesture_recognizer.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)
INSTALL_HINT = "pip install -r requirements-hands.txt, then python -m assistant.setup hands"

WIDTH, HEIGHT = 640, 480
ACTIVE_FPS = 15
IDLE_FPS = 5  # after a while with no hand: spare the CPU for speech
IDLE_AFTER_S = 3.0
MAX_BAD_READS = 30  # about two seconds of failed reads: the camera is gone
MIN_LABEL_SCORE = 0.6  # less sure than this: no label, the point rules decide


class HandsUnavailable(Exception):
    """A message safe to show in the chat."""


def check() -> None:
    """Raise HandsUnavailable unless the packages and the model are present.

    Only looks the packages up; importing MediaPipe takes a few seconds and
    happens on the camera thread, so the window never freezes."""
    if any(importlib.util.find_spec(name) is None for name in ("cv2", "mediapipe")):
        raise HandsUnavailable(f"Hand control needs MediaPipe: {INSTALL_HINT}.")
    if not MODEL_PATH.is_file():
        raise HandsUnavailable("The hand model is missing: run python -m assistant.setup hands.")


class HandTracker:
    """Calls on_points(points or None, time, label or None) from the camera
    thread, and on_stop(error or None) once when the camera closes."""

    def __init__(
        self,
        on_points: Callable[[Points | None, float, str | None], None],
        on_stop: Callable[[str | None], None],
        camera: int = 0,
    ) -> None:
        self._on_points = on_points
        self._on_stop = on_stop
        self._camera = camera
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        check()
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="camera")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float) -> None:
        """Wait (a little) for the camera thread to finish and release the camera."""
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        error: str | None = None
        capture = recognizer = None
        try:
            os.environ.setdefault("GLOG_minloglevel", "2")  # before MediaPipe loads: its startup chatter
            import cv2
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision

            recognizer = vision.GestureRecognizer.create_from_options(
                vision.GestureRecognizerOptions(
                    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=1,
                    # A little below the defaults: keep a motion-blurred hand in sight.
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.4,
                    min_tracking_confidence=0.4,
                )
            )
            capture = cv2.VideoCapture(self._camera, cv2.CAP_DSHOW)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            if not capture.isOpened():
                raise HandsUnavailable("I can't open the camera. Is another app using it?")
            bad_reads = 0
            last_hand = time.monotonic()
            next_frame = 0.0
            while not self._stop.is_set():
                # grab() every camera frame so the next one we look at is
                # fresh; decode only the frames we process.
                if not capture.grab():
                    bad_reads += 1
                    if bad_reads > MAX_BAD_READS:
                        raise HandsUnavailable("The camera stopped sending pictures.")
                    time.sleep(0.05)
                    continue
                bad_reads = 0
                now = time.monotonic()
                if now < next_frame:
                    continue
                fps = ACTIVE_FPS if now - last_hand < IDLE_AFTER_S else IDLE_FPS
                next_frame = now + 1 / fps
                ok, frame = capture.retrieve()
                if not ok:
                    continue
                rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                result = recognizer.recognize_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(now * 1000)
                )
                points = label = None
                if result.hand_landmarks:
                    points = [(p.x, p.y) for p in result.hand_landmarks[0]]
                    last_hand = now
                    if result.gestures and result.gestures[0]:
                        best = result.gestures[0][0]
                        if best.score >= MIN_LABEL_SCORE:
                            label = best.category_name
                self._on_points(points, now, label)
        except HandsUnavailable as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 — the camera thread must report, not vanish
            error = f"Hand control stopped: {type(exc).__name__}: {exc}"
        finally:
            if capture is not None:
                capture.release()
            if recognizer is not None:
                recognizer.close()
            self._on_stop(error)


def _debug(seconds: float) -> None:
    """Print what the camera sees and which gestures fire, acting on nothing:
    for checking the thresholds in gestures.py on a real hand."""
    from .gestures import INDEX_TIP, THUMB_TIP, CloseArming, GestureEngine, hand_size, pinched, pose
    from .handmenu import TILES

    labels = [tile.label for tile in TILES]
    engine = GestureEngine(len(labels))
    done = threading.Event()
    seen = {"line": None, "arming": False}
    start = time.monotonic()

    def on_points(points: Points | None, t: float, label: str | None) -> None:
        if points is None:
            line, gap = "no hand", ""
        else:
            ok = " pinch" if pinched(points) else ""
            line = f"label {label or '-':<12} shape {pose(points, label):<10}{ok}"
            gap = f"  (thumb-index {math.dist(points[THUMB_TIP], points[INDEX_TIP]) / hand_size(points):.2f})"
        if line != seen["line"]:
            seen["line"] = line
            print(f"{t - start:6.2f}  {line}{gap}", flush=True)
        for event in engine.feed(points, t, label):
            arming = isinstance(event, CloseArming)
            if not (arming and seen["arming"]):
                tile = getattr(event, "index", None)
                name = f" = {labels[tile]}" if tile is not None else ""
                print(f"{t - start:6.2f}  >> {event}{name}", flush=True)
            seen["arming"] = arming

    def on_stop(error: str | None) -> None:
        print(error or "camera off", flush=True)
        done.set()

    tracker = HandTracker(on_points, on_stop)
    try:
        tracker.start()
    except HandsUnavailable as exc:
        print(exc)
        return
    print(f"Watching for {seconds:.0f} s (Ctrl+C to stop). Nothing is opened, closed or switched.", flush=True)
    try:
        done.wait(seconds)
    except KeyboardInterrupt:
        pass
    tracker.stop()
    tracker.join(2.0)


if __name__ == "__main__":
    import sys

    _debug(float(sys.argv[1]) if len(sys.argv) > 1 else 60.0)
