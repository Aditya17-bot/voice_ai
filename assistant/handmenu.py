"""Hand control: camera points -> gestures -> the overlay and the actions.

The camera thread turns hand points into gesture events and posts them to
the window's main loop; everything here except `_feed` runs on Tk's main
thread. Actions go through the same read-only tools as voice commands
(tools.Workspace) plus window control (windows.py). Each one logs a single
chat line; nothing is spoken.
"""

import math
import tkinter as tk
from dataclasses import dataclass
from typing import Callable

from .gestures import (
    CloseArming,
    CloseCancel,
    CloseWindow,
    DialClose,
    DialOpen,
    Event,
    MENU_COLUMNS,
    GestureEngine,
    Hover,
    MenuClose,
    MenuOpen,
    Pick,
    Points,
    SwitchDesktop,
    SwitchTab,
    SwitchWindow,
    palm_centre,
    pose,
)
from .hands import HandsUnavailable, HandTracker
from .tools import ToolError
from .windows import WindowControl, WindowError

Log = Callable[[str, str], None]


@dataclass(frozen=True)
class Tile:
    label: str
    kind: str  # "site", "app" or "folder"
    target: str


# The hand menu, clockwise from the top. Every target is on an allowlist in
# tools.py; folders are the allowed roots, shown in File Explorer.
TILES = [
    Tile("YouTube", "site", "youtube"),
    Tile("Gmail", "site", "gmail"),
    Tile("GitHub", "site", "github"),
    Tile("Chrome", "app", "chrome"),
    Tile("Terminal", "app", "terminal"),
    Tile("VS Code", "app", "vscode"),
    Tile("Claude", "app", "claude"),
    Tile("adi", "folder", "adi"),
    Tile("Desktop", "folder", "desktop"),
    Tile("Downloads", "folder", "downloads"),
]


def tiles_for(workspace) -> list[Tile]:
    """The menu for this computer: folder tiles only for roots that exist."""
    roots = set(workspace.root_names)
    return [t for t in TILES if t.kind != "folder" or t.target in roots]


def run_tile(workspace, tile: Tile) -> str:
    if tile.kind == "site":
        return workspace.open_website(tile.target)
    if tile.kind == "app":
        return workspace.open_app(tile.target)
    return workspace.open_in_explorer(tile.target)


class HandControl:
    def __init__(
        self,
        post: Callable[[Callable[[], None]], None],
        log: Log,
        workspace: Callable[[], object],
        on_state: Callable[[bool], None],
        overlay: Callable[[], "Overlay"],
        windows: WindowControl | None = None,
        tracker: Callable[..., HandTracker] = HandTracker,
    ) -> None:
        """`post(fn)` runs fn on the main thread; `on_state(on)` follows the camera."""
        self._post = post
        self._log = log
        self._workspace = workspace
        self._on_state = on_state
        self._make_overlay = overlay
        self._overlay: Overlay | None = None
        self._windows = windows
        self._make_tracker = tracker
        self._tracker: HandTracker | None = None
        self._previous: HandTracker | None = None  # the last run's, maybe still closing the camera
        # Each camera run gets its own token: a stopped camera thread that
        # still delivers points, or reports its end late, can't touch a newer run.
        self._session: object | None = None
        self.tiles: list[Tile] = []
        self._close_target: tuple[int, str] | None = None  # (window, title) named; window 0: can't close

    @property
    def on(self) -> bool:
        return self._session is not None

    def start(self) -> str | None:
        """Turn the camera on; returns why it can't, or None."""
        if self.on:
            return None
        workspace = self._workspace()
        if workspace is None:
            return self._fail("Hand control needs the desktop tools, and they're off.")
        session = object()
        engine = GestureEngine(len(tiles_for(workspace)))
        seen = [True]  # whether the last frame had a hand: "no hand" is drawn once

        def feed(points: Points | None, t: float, label: str | None = None) -> None:
            # Camera thread. The engine belongs to this run alone.
            if self._session is not session:
                return
            if points is not None or seen[0]:
                shape = pose(points, label) if points is not None else ""
                self._post(lambda: self._deliver_hand(session, points, shape))
            seen[0] = points is not None
            for event in engine.feed(points, t, label):
                self._post(lambda e=event: self._deliver(session, e))

        def stopped(error: str | None) -> None:
            self._post(lambda: self._camera_stopped(session, error))

        try:
            if self._windows is None:
                self._windows = WindowControl()
            if self._overlay is None:  # before the camera, so a failure here leaves it off
                self._overlay = self._make_overlay()
            self.tiles = tiles_for(workspace)
            if self._previous is not None:
                self._previous.join(1.0)  # let a just-stopped run release the camera
            tracker = self._make_tracker(feed, stopped)
            tracker.start()
        except (HandsUnavailable, WindowError, tk.TclError) as exc:
            return self._fail(str(exc))
        self._session, self._tracker = session, tracker
        self._close_target = None
        self._on_state(True)
        self._log("note", "✋ Camera on. Hold 👍 for the menu (open your hand, move to a tile, ✊ grab or pinch "
                          "to open). Hold ✋ still, then move it left or right, to switch windows (like Alt+Tab); "
                          "✌ the same for desktops, ☝ for tabs; ✊ or hand away to stop. Hold 👎 to close.")
        return None

    def stop(self) -> None:
        if self._tracker is not None:
            self._tracker.stop()
            self._previous = self._tracker
        self._turned_off()

    def _fail(self, message: str) -> str:
        self._on_state(False)
        self._log("note", f"✋ {message}")
        return message

    def _turned_off(self) -> None:
        was_on = self.on
        self._session = self._tracker = None
        self._close_target = None
        if self._overlay is not None:
            self._overlay.clear()
        if was_on:
            self._on_state(False)

    def _camera_stopped(self, session: object, error: str | None) -> None:
        if session is not self._session:
            return  # an earlier run, already turned off
        if error:
            self._log("note", f"✋ {error}")
        self._turned_off()

    def _deliver(self, session: object, event: Event) -> None:
        if session is self._session:
            self.handle(event)

    def _deliver_hand(self, session: object, points: Points | None, shape: str) -> None:
        if session is self._session and self._overlay is not None:
            self._overlay.show_hand(points, shape)

    # ----- main thread -------------------------------------------------------

    def handle(self, event: Event) -> None:
        if not self.on or self._overlay is None:
            return
        overlay = self._overlay
        try:
            if isinstance(event, MenuOpen):
                overlay.show_menu([t.label for t in self.tiles])
            elif isinstance(event, Hover):
                overlay.hover(event.index)
            elif isinstance(event, MenuClose):
                overlay.hide_menu()
            elif isinstance(event, Pick):
                tile = self.tiles[event.index]
                self._log("note", f"✋ {run_tile(self._workspace(), tile)}")
            elif isinstance(event, SwitchWindow):
                title = self._windows.switch_window(event.step)
                overlay.toast("next window →" if event.step > 0 else "← previous window")
                self._log("note", f"✋ Switched to {title}.")
            elif isinstance(event, SwitchDesktop):
                self._windows.switch_desktop(event.step)
                overlay.toast("desktop →" if event.step > 0 else "← desktop")
                self._log("note", f"✋ Switched to the {'next' if event.step > 0 else 'previous'} desktop.")
            elif isinstance(event, DialOpen):
                overlay.show_dial(event.kind)
            elif isinstance(event, DialClose):
                overlay.hide_dial()
            elif isinstance(event, SwitchTab):
                title = self._windows.switch_tab(event.step)
                overlay.toast("next tab →" if event.step > 0 else "← previous tab")
                self._log("note", f"✋ {'Next' if event.step > 0 else 'Previous'} tab in {title}.")
            elif isinstance(event, CloseArming):
                if self._close_target is None:
                    hwnd, title = self._windows.front()
                    if self._windows.can_close(hwnd):
                        self._close_target = (hwnd, title or "this window")
                    else:  # say so now, not after a pointless countdown
                        self._close_target = (0, "")
                        overlay.toast("can't close this one")
                if self._close_target[0]:
                    overlay.close_progress(self._close_target[1], event.progress)
            elif isinstance(event, CloseCancel):
                self._close_target = None
                overlay.close_clear()
            elif isinstance(event, CloseWindow):
                # Only the window the overlay named, and only if it's still in front.
                target, self._close_target = self._close_target, None
                overlay.close_clear()
                if target is None or not target[0]:
                    return
                closed = self._windows.close_front(expected=target[0])
                self._log("note", f"✋ Closed {closed}.")
        except (ToolError, WindowError) as exc:
            overlay.close_clear()
            self._log("note", f"✋ {exc}")


# ----- the overlay -----------------------------------------------------------

OVERLAY_W, OVERLAY_H = 680, 440  # at 100 % display scaling
TRANSPARENT = "#010203"
TILE_BG = "#0b1220"
ACCENT = "#22d3ee"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
DANGER = "#f87171"
TOAST_MS = 900


def _front_window() -> int:
    try:
        import ctypes

        return ctypes.windll.user32.GetForegroundWindow() or 0
    except (AttributeError, OSError):
        return 0


def _give_focus_back(hwnd: int) -> None:
    """Creating a window takes the focus once; hand it back to where it was."""
    try:
        import ctypes

        if hwnd and ctypes.windll.user32.GetForegroundWindow() != hwnd:
            ctypes.windll.user32.SetForegroundWindow(hwnd)
    except (AttributeError, OSError):
        pass


def _click_through(top: tk.Toplevel) -> None:
    """Never take clicks or the keyboard focus, and stay out of Alt+Tab (Windows)."""
    try:
        import ctypes

        top.update_idletasks()
        hwnd = int(top.wm_frame(), 16)
        user32 = ctypes.windll.user32
        user32.GetWindowLongW.restype = ctypes.c_long
        style = user32.GetWindowLongW(hwnd, -20)  # GWL_EXSTYLE
        layered, transparent, tool, no_activate = 0x80000, 0x20, 0x80, 0x08000000
        user32.SetWindowLongW(hwnd, -20, style | layered | transparent | tool | no_activate)
    except (AttributeError, OSError, ValueError, tk.TclError):
        pass  # not Windows: it just isn't click-through


def _layer(root: tk.Tk, width: int, height: int, x: int, y: int) -> tuple[tk.Toplevel, tk.Canvas]:
    """A see-through, click-through, always-on-top window to draw on."""
    top = tk.Toplevel(root)
    top.overrideredirect(True)
    top.attributes("-topmost", True)
    try:
        top.attributes("-transparentcolor", TRANSPARENT)
    except tk.TclError:
        pass
    top.geometry(f"{width}x{height}+{x}+{y}")
    canvas = tk.Canvas(top, width=width, height=height, bg=TRANSPARENT, highlightthickness=0)
    canvas.pack()
    _click_through(top)
    return top, canvas


# MediaPipe's hand skeleton: which of the 21 points are joined.
BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
SHAPE_NAMES = {"palm": "✋ open palm", "two": "✌ two fingers", "one": "☝ one finger", "fist": "✊ fist",
               "thumb_down": "👎 thumbs down", "thumb_up": "👍 thumbs up", "other": "hand"}
DIAL_NAMES = {"window": "✋ window mode", "desktop": "✌ desktop mode", "tab": "☝ tab mode"}
VIEW_W, VIEW_H = 192, 144  # the camera frame, 4:3
EDGE_LOW, EDGE_SIDE = 0.8, 0.12  # palm centre past these (frame units): ask to move the hand in
WARN = "#fbbf24"


class Overlay:
    """A see-through, click-through, always-on-top square in the middle of
    the screen, and a small hand view in the bottom-right corner showing the
    hand as the camera sees it (its points, not the picture). Both stay
    shown and simply draw nothing when idle, so they never have to be
    raised (which would take the focus from your window)."""

    def __init__(self, root: tk.Tk) -> None:
        was_in_front = _front_window()
        scale = root.winfo_fpixels("1i") / 96
        self._w, self._h = w, h = int(OVERLAY_W * scale), int(OVERLAY_H * scale)
        self._scale = scale
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        self.top, self.canvas = _layer(root, w, h, (sw - w) // 2, (sh - h) // 2)
        pad = int(10 * scale)
        self._view = (int(VIEW_W * scale), int(VIEW_H * scale), pad)
        vw, vh = self._view[0] + 2 * pad, self._view[1] + int(44 * scale)
        self.hand_top, self.hand_canvas = _layer(
            root, vw, vh, sw - vw - int(24 * scale), sh - vh - int(72 * scale)  # above the taskbar
        )
        self.top.update()
        _give_focus_back(was_in_front)
        self._labels: list[str] = []
        self._hovered: int | None = None
        self._toast_job: str | None = None

    def _font(self, points: int, bold: bool = False) -> tuple:
        return ("Segoe UI", points, "bold") if bold else ("Segoe UI", points)

    def clear(self) -> None:
        self._labels = []
        try:
            self.canvas.delete("all")
            self.hand_canvas.delete("all")
        except tk.TclError:
            pass  # the window is already gone (quitting)

    def show_hand(self, points: Points | None, shape: str) -> None:
        """The camera frame as a box, the hand's skeleton in it, what shape
        it reads as, and a hint when the hand is near the frame's edge."""
        c, s = self.hand_canvas, self._scale
        w, h, pad = self._view
        try:
            c.delete("all")
        except tk.TclError:
            return
        c.create_rectangle(0, 0, w + 2 * pad, h + 44 * s, fill="#05080d", outline="#1f4b57", width=2)
        c.create_rectangle(pad, pad, pad + w, pad + h, outline="#334155", dash=(3, 3))
        text, colour = "no hand in view", MUTED
        if points is not None:
            xy = [(pad + min(max(x, 0.0), 1.0) * w, pad + min(max(y, 0.0), 1.0) * h) for x, y in points]
            for a, b in BONES:
                c.create_line(*xy[a], *xy[b], fill=ACCENT, width=2)
            for x, y in xy:
                c.create_oval(x - 2, y - 2, x + 2, y + 2, fill=TEXT, outline="")
            cx, cy = palm_centre(points)
            if cy > EDGE_LOW:
                text, colour = "raise your hand ↑", WARN
            elif cx < EDGE_SIDE or cx > 1 - EDGE_SIDE:
                text, colour = "move your hand to the middle", WARN
            else:
                text, colour = SHAPE_NAMES.get(shape, "hand"), TEXT
        c.create_text(pad + w / 2, pad + h + 22 * s, text=text, fill=colour, font=self._font(10, bold=True),
                      width=w)

    def show_menu(self, labels: list[str]) -> None:
        self.canvas.delete("menu", "toast")
        self._labels = labels
        self._hovered = None
        self._draw_menu()

    def hover(self, index: int | None) -> None:
        if self._labels:
            self._hovered = index
            self._draw_menu()

    def hide_menu(self) -> None:
        self._labels = []
        self.canvas.delete("menu")

    def _draw_menu(self) -> None:
        """Tiles in rows of MENU_COLUMNS, as the gesture engine lays them out."""
        c, s = self.canvas, self._scale
        c.delete("menu")
        n = len(self._labels)
        cols = max(1, min(MENU_COLUMNS, n))
        rows = math.ceil(n / cols) if n else 1
        tile_w, tile_h, gap = 118 * s, 74 * s, 10 * s
        grid_w, grid_h = cols * tile_w + (cols - 1) * gap, rows * tile_h + (rows - 1) * gap
        left, top = (self._w - grid_w) / 2, (self._h - grid_h) / 2
        c.create_rectangle(left - 20 * s, top - 44 * s, left + grid_w + 20 * s, top + grid_h + 40 * s,
                           fill="#05080d", outline="#1f4b57", width=2, tags="menu")
        c.create_text(self._w / 2, top - 22 * s, text="move your palm to a tile", fill=MUTED,
                      font=self._font(10), tags="menu")
        c.create_text(self._w / 2, top + grid_h + 20 * s, text="✊ grab or pinch to open · hand away to close",
                      fill=MUTED, font=self._font(9), tags="menu")
        for i, label in enumerate(self._labels):
            row, col = divmod(i, cols)
            x, y = left + col * (tile_w + gap), top + row * (tile_h + gap)
            lit = i == self._hovered
            c.create_rectangle(x, y, x + tile_w, y + tile_h, fill=ACCENT if lit else TILE_BG,
                               outline=ACCENT, width=3 if lit else 1, tags="menu")
            c.create_text(x + tile_w / 2, y + tile_h / 2, text=label, fill="#05080d" if lit else TEXT,
                          font=self._font(11, bold=lit), width=tile_w - 8 * s, justify="center", tags="menu")

    def close_progress(self, title: str, progress: float) -> None:
        c = self.canvas
        c.delete("close")
        x, y, r = self._w / 2, self._h / 2, self._h * 0.2
        c.create_oval(x - r * 1.25, y - r * 1.25, x + r * 1.25, y + r * 1.25,
                      fill="#05080d", outline="", tags="close")
        c.create_oval(x - r, y - r, x + r, y + r, outline="#3f1d1d", width=6, tags="close")
        c.create_arc(x - r, y - r, x + r, y + r, start=90, extent=-359.9 * min(1.0, progress),
                     style="arc", outline=DANGER, width=6, tags="close")
        c.create_text(x, y - 10 * self._scale, text=f"Close {title}?", fill=TEXT,
                      font=self._font(10, bold=True), width=r * 1.7, justify="center", tags="close")
        c.create_text(x, y + r * 0.62, text="keep 👎 to close", fill=MUTED,
                      font=self._font(9), tags="close")

    def close_clear(self) -> None:
        self.canvas.delete("close")

    def show_dial(self, kind: str) -> None:
        c, s = self.canvas, self._scale
        c.delete("dial")
        x, y, w, h = self._w / 2, 40 * s, 230 * s, 26 * s
        c.create_rectangle(x - w, y - h, x + w, y + h, fill="#05080d", outline=ACCENT, width=2, tags="dial")
        c.create_text(x, y - 8 * s, text=f"◀  {DIAL_NAMES.get(kind, kind)}  ▶", fill=TEXT,
                      font=self._font(12, bold=True), tags="dial")
        c.create_text(x, y + 12 * s, text="move left / right · ✊ or hand away to stop", fill=MUTED,
                      font=self._font(9), tags="dial")

    def hide_dial(self) -> None:
        self.canvas.delete("dial")

    def toast(self, text: str) -> None:
        c = self.canvas
        c.delete("toast")
        x, y = self._w / 2, self._h / 2
        w, h = 140 * self._scale, 22 * self._scale
        c.create_rectangle(x - w, y - h, x + w, y + h, fill="#05080d", outline=ACCENT, width=2,
                           tags="toast")
        c.create_text(x, y, text=text, fill=TEXT, font=self._font(12, bold=True), tags="toast")
        if self._toast_job is not None:
            self.top.after_cancel(self._toast_job)
        self._toast_job = self.top.after(TOAST_MS, lambda: c.delete("toast"))
