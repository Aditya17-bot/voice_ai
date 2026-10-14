"""Hand gestures from hand points: pure rules, no camera, no Windows calls.

Input: the 21 MediaPipe hand points of one hand (x, y normalized 0..1, from
a mirrored frame, y down), or None when no hand is seen, plus a timestamp
and MediaPipe's gesture label ("Open_Palm", "Victory", ...) when it has one.
The label, from a model trained on real hands, decides the hand shape; the
point rules below are the fallback when it has no opinion.
Output: events the window turns into overlay drawing and actions.

Gestures:
- thumbs up held: open the hand menu, a grid of tiles; then the open palm
  moves the highlight like a mouse moves a pointer; closing the hand into a
  fist (grab), or a quick pinch (tap, as in barehands), opens the
  highlighted tile; taking the hand away closes the menu
- dials, like a scroll wheel: hold a shape still, then each step of hand
  travel right / left is one step forward / back; a fist or taking the hand
  away ends it. Open hand: windows (Alt+Tab). Two fingers: virtual
  desktops. One finger: tabs in the front window.
- thumbs down held: close the front window

Windows and desktops were swipes at first. Seen live, a swipe swings
straight back, pauses about as long at the far end as before the next
swipe, and blurs out of sight in a dim room: no rule told swipes from swings
back. The dial's steps follow where the hand is, so moving back just steps
back.

Distances in hand rules are in units of hand size (wrist to middle knuckle),
so they work near or far from the camera. Motion thresholds are in frame
units. Every action is followed by a cooldown, so one motion is one action.
"""

import math
from dataclasses import dataclass

Points = list[tuple[float, float]]

WRIST, THUMB_MCP, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 2, 4, 8, 9
FINGERS = {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}  # (pip, tip)
KNUCKLES = (5, 9, 13, 17)

# MediaPipe gesture labels -> our shapes. Its other label (ILoveYou) is a
# shape we don't use; "None" means no opinion.
LABELS = {"Open_Palm": "palm", "Victory": "two", "Pointing_Up": "one", "Closed_Fist": "fist",
          "Thumb_Down": "thumb_down", "Thumb_Up": "thumb_up"}
NO_OPINION = (None, "", "None")
DIALS = {"palm": "window", "two": "desktop", "one": "tab"}  # shape held still -> dial

EXTENDED_RATIO = 1.1  # tip this much farther from the wrist than the middle joint
THUMB_LENGTH = 0.5  # thumbs down / up: thumb at least this long (knuckle to tip)...
THUMB_DOWN_DEGREES = 35  # ...within this angle of straight down (up), and the lowest (highest) point
LOWEST_SLACK = 0.05  # (give or take this much)
PINCH = 0.3  # thumb tip to index tip

MENU_HOLD_S = 0.6  # thumbs up held still this long opens the menu...
MENU_EDGE_Y = 0.88  # ...unless the hand is mostly below the frame (seen live: a hand cut off
# by the bottom edge read as a thumbs up)
MENU_STILL = 0.04  # "still": palm centre stays within this radius
MENU_PALM_S = 3.0  # no open palm this long after the thumbs up closes the menu
PALM_FRAMES = 3  # the palm that moves the highlight: open this many frames (not mid-way from the thumb)
MENU_COLUMNS = 5
CELL_X, CELL_Y = 0.07, 0.10  # palm travel (frame units) from one tile to the next
CELL_MARGIN = 0.25  # the palm must be this far (in cells) into a tile to move there
MENU_LOST_S = 0.8  # hand gone this long closes the menu
MENU_IDLE_S = 8.0  # nothing picked or hovered for this long closes it
GRAB_FRAMES = 3  # a fist this many frames in a row opens the tile
TAP_MAX_S = 0.6  # a pinch let go within this is a tap...
TAP_MIN_FRAMES = 2  # ...if it lasted at least this many frames
HOVER_SETTLE_S = 0.25  # a tile must stay highlighted this long before a grab or tap opens it

DIAL_HOLD_S = 0.5  # shape held still this long opens its dial
DIAL_STEP = 0.07  # hand travel (frame widths) per step (seen live: 7 tabs across half the frame, none missed)
DIAL_LOST_S = 0.6  # hand gone this long ends the dial
DIAL_IDLE_S = 6.0  # no step for this long ends it

CLOSE_HOLD_S = 1.0
CLOSE_GRACE_S = 0.2  # a dropped frame or two doesn't cancel the close
CLOSE_REARM_S = 0.6  # after a close, the thumb must be gone this long before the next

COOLDOWN_S = 1.0


@dataclass(frozen=True)
class MenuOpen:
    pass


@dataclass(frozen=True)
class Hover:
    index: int | None  # None: back in the middle, nothing hovered


@dataclass(frozen=True)
class Pick:
    index: int


@dataclass(frozen=True)
class MenuClose:
    pass


@dataclass(frozen=True)
class SwitchWindow:
    step: int  # +1 next, -1 previous


@dataclass(frozen=True)
class SwitchDesktop:
    step: int  # +1 the desktop to the right, -1 to the left


@dataclass(frozen=True)
class SwitchTab:
    step: int  # +1 next tab, -1 previous


@dataclass(frozen=True)
class DialOpen:
    kind: str  # "window", "desktop" or "tab"


@dataclass(frozen=True)
class DialClose:
    pass


@dataclass(frozen=True)
class CloseArming:
    progress: float  # 0..1


@dataclass(frozen=True)
class CloseCancel:
    pass


@dataclass(frozen=True)
class CloseWindow:
    pass


Event = (MenuOpen | Hover | Pick | MenuClose | SwitchWindow | SwitchDesktop | SwitchTab | DialOpen
         | DialClose | CloseArming | CloseCancel | CloseWindow)
STEPS = {"window": SwitchWindow, "desktop": SwitchDesktop, "tab": SwitchTab}


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hand_size(points: Points) -> float:
    return max(_dist(points[WRIST], points[MIDDLE_MCP]), 1e-6)


def palm_centre(points: Points) -> tuple[float, float]:
    ids = (WRIST, *KNUCKLES)
    return sum(points[i][0] for i in ids) / len(ids), sum(points[i][1] for i in ids) / len(ids)


def extended(points: Points, ratio: float = EXTENDED_RATIO) -> dict[str, bool]:
    wrist = points[WRIST]
    return {
        name: _dist(wrist, points[tip]) > _dist(wrist, points[pip]) * ratio
        for name, (pip, tip) in FINGERS.items()
    }


def _thumb(points: Points, down: bool) -> bool:
    """Thumb long and pointing down (up), its tip the lowest (highest) point:
    holds however the fist is turned, unlike "below the knuckles" on an
    upright hand."""
    sign = 1 if down else -1  # flip y for thumbs up
    mcp, tip = points[THUMB_MCP], points[THUMB_TIP]
    vx, vy = tip[0] - mcp[0], (tip[1] - mcp[1]) * sign
    length = math.hypot(vx, vy)
    if length < THUMB_LENGTH * hand_size(points):
        return False
    pointing = math.degrees(math.acos(max(-1.0, min(1.0, vy / length)))) <= THUMB_DOWN_DEGREES
    extreme_other = max(p[1] * sign for i, p in enumerate(points) if i != THUMB_TIP)
    return pointing and tip[1] * sign >= extreme_other - LOWEST_SLACK * hand_size(points)


def pose(points: Points, label: str | None = None) -> str:
    """"palm", "two" (index and middle up), "one" (index up), "fist",
    "thumb_down", "thumb_up" or "other"."""
    if label in LABELS:
        return LABELS[label]
    if label not in NO_OPINION:
        return "other"
    up = extended(points)
    if not any(up.values()):
        if _thumb(points, down=True):
            return "thumb_down"
        return "thumb_up" if _thumb(points, down=False) else "fist"
    if all(up.values()):
        return "palm"
    if up["index"] and not up["ring"] and not up["pinky"]:
        return "two" if up["middle"] else "one"
    return "other"


def pinched(points: Points) -> bool:
    """Thumb and index tips touching."""
    return _dist(points[THUMB_TIP], points[INDEX_TIP]) < PINCH * hand_size(points)


class GestureEngine:
    def __init__(self, tiles: int, columns: int = MENU_COLUMNS) -> None:
        """`tiles` laid out in rows of `columns`, left to right, top to bottom."""
        self.tiles = tiles
        self.columns = max(1, min(columns, tiles))
        self.rows = math.ceil(tiles / self.columns)
        self.reset()

    def reset(self) -> None:
        self._menu_open = False
        self._anchor: tuple[float, float] | None = None  # palm centre when the open palm first came
        self._palm_frames = 0
        self._menu_opened_at = 0.0
        self._cell = (0, 0)  # highlighted (column, row)
        self._hover: int | None = None
        self._menu_active_at = 0.0  # menu opened, or hover last changed
        self._hover_since = 0.0
        self._grab_frames = 0
        self._grab_settled = False  # the highlight had settled when the fist began
        self._pinch_frames = 0
        self._pinch_start = 0.0
        self._pinch_settled = False
        self._seen_at = 0.0
        self._still_from: tuple[float, float] | None = None
        self._still_since = 0.0
        self._still_shape = ""
        self._dial_kind = ""  # the open dial: "window", "desktop", "tab", or "" for none
        self._dial_shape = ""
        self._dial_anchor = 0.0  # palm centre x when the dial opened
        self._dial_at = 0  # steps so far (+ right, - left)
        self._dial_active_at = 0.0
        self._close_since: float | None = None
        self._close_seen_at = 0.0
        self._close_blocked = False  # thumb must come up before the next close
        self._thumb_gone_since: float | None = None
        self._cooldown_until = 0.0

    # ----- feed --------------------------------------------------------

    def feed(self, points: Points | None, t: float, label: str | None = None) -> list[Event]:
        if points is None:
            return self._lost(t)
        self._seen_at = t
        shape = pose(points, label)
        centre = palm_centre(points)
        self._rearm(shape != "thumb_down", t)
        if self._menu_open:
            return self._menu(points, shape, centre, t)
        if self._dial_kind:
            return self._dial(shape, centre, t)
        events = self._close(shape, t)
        if events or self._close_since is not None:
            self._still_from = None
            return events
        if t < self._cooldown_until:
            self._still_from = None
            return []
        return self._hold_still(shape, centre, t)

    def _lost(self, t: float) -> list[Event]:
        self._still_from = None
        self._grab_frames = self._pinch_frames = 0
        self._rearm(True, t)
        events: list[Event] = []
        if self._menu_open and t - self._seen_at >= MENU_LOST_S:
            events += self._close_menu(t)
        if self._dial_kind and t - self._seen_at >= DIAL_LOST_S:
            events += self._close_dial(t)
        if self._close_since is not None and t - self._close_seen_at > CLOSE_GRACE_S:
            self._close_since = None
            events.append(CloseCancel())
        return events

    # ----- menu ----------------------------------------------------------

    def _hold_still(self, shape: str, centre: tuple[float, float], t: float) -> list[Event]:
        """A thumbs up held still opens the menu; an open hand, two fingers or
        one finger, its dial."""
        if shape not in ("thumb_up", *DIALS) or (shape == "thumb_up" and centre[1] > MENU_EDGE_Y):
            self._still_from = None
            return []
        if self._still_from is None or shape != self._still_shape or _dist(centre, self._still_from) > MENU_STILL:
            self._still_from, self._still_since, self._still_shape = centre, t, shape
            return []
        if shape in DIALS:
            if t - self._still_since < DIAL_HOLD_S:
                return []
            self._dial_kind, self._dial_shape = DIALS[shape], shape
            self._dial_anchor, self._dial_at, self._dial_active_at = centre[0], 0, t
            self._still_from = None
            return [DialOpen(self._dial_kind)]
        if t - self._still_since < MENU_HOLD_S:
            return []
        self._menu_open = True
        self._anchor = None
        self._palm_frames = 0
        self._cell = (self.columns // 2, 0)  # the palm starts on the middle of the top row
        self._hover = self._index(self._cell)
        self._menu_active_at = self._hover_since = self._menu_opened_at = t
        self._grab_frames = self._pinch_frames = 0
        self._still_from = None
        return [MenuOpen(), Hover(self._hover)]

    def _menu(self, points: Points, shape: str, centre: tuple[float, float], t: float) -> list[Event]:
        events: list[Event] = []
        if self._anchor is None:  # waiting for the open palm; the thumbs up lowering is no grab
            self._palm_frames = self._palm_frames + 1 if shape == "palm" else 0
            if shape == "thumb_up":  # seen live: the thumb stays up a few seconds, watching the menu appear
                self._menu_opened_at = t
            if self._palm_frames >= PALM_FRAMES:
                self._anchor = centre
                self._menu_active_at = self._hover_since = t
            elif t - self._menu_opened_at >= MENU_PALM_S:
                events += self._close_menu(t)
            return events
        if shape == "palm":  # any other shape freezes the highlight: closing the hand moves it
            hover = self._move_highlight(centre)
            if hover != self._hover:
                self._hover = hover
                self._menu_active_at = self._hover_since = t
                events.append(Hover(hover))
        settled = t - self._hover_since >= HOVER_SETTLE_S
        # Grab: the open hand closes into a fist.
        if shape == "fist":
            if self._grab_frames == 0:
                self._grab_settled = settled
            self._grab_frames += 1
        else:
            self._grab_frames = 0
        # Tap (barehands): a quick pinch, let go.
        tapped = False
        if pinched(points) and shape != "fist":
            if self._pinch_frames == 0:
                self._pinch_start, self._pinch_settled = t, settled
            self._pinch_frames += 1
        elif self._pinch_frames:
            tapped = (self._pinch_frames >= TAP_MIN_FRAMES and t - self._pinch_start <= TAP_MAX_S
                      and self._pinch_settled)
            self._pinch_frames = 0
        grabbed = self._grab_frames >= GRAB_FRAMES and self._grab_settled
        if (grabbed or tapped) and self._hover is not None:
            events.append(Pick(self._hover))
            events += self._close_menu(t, acted=True)
        elif t - self._menu_active_at >= MENU_IDLE_S:
            events += self._close_menu(t)
        return events

    def _index(self, cell: tuple[int, int]) -> int:
        return cell[1] * self.columns + cell[0]

    def _move_highlight(self, centre: tuple[float, float]) -> int:
        """The palm moves the highlight like a pointer, from the tile it
        started on; it must be well into the next tile to move there."""
        fx = (centre[0] - self._anchor[0]) / CELL_X + self.columns // 2 + 0.5
        fy = (centre[1] - self._anchor[1]) / CELL_Y + 0.5
        col, row = self._cell
        if fx >= col + 1 + CELL_MARGIN or fx < col - CELL_MARGIN:
            col = min(max(int(math.floor(fx)), 0), self.columns - 1)
        if fy >= row + 1 + CELL_MARGIN or fy < row - CELL_MARGIN:
            row = min(max(int(math.floor(fy)), 0), self.rows - 1)
        last_row = self.tiles - (self.rows - 1) * self.columns  # tiles in a short last row
        if row == self.rows - 1:
            col = min(col, last_row - 1)
        self._cell = (col, row)
        return self._index(self._cell)

    def _close_menu(self, t: float, acted: bool = False) -> list[Event]:
        self._menu_open = False
        self._hover = None
        self._grab_frames = self._pinch_frames = 0
        if acted:
            self._cooldown_until = t + COOLDOWN_S
        return [MenuClose()]

    # ----- dials ----------------------------------------------------------

    def _dial(self, shape: str, centre: tuple[float, float], t: float) -> list[Event]:
        """Like a scroll wheel: each DIAL_STEP of hand travel is one step,
        right for forward, left for back; at most one per frame."""
        if shape == "fist":
            return self._close_dial(t)
        events: list[Event] = []
        if shape == self._dial_shape:  # any other shape holds still
            offset = (centre[0] - self._dial_anchor) / DIAL_STEP
            step = 0
            if offset >= self._dial_at + 0.5 + CELL_MARGIN:
                step = 1
            elif offset <= self._dial_at - 0.5 - CELL_MARGIN:
                step = -1
            if step:
                self._dial_at += step
                self._dial_active_at = t
                events.append(STEPS[self._dial_kind](step))
        if t - self._dial_active_at >= DIAL_IDLE_S:
            events += self._close_dial(t)
        return events

    def _close_dial(self, t: float) -> list[Event]:
        self._dial_kind = self._dial_shape = ""
        self._cooldown_until = t + COOLDOWN_S  # the hand dropping afterwards opens nothing
        return [DialClose()]

    # ----- close ---------------------------------------------------------

    def _rearm(self, thumb_gone: bool, t: float) -> None:
        """Allow the next close once the thumb has been gone a while: one frame
        read as a fist in the middle of a held thumbs down is not lowering it."""
        if not thumb_gone:
            self._thumb_gone_since = None
        elif self._thumb_gone_since is None:
            self._thumb_gone_since = t
        elif t - self._thumb_gone_since >= CLOSE_REARM_S:
            self._close_blocked = False

    def _close(self, shape: str, t: float) -> list[Event]:
        if shape == "thumb_down" and not self._close_blocked and t >= self._cooldown_until:
            self._close_seen_at = t
            if self._close_since is None:
                self._close_since = t
            progress = (t - self._close_since) / CLOSE_HOLD_S
            if progress < 1:
                return [CloseArming(progress)]
            self._close_since = None
            self._close_blocked = True
            self._cooldown_until = t + COOLDOWN_S
            return [CloseWindow()]
        if self._close_since is not None and t - self._close_seen_at > CLOSE_GRACE_S:
            self._close_since = None
            return [CloseCancel()]
        return []
