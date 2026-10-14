"""Gesture rules, fed synthetic hands (no camera)."""

import math
import unittest

from assistant import gestures as g
from assistant.gestures import (
    CloseArming,
    CloseCancel,
    CloseWindow,
    DialClose,
    DialOpen,
    GestureEngine,
    Hover,
    MenuClose,
    MenuOpen,
    Pick,
    SwitchDesktop,
    SwitchTab,
    SwitchWindow,
)

FPS = 15
DT = 1 / FPS

# Finger knuckles (MCP) relative to the wrist, in units of hand size, y down.
_MCP = {"index": (-0.35, -0.95), "middle": (0.0, -1.0), "ring": (0.3, -0.95), "pinky": (0.55, -0.85)}
_FIRST = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


def hand(up=("index", "middle", "ring", "pinky"), thumb="side", at=(0.5, 0.6), size=0.15, turn=0.0):
    """21 points of a hand, upright unless turned. `at` is the palm centre.

    up: fingers pointing up; the rest are curled into the palm.
    thumb: "side" (out to the side), "tucked" (across the fist), "up"
    (thumbs up; turn=180 makes it a thumbs down: wrist on top, thumb under
    the knuckles), "pinch" (the "OK" sign: index curled onto the thumb tip).
    turn: degrees clockwise, around the palm centre.
    """
    rel = [(0.0, 0.0)] * 21
    for name, (mx, my) in _MCP.items():
        i = _FIRST[name]
        if name in up:
            chain = [(mx, my), (mx, my - 0.45), (mx, my - 0.75), (mx, my - 1.0)]
        else:  # curled: the tip comes back down next to the knuckle
            chain = [(mx, my), (mx, my - 0.35), (mx + 0.05, my - 0.15), (mx + 0.05, my + 0.15)]
        rel[i:i + 4] = chain
    thumbs = {
        "side": [(-0.35, -0.25), (-0.65, -0.45), (-0.9, -0.62), (-1.1, -0.8)],
        "tucked": [(-0.35, -0.25), (-0.45, -0.5), (-0.3, -0.65), (-0.15, -0.7)],
        "up": [(-0.3, -0.3), (-0.45, -0.75), (-0.5, -1.15), (-0.55, -1.6)],
    }
    if thumb == "pinch":
        # The index bends forward so its tip meets the thumb tip: a ring.
        rel[5:9] = [(-0.35, -0.95), (-0.4, -1.35), (-0.55, -1.3), (-0.6, -1.12)]
        rel[1:5] = [(-0.35, -0.25), (-0.55, -0.55), (-0.65, -0.85), (-0.62, -1.08)]
    else:
        rel[1:5] = thumbs[thumb]
    # Palm centre (wrist + knuckles average) is (0.1, -0.75) from the wrist.
    cx, cy = 0.1, -0.75
    a = math.radians(turn)
    rel = [(cx + (x - cx) * math.cos(a) - (y - cy) * math.sin(a),
            cy + (x - cx) * math.sin(a) + (y - cy) * math.cos(a)) for x, y in rel]
    return [(at[0] + (x - cx) * size, at[1] + (y - cy) * size) for x, y in rel]


PALM = dict(up=("index", "middle", "ring", "pinky"))
TWO = dict(up=("index", "middle"), thumb="tucked")
ONE = dict(up=("index",), thumb="tucked")
FIST = dict(up=(), thumb="tucked")
THUMBS_DOWN = dict(up=(), thumb="up", turn=180)
THUMBS_UP = dict(up=(), thumb="up")
OK_PINCH = dict(up=("middle", "ring", "pinky"), thumb="pinch")


class Run:
    """Feeds frames at a steady frame rate and collects events."""

    def __init__(self, tiles=10) -> None:
        self.engine = GestureEngine(tiles)
        self.t = 100.0
        self.events: list = []

    def frames(self, seconds, label=None, **pose):
        out = []
        for _ in range(round(seconds * FPS)):
            self.t += DT
            out += self.engine.feed(hand(**pose), self.t, label)
        self.events += out
        return out

    def none(self, seconds):
        out = []
        for _ in range(round(seconds * FPS)):
            self.t += DT
            out += self.engine.feed(None, self.t)
        self.events += out
        return out

    def move(self, seconds, start, end, label=None, **pose):
        out = []
        n = max(1, round(seconds * FPS))
        for k in range(1, n + 1):
            self.t += DT
            f = k / n
            at = (start[0] + (end[0] - start[0]) * f, start[1] + (end[1] - start[1]) * f)
            out += self.engine.feed(hand(at=at, **pose), self.t, label)
        self.events += out
        return out

    def open_menu(self, at=(0.5, 0.6)):
        """Thumbs up held, then the open palm the highlight follows."""
        out = self.frames(g.MENU_HOLD_S + 0.2, at=at, **THUMBS_UP)
        assert any(isinstance(e, MenuOpen) for e in out), out
        return out + self.frames(0.3, at=at, **PALM)


def kinds(events):
    return [type(e) for e in events]


class PoseTests(unittest.TestCase):
    def test_poses_are_recognized(self):
        self.assertEqual(g.pose(hand(**PALM)), "palm")
        self.assertEqual(g.pose(hand(**TWO)), "two")
        self.assertEqual(g.pose(hand(**FIST)), "fist")
        self.assertEqual(g.pose(hand(**THUMBS_DOWN)), "thumb_down")
        self.assertEqual(g.pose(hand(**THUMBS_UP)), "thumb_up")

    def test_thumbs_down_at_an_angle(self):
        self.assertEqual(g.pose(hand(up=(), thumb="up", turn=160)), "thumb_down")
        self.assertEqual(g.pose(hand(up=(), thumb="up", turn=205)), "thumb_down")
        self.assertNotEqual(g.pose(hand(up=(), thumb="up", turn=90)), "thumb_down")  # thumb sideways

    def test_recognizer_label_wins(self):
        self.assertEqual(g.pose(hand(**PALM), "Victory"), "two")
        self.assertEqual(g.pose(hand(**PALM), "Thumb_Down"), "thumb_down")
        self.assertEqual(g.pose(hand(**PALM), "Thumb_Up"), "thumb_up")
        self.assertEqual(g.pose(hand(**PALM), "ILoveYou"), "other")
        self.assertEqual(g.pose(hand(**TWO), "None"), "two")  # no opinion: geometry decides
        self.assertEqual(g.pose(hand(**TWO), None), "two")

    def test_one_finger(self):
        self.assertEqual(g.pose(hand(**ONE)), "one")
        self.assertEqual(g.pose(hand(**PALM), "Pointing_Up"), "one")

    def test_pinch(self):
        self.assertTrue(g.pinched(hand(**OK_PINCH)))
        self.assertFalse(g.pinched(hand(**PALM)))

    def test_size_and_turn_do_not_matter(self):
        for size in (0.07, 0.3):
            self.assertEqual(g.pose(hand(size=size, **PALM)), "palm")
            self.assertEqual(g.pose(hand(size=size, **THUMBS_DOWN)), "thumb_down")
            self.assertTrue(g.pinched(hand(size=size, **OK_PINCH)))
        self.assertEqual(g.pose(hand(turn=25, **TWO)), "two")
        self.assertTrue(g.pinched(hand(turn=-30, **OK_PINCH)))


class MenuTests(unittest.TestCase):
    def test_thumbs_up_opens_menu(self):
        run = Run()
        self.assertNotIn(MenuOpen, kinds(run.frames(g.MENU_HOLD_S - 0.15, **THUMBS_UP)))
        self.assertIn(MenuOpen, kinds(run.frames(0.3, **THUMBS_UP)))

    def test_still_palm_does_not_open_menu(self):
        """A palm resting is the window dial, never the menu."""
        run = Run()
        self.assertNotIn(MenuOpen, kinds(run.frames(2, **PALM)))

    def test_thumbs_up_at_the_bottom_edge_does_not_open_menu(self):
        """Seen live: a hand cut off by the bottom of the frame read as a thumbs up."""
        run = Run()
        self.assertEqual(run.frames(2, at=(0.5, 0.95), label="Thumb_Up", **FIST), [])

    def test_lowering_the_thumb_picks_nothing(self):
        """The thumbs up becomes a fist on the way to an open hand: no grab."""
        run = Run()
        run.frames(g.MENU_HOLD_S + 0.2, **THUMBS_UP)
        self.assertNotIn(Pick, kinds(run.frames(0.5, **FIST)))

    def test_menu_without_an_open_palm_closes(self):
        run = Run()
        run.frames(g.MENU_HOLD_S + 0.2, **THUMBS_UP)
        self.assertNotIn(MenuClose, kinds(run.frames(g.MENU_PALM_S - 0.5, **FIST)))
        self.assertIn(MenuClose, kinds(run.frames(0.7, **FIST)))

    def test_long_thumbs_up_keeps_the_menu(self):
        """Seen live: the thumb stayed up 2.6 s after the menu opened, then the palm came."""
        run = Run()
        run.frames(g.MENU_HOLD_S + g.MENU_PALM_S + 0.5, **THUMBS_UP)
        out = run.frames(0.3, **FIST)
        out += run.frames(0.5, at=(0.5 + g.CELL_X * 1.5, 0.6), **PALM)
        self.assertNotIn(MenuClose, kinds(run.events))
        out += run.frames(0.4, at=(0.5 + g.CELL_X * 3, 0.6), **PALM)
        self.assertEqual([e for e in out if isinstance(e, Hover)], [Hover(3)])

    def test_highlight_starts_where_the_palm_opens(self):
        """The palm centre moves as the thumbs up opens into a palm: the
        highlight starts from where the open palm settles, not the thumb."""
        run = Run()
        run.frames(g.MENU_HOLD_S + 0.2, at=(0.5, 0.6), **THUMBS_UP)
        out = run.frames(0.6, at=(0.5 + g.CELL_X * 2, 0.6), **PALM)
        self.assertEqual([e for e in out if isinstance(e, Hover)], [])

    def test_menu_starts_on_the_middle_of_the_top_row(self):
        run = Run()  # 10 tiles: two rows of five
        out = run.open_menu()
        self.assertEqual([e for e in out if isinstance(e, Hover)], [Hover(2)])

    def hovers(self, out):
        return [e.index for e in out if isinstance(e, Hover)]

    def test_palm_moves_the_highlight_like_a_pointer(self):
        run = Run()
        run.open_menu()
        x, y = 0.5, 0.6
        out = run.move(0.3, (x, y), (x + g.CELL_X * 1.4, y), **PALM)  # one tile right
        self.assertEqual(self.hovers(out), [3])
        out = run.move(0.3, (x + g.CELL_X * 1.4, y), (x + g.CELL_X * 1.4, y + g.CELL_Y * 1.4), **PALM)  # down
        self.assertEqual(self.hovers(out), [8])
        out = run.move(0.4, (x + g.CELL_X * 1.4, y + g.CELL_Y * 1.4), (x - g.CELL_X * 2.4, y + g.CELL_Y * 1.4),
                       **PALM)  # three left, one tile at a time
        self.assertEqual(self.hovers(out), [7, 6, 5])

    def test_past_the_edge_stays_on_the_edge_tile(self):
        run = Run()
        run.open_menu()
        out = run.move(0.5, (0.5, 0.6), (0.95, 0.3), **PALM)
        self.assertEqual(self.hovers(out)[-1], 4)

    def test_short_last_row(self):
        run = Run(tiles=9)  # five on top, four below
        run.open_menu()
        out = run.move(0.5, (0.5, 0.6), (0.95, 0.9), **PALM)
        self.assertEqual(self.hovers(out)[-1], 8)

    def test_jitter_on_a_border_does_not_flicker(self):
        run = Run()
        run.open_menu()
        border = 0.5 + g.CELL_X * 0.5  # between tiles 2 and 3
        out = []
        for k in range(30):
            run.t += DT
            out += run.engine.feed(hand(at=(border + 0.01 * (-1) ** k, 0.6), **PALM), run.t)
        self.assertEqual(self.hovers(out), [])

    def test_hover_is_reported_once(self):
        run = Run()
        run.open_menu()
        run.move(0.2, (0.5, 0.6), (0.6, 0.6), **PALM)
        self.assertEqual(run.frames(0.5, at=(0.6, 0.6), **PALM), [])

    def test_grab_picks_highlighted_tile(self):
        run = Run()
        run.open_menu()
        run.move(0.2, (0.5, 0.6), (0.6, 0.6), **PALM)
        run.frames(0.4, at=(0.6, 0.6), **PALM)
        out = run.frames(0.3, at=(0.6, 0.6), **FIST)
        self.assertIn(Pick(3), out)
        self.assertIn(MenuClose, kinds(out))

    def test_closing_the_hand_does_not_move_the_highlight(self):
        """The palm centre shifts as the fingers curl: the tile is the one
        highlighted while the hand was open."""
        run = Run()
        run.open_menu()
        run.frames(0.4, **PALM)
        out = run.frames(0.3, at=(0.5 + g.CELL_X * 1.5, 0.6), **FIST)
        self.assertIn(Pick(2), out)

    def test_tap_picks_highlighted_tile(self):
        run = Run()
        run.open_menu()
        run.frames(0.4, **PALM)
        out = run.frames(3 * DT, **OK_PINCH)
        self.assertNotIn(Pick, kinds(out))  # not until it's let go
        out = run.frames(DT, **PALM)
        self.assertIn(Pick(2), out)

    def test_long_pinch_is_not_a_tap(self):
        run = Run()
        run.open_menu()
        run.frames(0.4, **PALM)
        run.frames(g.TAP_MAX_S + 0.2, **OK_PINCH)
        self.assertNotIn(Pick, kinds(run.frames(0.2, **PALM)))

    def test_one_frame_pinch_is_not_a_tap(self):
        run = Run()
        run.open_menu()
        run.frames(0.4, **PALM)
        run.frames(DT, **OK_PINCH)
        self.assertNotIn(Pick, kinds(run.frames(0.2, **PALM)))

    def test_grab_needs_a_settled_highlight(self):
        """Grabbing just as the highlight lands on a tile opens nothing."""
        run = Run()
        run.open_menu()
        run.move(2 * DT, (0.5, 0.6), (0.5 + g.CELL_X * 1.4, 0.6), **PALM)
        out = run.frames(0.4, at=(0.5 + g.CELL_X * 1.4, 0.6), **FIST)
        self.assertNotIn(Pick, kinds(out))

    def test_short_fist_is_not_a_grab(self):
        run = Run()
        run.open_menu()
        run.frames(0.4, **PALM)
        run.frames((g.GRAB_FRAMES - 1) * DT, **FIST)
        self.assertNotIn(Pick, kinds(run.frames(0.3, **PALM)))

    def test_lost_hand_closes_menu(self):
        run = Run()
        run.open_menu()
        self.assertEqual(run.none(g.MENU_LOST_S - 0.2), [])
        self.assertIn(MenuClose, kinds(run.none(0.4)))

    def test_idle_menu_times_out(self):
        run = Run()
        run.open_menu()
        self.assertIn(MenuClose, kinds(run.frames(g.MENU_IDLE_S + 0.2, **PALM)))

    def test_no_dial_while_menu_is_open(self):
        run = Run()
        run.open_menu()
        out = run.frames(2, **PALM)
        self.assertNotIn(DialOpen, kinds(out))

    def test_menu_does_not_reopen_right_after_pick(self):
        run = Run()
        run.open_menu()
        run.frames(0.4, **PALM)
        self.assertIn(Pick, kinds(run.frames(0.3, **FIST)))
        out = run.frames(g.COOLDOWN_S, **THUMBS_UP)
        self.assertNotIn(MenuOpen, kinds(out))


class DialTests(unittest.TestCase):
    def open_dial(self, run, shape=ONE, kind="tab", at=(0.5, 0.6), label=None):
        out = run.frames(g.DIAL_HOLD_S + 0.15, at=at, label=label, **shape)
        self.assertIn(DialOpen(kind), out)

    def test_still_finger_opens_the_tab_dial(self):
        run = Run()
        self.assertNotIn(DialOpen, kinds(run.frames(g.DIAL_HOLD_S - 0.15, **ONE)))
        self.assertIn(DialOpen("tab"), run.frames(0.3, **ONE))

    def test_each_shape_has_its_dial(self):
        self.open_dial(Run(), PALM, "window")
        self.open_dial(Run(), TWO, "desktop")
        self.open_dial(Run(), PALM, "desktop", label="Victory")

    def test_moving_finger_steps_tabs(self):
        run = Run()
        self.open_dial(run)
        out = run.move(0.6, (0.5, 0.6), (0.5 + g.DIAL_STEP * 2.5, 0.6), **ONE)
        self.assertEqual(out, [SwitchTab(1), SwitchTab(1)])
        out = run.move(0.6, (0.5 + g.DIAL_STEP * 2.5, 0.6), (0.5 + g.DIAL_STEP * 0.2, 0.6), **ONE)
        self.assertEqual(out, [SwitchTab(-1), SwitchTab(-1)])

    def test_palm_dial_steps_windows(self):
        run = Run()
        self.open_dial(run, PALM, "window")
        out = run.move(0.4, (0.5, 0.6), (0.5 + g.DIAL_STEP * 1.5, 0.6), **PALM)
        self.assertEqual(out, [SwitchWindow(1)])
        out = run.move(0.4, (0.5 + g.DIAL_STEP * 1.5, 0.6), (0.5 - g.DIAL_STEP * 0.2, 0.6), **PALM)
        self.assertEqual(out, [SwitchWindow(-1)])  # moving back steps back

    def test_two_finger_dial_steps_desktops(self):
        run = Run()
        self.open_dial(run, TWO, "desktop")
        out = run.move(0.4, (0.5, 0.6), (0.5 - g.DIAL_STEP * 1.5, 0.6), **TWO)
        self.assertEqual(out, [SwitchDesktop(-1)])

    def test_moving_hand_opens_no_dial(self):
        """Swipes are gone: a hand moving across, fast or slow, does nothing."""
        for seconds in (0.33, 1.5):
            run = Run()
            self.assertEqual(run.move(seconds, (0.8, 0.6), (0.3, 0.6), **PALM), [])

    def test_steps_only_with_its_shape(self):
        run = Run()
        self.open_dial(run, PALM, "window")
        out = run.move(0.4, (0.5, 0.6), (0.7, 0.6), **ONE)  # another shape holds still
        self.assertEqual(out, [])

    def test_one_step_per_frame(self):
        """A jump (tracking hop) doesn't flip through many tabs at once."""
        run = Run()
        self.open_dial(run)
        out = run.frames(DT, at=(0.9, 0.6), **ONE)
        self.assertEqual(out, [SwitchTab(1)])

    def test_jitter_does_not_step(self):
        run = Run()
        self.open_dial(run)
        out = []
        for k in range(30):
            run.t += DT
            out += run.engine.feed(hand(at=(0.5 + g.DIAL_STEP * (0.5 + 0.1 * (-1) ** k), 0.6), **ONE), run.t)
        self.assertNotIn(SwitchTab, kinds(out))

    def test_fist_or_hand_away_ends_the_dial(self):
        run = Run()
        self.open_dial(run)
        self.assertIn(DialClose, kinds(run.frames(DT, **FIST)))
        run.none(g.COOLDOWN_S + 0.2)
        self.open_dial(run)
        self.assertEqual(run.none(g.DIAL_LOST_S - 0.2), [])
        self.assertIn(DialClose, kinds(run.none(0.4)))

    def test_hand_dropping_after_the_dial_opens_nothing(self):
        run = Run()
        self.open_dial(run, PALM, "window")
        run.frames(DT, **FIST)
        self.assertEqual(run.frames(g.COOLDOWN_S - 0.1, **PALM), [])

    def test_idle_dial_ends(self):
        run = Run()
        self.open_dial(run)
        self.assertIn(DialClose, kinds(run.frames(g.DIAL_IDLE_S + 0.2, **ONE)))

    def test_no_close_in_the_dial(self):
        run = Run()
        self.open_dial(run)
        out = run.frames(g.CLOSE_HOLD_S + 0.2, **THUMBS_DOWN)
        self.assertEqual(set(kinds(out)) & {CloseWindow, CloseArming}, set())


class CloseTests(unittest.TestCase):
    def test_thumbs_down_hold_closes(self):
        run = Run()
        out = run.frames(g.CLOSE_HOLD_S + 0.2, **THUMBS_DOWN)
        arming = [e for e in out if isinstance(e, CloseArming)]
        self.assertGreater(len(arming), 3)
        self.assertLess(arming[0].progress, 0.2)
        self.assertEqual(kinds(out).count(CloseWindow), 1)

    def test_recognizer_thumb_down_closes(self):
        run = Run()
        out = run.frames(g.CLOSE_HOLD_S + 0.2, label="Thumb_Down", **FIST)
        self.assertIn(CloseWindow, kinds(out))

    def test_thumbs_up_does_not_close(self):
        run = Run()
        out = run.frames(g.CLOSE_HOLD_S + 0.5, **THUMBS_UP)
        self.assertEqual(set(kinds(out)) & {CloseWindow, CloseArming}, set())

    def test_short_thumbs_down_cancels(self):
        run = Run()
        out = run.frames(g.CLOSE_HOLD_S / 2, **THUMBS_DOWN)
        out += run.frames(0.5, **FIST)
        self.assertIn(CloseCancel, kinds(out))
        self.assertNotIn(CloseWindow, kinds(out))

    def test_one_dropped_frame_does_not_cancel(self):
        run = Run()
        run.frames(0.4, **THUMBS_DOWN)
        run.none(DT)
        out = run.frames(g.CLOSE_HOLD_S, **THUMBS_DOWN)
        self.assertIn(CloseWindow, kinds(out))
        self.assertNotIn(CloseCancel, kinds(run.events))

    def test_no_second_close_while_still_held(self):
        run = Run()
        run.frames(g.CLOSE_HOLD_S + 0.2, **THUMBS_DOWN)
        out = run.frames(g.CLOSE_HOLD_S * 2, **THUMBS_DOWN)
        self.assertNotIn(CloseWindow, kinds(out))

    def test_flickering_label_does_not_close_twice(self):
        """Seen live: a held thumbs down, its label dipping under the score
        cut for a frame now and then (read as a fist), closed two windows."""
        run = Run()
        out = []
        for _ in range(8):
            out += run.frames(0.4, label="Thumb_Down", **FIST)
            out += run.frames(DT, **FIST)
        self.assertEqual(kinds(out).count(CloseWindow), 1)

    def test_thumb_lowered_then_down_again_closes_again(self):
        run = Run()
        run.frames(g.CLOSE_HOLD_S + 0.2, **THUMBS_DOWN)
        run.none(max(g.CLOSE_REARM_S, g.COOLDOWN_S) + 0.1)
        out = run.frames(g.CLOSE_HOLD_S + 0.2, **THUMBS_DOWN)
        self.assertEqual(kinds(out).count(CloseWindow), 1)

    def test_menu_ignores_thumbs_down(self):
        run = Run()
        run.open_menu()
        out = run.frames(g.CLOSE_HOLD_S + 0.3, **THUMBS_DOWN)
        self.assertNotIn(CloseWindow, kinds(out))


class NoiseTests(unittest.TestCase):
    def test_jittery_palm_steps_nothing(self):
        run = Run()
        out = []
        for k in range(60):
            run.t += DT
            jitter = 0.01 * math.sin(k * 2.1)
            out += run.engine.feed(hand(at=(0.5 + jitter, 0.6 - jitter), **PALM), run.t)
        self.assertEqual(out, [DialOpen("window")])

    def test_no_hand_fires_nothing(self):
        self.assertEqual(Run().none(3), [])

    def test_reset_clears_state(self):
        run = Run()
        run.open_menu()
        run.engine.reset()
        self.assertEqual(run.frames(0.2, **FIST), [])


if __name__ == "__main__":
    unittest.main()
