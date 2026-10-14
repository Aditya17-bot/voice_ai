"""Hand control glue: gesture events -> overlay and actions (no camera, no Tk)."""

import tkinter as tk
import unittest

from assistant.gestures import (
    CloseArming,
    CloseCancel,
    CloseWindow,
    DialClose,
    DialOpen,
    Hover,
    MenuClose,
    MenuOpen,
    Pick,
    SwitchDesktop,
    SwitchTab,
    SwitchWindow,
)
from assistant.handmenu import HandControl, tiles_for
from assistant.hands import HandsUnavailable
from assistant.tools import ToolError
from assistant.windows import WindowError


class FakeWorkspace:
    root_names = ["adi", "downloads"]  # no desktop on this machine

    def __init__(self):
        self.calls = []

    def open_website(self, site):
        self.calls.append(("site", site))
        return f"Opening {site}."

    def open_app(self, app):
        if app == "claude":
            raise ToolError("I can't find the Claude app on this computer.")
        self.calls.append(("app", app))
        return f"Opening {app}."

    def open_in_explorer(self, root):
        self.calls.append(("folder", root))
        return f"Opening {root}."


class FakeOverlay:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args: self.calls.append((name, *args))


class FakeWindows:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.protected = False

    def front(self):
        return 7, "Notepad"

    def can_close(self, hwnd):
        return not self.protected

    def close_front(self, expected=None):
        self.calls.append(("close", expected))
        return "Notepad"

    def switch_window(self, step):
        if self.fail:
            raise WindowError("Windows didn't let me switch.")
        self.calls.append(("window", step))
        return "Chrome"

    def switch_desktop(self, step):
        self.calls.append(("desktop", step))

    def switch_tab(self, step):
        self.calls.append(("tab", step))
        return "Chrome"


class FakeTracker:
    instances: list["FakeTracker"] = []
    fail = False

    def __init__(self, on_points, on_stop):
        self.on_points, self.on_stop = on_points, on_stop
        self.started = self.stopped = False
        FakeTracker.instances.append(self)

    def start(self):
        if FakeTracker.fail:
            raise HandsUnavailable("Hand control needs MediaPipe.")
        self.started = True

    def stop(self):
        self.stopped = True

    def join(self, timeout):
        self.joined = timeout


class HandControlTest(unittest.TestCase):
    def setUp(self):
        FakeTracker.instances = []
        FakeTracker.fail = False
        self.posted = []
        self.logged = []
        self.states = []
        self.ws = FakeWorkspace()
        self.overlay = FakeOverlay()
        self.windows = FakeWindows()
        self.hc = HandControl(
            post=self.posted.append,
            log=lambda kind, text: self.logged.append(text),
            workspace=lambda: self.ws,
            on_state=self.states.append,
            overlay=lambda: self.overlay,
            windows=self.windows,
            tracker=FakeTracker,
        )

    def labels(self):
        return [t.label for t in self.hc.tiles]

    def test_tiles_skip_missing_folders(self):
        labels = [t.label for t in tiles_for(self.ws)]
        self.assertEqual(
            labels, ["YouTube", "Gmail", "GitHub", "Chrome", "Terminal", "VS Code", "Claude", "adi", "Downloads"]
        )

    def test_start_and_stop(self):
        self.assertIsNone(self.hc.start())
        self.assertTrue(FakeTracker.instances[-1].started)
        self.assertEqual(self.states, [True])
        self.hc.stop()
        self.assertTrue(FakeTracker.instances[-1].stopped)

    def test_start_reports_missing_packages(self):
        FakeTracker.fail = True
        self.assertIn("MediaPipe", self.hc.start())
        self.assertEqual(self.states, [False])

    def test_camera_failure_turns_off(self):
        self.hc.start()
        FakeTracker.instances[-1].on_stop("I can't open the camera.")
        for call in self.posted:
            call()
        self.assertEqual(self.states, [True, False])
        self.assertIn("I can't open the camera.", self.logged[-1])

    def run_posted(self):
        calls, self.posted[:] = list(self.posted), []
        for call in calls:
            call()

    def test_old_camera_stopping_late_does_not_turn_off_the_new_one(self):
        self.hc.start()
        old = FakeTracker.instances[-1]
        self.hc.stop()
        self.hc.start()
        new = FakeTracker.instances[-1]
        old.on_stop(None)  # the first camera thread only now finishes
        self.run_posted()
        self.assertTrue(self.hc.on)
        self.assertEqual(self.states, [True, False, True])
        self.hc.stop()
        self.assertTrue(new.stopped)

    def test_old_camera_events_are_dropped(self):
        self.hc.start()
        old = FakeTracker.instances[-1]
        self.hc.stop()
        self.hc.start()
        from tests.test_gestures import THUMBS_UP, hand
        for k in range(25):
            old.on_points(hand(**THUMBS_UP), 1.0 + k / 15, None)
        self.run_posted()
        self.assertNotIn("show_menu", [c[0] for c in self.overlay.calls])

    def test_camera_points_become_overlay_drawing(self):
        self.hc.start()
        from tests.test_gestures import THUMBS_UP, hand
        for k in range(25):
            FakeTracker.instances[-1].on_points(hand(**THUMBS_UP), 1.0 + k / 15, "Thumb_Up")
        self.run_posted()
        self.assertIn("show_menu", [c[0] for c in self.overlay.calls])

    def test_hand_view_follows_the_camera(self):
        self.hc.start()
        from tests.test_gestures import PALM, hand
        tracker = FakeTracker.instances[-1]
        tracker.on_points(hand(**PALM), 1.0, "Open_Palm")
        for k in range(5):
            tracker.on_points(None, 1.1 + k / 10, None)
        self.run_posted()
        drawn = [c for c in self.overlay.calls if c[0] == "show_hand"]
        self.assertEqual(drawn[0][2], "palm")
        self.assertEqual(drawn[1:], [("show_hand", None, "")])  # "no hand" drawn once, not every frame

    def test_menu_pick_opens_tile(self):
        self.hc.start()
        self.hc.handle(MenuOpen())
        self.hc.handle(Hover(0))
        self.hc.handle(Pick(0))
        self.hc.handle(MenuClose())
        self.assertEqual(self.ws.calls, [("site", "youtube")])
        self.assertEqual(self.overlay.calls[0], ("show_menu", self.labels()))
        self.assertIn(("hover", 0), self.overlay.calls)
        self.assertIn(("hide_menu",), self.overlay.calls)
        self.assertIn("Opening youtube.", self.logged[-1])

    def test_folder_tile_opens_explorer(self):
        self.hc.start()
        self.hc.handle(Pick(self.labels().index("Downloads")))
        self.assertEqual(self.ws.calls, [("folder", "downloads")])

    def test_tool_error_is_logged(self):
        self.hc.start()
        self.hc.handle(Pick(self.labels().index("Claude")))
        self.assertIn("can't find the Claude app", self.logged[-1])

    def test_switches(self):
        self.hc.start()
        self.hc.handle(SwitchWindow(1))
        self.hc.handle(SwitchDesktop(-1))
        self.hc.handle(SwitchTab(1))
        self.assertEqual(self.windows.calls, [("window", 1), ("desktop", -1), ("tab", 1)])
        self.assertTrue(any(c[0] == "toast" for c in self.overlay.calls))

    def test_tab_dial(self):
        self.hc.start()
        self.hc.handle(DialOpen("tab"))
        self.hc.handle(SwitchTab(1))
        self.hc.handle(SwitchTab(-1))
        self.hc.handle(DialClose())
        self.assertEqual(self.windows.calls, [("tab", 1), ("tab", -1)])
        names = [c[0] for c in self.overlay.calls]
        self.assertLess(names.index("show_dial"), names.index("hide_dial"))
        self.assertIn(("show_dial", "tab"), self.overlay.calls)

    def test_window_error_is_logged(self):
        self.hc.start()
        self.windows.fail = True
        self.hc.handle(SwitchWindow(1))
        self.assertIn("didn't let me switch", self.logged[-1])

    def test_close_flow(self):
        self.hc.start()
        self.hc.handle(CloseArming(0.0))
        self.hc.handle(CloseArming(0.5))
        self.hc.handle(CloseWindow())
        self.assertEqual(self.windows.calls, [("close", 7)])
        self.assertIn(("close_progress", "Notepad", 0.5), self.overlay.calls)
        self.assertIn(("close_clear",), self.overlay.calls)
        self.assertIn("Closed Notepad", self.logged[-1])

    def test_close_cancel(self):
        self.hc.start()
        self.hc.handle(CloseArming(0.2))
        self.hc.handle(CloseCancel())
        self.assertEqual(self.windows.calls, [])
        self.assertIn(("close_clear",), self.overlay.calls)

    def test_stop_mid_close_forgets_the_window(self):
        self.hc.start()
        self.hc.handle(CloseArming(0.5))
        self.hc.stop()
        self.hc.start()
        self.hc.handle(CloseWindow())
        self.assertEqual(self.windows.calls, [])  # never "whatever is in front now"

    def test_protected_window_says_so_at_once(self):
        self.hc.start()
        self.windows.protected = True
        self.hc.handle(CloseArming(0.0))
        self.hc.handle(CloseArming(0.5))
        self.hc.handle(CloseWindow())
        self.assertEqual(self.windows.calls, [])
        self.assertNotIn("close_progress", [c[0] for c in self.overlay.calls])
        self.assertEqual([c for c in self.overlay.calls if c[0] == "toast"], [("toast", "can't close this one")])

    def test_overlay_failure_leaves_camera_off(self):
        def broken():
            raise tk.TclError("no display")

        self.hc._make_overlay = broken
        self.assertIsNotNone(self.hc.start())
        self.assertFalse(self.hc.on)
        self.assertEqual(FakeTracker.instances, [])  # the camera was never started

    def test_quick_restart_waits_for_old_camera(self):
        self.hc.start()
        old = FakeTracker.instances[-1]
        self.hc.stop()
        self.hc.start()
        self.assertTrue(getattr(old, "joined", None))

    def test_events_after_stop_are_ignored(self):
        self.hc.start()
        self.hc.stop()
        self.hc.handle(Pick(0))
        self.assertEqual(self.ws.calls, [])


if __name__ == "__main__":
    unittest.main()
