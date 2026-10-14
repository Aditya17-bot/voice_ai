"""Window control rules, against a fake Win32 API (no real windows touched)."""

import unittest
from dataclasses import dataclass

from assistant.windows import (
    VK_LCONTROL,
    VK_LEFT,
    VK_LSHIFT,
    VK_LWIN,
    VK_RIGHT,
    VK_TAB,
    WindowControl,
    WindowError,
)

OWN_PID = 4242
TERMINAL_PID = 77  # the terminal (or VS Code) the assistant was started from


@dataclass
class W:
    hwnd: int
    title: str = "Window"
    cls: str = "AppWindow"
    pid: int = 100
    visible: bool = True
    owner: int = 0
    tool: bool = False
    cloaked: bool = False
    minimized: bool = False


class FakeApi:
    def __init__(self, windows, front=None):
        self.windows = list(windows)  # z-order, top first
        self.front = front if front is not None else (self.windows[0].hwnd if self.windows else 0)
        self.closed: list[int] = []
        self.keys: list[tuple[int, bool]] = []
        self.restored: list[int] = []
        self.refuse_focus = False
        self.refuse_close = False

    def _w(self, hwnd):
        return next(w for w in self.windows if w.hwnd == hwnd)

    def foreground(self):
        return self.front

    def root(self, hwnd):
        return hwnd

    def title(self, hwnd):
        return self._w(hwnd).title

    def class_name(self, hwnd):
        return self._w(hwnd).cls

    def pid(self, hwnd):
        return self._w(hwnd).pid

    def visible(self, hwnd):
        return self._w(hwnd).visible

    def owner(self, hwnd):
        return self._w(hwnd).owner

    def tool_window(self, hwnd):
        return self._w(hwnd).tool

    def cloaked(self, hwnd):
        return self._w(hwnd).cloaked

    def top_level_windows(self):
        return [w.hwnd for w in self.windows]

    def post_close(self, hwnd):
        if self.refuse_close:
            return False
        self.closed.append(hwnd)
        return True

    def activate(self, hwnd):
        if self._w(hwnd).minimized:
            self.restored.append(hwnd)
        if self.refuse_focus:
            return False
        w = self._w(hwnd)
        self.windows.remove(w)
        self.windows.insert(0, w)
        self.front = hwnd
        return True

    def send_to_back(self, hwnd):
        w = self._w(hwnd)
        self.windows.remove(w)
        self.windows.append(w)

    def send_keys(self, keys):
        self.keys += keys


def control(*windows, front=None):
    api = FakeApi(windows, front)
    return WindowControl(api, protected_pids={OWN_PID, TERMINAL_PID}, own_pid=OWN_PID), api


class CloseTests(unittest.TestCase):
    def test_closes_front_window_politely(self):
        wc, api = control(W(1, "Notepad"), W(2, "Chrome"))
        self.assertEqual(wc.close_front(), "Notepad")
        self.assertEqual(api.closed, [1])

    def test_refuses_desktop_and_taskbar(self):
        for cls in ("Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"):
            wc, api = control(W(1, "", cls=cls))
            with self.assertRaises(WindowError):
                wc.close_front()
            self.assertEqual(api.closed, [])

    def test_refuses_own_windows(self):
        wc, api = control(W(1, "Jarvis", pid=OWN_PID))
        with self.assertRaises(WindowError):
            wc.close_front()
        self.assertEqual(api.closed, [])

    def test_refuses_the_terminal_we_run_in(self):
        wc, api = control(W(1, "PowerShell", pid=TERMINAL_PID))
        with self.assertRaises(WindowError):
            wc.close_front()
        self.assertEqual(api.closed, [])

    def test_closes_only_the_window_it_named(self):
        wc, api = control(W(1, "Notepad"), W(2, "Chrome"))
        hwnd, title = wc.front()
        self.assertEqual((hwnd, title), (1, "Notepad"))
        api.front = 2  # something else came to the front during the hold
        with self.assertRaises(WindowError):
            wc.close_front(expected=hwnd)
        self.assertEqual(api.closed, [])
        api.front = 1
        self.assertEqual(wc.close_front(expected=hwnd), "Notepad")
        self.assertEqual(api.closed, [1])

    def test_can_close_tells_ahead(self):
        wc, api = control(W(1, "Notepad"), W(2, "PowerShell", pid=TERMINAL_PID), W(3, "", cls="Progman"))
        self.assertEqual([wc.can_close(h) for h in (1, 2, 3, 0)], [True, False, False, False])

    def test_close_refused_by_windows(self):
        wc, api = control(W(1, "Task Manager"))
        api.refuse_close = True
        with self.assertRaises(WindowError):
            wc.close_front()

    def test_nothing_in_front(self):
        wc, api = control(front=0)
        with self.assertRaises(WindowError):
            wc.close_front()

    def test_title_is_shortened(self):
        wc, _ = control(W(1, "x" * 200))
        self.assertLessEqual(len(wc.front_title()), 60)


class SwitchTests(unittest.TestCase):
    def test_next_brings_second_window_and_cycles(self):
        wc, api = control(W(1, "A"), W(2, "B"), W(3, "C"))
        self.assertEqual(wc.switch_window(1), "B")
        self.assertEqual(wc.switch_window(1), "C")
        self.assertEqual(wc.switch_window(1), "A")
        self.assertEqual(wc.switch_window(1), "B")

    def test_previous_undoes_next(self):
        wc, api = control(W(1, "A"), W(2, "B"), W(3, "C"))
        wc.switch_window(1)
        self.assertEqual(wc.switch_window(-1), "A")

    def test_previous_goes_to_bottom(self):
        wc, api = control(W(1, "A"), W(2, "B"), W(3, "C"))
        self.assertEqual(wc.switch_window(-1), "C")

    def test_skips_windows_not_in_alt_tab(self):
        wc, api = control(
            W(1, "A"),
            W(2, "hidden", visible=False),
            W(3, "dialog", owner=1),
            W(4, "tool", tool=True),
            W(5, "", cls="Untitled"),
            W(6, "other desktop", cloaked=True),
            W(7, "Jarvis", pid=OWN_PID),
            W(8, "", cls="Shell_TrayWnd"),
            W(9, "B"),
        )
        self.assertEqual(wc.switch_window(1), "B")

    def test_can_switch_to_our_terminal(self):
        """Closing it is refused, but bringing it to the front is fine."""
        wc, api = control(W(1, "A"), W(2, "PowerShell", pid=TERMINAL_PID))
        self.assertEqual(wc.switch_window(1), "PowerShell")

    def test_from_desktop_goes_to_top_window(self):
        wc, api = control(W(1, "", cls="Progman"), W(2, "A"), W(3, "B"), front=1)
        self.assertEqual(wc.switch_window(1), "A")

    def test_restores_minimized(self):
        wc, api = control(W(1, "A"), W(2, "B", minimized=True))
        wc.switch_window(1)
        self.assertEqual(api.restored, [2])

    def test_only_one_window(self):
        wc, api = control(W(1, "A"))
        with self.assertRaises(WindowError):
            wc.switch_window(1)

    def test_focus_refused(self):
        wc, api = control(W(1, "A"), W(2, "B"))
        api.refuse_focus = True
        with self.assertRaises(WindowError):
            wc.switch_window(1)


class DesktopTests(unittest.TestCase):
    def test_next_desktop_keys(self):
        wc, api = control(W(1, "A"))
        wc.switch_desktop(1)
        self.assertEqual(
            api.keys,
            [(VK_LWIN, False), (VK_LCONTROL, False), (VK_RIGHT, False),
             (VK_RIGHT, True), (VK_LCONTROL, True), (VK_LWIN, True)],
        )

    def test_previous_desktop_keys(self):
        wc, api = control(W(1, "A"))
        wc.switch_desktop(-1)
        self.assertEqual([k for k, up in api.keys if not up], [VK_LWIN, VK_LCONTROL, VK_LEFT])


class TabTests(unittest.TestCase):
    def test_next_tab_keys(self):
        wc, api = control(W(1, "Chrome"))
        self.assertEqual(wc.switch_tab(1), "Chrome")
        self.assertEqual(api.keys, [(VK_LCONTROL, False), (VK_TAB, False), (VK_TAB, True), (VK_LCONTROL, True)])

    def test_previous_tab_keys(self):
        wc, api = control(W(1, "Chrome"))
        wc.switch_tab(-1)
        self.assertEqual([k for k, up in api.keys if not up], [VK_LCONTROL, VK_LSHIFT, VK_TAB])
        self.assertEqual([k for k, up in api.keys if up], [VK_TAB, VK_LSHIFT, VK_LCONTROL])

    def test_no_keys_to_the_desktop(self):
        wc, api = control(W(1, "", cls="Progman"))
        with self.assertRaises(WindowError):
            wc.switch_tab(1)
        self.assertEqual(api.keys, [])


if __name__ == "__main__":
    unittest.main()
