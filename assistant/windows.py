"""Window control for hand gestures: close the front window, switch windows,
switch tabs, switch virtual desktops (Windows only, through ctypes).

Kept narrow on purpose:
- Close posts WM_CLOSE, the same as clicking the window's X: the app still
  asks to save, and may refuse. Only the window the overlay named (if
  another came to the front meanwhile, nothing closes). Never the desktop,
  the taskbar, our own windows, or the terminal / editor we were started
  from (closing it would kill the assistant).
- Switching windows only brings an existing window to the front.
- The only keystrokes ever sent are Win+Ctrl+Left/Right (switch desktop) and
  Ctrl+Tab / Ctrl+Shift+Tab (switch tab), with the left Ctrl so the Right
  Ctrl talk key never sees them. No typing.

Window titles are shown in the overlay and the chat only; they never reach
the LLM, an action, or memory.
"""

import os
import sys
from typing import Protocol

VK_LWIN = 0x5B
VK_LCONTROL = 0xA2
VK_LEFT = 0x25
VK_RIGHT = 0x27
VK_TAB = 0x09
VK_LSHIFT = 0xA0
MAX_TITLE = 60

# The desktop, its wallpaper workers, and the taskbars.
SHELL_CLASSES = {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"}


class WindowError(Exception):
    """A message safe to show in the chat."""


class WinApi(Protocol):
    def foreground(self) -> int: ...
    def root(self, hwnd: int) -> int: ...
    def title(self, hwnd: int) -> str: ...
    def class_name(self, hwnd: int) -> str: ...
    def pid(self, hwnd: int) -> int: ...
    def visible(self, hwnd: int) -> bool: ...
    def owner(self, hwnd: int) -> int: ...
    def tool_window(self, hwnd: int) -> bool: ...
    def cloaked(self, hwnd: int) -> bool: ...
    def top_level_windows(self) -> list[int]: ...
    def post_close(self, hwnd: int) -> bool: ...
    def protected_pids(self) -> set[int]: ...
    def activate(self, hwnd: int) -> bool: ...
    def send_to_back(self, hwnd: int) -> None: ...
    def send_keys(self, keys: list[tuple[int, bool]]) -> None: ...


class WindowControl:
    def __init__(
        self, api: WinApi | None = None, protected_pids: set[int] | None = None, own_pid: int | None = None
    ) -> None:
        """protected_pids: processes whose windows are never closed; by
        default ours and the ones that started us. Our own windows (orb,
        overlay) are also never switched to."""
        self._api = api or Win32Api()
        self._protected = self._api.protected_pids() if protected_pids is None else protected_pids
        self._own_pid = os.getpid() if own_pid is None else own_pid

    def front(self) -> tuple[int, str]:
        """(handle, short title) of the front window; (0, "") if none."""
        hwnd = self._front()
        return (hwnd, _short(self._api.title(hwnd))) if hwnd else (0, "")

    def front_title(self) -> str:
        return self.front()[1]

    def close_front(self, expected: int | None = None) -> str:
        """Ask the front window to close; returns its title. With `expected`,
        only if that window is still the one in front."""
        hwnd = self._front()
        if expected is not None and hwnd != expected:
            raise WindowError("A different window came to the front, so I didn't close anything.")
        if not hwnd or not self._closable(hwnd):
            raise WindowError("There's no window in front I can close.")
        title = _short(self._api.title(hwnd)) or "the window"
        if not self._api.post_close(hwnd):
            raise WindowError(f"Windows wouldn't let me close {title}.")
        return title

    def switch_window(self, step: int) -> str:
        """Bring the next (+1) or previous (-1) window to the front; returns its title.

        "Next" also sends the old front window to the back, so repeated
        steps cycle through every window instead of flipping between two.
        """
        windows = self._alt_tab()
        front = self._front()
        others = [w for w in windows if w != front]
        if not others:
            raise WindowError("There's no other window to switch to.")
        if front in windows:
            i = windows.index(front)
            target = windows[(i + step) % len(windows)]
        else:
            target = windows[0] if step > 0 else windows[-1]
        if not self._api.activate(target):
            raise WindowError("Windows didn't let me switch.")
        if step > 0 and front in windows:
            self._api.send_to_back(front)
        return _short(self._api.title(target))

    def switch_desktop(self, step: int) -> None:
        arrow = VK_RIGHT if step > 0 else VK_LEFT
        self._chord([VK_LWIN, VK_LCONTROL, arrow])

    def switch_tab(self, step: int) -> str:
        """Next (+1) or previous (-1) tab in the front window (Ctrl+Tab /
        Ctrl+Shift+Tab: browsers, editors, terminals); returns its title."""
        hwnd = self._front()
        if not hwnd or self._api.class_name(hwnd) in SHELL_CLASSES:
            raise WindowError("There's no window in front to switch tabs in.")
        self._chord([VK_LCONTROL, VK_TAB] if step > 0 else [VK_LCONTROL, VK_LSHIFT, VK_TAB])
        return _short(self._api.title(hwnd)) or "the window"

    def _chord(self, combo: list[int]) -> None:
        self._api.send_keys([(k, False) for k in combo] + [(k, True) for k in reversed(combo)])

    # ----- helpers -------------------------------------------------------

    def _front(self) -> int:
        hwnd = self._api.foreground()
        return self._api.root(hwnd) if hwnd else 0

    def can_close(self, hwnd: int) -> bool:
        """Whether close_front would close this window (when it's in front)."""
        return bool(hwnd) and self._closable(hwnd)

    def _closable(self, hwnd: int) -> bool:
        return (
            self._api.visible(hwnd)
            and self._api.class_name(hwnd) not in SHELL_CLASSES
            and self._api.pid(hwnd) not in self._protected
        )

    def _alt_tab(self) -> list[int]:
        """Windows Alt+Tab would show on this desktop, front first, minus ours."""
        api = self._api
        return [
            hwnd
            for hwnd in api.top_level_windows()
            if api.visible(hwnd)
            and not api.owner(hwnd)
            and not api.tool_window(hwnd)
            and api.title(hwnd).strip()
            and not api.cloaked(hwnd)  # on another desktop, or a suspended app
            and api.class_name(hwnd) not in SHELL_CLASSES
            and api.pid(hwnd) != self._own_pid
        ]


def _short(title: str) -> str:
    title = " ".join(title.split())
    return title if len(title) <= MAX_TITLE else title[: MAX_TITLE - 1] + "…"


class Win32Api:
    """The real thing: user32 and dwmapi through ctypes."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise WindowError("Window control needs Windows.")
        import ctypes
        from ctypes import wintypes

        self._ct = ctypes
        self._wt = wintypes
        u = ctypes.WinDLL("user32", use_last_error=True)
        H = wintypes.HWND
        sigs = {
            "GetForegroundWindow": ([], H),
            "GetAncestor": ([H, wintypes.UINT], H),
            "GetWindowTextLengthW": ([H], ctypes.c_int),
            "GetWindowTextW": ([H, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int),
            "GetClassNameW": ([H, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int),
            "GetWindowThreadProcessId": ([H, ctypes.POINTER(wintypes.DWORD)], wintypes.DWORD),
            "IsWindowVisible": ([H], wintypes.BOOL),
            "IsIconic": ([H], wintypes.BOOL),
            "GetWindow": ([H, wintypes.UINT], H),
            "GetWindowLongPtrW": ([H, ctypes.c_int], ctypes.c_ssize_t),
            "PostMessageW": ([H, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM], wintypes.BOOL),
            "ShowWindow": ([H, ctypes.c_int], wintypes.BOOL),
            "BringWindowToTop": ([H], wintypes.BOOL),
            "SetForegroundWindow": ([H], wintypes.BOOL),
            "SwitchToThisWindow": ([H, wintypes.BOOL], None),
            "AttachThreadInput": ([wintypes.DWORD, wintypes.DWORD, wintypes.BOOL], wintypes.BOOL),
            "SetWindowPos": ([H, H, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT],
                             wintypes.BOOL),
            "keybd_event": ([wintypes.BYTE, wintypes.BYTE, wintypes.DWORD, ctypes.c_size_t], None),
        }
        for name, (args, res) in sigs.items():
            fn = getattr(u, name)
            fn.argtypes, fn.restype = args, res
        self._u = u
        self._enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, H, wintypes.LPARAM)
        u.EnumWindows.argtypes = [self._enum_proc, wintypes.LPARAM]
        u.EnumWindows.restype = wintypes.BOOL
        self._dwm = ctypes.WinDLL("dwmapi")
        self._dwm.DwmGetWindowAttribute.argtypes = [H, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
        self._dwm.DwmGetWindowAttribute.restype = ctypes.c_long
        self._this_thread = ctypes.WinDLL("kernel32").GetCurrentThreadId

    def foreground(self) -> int:
        return self._u.GetForegroundWindow() or 0

    def root(self, hwnd: int) -> int:
        return self._u.GetAncestor(hwnd, 2) or hwnd  # GA_ROOT

    def title(self, hwnd: int) -> str:
        n = self._u.GetWindowTextLengthW(hwnd)
        buf = self._ct.create_unicode_buffer(n + 1)
        self._u.GetWindowTextW(hwnd, buf, n + 1)
        return buf.value

    def class_name(self, hwnd: int) -> str:
        buf = self._ct.create_unicode_buffer(256)
        self._u.GetClassNameW(hwnd, buf, 256)
        return buf.value

    def pid(self, hwnd: int) -> int:
        pid = self._wt.DWORD()
        self._u.GetWindowThreadProcessId(hwnd, self._ct.byref(pid))
        return pid.value

    def visible(self, hwnd: int) -> bool:
        return bool(self._u.IsWindowVisible(hwnd))

    def owner(self, hwnd: int) -> int:
        return self._u.GetWindow(hwnd, 4) or 0  # GW_OWNER

    def tool_window(self, hwnd: int) -> bool:
        ex = self._u.GetWindowLongPtrW(hwnd, -20)  # GWL_EXSTYLE
        tool, app, no_activate = 0x80, 0x40000, 0x08000000
        return bool(ex & no_activate) or (bool(ex & tool) and not ex & app)

    def cloaked(self, hwnd: int) -> bool:
        value = self._wt.DWORD()
        ok = self._dwm.DwmGetWindowAttribute(hwnd, 14, self._ct.byref(value), 4) == 0  # DWMWA_CLOAKED
        return ok and value.value != 0

    def top_level_windows(self) -> list[int]:
        found: list[int] = []

        def add(hwnd, _):
            found.append(hwnd)
            return True

        self._u.EnumWindows(self._enum_proc(add), 0)
        return found

    def post_close(self, hwnd: int) -> bool:
        # Fails for windows of programs running as administrator.
        return bool(self._u.PostMessageW(hwnd, 0x0010, 0, 0))  # WM_CLOSE

    def protected_pids(self) -> set[int]:
        """Us, our console window, and our parent processes (the terminal or
        VS Code we were started from), up to but not including Explorer,
        whose windows are ordinary folders."""
        ct, wt = self._ct, self._wt
        protected = {os.getpid()}
        kernel32 = ct.WinDLL("kernel32", use_last_error=True)
        kernel32.GetConsoleWindow.restype = wt.HWND
        console = kernel32.GetConsoleWindow()
        if console:
            protected.add(self.pid(console))

        class Entry(ct.Structure):
            _fields_ = [
                ("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ProcessID", wt.DWORD),
                ("th32DefaultHeapID", ct.c_size_t), ("th32ModuleID", wt.DWORD), ("cntThreads", wt.DWORD),
                ("th32ParentProcessID", wt.DWORD), ("pcPriClassBase", ct.c_long), ("dwFlags", wt.DWORD),
                ("szExeFile", ct.c_wchar * 260),
            ]

        kernel32.CreateToolhelp32Snapshot.restype = wt.HANDLE
        kernel32.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
        kernel32.Process32FirstW.argtypes = kernel32.Process32NextW.argtypes = [wt.HANDLE, ct.POINTER(Entry)]
        kernel32.CloseHandle.argtypes = [wt.HANDLE]
        snapshot = kernel32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
        if not snapshot or snapshot == wt.HANDLE(-1).value:
            return protected
        parents: dict[int, tuple[int, str]] = {}
        try:
            entry = Entry()
            entry.dwSize = ct.sizeof(Entry)
            ok = kernel32.Process32FirstW(snapshot, ct.byref(entry))
            while ok:
                parents[entry.th32ProcessID] = (entry.th32ParentProcessID, entry.szExeFile.lower())
                ok = kernel32.Process32NextW(snapshot, ct.byref(entry))
        finally:
            kernel32.CloseHandle(snapshot)
        pid = os.getpid()
        for _ in range(12):
            parent = parents.get(pid, (0, ""))[0]
            if not parent or parent in protected or parent not in parents:
                break
            if parents[parent][1] == "explorer.exe":
                break
            protected.add(parent)
            pid = parent
        return protected

    def activate(self, hwnd: int) -> bool:
        if self._u.IsIconic(hwnd):
            self._u.ShowWindow(hwnd, 9)  # SW_RESTORE
        # Windows only lets the foreground app hand over the focus; joining
        # its input queue for a moment makes us count as part of it.
        front = self._u.GetForegroundWindow()
        theirs = self._u.GetWindowThreadProcessId(front, None) if front else 0
        ours = self._this_thread()
        joined = bool(theirs and theirs != ours and self._u.AttachThreadInput(ours, theirs, True))
        try:
            self._u.BringWindowToTop(hwnd)
            self._u.SetForegroundWindow(hwnd)
        finally:
            if joined:
                self._u.AttachThreadInput(ours, theirs, False)
        if self._u.GetForegroundWindow() != hwnd:
            self._u.SwitchToThisWindow(hwnd, True)  # what Alt+Tab itself uses
        return self.root(self._u.GetForegroundWindow() or 0) == hwnd

    def send_to_back(self, hwnd: int) -> None:
        self._u.SetWindowPos(hwnd, 1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)  # HWND_BOTTOM, no size/move/activate

    def send_keys(self, keys: list[tuple[int, bool]]) -> None:
        extended = {VK_LWIN, VK_LEFT, VK_RIGHT}
        for vk, up in keys:
            flags = (0x0001 if vk in extended else 0) | (0x0002 if up else 0)  # EXTENDEDKEY, KEYUP
            self._u.keybd_event(vk, 0, flags, 0)
