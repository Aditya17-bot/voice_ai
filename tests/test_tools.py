"""Unit tests for the read-only tool layer. No apps are actually launched.

Run: python -m unittest tests.test_tools
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from assistant import tools
from assistant.tools import ToolError, Workspace


class WorkspaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.adi = base / "adi"
        self.downloads = base / "Downloads"
        self.outside = base / "outside"
        for folder in (
            self.adi / "voice_input" / "assistant",
            self.adi / "budget_app",
            self.adi / ".git",
            self.adi / ".ssh",
            self.downloads,
            self.outside,
        ):
            folder.mkdir(parents=True)
        (self.adi / "voice_input" / "README.md").write_text("# Voice assistant\nOffline.")
        (self.adi / "voice_input" / "main.py").write_text("print('hi')\n")
        (self.adi / "voice_input" / "big.txt").write_text("x" * (tools.MAX_READ_BYTES + 50))
        (self.adi / "voice_input" / "model.bin").write_bytes(b"\0\1\2binary")
        (self.adi / "voice_input" / ".env").write_text("API_KEY=nope")
        (self.adi / "voice_input" / "secrets.json").write_text("{}")
        (self.outside / "notes.txt").write_text("outside")

        self.launched: list[tuple[list[str], Path | None, bool]] = []
        self.urls: list[str] = []
        self.ws = Workspace(
            roots={"adi": self.adi, "downloads": self.downloads},
            launcher=lambda argv, cwd, new: self.launched.append((argv, cwd, new)),
            open_url=self.urls.append,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # ----- navigation --------------------------------------------------

    def test_starts_in_first_root(self) -> None:
        self.assertEqual(self.ws.cwd, self.adi.resolve())

    def test_spoken_folder_name_fuzzy_matches(self) -> None:
        self.ws.go_to_folder("the Voice Input folder.")
        self.assertEqual(self.ws.cwd.name, "voice_input")

    def test_root_reachable_from_anywhere(self) -> None:
        self.ws.go_to_folder("voice input")
        self.ws.go_to_folder("downloads")
        self.assertEqual(self.ws.cwd, self.downloads.resolve())

    def test_root_child_reachable_from_other_root(self) -> None:
        self.ws.go_to_folder("downloads")
        self.ws.go_to_folder("budget app")
        self.assertEqual(self.ws.cwd.name, "budget_app")

    def test_up_stops_at_root(self) -> None:
        self.ws.go_to_folder("voice input")
        self.ws.go_to_folder("up")
        self.assertEqual(self.ws.cwd, self.adi.resolve())
        with self.assertRaises(ToolError):
            self.ws.go_to_folder("go back")

    def test_unknown_folder_refused(self) -> None:
        with self.assertRaises(ToolError):
            self.ws.go_to_folder("zebra")

    def test_hidden_and_sensitive_folders_not_reachable(self) -> None:
        with self.assertRaises(ToolError):
            self.ws.go_to_folder("ssh")
        with self.assertRaises(ToolError):
            self.ws.go_to_folder("git")

    def test_guard_blocks_outside_and_traversal(self) -> None:
        with self.assertRaises(ToolError):
            self.ws._guard(self.outside)
        with self.assertRaises(ToolError):
            self.ws._guard(self.adi / ".." / "outside")
        with self.assertRaises(ToolError):
            self.ws._guard(self.adi / ".ssh" / "id_rsa")

    # ----- reading -----------------------------------------------------

    def test_listing_hides_dotfiles_and_secrets(self) -> None:
        self.ws.go_to_folder("voice input")
        listing = self.ws.list_folder()
        self.assertIn("main.py", listing)
        self.assertIn("assistant", listing)
        self.assertNotIn(".env", listing)
        self.assertNotIn("secrets.json", listing)

    def test_read_file_by_spoken_name(self) -> None:
        self.ws.go_to_folder("voice input")
        self.assertIn("print('hi')", self.ws.read_file("main dot py"))
        self.assertIn("print('hi')", self.ws.read_file("main"))

    def test_read_file_truncates(self) -> None:
        self.ws.go_to_folder("voice input")
        result = self.ws.read_file("big.txt")
        self.assertIn("only the beginning", result)
        self.assertLess(len(result), tools.MAX_READ_BYTES + 200)

    def test_binary_and_sensitive_files_refused(self) -> None:
        self.ws.go_to_folder("voice input")
        with self.assertRaises(ToolError):
            self.ws.read_file("model.bin")
        with self.assertRaises(ToolError):
            self.ws.read_file(".env")
        with self.assertRaises(ToolError):
            self.ws.read_file("secrets")

    def test_explain_folder_includes_readme(self) -> None:
        self.ws.go_to_folder("voice input")
        self.assertIn("Offline.", self.ws.explain_folder())

    def test_private_key_content_refused(self) -> None:
        (self.adi / "notes.txt").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nabc")
        with self.assertRaises(ToolError):
            self.ws.read_file("notes")

    def test_utf16_text_readable(self) -> None:
        (self.adi / "log.txt").write_bytes("hello from powershell".encode("utf-16"))
        self.assertIn("hello from powershell", self.ws.read_file("log"))

    # ----- matching ----------------------------------------------------

    def test_ambiguous_name_asks(self) -> None:
        self.ws.go_to_folder("voice input")
        (self.ws.cwd / "main.txt").write_text("notes")
        with self.assertRaisesRegex(ToolError, "Did you mean"):
            self.ws.read_file("main")
        self.assertIn("print('hi')", self.ws.read_file("main.py"))

    def test_prefix_match(self) -> None:
        self.ws.go_to_folder("voice")
        self.assertEqual(self.ws.cwd.name, "voice_input")

    def test_short_near_miss_not_matched(self) -> None:
        (self.adi / "srv").mkdir()
        with self.assertRaises(ToolError):
            self.ws.go_to_folder("src")

    def test_exact_root_beats_fuzzy_subfolder(self) -> None:
        (self.adi / "adi2").mkdir()
        self.ws.go_to_folder("voice input")
        self.ws.go_to_folder("adi")
        self.assertEqual(self.ws.cwd, self.adi.resolve())

    def test_cross_root_jump_needs_exact_name(self) -> None:
        self.ws.go_to_folder("downloads")
        with self.assertRaises(ToolError):
            self.ws.go_to_folder("budget")  # only a prefix of budget_app
        self.ws.go_to_folder("budget app")
        self.assertEqual(self.ws.cwd.name, "budget_app")

    def test_junction_escape_hidden(self) -> None:
        try:
            import _winapi

            _winapi.CreateJunction(str(self.outside), str(self.adi / "shortcut"))
        except (ImportError, OSError) as exc:
            self.skipTest(f"junctions unavailable: {exc}")
        self.assertNotIn("shortcut", self.ws.list_folder())
        with self.assertRaises(ToolError):
            self.ws.go_to_folder("shortcut")

    # ----- launching ---------------------------------------------------

    def test_open_app_aliases(self) -> None:
        with mock.patch.object(tools, "_find_chrome", return_value="chrome.exe"):
            self.assertEqual(self.ws.open_app("Google Chrome"), "Opening Chrome.")
        self.assertEqual(self.launched[-1][0], ["chrome.exe", "--profile-directory=Default"])

    def test_explorer_not_allowed(self) -> None:
        with self.assertRaises(ToolError):
            self.ws.open_app("explorer")
        with self.assertRaises(ToolError):
            self.ws.open_app("notepad")
        self.assertEqual(self.launched, [])

    def _fake_install(self, *parts: str) -> Path:
        exe = Path(self._tmp.name, "localappdata", *parts)
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_bytes(b"")
        return exe

    def test_open_here_uses_absolute_wt(self) -> None:
        wt = self._fake_install("Microsoft", "WindowsApps", "wt.exe")
        self.ws.go_to_folder("voice input")
        with mock.patch.dict(os.environ, {"LOCALAPPDATA": str(wt.parents[2])}):
            self.ws.open_here("terminal")
        argv, cwd, _ = self.launched[-1]
        self.assertEqual(argv, [str(wt), "-d", str(self.ws.cwd)])
        self.assertEqual(cwd, self.ws.cwd)

    def test_vscode_found_by_install_path(self) -> None:
        code = self._fake_install("Programs", "Microsoft VS Code", "Code.exe")
        with mock.patch.dict(os.environ, {"LOCALAPPDATA": str(code.parents[2])}):
            self.ws.open_here("vs code")
        self.assertEqual(self.launched[-1][0], [str(code), str(self.ws.cwd)])

    def test_open_site_allowlist(self) -> None:
        with mock.patch.object(tools, "_find_chrome", return_value=None):
            self.assertEqual(self.ws.open_website("G mail"), "Opening Gmail.")
            self.assertEqual(self.ws.open_website("youtube.com"), "Opening YouTube.")
        self.assertEqual(self.urls, ["https://mail.google.com", "https://www.youtube.com"])

    def test_search_encodes_the_query(self) -> None:
        with mock.patch.object(tools, "_find_chrome", return_value="chrome.exe"):
            said = self.ws.search("youtube", "lofi & chill?  music")
        self.assertEqual(said, "Searching YouTube for lofi & chill? music.")
        self.assertEqual(
            self.launched[-1][0],
            ["chrome.exe", "--profile-directory=Default", "https://www.youtube.com/results?search_query=lofi+%26+chill%3F+music"],
        )

    def test_search_only_known_sites_and_bounded_query(self) -> None:
        with self.assertRaises(ToolError):
            self.ws.search("evil.example", "x")
        with self.assertRaises(ToolError):
            self.ws.search("google", "   ")
        with mock.patch.object(tools, "_find_chrome", return_value=None):
            self.ws.search("google", "a" * 500)
        self.assertEqual(len(self.urls[-1]), len("https://www.google.com/search?q=") + tools.MAX_QUERY_CHARS)

    def test_chrome_profile_by_name(self) -> None:
        profiles = {
            "Default": ["Aditya", "Aditya"],
            "Profile 2": ["Robert", "Robert"],
            "Profile 4": ["Aditya", "Aditya"],
            "Profile 6": ["vitstudent.ac.in"],
            "Profile 7": ["vitstudent.ac.in", "Aditya", "Aditya Sridhar"],
        }
        self.ws._profiles = lambda: profiles
        resolve = lambda name: tools.resolve_chrome_profile(name, profiles)  # noqa: E731
        self.assertEqual(resolve("robert"), "Profile 2")
        self.assertEqual(resolve("Aditya"), "Default")  # exactly the usual one's name
        self.assertEqual(resolve("aditya sridhar"), "Profile 7")  # full name is exact
        self.assertEqual(resolve("profile 4"), "Profile 4")  # folder names work too
        with self.assertRaises(ToolError) as caught:
            resolve("vit")  # two VIT profiles, neither the usual one
        self.assertIn("Profile 6 or Profile 7", str(caught.exception))
        with self.assertRaises(ToolError):
            resolve("adit")  # a prefix tie never quietly picks the usual one
        with self.assertRaises(ToolError) as caught:
            resolve("bob")
        self.assertIn("Robert", str(caught.exception))
        with mock.patch.object(tools, "_find_chrome", return_value="chrome.exe"):
            self.assertEqual(self.ws.open_app("chrome", "robert"), "Opening Chrome as robert.")
            self.ws.open_website("gmail", "robert")
        self.assertEqual(self.launched[-2][0], ["chrome.exe", "--profile-directory=Profile 2"])
        self.assertEqual(self.launched[-1][0], ["chrome.exe", "--profile-directory=Profile 2", "https://mail.google.com"])
        with self.assertRaises(ToolError):
            self.ws.open_app("terminal", "robert")

    def test_other_urls_refused(self) -> None:
        # Arbitrary URLs would let text read from a file make the model send
        # data to any host, so only the allowlisted sites open.
        for bad in (
            "example.com",
            "https://evil.example/?data=secret",
            "file:///C:/secret.txt",
            "javascript:alert(1)",
            "--disable-web-security",
        ):
            with self.subTest(bad=bad), self.assertRaises(ToolError):
                self.ws.open_website(bad)
        self.assertEqual(self.urls, [])


if __name__ == "__main__":
    unittest.main()
