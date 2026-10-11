"""Unit tests for the fast-path router. Utterances are written the way Whisper
transcribes them (capitalized, with punctuation).

Run: python -m unittest tests.test_router
"""

import tempfile
import unittest
from pathlib import Path

from assistant.router import Intent, Outcome, execute, route
from assistant.tools import ToolError, Workspace

ROUTED = {
    # apps
    "Open Chrome.": Intent("open_app", "chrome"),
    "Okay, open Google Chrome please.": Intent("open_app", "google chrome"),
    "Can you open the terminal?": Intent("open_app", "terminal"),
    "Launch VS Code.": Intent("open_app", "vs code"),
    "Open a terminal here.": Intent("open_here", "terminal"),
    "Open VS Code in this folder.": Intent("open_here", "vscode"),
    "Open this folder in VS Code.": Intent("open_here", "vscode"),
    # sites
    "Open Gmail.": Intent("open_website", "gmail"),
    "Open YouTube.": Intent("open_website", "youtube"),
    "Pull up GitHub.": Intent("open_website", "github"),
    "Open my email.": Intent("open_website", "gmail"),
    "Go to YouTube.": Intent("open_website", "youtube"),
    # navigation: explicit (fuzzy, refuses on a miss) vs loose (exact or LLM)
    "Go to the voice input folder.": Intent("go_to_folder", "voice input folder"),
    "Change directory to assistant.": Intent("go_to_folder", "assistant"),
    "cd tests": Intent("go_to_folder", "tests"),
    "Open the budget app folder.": Intent("go_to_folder", "budget app folder"),
    "Open the GitHub website.": Intent("open_website", "github"),
    "Navigate to downloads.": Intent("go_to_folder", "downloads", loose=True),
    "Take me to the desktop.": Intent("go_to_folder", "desktop", loose=True),
    "Go back.": Intent("go_to_folder", "up"),
    "Go up one folder.": Intent("go_to_folder", "up"),
    "Where am I?": Intent("where"),
    # looking around
    "What's in this folder?": Intent("list_folder"),
    "List the files.": Intent("list_folder"),
    "Show me the files.": Intent("list_folder"),
    "Explain this folder.": Intent("explain_folder"),
    "What is this project about?": Intent("explain_folder"),
    "Summarize this project.": Intent("explain_folder"),
    "Explain this project in one sentence.": Intent("explain_folder"),
    # files
    "Read main.py.": Intent("read_file", "main.py"),
    "Open main.py.": Intent("read_file", "main.py"),
    "Explain main dot py.": Intent("read_file", "main dot py", loose=True),
    "Explain main.py in simple terms.": Intent("read_file", "main.py", loose=True),
    "What does main.py do?": Intent("read_file", "main.py", loose=True),
    "Explain the readme file.": Intent("read_file", "readme", loose=True),
    "What's in requirements.txt?": Intent("read_file", "requirements.txt", loose=True),
    "Read readme.": Intent("read_file", "readme", loose=True),
    # unknown "open X" resolved at execution time
    "Open voice input.": Intent("open_any", "voice input"),
    "Start over.": Intent("open_any", "over", loose=True),
}

# Must reach the LLM — either no pattern matches, or a loose match finds
# nothing on disk.
CONVERSATION = [
    "What's the weather like today?",
    "Tell me a joke.",
    "How do I open a file in Python?",
    "Can you explain recursion?",
    "Explain how photosynthesis works.",
    "What is the capital of France?",
    "I went to the store yesterday.",
    "Should I go to bed?",
    "Back.",
    "Start over.",
    "Start a timer for five minutes.",
    "Open up about your day.",
    "Go to sleep.",
    "Take me to the airport.",
    "Explain node.js.",
    "What's in vue.js?",
    "Explain the file system.",
    "Describe the file format of PNG.",
    "Read me a story.",
    "What does a compiler do?",
    "Go to school.",
    "",
    "   ",
]


class RouteTest(unittest.TestCase):
    def test_commands_routed(self) -> None:
        for utterance, expected in ROUTED.items():
            with self.subTest(utterance=utterance):
                self.assertEqual(route(utterance), expected)


class ExecuteTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.adi = Path(self._tmp.name) / "adi"
        (self.adi / "voice_input").mkdir(parents=True)
        (self.adi / "windows").mkdir()
        (self.adi / "school_2019").mkdir()
        (self.adi / "voice_input" / "README.md").write_text("# Voice\nOffline assistant.")
        (self.adi / "voice_input" / "main.py").write_text("print('hi')\n")
        self.launched: list[list[str]] = []
        self.ws = Workspace(
            roots={"adi": self.adi},
            launcher=lambda argv, cwd, new: self.launched.append(argv),
            open_url=lambda url: None,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _handle(self, text: str) -> Outcome | None:
        intent = route(text)
        return execute(intent, self.ws) if intent else None

    def test_conversation_reaches_llm(self) -> None:
        start = self.ws.cwd
        for utterance in CONVERSATION:
            with self.subTest(utterance=utterance):
                self.assertIsNone(self._handle(utterance))
        self.assertEqual(self.ws.cwd, start)
        self.assertEqual(self.launched, [])

    def test_navigation_spoken_directly(self) -> None:
        outcome = self._handle("Go to voice input.")
        self.assertFalse(outcome.explain)
        self.assertEqual(outcome.text, "Now in voice_input, inside adi.")
        self.assertEqual(self._handle("Where am I?").text, "We're in voice_input, inside adi.")

    def test_explicit_navigation_is_fuzzy(self) -> None:
        self.assertIn("voice_input", self._handle("Go to the voice in put folder.").text)

    def test_file_goes_to_llm(self) -> None:
        self.ws.go_to_folder("voice input")
        for utterance in ("Read the readme file.", "What does main.py do?", "Read readme."):
            with self.subTest(utterance=utterance):
                outcome = self._handle(utterance)
                self.assertTrue(outcome.explain)

    def test_open_any_tries_folder_then_file(self) -> None:
        self.assertFalse(self._handle("Open voice input.").explain)
        self.assertTrue(self._handle("Open readme.").explain)

    def test_near_miss_never_moves(self) -> None:
        # "window" is close to the folder "windows" but must not jump there.
        with self.assertRaisesRegex(ToolError, "I can open Chrome"):
            self._handle("Open the window.")
        self.assertEqual(self.ws.cwd, self.adi.resolve())

    def test_explicit_miss_is_spoken(self) -> None:
        with self.assertRaisesRegex(ToolError, "I can open Chrome"):
            self._handle("Open Spotify.")
        with self.assertRaisesRegex(ToolError, "couldn't find a folder"):
            self._handle("cd zebra")
        with self.assertRaisesRegex(ToolError, "couldn't find a file"):
            self._handle("Read notes.txt.")
        self.assertEqual(self.launched, [])

    def test_listing_is_separate_sentences(self) -> None:
        self.ws.go_to_folder("voice input")
        (self.ws.cwd / "assistant").mkdir()
        text = self._handle("What's in this folder?").text
        self.assertEqual(
            text,
            "Voice_input, inside adi has 1 folder: assistant. "
            "It has 2 files: main.py, README.md.",
        )


if __name__ == "__main__":
    unittest.main()
