"""Unit tests for the fast-path router. Utterances are written the way Whisper
transcribes them (capitalized, with punctuation).

Run: python -m unittest tests.test_router
"""

import tempfile
import unittest
from pathlib import Path

from assistant.router import Intent, Outcome, execute, route, wants_to_quit
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
    "Start over.": Intent("clear_chat"),
    "Start the engine.": Intent("open_any", "engine", loose=True),
    # voice console
    "Go hands free.": Intent("ui", "hands_free_on"),
    "Push to talk mode.": Intent("ui", "hands_free_off"),
    "Clear the conversation.": Intent("clear_chat"),
    "Full screen.": Intent("ui", "fullscreen_on"),
    "Exit full screen.": Intent("ui", "fullscreen_off"),
    "Change your face.": Intent("ui", "next_face"),
    "Hide.": Intent("ui", "hide"),
    "Come back.": Intent("ui", "show"),
    "Camera on.": Intent("ui", "camera_on"),
    "Turn on the camera.": Intent("ui", "camera_on"),
    "Start hand control.": Intent("ui", "camera_on"),
    "Gestures on.": Intent("ui", "camera_on"),
    "Turn off the camera.": Intent("ui", "camera_off"),
    "Hand control off.": Intent("ui", "camera_off"),
    "Stop gestures.": Intent("ui", "camera_off"),
    "Open Claude.": Intent("open_app", "claude"),
    "Open the Claude app.": Intent("open_app", "claude"),
    # "check X"
    "Check the patents folder.": Intent("go_to_folder", "patents folder", loose=True),
    "Check my email.": Intent("open_website", "gmail"),
    "Check out YouTube.": Intent("open_website", "youtube"),
    "Check patents.": Intent("open_any", "patents", loose=True),
    # searches: the query is the user's words
    "Search YouTube for lofi music.": Intent("search", "lofi music", site="youtube"),
    "Can you search for some rust tutorials on YouTube?": Intent("search", "rust tutorials", site="youtube"),
    "Look up voice assistants on GitHub.": Intent("search", "voice assistants", site="github"),
    "Google pasta recipes.": Intent("search", "pasta recipes", site="google"),
    "Search for the best pizza near me.": Intent("search", "best pizza near me", site="google"),
    # YouTube: play by search, pick from the last search
    "Play Coldplay on YouTube.": Intent("play_video", "coldplay"),
    "Open a music video on YouTube.": Intent("play_video", "music video"),
    "Open on YouTube Clash Royale.": Intent("play_video", "clash royale"),
    "Find an anime video.": Intent("search", "anime video", site="youtube"),
    "Play some lofi music.": Intent("play_video", "lofi music"),
    "Click on the first video.": Intent("pick_video", "1"),
    "Play the second one.": Intent("pick_video", "2"),
    "Open video number 3.": Intent("pick_video", "3"),
    "Play the next one.": Intent("pick_video", "next"),
    "The third one.": Intent("pick", "3", loose=True),
    "Read the second one.": Intent("pick", "2"),
    "Open the second one.": Intent("pick", "2"),
    # memory
    "Remember that my exam is on Friday.": Intent("remember", "my exam is on friday"),
    "What do you remember about me?": Intent("recall"),
    "Forget about my exam.": Intent("forget", "my exam"),
    "Forget everything.": Intent("forget_all"),
    # Gmail
    "How many emails did I get today?": Intent("email_today"),
    "Do I have any new emails?": Intent("email_unread"),
    "What are my latest emails?": Intent("email_latest"),
    "Read my unread emails.": Intent("email_latest", "unread"),
    "Find the email from Amazon.": Intent("email_from", "amazon"),
    "Did I get any emails from my professor today?": Intent("email_from", "my professor"),
    "Any emails about the internship?": Intent("email_about", "the internship"),
    "Read the latest email.": Intent("read_email", "latest"),
    "Check the weather in Paris.": Intent("search", "weather paris", site="google"),
    "Read the third email.": Intent("read_email", "3"),
    # GitHub
    "List my repos.": Intent("gh_repos"),
    "What's new on my voice AI repo?": Intent("gh_new", "voice ai"),
    "Any open issues on my voice AI repo?": Intent("gh_issues", "voice ai"),
    "Any pull requests on the portfolio repo?": Intent("gh_pulls", "portfolio"),
    "How many stars does my voice AI repo have?": Intent("gh_stats", "voice ai"),
    "What's new on voice ai on GitHub?": Intent("gh_new", "voice ai"),
    "What time is it in Tokyo?": Intent("time_in", "tokyo"),
    "What's the weather in Paris?": Intent("search", "weather paris", site="google"),
    "Will it rain tomorrow?": Intent("search", "weather", site="google"),
    "Forget the conversation.": Intent("clear_chat"),
    "What is my stonks repo about?": Intent("gh_readme", "stonks"),
    "What are the videos?": Intent("list_videos"),
    # a named folder, help, time
    "Tell me what you see in downloads folder.": Intent("look_in", "downloads"),
    "What's in my patents folder?": Intent("look_in", "patents"),
    "What's in downloads?": Intent("look_in", "downloads", loose=True),
    "List the files in the assistant folder.": Intent("look_in", "assistant"),
    "Find the folder named Adi.": Intent("go_to_folder", "adi"),
    "List of files.": Intent("list_folder"),
    "What can you do?": Intent("help"),
    "What time is it?": Intent("time"),
    "What's the date today?": Intent("date"),
    "Search YouTube for lofi in Robert's profile.": Intent("search", "lofi", site="youtube", profile="robert"),
    "Google dress as a pirate.": Intent("search", "dress as a pirate", site="google"),
    # Chrome profiles
    "Open Chrome as Robert.": Intent("open_app", "chrome", profile="robert"),
    "Open YouTube with my VIT profile.": Intent("open_website", "youtube", profile="vit"),
    "Open Gmail in Robert's account.": Intent("open_website", "gmail", profile="robert"),
    "Open Chrome as profile 4.": Intent("open_app", "chrome", profile="profile 4"),
    "Search GitHub for jobs on my account.": Intent("search", "jobs on my account", site="github"),
    "Search YouTube for sign in to my account.": Intent("search", "sign in to my account", site="youtube"),
}

# Must reach the LLM — either no pattern matches, or a loose match finds
# nothing on disk.
CONVERSATION = [
    "Which camera should I buy?",
    "Is my camera any good for video calls?",
    "Who made Claude?",
    "Tell me a joke.",
    "How do I open a file in Python?",
    "Can you explain recursion?",
    "Explain how photosynthesis works.",
    "What is the capital of France?",
    "I went to the store yesterday.",
    "Should I go to bed?",
    "Back.",
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
    "Check the terminal.",
    "Check on Chrome.",
    "Google Chrome.",
    "Search for the budget file.",
    "Play a game with me.",
    "Find my keys.",
    "Search me.",
    "Google is a company.",
    "Google Drive.",
    "Search my notes.",
    "Search for my notes.",
    "Search the project for TODO.",
    "Open YouTube as fast as possible.",
    "What's in the news today?",
    "Find the video files.",
    "Search for jazz music on my computer.",
    "Do you remember the movie we talked about?",
    "Remember me.",
    "Forget it.",
    "Forget about it.",
    "Remember to buy milk.",
    "What's new on YouTube?",
    "Any issues on my mind?",
    "I forgot my keys.",
    "",
    "   ",
]


class QuitTest(unittest.TestCase):
    def test_quit_phrases_alone(self) -> None:
        for text in ("Stop.", "Goodbye!", "Okay, stop listening.", "Quit.", "Exit, please."):
            with self.subTest(text=text):
                self.assertTrue(wants_to_quit(text))

    def test_quit_word_inside_a_request_is_not_a_quit(self) -> None:
        for text in (
            "Stop the music.",
            "Don't stop.",
            "Say goodbye to my friend.",
            "How do I exit vim?",
            "That's all.",  # answers "anything else?"
            "Bye.",  # Whisper's favourite word for noise
        ):
            with self.subTest(text=text):
                self.assertFalse(wants_to_quit(text))


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
        # "windws" is close to the folder "windows" but must not jump there.
        with self.assertRaisesRegex(ToolError, "I can open Chrome"):
            self._handle("Open windws.")
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
