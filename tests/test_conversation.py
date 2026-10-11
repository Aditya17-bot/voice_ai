"""Multi-turn behaviour: answering "did you mean X or Y?", follow-up questions
about the file just explained, and how names are spoken. No Ollama needed.

Run: python -m unittest tests.test_conversation
"""

import tempfile
import time
import unittest
from pathlib import Path

from assistant.main import Assistant, speakable
from assistant.router import pick_option, refers_back
from assistant.tools import ToolError, Workspace
from tests.fakes import FakeBrain, FakeFallback, FakePlayer, FakeTTS


class PickOptionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.options = [Path("/adi/voice_input"), Path("/adi/voice_notes")]

    def test_by_order_or_name(self) -> None:
        cases = {
            "The first one.": self.options[0],
            "The second one, please.": self.options[1],
            "The other.": self.options[1],
            "Voice input.": self.options[0],
            "I meant voice notes.": self.options[1],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(pick_option(text, self.options), expected)

    def test_unclear_answers_pick_nothing(self) -> None:
        for text in ("Yes.", "Both.", "Tell me a joke.", "Voice input or voice notes.", "The first or second."):
            with self.subTest(text=text):
                self.assertIsNone(pick_option(text, self.options))

    def test_new_requests_and_negations_are_not_answers(self) -> None:
        for text in (
            "Go to downloads.",
            "Not the first one.",
            "The first time I ran it, it crashed.",
            "What was the last thing you said?",
            "No, some other file.",
            "Open voice input in VS Code.",
        ):
            with self.subTest(text=text):
                self.assertIsNone(pick_option(text, self.options))

    def test_single_offer_needs_a_plain_yes(self) -> None:
        offer = [Path("/adi/voice_input/assistant/main.py")]
        for text in ("Yes.", "Yes, please.", "Sure.", "Okay, that one."):
            with self.subTest(text=text):
                self.assertEqual(pick_option(text, offer), offer[0])
        for text in ("That is wrong.", "Yeah, nah.", "That doesn't matter.", "Sure, but first open Chrome.", "No."):
            with self.subTest(text=text):
                self.assertIsNone(pick_option(text, offer))

    def test_same_name_told_apart_by_folder(self) -> None:
        options = [Path("/adi/school/notes"), Path("/adi/work/notes")]
        self.assertEqual(pick_option("The one in work.", options), options[1])


class RefersBackTest(unittest.TestCase):
    def test_follow_ups(self) -> None:
        for text in (
            "What else does it do?",
            "Tell me more.",
            "Explain the code again.",
            "What does it say about errors?",
            "What is this file for?",
            "Explain it again.",
        ):
            with self.subTest(text=text):
                self.assertTrue(refers_back(text))

    def test_fresh_questions(self) -> None:
        for text in (
            "Tell me a joke.",
            "What's the capital of France?",
            "Why is that?",
            "Is it going to rain?",
            "What time is it?",
            "What's this song?",
            "Thanks, that's all.",
            "What does the brain.py file do?",  # names another file
        ):
            with self.subTest(text=text):
                self.assertFalse(refers_back(text))


class SpeakableTest(unittest.TestCase):
    def test_names_sound_natural_and_numbers_stay(self) -> None:
        self.assertEqual(
            speakable("Now in voice_input. It has main.py, README.md."),
            "Now in voice input. It has main dot py, README dot md.",
        )
        self.assertEqual(speakable("Version 3.5 is out."), "Version 3.5 is out.")

    def test_abbreviations_and_dunders(self) -> None:
        self.assertEqual(speakable("e.g. at 9 a.m. in the U.S.A."), "e.g. at 9 a.m. in the U.S.A.")
        self.assertEqual(speakable("__init__.py"), "dunder init dot py")


class ConversationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.adi = Path(self._tmp.name) / "adi"
        for folder in ("voice_input", "voice_notes"):
            (self.adi / folder).mkdir(parents=True)
        (self.adi / "voice_input" / "README.md").write_text("# Voice\nAn offline assistant.\n")
        (self.adi / "voice_input" / "main.py").write_text("print('hi')\n")
        self.ws = Workspace(roots={"adi": self.adi}, launcher=lambda *a: None, open_url=lambda u: None)
        self.player = FakePlayer()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _session(self, brain: FakeBrain) -> Assistant:
        return Assistant(brain, FakeTTS(), self.player, FakeFallback(), self.ws)

    def test_answering_did_you_mean(self) -> None:
        assistant = self._session(FakeBrain("none"))
        assistant.respond("cd voice", time.perf_counter())
        self.assertEqual(self.player.said, ["Did you mean voice input or voice notes?"])
        assistant.respond("The second one.", time.perf_counter())
        self.assertEqual(self.ws.cwd, (self.adi / "voice_notes").resolve())

    def test_unrelated_answer_is_a_fresh_request(self) -> None:
        brain = FakeBrain("none", ["Here's one."], ["The first what?"])
        assistant = self._session(brain)
        assistant.respond("cd voice", time.perf_counter())
        assistant.respond("Tell me a joke.", time.perf_counter())
        self.assertEqual(self.player.said[-1], "Here's one.")
        self.assertEqual(self.ws.cwd, self.adi.resolve())
        # The question is only open for one turn.
        assistant.respond("The first one.", time.perf_counter())
        self.assertEqual(self.ws.cwd, self.adi.resolve())

    def test_follow_up_reuses_the_file(self) -> None:
        brain = FakeBrain("none", ["It's a readme."], ["It mentions being offline."])
        assistant = self._session(brain)
        self.ws.go_to_folder("voice input")
        assistant.respond("Read the readme.", time.perf_counter())
        assistant.respond("What else does it say?", time.perf_counter())
        self.assertIn("An offline assistant", brain.streamed[1]["context"])
        self.assertEqual(brain.streamed[1]["text"], "What else does it say?")

    def test_follow_up_window_closes(self) -> None:
        brain = FakeBrain("none", ["A readme."], ["Ha."], ["Paris."], ["Hmm."], ["Sure."])
        assistant = self._session(brain)
        self.ws.go_to_folder("voice input")
        assistant.respond("Read the readme.", time.perf_counter())
        assistant.respond("Tell me a joke.", time.perf_counter())
        assistant.respond("What's the capital of France?", time.perf_counter())
        assistant.respond("Tell me more about it.", time.perf_counter())
        self.assertIsNone(brain.streamed[1]["context"])  # no back-reference
        self.assertIsNone(brain.streamed[3]["context"])  # three turns later: stale

    def test_moving_on_forgets_the_file(self) -> None:
        brain = FakeBrain("none", ["A readme."], ["Hmm."])
        assistant = self._session(brain)
        self.ws.go_to_folder("voice input")
        assistant.respond("Read the readme.", time.perf_counter())
        assistant.respond("Go up.", time.perf_counter())
        assistant.respond("What else does it say?", time.perf_counter())
        self.assertIsNone(brain.streamed[1]["context"])

    def test_file_in_a_subfolder_is_offered(self) -> None:
        (self.adi / "voice_input" / "assistant").mkdir()
        (self.adi / "voice_input" / "assistant" / "stt.py").write_text("# speech to text\n")
        brain = FakeBrain("none", ["It turns speech into text."])
        assistant = self._session(brain)
        self.ws.go_to_folder("voice input")
        assistant.respond("What does stt.py do?", time.perf_counter())
        self.assertEqual(
            self.player.said, ["stt dot py isn't here, but there's one in assistant.", "Want that one?"]
        )
        assistant.respond("Yes, please.", time.perf_counter())
        self.assertEqual(brain.streamed[0]["text"], "What does stt.py do?")
        self.assertIn("speech to text", brain.streamed[0]["context"])
        self.assertEqual(self.ws.cwd, (self.adi / "voice_input").resolve())  # reading doesn't move

    def test_declining_the_offer(self) -> None:
        (self.adi / "voice_input" / "assistant").mkdir()
        (self.adi / "voice_input" / "assistant" / "stt.py").write_text("# speech to text\n")
        brain = FakeBrain("none", ["Okay."])
        assistant = self._session(brain)
        self.ws.go_to_folder("voice input")
        assistant.respond("What does stt.py do?", time.perf_counter())
        assistant.respond("No, not that one.", time.perf_counter())
        self.assertIsNone(brain.streamed[0]["context"])

    def test_explicit_command_beats_a_pending_question(self) -> None:
        assistant = self._session(FakeBrain("none"))
        assistant.respond("cd voice", time.perf_counter())
        assistant.respond("Where am I?", time.perf_counter())
        self.assertEqual(self.player.said[-1], "We're in the adi folder.")
        self.assertEqual(self.ws.cwd, self.adi.resolve())

    def test_loose_question_without_extension_is_not_offered(self) -> None:
        (self.adi / "web" / "deploy").mkdir(parents=True)
        (self.adi / "web" / "deploy" / "docker-compose.yml").write_text("services: {}\n")
        brain = FakeBrain("none", ["It runs several containers together."])
        assistant = self._session(brain)
        self.ws.go_to_folder("web")
        assistant.respond("What does docker compose do?", time.perf_counter())
        self.assertEqual(self.player.said, ["It runs several containers together."])
        self.assertIsNone(brain.streamed[0]["context"])

    def test_subfolder_search_skips_hidden_and_private_and_has_no_folder_cap(self) -> None:
        project = self.adi / "voice_input"
        for name in (".git", "keys"):
            (project / name).mkdir()
        (project / ".git" / "config.txt").write_text("token\n")
        (project / "keys" / "secret.txt").write_text("hunter2\n")
        for i in range(40):
            (project / f"sub{i:02d}").mkdir()
        (project / "sub35" / "target.py").write_text("x = 1\n")
        self.ws.go_to_folder("voice input")
        self.assertEqual(self.ws._files_one_level_down("config.txt"), [])
        self.assertEqual(self.ws._files_one_level_down("secret.txt"), [])
        self.assertEqual(
            self.ws._files_one_level_down("target.py"), [(project / "sub35" / "target.py").resolve()]
        )
        with self.assertRaises(ToolError):
            self.ws.read(project / ".git" / "config.txt")

    def test_declined_offer_forgets_the_old_file(self) -> None:
        (self.adi / "voice_input" / "assistant").mkdir()
        (self.adi / "voice_input" / "assistant" / "brain.py").write_text("# llm\n")
        brain = FakeBrain("none", ["It prints hi."], ["No problem."], ["Not sure what you mean."])
        assistant = self._session(brain)
        self.ws.go_to_folder("voice input")
        assistant.respond("What does main.py do?", time.perf_counter())
        assistant.respond("What does brain.py do?", time.perf_counter())  # offered from assistant/
        assistant.respond("No.", time.perf_counter())
        assistant.respond("What does it do?", time.perf_counter())
        self.assertIsNone(brain.streamed[2]["context"])

    def test_follow_up_window_is_not_extended_by_use(self) -> None:
        brain = FakeBrain("none", ["A readme."], ["More."], ["Even more."], ["Hmm."])
        assistant = self._session(brain)
        self.ws.go_to_folder("voice input")
        assistant.respond("Read the readme.", time.perf_counter())
        assistant.respond("Tell me more.", time.perf_counter())
        assistant.respond("Tell me more.", time.perf_counter())
        assistant.respond("Tell me more.", time.perf_counter())
        self.assertIsNotNone(brain.streamed[2]["context"])  # two turns after
        self.assertIsNone(brain.streamed[3]["context"])  # three turns after

    def test_interrupt_drops_the_rest_of_the_reply(self) -> None:
        assistant = None

        class InterruptedBrain(FakeBrain):
            def stream_reply(self, text, context=None):
                yield "First sentence."
                assistant.interrupt()  # the user pressed Stop while it talked
                yield "Second sentence."
                yield "Third sentence."

        self.player.stop = lambda: None
        assistant = self._session(InterruptedBrain("none"))
        assistant.respond("Tell me a story.", time.perf_counter())
        self.assertEqual(self.player.said, ["First sentence."])
        # The next request is answered normally.
        assistant._brain = FakeBrain("none", ["Sure."])
        assistant.respond("Tell me a joke.", time.perf_counter())
        self.assertEqual(self.player.said[-1], "Sure.")

    def test_listing_is_spoken_with_friendly_names(self) -> None:
        assistant = self._session(FakeBrain("none"))
        self.ws.go_to_folder("voice input")
        assistant.respond("List the files.", time.perf_counter())
        self.assertEqual(self.player.said, ["Voice input, inside adi has 2 files: main dot py, README dot md."])


if __name__ == "__main__":
    unittest.main()
