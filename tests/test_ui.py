"""The window's background session: requests run in order, "stop" quits,
Stop interrupts. No window, mic, or models: the assistant is a stub.

Run: python -m unittest tests.test_ui
"""

import queue
import time
import unittest

from assistant.ui import Session


class StubPlayer:
    busy = False
    level = 0.0

    def wait_until_done(self, timeout=None):
        return True


class StubAssistant:
    def __init__(self) -> None:
        self.player = StubPlayer()
        self.calls: list[tuple[str, str]] = []

    def respond(self, text, started_at):
        self.calls.append(("respond", text))

    def say(self, text):
        self.calls.append(("say", text))

    def interrupt(self):
        self.calls.append(("interrupt", ""))


class SessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.events: queue.Queue = queue.Queue()
        self.logged: list[tuple[str, str]] = []
        self.assistant = StubAssistant()
        self.session = Session(self.assistant, None, None, self.events, lambda k, t: self.logged.append((k, t)))
        self.addCleanup(self.session.close)

    def _settle(self) -> None:
        deadline = time.monotonic() + 2
        while (self.session.busy or not self.session._requests.empty()) and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.05)

    def test_typed_requests_are_answered_in_order(self) -> None:
        self.session.say("Hello.")
        self.session.submit_text("Open Chrome.")
        self.session.submit_text("Tell me a joke.")
        self._settle()
        self.assertEqual(
            self.assistant.calls,
            [("say", "Hello."), ("respond", "Open Chrome."), ("respond", "Tell me a joke.")],
        )
        self.assertIn(("you", "Tell me a joke."), self.logged)

    def test_stop_says_goodbye_and_quits(self) -> None:
        self.session.submit_text("Stop.")
        self._settle()
        self.assertEqual(self.assistant.calls, [("say", "Goodbye.")])
        self.assertEqual(self.events.get(timeout=1), ("quit",))

    def test_stop_button_and_talking_interrupt(self) -> None:
        self.session.interrupt()
        self.assertEqual(self.assistant.calls, [("interrupt", "")])


if __name__ == "__main__":
    unittest.main()
