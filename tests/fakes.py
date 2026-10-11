"""Stand-ins for the LLM, TTS, player, and fillers in assistant flow tests."""

import contextlib
import unittest.mock


class FakeBrain:
    """Scripted classifier answers and replies; records what each call carried."""

    def __init__(self, choice: str, *replies: list[str]) -> None:
        self.choice = choice
        self.replies = list(replies)
        self.classified: list[str] = []
        self.streamed: list[dict] = []

    def classify(self, instructions, text, schema):
        self.classified.append(text)
        return {"choice": self.choice}

    def stream_reply(self, text, context=None):
        self.streamed.append({"text": text, "context": context})
        yield from self.replies.pop(0)


class FakeTTS:
    def synthesize(self, text):
        return text, 22050


class FakePlayer:
    def __init__(self) -> None:
        self.said: list[str] = []

    def enqueue(self, audio, rate) -> None:
        self.said.append(audio)


class FakeFallback:
    def __init__(self) -> None:
        self.watchdogs = 0

    @contextlib.contextmanager
    def watchdog(self):
        self.watchdogs += 1
        yield unittest.mock.Mock()

    def speak_recovery(self) -> None:
        raise AssertionError("no recovery expected")
