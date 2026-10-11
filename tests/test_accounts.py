"""Memory, read-only Gmail and read-only GitHub, against fake servers.

The fakes record every request, so the tests also check that nothing but
reads is ever sent (and the token refresh, the one POST Gmail needs).

Run: python -m unittest tests.test_accounts
"""

import base64
import tempfile
import time
import unittest
from pathlib import Path

from assistant.github import GitHub, GitHubError
from assistant.gmail import API as GMAIL_API, TOKEN_URL, Gmail, GmailError
from assistant.main import Assistant
from assistant.memory import Memory, as_said_back
from assistant.tools import Workspace
from tests.fakes import FakeBrain, FakeFallback, FakePlayer, FakeTTS


class FakeResponse:
    def __init__(self, status: int, body) -> None:
        self.status_code, self._body = status, body
        self.ok = 200 <= status < 300

    def json(self):
        return self._body

    def raise_for_status(self) -> None:
        if not self.ok:
            raise RuntimeError(self.status_code)


class FakeHttp:
    """Answers by URL prefix; records (method, url, params)."""

    def __init__(self, routes: dict[str, object]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, str, dict]] = []

    def _answer(self, method, url, params):
        self.calls.append((method, url, params or {}))
        for prefix in sorted(self.routes, key=len, reverse=True):
            if url.startswith(prefix):
                answer = self.routes[prefix]
                answer = answer(params or {}) if callable(answer) else answer
                return answer if isinstance(answer, FakeResponse) else FakeResponse(200, answer)
        return FakeResponse(404, {})

    def get(self, url, params=None, headers=None, timeout=None):
        return self._answer("GET", url, params)

    def post(self, url, data=None, timeout=None):
        return self._answer("POST", url, data)


def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


LOGIN = {"client_id": "id", "client_secret": "secret", "refresh_token": "refresh"}
NOW_MS = str(int(time.time() * 1000))


def gmail_routes() -> dict:
    def messages(params):
        q = params.get("q", "").lower()
        if "from:(amazon)" in q:
            return {"messages": [{"id": "m1"}, {"id": "m2"}]}
        if "is:unread" in q:
            return {"messages": [{"id": "m1"}]}
        if "in:inbox" in q:
            return {"messages": [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]}
        return {}

    def meta(n, sender, subject):
        return {"internalDate": NOW_MS, "payload": {"headers": [
            {"name": "From", "value": sender}, {"name": "Subject", "value": subject}]}}

    return {
        TOKEN_URL: {"access_token": "access", "expires_in": 3600},
        f"{GMAIL_API}/messages/m1": lambda p: (
            meta(1, '"Amazon" <ship@amazon.com>', "Your order shipped 📦") if p.get("format") == "metadata"
            else {"payload": {"mimeType": "multipart/alternative", "parts": [
                {"mimeType": "text/plain", "body": {"data": _b64("Your parcel arrives Tuesday.\n\nThanks.")}},
                {"mimeType": "text/html", "body": {"data": _b64("<p>ignored</p>")}},
            ]}}
        ),
        f"{GMAIL_API}/messages/m2": lambda p: (
            meta(2, "deals@amazon.com", "Deals for you") if p.get("format") == "metadata"
            else {"payload": {"mimeType": "text/html", "body": {"data": _b64(
                "<style>x{}</style><p>Big <b>sale</b> &amp; more</p>")}}}
        ),
        f"{GMAIL_API}/messages/m3": lambda p: meta(3, "Prof <p@uni.edu>", "Exam moved"),
        f"{GMAIL_API}/messages": messages,
    }


class GmailTest(unittest.TestCase):
    def setUp(self) -> None:
        self.http = FakeHttp(gmail_routes())
        self.gmail = Gmail(lambda: LOGIN, self.http)

    def test_counts(self) -> None:
        self.assertEqual(self.gmail.count_today(), "You got 3 emails today, 1 still unread.")
        self.assertEqual(self.gmail.count_unread(), "You have 1 unread email.")

    def test_from_sender_then_read_one(self) -> None:
        said = self.gmail.from_sender("Amazon")
        self.assertIn("1, from Amazon, today at", said)
        self.assertIn("Your order shipped.", said)  # emoji dropped for speech
        self.assertIn("2, from deals", said)
        text = self.gmail.read(1)
        self.assertIn("Your parcel arrives Tuesday.", text)
        self.assertNotIn("ignored", text)  # plain text preferred over HTML
        self.assertIn("Big sale & more", self.gmail.read("next"))  # HTML stripped
        with self.assertRaises(GmailError):
            self.gmail.read("next")

    def test_only_reads_are_sent(self) -> None:
        self.gmail.count_today()
        self.gmail.latest()
        self.gmail.read(2)
        for method, url, _ in self.http.calls:
            self.assertTrue(method == "GET" or url == TOKEN_URL, (method, url))
        self.assertEqual(sum(1 for m, *_ in self.http.calls if m == "POST"), 1)  # token reused

    def test_not_connected_and_signed_out(self) -> None:
        with self.assertRaisesRegex(GmailError, "isn't connected"):
            Gmail(lambda: None, self.http).count_unread()
        expired = FakeHttp({TOKEN_URL: FakeResponse(400, {"error": "invalid_grant"})})
        with self.assertRaisesRegex(GmailError, "sign in again"):
            Gmail(lambda: LOGIN, expired).count_unread()


def github_routes() -> dict:
    api = "https://api.github.com"
    return {
        f"{api}/user/repos": [
            {"name": "voice_ai", "full_name": "me/voice_ai", "stargazers_count": 3, "forks_count": 1,
             "pushed_at": "2026-10-10T10:00:00Z", "description": "Offline assistant"},
            {"name": "portfolio", "full_name": "me/portfolio", "stargazers_count": 0, "forks_count": 0,
             "pushed_at": "2026-01-01T00:00:00Z"},
        ],
        f"{api}/repos/me/voice_ai/commits": [
            {"commit": {"message": "Add window\n\nlong body", "author": {"date": "2026-10-11T08:00:00Z"}}},
        ],
        f"{api}/repos/me/voice_ai/issues": [
            {"title": "Mic cuts out"}, {"title": "A PR", "pull_request": {}},
        ],
        f"{api}/repos/me/voice_ai/readme": {"content": base64.b64encode(b"# Voice\nOffline assistant.").decode()},
    }


class GitHubTest(unittest.TestCase):
    def setUp(self) -> None:
        self.http = FakeHttp(github_routes())
        self.github = GitHub(lambda: {"token": "github_pat_x"}, self.http)

    def test_spoken_repo_names(self) -> None:
        self.assertIn("voice ai has 3 stars", self.github.stats("voice AI"))
        self.assertIn("Add window", self.github.whats_new("the voice ai"))
        with self.assertRaisesRegex(GitHubError, "don't see a repo"):
            self.github.stats("banana bread")

    def test_issues_exclude_pull_requests(self) -> None:
        said = self.github.issues("voice ai")
        self.assertIn("1 open issue.", said)
        self.assertNotIn("A PR", said)

    def test_readme_and_only_gets(self) -> None:
        self.assertIn("Offline assistant.", self.github.readme("voice ai"))
        self.assertTrue(all(method == "GET" for method, *_ in self.http.calls))

    def test_not_connected(self) -> None:
        with self.assertRaisesRegex(GitHubError, "isn't connected"):
            GitHub(lambda: None, self.http).repos()


class MemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "data" / "memory.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_remember_recall_forget_and_persist(self) -> None:
        memory = Memory(self.path)
        self.assertEqual(memory.remember("my exam is on friday"), "Got it. I'll remember your exam is on friday.")
        self.assertEqual(memory.remember("my exam is on friday"), "I already remember that.")
        memory.remember("I am allergic to peanuts")
        self.assertEqual(Memory(self.path).facts, memory.facts)  # survives a restart
        self.assertIn("You are allergic to peanuts.", memory.recall())
        self.assertEqual(memory.find("my exam"), "my exam is on friday")
        self.assertIn("forgotten", memory.forget("my exam is on friday"))
        self.assertEqual(memory.facts, ["I am allergic to peanuts"])
        self.assertIsNone(memory.find("my car"))
        self.assertIsNone(memory.find("me"))  # too vague to pick one

    def test_note_for_the_chat_model_is_bounded(self) -> None:
        memory = Memory(self.path)
        for n in range(30):
            memory.remember(f"fact number {n} " + "x" * 60)
        note = memory.prompt_note()
        self.assertLess(len(note), 700)
        self.assertIn("fact number 29", note)  # newest kept

    def test_pronouns(self) -> None:
        self.assertEqual(as_said_back("I'm moving my desk"), "you're moving your desk")


class AssistantAccountsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.ws = Workspace(roots={"adi": root}, launcher=lambda *a: None, open_url=lambda u: None,
                            gmail=Gmail(lambda: LOGIN, FakeHttp(gmail_routes())),
                            github=GitHub(lambda: None, FakeHttp({})))
        self.memory = Memory(root / "memory.json")
        self.player = FakePlayer()
        self.brain = FakeBrain("none", ["The parcel comes Tuesday."])
        self.assistant = Assistant(self.brain, FakeTTS(), self.player, FakeFallback(), self.ws, memory=self.memory)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def say(self, text: str) -> None:
        self.assistant.respond(text, time.perf_counter())

    def test_memory_reaches_the_chat_model_and_forget_all_needs_yes(self) -> None:
        self.say("Remember that my exam is on Friday.")
        self.assertIn("my exam is on friday", self.brain.memory_note)
        self.say("Forget everything.")
        self.assertIn("Say yes to confirm.", self.player.said[-1])
        self.say("No.")
        self.assertEqual(self.memory.facts, ["my exam is on friday"])
        self.say("Forget everything.")
        self.say("Yes.")
        self.assertEqual(self.memory.facts, [])
        self.assertEqual(self.brain.memory_note, "")  # the chat model forgot too

    def test_forgetting_one_thing_asks_first(self) -> None:
        self.say("Remember that my exam is on Friday.")
        self.say("Forget about my exam.")
        self.assertIn("Forget that your exam is on friday?", self.player.said[-2] + " " + self.player.said[-1])
        self.assertEqual(len(self.memory.facts), 1)
        self.say("Yes.")
        self.assertEqual(self.memory.facts, [])

    def test_stale_email_list_is_not_read(self) -> None:
        self.say("Find the email from Amazon.")
        self.say("Where am I?")
        self.say("Read the second one.")
        self.assertEqual(self.brain.streamed, [])  # nothing read out

    def test_email_list_then_read_the_first_one(self) -> None:
        self.say("Find the email from Amazon.")
        self.assertIn("Emails from amazon.", self.player.said[0])
        self.say("Read the first one.")
        self.assertIn("Your parcel arrives Tuesday.", self.brain.streamed[0]["context"])
        self.assertEqual(self.player.said[-1], "The parcel comes Tuesday.")

    def test_github_not_set_up_is_said_plainly(self) -> None:
        self.say("List my repos.")
        self.assertIn("GitHub isn't connected yet.", " ".join(self.player.said))


if __name__ == "__main__":
    unittest.main()
