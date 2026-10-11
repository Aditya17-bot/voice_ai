"""Read-only Gmail: count today's emails, list the latest, find emails from
someone or about something, and read one out (the LLM summarizes it).

Access uses Google's `gmail.readonly` scope, so Google itself refuses any
change: nothing can be sent, deleted, labelled or marked read, whatever this
code does. The login is set up once with `python -m assistant.setup gmail`
and kept encrypted (see credentials.py).

Email text is data from the outside world. Subjects and senders are spoken
directly; a body goes to the LLM only as delimited data in a tool-less turn.
None of it reaches the classifier that chooses actions.
"""

import base64
import html
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import requests

from . import credentials

SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
API = "https://gmail.googleapis.com/gmail/v1/users/me"
TOKEN_URL = "https://oauth2.googleapis.com/token"
TIMEOUT_S = 8
LISTED = 5  # emails read out per list
MAX_COUNT = 500  # "more than 500" beyond this
# An email goes to a 2048-token context with the prompt, memory and a short
# history; links and non-English text cost more tokens per character.
MAX_BODY_CHARS = 2400
SECRET_NAME = "gmail"
_SPEAKABLE = re.compile(r"[^\w\s.,:;!?'&()/@-]")
_TAGS = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.DOTALL | re.IGNORECASE)
_SENDER_NAME = re.compile(r'^\s*"?([^"<]+?)"?\s*<')


class GmailError(Exception):
    """Speakable reason a Gmail request didn't work."""


class NoList(GmailError):
    """No emails were read out to pick from: a bare "the second one" isn't ours."""


@dataclass(frozen=True)
class Email:
    id: str
    sender: str
    subject: str
    when: str


def clean(text: str, limit: int = 80) -> str:
    """Safe to speak and show: plain characters, bounded length."""
    text = "".join(ch for ch in text if ord(ch) <= 0xFFFF)
    text = " ".join(_SPEAKABLE.sub(" ", text).split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def _sender(raw: str) -> str:
    name = _SENDER_NAME.match(raw)
    return clean(name.group(1) if name else raw.split("@")[0], 40)


def _when(raw_ms: str) -> str:
    try:
        sent = datetime.fromtimestamp(int(raw_ms) / 1000)
    except (TypeError, ValueError):
        return ""
    today = datetime.now().date()
    if sent.date() == today:
        return sent.strftime("today at %I:%M %p").replace(" 0", " ")
    if (today - sent.date()).days == 1:
        return "yesterday"
    return sent.strftime("on %A %d %B").replace(" 0", " ")


def _text_of(payload: dict) -> str:
    """Plain text of a message: the text/plain part, else HTML with tags stripped."""
    plain, rich = [], []
    stack = [payload]
    while stack:
        part = stack.pop()
        mime = part.get("mimeType", "")
        data = (part.get("body") or {}).get("data")
        if data and mime in ("text/plain", "text/html"):
            text = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", "replace")
            (plain if mime == "text/plain" else rich).append(text)
        stack.extend(reversed(part.get("parts") or []))
    if plain:
        text = "\n".join(plain)
    else:
        text = html.unescape(_TAGS.sub(" ", "\n".join(rich)))
    return re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


class Gmail:
    def __init__(
        self,
        load_login: Callable[[], dict | None] = lambda: credentials.load(SECRET_NAME),
        http: requests.Session | None = None,
    ) -> None:
        self._load_login = load_login
        self._http = http or requests.Session()
        self._token = ""
        self._token_expires = 0.0
        self.listed: list[Email] = []  # the last emails read out, for "read the second one"
        self._read_number = 0  # the one read last, for "read the next one"

    # ----- access ----------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._load_login() is not None

    def _access_token(self) -> str:
        if self._token and time.time() < self._token_expires - 60:
            return self._token
        login = self._load_login()
        if login is None:
            raise GmailError("Gmail isn't connected yet. Run the Gmail setup once, it's in the read me.")
        try:
            response = self._http.post(TOKEN_URL, data={
                "client_id": login["client_id"],
                "client_secret": login["client_secret"],
                "refresh_token": login["refresh_token"],
                "grant_type": "refresh_token",
            }, timeout=TIMEOUT_S)
        except requests.RequestException:
            raise GmailError("I couldn't reach Gmail just now.") from None
        if response.status_code in (400, 401):
            raise GmailError("Gmail asked to sign in again. Run the Gmail setup once more.")
        response.raise_for_status()
        body = response.json()
        self._token = body["access_token"]
        self._token_expires = time.time() + int(body.get("expires_in", 3600))
        return self._token

    def _get(self, path: str, **params) -> dict:
        """GET only: this class never sends a request that could change anything."""
        try:
            response = self._http.get(
                f"{API}/{path}", params=params, timeout=TIMEOUT_S,
                headers={"Authorization": f"Bearer {self._access_token()}"},
            )
        except requests.RequestException:
            raise GmailError("I couldn't reach Gmail just now.") from None
        if response.status_code == 401:
            self._token = ""
            raise GmailError("Gmail asked to sign in again. Run the Gmail setup once more.")
        if not response.ok:
            raise GmailError("Gmail didn't answer that one.")
        return response.json()

    def _ids(self, query: str, limit: int) -> list[str]:
        ids: list[str] = []
        page = None
        while len(ids) < limit:
            params = {"q": query, "maxResults": min(limit - len(ids), 500)}
            if page:
                params["pageToken"] = page
            body = self._get("messages", **params)
            ids += [m["id"] for m in body.get("messages", [])]
            page = body.get("nextPageToken")
            if not page:
                break
        return ids

    def _email(self, message_id: str) -> Email:
        body = self._get(f"messages/{message_id}", format="metadata", metadataHeaders=["From", "Subject"])
        headers = {h["name"].lower(): h["value"] for h in body.get("payload", {}).get("headers", [])}
        return Email(
            message_id,
            _sender(headers.get("from", "someone")),
            clean(headers.get("subject", "")) or "no subject",
            _when(body.get("internalDate")),
        )

    # ----- what the assistant can say -----------------------------------------

    @staticmethod
    def _since_midnight() -> str:
        midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return f"after:{int(midnight.timestamp())}"

    def count_today(self) -> str:
        today = self._since_midnight()
        total = len(self._ids(f"in:inbox {today}", MAX_COUNT))
        unread = len(self._ids(f"in:inbox is:unread {today}", MAX_COUNT))
        if total == 0:
            return "No new emails today."
        many = "more than " if total >= MAX_COUNT else ""
        return f"You got {many}{total} email{'s' if total != 1 else ''} today, {unread} still unread."

    def count_unread(self) -> str:
        unread = len(self._ids("in:inbox is:unread", MAX_COUNT))
        if unread == 0:
            return "Your inbox has no unread emails."
        many = "more than " if unread >= MAX_COUNT else ""
        return f"You have {many}{unread} unread email{'s' if unread != 1 else ''}."

    def _list(self, query: str, intro: str, empty: str) -> str:
        ids = self._ids(query, LISTED)
        self._read_number = 0
        if not ids:
            self.listed = []
            return empty
        self.listed = [self._email(i) for i in ids]
        lines = " ".join(
            f"{n}, from {e.sender}{', ' + e.when if e.when else ''}: {e.subject}."
            for n, e in enumerate(self.listed, 1)
        )
        return f"{intro} {lines} Say read the first one, or another number."

    def latest(self, unread_only: bool = False) -> str:
        query = "in:inbox is:unread" if unread_only else "in:inbox"
        kind = "unread emails" if unread_only else "latest emails"
        return self._list(query, f"Your {kind}.", "Nothing new in your inbox.")

    def from_sender(self, who: str) -> str:
        who = clean(who, 60)
        return self._list(f"from:({who})", f"Emails from {who}.", f"I don't see any emails from {who}.")

    def about(self, topic: str) -> str:
        topic = clean(topic, 60)
        return self._list(topic, f"Emails about {topic}.", f"I don't see any emails about {topic}.")

    def read_latest(self) -> str:
        """Text of the newest email in the inbox."""
        self._list("in:inbox", "", "")
        if not self.listed:
            raise GmailError("Your inbox is empty.")
        return self.read(1)

    def read(self, number: int | str) -> str:
        """Text of email `number` (1-based, or "next") from the last list, for the LLM to summarize."""
        if not self.listed:
            raise NoList("Ask for your emails first, like: what are my latest emails.")
        if number == "next":
            number = self._read_number + 1
        if not 1 <= number <= len(self.listed):
            raise GmailError(f"I only listed {len(self.listed)} emails.")
        self._read_number = number
        email = self.listed[number - 1]
        body = self._get(f"messages/{email.id}", format="full")
        text = _text_of(body.get("payload", {}))[:MAX_BODY_CHARS] or "(no text)"
        return f"Email from {email.sender}, {email.when}.\nSubject: {email.subject}\n\n{text}"
