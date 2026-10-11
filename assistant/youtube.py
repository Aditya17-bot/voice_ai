"""Play YouTube videos by search: "play lofi on YouTube", "play the second one".

The assistant can't click inside Chrome (pressing keys could land in a
terminal), so it reads YouTube's own results page for the user's words and
opens the chosen video's watch page. Only youtube.com is fetched or opened,
and only video ids that look like video ids are used.

Video titles are text from the web: they are spoken and shown, but never
given to the LLM that chooses actions.
"""

import json
import re
import threading
from dataclasses import dataclass
from typing import Callable

import requests

RESULTS_URL = "https://www.youtube.com/results"
WATCH_URL = "https://www.youtube.com/watch?v="
FETCH_TIMEOUT_S = 6
MAX_RESULTS = 10
TITLES_SPOKEN = 3
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
_INITIAL_DATA = re.compile(r"var ytInitialData\s*=\s*(\{.*?\});\s*</script>", re.DOTALL)
_HEADERS = {
    # The plain desktop page; without a browser-like agent YouTube sends a consent or app page.
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
# Titles are read aloud: keep letters, digits and ordinary punctuation only.
_UNSPEAKABLE = re.compile(r"[^\w\s.,:;!?'&()/-]")


@dataclass(frozen=True)
class Video:
    id: str
    title: str
    channel: str = ""


class NoVideos(Exception):
    """Speakable reason there's nothing to play."""


def _clean(text: str, limit: int = 90) -> str:
    # Characters beyond U+FFFF (stylized "𝑳𝒐𝒇𝒊" letters, most emoji) can't be
    # shown by Tk on Windows and aren't read aloud anyway.
    text = "".join(ch for ch in text if ord(ch) <= 0xFFFF)
    text = " ".join(_UNSPEAKABLE.sub(" ", text).split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def _runs_text(node: dict | None) -> str:
    if not isinstance(node, dict):
        return ""
    if "simpleText" in node:
        return str(node["simpleText"])
    return "".join(str(r.get("text", "")) for r in node.get("runs") or [] if isinstance(r, dict))


def parse_results(html: str) -> list[Video]:
    """Videos in page order from a YouTube results page (ads, shorts, playlists skipped)."""
    found: list[Video] = []
    match = _INITIAL_DATA.search(html)
    if match:
        try:
            data = json.loads(match.group(1))
        except ValueError:
            data = None
        stack = [data]
        while stack and len(found) < MAX_RESULTS:
            node = stack.pop()
            if isinstance(node, dict):
                renderer = node.get("videoRenderer")
                if isinstance(renderer, dict) and _VIDEO_ID.fullmatch(str(renderer.get("videoId", ""))):
                    found.append(Video(
                        renderer["videoId"],
                        _clean(_runs_text(renderer.get("title"))) or "a video",
                        _clean(_runs_text(renderer.get("ownerText")), 40),
                    ))
                    continue
                stack.extend(reversed(list(node.values())))
            elif isinstance(node, list):
                stack.extend(reversed(node))
    if not found:
        # Page layout changed: the first id on the page is still very likely
        # the top result, but the rest may be ads or sidebar videos, so only
        # one is kept and numbered picks beyond it are refused.
        first = re.search(r'"videoId":"([A-Za-z0-9_-]{11})"', html)
        found = [Video(first.group(1), "the top video")] if first else []
    return found


def fetch_results_page(query: str) -> str:
    response = requests.get(
        RESULTS_URL, params={"search_query": query}, headers=_HEADERS, timeout=FETCH_TIMEOUT_S
    )
    response.raise_for_status()
    return response.text


class YouTube:
    """Remembers the last search so "play the second one" knows what to play."""

    def __init__(
        self, open_url: Callable[[str, str], None], fetch: Callable[[str], str] | None = None
    ) -> None:
        """`open_url(url, profile)` opens a youtube.com page in Chrome."""
        self._open = open_url
        self._fetch = fetch or fetch_results_page
        self.query = ""
        self._profile = ""
        self._videos: list[Video] = []
        self._playing = -1
        self._lock = threading.Lock()
        self._prefetch: threading.Thread | None = None

    def remember_search(self, query: str, profile: str = "", prefetch: bool = True) -> None:
        """A results page was opened. Its videos are fetched in the background
        (the page takes a few seconds) so "play the second one" is instant."""
        self._profile = profile
        if query == self.query:
            return
        with self._lock:
            self.query, self._videos, self._playing = query, [], -1
            self._prefetch = None  # an older search's fetch is no longer wanted
        if prefetch:
            self._prefetch = threading.Thread(target=self._load, args=(query,), daemon=True)
            self._prefetch.start()

    def forget(self) -> None:
        """Moved on to something else: "the second one" no longer means a video."""
        with self._lock:
            self.query, self._videos, self._playing, self._prefetch = "", [], -1, None

    def _load(self, query: str) -> None:
        try:
            videos = parse_results(self._fetch(query))
        except Exception:  # noqa: BLE001 — background; tried again when a video is picked
            return
        with self._lock:
            if query == self.query:
                self._videos = videos

    def _results(self) -> list[Video]:
        if not self.query:
            raise NoVideos("Search YouTube for something first, like: search YouTube for lofi music.")
        prefetch = self._prefetch
        if prefetch is not None and prefetch.is_alive():
            prefetch.join(FETCH_TIMEOUT_S)
        if not self._videos:
            query = self.query
            try:
                videos = parse_results(self._fetch(query))
            except requests.RequestException:
                raise NoVideos("I couldn't reach YouTube just now.") from None
            with self._lock:
                if query == self.query:
                    self._videos = videos
            if not videos:
                raise NoVideos(f"I couldn't find any videos for {query}.")
        return self._videos

    def play(self, query: str, profile: str = "") -> str:
        """Play the top result for `query`."""
        self.remember_search(query, profile, prefetch=False)  # fetched right now instead
        return self.pick(1)

    def pick(self, number: int | str) -> str:
        """Play result `number` (1-based), or "next" after the one playing."""
        videos = self._results()
        index = self._playing + 1 if number == "next" else int(number) - 1
        if not 0 <= index < len(videos):
            if len(videos) == 1:
                raise NoVideos("I could only make out the top result this time.")
            raise NoVideos(f"There are only {len(videos)} results I can see.")
        video = videos[index]
        self._open(WATCH_URL + video.id, self._profile)
        self._playing = index
        by = f" by {video.channel}" if video.channel else ""
        return f"Playing {video.title}{by}."

    def titles(self) -> str:
        videos = self._results()[:TITLES_SPOKEN]
        listed = " ".join(f"Number {i}, {v.title}." for i, v in enumerate(videos, 1))
        return f"Top results for {self.query}. {listed} Say play the first one, or another number."
