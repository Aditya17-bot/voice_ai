"""YouTube results: parsing the page, playing by number, never opening anything
but a youtube.com watch page for a well-formed video id. No network.

Run: python -m unittest tests.test_youtube
"""

import json
import tempfile
import time
import unittest
from pathlib import Path

import requests

from assistant.main import Assistant
from assistant.router import Intent, execute
from assistant.tools import ToolError, Workspace
from assistant.youtube import NoVideos, YouTube, parse_results
from tests.fakes import FakeBrain, FakeFallback, FakePlayer, FakeTTS


def _renderer(video_id: str, title: str, channel: str) -> dict:
    return {"videoRenderer": {
        "videoId": video_id,
        "title": {"runs": [{"text": title}]},
        "ownerText": {"runs": [{"text": channel}]},
    }}


def results_page(*videos: tuple[str, str, str]) -> str:
    data = {"contents": {"sectionList": [
        {"adSlotRenderer": {"videoId": "AAAAAAAAAAA"}},  # not a videoRenderer: skipped
        {"items": [_renderer(*v) for v in videos]},
    ]}}
    return f"<html><script>var ytInitialData = {json.dumps(data)};</script></html>"


PAGE = results_page(
    ("abcdefghijk", "Lofi beats 🎧 to study", "Lofi Girl"),
    ("BBBBBBBBBBB", "Rainy lofi", "Chill"),
    ("bad id", "Broken", "x"),
    ("CCCCCCCCCCC", "Third", "Someone"),
)


class ParseTest(unittest.TestCase):
    def test_videos_in_order_with_clean_titles(self) -> None:
        videos = parse_results(PAGE)
        self.assertEqual([v.id for v in videos], ["abcdefghijk", "BBBBBBBBBBB", "CCCCCCCCCCC"])
        self.assertEqual(videos[0].title, "Lofi beats to study")  # emoji dropped for speech
        self.assertEqual(videos[0].channel, "Lofi Girl")

    def test_layout_change_keeps_only_the_top_id(self) -> None:
        # The rest of the page may be ads or sidebar videos: never picked by number.
        page = '... "videoId":"abcdefghijk" ... "videoId":"BBBBBBBBBBB"'
        self.assertEqual([v.id for v in parse_results(page)], ["abcdefghijk"])

    def test_titles_beyond_the_basic_plane_are_dropped(self) -> None:
        page = results_page(("abcdefghijk", "\U0001d473\U0001d490 Lofi", "x"))
        self.assertEqual(parse_results(page)[0].title, "Lofi")

    def test_nothing_found(self) -> None:
        self.assertEqual(parse_results("<html>no data</html>"), [])


class YouTubeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.opened: list[tuple[str, str]] = []
        self.fetched: list[str] = []

        def fetch(query: str) -> str:
            self.fetched.append(query)
            return PAGE

        self.yt = YouTube(lambda url, profile: self.opened.append((url, profile)), fetch)

    def test_play_opens_top_result(self) -> None:
        said = self.yt.play("lofi", "robert")
        self.assertEqual(said, "Playing Lofi beats to study by Lofi Girl.")
        self.assertEqual(self.opened, [("https://www.youtube.com/watch?v=abcdefghijk", "robert")])

    def test_pick_by_number_and_next(self) -> None:
        self.yt.remember_search("lofi")
        self.yt.pick(2)
        self.yt.pick("next")
        self.assertEqual([u for u, _ in self.opened], [
            "https://www.youtube.com/watch?v=BBBBBBBBBBB",
            "https://www.youtube.com/watch?v=CCCCCCCCCCC",
        ])
        with self.assertRaises(NoVideos):
            self.yt.pick("next")  # past the end
        self.assertEqual(self.fetched, ["lofi"])  # the prefetch is reused

    def test_pick_without_a_search(self) -> None:
        with self.assertRaises(NoVideos):
            self.yt.pick(1)
        self.assertEqual(self.opened, [])

    def test_unreachable(self) -> None:
        def down(query):
            raise requests.ConnectionError("offline")

        yt = YouTube(lambda url, profile: self.opened.append(url), down)
        with self.assertRaises(NoVideos) as caught:
            yt.play("lofi")
        self.assertIn("couldn't reach YouTube", str(caught.exception))
        self.assertEqual(self.opened, [])

    def test_titles(self) -> None:
        self.yt.remember_search("lofi")
        self.assertIn("Number 2, Rainy lofi.", self.yt.titles())


class FlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.urls: list[str] = []
        self.ws = Workspace(
            roots={"adi": root}, launcher=lambda *a: None, open_url=self.urls.append, fetch=lambda q: PAGE
        )
        self.player = FakePlayer()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_search_then_play_the_second_one(self) -> None:
        unittest.mock.patch("assistant.tools._find_chrome", return_value=None).start()
        self.addCleanup(unittest.mock.patch.stopall)
        assistant = Assistant(FakeBrain("none"), FakeTTS(), self.player, FakeFallback(), self.ws)
        assistant.respond("Search YouTube for lofi.", time.perf_counter())
        assistant.respond("Play the second one.", time.perf_counter())
        self.assertEqual(self.urls[-1], "https://www.youtube.com/watch?v=BBBBBBBBBBB")
        self.assertEqual(self.player.said[-1], "Playing Rainy lofi by Chill.")

    def test_other_actions_end_the_youtube_thread(self) -> None:
        unittest.mock.patch("assistant.tools._find_chrome", return_value=None).start()
        self.addCleanup(unittest.mock.patch.stopall)
        assistant = Assistant(FakeBrain("none", ["Hmm."]), FakeTTS(), self.player, FakeFallback(), self.ws)
        assistant.respond("Search YouTube for lofi.", time.perf_counter())
        assistant.respond("Where am I?", time.perf_counter())
        assistant.respond("The second one.", time.perf_counter())  # not a video any more
        self.assertFalse(any("watch?v=" in url for url in self.urls))

    def test_bare_ordinal_without_search_is_not_ours(self) -> None:
        self.assertIsNone(execute(Intent("pick_video", "2", loose=True), self.ws))
        with self.assertRaises(ToolError):
            execute(Intent("pick_video", "2"), self.ws)


if __name__ == "__main__":
    unittest.main()
