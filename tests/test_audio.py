"""The playback buffer: clips play back to back, stop() empties it, idle is
reported once the audio has run out. The sound card is mocked; the stream
callback is driven by hand.

Run: python -m unittest tests.test_audio
"""

import unittest
from unittest import mock

import numpy as np

from assistant import audio_io


class PlayerTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.object(audio_io.sd, "OutputStream")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.player = audio_io.Player(sample_rate=100)

    def _pull(self, frames: int) -> np.ndarray:
        out = np.full((frames, 1), 99, dtype=np.int16)
        self.player._fill(out, frames, None, None)
        return out[:, 0].copy()

    def test_clips_play_back_to_back(self) -> None:
        self.player.enqueue(np.array([1, 2, 3], np.int16), 100)
        self.player.enqueue(np.array([4, 5], np.int16), 100)
        self.assertTrue(self.player.busy)
        self.assertEqual(self._pull(4).tolist(), [1, 2, 3, 4])
        self.assertEqual(self._pull(4).tolist(), [5, 0, 0, 0])  # tail padded with silence
        self.assertTrue(self.player.busy)  # last block may still be in the device
        self._pull(4)
        self.assertFalse(self.player.busy)
        self.assertTrue(self.player.wait_until_done(timeout=0))

    def test_stop_empties_everything(self) -> None:
        self.player.enqueue(np.arange(1, 10, dtype=np.int16), 100)
        self._pull(2)
        self.player.stop()
        self.assertFalse(self.player.busy)
        self.assertEqual(self._pull(4).tolist(), [0, 0, 0, 0])

    def test_other_rates_and_float_audio_are_converted(self) -> None:
        self.player.enqueue(np.ones(200, np.int16), 200)  # twice our rate -> half the samples
        self.assertEqual(len(self.player._clips[0]), 100)
        self.player.stop()
        self.player.enqueue(np.array([0.5, -0.5], np.float32), 100)
        self.assertEqual(self.player._clips[0].dtype, np.int16)

    def test_empty_clip_is_ignored(self) -> None:
        self.player.enqueue(np.zeros(0, np.int16), 100)
        self.assertFalse(self.player.busy)


if __name__ == "__main__":
    unittest.main()
