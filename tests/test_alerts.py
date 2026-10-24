import sys
import unittest
from unittest.mock import MagicMock, patch

from alerts import AlertPlayer


class AlarmTests(unittest.TestCase):
    def player(self, enabled=True):
        sound = MagicMock(SND_FILENAME=1, SND_ASYNC=2, SND_LOOP=4, SND_ALIAS=8)
        with patch.dict(sys.modules, {'winsound': sound}):
            player = AlertPlayer(enabled=enabled)
        return player, sound

    def test_critical_alarm_starts_once_and_stops_on_recovery(self):
        player, sound = self.player()
        player.update(2, 0)
        player.update(2, 1)
        self.assertEqual(sound.PlaySound.call_count, 1)
        player.update(0, 2)
        sound.PlaySound.assert_called_with(None, 0)

    def test_warnings_are_rate_limited(self):
        player, sound = self.player()
        for t in (0, 1, 2, 7, 8):
            player.update(1, t)
        self.assertEqual(sound.PlaySound.call_count, 2)
        sound.PlaySound.assert_called_with(str(player.path), sound.SND_FILENAME | sound.SND_ASYNC)

    def test_muted_session_can_enable_audio(self):
        player, sound = self.player(False)
        player.update(2, 0)
        sound.PlaySound.assert_not_called()
        player.toggle()
        player.update(2, 1)
        self.assertEqual(sound.PlaySound.call_count, 1)
        player.toggle()
        self.assertFalse(player.enabled)
        sound.PlaySound.assert_called_with(None, 0)


if __name__ == '__main__':
    unittest.main()
