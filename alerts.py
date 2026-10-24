"""One nonblocking Windows audio owner; no racing playback threads."""
from pathlib import Path
import time


class AlertPlayer:
    def __init__(self, enabled=True, warning_repeat=8.0):
        self.enabled = enabled
        self.warning_repeat = warning_repeat
        self.level = 0
        self.last_warning = float('-inf')
        self.error = None
        self.sound = None
        try:
            import winsound
            self.sound = winsound
            self.path = Path(__file__).resolve().parent / 'alarm.wav'
            if not self.path.exists():
                from generate_alarm import generate_alarm
                generate_alarm(str(self.path))
        except (ImportError, OSError) as exc:
            self.error = f'Audio unavailable: {exc}'
            self.enabled = False

    def update(self, level, now=None):
        if not self.enabled or self.sound is None:
            return
        now = time.monotonic() if now is None else now
        try:
            if self.level == 2 and level != 2:
                self.sound.PlaySound(None, 0)
            if level == 2 and self.level != 2:
                self.sound.PlaySound(str(self.path), self.sound.SND_FILENAME | self.sound.SND_ASYNC | self.sound.SND_LOOP)
            elif level == 1 and now - self.last_warning >= self.warning_repeat:
                self.sound.PlaySound(str(self.path), self.sound.SND_FILENAME | self.sound.SND_ASYNC)
                self.last_warning = now
            elif level == 0 and self.level == 1:
                self.sound.PlaySound(None, 0)
            self.level = level
        except RuntimeError as exc:
            self.error = f'Audio playback failed: {exc}'
            self.close()
            self.enabled = False

    def toggle(self):
        if self.enabled:
            self.close()
            self.enabled = False
        elif self.sound is not None:
            self.enabled = True
            self.error = None
            self.last_warning = float('-inf')

    def close(self):
        if self.sound is not None:
            try:
                self.sound.PlaySound(None, 0)
            except RuntimeError:
                pass
        self.level = 0
