from __future__ import annotations

import time
from rich.text import Text
from textual.widget import Widget


# Log level colors
LOG_STYLES = {
    "SYS":    ("dim",          "░"),
    "TRAIN":  ("green",        "▸"),
    "GEN":    ("cyan",         "▸"),
    "SCORE":  ("yellow",       "▸"),
    "SELECT": ("magenta",      "▸"),
    "SAVE":   ("blue",         "▸"),
    "NEURAL": ("bright_cyan",  "▸"),
    "OK":     ("bright_green", "✓"),
    "WARN":   ("bright_yellow", "⚠"),
    "ERR":    ("bright_red",   "✗"),
}


class SystemLog(Widget):
    """Scrolling terminal-style system log with colored entries."""

    DEFAULT_CSS = """
    SystemLog {
        width: 100%;
        height: 100%;
        padding: 0;
    }
    """

    def __init__(self, max_lines: int = 100, **kwargs):
        super().__init__(**kwargs)
        self._entries: list[tuple[float, str, str, str]] = []
        self._max_lines = max_lines
        self._start_time = time.time()

    def log(self, level: str, source: str, message: str):
        self._entries.append((time.time(), level, source, message))
        if len(self._entries) > self._max_lines:
            self._entries = self._entries[-self._max_lines:]
        self.refresh()

    def render(self) -> Text:
        result = Text()
        # self.size.height is content height (Textual excludes border from size)
        # subtract 1 for the "SYSTEM LOG" title line
        visible_lines = max(1, self.size.height - 1)

        result.append(" SYSTEM LOG\n", style="bold magenta")

        if not self._entries:
            result.append("  waiting for events...\n", style="dim")
            return result

        visible = self._entries[-visible_lines:]

        for ts, level, source, msg in visible:
            elapsed = ts - self._start_time
            mins = int(elapsed) // 60
            secs = int(elapsed) % 60

            style, marker = LOG_STYLES.get(level, ("dim", "·"))

            result.append(f" {mins:02d}:{secs:02d} ", style="bright_black")
            result.append(f"{marker} ", style=style)
            result.append(f"{source:<6s} ", style=style)

            # Truncate message to fit width — leave more room
            max_msg = max(10, self.size.width - 18)
            truncated = msg[:max_msg] + "…" if len(msg) > max_msg else msg
            msg_style = "white" if level not in ("SYS", "WARN", "ERR") else style
            result.append(f"{truncated}\n", style=msg_style)

        return result
