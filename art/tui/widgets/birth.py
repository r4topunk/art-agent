from __future__ import annotations
import numpy as np
from rich.text import Text
from textual.widget import Widget

UPPER_HALF = "▀"
LOWER_HALF = "▄"
FULL_BLOCK = "█"
EMPTY = " "
DOT = "·"
SPARK = "▁▂▃▄▅▆▇█"


class BirthWidget(Widget):
    DEFAULT_CSS = """
    BirthWidget {
        width: 100%;
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._grid: np.ndarray | None = None  # 16x16 binary
        self._confidences: np.ndarray | None = None  # 256 floats
        self._filled: int = 0  # how many pixels filled so far
        self._grid_size: int = 16

    def update_birth(self, grid: np.ndarray, confidences: np.ndarray, piece_index: int = 0):
        """Show a completed piece with its confidence map."""
        self._grid = grid
        self._confidences = confidences
        self._filled = self._grid_size * self._grid_size
        self.refresh()

    def show_progress(self, grid_partial: np.ndarray, confidences_partial: np.ndarray, filled: int):
        """Update with partial generation progress."""
        self._grid = grid_partial
        self._confidences = confidences_partial
        self._filled = filled
        self.refresh()

    def _conf_color(self, conf: float) -> str:
        if conf >= 0.8:
            return "bright_green"
        elif conf >= 0.6:
            return "green"
        elif conf >= 0.4:
            return "yellow"
        elif conf >= 0.2:
            return "red"
        return "bright_red"

    def render(self) -> Text:
        result = Text()
        result.append("◉ NASCIMENTO\n", style="bold magenta")

        if self._grid is None:
            result.append("  Awaiting generation...\n", style="dim")
            return result

        size = self._grid_size

        # Render grid with confidence coloring - half blocks
        # Two layouts side by side: PIECE and CONFIDENCE MAP
        piece_lines = []
        conf_lines = []

        for y in range(0, size, 2):
            piece_line = Text()
            conf_line = Text()
            for x in range(size):
                top_idx = y * size + x
                bot_idx = (y + 1) * size + x

                # Piece rendering
                top_filled = top_idx < self._filled
                bot_filled = bot_idx < self._filled and y + 1 < size

                top_val = self._grid[y, x] if top_filled else 0
                bot_val = self._grid[y + 1, x] if bot_filled and y + 1 < size else 0

                if not top_filled and not bot_filled:
                    piece_line.append(DOT, style="bright_black")
                elif top_filled and not bot_filled:
                    piece_line.append(UPPER_HALF if top_val else " ", style="bright_green")
                elif not top_filled and bot_filled:
                    piece_line.append(LOWER_HALF if bot_val else " ", style="bright_green")
                else:
                    if top_val and bot_val:
                        piece_line.append(FULL_BLOCK, style="bright_green")
                    elif top_val:
                        piece_line.append(UPPER_HALF, style="bright_green")
                    elif bot_val:
                        piece_line.append(LOWER_HALF, style="bright_green")
                    else:
                        piece_line.append(EMPTY)

                # Confidence rendering
                if self._confidences is not None:
                    top_conf = self._confidences[top_idx] if top_idx < len(self._confidences) and top_filled else 0
                    bot_conf = self._confidences[bot_idx] if bot_idx < len(self._confidences) and bot_filled else 0

                    if not top_filled and not bot_filled:
                        conf_line.append(DOT, style="bright_black")
                    elif top_filled and bot_filled:
                        avg_conf = (top_conf + bot_conf) / 2
                        conf_line.append(FULL_BLOCK, style=self._conf_color(avg_conf))
                    elif top_filled:
                        conf_line.append(UPPER_HALF, style=self._conf_color(top_conf))
                    else:
                        conf_line.append(LOWER_HALF, style=self._conf_color(bot_conf))

            piece_lines.append(piece_line)
            conf_lines.append(conf_line)

        # Side by side: piece | confidence
        result.append("  PIECE            CONFIDENCE\n", style="dim")
        for pl, cl in zip(piece_lines, conf_lines):
            result.append("  ")
            result.append(pl)
            result.append("    ")
            result.append(cl)
            result.append("\n")

        # Progress bar
        pct = self._filled / (size * size)
        filled_bar = int(pct * 20)
        result.append(f"\n  pixel {self._filled}/{size*size}  ", style="dim")
        result.append("█" * filled_bar, style="cyan")
        result.append("░" * (20 - filled_bar), style="bright_black")
        result.append("\n")

        # Confidence waveform (last 40 pixels)
        if self._confidences is not None and self._filled > 0:
            recent = self._confidences[max(0, self._filled - 40):self._filled]
            if len(recent) > 0:
                result.append("  Confidence: ", style="dim")
                for c in recent:
                    idx = min(7, max(0, int(c * 7.99)))
                    color = self._conf_color(c)
                    result.append(SPARK[idx], style=color)
                result.append("\n")

        return result
