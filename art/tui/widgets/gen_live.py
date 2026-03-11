from __future__ import annotations

import numpy as np
from rich.text import Text
from textual.widget import Widget

UPPER_HALF = "▀"
LOWER_HALF = "▄"
FULL_BLOCK = "█"
EMPTY = " "


def _render_partial(grid: np.ndarray, filled_pixels: int) -> list[str]:
    """Render a grid with only the first N pixels filled (raster order), rest dimmed."""
    rows, cols = grid.shape
    lines = []
    for y in range(0, rows, 2):
        line = ""
        for x in range(cols):
            top_idx = y * cols + x
            bot_idx = (y + 1) * cols + x

            top = grid[y, x] if top_idx < filled_pixels else -1
            bot = grid[y + 1, x] if y + 1 < rows and bot_idx < filled_pixels else -1

            if top == -1 and bot == -1:
                line += "·"  # unfilled
            elif top == -1:
                line += LOWER_HALF if bot else " "
            elif bot == -1:
                line += UPPER_HALF if top else " "
            else:
                if top and bot:
                    line += FULL_BLOCK
                elif top:
                    line += UPPER_HALF
                elif bot:
                    line += LOWER_HALF
                else:
                    line += EMPTY
        lines.append(line)
    return lines


class LiveGenWidget(Widget):
    """Shows pieces being generated with a typing animation effect."""

    DEFAULT_CSS = """
    LiveGenWidget {
        width: 100%;
        height: 100%;
        padding: 1;
        content-align: center middle;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pieces: list[np.ndarray] = []
        self._display_count = 0
        self._total_count = 0
        self._show_cols = 8
        self._max_display = 16

    def update_pieces(self, pieces: list[np.ndarray]):
        """Update with new completed pieces from generation."""
        self._pieces = pieces[:self._max_display]
        self._display_count = len(self._pieces)
        self._total_count = len(pieces)
        self.refresh()

    def clear(self):
        self._pieces.clear()
        self._display_count = 0
        self._total_count = 0
        self.refresh()

    def render(self) -> Text:
        result = Text()
        result.append("LIVE GENERATION\n", style="bold magenta")
        result.append(f"Showing {self._display_count} of {self._total_count} pieces\n", style="dim")
        result.append("─" * 60 + "\n", style="dim")

        if not self._pieces:
            result.append("\n  Waiting for generation...\n", style="dim italic")
            return result

        # Render pieces in grid
        for row_start in range(0, len(self._pieces), self._show_cols):
            row = self._pieces[row_start:row_start + self._show_cols]

            rendered = []
            for grid in row:
                rows, cols = grid.shape
                lines = []
                for y in range(0, rows, 2):
                    line = ""
                    for x in range(cols):
                        top = grid[y, x]
                        bot = grid[y + 1, x] if y + 1 < rows else 0
                        if top and bot:
                            line += FULL_BLOCK
                        elif top:
                            line += UPPER_HALF
                        elif bot:
                            line += LOWER_HALF
                        else:
                            line += EMPTY
                    lines.append(line)
                rendered.append(lines)

            n_lines = max(len(r) for r in rendered)
            for line_idx in range(n_lines):
                for piece_lines in rendered:
                    if line_idx < len(piece_lines):
                        result.append(" " + piece_lines[line_idx], style="white on #1a1a1a")
                    else:
                        result.append(" " + " " * 16)
                    result.append(" ")
                result.append("\n")
            result.append("\n")

        return result
