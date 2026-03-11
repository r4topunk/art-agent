from __future__ import annotations
import numpy as np
from rich.text import Text
from textual.widget import Widget

UPPER_HALF = "▀"
LOWER_HALF = "▄"
FULL_BLOCK = "█"
EMPTY = " "
SPARK = "▁▂▃▄▅▆▇█"


def _render_row(grid: np.ndarray) -> Text:
    """Render a 16x16 grid as a single compact row: 8 chars tall squeezed into 2-line representation using half blocks."""
    # Ultra-compact: just show middle 2 rows as a single line of half-blocks
    size = grid.shape[0]
    mid = size // 2
    line = Text()
    for x in range(size):
        top = grid[mid - 1, x]
        bot = grid[mid, x]
        if top and bot:
            line.append(FULL_BLOCK, style="bright_green")
        elif top:
            line.append(UPPER_HALF, style="bright_green")
        elif bot:
            line.append(LOWER_HALF, style="bright_green")
        else:
            line.append(" ")
    return line


class TimelineWidget(Widget):
    DEFAULT_CSS = """
    TimelineWidget {
        width: 100%;
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(self, max_visible: int = 20, **kwargs):
        super().__init__(**kwargs)
        self._entries: list[tuple[int, np.ndarray, float]] = []  # (gen, best_piece, best_score)
        self.max_visible = max_visible

    def add_generation(self, generation: int, best_piece: np.ndarray, best_score: float):
        self._entries.append((generation, best_piece, best_score))
        self.refresh()

    def render(self) -> Text:
        result = Text()
        result.append("⏳ TIMELINE\n", style="bold yellow")

        if not self._entries:
            result.append("  No generations yet\n", style="dim")
            return result

        # Show last max_visible entries
        visible = self._entries[-self.max_visible:]

        # Score sparkline on the side
        all_scores = [s for _, _, s in self._entries]
        min_s = min(all_scores) if all_scores else 0
        max_s = max(all_scores) if all_scores else 1
        range_s = max_s - min_s if max_s > min_s else 1.0

        for gen, piece, score in visible:
            # Score color
            score_color = "green" if score >= 0.6 else "yellow" if score >= 0.4 else "red"

            # Generation label
            result.append(f"  G{gen:<3d} ", style="dim cyan")

            # Compact piece render (middle slice)
            result.append(_render_row(piece))

            # Score + sparkbar
            result.append(f" {score:.2f} ", style=score_color)

            # Mini bar
            bar_len = int((score - min_s) / range_s * 8) if range_s > 0 else 4
            result.append("█" * bar_len, style=score_color)

            # Mark best ever
            if score == max_s and len(self._entries) > 1:
                result.append(" ★", style="bright_yellow")

            result.append("\n")

        if len(self._entries) > self.max_visible:
            result.append(f"  ... {len(self._entries) - self.max_visible} earlier generations\n", style="dim")

        return result
