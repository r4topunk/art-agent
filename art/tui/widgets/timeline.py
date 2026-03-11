from __future__ import annotations
import numpy as np
from rich.text import Text
from textual.widget import Widget

from art.config import PALETTE_TERM

UPPER_HALF = "▀"
SPARK = "▁▂▃▄▅▆▇█"


def _render_row(grid: np.ndarray) -> Text:
    """Render middle 2 rows of grid as a single line using half-blocks with color."""
    size = grid.shape[0]
    mid = size // 2
    line = Text()
    for x in range(size):
        top = int(grid[mid - 1, x])
        bot = int(grid[mid, x])
        fg = PALETTE_TERM[top]
        bg = PALETTE_TERM[bot]
        line.append(UPPER_HALF, style=f"{fg} on {bg}")
    return line


class TimelineWidget(Widget):
    DEFAULT_CSS = """
    TimelineWidget {
        width: 100%;
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._entries: list[tuple[int, np.ndarray, float, float | None]] = []

    @property
    def _max_visible(self) -> int:
        """Fit as many entries as the widget height allows."""
        return max(3, self.size.height - 4)  # border(2) + title(1) + footer(1)

    def add_generation(self, generation: int, best_piece: np.ndarray, best_score: float, vlm_score: float | None = None):
        self._entries.append((generation, best_piece, best_score, vlm_score))
        self.refresh()

    def render(self) -> Text:
        cw = max(16, self.size.width - 4)
        result = Text()
        result.append("⏳ TIMELINE\n", style="bold yellow")

        if not self._entries:
            result.append("  No generations yet\n", style="dim")
            return result

        max_vis = self._max_visible
        visible = self._entries[-max_vis:]

        all_scores = [s for _, _, s, _ in self._entries]
        min_s = min(all_scores) if all_scores else 0
        max_s = max(all_scores) if all_scores else 1
        range_s = max_s - min_s if max_s > min_s else 1.0

        # Bar width fills remaining space after: "  G999 " (7) + piece(16) + " 0.65 " (6) + vlm(5) + star(2)
        bar_w = max(3, cw - 34)

        for gen, piece, score, vlm_score in visible:
            score_color = "green" if score >= 0.6 else "yellow" if score >= 0.4 else "red"

            result.append(f"  G{gen:<3d} ", style="dim cyan")
            result.append(_render_row(piece))
            result.append(f" {score:.2f} ", style=score_color)

            bar_len = int((score - min_s) / range_s * bar_w) if range_s > 0 else bar_w // 2
            result.append("█" * bar_len, style=score_color)

            if vlm_score is not None:
                result.append(f" v{vlm_score:.1f}", style="bright_cyan")

            if score == max_s and len(self._entries) > 1:
                result.append(" ★", style="bright_yellow")

            result.append("\n")

        if len(self._entries) > max_vis:
            result.append(f"  ... {len(self._entries) - max_vis} earlier\n", style="dim")

        return result
