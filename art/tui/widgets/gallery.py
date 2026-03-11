import numpy as np
from textual.widget import Widget
from rich.text import Text

UPPER_HALF = "▀"
LOWER_HALF = "▄"
FULL_BLOCK = "█"
EMPTY = " "


def _render_small(grid: np.ndarray) -> list[str]:
    """Render 16x16 grid as 8 lines of 16 chars using half-blocks."""
    rows, cols = grid.shape
    lines = []
    for y in range(0, rows, 2):
        line = ""
        for x in range(cols):
            top = grid[y, x] if y < rows else 0
            bot = grid[y + 1, x] if y + 1 < rows else 0
            if top and bot:
                line += FULL_BLOCK
            elif top and not bot:
                line += UPPER_HALF
            elif not top and bot:
                line += LOWER_HALF
            else:
                line += EMPTY
        lines.append(line)
    return lines


class GalleryGrid(Widget):
    """Shows a grid of pixel art pieces with scores."""

    DEFAULT_CSS = """
    GalleryGrid {
        width: 100%;
        height: 100%;
    }
    """

    def __init__(self, cols: int = 4, max_pieces: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.cols = cols
        self.max_pieces = max_pieces
        self._pieces: list[tuple[np.ndarray, int, float]] = []  # (grid, index, score)

    def update_pieces(self, pieces: list[np.ndarray], scores: list[dict]):
        """Update with new pieces, sorted by score descending. Keep top max_pieces."""
        indexed = sorted(
            enumerate(scores),
            key=lambda x: x[1].get("composite", 0),
            reverse=True
        )[:self.max_pieces]

        self._pieces = [
            (pieces[idx], idx, scores[idx].get("composite", 0))
            for idx, _ in indexed
            if idx < len(pieces)
        ]
        self.refresh()

    def render(self) -> Text:
        if not self._pieces:
            result = Text()
            result.append("GALLERY\n", style="bold magenta")
            result.append("─" * 30 + "\n", style="dim")
            result.append("  Waiting for generation...", style="dim italic")
            return result

        result = Text()
        result.append("BEST OF GENERATION\n", style="bold magenta")
        result.append("─" * (self.cols * 18) + "\n", style="dim")

        # Arrange pieces in rows of self.cols
        for row_start in range(0, len(self._pieces), self.cols):
            row_pieces = self._pieces[row_start : row_start + self.cols]

            # Header line: indices
            for grid, idx, score in row_pieces:
                label = f"  #{idx:<4d}           "[:18]
                result.append(label, style="bold cyan")
            result.append("\n")

            # Render all pieces in this row side by side
            rendered = [_render_small(g) for g, _, _ in row_pieces]
            n_lines = max(len(r) for r in rendered) if rendered else 0

            for line_idx in range(n_lines):
                for piece_lines in rendered:
                    if line_idx < len(piece_lines):
                        result.append("  " + piece_lines[line_idx], style="bright_green on #0a0f0a")
                    else:
                        result.append("  " + " " * 16)
                    result.append("  ")  # gap between pieces
                result.append("\n")

            # Score line
            for grid, idx, score in row_pieces:
                score_str = f"  {score:.3f}          "[:18]
                color = "green" if score >= 0.6 else "yellow" if score >= 0.4 else "red"
                result.append(score_str, style=color)
            result.append("\n\n")

        return result
