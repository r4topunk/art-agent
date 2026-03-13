from __future__ import annotations

import numpy as np
from rich.text import Text
from textual.widget import Widget
from textual.reactive import reactive

from art.config import PALETTE_TERM

UPPER_HALF = "▀"
FULL_BLOCK = "█"
EMPTY = " "
BAR_FULL = "█"
BAR_EMPTY = "░"


def _render_small(grid: np.ndarray) -> list[Text]:
    """Render grid using half-blocks with 16-color palette."""
    rows, cols = grid.shape
    lines = []
    for y in range(0, rows, 2):
        line = Text()
        for x in range(cols):
            top = int(grid[y, x]) if y < rows else 0
            bot = int(grid[y + 1, x]) if y + 1 < rows else 0
            fg = PALETTE_TERM[top]
            bg = PALETTE_TERM[bot]
            line.append(UPPER_HALF, style=f"{fg} on {bg}")
        lines.append(line)
    return lines


def _render_large(grid: np.ndarray) -> list[Text]:
    """Render grid at 2x scale with color."""
    rows, cols = grid.shape
    lines = []
    for y in range(rows):
        line = Text()
        for x in range(cols):
            color = PALETTE_TERM[int(grid[y, x])]
            line.append(FULL_BLOCK * 2, style=color)
        lines.append(line)
    return lines


def _score_bar(value: float, width: int = 10) -> str:
    filled = int(value * width)
    return BAR_FULL * filled + BAR_EMPTY * (width - filled)


class ReviewGrid(Widget):
    """Browsable grid of pieces with cursor selection."""

    DEFAULT_CSS = """
    ReviewGrid {
        width: 100%;
        height: 100%;
        padding: 1;
    }
    """

    cursor = reactive(0)

    def __init__(self, cols: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.cols = cols
        self._pieces: list[tuple[np.ndarray, int, dict]] = []
        self._favorites: set[int] = set()

    def set_pieces(self, pieces: list[np.ndarray], scores: list[dict]):
        indexed = sorted(
            enumerate(scores),
            key=lambda x: x[1].get("composite", 0),
            reverse=True,
        )
        self._pieces = [(pieces[i], i, scores[i]) for i, _ in indexed if i < len(pieces)]
        self._favorites.clear()
        self.cursor = 0
        self.refresh()

    def toggle_favorite(self):
        if not self._pieces:
            return
        _, orig_idx, _ = self._pieces[self.cursor]
        if orig_idx in self._favorites:
            self._favorites.discard(orig_idx)
        else:
            self._favorites.add(orig_idx)
        self.refresh()

    def get_favorites(self) -> list[int]:
        return sorted(self._favorites)

    def get_current(self) -> tuple[np.ndarray, int, dict] | None:
        if not self._pieces:
            return None
        return self._pieces[self.cursor]

    def move_cursor(self, delta: int):
        if self._pieces:
            self.cursor = max(0, min(len(self._pieces) - 1, self.cursor + delta))
            self.refresh()

    def render(self) -> Text:
        if not self._pieces:
            return Text("No pieces to review", style="dim")

        result = Text()
        result.append(f"REVIEW  ({len(self._favorites)} favorites selected)\n", style="bold magenta")
        result.append("─" * (self.cols * 20) + "\n", style="dim")

        for row_start in range(0, len(self._pieces), self.cols):
            row = self._pieces[row_start : row_start + self.cols]

            for i, (grid, orig_idx, scores) in enumerate(row):
                pos = row_start + i
                is_cursor = pos == self.cursor
                is_fav = orig_idx in self._favorites

                marker = ">" if is_cursor else " "
                fav = "★" if is_fav else " "
                label = f"{marker}{fav}#{orig_idx:<4d}          "[:18]
                style = "bold reverse" if is_cursor else ("bold yellow" if is_fav else "cyan")
                result.append(label, style=style)
            result.append("\n")

            rendered = [_render_small(g) for g, _, _ in row]
            n_lines = max(len(r) for r in rendered) if rendered else 0

            for line_idx in range(n_lines):
                for j, piece_lines in enumerate(rendered):
                    result.append("  ")
                    if line_idx < len(piece_lines):
                        result.append(piece_lines[line_idx])
                    else:
                        result.append(" " * 16)
                    result.append("  ")
                result.append("\n")

            for grid, orig_idx, scores in row:
                score = scores.get("composite", 0)
                color = "green" if score >= 0.6 else "yellow" if score >= 0.4 else "red"
                result.append(f"  {score:.3f}          "[:18], style=color)
            result.append("\n\n")

        return result


class DetailPanel(Widget):
    """Shows enlarged piece + score breakdown for the selected piece."""

    DEFAULT_CSS = """
    DetailPanel {
        width: 100%;
        height: auto;
        padding: 1;
        border-top: solid $accent;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._grid: np.ndarray | None = None
        self._scores: dict = {}
        self._index: int = 0
        self._is_favorite: bool = False

    def update_detail(self, grid: np.ndarray, index: int, scores: dict, is_favorite: bool):
        self._grid = grid
        self._index = index
        self._scores = scores
        self._is_favorite = is_favorite
        self.refresh()

    def render(self) -> Text:
        if self._grid is None:
            return Text("Select a piece to view details", style="dim")

        result = Text()
        fav_mark = " ★ FAVORITE" if self._is_favorite else ""
        result.append(f"DETAIL — Piece #{self._index}{fav_mark}\n", style="bold magenta")
        result.append("─" * 50 + "\n", style="dim")

        large_lines = _render_large(self._grid)

        score_lines = []
        for key in ["symmetry", "complexity", "structure", "aesthetics", "diversity", "composite"]:
            val = self._scores.get(key, 0)
            bar = _score_bar(val)
            color = "green" if val >= 0.6 else "yellow" if val >= 0.4 else "red"
            label = key.capitalize()[:10]
            score_lines.append((f"  {label:<11} {val:.3f} ", bar, color))

        max_lines = max(len(large_lines), len(score_lines))

        for i in range(max_lines):
            if i < len(large_lines):
                result.append(large_lines[i])
            else:
                result.append(" " * 32)

            result.append("    ")

            if i < len(score_lines):
                label, bar, color = score_lines[i]
                result.append(label, style="white")
                result.append(bar, style=color)

            result.append("\n")

        result.append("\n")
        result.append(
            "  [←][→] Navigate   [↑][↓] Rows   [SPACE] Toggle favorite   [ENTER] Confirm   [ESC] Back\n",
            style="dim",
        )

        return result
