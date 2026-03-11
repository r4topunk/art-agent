"""Full-screen generation watch — shows all images being drawn pixel by pixel."""
from __future__ import annotations

import numpy as np
from rich.text import Text
from textual.widget import Widget

from art.config import PALETTE_TERM

UPPER_HALF = "▀"


def _render_small(grid: np.ndarray) -> list[Text]:
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


class GenWatchPanel(Widget):
    """Displays all generation images in a dense grid with live progress."""

    DEFAULT_CSS = """
    GenWatchPanel {
        width: 100%;
        height: 100%;
        padding: 1 2;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._grids: list[np.ndarray] = []
        self._pixel: int = 0
        self._total_pixels: int = 256
        self._scores: list[dict] = []
        self._selected: set[int] = set()
        self._generation: int = 0
        self._temperature: float = 0.0
        self._phase: str = "waiting"  # waiting, generating, scoring, scored
        self._scoring_done: int = 0
        self._scoring_total: int = 0

    def update_gen_start(self, generation: int, temperature: float):
        self._generation = generation
        self._temperature = temperature
        self._phase = "generating"
        self._grids = []
        self._scores = []
        self._selected = set()
        self._pixel = 0
        self.refresh()

    def update_progress(self, grids: list[np.ndarray], pixel: int, total_pixels: int):
        self._grids = grids
        self._pixel = pixel
        self._total_pixels = total_pixels
        self._phase = "generating"
        self.refresh()

    def update_scoring(self, done: int, total: int):
        self._scoring_done = done
        self._scoring_total = total
        self._phase = "scoring"
        self.refresh()

    def update_scored(self, pieces: list[np.ndarray], scores: list[dict]):
        self._grids = pieces
        self._scores = scores
        self._phase = "scored"
        self.refresh()

    def update_selected(self, indices: list[int]):
        self._selected = set(indices)
        self.refresh()

    @property
    def _cw(self) -> int:
        return max(20, self.size.width - 6)

    @property
    def _fit_cols(self) -> int:
        return max(1, (self._cw + 1) // 17)  # 16px + 1 spacing

    def render(self) -> Text:
        result = Text()
        cols = self._fit_cols

        # Header
        result.append("GENERATION WATCH", style="bold magenta")
        result.append(f"  gen {self._generation}", style="bold cyan")
        result.append(f"  temp={self._temperature:.3f}", style="dim cyan")
        result.append(f"  {len(self._grids)} pieces\n", style="dim")

        # Phase-specific status
        if self._phase == "waiting":
            result.append("  Waiting for generation to start...\n", style="dim italic")
            return result

        if self._phase == "generating":
            pct = self._pixel / max(1, self._total_pixels) * 100
            row = self._pixel // 16
            result.append(f"  Drawing row {row}/16  ", style="dim")
            result.append(f"{pct:.0f}%  ", style="bold cyan")
            result.append(self._bar(pct))
            result.append("\n\n")
        elif self._phase == "scoring":
            pct = self._scoring_done / max(1, self._scoring_total) * 100
            result.append(f"  Scoring {self._scoring_done}/{self._scoring_total}  ", style="dim")
            result.append(f"{pct:.0f}%  ", style="bold yellow")
            result.append(self._bar(pct, style="yellow"))
            result.append("\n\n")
        elif self._phase == "scored":
            composites = [s.get("composite", 0) for s in self._scores]
            if composites:
                best = max(composites)
                mean = sum(composites) / len(composites)
                result.append(f"  best={best:.3f}  mean={mean:.3f}", style="bold bright_green")
                result.append(f"  {len(self._selected)} selected\n\n", style="bold bright_yellow")
            else:
                result.append("\n")

        if not self._grids:
            return result

        # Render all pieces in a dense grid
        has_scores = self._phase == "scored" and self._scores
        for row_start in range(0, len(self._grids), cols):
            row_grids = self._grids[row_start: row_start + cols]
            row_indices = list(range(row_start, min(row_start + cols, len(self._grids))))
            piece_rows = [_render_small(g) for g in row_grids]
            n_lines = max((len(r) for r in piece_rows), default=0)

            # Index labels
            result.append(" ")
            for i, idx in enumerate(row_indices):
                if i > 0:
                    result.append(" ")
                is_sel = idx in self._selected
                style = "bold bright_yellow" if is_sel else "dim"
                label = f"{'*' if is_sel else ' '}{idx:>2d}"
                pad = 16 - len(label)
                result.append(label, style=style)
                result.append(" " * pad)
            result.append("\n")

            # Pixel art rows
            for line_idx in range(n_lines):
                result.append(" ")
                for i, lines in enumerate(piece_rows):
                    if i > 0:
                        result.append(" ")
                    if line_idx < len(lines):
                        result.append(lines[line_idx])
                result.append("\n")

            # Score labels
            if has_scores:
                result.append(" ")
                for i, idx in enumerate(row_indices):
                    if i > 0:
                        result.append(" ")
                    score = self._scores[idx].get("composite", 0) if idx < len(self._scores) else 0
                    is_sel = idx in self._selected
                    if is_sel:
                        color = "bold bright_yellow"
                    elif score >= 0.6:
                        color = "bright_green"
                    elif score >= 0.4:
                        color = "yellow"
                    else:
                        color = "red"
                    label = f"{score:.3f}"
                    pad = 16 - len(label)
                    result.append(label, style=color)
                    result.append(" " * pad)
                result.append("\n")

            result.append("\n")

        # Footer
        result.append("  [D] Dashboard  [R] Review  [ESC] Back  [Q] Quit\n", style="dim")
        return result

    def _bar(self, pct: float, width: int | None = None, style: str = "green") -> Text:
        w = width or min(40, self._cw - 30)
        filled = int(pct / 100 * w)
        t = Text()
        for i in range(w):
            t.append("█" if i < filled else "░", style=style if i < filled else "bright_black")
        return t
