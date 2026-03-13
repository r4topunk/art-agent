"""Full-screen mural — tiles generation pieces across the terminal, live pixel-by-pixel."""
from __future__ import annotations

import numpy as np
from rich.text import Text
from textual.widget import Widget

from art.config import PALETTE_TERM

UPPER_HALF = "▀"
BLACK = PALETTE_TERM[0]


class WallpaperWidget(Widget):
    """Tiles pieces edge-to-edge across the full terminal, updated live during generation."""

    DEFAULT_CSS = """
    WallpaperWidget {
        width: 100%;
        height: 100%;
        padding: 0;
        overflow: hidden hidden;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._grids: list[np.ndarray] = []
        self._phase: str = "waiting"
        self._generation: int = 0
        self._finetune_step: int = 0
        self._finetune_total: int = 0
        self._selected_pieces: list[np.ndarray] = []  # pieces selected for finetuning
        self._cycle_index: int = 0
        self._display_mode: str = "wallpaper"  # "wallpaper" or "grid"

    def on_mount(self):
        self.set_interval(0.33, self._cycle_next_piece)

    def _cycle_next_piece(self):
        """During finetuning, flash through selected pieces one at a time."""
        if self._phase != "finetuning" or not self._selected_pieces:
            return
        n = len(self._selected_pieces)
        self._cycle_index = (self._cycle_index + 1) % max(1, n)
        self._grids = [self._selected_pieces[self._cycle_index]]
        self.refresh()

    def toggle_display_mode(self):
        self._display_mode = "grid" if self._display_mode == "wallpaper" else "wallpaper"
        self.refresh()

    # ------------------------------------------------------------------ events from app

    def update_gen_start(self, generation: int, temperature: float):
        self._generation = generation
        if generation == 0:
            return
        self._phase = "generating"
        self.refresh()

    def update_progress(self, grids: list[np.ndarray], pixel: int, total_pixels: int):
        if self._generation == 0:
            return
        self._grids = grids
        self._phase = "generating"
        self.refresh()

    def update_scoring(self, done: int, total: int):
        self._phase = "scoring"
        self.refresh()

    def update_scored(self, pieces: list[np.ndarray], scores: list[dict]):
        self._grids = pieces
        self._phase = "scored"
        self.refresh()

    def update_selected(self, indices: list[int]):
        self.refresh()

    def update_selected_pieces(self, pieces: list[np.ndarray]):
        self._selected_pieces = pieces

    def update_finetune(self, step: int, total: int):
        self._phase = "finetuning"
        self._finetune_step = step
        self._finetune_total = total
        if step % 5 == 0:
            self.refresh()

    # ------------------------------------------------------------------ render

    def render(self) -> Text:
        w = self.size.width
        h = self.size.height

        if not self._grids or w == 0 or h == 0:
            result = Text()
            pad = max(0, h // 2 - 1)
            result.append("\n" * pad)
            msg = f"  gen {self._generation}  waiting for pixels..." if self._phase != "waiting" else "  [M]ural — waiting for generation..."
            left = max(0, (w - len(msg)) // 2)
            result.append(" " * left + msg, style="dim italic")
            return result

        if self._display_mode == "grid":
            return self._render_grid(w, h)
        return self._render_wallpaper(w, h)

    def _render_wallpaper(self, w: int, h: int) -> Text:
        pieces_x = max(1, (w + 15) // 16)
        pieces_y = max(1, (h * 2 + 15) // 16)
        n = len(self._grids)

        mural_pw = pieces_x * 16
        mural_ph = pieces_y * 16
        mural = np.zeros((mural_ph, mural_pw), dtype=np.int32)

        for py in range(pieces_y):
            for px in range(pieces_x):
                idx = (py * pieces_x + px) % n
                piece = self._grids[idx]
                r0, c0 = py * 16, px * 16
                pr, pc = piece.shape[:2]
                rr, cc = min(pr, 16), min(pc, 16)
                mural[r0:r0 + rr, c0:c0 + cc] = piece[:rr, :cc]

        mode_hint = "[F]Grid" if self._display_mode == "wallpaper" else "[F]Full"
        status = ""
        if self._phase == "finetuning":
            pct = self._finetune_step / max(1, self._finetune_total) * 100
            status = f" gen {self._generation}  finetuning {pct:.0f}%  {mode_hint} "
        elif self._phase == "scoring":
            status = f" gen {self._generation}  scoring...  {mode_hint} "
        else:
            status = f" {mode_hint} "

        result = Text()
        for tr in range(h):
            py0 = tr * 2
            py1 = py0 + 1
            is_status_row = tr == h - 1 and status
            status_start = max(0, (w - len(status)) // 2) if is_status_row else w
            status_end = status_start + len(status) if is_status_row else w
            for tc in range(w):
                if is_status_row and status_start <= tc < status_end:
                    result.append(status[tc - status_start], style="bold white on #1a1a2e")
                else:
                    top = int(mural[py0, tc]) if py0 < mural_ph else 0
                    bot = int(mural[py1, tc]) if py1 < mural_ph else 0
                    result.append(UPPER_HALF, style=f"{PALETTE_TERM[top]} on {PALETTE_TERM[bot]}")
            if tr < h - 1:
                result.append("\n")

        return result

    def _render_grid(self, w: int, h: int) -> Text:
        pieces = self._grids[:36]
        n = len(pieces)
        if n == 0:
            return Text("  no pieces  [F]Full", style="dim italic")

        cols = 6
        rows = min(6, (n + cols - 1) // cols)
        cell_w = max(4, (w - cols - 1) // cols)
        cell_h = max(4, (h * 2 - rows - 1) // rows)
        cell_w = min(cell_w, 16)
        cell_h = min(cell_h, 16)

        grid_w = cols * (cell_w + 1) + 1
        grid_h = rows * (cell_h + 1) + 1
        left_pad = max(0, (w - grid_w) // 2)
        top_pad = max(0, (h - (grid_h + 1) // 2) // 2)

        result = Text()
        result.append("\n" * top_pad)

        for row in range(rows):
            result.append(" " * left_pad + "─" * grid_w + "\n", style="dim")
            term_lines = cell_h // 2
            for tl in range(term_lines):
                result.append(" " * left_pad)
                result.append("│", style="dim")
                for col in range(cols):
                    idx = row * cols + col
                    if idx < n:
                        piece = pieces[idx]
                        py0 = tl * 2
                        py1 = py0 + 1
                        for tc in range(cell_w):
                            px_col = tc * 16 // cell_w
                            top_px = int(piece[py0 * 16 // cell_h, px_col]) if py0 < cell_h else 0
                            bot_px = int(piece[py1 * 16 // cell_h, px_col]) if py1 < cell_h else 0
                            result.append(UPPER_HALF, style=f"{PALETTE_TERM[top_px]} on {PALETTE_TERM[bot_px]}")
                    else:
                        result.append(" " * cell_w)
                    result.append("│", style="dim")
                if tl < term_lines - 1 or row < rows - 1:
                    result.append("\n")

        result.append("\n" + " " * left_pad + "─" * grid_w, style="dim")
        return result
