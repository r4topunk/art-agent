"""Full-screen mural — tiles generation pieces across the terminal, live pixel-by-pixel."""
from __future__ import annotations

import random

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
        self._selected_pieces: list[np.ndarray] = []
        self._display_mode: str = "wallpaper"

    def on_mount(self):
        self.set_interval(0.111, self._flash_tick_fn)

    @staticmethod
    def _random_transform(piece: np.ndarray) -> np.ndarray:
        p = np.rot90(piece, random.randint(0, 3))
        if random.random() < 0.5:
            p = np.fliplr(p)
        return p

    def _flash_tick_fn(self):
        """Drive the finetuning kaleidoscope animation."""
        if self._phase != "finetuning" or not self._selected_pieces:
            return
        self.refresh()

    def _pick_pool(self) -> list[np.ndarray]:
        """Pick 1–5 random source pieces for this frame."""
        k = random.randint(1, min(5, len(self._selected_pieces)))
        return random.sample(self._selected_pieces, k)

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

        if self._phase == "finetuning" and self._selected_pieces:
            return self._render_kaleidoscope(w, h)
        if self._display_mode == "grid":
            return self._render_grid(w, h)
        return self._render_wallpaper(w, h)

    # ------------------------------------------------------------------ kaleidoscope flash

    def _render_kaleidoscope(self, w: int, h: int) -> Text:
        """4-way mirrored symmetry centered on screen."""
        pixel_h = h * 2
        cols = max(1, (w + 15) // 16)
        rows = max(1, (pixel_h + 15) // 16)
        cell = 16

        half_cols = max(1, (cols + 1) // 2)
        half_rows = max(1, (rows + 1) // 2)

        # Build top-left quadrant
        pool = self._pick_pool()
        quad_h = half_rows * cell
        quad_w = half_cols * cell
        quad = np.zeros((quad_h, quad_w), dtype=np.int32)
        for gy in range(half_rows):
            for gx in range(half_cols):
                piece = self._random_transform(random.choice(pool))
                r0, c0 = gy * cell, gx * cell
                pr, pc = piece.shape[:2]
                rr, cc = min(pr, cell, quad_h - r0), min(pc, cell, quad_w - c0)
                if rr > 0 and cc > 0:
                    quad[r0:r0 + rr, c0:c0 + cc] = piece[:rr, :cc]

        # Mirror into full symmetric mural: right = fliplr, bottom = flipud
        right = np.fliplr(quad)
        top_half = np.concatenate([quad, right], axis=1)
        bottom_half = np.flipud(top_half)
        full = np.concatenate([top_half, bottom_half], axis=0)

        mural_h, mural_w = full.shape

        # Center on screen
        off_x = max(0, (mural_w - w) // 2)
        off_y = max(0, (mural_h - pixel_h) // 2)

        # Status overlay
        pct = self._finetune_step / max(1, self._finetune_total) * 100
        status = f" finetuning {pct:.0f}% "

        result = Text()
        for tr in range(h):
            py0 = off_y + tr * 2
            py1 = py0 + 1
            is_status_row = tr == h - 1
            status_start = max(0, (w - len(status)) // 2) if is_status_row else w
            status_end = status_start + len(status) if is_status_row else w
            for tc in range(w):
                if is_status_row and status_start <= tc < status_end:
                    result.append(status[tc - status_start], style="bold white on #1a1a2e")
                else:
                    mx = off_x + tc
                    top = int(full[py0, mx]) if 0 <= py0 < mural_h and 0 <= mx < mural_w else 0
                    bot = int(full[py1, mx]) if 0 <= py1 < mural_h and 0 <= mx < mural_w else 0
                    result.append(UPPER_HALF, style=f"{PALETTE_TERM[top]} on {PALETTE_TERM[bot]}")
            if tr < h - 1:
                result.append("\n")

        return result

    # ------------------------------------------------------------------ normal renders

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

        status = ""
        if self._phase == "scoring":
            status = f" gen {self._generation}  scoring... "

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
            return Text("  no pieces", style="dim italic")

        cols = 6
        rows = min(6, (n + cols - 1) // cols)
        cell_w = max(4, w // cols)
        cell_h = max(4, (h * 2) // rows)
        cell_w = min(cell_w, 16)
        cell_h = min(cell_h, 16)

        grid_w = cols * cell_w
        left_pad = max(0, (w - grid_w) // 2)

        total_term_rows = rows * (cell_h // 2)
        top_pad = max(0, (h - total_term_rows) // 2)

        result = Text()
        result.append("\n" * top_pad)

        for row in range(rows):
            term_lines = cell_h // 2
            for tl in range(term_lines):
                result.append(" " * left_pad)
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
                result.append("\n")

        return result
