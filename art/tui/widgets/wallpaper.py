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
        self._all_gens: list[list[np.ndarray]] = []  # scored grids per gen, for cycling
        self._cycle_index: int = 0

    def on_mount(self):
        self.set_interval(2.5, self._cycle_next_gen)

    def _cycle_next_gen(self):
        """During finetuning, rotate through all scored generations on the mural."""
        if self._phase == "finetuning" and len(self._all_gens) > 1:
            self._cycle_index = (self._cycle_index + 1) % len(self._all_gens)
            self._grids = self._all_gens[self._cycle_index]
            self.refresh()

    # ------------------------------------------------------------------ events from app

    def update_gen_start(self, generation: int, temperature: float):
        self._generation = generation
        if generation == 0:
            # Hide gen 0 — keep waiting state, nothing to show yet
            return
        # Keep previous grids visible until new pixels arrive
        self._phase = "generating"
        self.refresh()

    def update_progress(self, grids: list[np.ndarray], pixel: int, total_pixels: int):
        if self._generation == 0:
            return  # Don't show gen 0 live pixels
        self._grids = grids
        self._phase = "generating"
        self.refresh()

    def update_scoring(self, done: int, total: int):
        self._phase = "scoring"
        self.refresh()

    def update_scored(self, pieces: list[np.ndarray], scores: list[dict]):
        self._grids = pieces
        self._phase = "scored"
        # Accumulate for finetuning slideshow (deduplicate by identity)
        if not self._all_gens or self._all_gens[-1] is not pieces:
            self._all_gens.append(pieces)
            self._cycle_index = len(self._all_gens) - 1
        self.refresh()

    def update_selected(self, indices: list[int]):
        self.refresh()

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

        # Tile enough pieces to cover the full terminal, including partial edges
        # w terminal cols = w pixel cols; h terminal lines = h*2 pixel rows
        pieces_x = max(1, (w + 15) // 16)   # ceil division — includes partial right column
        pieces_y = max(1, (h * 2 + 15) // 16)  # ceil — includes partial bottom row
        n = len(self._grids)

        # Build oversized mural pixel buffer, then crop to exact terminal size
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

        # Status overlay for the bottom row
        status = ""
        if self._phase == "finetuning":
            pct = self._finetune_step / max(1, self._finetune_total) * 100
            status = f" gen {self._generation}  finetuning {pct:.0f}% "
        elif self._phase == "scoring":
            status = f" gen {self._generation}  scoring... "
        elif self._phase == "generating":
            status = f" gen {self._generation}  generating... "

        # Render exactly w cols × h lines (crop partial pieces at right/bottom)
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
