from __future__ import annotations
import numpy as np
from collections import Counter
from rich.text import Text
from textual.widget import Widget

UPPER_HALF = "▀"
LOWER_HALF = "▄"
FULL_BLOCK = "█"
EMPTY = " "


def _render_motif_2x4(block: np.ndarray) -> Text:
    """Render a 4x4 block as 2 rows of 4 chars using half-blocks."""
    text = Text()
    for y in range(0, 4, 2):
        for x in range(4):
            top = block[y, x]
            bot = block[y + 1, x] if y + 1 < 4 else 0
            if top and bot:
                text.append(FULL_BLOCK, style="white")
            elif top:
                text.append(UPPER_HALF, style="white")
            elif bot:
                text.append(LOWER_HALF, style="white")
            else:
                text.append(EMPTY)
        if y < 2:
            text.append("\n")
    return text


def extract_motifs(pieces: list[np.ndarray], top_n: int = 8) -> list[tuple[np.ndarray, int]]:
    """Extract most common 4x4 sub-patterns from pieces."""
    counter: Counter = Counter()
    blocks: dict[bytes, np.ndarray] = {}

    for piece in pieces:
        size = piece.shape[0]
        for r in range(0, size - 3, 2):  # step by 2 for some overlap reduction
            for c in range(0, size - 3, 2):
                block = piece[r:r+4, c:c+4]
                key = block.tobytes()
                counter[key] += 1
                if key not in blocks:
                    blocks[key] = block.copy()

    # Filter out trivial (all black or all white)
    results = []
    for key, count in counter.most_common(top_n + 10):
        block = blocks[key]
        density = np.mean(block)
        if 0.1 < density < 0.9:  # skip trivial
            results.append((block, count))
        if len(results) >= top_n:
            break

    # Pad with most common if not enough non-trivial
    if len(results) < top_n:
        for key, count in counter.most_common(top_n * 2):
            block = blocks[key]
            if not any(np.array_equal(block, r[0]) for r in results):
                results.append((block, count))
            if len(results) >= top_n:
                break

    return results[:top_n]


class MotifsWidget(Widget):
    DEFAULT_CSS = """
    MotifsWidget {
        width: 100%;
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(self, n_motifs: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.n_motifs = n_motifs
        self._current_motifs: list[tuple[np.ndarray, int]] = []
        self._generation: int = 0

    def update_motifs(self, pieces: list[np.ndarray], generation: int):
        self._current_motifs = extract_motifs(pieces, top_n=self.n_motifs)
        self._generation = generation
        self.refresh()

    def render(self) -> Text:
        result = Text()
        result.append("✦ MOTIFS\n", style="bold yellow")

        if not self._current_motifs:
            result.append("  No patterns yet\n", style="dim")
            return result

        result.append(f"  Gen {self._generation} vocabulary:\n", style="dim")

        # Render motifs in a row, 2 lines each (4x4 blocks as 2x4 half-blocks)
        top_lines = []
        bot_lines = []
        count_line = Text("  ")

        for block, count in self._current_motifs:
            # Render each 4x4 as 2 rows
            top = Text()
            bot = Text()
            for x in range(4):
                t = block[0, x]
                b = block[1, x]
                if t and b: top.append(FULL_BLOCK, style="white")
                elif t: top.append(UPPER_HALF, style="white")
                elif b: top.append(LOWER_HALF, style="white")
                else: top.append(" ")

                t2 = block[2, x]
                b2 = block[3, x]
                if t2 and b2: bot.append(FULL_BLOCK, style="white")
                elif t2: bot.append(UPPER_HALF, style="white")
                elif b2: bot.append(LOWER_HALF, style="white")
                else: bot.append(" ")

            top_lines.append(top)
            bot_lines.append(bot)
            count_line.append(f"×{count:<4d}", style="dim yellow")
            count_line.append(" ")

        # Print rows
        line1 = Text("  ")
        line2 = Text("  ")
        for t in top_lines:
            line1.append(t)
            line1.append("  ")
        for b in bot_lines:
            line2.append(b)
            line2.append("  ")

        result.append(line1)
        result.append("\n")
        result.append(line2)
        result.append("\n")
        result.append(count_line)
        result.append("\n")

        return result
