from __future__ import annotations
import numpy as np
from rich.text import Text
from textual.widget import Widget


class DNAWidget(Widget):
    DEFAULT_CSS = """
    DNAWidget {
        width: 100%;
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(self, max_strands: int = 6, display_width: int = 64, **kwargs):
        super().__init__(**kwargs)
        self._strands: list[tuple[int, np.ndarray, float]] = []  # (gen, piece, score) - best per gen
        self.max_strands = max_strands
        self.display_width = display_width

    def add_strand(self, generation: int, piece: np.ndarray, score: float):
        self._strands.append((generation, piece, score))
        if len(self._strands) > self.max_strands:
            self._strands = self._strands[-self.max_strands:]
        self.refresh()

    def render(self) -> Text:
        result = Text()
        result.append("🧬 DNA\n", style="bold green")

        if not self._strands:
            result.append("  No genetic data\n", style="dim")
            return result

        # Each strand is the flattened 256 pixels, shown as colored blocks
        # Subsample to fit display_width
        for gen, piece, score in self._strands:
            flat = piece.flatten()
            # Subsample
            if len(flat) > self.display_width:
                indices = np.linspace(0, len(flat) - 1, self.display_width, dtype=int)
                sampled = flat[indices]
            else:
                sampled = flat

            result.append(f"  G{gen:<3d} ", style="dim cyan")
            for pixel in sampled:
                if pixel:
                    result.append("█", style="white")
                else:
                    result.append("░", style="bright_black")
            result.append(f" {score:.2f}\n", style="yellow")

        # Show mutations: where consecutive strands differ
        if len(self._strands) >= 2:
            _, prev_piece, _ = self._strands[-2]
            _, curr_piece, _ = self._strands[-1]
            prev_flat = prev_piece.flatten()
            curr_flat = curr_piece.flatten()

            if len(prev_flat) > self.display_width:
                indices = np.linspace(0, len(prev_flat) - 1, self.display_width, dtype=int)
                prev_sampled = prev_flat[indices]
                curr_sampled = curr_flat[indices]
            else:
                prev_sampled = prev_flat
                curr_sampled = curr_flat

            result.append("  mut  ", style="dim red")
            for p, c in zip(prev_sampled, curr_sampled):
                if p != c:
                    result.append("▲", style="bright_red")  # mutation
                else:
                    result.append("·", style="bright_black")  # conserved

            n_mutations = int(np.sum(prev_flat != curr_flat))
            pct = n_mutations / len(prev_flat) * 100
            result.append(f" {pct:.0f}%\n", style="red")

        return result
