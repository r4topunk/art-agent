from __future__ import annotations
import numpy as np
from rich.text import Text
from textual.widget import Widget

UPPER_HALF = "▀"
LOWER_HALF = "▄"
FULL_BLOCK = "█"
EMPTY = " "
# Braille patterns: 2x4 dot matrix per character
BRAILLE_BASE = 0x2800

def _braille_dot(x: int, y: int) -> int:
    """Get braille bit offset for position (x=0..1, y=0..3)."""
    if x == 0:
        return [0, 1, 2, 6][y]
    else:
        return [3, 4, 5, 7][y]


def _render_mini(grid: np.ndarray) -> list[str]:
    """Render 16x16 as 8 lines of 16 chars using half-blocks."""
    rows, cols = grid.shape
    lines = []
    for y in range(0, rows, 2):
        line = ""
        for x in range(cols):
            top = grid[y, x]
            bot = grid[y + 1, x] if y + 1 < rows else 0
            if top and bot: line += FULL_BLOCK
            elif top: line += UPPER_HALF
            elif bot: line += LOWER_HALF
            else: line += EMPTY
        lines.append(line)
    return lines


class DiversityWidget(Widget):
    DEFAULT_CSS = """
    DiversityWidget {
        width: 100%;
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pieces: list[np.ndarray] = []
        self._selected_indices: list[int] = []
        self._scores: list[dict] = []

    def update_generation(self, pieces: list[np.ndarray], scores: list[dict], selected_indices: list[int]):
        self._pieces = pieces
        self._scores = scores
        self._selected_indices = selected_indices
        self.refresh()

    def _make_scatter(self, width: int = 30, height: int = 12) -> Text:
        """Simple 2D scatter of pieces using braille dots."""
        if len(self._pieces) < 2:
            return Text("  Not enough pieces\n", style="dim")

        # Project 256-dim -> 2D using two fixed random projections
        rng = np.random.RandomState(42)
        proj = rng.randn(2, 256)

        flats = np.array([p.flatten().astype(np.float32) for p in self._pieces])
        coords = flats @ proj.T  # (n, 2)

        # Normalize to grid
        for dim in range(2):
            mn, mx = coords[:, dim].min(), coords[:, dim].max()
            rng_d = mx - mn if mx > mn else 1.0
            coords[:, dim] = (coords[:, dim] - mn) / rng_d

        # Create braille canvas
        char_w = width
        char_h = height
        canvas = [[0] * char_w for _ in range(char_h)]

        selected_set = set(self._selected_indices)
        point_types = {}  # (cx, cy) -> is_selected

        for i, (x, y) in enumerate(coords):
            cx = min(char_w * 2 - 1, max(0, int(x * (char_w * 2 - 1))))
            cy = min(char_h * 4 - 1, max(0, int(y * (char_h * 4 - 1))))
            char_x = cx // 2
            char_y = cy // 4
            dot_x = cx % 2
            dot_y = cy % 4
            canvas[char_y][char_x] |= (1 << _braille_dot(dot_x, dot_y))
            if i in selected_set:
                point_types[(char_x, char_y)] = True
            elif (char_x, char_y) not in point_types:
                point_types[(char_x, char_y)] = False

        result = Text()
        for y in range(char_h):
            result.append("  ")
            for x in range(char_w):
                ch = chr(BRAILLE_BASE + canvas[y][x])
                is_sel = point_types.get((x, y), None)
                if is_sel is True:
                    result.append(ch, style="bright_yellow")
                elif is_sel is False:
                    result.append(ch, style="bright_black")
                else:
                    result.append(" ")
            result.append("\n")

        return result

    def render(self) -> Text:
        result = Text()
        result.append("⬡ DIVERSITY\n", style="bold blue")

        if not self._pieces:
            result.append("  No generation data\n", style="dim")
            return result

        # Scatter plot
        result.append(self._make_scatter())

        n_sel = len(self._selected_indices)
        n_total = len(self._pieces)
        result.append(f"  ★ selected: {n_sel}  ", style="bright_yellow")
        result.append(f"· rejected: {n_total - n_sel}\n", style="bright_black")

        # Museum vs Cemetery: top 3 selected vs bottom 3 unselected
        if self._scores:
            ranked = sorted(range(len(self._scores)), key=lambda i: self._scores[i].get("composite", 0), reverse=True)
            selected_set = set(self._selected_indices)

            museum = [i for i in ranked if i in selected_set][:3]
            cemetery = [i for i in reversed(ranked) if i not in selected_set][:3]

            if museum and cemetery:
                result.append("\n  MUSEUM          CEMETERY\n", style="bold")

                # Render side by side, using just 2 lines per piece
                museum_rendered = [_render_mini(self._pieces[i]) for i in museum]
                cemetery_rendered = [_render_mini(self._pieces[i]) for i in cemetery]

                for line_idx in range(min(4, max(len(r) for r in museum_rendered))):  # Show first 4 rows
                    result.append("  ")
                    for piece_lines in museum_rendered:
                        if line_idx < len(piece_lines):
                            result.append(piece_lines[line_idx][:8], style="white")  # half width
                        result.append(" ")
                    result.append("  ")
                    for piece_lines in cemetery_rendered:
                        if line_idx < len(piece_lines):
                            result.append(piece_lines[line_idx][:8], style="bright_black")  # dimmed
                        result.append(" ")
                    result.append("\n")

        return result
