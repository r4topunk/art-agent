import numpy as np
from rich.text import Text
from textual.widget import Widget
from textual.reactive import reactive

UPPER_HALF = "▀"
LOWER_HALF = "▄"
FULL_BLOCK = "█"
EMPTY = " "

def render_piece(grid: np.ndarray, fg_color: str = "white", bg_color: str = "black") -> Text:
    """Render a 16x16 binary grid as Rich Text using half-block characters.

    Grid values: 1 = white/filled, 0 = black/empty.
    Returns Text with 8 lines (2 pixels per row).
    """
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

    text = Text("\n".join(lines))
    text.stylize(f"{fg_color} on {bg_color}")
    return text


def render_piece_large(grid: np.ndarray, fg_color: str = "white", bg_color: str = "black") -> Text:
    """Render at 2x width (each pixel = 2 chars wide). 16x16 becomes 16 rows x 32 cols."""
    rows, cols = grid.shape
    lines = []
    for y in range(rows):
        line = ""
        for x in range(cols):
            if grid[y, x]:
                line += FULL_BLOCK * 2
            else:
                line += EMPTY * 2
        lines.append(line)

    text = Text("\n".join(lines))
    text.stylize(f"{fg_color} on {bg_color}")
    return text


def render_gallery_item(grid: np.ndarray, index: int, score: float) -> Text:
    """Render a single gallery item: index label + pixel art + score."""
    header = Text(f"#{index:>3d}", style="bold cyan")
    art = render_piece(grid)
    footer = Text(f"{score:.2f}", style="yellow")

    result = Text()
    result.append(header)
    result.append("\n")
    result.append(art)
    result.append("\n")
    result.append(footer)
    return result


class PixelArtWidget(Widget):
    """Textual widget that displays a single piece of pixel art."""

    DEFAULT_CSS = """
    PixelArtWidget {
        width: auto;
        height: auto;
        padding: 0 1;
    }
    """

    grid = reactive(None)
    label = reactive("")
    score = reactive(0.0)

    def __init__(self, grid: np.ndarray | None = None, label: str = "", score: float = 0.0, large: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._grid = grid
        self.label = label
        self.score = score
        self.large = large

    def on_mount(self):
        if self._grid is not None:
            self.grid = self._grid

    def render(self) -> Text:
        if self.grid is None:
            return Text("(empty)", style="dim")

        result = Text()
        if self.label:
            result.append(Text(self.label, style="bold cyan"))
            result.append("\n")

        if self.large:
            result.append(render_piece_large(self.grid))
        else:
            result.append(render_piece(self.grid))

        if self.score > 0:
            result.append("\n")
            result.append(Text(f"{self.score:.2f}", style="yellow"))

        return result

    def update_piece(self, grid: np.ndarray, label: str = "", score: float = 0.0):
        self._grid = grid
        self.grid = grid
        self.label = label
        self.score = score
        self.refresh()
