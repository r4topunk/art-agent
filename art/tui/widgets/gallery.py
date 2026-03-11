import numpy as np
from textual.widget import Widget
from rich.text import Text

from art.config import PALETTE_TERM

UPPER_HALF = "▀"
SPARK = "▁▂▃▄▅▆▇█"

# Box-drawing for green frames
FRAME_TL = "┌"
FRAME_TR = "┐"
FRAME_BL = "└"
FRAME_BR = "┘"
FRAME_H = "─"
FRAME_V = "│"
FRAME_STYLE = "green"

# Heat color scale for neural activity visualization
HEAT_SCALE = [
    "#0a0a2e",  # deep dark
    "#1a1a6e",  # dark blue
    "#2c5faa",  # blue
    "#00a89e",  # teal
    "#4ec44e",  # green
    "#c8c800",  # yellow
    "#e06600",  # orange
    "#ff2222",  # red
]


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


def _render_heatmap(data: np.ndarray) -> list[Text]:
    rows, cols = data.shape
    min_v = float(data.min())
    max_v = float(data.max())
    range_v = max_v - min_v if max_v > min_v else 1.0
    norm = (data - min_v) / range_v

    lines = []
    for y in range(0, rows, 2):
        line = Text()
        for x in range(cols):
            top = float(norm[y, x])
            bot = float(norm[y + 1, x]) if y + 1 < rows else 0.0
            fg = HEAT_SCALE[min(7, int(top * 7.99))]
            bg = HEAT_SCALE[min(7, int(bot * 7.99))]
            line.append(UPPER_HALF, style=f"{fg} on {bg}")
        lines.append(line)
    return lines


def _render_framed(grid: np.ndarray, label: str = "", score: float | None = None,
                   heatmap: bool = False) -> list[Text]:
    cols = grid.shape[1]
    inner_lines = _render_heatmap(grid) if heatmap else _render_small(grid)
    result = []

    top = Text()
    top.append(FRAME_TL, style=FRAME_STYLE)
    if label:
        lbl = label[:cols]
        top.append(lbl, style="dim cyan")
        top.append(FRAME_H * (cols - len(lbl)), style=FRAME_STYLE)
    else:
        top.append(FRAME_H * cols, style=FRAME_STYLE)
    top.append(FRAME_TR, style=FRAME_STYLE)
    result.append(top)

    for line in inner_lines:
        row = Text()
        row.append(FRAME_V, style=FRAME_STYLE)
        row.append(line)
        row.append(FRAME_V, style=FRAME_STYLE)
        result.append(row)

    bot = Text()
    bot.append(FRAME_BL, style=FRAME_STYLE)
    if score is not None:
        score_str = f"{score:.3f}"
        color = "bright_green" if score >= 0.6 else "yellow" if score >= 0.4 else "red"
        bot.append(score_str, style=color)
        bot.append(FRAME_H * (cols - len(score_str)), style=FRAME_STYLE)
    else:
        bot.append(FRAME_H * cols, style=FRAME_STYLE)
    bot.append(FRAME_BR, style=FRAME_STYLE)
    result.append(bot)

    return result


def _render_framed_row(items: list[list[Text]], spacing: int = 1) -> list[Text]:
    if not items:
        return []
    n_lines = max(len(frame) for frame in items)
    result = []
    for line_idx in range(n_lines):
        combined = Text()
        for i, frame in enumerate(items):
            if i > 0:
                combined.append(" " * spacing)
            if line_idx < len(frame):
                combined.append(frame[line_idx])
            else:
                w = len(frame[0].plain) if frame else 18
                combined.append(" " * w)
        result.append(combined)
    return result


class GalleryGrid(Widget):
    DEFAULT_CSS = """
    GalleryGrid {
        width: 100%;
        height: 100%;
    }
    """

    def __init__(self, cols: int = 4, max_pieces: int = 12, **kwargs):
        super().__init__(**kwargs)
        self.cols = cols
        self.max_pieces = max_pieces
        self._pieces: list[tuple[np.ndarray, int, float]] = []
        self._selected_indices: set[int] = set()
        self._training_previews: list[np.ndarray] = []
        self._training_step: int = 0
        self._training_total: int = 0
        self._is_training: bool = False
        self._gen_grids: list[np.ndarray] = []
        self._gen_pixel: int = 0
        self._gen_total_pixels: int = 256
        self._is_generating: bool = False
        self._neural_layer_maps: list[np.ndarray] = []
        self._neural_embedding_sim: np.ndarray | None = None
        self._neural_weight_norms: list[float] = []
        self._neural_step: int = 0
        self._neural_total: int = 0
        self._has_neural: bool = False
        self._scoring_done: int = 0
        self._scoring_total: int = 0
        self._is_scoring: bool = False

    @property
    def _cw(self) -> int:
        """Content width (widget width minus border + padding)."""
        return max(20, self.size.width - 4)

    @property
    def _fit_cols(self) -> int:
        """How many 16px items fit side by side."""
        return max(1, (self._cw + 2) // 18)  # 16px art + 2 spacing

    def update_pieces(self, pieces: list[np.ndarray], scores: list[dict]):
        self._is_training = False
        self._is_generating = False
        self._is_scoring = False
        self._has_neural = False
        self._selected_indices = set()
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

    def mark_selected(self, indices: list[int]):
        self._selected_indices = set(indices)
        self.refresh()

    def update_training_preview(self, grids: list[np.ndarray], step: int, total_steps: int):
        self._training_previews = grids
        self._training_step = step
        self._training_total = total_steps
        self._is_training = True
        self._is_generating = False
        self.refresh()

    def update_generation_progress(self, grids: list[np.ndarray], pixel: int, total_pixels: int):
        self._gen_grids = grids
        self._gen_pixel = pixel
        self._gen_total_pixels = total_pixels
        self._is_generating = True
        self._is_training = False
        self._is_scoring = False
        self._has_neural = False
        self.refresh()

    def update_scoring_progress(self, done: int, total: int):
        self._scoring_done = done
        self._scoring_total = total
        self._is_scoring = True
        self._is_generating = False
        self._is_training = False
        self._has_neural = False
        self.refresh()

    def update_neural_activity(self, layer_maps, embedding_sim, weight_norms, step, total_steps):
        self._neural_layer_maps = layer_maps
        self._neural_embedding_sim = embedding_sim
        self._neural_weight_norms = weight_norms
        self._neural_step = step
        self._neural_total = total_steps
        self._has_neural = True
        self._is_generating = False
        self.refresh()

    def render(self) -> Text:
        if self._is_generating and self._gen_grids:
            return self._render_generating()
        if self._is_scoring:
            return self._render_scoring()
        if self._has_neural and self._neural_layer_maps:
            return self._render_neural()
        if self._is_training and self._training_previews:
            return self._render_training()
        if self._pieces:
            return self._render_gallery()
        # Default: show neural placeholder until data arrives
        result = Text()
        result.append("NEURAL ACTIVITY\n", style="bold magenta")
        result.append("─" * (self._cw) + "\n", style="dim")
        result.append("  Warming up...\n", style="dim italic")
        return result

    def _bar(self, pct: float, width: int | None = None) -> Text:
        w = width or (self._cw - 4)
        filled = int(pct / 100 * w)
        t = Text()
        for i in range(w):
            t.append("█" if i < filled else "░", style="green" if i < filled else "bright_black")
        return t

    def _render_neural(self) -> Text:
        result = Text()
        cw = self._cw
        pct = self._neural_step / max(1, self._neural_total) * 100
        result.append("NEURAL ACTIVITY", style="bold magenta")
        result.append(f"  step {self._neural_step}/{self._neural_total}", style="dim cyan")
        result.append(f"  {pct:.0f}%\n", style="bold cyan")

        maps = self._neural_layer_maps
        cols = self._fit_cols

        # Collect all framed items: all layers + embedding sim
        all_frames: list[tuple[list[Text], str]] = []
        for i, m in enumerate(maps):
            label = "L0 input" if i == 0 else (f"L{i} output" if i == len(maps) - 1 else f"L{i}")
            all_frames.append(_render_framed(m, label=label, heatmap=True))

        if self._neural_embedding_sim is not None:
            sim_clamped = np.clip(self._neural_embedding_sim, 0, 1)
            all_frames.append(_render_framed(sim_clamped, label="COLORS", heatmap=True))

        # Render in rows filling available width
        for row_start in range(0, len(all_frames), cols):
            row_items = all_frames[row_start : row_start + cols]
            for line in _render_framed_row(row_items):
                result.append(" ")
                result.append(line)
                result.append("\n")
            result.append("\n")

        # Layer energy — fill available width
        if self._neural_weight_norms:
            result.append("  Layer Energy ", style="dim")
            norms = self._neural_weight_norms
            min_n = min(norms)
            max_n = max(norms)
            range_n = max_n - min_n if max_n > min_n else 1.0
            bar_each = max(2, (cw - 16 - len(norms) * 4) // len(norms))
            for i, n in enumerate(norms):
                v = (n - min_n) / range_n
                idx = min(7, int(v * 7.99))
                color = HEAT_SCALE[idx]
                result.append(f" L{i}", style="dim")
                result.append(SPARK[idx] * bar_each, style=color)
            result.append("\n")

        # Show preview samples if available
        if self._training_previews:
            result.append("\n")
            previews = self._training_previews[:cols]
            frames = [_render_framed(g, label=f"sample {i+1}") for i, g in enumerate(previews)]
            for line in _render_framed_row(frames):
                result.append(" ")
                result.append(line)
                result.append("\n")

        return result

    def _render_piece_grid(self, grids: list[np.ndarray], cols: int) -> Text:
        """Render a grid of pixel art pieces without frames — raw pixels side by side."""
        result = Text()
        for row_start in range(0, len(grids), cols):
            row_grids = grids[row_start : row_start + cols]
            piece_rows = [_render_small(g) for g in row_grids]
            n_lines = max((len(r) for r in piece_rows), default=0)
            for line_idx in range(n_lines):
                result.append(" ")
                for i, lines in enumerate(piece_rows):
                    if i > 0:
                        result.append("  ")
                    if line_idx < len(lines):
                        result.append(lines[line_idx])
                result.append("\n")
            result.append("\n")
        return result

    def _render_scoring(self) -> Text:
        result = Text()
        cols = self._fit_cols
        done = self._scoring_done
        total = max(1, self._scoring_total)
        pct = done / total * 100

        result.append("ANALYZING", style="bold magenta")
        result.append(f"  {done}/{total} pieces", style="dim cyan")
        result.append(f"  {pct:.0f}%\n", style="bold yellow")

        result.append("  ")
        result.append(self._bar(pct))
        result.append("\n\n")

        if self._gen_grids:
            result.append(self._render_piece_grid(self._gen_grids, cols))

        metrics = ["symmetry", "complexity", "structure", "aesthetics", "diversity"]
        result.append("  evaluating: ", style="dim")
        for i, m in enumerate(metrics):
            if (done / total) > (i / len(metrics)):
                result.append(f"{m} ", style="bright_green")
            else:
                result.append(f"{m} ", style="bright_black")
        result.append("\n")

        return result

    def _render_generating(self) -> Text:
        result = Text()
        cols = self._fit_cols
        pct = self._gen_pixel / max(1, self._gen_total_pixels) * 100
        row = self._gen_pixel // 16
        result.append("GENERATING", style="bold magenta")
        result.append(f"  pixel {self._gen_pixel}/{self._gen_total_pixels}", style="dim cyan")
        result.append(f"  row {row}/16", style="dim")
        result.append(f"  {pct:.0f}%\n", style="bold cyan")

        result.append("  ")
        result.append(self._bar(pct))
        result.append("\n\n")

        result.append(self._render_piece_grid(self._gen_grids, cols))

        return result

    def _render_training(self) -> Text:
        result = Text()
        cols = self._fit_cols
        pct = self._training_step / max(1, self._training_total) * 100
        result.append("LEARNING", style="bold magenta")
        result.append(f"  step {self._training_step}/{self._training_total}", style="dim cyan")
        result.append(f"  {pct:.0f}%\n", style="bold cyan")

        result.append("  ")
        result.append(self._bar(pct))
        result.append("\n\n")

        result.append(self._render_piece_grid(self._training_previews[:cols * 2], cols))
        result.append("  the model is learning to draw...\n", style="dim italic")
        return result

    def _render_gallery(self) -> Text:
        result = Text()
        cols = self._fit_cols
        has_selection = bool(self._selected_indices)
        n_selected = len(self._selected_indices)
        n_total = len(self._pieces)

        if has_selection:
            result.append("GALLERY", style="bold magenta")
            result.append(f"  {n_selected}/{n_total} selected\n", style="bold bright_green")
        else:
            result.append("GALLERY", style="bold magenta")
            result.append(f"  {n_total} pieces\n", style="dim cyan")

        for row_start in range(0, len(self._pieces), cols):
            row_pieces = self._pieces[row_start : row_start + cols]
            indices = [idx for _, idx, _ in row_pieces]
            is_selected = [idx in self._selected_indices for idx in indices]
            piece_rows = [_render_small(g) for g, _, _ in row_pieces]
            scores = [score for _, _, score in row_pieces]
            art_width = row_pieces[0][0].shape[1] if row_pieces else 16
            n_lines = max((len(r) for r in piece_rows), default=0)

            # Top border for selected pieces
            if has_selection:
                result.append(" ")
                for i, sel in enumerate(is_selected):
                    if i > 0:
                        result.append("  ")
                    if sel:
                        result.append("+" + "-" * art_width + "+", style="bold bright_yellow")
                    else:
                        result.append(" " * (art_width + 2))
                result.append("\n")

            for line_idx in range(n_lines):
                result.append(" ")
                for i, lines in enumerate(piece_rows):
                    if i > 0:
                        result.append("  ")
                    sel = is_selected[i]
                    if has_selection and sel:
                        result.append("|", style="bold bright_yellow")
                    elif has_selection:
                        result.append(" ")
                    if line_idx < len(lines):
                        result.append(lines[line_idx])
                    if has_selection and sel:
                        result.append("|", style="bold bright_yellow")
                    elif has_selection:
                        result.append(" ")
                result.append("\n")

            # Bottom border for selected pieces
            if has_selection:
                result.append(" ")
                for i, sel in enumerate(is_selected):
                    if i > 0:
                        result.append("  ")
                    if sel:
                        result.append("+" + "-" * art_width + "+", style="bold bright_yellow")
                    else:
                        result.append(" " * (art_width + 2))
                result.append("\n")

            # Score + label row
            result.append(" ")
            for i, score in enumerate(scores):
                if i > 0:
                    result.append("  ")
                sel = is_selected[i]
                label_width = art_width + (2 if has_selection else 0)
                if sel:
                    tag = f"*{score:.3f}*"
                    result.append(tag.center(label_width), style="bold bright_yellow")
                else:
                    color = "bright_green" if score >= 0.6 else "yellow" if score >= 0.4 else "red"
                    if has_selection:
                        color = "dim"
                    result.append(f"{score:.3f}".center(label_width), style=color)
            result.append("\n\n")

        return result
