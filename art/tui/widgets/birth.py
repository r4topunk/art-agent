from __future__ import annotations
import numpy as np
from rich.text import Text
from textual.widget import Widget

from art.config import PALETTE_TERM

UPPER_HALF = "▀"
DOT = "·"
SPARK = "▁▂▃▄▅▆▇█"
BAR_FULL = "█"
BAR_EMPTY = "░"


class BirthWidget(Widget):
    DEFAULT_CSS = """
    BirthWidget {
        width: 100%;
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._grid: np.ndarray | None = None
        self._confidences: np.ndarray | None = None
        self._filled: int = 0
        self._grid_size: int = 16
        self._vlm_description: str | None = None
        self._vlm_scores: dict | None = None
        self._vlm_history: list[tuple[int, str]] = []
        self._max_history: int = 5
        # Training state
        self._training_losses: list[float] = []
        self._training_step: int = 0
        self._training_total: int = 0
        self._training_lr: float = 0.0
        self._training_preview: np.ndarray | None = None
        self._is_training: bool = False

    def update_training(self, step: int, total_steps: int, loss: float, lr: float):
        self._is_training = True
        self._training_step = step
        self._training_total = total_steps
        self._training_lr = lr
        self._training_losses.append(loss)
        if len(self._training_losses) > 200:
            self._training_losses = self._training_losses[-200:]
        self.refresh()

    def update_training_preview(self, grid: np.ndarray):
        self._training_preview = grid
        self.refresh()

    def end_training(self):
        self._is_training = False
        self.refresh()

    def update_birth(
        self,
        grid: np.ndarray,
        confidences: np.ndarray,
        piece_index: int = 0,
        vlm_description: str | None = None,
        vlm_scores: dict | None = None,
        generation: int = 0,
    ):
        self._grid = grid
        self._confidences = confidences
        self._vlm_description = vlm_description
        self._vlm_scores = vlm_scores
        self._filled = self._grid_size * self._grid_size
        if vlm_description:
            self._vlm_history.append((generation, vlm_description))
            if len(self._vlm_history) > self._max_history:
                self._vlm_history = self._vlm_history[-self._max_history:]
        self.refresh()

    @property
    def _cw(self) -> int:
        return max(20, self.size.width - 4)

    def _conf_color(self, conf: float) -> str:
        if conf >= 0.8:
            return "bright_green"
        elif conf >= 0.6:
            return "green"
        elif conf >= 0.4:
            return "yellow"
        elif conf >= 0.2:
            return "red"
        return "bright_red"

    def _score_bar(self, value: float, width: int = 12) -> Text:
        filled = int(value * width)
        bar = Text()
        color = "bright_cyan" if value >= 0.5 else "cyan" if value >= 0.3 else "bright_black"
        bar.append(BAR_FULL * filled, style=color)
        bar.append(BAR_EMPTY * (width - filled), style="bright_black")
        return bar

    def _render_loss_landscape(self, width: int = 40, height: int = 8) -> list[Text]:
        losses = self._training_losses
        if not losses:
            return []

        # Bucket into width columns
        if len(losses) > width:
            step = len(losses) / width
            buckets = [losses[int(i * step)] for i in range(width)]
        else:
            buckets = list(losses)
            # Pad left with empty
            buckets = [None] * (width - len(buckets)) + buckets

        real_vals = [v for v in buckets if v is not None]
        if not real_vals:
            return []
        min_l = min(real_vals)
        max_l = max(real_vals)
        range_l = max_l - min_l if max_l > min_l else 1.0

        lines = []
        for row in range(height):
            line = Text("  ")
            threshold = 1.0 - (row + 1) / height
            for val in buckets:
                if val is None:
                    line.append(" ")
                    continue
                normalized = (val - min_l) / range_l
                if normalized >= threshold:
                    if normalized > 0.7:
                        color = "bright_red"
                    elif normalized > 0.4:
                        color = "yellow"
                    elif normalized > 0.15:
                        color = "green"
                    else:
                        color = "bright_green"
                    line.append("█", style=color)
                else:
                    line.append(" ")
            lines.append(line)
        return lines

    def _render_training(self) -> Text:
        result = Text()
        result.append("◉ SPECIMEN", style="bold magenta")
        result.append(" — Training\n", style="dim magenta")

        pct = self._training_step / max(1, self._training_total) * 100

        # Loss landscape chart — fill available width
        cw = self._cw
        result.append("\n  LOSS LANDSCAPE\n", style="bold cyan")
        landscape = self._render_loss_landscape(width=cw - 4, height=8)
        for line in landscape:
            result.append(line)
            result.append("\n")

        if self._training_losses:
            latest = self._training_losses[-1]
            first = self._training_losses[0]
            result.append(f"  {first:.2f}", style="red")
            result.append(" ─── ", style="dim")
            result.append(f"{latest:.4f}", style="bright_green")
            result.append(f"  lr {self._training_lr:.6f}\n", style="dim")

        # Show best training preview alongside stats
        result.append("\n")
        if self._training_preview is not None:
            result.append("  PREVIEW                 STATS\n", style="dim")
            size = self._grid_size
            preview_lines = []
            for y in range(0, size, 2):
                line = Text()
                for x in range(size):
                    top = int(self._training_preview[y, x])
                    bot = int(self._training_preview[y + 1, x]) if y + 1 < size else 0
                    fg = PALETTE_TERM[top]
                    bg = PALETTE_TERM[bot]
                    line.append(UPPER_HALF, style=f"{fg} on {bg}")
                preview_lines.append(line)

            stat_lines = [
                (f"  Step {self._training_step}/{self._training_total}", "cyan"),
                (f"  Loss {self._training_losses[-1]:.4f}" if self._training_losses else "", "yellow"),
                (f"  LR   {self._training_lr:.6f}", "dim"),
                ("", ""),
                (f"  Progress {pct:.0f}%", "green"),
            ]

            for i in range(max(len(preview_lines), len(stat_lines))):
                result.append("  ")
                if i < len(preview_lines):
                    result.append(preview_lines[i])
                else:
                    result.append(" " * size)
                result.append("    ")
                if i < len(stat_lines) and stat_lines[i][0]:
                    result.append(stat_lines[i][0], style=stat_lines[i][1])
                result.append("\n")
        else:
            result.append(f"  Step {self._training_step}/{self._training_total}", style="cyan")
            result.append(f"  {pct:.0f}%\n", style="green")
            result.append("  Preview at step 500...\n", style="dim italic")

        return result

    def render(self) -> Text:
        if self._is_training:
            return self._render_training()

        result = Text()
        has_vlm = self._vlm_description is not None

        if has_vlm:
            result.append("◉ SPECIMEN + VLM\n", style="bold magenta")
        else:
            result.append("◉ SPECIMEN\n", style="bold magenta")

        if self._grid is None:
            result.append("  Awaiting generation...\n", style="dim")
            return result

        size = self._grid_size

        piece_lines = []
        conf_lines = []

        for y in range(0, size, 2):
            piece_line = Text()
            conf_line = Text()
            for x in range(size):
                top_idx = y * size + x
                bot_idx = (y + 1) * size + x

                top_filled = top_idx < self._filled
                bot_filled = bot_idx < self._filled and y + 1 < size

                if not top_filled and not bot_filled:
                    piece_line.append(DOT, style="bright_black")
                else:
                    top_val = int(self._grid[y, x]) if top_filled else 0
                    bot_val = int(self._grid[y + 1, x]) if bot_filled and y + 1 < size else 0
                    fg = PALETTE_TERM[top_val]
                    bg = PALETTE_TERM[bot_val]
                    piece_line.append(UPPER_HALF, style=f"{fg} on {bg}")

                if self._confidences is not None:
                    top_conf = self._confidences[top_idx] if top_idx < len(self._confidences) and top_filled else 0
                    bot_conf = self._confidences[bot_idx] if bot_idx < len(self._confidences) and bot_filled else 0

                    if not top_filled and not bot_filled:
                        conf_line.append(DOT, style="bright_black")
                    else:
                        avg_conf = (top_conf + bot_conf) / 2
                        conf_line.append(UPPER_HALF, style=self._conf_color(avg_conf))

            piece_lines.append(piece_line)
            conf_lines.append(conf_line)

        if has_vlm:
            vlm_lines = self._build_vlm_lines()
            max_lines = max(len(piece_lines), len(vlm_lines))

            result.append("  PIECE            CONF             VLM PERCEPTION\n", style="dim")
            for i in range(max_lines):
                result.append("  ")
                if i < len(piece_lines):
                    result.append(piece_lines[i])
                else:
                    result.append(" " * size)
                result.append("  ")
                if i < len(conf_lines):
                    result.append(conf_lines[i])
                else:
                    result.append(" " * size)
                result.append("  ")
                if i < len(vlm_lines):
                    result.append(vlm_lines[i])
                result.append("\n")
        else:
            result.append("  PIECE            CONFIDENCE\n", style="dim")
            for pl, cl in zip(piece_lines, conf_lines):
                result.append("  ")
                result.append(pl)
                result.append("    ")
                result.append(cl)
                result.append("\n")

        pct = self._filled / (size * size)
        filled_bar = int(pct * 20)
        result.append(f"\n  pixel {self._filled}/{size * size}  ", style="dim")
        result.append(BAR_FULL * filled_bar, style="cyan")
        result.append(BAR_EMPTY * (20 - filled_bar), style="bright_black")

        if self._confidences is not None and self._filled > 0:
            recent = self._confidences[max(0, self._filled - 30):self._filled]
            if len(recent) > 0:
                result.append("  conf: ", style="dim")
                for c in recent:
                    idx = min(7, max(0, int(c * 7.99)))
                    result.append(SPARK[idx], style=self._conf_color(c))
        result.append("\n")

        if has_vlm and len(self._vlm_history) > 1:
            result.append("\n  VLM HISTORY\n", style="bold dim")
            for gen, desc in self._vlm_history[-4:-1]:
                short = desc[:50] + "..." if len(desc) > 50 else desc
                result.append(f"  G{gen:<3d} ", style="dim cyan")
                result.append(f"{short}\n", style="dim")

        return result

    def _build_vlm_lines(self) -> list[Text]:
        lines: list[Text] = []
        if not self._vlm_description:
            return lines

        desc = self._vlm_description.strip()
        # piece(16) + 2 gaps + conf(16) + 2 gaps = 36 chars used; rest is VLM column
        wrap_width = max(28, self._cw - 40)
        words = desc.split()
        current_line = ""
        desc_lines = []
        for word in words:
            if len(current_line) + len(word) + 1 <= wrap_width:
                current_line = f"{current_line} {word}" if current_line else word
            else:
                if current_line:
                    desc_lines.append(current_line)
                current_line = word
        if current_line:
            desc_lines.append(current_line)

        header = Text()
        header.append("VLM says:", style="bold bright_cyan")
        lines.append(header)

        for dl in desc_lines[:4]:
            line = Text()
            line.append(dl, style="white")
            lines.append(line)
        if len(desc_lines) > 4:
            lines.append(Text("...", style="dim"))

        lines.append(Text(""))

        if self._vlm_scores:
            bar_w = max(10, wrap_width - 16)
            for key, label in [("vlm_interest", "Interest "), ("vlm_composition", "Composit."), ("vlm_creativity", "Creative "), ("vlm_composite", "VLM Total")]:
                val = self._vlm_scores.get(key, 0)
                line = Text()
                line.append(f"{label} ", style="dim")
                line.append(f"{val:.2f} ", style="bright_cyan" if val >= 0.4 else "dim")
                line.append(self._score_bar(val, width=bar_w))
                lines.append(line)

        return lines
