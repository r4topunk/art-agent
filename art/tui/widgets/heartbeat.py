from __future__ import annotations
from rich.text import Text
from textual.widget import Widget

SPARK = "▁▂▃▄▅▆▇█"


class HeartbeatWidget(Widget):
    DEFAULT_CSS = """
    HeartbeatWidget {
        width: 100%;
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(self, max_points: int = 60, **kwargs):
        super().__init__(**kwargs)
        self._grad_norms: list[float] = []
        self._max_points = max_points
        self._token_difficulties: list[float] = []  # per-position difficulty

    def add_grad_norm(self, norm: float):
        self._grad_norms.append(norm)
        if len(self._grad_norms) > self._max_points:
            self._grad_norms = self._grad_norms[-self._max_points:]
        self.refresh()

    def update_token_difficulty(self, difficulties: list[float]):
        self._token_difficulties = difficulties
        self.refresh()

    def render(self) -> Text:
        result = Text()
        result.append("♥ HEARTBEAT\n", style="bold red")

        if not self._grad_norms:
            result.append("  ─── flatline ───\n", style="dim")
            return result

        # Gradient norm waveform
        values = self._grad_norms
        min_v = min(values)
        max_v = max(values)
        range_v = max_v - min_v if max_v > min_v else 1.0

        result.append("  ")
        for v in values:
            normalized = (v - min_v) / range_v
            idx = min(7, max(0, int(normalized * 7.99)))
            # Color: calm=green, intense=red
            if normalized < 0.3:
                color = "green"
            elif normalized < 0.6:
                color = "yellow"
            else:
                color = "red"
            result.append(SPARK[idx], style=color)
        result.append("\n")

        result.append(f"  grad: {values[-1]:.4f}", style="dim")
        if len(values) > 1:
            trend = values[-1] - values[-2]
            arrow = "↑" if trend > 0 else "↓" if trend < 0 else "→"
            result.append(f" {arrow}", style="red" if trend > 0 else "green")
        result.append("\n")

        # Token difficulty heatmap (which pixel positions are hardest)
        if self._token_difficulties:
            result.append("\n  PIXEL DIFFICULTY MAP\n", style="bold")
            diffs = self._token_difficulties
            if len(diffs) >= 256:
                # Reshape to 16x16 and render as heatmap with half-blocks
                max_d = max(diffs[:256]) if max(diffs[:256]) > 0 else 1.0
                min_d = min(diffs[:256])
                range_d = max_d - min_d if max_d > min_d else 1.0

                result.append("  ")
                for y in range(0, 16, 2):
                    line = Text("  ")
                    for x in range(16):
                        top_d = (diffs[y * 16 + x] - min_d) / range_d
                        bot_d = (diffs[(y + 1) * 16 + x] - min_d) / range_d if y + 1 < 16 else 0
                        avg = (top_d + bot_d) / 2

                        if avg < 0.2:
                            color = "bright_green"  # easy
                        elif avg < 0.4:
                            color = "green"
                        elif avg < 0.6:
                            color = "yellow"
                        elif avg < 0.8:
                            color = "red"
                        else:
                            color = "bright_red"  # hardest
                        line.append("█", style=color)
                    result.append(line)
                    result.append("\n")
                result.append("  easy ", style="bright_green")
                result.append("██", style="green")
                result.append("██", style="yellow")
                result.append("██", style="red")
                result.append(" hard\n", style="bright_red")

        return result
