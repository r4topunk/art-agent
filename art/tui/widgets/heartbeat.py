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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._grad_norms: list[float] = []
        self._max_points = 200
        self._token_difficulties: list[float] = []

    @property
    def _cw(self) -> int:
        return max(16, self.size.width - 4)

    def add_grad_norm(self, norm: float):
        self._grad_norms.append(norm)
        if len(self._grad_norms) > self._max_points:
            self._grad_norms = self._grad_norms[-self._max_points:]
        self.refresh()

    def update_token_difficulty(self, difficulties: list[float]):
        self._token_difficulties = difficulties
        self.refresh()

    def render(self) -> Text:
        cw = self._cw
        result = Text()
        result.append("♥ HEARTBEAT\n", style="bold red")

        if not self._grad_norms:
            result.append("  " + "─" * (cw - 4) + "\n", style="dim")
            result.append("  flatline\n", style="dim")
            return result

        # Sparkline uses full available width
        spark_w = cw - 4
        values = self._grad_norms
        if len(values) > spark_w:
            step = len(values) / spark_w
            values = [values[int(i * step)] for i in range(spark_w)]

        clean = [v for v in values if v == v]  # filter NaN
        if not clean:
            return result
        min_v = min(clean)
        max_v = max(clean)
        range_v = max_v - min_v if max_v > min_v else 1.0

        result.append("  ")
        for v in values:
            if v != v:  # NaN check
                continue
            normalized = (v - min_v) / range_v
            idx = min(7, max(0, int(normalized * 7.99)))
            if normalized < 0.3:
                color = "green"
            elif normalized < 0.6:
                color = "yellow"
            else:
                color = "red"
            result.append(SPARK[idx], style=color)
        result.append("\n")

        result.append(f"  grad: {self._grad_norms[-1]:.4f}", style="dim")
        if len(self._grad_norms) > 1:
            trend = self._grad_norms[-1] - self._grad_norms[-2]
            arrow = "↑" if trend > 0 else "↓" if trend < 0 else "→"
            result.append(f" {arrow}", style="red" if trend > 0 else "green")
        result.append("\n")

        if self._token_difficulties:
            result.append("\n  PIXEL DIFFICULTY MAP\n", style="bold")
            diffs = self._token_difficulties
            if len(diffs) >= 256:
                max_d = max(diffs[:256]) if max(diffs[:256]) > 0 else 1.0
                min_d = min(diffs[:256])
                range_d = max_d - min_d if max_d > min_d else 1.0

                for y in range(0, 16, 2):
                    line = Text("  ")
                    for x in range(16):
                        top_d = (diffs[y * 16 + x] - min_d) / range_d
                        bot_d = (diffs[(y + 1) * 16 + x] - min_d) / range_d if y + 1 < 16 else 0
                        avg = (top_d + bot_d) / 2

                        if avg < 0.2:
                            color = "bright_green"
                        elif avg < 0.4:
                            color = "green"
                        elif avg < 0.6:
                            color = "yellow"
                        elif avg < 0.8:
                            color = "red"
                        else:
                            color = "bright_red"
                        line.append("█", style=color)
                    result.append(line)
                    result.append("\n")
                result.append("  easy ", style="bright_green")
                result.append("██", style="green")
                result.append("██", style="yellow")
                result.append("██", style="red")
                result.append(" hard\n", style="bright_red")

        return result
