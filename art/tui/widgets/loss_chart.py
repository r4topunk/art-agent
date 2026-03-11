from textual.widget import Widget
from textual.reactive import reactive
from rich.text import Text

SPARK_CHARS = "▁▂▃▄▅▆▇█"


class LossChart(Widget):
    DEFAULT_CSS = """
    LossChart {
        height: 3;
        width: 100%;
    }
    """

    losses: reactive[list[float]] = reactive(list, init=False)

    def __init__(self, max_points: int = 60, **kwargs):
        super().__init__(**kwargs)
        self._losses: list[float] = []
        self.max_points = max_points

    def add_loss(self, loss: float) -> None:
        self._losses.append(loss)
        if len(self._losses) > self.max_points:
            self._losses = self._losses[-self.max_points:]
        self.refresh()

    def clear(self) -> None:
        self._losses.clear()
        self.refresh()

    def render(self) -> Text:
        if not self._losses:
            return Text("No data yet", style="dim")

        values = self._losses
        min_v = min(values)
        max_v = max(values)
        range_v = max_v - min_v if max_v > min_v else 1.0

        # Invert: high loss = tall bar (we want loss going DOWN to look like descent)
        inverted = ""
        for v in values:
            idx = int((1 - (v - min_v) / range_v) * (len(SPARK_CHARS) - 1))
            idx = max(0, min(len(SPARK_CHARS) - 1, idx))
            inverted += SPARK_CHARS[idx]

        result = Text()
        result.append(Text(f"Loss: {values[-1]:.4f}", style="bold"))
        result.append("\n")
        result.append(Text(inverted, style="green"))
        result.append("\n")
        result.append(Text(f"↓{min_v:.4f}  ↑{max_v:.4f}", style="dim"))
        return result
