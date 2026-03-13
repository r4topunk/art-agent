from textual.widget import Widget
from rich.text import Text

SPARK_CHARS = "▁▂▃▄▅▆▇█"
BAR_FULL = "█"
BAR_EMPTY = "░"


def score_bar(value: float, width: int = 10) -> Text:
    filled = int(value * width)
    bar = BAR_FULL * filled + BAR_EMPTY * (width - filled)
    color = "green" if value >= 0.6 else "yellow" if value >= 0.4 else "red"
    result = Text()
    result.append(bar, style=color)
    return result


def mini_sparkline(values: list[float], width: int = 30) -> Text:
    if not values:
        return Text("─" * width, style="dim")

    if len(values) > width:
        step = len(values) / width
        resampled = [values[int(i * step)] for i in range(width)]
    else:
        resampled = values

    min_v = min(resampled)
    max_v = max(resampled)
    range_v = max_v - min_v if max_v > min_v else 1.0

    result = Text()
    for v in resampled:
        idx = int((v - min_v) / range_v * (len(SPARK_CHARS) - 1))
        idx = max(0, min(len(SPARK_CHARS) - 1, idx))
        result.append(SPARK_CHARS[idx], style="cyan")
    return result


class EvolutionPanel(Widget):
    DEFAULT_CSS = """
    EvolutionPanel {
        width: 100%;
        height: 100%;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mean_scores: list[float] = []
        self._max_scores: list[float] = []
        self._latest_summary: dict | None = None
        self._latest_scores: list[dict] | None = None

    @property
    def _cw(self) -> int:
        return max(16, self.size.width - 4)

    def update_generation(self, summary: dict) -> None:
        self._latest_summary = summary
        self._mean_scores.append(summary.get("mean_score", 0))
        self._max_scores.append(summary.get("max_score", 0))
        self.refresh()

    def update_scores(self, scores: list[dict]) -> None:
        self._latest_scores = scores
        self.refresh()

    def render(self) -> Text:
        cw = self._cw
        spark_w = cw - 2
        bar_w = max(6, cw - 14)

        result = Text()
        result.append("EVOLUTION\n", style="bold magenta")
        result.append("─" * cw + "\n", style="dim")

        result.append("Mean Score Trend\n", style="bold")
        result.append(mini_sparkline(self._mean_scores, width=spark_w))
        result.append("\n")

        result.append("Max Score Trend\n", style="bold")
        result.append(mini_sparkline(self._max_scores, width=spark_w))
        result.append("\n\n")

        if self._latest_summary:
            s = self._latest_summary
            result.append(f"  Mean  {s.get('mean_score', 0):.3f}\n", style="white")
            result.append(f"  Max   {s.get('max_score', 0):.3f}\n", style="green")
            result.append(f"  Min   {s.get('min_score', 0):.3f}\n", style="red")
            result.append(f"  Temp  {s.get('temperature', 0):.3f}\n", style="yellow")
            n_sel = s.get("n_selected", 0)
            n_tot = s.get("n_pieces", 0)
            result.append(f"  Sel   {n_sel}/{n_tot}\n", style="cyan")
            result.append("\n")

        if self._latest_scores:
            result.append("SCORE BREAKDOWN (avg)\n", style="bold")
            result.append("─" * cw + "\n", style="dim")

            keys = ["symmetry", "complexity", "structure", "aesthetics", "diversity"]
            for key in keys:
                vals = [s.get(key, 0) for s in self._latest_scores]
                avg = sum(vals) / len(vals) if vals else 0
                label = key[:4].capitalize()
                result.append(f"  {label:<5} {avg:.2f} ", style="white")
                result.append(score_bar(avg, width=bar_w))
                result.append("\n")

        return result


class TrainingPanel(Widget):
    DEFAULT_CSS = """
    TrainingPanel {
        width: 100%;
        height: 100%;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._step = 0
        self._total_steps = 0
        self._loss = 0.0
        self._lr = 0.0
        self._phase = ""
        self._losses: list[float] = []
        self._max_display = 80

    @property
    def _cw(self) -> int:
        return max(16, self.size.width - 4)

    def update_step(self, step: int, loss: float, lr: float) -> None:
        self._step = step
        self._loss = loss
        self._lr = lr
        self._losses.append(loss)
        if len(self._losses) > self._max_display:
            self._losses = self._losses[-self._max_display:]
        self.refresh()

    def update_phase(self, phase: str, total_steps: int) -> None:
        self._phase = phase
        self._total_steps = total_steps
        self._step = 0
        self._losses.clear()
        self.refresh()

    def render(self) -> Text:
        cw = self._cw
        bar_w = cw - 4

        result = Text()
        result.append("TRAINING\n", style="bold magenta")
        result.append("─" * cw + "\n", style="dim")

        result.append("  Phase: ", style="dim")
        result.append(f"{self._phase}\n", style="bold white")

        result.append("  Step:  ", style="dim")
        result.append(f"{self._step}/{self._total_steps}\n", style="bold cyan")

        if self._total_steps > 0:
            pct = self._step / self._total_steps
            filled = int(pct * bar_w)
            bar = BAR_FULL * filled + BAR_EMPTY * (bar_w - filled)
            result.append(f"  {bar}\n", style="green")

        result.append("  Loss:  ", style="dim")
        result.append(f"{self._loss:.4f}\n", style="bold yellow")

        result.append("  LR:    ", style="dim")
        result.append(f"{self._lr:.6f}\n", style="white")
        result.append("\n")

        if self._losses:
            result.append("  Loss curve:\n", style="dim")
            result.append("  ")

            spark_w = cw - 4
            values = self._losses
            if len(values) > spark_w:
                step = len(values) / spark_w
                values = [values[int(i * step)] for i in range(spark_w)]

            min_v = min(values)
            max_v = max(values)
            range_v = max_v - min_v if max_v > min_v else 1.0

            for v in values:
                idx = int((1 - (v - min_v) / range_v) * 7)
                idx = max(0, min(7, idx))
                result.append(SPARK_CHARS[idx], style="green")
            result.append("\n")

        return result
