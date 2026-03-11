from __future__ import annotations
import math
from rich.text import Text
from textual.widget import Widget

SPARK = "▁▂▃▄▅▆▇█"


class EntropyMeter(Widget):
    DEFAULT_CSS = """
    EntropyMeter {
        width: 100%;
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._entropies: list[float] = []  # per-generation average entropy
        self._current_confidences: list[float] = []  # current gen pixel confidences

    def update_from_confidences(self, confidences: list[float] | None = None):
        """Compute entropy from confidence values of latest generation.
        For binary-ish distributions, entropy ~ -conf*log2(conf) - (1-conf)*log2(1-conf)
        """
        if confidences is None or len(confidences) == 0:
            return
        self._current_confidences = confidences

        # Approximate entropy from confidence
        total_entropy = 0.0
        for c in confidences:
            c = max(0.001, min(0.999, c))
            # Shannon entropy of Bernoulli-like: higher when c close to 0.5
            h = -c * math.log2(c)
            total_entropy += h
        avg_entropy = total_entropy / len(confidences)
        self._entropies.append(avg_entropy)
        self.refresh()

    def render(self) -> Text:
        result = Text()
        result.append("✧ ARTIST'S MIND\n", style="bold cyan")

        if not self._entropies:
            result.append("  Waiting...\n", style="dim")
            return result

        # Current creativity level
        current = self._entropies[-1]
        max_entropy = math.log2(5)  # max for 5-token vocab
        creativity = min(1.0, current / max_entropy)

        # Creativity gauge
        gauge_width = 20
        filled = int(creativity * gauge_width)
        result.append("  ")
        result.append("█" * filled, style="magenta")
        result.append("░" * (gauge_width - filled), style="bright_black")
        result.append(f" {creativity:.0%}\n")

        result.append("  focused ", style="dim cyan")
        result.append("─" * 10, style="dim")
        result.append(" exploring\n", style="dim magenta")

        # Entropy trend over generations
        if len(self._entropies) > 1:
            result.append("  Trend: ", style="dim")
            values = self._entropies[-40:]
            min_v = min(values)
            max_v = max(values)
            range_v = max_v - min_v if max_v > min_v else 1.0
            for v in values:
                idx = min(7, max(0, int((v - min_v) / range_v * 7.99)))
                result.append(SPARK[idx], style="magenta")
            result.append("\n")

        # Per-pixel confidence distribution of current generation
        if self._current_confidences:
            # Histogram of confidences
            bins = [0, 0, 0, 0, 0]  # 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
            for c in self._current_confidences:
                b = min(4, int(c * 5))
                bins[b] += 1
            total = sum(bins) or 1
            result.append("  Confidence dist:\n", style="dim")
            labels = ["sure ", "conf ", "mixed", "unsur", "guess"]
            colors = ["bright_green", "green", "yellow", "red", "bright_red"]
            for i in range(4, -1, -1):
                pct = bins[i] / total
                bar_w = int(pct * 20)
                result.append(f"  {labels[i]} ", style="dim")
                result.append("█" * bar_w, style=colors[i])
                result.append(f" {pct:.0%}\n")

        return result
