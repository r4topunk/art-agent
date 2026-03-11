import time
from textual.widget import Widget
from textual.reactive import reactive
from rich.text import Text


class HeaderWidget(Widget):
    DEFAULT_CSS = """
    HeaderWidget {
        dock: top;
        height: 3;
        background: #1a1a2e;
        padding: 0 2;
        content-align: center middle;
    }
    """

    generation = reactive(0)
    total_generations = reactive(0)
    temperature = reactive(1.0)
    phase = reactive("idle")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._start_time = time.time()

    def render(self) -> Text:
        elapsed = time.time() - self._start_time
        h = int(elapsed // 3600)
        m = int(elapsed % 3600 // 60)
        s = int(elapsed % 60)
        elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"

        result = Text()
        result.append("  ArtAgent  ", style="bold magenta")
        result.append("│ ", style="dim")
        result.append(f"Gen {self.generation}", style="bold cyan")
        if self.total_generations:
            result.append(f"/{self.total_generations}", style="dim cyan")
        result.append("  │ ", style="dim")
        result.append(f"Temp {self.temperature:.2f}", style="bold yellow")
        result.append("  │ ", style="dim")
        result.append(f"⏱ {elapsed_str}", style="bold green")
        result.append("  │ ", style="dim")
        result.append(f"{self.phase}", style="bold white")
        return result
