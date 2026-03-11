from __future__ import annotations

import numpy as np
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Static
from textual import work

from art.config import ArtConfig
from art.events import EventBus
from art.runner import OvernightRunner
from art.utils import ensure_dirs
from art.tui.styles import APP_CSS
from art.tui.widgets.header import HeaderWidget
from art.tui.widgets.dashboard import TrainingPanel, EvolutionPanel
from art.tui.widgets.gallery import GalleryGrid
from art.tui.widgets.birth import BirthWidget
from art.tui.widgets.heartbeat import HeartbeatWidget
from art.tui.widgets.timeline import TimelineWidget
from art.tui.widgets.review import ReviewGrid, DetailPanel


class StatusBar(Static):
    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: #1a1a2e;
        padding: 0 2;
    }
    """

    def __init__(self):
        super().__init__("")
        self._status = "Initializing..."
        self._keys = "[D]ash [R]eview [Q]uit"

    def update_status(self, text: str):
        self._status = text
        self.update(f" {self._status}  │  {self._keys}")

    def on_mount(self):
        self.update(f" {self._status}  │  {self._keys}")


class DashboardScreen(Screen):
    def compose(self) -> ComposeResult:
        yield HeaderWidget(id="header")
        with Vertical(id="main-grid"):
            with Horizontal(id="top-row"):
                yield TrainingPanel(id="training-panel")
                yield GalleryGrid(cols=4, max_pieces=8, id="gallery-panel")
                yield EvolutionPanel(id="evolution-panel")
            with Horizontal(id="bottom-row"):
                yield HeartbeatWidget(id="heartbeat")
                yield BirthWidget(id="birth")
                yield TimelineWidget(id="timeline")
        yield StatusBar()

    def on_mount(self):
        self.set_interval(1.0, self._refresh_header)

    def _refresh_header(self):
        try:
            self.query_one(HeaderWidget).refresh()
        except Exception:
            pass


class ReviewScreen(Screen):
    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("left", "move_left", "Left"),
        ("right", "move_right", "Right"),
        ("up", "move_up", "Up"),
        ("down", "move_down", "Down"),
        ("space", "toggle_fav", "Favorite"),
    ]

    def compose(self) -> ComposeResult:
        yield ReviewGrid(cols=8, id="review-grid")
        yield DetailPanel(id="review-detail")

    def action_move_left(self):
        g = self.query_one("#review-grid", ReviewGrid)
        g.move_cursor(-1)
        self._sync()

    def action_move_right(self):
        g = self.query_one("#review-grid", ReviewGrid)
        g.move_cursor(1)
        self._sync()

    def action_move_up(self):
        g = self.query_one("#review-grid", ReviewGrid)
        g.move_cursor(-g.cols)
        self._sync()

    def action_move_down(self):
        g = self.query_one("#review-grid", ReviewGrid)
        g.move_cursor(g.cols)
        self._sync()

    def action_toggle_fav(self):
        g = self.query_one("#review-grid", ReviewGrid)
        g.toggle_favorite()
        self._sync()

    def _sync(self):
        g = self.query_one("#review-grid", ReviewGrid)
        d = self.query_one("#review-detail", DetailPanel)
        cur = g.get_current()
        if cur:
            grid, idx, scores = cur
            d.update_detail(grid, idx, scores, idx in g._favorites)


class ArtApp(App):
    CSS = APP_CSS
    TITLE = "ArtAgent"
    BINDINGS = [
        ("d", "switch_dashboard", "Dashboard"),
        ("r", "switch_review", "Review"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        generations: int = 50,
        resume: bool = True,
        use_vlm: bool = False,
        vlm_model: str = "moondream",
    ):
        super().__init__()
        self.generations = generations
        self.do_resume = resume
        self.use_vlm = use_vlm
        self.vlm_model = vlm_model
        self.event_bus = EventBus()
        self.config = ArtConfig()
        ensure_dirs(self.config)
        self._latest_pieces: list[np.ndarray] = []
        self._latest_scores: list[dict] = []
        self._latest_selected_indices: list[int] = []
        self._latest_confidences: np.ndarray | None = None

    def on_mount(self):
        self.push_screen(DashboardScreen())
        self._wire_events()
        self._run_evolution()

    def _wire_events(self):
        bus = self.event_bus
        bus.on("train_start", self._on_train_start)
        bus.on("train_step", self._on_train_step)
        bus.on("train_end", self._on_train_end)
        bus.on("gen_start", self._on_gen_start)
        bus.on("gen_scored", self._on_gen_scored)
        bus.on("gen_selected", self._on_gen_selected)
        bus.on("gen_complete", self._on_gen_complete)
        bus.on("gen_confidences", self._on_gen_confidences)
        bus.on("evolution_step", self._on_evolution_step)
        bus.on("token_difficulty", self._on_token_difficulty)

    # --- Train events ---

    def _on_train_start(self, total_steps: int, phase: str):
        self.call_from_thread(self._u_train_start, total_steps, phase)

    def _u_train_start(self, total_steps: int, phase: str):
        try:
            s = self.screen
            s.query_one("#training-panel", TrainingPanel).update_phase(phase, total_steps)
            s.query_one(HeaderWidget).phase = f"Training ({phase})"
            s.query_one(StatusBar).update_status(f"Training {phase}... 0/{total_steps}")
        except Exception:
            pass

    def _on_train_step(self, step: int, loss: float, lr: float, grad_norm: float = 0.0):
        if step % 5 == 0:
            self.call_from_thread(self._u_train_step, step, loss, lr, grad_norm)

    def _u_train_step(self, step: int, loss: float, lr: float, grad_norm: float):
        try:
            s = self.screen
            s.query_one("#training-panel", TrainingPanel).update_step(step, loss, lr)
            s.query_one("#heartbeat", HeartbeatWidget).add_grad_norm(grad_norm)
        except Exception:
            pass

    def _on_train_end(self, losses: list[float]):
        self.call_from_thread(self._u_train_end)

    def _u_train_end(self):
        try:
            self.screen.query_one(StatusBar).update_status("Training complete")
        except Exception:
            pass

    def _on_token_difficulty(self, difficulties: list[float]):
        self.call_from_thread(self._u_token_difficulty, difficulties)

    def _u_token_difficulty(self, difficulties: list[float]):
        try:
            self.screen.query_one("#heartbeat", HeartbeatWidget).update_token_difficulty(difficulties)
        except Exception:
            pass

    # --- Generation events ---

    def _on_gen_start(self, generation: int, temperature: float):
        self.call_from_thread(self._u_gen_start, generation, temperature)

    def _u_gen_start(self, generation: int, temperature: float):
        try:
            s = self.screen
            h = s.query_one(HeaderWidget)
            h.generation = generation
            h.temperature = temperature
            h.phase = "Generating..."
            s.query_one(StatusBar).update_status(f"Gen {generation} — Generating pieces...")
        except Exception:
            pass

    def _on_gen_confidences(self, confidences):
        self.call_from_thread(self._u_gen_confidences, confidences)

    def _u_gen_confidences(self, confidences):
        try:
            self._latest_confidences = confidences
        except Exception:
            pass

    def _on_gen_scored(self, pieces: list, scores: list[dict]):
        self._latest_pieces = pieces
        self._latest_scores = scores
        self.call_from_thread(self._u_gen_scored, pieces, scores)

    def _u_gen_scored(self, pieces: list, scores: list):
        try:
            s = self.screen
            s.query_one("#gallery-panel", GalleryGrid).update_pieces(pieces, scores)
            s.query_one("#evolution-panel", EvolutionPanel).update_scores(scores)

            # Birth: show best piece with its confidence + VLM description
            ranked = sorted(range(len(scores)), key=lambda i: scores[i].get("composite", 0), reverse=True)
            if ranked:
                best_idx = ranked[0]
                conf = np.zeros(256)
                if self._latest_confidences is not None and best_idx < self._latest_confidences.shape[0]:
                    conf = self._latest_confidences[best_idx, 1:257]
                vlm_desc = scores[best_idx].get("vlm_description")
                s.query_one("#birth", BirthWidget).update_birth(
                    pieces[best_idx], conf, best_idx, vlm_description=vlm_desc,
                )

            s.query_one(HeaderWidget).phase = "Selecting & Finetuning..."
            s.query_one(StatusBar).update_status("Scored — selecting & finetuning...")
        except Exception:
            pass

    def _on_gen_selected(self, selected: list, indices: list[int]):
        self._latest_selected_indices = indices
        self.call_from_thread(self._u_gen_selected, indices)

    def _u_gen_selected(self, indices: list[int]):
        pass

    def _on_gen_complete(self, summary: dict):
        self.call_from_thread(self._u_gen_complete, summary)

    def _u_gen_complete(self, summary: dict):
        try:
            s = self.screen
            gen = summary.get("generation", 0)
            self._current_gen = gen

            s.query_one("#evolution-panel", EvolutionPanel).update_generation(summary)
            s.query_one(HeaderWidget).phase = "Generation complete"
            s.query_one(StatusBar).update_status(
                f"Gen {gen} complete — mean: {summary.get('mean_score', 0):.3f}"
            )

            # Timeline: add best piece
            if self._latest_pieces and self._latest_scores:
                ranked = sorted(
                    range(len(self._latest_scores)),
                    key=lambda i: self._latest_scores[i].get("composite", 0),
                    reverse=True,
                )
                if ranked:
                    best_idx = ranked[0]
                    best_score = self._latest_scores[best_idx].get("composite", 0)
                    s.query_one("#timeline", TimelineWidget).add_generation(
                        gen, self._latest_pieces[best_idx], best_score
                    )
        except Exception:
            pass

    def _on_evolution_step(self, summary: dict, log: list[dict]):
        self.call_from_thread(self._u_evolution_step)

    def _u_evolution_step(self):
        try:
            self.screen.query_one(HeaderWidget).total_generations = self.generations
        except Exception:
            pass

    # --- Worker ---

    @work(thread=True)
    def _run_evolution(self):
        runner = OvernightRunner(
            self.config,
            event_bus=self.event_bus,
            use_vlm=self.use_vlm,
            vlm_model=self.vlm_model,
        )
        if self.do_resume:
            if not runner.resume():
                runner.initialize()
        else:
            runner.initialize()
        runner.run(self.generations)

    # --- Actions ---

    def action_switch_dashboard(self):
        while len(self.screen_stack) > 1:
            self.pop_screen()

    def action_switch_review(self):
        if self._latest_pieces and self._latest_scores:
            self.push_screen(ReviewScreen())
            self.call_later(self._load_review)

    def _load_review(self):
        try:
            s = self.screen
            g = s.query_one("#review-grid", ReviewGrid)
            g.set_pieces(self._latest_pieces, self._latest_scores)
            cur = g.get_current()
            if cur:
                grid, idx, scores = cur
                s.query_one("#review-detail", DetailPanel).update_detail(grid, idx, scores, False)
        except Exception:
            pass

    def action_quit(self):
        self.exit()
