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
from art.tui.widgets.genwatch import GenWatchPanel
from art.tui.widgets.log import SystemLog
from art.tui import audio


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
        self._keys = "[D]ash [G]en Watch [R]eview [Q]uit"

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
                yield GalleryGrid(cols=4, max_pieces=32, id="gallery-panel")
                yield EvolutionPanel(id="evolution-panel")
            with Horizontal(id="bottom-row"):
                yield HeartbeatWidget(id="heartbeat")
                yield BirthWidget(id="birth")
                yield TimelineWidget(id="timeline")
            yield SystemLog(id="system-log")
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
        ("enter", "confirm", "Confirm"),
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
        audio.play_nav_click()
        g = self.query_one("#review-grid", ReviewGrid)
        g.move_cursor(-1)
        self._sync()

    def action_move_right(self):
        audio.play_nav_click()
        g = self.query_one("#review-grid", ReviewGrid)
        g.move_cursor(1)
        self._sync()

    def action_move_up(self):
        audio.play_nav_click()
        g = self.query_one("#review-grid", ReviewGrid)
        g.move_cursor(-g.cols)
        self._sync()

    def action_move_down(self):
        audio.play_nav_click()
        g = self.query_one("#review-grid", ReviewGrid)
        g.move_cursor(g.cols)
        self._sync()

    def action_toggle_fav(self):
        audio.play_favorite()
        g = self.query_one("#review-grid", ReviewGrid)
        g.toggle_favorite()
        self._sync()

    def action_confirm(self):
        g = self.query_one("#review-grid", ReviewGrid)
        favs = g.get_favorites()
        self.app.set_human_picks(favs)
        self.app.pop_screen()

    def _sync(self):
        g = self.query_one("#review-grid", ReviewGrid)
        d = self.query_one("#review-detail", DetailPanel)
        cur = g.get_current()
        if cur:
            grid, idx, scores = cur
            d.update_detail(grid, idx, scores, idx in g._favorites)


class GenerationScreen(Screen):
    """Full-screen view to watch all images being generated live."""
    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("d", "app.pop_screen", "Dashboard"),
    ]

    def compose(self) -> ComposeResult:
        yield GenWatchPanel(id="genwatch-panel")

    def on_mount(self):
        # Sync current state from app
        app = self.app
        panel = self.query_one("#genwatch-panel", GenWatchPanel)
        if hasattr(app, "_current_gen"):
            panel._generation = app._current_gen
        if app._latest_pieces:
            panel.update_scored(app._latest_pieces, app._latest_scores)
            if app._latest_selected_indices:
                panel.update_selected(app._latest_selected_indices)



class ArtApp(App):
    CSS = APP_CSS
    TITLE = "ArtAgent"
    BINDINGS = [
        ("d", "switch_dashboard", "Dashboard"),
        ("g", "switch_genwatch", "Gen Watch"),
        ("r", "switch_review", "Review"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        generations: int = 50,
        resume: bool = True,
        web: bool = False,
        web_port: int = 8765,
    ):
        super().__init__()
        self.generations = generations
        self.do_resume = resume
        self.event_bus = EventBus()
        self.config = ArtConfig()

        if web:
            from art.web.bridge import WebBridge
            self._web_bridge = WebBridge(self.event_bus, port=web_port)
            self._web_bridge.start()
        else:
            self._web_bridge = None
        ensure_dirs(self.config)
        self._latest_pieces: list[np.ndarray] = []
        self._latest_scores: list[dict] = []
        self._latest_selected_indices: list[int] = []
        self._latest_confidences: np.ndarray | None = None
        self._train_total_steps: int = 0
        self._human_picks: list[int] | None = None  # set by review confirm
        self._is_finetuning: bool = False

    def on_mount(self):
        self.push_screen(DashboardScreen())
        self._wire_events()
        audio.play_startup()
        self._log("SYS", "init", f"ArtAgent started — {self.generations} generations")
        self._log("SYS", "init", f"device={self.config.data_dir}  vocab={self.config.vocab_size}  colors={self.config.n_colors}")
        self._log("SYS", "init", f"model: d={self.config.d_model} h={self.config.n_heads} L={self.config.n_layers}")
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
        bus.on("train_preview", self._on_train_preview)
        bus.on("gen_progress", self._on_gen_progress)
        bus.on("neural_activity", self._on_neural_activity)
        # Phase transition events
        bus.on("scoring_start", self._on_scoring_start)
        bus.on("scoring_progress", self._on_scoring_progress)
        bus.on("finetune_start", self._on_finetune_start)
        bus.on("saving_start", self._on_saving_start)
        bus.on("saving_complete", self._on_saving_complete)
        bus.on("init_phase", self._on_init_phase)
        bus.on("init_bootstrap_done", self._on_init_bootstrap_done)
        bus.on("resume_found", self._on_resume_found)
        bus.on("resume_checkpoint", self._on_resume_checkpoint)
        bus.on("mps_cache_cleared", self._on_mps_cache_cleared)
        bus.on("bootstrap_progress", self._on_bootstrap_progress)
        bus.on("bootstrap_save_progress", self._on_bootstrap_save_progress)
        bus.on("saving_piece", self._on_saving_piece)

    def _log(self, level: str, source: str, message: str):
        try:
            self.screen.query_one("#system-log", SystemLog).log(level, source, message)
        except Exception:
            pass

    # --- Train events ---

    def _on_train_start(self, total_steps: int, phase: str):
        self.call_from_thread(self._u_train_start, total_steps, phase)

    def _u_train_start(self, total_steps: int, phase: str):
        try:
            audio.play_train_start()
            self._train_total_steps = total_steps
            s = self.screen
            s.query_one("#training-panel", TrainingPanel).update_phase(phase, total_steps)
            s.query_one(HeaderWidget).phase = f"Training ({phase})"
            s.query_one(StatusBar).update_status(f"Training {phase}... 0/{total_steps}")
            self._log("TRAIN", "train", f"started {phase} phase — {total_steps} steps")
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
            s.query_one("#birth", BirthWidget).update_training(
                step, self._train_total_steps, loss, lr,
            )
            pct = step / max(1, self._train_total_steps) * 100
            if step % 10 == 0:
                self._log("TRAIN", "step", f"{step}/{self._train_total_steps} ({pct:.0f}%)  loss={loss:.4f}  grad={grad_norm:.4f}")
        except Exception:
            pass

    def _on_train_preview(self, grids: list, step: int, total_steps: int):
        self.call_from_thread(self._u_train_preview, grids, step, total_steps)

    def _u_train_preview(self, grids: list, step: int, total_steps: int):
        try:
            s = self.screen
            s.query_one("#gallery-panel", GalleryGrid).update_training_preview(
                grids, step, total_steps,
            )
            if grids:
                s.query_one("#birth", BirthWidget).update_training_preview(grids[0])
            self._log("GEN", "model", f"preview generated — {len(grids)} samples at step {step}")
        except Exception:
            pass

    def _on_train_end(self, losses: list[float]):
        self._is_finetuning = False
        self.call_from_thread(self._u_train_end)

    def _u_train_end(self):
        try:
            s = self.screen
            audio.play_train_end()
            s.query_one(StatusBar).update_status("Training complete")
            s.query_one("#birth", BirthWidget).end_training()
            self._log("OK", "train", "training phase complete")
        except Exception:
            pass

    def _on_neural_activity(self, layer_maps, embedding_sim, weight_norms, step, total_steps):
        self.call_from_thread(
            self._u_neural_activity, layer_maps, embedding_sim, weight_norms, step, total_steps,
        )

    def _u_neural_activity(self, layer_maps, embedding_sim, weight_norms, step, total_steps):
        try:
            self.screen.query_one("#gallery-panel", GalleryGrid).update_neural_activity(
                layer_maps, embedding_sim, weight_norms, step, total_steps,
            )
            max_act = max(m.max() for m in layer_maps)
            self._log("NEURAL", "model", f"activations captured — peak={max_act:.3f}  layers={len(layer_maps)}")
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

    # --- Phase transition events ---

    def _on_scoring_start(self, n_pieces: int):
        self.call_from_thread(self._u_scoring_start, n_pieces)

    def _u_scoring_start(self, n_pieces: int):
        try:
            s = self.screen
            s.query_one(HeaderWidget).phase = "Scoring..."
            s.query_one(StatusBar).update_status(f"Analyzing {n_pieces} pieces with critic...")
            self._log("SCORE", "critic", f"analyzing {n_pieces} pieces — symmetry, complexity, structure, aesthetics")
        except Exception:
            pass
        self._genwatch_call("update_scoring", 0, n_pieces)

    def _on_scoring_progress(self, done: int, total: int, latest_composite: float):
        self.call_from_thread(self._u_scoring_progress, done, total, latest_composite)

    def _u_scoring_progress(self, done: int, total: int, latest_composite: float):
        try:
            pct = done / max(1, total) * 100
            s = self.screen
            s.query_one(StatusBar).update_status(f"Scoring... {done}/{total} ({pct:.0f}%)")
            s.query_one("#gallery-panel", GalleryGrid).update_scoring_progress(done, total)
            self._log("SCORE", "critic", f"scored {done}/{total} ({pct:.0f}%) — latest={latest_composite:.3f}")
        except Exception:
            pass
        self._genwatch_call("update_scoring", done, total)

    def _on_finetune_start(self, n_selected: int, generation: int):
        self._is_finetuning = True
        self.call_from_thread(self._u_finetune_start, n_selected, generation)

    def _u_finetune_start(self, n_selected: int, generation: int):
        try:
            s = self.screen
            s.query_one(HeaderWidget).phase = "Finetuning..."
            s.query_one(StatusBar).update_status(f"Finetuning on {n_selected} selected pieces...")
            self._log("TRAIN", "gas", f"finetuning model on {n_selected} selected pieces")
        except Exception:
            pass

    def _on_saving_start(self, generation: int):
        self.call_from_thread(self._u_saving_start, generation)

    def _u_saving_start(self, generation: int):
        try:
            s = self.screen
            s.query_one(HeaderWidget).phase = "Saving..."
            s.query_one(StatusBar).update_status(f"Saving gen {generation}...")
            self._log("SAVE", "io", f"saving generation {generation} — pieces, scores, checkpoint")
        except Exception:
            pass

    def _on_saving_complete(self, generation: int, n_pieces: int):
        self.call_from_thread(self._u_saving_complete, generation, n_pieces)

    def _u_saving_complete(self, generation: int, n_pieces: int):
        try:
            audio.play_save()
            self._log("OK", "io", f"saved {n_pieces} pieces + checkpoint to gen_{generation:03d}/")
        except Exception:
            pass

    def _on_init_phase(self, phase: str):
        self.call_from_thread(self._u_init_phase, phase)

    def _u_init_phase(self, phase: str):
        try:
            labels = {
                "bootstrap_gen": "Generating bootstrap patterns...",
                "bootstrap_train": "Bootstrap training starting...",
                "resume": "Resuming from checkpoint...",
            }
            label = labels.get(phase, phase)
            s = self.screen
            s.query_one(HeaderWidget).phase = label
            s.query_one(StatusBar).update_status(label)
            self._log("SYS", "init", label)
        except Exception:
            pass

    def _on_init_bootstrap_done(self, n_patterns: int):
        self.call_from_thread(self._u_init_bootstrap_done, n_patterns)

    def _u_init_bootstrap_done(self, n_patterns: int):
        try:
            self._log("OK", "init", f"generated {n_patterns} bootstrap patterns — lines, shapes, symmetries")
        except Exception:
            pass

    def _on_resume_found(self, generation: int):
        self.call_from_thread(self._u_resume_found, generation)

    def _u_resume_found(self, generation: int):
        try:
            self._log("SYS", "resume", f"found existing run — latest generation: {generation}")
        except Exception:
            pass

    def _on_resume_checkpoint(self, generation: int):
        self.call_from_thread(self._u_resume_checkpoint, generation)

    def _u_resume_checkpoint(self, generation: int):
        try:
            audio.play_resume()
            self._log("OK", "resume", f"loaded model checkpoint from gen_{generation:03d}/")
        except Exception:
            pass

    def _on_mps_cache_cleared(self):
        self.call_from_thread(self._u_mps_cache_cleared)

    def _u_mps_cache_cleared(self):
        try:
            self._log("SYS", "mem", "MPS cache cleared — freed GPU memory")
        except Exception:
            pass

    def _on_bootstrap_progress(self, done: int, total: int, category: str):
        self.call_from_thread(self._u_bootstrap_progress, done, total, category)

    def _u_bootstrap_progress(self, done: int, total: int, category: str):
        try:
            pct = done / max(1, total) * 100
            s = self.screen
            s.query_one(StatusBar).update_status(f"Generating patterns... {done}/{total} ({pct:.0f}%)")
            self._log("GEN", "boot", f"{done}/{total} patterns ({pct:.0f}%) — {category}")
        except Exception:
            pass

    def _on_bootstrap_save_progress(self, done: int, total: int):
        self.call_from_thread(self._u_bootstrap_save_progress, done, total)

    def _u_bootstrap_save_progress(self, done: int, total: int):
        try:
            pct = done / max(1, total) * 100
            s = self.screen
            s.query_one(StatusBar).update_status(f"Saving patterns... {done}/{total}")
            self._log("SAVE", "boot", f"saved {done}/{total} pattern PNGs ({pct:.0f}%)")
        except Exception:
            pass

    def _on_saving_piece(self, done: int, total: int):
        self.call_from_thread(self._u_saving_piece, done, total)

    def _u_saving_piece(self, done: int, total: int):
        try:
            pct = done / max(1, total) * 100
            s = self.screen
            s.query_one(StatusBar).update_status(f"Saving pieces... {done}/{total}")
            self._log("SAVE", "io", f"writing piece PNGs... {done}/{total} ({pct:.0f}%)")
        except Exception:
            pass

    # --- Generation events ---

    def _on_gen_start(self, generation: int, temperature: float):
        self.call_from_thread(self._u_gen_start, generation, temperature)

    def _u_gen_start(self, generation: int, temperature: float):
        try:
            audio.play_gen_start()
            self._current_gen = generation
            s = self.screen
            h = s.query_one(HeaderWidget)
            h.generation = generation
            h.temperature = temperature
            h.phase = "Generating..."
            s.query_one(StatusBar).update_status(f"Gen {generation} — Generating...")
            self._log("GEN", "gas", f"generation {generation} started — temp={temperature:.3f}")
        except Exception:
            pass
        self._genwatch_call("update_gen_start", generation, temperature)

    def _on_gen_progress(self, grids: list, pixel: int, total_pixels: int):
        self.call_from_thread(self._u_gen_progress, grids, pixel, total_pixels)

    def _u_gen_progress(self, grids: list, pixel: int, total_pixels: int):
        try:
            s = self.screen
            s.query_one("#gallery-panel", GalleryGrid).update_generation_progress(
                grids, pixel, total_pixels,
            )
            pct = pixel / max(1, total_pixels) * 100
            s.query_one(StatusBar).update_status(
                f"Gen {getattr(self, '_current_gen', '?')} — Drawing... {pct:.0f}%"
            )
            row = pixel // 16
            self._log("GEN", "draw", f"row {row}/16 ({pct:.0f}%) — {len(grids)} pieces live")
        except Exception:
            pass
        self._genwatch_call("update_progress", grids, pixel, total_pixels)

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
        self.call_from_thread(self._genwatch_call, "update_scored", pieces, scores)

    def _u_gen_scored(self, pieces: list, scores: list):
        try:
            s = self.screen
            s.query_one("#gallery-panel", GalleryGrid).update_pieces(pieces, scores)
            s.query_one("#evolution-panel", EvolutionPanel).update_scores(scores)

            # Birth: show best piece with its confidence
            ranked = sorted(range(len(scores)), key=lambda i: scores[i].get("composite", 0), reverse=True)
            if ranked:
                best_idx = ranked[0]
                conf = np.zeros(256)
                if self._latest_confidences is not None and best_idx < self._latest_confidences.shape[0]:
                    conf = self._latest_confidences[best_idx, 1:257]
                s.query_one("#birth", BirthWidget).update_birth(
                    pieces[best_idx], conf, best_idx,
                    generation=getattr(self, "_current_gen", 0),
                )

            audio.play_masterpiece()
            s.query_one(HeaderWidget).phase = "Selecting & Finetuning..."
            s.query_one(StatusBar).update_status("Scored — selecting & finetuning...")

            composites = [s.get("composite", 0) for s in scores]
            best = max(composites)
            mean = sum(composites) / len(composites)
            self._log("SCORE", "critic", f"scored {len(scores)} pieces — best={best:.3f}  mean={mean:.3f}")
        except Exception:
            pass

    def _on_gen_selected(self, selected: list, indices: list[int]):
        self._latest_selected_indices = indices
        self.call_from_thread(self._u_gen_selected, indices)
        self.call_from_thread(self._genwatch_call, "update_selected", indices)

    def _u_gen_selected(self, indices: list[int]):
        try:
            self.screen.query_one("#gallery-panel", GalleryGrid).mark_selected(indices)
        except Exception:
            pass
        self._log("SELECT", "gas", f"selected {len(indices)} pieces — indices: {indices[:8]}{'...' if len(indices) > 8 else ''}")

    def _on_gen_complete(self, summary: dict):
        self.call_from_thread(self._u_gen_complete, summary)

    def _u_gen_complete(self, summary: dict):
        try:
            s = self.screen
            gen = summary.get("generation", 0)
            self._current_gen = gen

            audio.play_gen_complete()
            s.query_one("#evolution-panel", EvolutionPanel).update_generation(summary)
            s.query_one(HeaderWidget).phase = "Generation complete"
            s.query_one(StatusBar).update_status(
                f"Gen {gen} complete — mean: {summary.get('mean_score', 0):.3f}"
            )
            self._log("OK", "gas", f"gen {gen} complete — mean={summary.get('mean_score', 0):.3f}  max={summary.get('max_score', 0):.3f}  temp={summary.get('temperature', 0):.3f}")
            self._log("SAVE", "io", f"checkpoint saved to data/collections/gen_{gen:03d}/")

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
                        gen, self._latest_pieces[best_idx], best_score,
                    )
        except Exception:
            pass

    def _on_evolution_step(self, summary: dict, log: list[dict]):
        self.call_from_thread(self._u_evolution_step)

    def _u_evolution_step(self):
        try:
            self.screen.query_one(HeaderWidget).total_generations = self.generations
            self._log("SYS", "gas", f"evolution step — target={self.generations} generations")
        except Exception:
            pass

    # --- Worker ---

    def _genwatch_call(self, method: str, *args):
        """Forward event to GenWatchPanel if the GenerationScreen is active."""
        try:
            if isinstance(self.screen, GenerationScreen):
                panel = self.screen.query_one("#genwatch-panel", GenWatchPanel)
                getattr(panel, method)(*args)
        except Exception:
            pass

    def _consume_human_picks(self) -> list[int] | None:
        """Called by the runner each generation to collect human picks."""
        picks = self._human_picks
        self._human_picks = None
        return picks

    @work(thread=True)
    def _run_evolution(self):
        runner = OvernightRunner(
            self.config,
            event_bus=self.event_bus,
        )
        if self.do_resume:
            if not runner.resume():
                runner.initialize()
        else:
            runner.initialize()
        runner.run(self.generations, human_picks_fn=self._consume_human_picks)

    # --- Actions ---

    def action_switch_dashboard(self):
        if not isinstance(self.screen, DashboardScreen):
            self.pop_screen()

    def action_switch_genwatch(self):
        if isinstance(self.screen, GenerationScreen):
            return
        if not isinstance(self.screen, DashboardScreen):
            self.pop_screen()
        self.push_screen(GenerationScreen())

    def action_switch_review(self):
        if not self._latest_pieces or not self._latest_scores:
            return
        if not isinstance(self.screen, DashboardScreen):
            self.pop_screen()
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

    def set_human_picks(self, indices: list[int]):
        """Called from ReviewScreen when user confirms favorites."""
        self._human_picks = indices if indices else None
        self._log("SELECT", "human", f"human override: {len(indices)} picks — {indices}")

    def action_quit(self):
        self.exit()
