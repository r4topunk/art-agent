from __future__ import annotations

import copy
import json
import random
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from art.config import ArtConfig
from art.critic import ArtCritic
from art.data import PixelDataset
from art.events import EventBus
from art.model import PixelGPT
from art.tokenizer import PixelTokenizer
from art.trainer import Trainer


class GASLoop:
    def __init__(
        self,
        model: PixelGPT,
        config: ArtConfig,
        device: torch.device,
        event_bus: EventBus | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.event_bus = event_bus
        self.trainer = Trainer(model, config, device, event_bus=event_bus)
        self.critic = ArtCritic(config)
        self.tokenizer = PixelTokenizer(config)
        self.generation = 0
        self._diversity_low = False
        # Archive of past selected pieces for repetition penalty
        self.archive: list[np.ndarray] = []
        # Style fingerprints (top-2 dominant color indices) of archived pieces
        self.archive_fingerprints: list[frozenset] = []
        # Background save thread pool (single worker to serialize saves)
        self._save_pool = ThreadPoolExecutor(max_workers=1)
        self._save_future: Future | None = None

    def get_temperature(self) -> float:
        if self._diversity_low:
            return min(1.3, self.config.temp_start + 0.3)
        if self.generation >= self.config.temp_generations:
            base = self.config.temp_end
        else:
            t = self.generation / max(1, self.config.temp_generations)
            base = self.config.temp_start + t * (self.config.temp_end - self.config.temp_start)
        # Periodic spike every 50 gens to escape local optima
        if self.generation > 0 and self.generation % 50 == 0:
            base = max(base, 1.3)
        return base

    def generate_pieces(self, n: int | None = None) -> list[np.ndarray]:
        if n is None:
            n = self.config.images_per_gen
        temperature = self.get_temperature()
        self.model.eval()

        # Stream live progress to TUI: decode every 4 pixels for smooth animation
        display_n = n
        grid_size = self.config.grid_size
        total_pixels = grid_size * grid_size
        bus = self.event_bus

        def on_token(t: int, seq, conf):
            pixel = t - 1  # token 1 = pixel 0
            if pixel >= 0 and pixel % 4 == 0 and bus:
                partial_grids = []
                for i in range(display_n):
                    grid = self.tokenizer.decode_to_grid(seq[i].tolist())
                    partial_grids.append(grid)
                bus.emit(
                    "gen_progress",
                    grids=partial_grids,
                    pixel=pixel,
                    total_pixels=total_pixels,
                )

        tokens, confidences = self.model.generate_with_confidence(
            batch_size=n,
            temperature=temperature,
            top_p=self.config.top_p,
            device=str(self.device),
            on_token=on_token if bus else None,
        )

        if self.event_bus:
            self.event_bus.emit("gen_confidences", confidences=confidences.numpy())

        pieces: list[np.ndarray] = []
        for i in range(tokens.shape[0]):
            token_list = tokens[i].tolist()
            grid = self.tokenizer.decode_to_grid(token_list)
            pieces.append(grid)
        return pieces

    def evaluate(self, pieces: list[np.ndarray]) -> list[dict]:
        if self.event_bus:
            self.event_bus.emit("scoring_start", n_pieces=len(pieces))

        def on_score_progress(done, total, latest_score):
            if self.event_bus:
                self.event_bus.emit(
                    "scoring_progress",
                    done=done, total=total,
                    latest_composite=latest_score.get("composite", 0),
                )

        scores = self.critic.score_batch(pieces, on_progress=on_score_progress)

        # Intra-generation penalty: punish duplicates within the same batch
        # harder (0.7 multiplier) so the model is pushed toward variety.
        # Also groups "dominant color" pieces together — if multiple pieces
        # in the batch are 60%+ one color, they're treated as similar.
        dominant_seen = 0  # count of dominant-color pieces already seen
        for i in range(len(pieces)):
            _, counts = np.unique(pieces[i], return_counts=True)
            dominance_i = float(counts.max()) / pieces[i].size
            is_dominant = dominance_i > 0.4

            max_sim = 0.0
            for j in range(i):
                pixel_sim = 1.0 - self._hamming(pieces[i], pieces[j])
                struct_sim = self._structural_similarity(pieces[i], pieces[j])
                sim = max(pixel_sim, struct_sim)
                # Boost similarity between dominant-color pieces even if
                # their patterns differ slightly — they look the same
                if is_dominant:
                    _, counts_j = np.unique(pieces[j], return_counts=True)
                    if float(counts_j.max()) / pieces[j].size > 0.4:
                        sim = max(sim, 0.85)
                max_sim = max(max_sim, sim)

            if max_sim > 0.6:
                penalty = (max_sim - 0.6) / 0.4  # 0..1
                scores[i]["intra_gen_penalty"] = penalty
                scores[i]["composite"] *= (1.0 - 0.7 * penalty)
            else:
                scores[i]["intra_gen_penalty"] = 0.0

            if is_dominant:
                dominant_seen += 1

        # Repetition penalty: punish pieces too similar to past archived pieces
        # Uses both pixel-level AND structural (color-agnostic) similarity
        # so same pattern with different colors is also penalised.
        if self.archive:
            for i, piece in enumerate(pieces):
                max_pixel_sim = max(
                    1.0 - self._hamming(piece, past) for past in self.archive
                )
                max_struct_sim = max(
                    self._structural_similarity(piece, past)
                    for past in self.archive
                )
                # Take the worse of the two similarities
                max_sim = max(max_pixel_sim, max_struct_sim)
                # Penalty ramps up: 0 when similarity < 0.7, full at 1.0
                if max_sim > 0.5:
                    penalty = (max_sim - 0.5) / 0.5  # 0..1
                    scores[i]["repetition_penalty"] = penalty
                    scores[i]["composite"] *= (1.0 - 0.6 * penalty)
                else:
                    scores[i]["repetition_penalty"] = 0.0

        # Style-level penalty: penalize pieces whose top-2 color style is
        # over-represented in the archive. This catches cases where the model
        # keeps regenerating the same color scheme (e.g. always yellow+orange)
        # even when pixel-level Hamming distances are sufficient to avoid the
        # repetition penalty above.
        if self.archive_fingerprints:
            for i, piece in enumerate(pieces):
                fp = self._color_fingerprint(piece)
                matches = sum(1 for afp in self.archive_fingerprints if afp == fp)
                if matches > 4:
                    style_penalty = min(0.40, (matches - 4) * 0.05)
                    scores[i]["composite"] *= (1.0 - style_penalty)

        return scores

    def _color_fingerprint(self, grid: np.ndarray) -> frozenset:
        """Top-2 dominant color indices as an order-independent style fingerprint."""
        counts = np.bincount(grid.flatten(), minlength=self.config.n_colors)
        top2 = np.argsort(counts)[-2:]
        return frozenset(top2.tolist())

    def _hamming(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a != b)) / a.size

    @staticmethod
    def _canonicalize(grid: np.ndarray) -> np.ndarray:
        """Remap colors by order of first appearance so structure is color-agnostic."""
        flat = grid.ravel()
        mapping = {}
        next_id = 0
        canon = np.empty_like(flat)
        for i, v in enumerate(flat):
            if v not in mapping:
                mapping[v] = next_id
                next_id += 1
            canon[i] = mapping[v]
        return canon.reshape(grid.shape)

    def _structural_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Similarity ignoring color identity — same pattern different colors = 1.0."""
        ca = self._canonicalize(a)
        cb = self._canonicalize(b)
        return 1.0 - float(np.sum(ca != cb)) / ca.size

    def select(
        self,
        pieces: list[np.ndarray],
        scores: list[dict],
        human_picks: list[int] | None = None,
    ) -> tuple[list[np.ndarray], list[int]]:
        n_select = self.config.select_top
        composites = [s["composite"] for s in scores]

        # When archive exists, reserve 1 slot for an "exploration pick" —
        # the piece most distant from recent archive history regardless of score.
        # This prevents the selection pool from converging prematurely.
        use_exploration = len(self.archive) >= 5
        n_greedy = n_select - 1 if use_exploration else n_select

        # Greedy max-min diversity selection:
        # Pick #1 by score, then each next pick maximizes
        # min hamming distance to already-selected pieces
        best_idx = int(np.argmax(composites))
        selected_indices = [best_idx]

        for _ in range(n_greedy - 1):
            best_score = -1.0
            best_i = -1
            for i in range(len(pieces)):
                if i in selected_indices:
                    continue
                # Min distance to any already-selected piece (pixel + structural)
                min_dist = min(
                    min(self._hamming(pieces[i], pieces[j]),
                        1.0 - self._structural_similarity(pieces[i], pieces[j]))
                    for j in selected_indices
                )
                # Blend: 50% composite score + 50% diversity from selected set
                blended = 0.5 * composites[i] + 0.5 * min_dist
                if blended > best_score:
                    best_score = blended
                    best_i = i
            if best_i >= 0:
                selected_indices.append(best_i)

        # Exploration pick: piece with highest mean distance to recent archive,
        # subject to a minimum quality gate (composite > 0.15).
        if use_exploration:
            recent_archive = self.archive[-20:]
            best_novel_i = -1
            best_novelty = -1.0
            for i in range(len(pieces)):
                if i in selected_indices:
                    continue
                if composites[i] < 0.15:
                    continue
                novelty = float(np.mean([self._hamming(pieces[i], p) for p in recent_archive]))
                if novelty > best_novelty:
                    best_novelty = novelty
                    best_novel_i = i
            if best_novel_i >= 0:
                selected_indices.append(best_novel_i)

        if human_picks:
            valid_picks = [i for i in human_picks if 0 <= i < len(pieces)]
            selected_indices = list(dict.fromkeys(selected_indices + valid_picks))

        return [pieces[i] for i in selected_indices], selected_indices

    def _augment_rotations(self, patterns: list[np.ndarray]) -> list[np.ndarray]:
        """Expand each pattern into 4 rotations (0°, 90°, 180°, 270°)."""
        augmented = []
        for p in patterns:
            for k in range(4):
                augmented.append(np.rot90(p, k).copy())
        return augmented

    def _bootstrap_ratio(self) -> float:
        """Bootstrap mix ratio decays from start to end over decay_generations."""
        cfg = self.config
        if self.generation >= cfg.bootstrap_decay_generations:
            return cfg.bootstrap_mix_ratio_end
        t = self.generation / max(1, cfg.bootstrap_decay_generations)
        return cfg.bootstrap_mix_ratio_start + t * (cfg.bootstrap_mix_ratio_end - cfg.bootstrap_mix_ratio_start)

    def finetune(
        self,
        selected: list[np.ndarray],
        bootstrap_patterns: list[np.ndarray] | None = None,
    ) -> None:
        training_patterns = list(selected)

        if (
            self.generation % self.config.bootstrap_mix_interval == 0
            and bootstrap_patterns
        ):
            n_bootstrap = max(1, int(len(selected) * self.config.bootstrap_mix_ratio))
            sampled = random.sample(bootstrap_patterns, min(n_bootstrap, len(bootstrap_patterns)))
            training_patterns = training_patterns + sampled

        dataset = PixelDataset(training_patterns, self.config)
        self.trainer.train(
            dataset,
            steps=self.config.finetune_steps,
            lr=self.config.finetune_lr,
        )

    @staticmethod
    def _make_kaleidoscope(pil_images: list, size: int = 1200) -> "PIL.Image.Image":
        """Build a 4-way mirrored kaleidoscope from random pieces."""
        import PIL.Image

        # Pick 1-5 source images, tile into a quadrant, then mirror
        k = random.randint(1, min(5, len(pil_images)))
        pool = random.sample(pil_images, k)

        # Quadrant: fill with randomly transformed tiles
        half = size // 2
        n_tiles = random.randint(4, 8)  # tiles per side in quadrant
        tile_size = max(1, half // n_tiles)
        cols = (half + tile_size - 1) // tile_size
        rows = (half + tile_size - 1) // tile_size

        quad = PIL.Image.new("RGB", (half, half), color=(0, 0, 0))
        for gy in range(rows):
            for gx in range(cols):
                img = random.choice(pool).resize((tile_size, tile_size), PIL.Image.NEAREST)
                # Random rotation (0, 90, 180, 270)
                rot = random.choice([0, 90, 180, 270])
                if rot:
                    img = img.rotate(rot)
                # Random horizontal flip
                if random.random() < 0.5:
                    img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                quad.paste(img, (gx * tile_size, gy * tile_size))

        # Crop quadrant to exact half size
        quad = quad.crop((0, 0, half, half))

        # Mirror: right = fliplr, bottom = flipud
        right = quad.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        top = PIL.Image.new("RGB", (size, half))
        top.paste(quad, (0, 0))
        top.paste(right, (half, 0))
        bottom = top.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        full = PIL.Image.new("RGB", (size, size))
        full.paste(top, (0, 0))
        full.paste(bottom, (0, half))
        return full

    def _save_generation_sync(
        self,
        gen_dir: Path,
        pieces: list[np.ndarray],
        scores: list[dict],
        selections: list[int],
        generation: int,
        checkpoint_data: dict,
    ) -> None:
        """I/O-heavy save that runs in a background thread."""
        from art.config import PALETTE_16
        import PIL.Image

        # Only save the selected pieces (not all generated)
        selected_scores = [scores[i] for i in selections]

        # Find the best piece among selections (highest composite)
        best_local_idx = int(np.argmax([s["composite"] for s in selected_scores]))
        best_global_idx = selections[best_local_idx]

        # Convert all pieces to PIL images
        all_pil = []
        for idx, piece in enumerate(pieces):
            h, w = piece.shape
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for ci in range(self.config.n_colors):
                mask = piece == ci
                rgb[mask] = PALETTE_16[ci]
            all_pil.append(PIL.Image.fromarray(rgb, mode="RGB"))

        # --- Organized image folders ---
        col_dir = self.config.collections_dir

        # all/ — every piece from this generation (native size)
        all_dir = col_dir / "all"
        all_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(len(all_pil)):
            all_pil[idx].save(all_dir / f"gen_{generation:03d}_piece_{idx:02d}.png")

        # top5/ — top 5 selected pieces upscaled to 1024x1024
        top5_dir = col_dir / "top5"
        top5_dir.mkdir(parents=True, exist_ok=True)
        ranked_selections = sorted(selections, key=lambda i: scores[i]["composite"], reverse=True)
        for rank, idx in enumerate(ranked_selections[:5]):
            upscaled = all_pil[idx].resize((1200, 1200), PIL.Image.NEAREST)
            upscaled.save(top5_dir / f"gen_{generation:03d}_top{rank + 1}.png")

        # hall_of_fame/ — single best piece per generation (1024x1024)
        hall_dir = col_dir / "hall_of_fame"
        hall_dir.mkdir(parents=True, exist_ok=True)
        best_hof = all_pil[best_global_idx].resize((1200, 1200), PIL.Image.NEAREST)
        best_hof.save(hall_dir / f"gen_{generation:03d}_best.png")

        # grids/ — 6x6 grid of all pieces (1020x1020, no border gap)
        grids_dir = col_dir / "grids"
        grids_dir.mkdir(parents=True, exist_ok=True)
        cell = 200
        grid_size = cell * 6  # 1200
        grid_img = PIL.Image.new("RGB", (grid_size, grid_size), color=(0, 0, 0))
        for i, img in enumerate(all_pil[:36]):
            row, col_i = divmod(i, 6)
            x, y = col_i * cell, row * cell
            grid_img.paste(img.resize((cell, cell), PIL.Image.NEAREST), (x, y))
        grid_img.save(grids_dir / f"gen_{generation:03d}_grid.png")

        # kaleidoscope/ — 3 kaleidoscope images from selected pieces (1024x1024)
        kalei_dir = col_dir / "kaleidoscope"
        kalei_dir.mkdir(parents=True, exist_ok=True)
        selected_pil = [all_pil[i] for i in selections if i < len(all_pil)]
        if len(selected_pil) >= 2:
            for ki in range(3):
                kalei_img = self._make_kaleidoscope(selected_pil, size=1200)
                kalei_img.save(kalei_dir / f"gen_{generation:03d}_kalei_{ki + 1}.png")

        if self.event_bus:
            self.event_bus.emit("saving_piece", done=len(selections), total=len(selections))

        # Save only scores for selected pieces, plus metadata
        with open(gen_dir / "scores.json", "w") as f:
            json.dump(selected_scores, f, indent=2)

        with open(gen_dir / "selections.json", "w") as f:
            json.dump(selections, f, indent=2)

        with open(gen_dir / "best.json", "w") as f:
            json.dump({"best_index": best_global_idx, "score": selected_scores[best_local_idx]}, f, indent=2)

        # Save checkpoint to a single fixed location (overwritten each gen, ~55MB)
        checkpoint_path = self.config.collections_dir / "checkpoint.pt"
        torch.save(checkpoint_data, checkpoint_path)

    def save_generation(
        self,
        gen_dir: Path,
        pieces: list[np.ndarray],
        scores: list[dict],
        selections: list[int],
    ) -> None:
        """Snapshot model state and submit save to background thread."""
        # Snapshot checkpoint data now, before the model gets mutated by next finetune
        checkpoint_data = {
            "model_state_dict": copy.deepcopy(self.model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(self.trainer.optimizer.state_dict()),
            "extra": {
                "generation": self.generation,
                "temperature": self.get_temperature(),
            },
        }

        # Wait for any previous save to finish before submitting a new one
        if self._save_future is not None:
            self._save_future.result()

        self._save_future = self._save_pool.submit(
            self._save_generation_sync,
            gen_dir,
            pieces,
            scores,
            selections,
            self.generation,
            checkpoint_data,
        )

    def wait_for_save(self) -> None:
        """Block until the background save completes. Call before exit."""
        if self._save_future is not None:
            self._save_future.result()
            self._save_future = None

    def run_generation(
        self,
        bootstrap_patterns: list[np.ndarray] | None = None,
        human_picks: list[int] | None = None,
    ) -> dict:
        gen_dir = self.config.collections_dir / f"gen_{self.generation:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        print(f"[GAS] Generation {self.generation} | temperature={self.get_temperature():.4f}")

        if self.event_bus:
            self.event_bus.emit("gen_start", generation=self.generation, temperature=self.get_temperature())

        pieces = self.generate_pieces()

        if self.event_bus:
            self.event_bus.emit("gen_pieces", pieces=pieces)

        scores = self.evaluate(pieces)

        # Check batch diversity — spike temperature next gen if collapsing
        batch_div = np.mean([s.get("diversity", 0) for s in scores])
        self._diversity_low = batch_div < self.config.temp_diversity_floor

        if self.event_bus:
            self.event_bus.emit("gen_scored", pieces=pieces, scores=scores)

        selected, selection_indices = self.select(pieces, scores, human_picks=human_picks)

        if self.event_bus:
            self.event_bus.emit("gen_selected", selected=selected, indices=selection_indices)

        if self.event_bus:
            self.event_bus.emit("finetune_start", n_selected=len(selected), generation=self.generation)
        self.finetune(selected, bootstrap_patterns=bootstrap_patterns)

        if self.event_bus:
            self.event_bus.emit("saving_start", generation=self.generation)
        self.save_generation(gen_dir, pieces, scores, selection_indices)
        if self.event_bus:
            self.event_bus.emit("saving_complete", generation=self.generation, n_pieces=len(selected))

        # Add selected pieces to archive for future repetition penalty
        self.archive.extend(selected)
        self.archive_fingerprints.extend(self._color_fingerprint(p) for p in selected)

        composites = [s["composite"] for s in scores]
        # Find the best piece index among selections
        sel_composites = [scores[i]["composite"] for i in selection_indices]
        best_sel_local = int(np.argmax(sel_composites))
        best_idx = selection_indices[best_sel_local]

        summary = {
            "generation": self.generation,
            "temperature": self.get_temperature(),
            "mean_score": float(np.mean(composites)),
            "max_score": float(np.max(composites)),
            "min_score": float(np.min(composites)),
            "n_pieces": len(pieces),
            "n_selected": len(selected),
            "best_index": best_idx,
            "best_score": float(scores[best_idx]["composite"]),
        }

        if self.event_bus:
            self.event_bus.emit("gen_complete", summary=summary)

        self.generation += 1
        return summary
