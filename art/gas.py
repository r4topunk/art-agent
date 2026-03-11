from __future__ import annotations

import json
import random
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
        use_vlm: bool = False,
        vlm_model: str = "moondream",
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.event_bus = event_bus
        self.trainer = Trainer(model, config, device, event_bus=event_bus)
        self.critic = ArtCritic(config)
        self.tokenizer = PixelTokenizer(config)
        self.generation = 0
        self.use_vlm = use_vlm
        self.vlm_model = vlm_model
        self._diversity_low = False
        # Archive of past selected pieces for repetition penalty
        self.archive: list[np.ndarray] = []

    def get_temperature(self) -> float:
        if self._diversity_low:
            return min(1.3, self.config.temp_start + 0.3)
        if self.generation >= self.config.temp_generations:
            return self.config.temp_end
        t = self.generation / max(1, self.config.temp_generations)
        return self.config.temp_start + t * (self.config.temp_end - self.config.temp_start)

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
            is_dominant = dominance_i > 0.6

            max_sim = 0.0
            for j in range(i):
                pixel_sim = 1.0 - self._hamming(pieces[i], pieces[j])
                struct_sim = self._structural_similarity(pieces[i], pieces[j])
                sim = max(pixel_sim, struct_sim)
                # Boost similarity between dominant-color pieces even if
                # their patterns differ slightly — they look the same
                if is_dominant:
                    _, counts_j = np.unique(pieces[j], return_counts=True)
                    if float(counts_j.max()) / pieces[j].size > 0.6:
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
                if max_sim > 0.7:
                    penalty = (max_sim - 0.7) / 0.3  # 0..1
                    scores[i]["repetition_penalty"] = penalty
                    scores[i]["composite"] *= (1.0 - 0.5 * penalty)
                else:
                    scores[i]["repetition_penalty"] = 0.0

        if self.use_vlm:
            try:
                from art.vlm_critic import score_batch_with_vlm

                def on_vlm_progress(done, total, result):
                    if self.event_bus:
                        idx = done - 1
                        desc = result.get("vlm_description", "") if result else ""
                        piece = pieces[idx] if idx < len(pieces) else None
                        algo_score = scores[idx] if idx < len(scores) else {}
                        vlm_scores_partial = {k: v for k, v in (result or {}).items()
                                              if k.startswith("vlm_") and k != "vlm_description"}
                        self.event_bus.emit(
                            "vlm_progress",
                            done=done, total=total,
                            description=desc,
                            piece=piece,
                            algo_scores=algo_score,
                            vlm_scores=vlm_scores_partial,
                        )

                vlm_scores = score_batch_with_vlm(pieces, model=self.vlm_model, on_progress=on_vlm_progress)
                for i, vlm in enumerate(vlm_scores):
                    if vlm is not None:
                        scores[i]["vlm_interest"] = vlm["interest"]
                        scores[i]["vlm_composition"] = vlm["composition"]
                        scores[i]["vlm_creativity"] = vlm["creativity"]
                        scores[i]["vlm_composite"] = vlm["vlm_composite"]
                        if "vlm_description" in vlm:
                            scores[i]["vlm_description"] = vlm["vlm_description"]
                        # Blend: 50% algorithmic + 50% VLM
                        scores[i]["composite"] = (
                            0.5 * scores[i]["composite"]
                            + 0.5 * vlm["vlm_composite"]
                        )
            except Exception as e:
                print(f"[GAS] VLM scoring failed: {e}")

        return scores

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

        # Greedy max-min diversity selection:
        # Pick #1 by score, then each next pick maximizes
        # min hamming distance to already-selected pieces
        best_idx = int(np.argmax(composites))
        selected_indices = [best_idx]

        for _ in range(n_select - 1):
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

        if human_picks:
            valid_picks = [i for i in human_picks if 0 <= i < len(pieces)]
            selected_indices = list(dict.fromkeys(selected_indices + valid_picks))

        return [pieces[i] for i in selected_indices], selected_indices

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

    def save_generation(
        self,
        gen_dir: Path,
        pieces: list[np.ndarray],
        scores: list[dict],
        selections: list[int],
    ) -> None:
        from art.config import PALETTE_16

        pieces_dir = gen_dir / "pieces"
        pieces_dir.mkdir(parents=True, exist_ok=True)

        # Only save the selected pieces (not all generated)
        selected_scores = [scores[i] for i in selections]

        # Find the best piece among selections (highest composite)
        best_local_idx = int(np.argmax([s["composite"] for s in selected_scores]))
        best_global_idx = selections[best_local_idx]

        for rank, idx in enumerate(selections):
            piece = pieces[idx]
            h, w = piece.shape
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for ci in range(self.config.n_colors):
                mask = piece == ci
                rgb[mask] = PALETTE_16[ci]
            img = __import__("PIL").Image.fromarray(rgb, mode="RGB")
            img.save(pieces_dir / f"piece_{idx:04d}.png")

            # Also save the generation's best piece upscaled to 1024x1024
            if idx == best_global_idx:
                img_upscaled = img.resize((1024, 1024), __import__("PIL").Image.NEAREST)
                img_upscaled.save(gen_dir / "best.png")

        if self.event_bus:
            self.event_bus.emit("saving_piece", done=len(selections), total=len(selections))

        # Save only scores for selected pieces, plus metadata
        with open(gen_dir / "scores.json", "w") as f:
            json.dump(selected_scores, f, indent=2)

        with open(gen_dir / "selections.json", "w") as f:
            json.dump(selections, f, indent=2)

        with open(gen_dir / "best.json", "w") as f:
            json.dump({"best_index": best_global_idx, "score": selected_scores[best_local_idx]}, f, indent=2)

        checkpoint_path = gen_dir / "checkpoint.pt"
        self.trainer.save_checkpoint(
            checkpoint_path,
            extra={
                "generation": self.generation,
                "temperature": self.get_temperature(),
            },
        )

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
