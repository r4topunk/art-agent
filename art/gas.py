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
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.event_bus = event_bus
        self.trainer = Trainer(model, config, device, event_bus=event_bus)
        self.critic = ArtCritic(config)
        self.tokenizer = PixelTokenizer(config)
        self.generation = 0

    def get_temperature(self) -> float:
        if self.generation >= self.config.temp_generations:
            return self.config.temp_end
        t = self.generation / max(1, self.config.temp_generations)
        return self.config.temp_start + t * (self.config.temp_end - self.config.temp_start)

    def generate_pieces(self, n: int | None = None) -> list[np.ndarray]:
        if n is None:
            n = self.config.images_per_gen
        temperature = self.get_temperature()
        self.model.eval()

        tokens, confidences = self.model.generate_with_confidence(
            batch_size=n,
            temperature=temperature,
            device=str(self.device),
        )

        if self.event_bus:
            self.event_bus.emit("gen_confidences", confidences=confidences.numpy())

        pieces: list[np.ndarray] = []
        for i in range(tokens.shape[0]):
            token_list = tokens[i].tolist()
            img = self.tokenizer.decode(token_list)
            arr = np.array(img)
            binary = (arr >= 128).astype(np.uint8)
            pieces.append(binary)
        return pieces

    def evaluate(self, pieces: list[np.ndarray]) -> list[dict]:
        return self.critic.score_batch(pieces)

    def select(
        self,
        pieces: list[np.ndarray],
        scores: list[dict],
        human_picks: list[int] | None = None,
    ) -> list[np.ndarray]:
        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i]["composite"],
            reverse=True,
        )
        top_indices = ranked[: self.config.select_top]

        if human_picks:
            valid_picks = [i for i in human_picks if 0 <= i < len(pieces)]
            merged: list[int] = list(dict.fromkeys(top_indices + valid_picks))
        else:
            merged = top_indices

        return [pieces[i] for i in merged]

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
        import PIL.Image

        pieces_dir = gen_dir / "pieces"
        pieces_dir.mkdir(parents=True, exist_ok=True)

        for idx, piece in enumerate(pieces):
            img_array = (piece * 255).astype(np.uint8)
            img = PIL.Image.fromarray(img_array, mode="L")
            img.save(pieces_dir / f"piece_{idx:04d}.png")

        with open(gen_dir / "scores.json", "w") as f:
            json.dump(scores, f, indent=2)

        with open(gen_dir / "selections.json", "w") as f:
            json.dump(selections, f, indent=2)

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

        if self.event_bus:
            self.event_bus.emit("gen_scored", pieces=pieces, scores=scores)

        selected = self.select(pieces, scores, human_picks=human_picks)

        # Determine which original indices were selected for the selections manifest
        composite = [s["composite"] for s in scores]
        ranked_indices = sorted(range(len(composite)), key=lambda i: composite[i], reverse=True)
        selection_indices = ranked_indices[: self.config.select_top]
        if human_picks:
            merged_set: list[int] = list(dict.fromkeys(selection_indices + human_picks))
            selection_indices = merged_set

        if self.event_bus:
            self.event_bus.emit("gen_selected", selected=selected, indices=selection_indices)

        self.finetune(selected, bootstrap_patterns=bootstrap_patterns)
        self.save_generation(gen_dir, pieces, scores, selection_indices)

        composites = [s["composite"] for s in scores]
        summary = {
            "generation": self.generation,
            "temperature": self.get_temperature(),
            "mean_score": float(np.mean(composites)),
            "max_score": float(np.max(composites)),
            "min_score": float(np.min(composites)),
            "n_pieces": len(pieces),
            "n_selected": len(selected),
        }

        if self.event_bus:
            self.event_bus.emit("gen_complete", summary=summary)

        self.generation += 1
        return summary
