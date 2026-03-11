from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from art.config import ArtConfig
from art.data import PixelDataset, generate_bootstrap_patterns, save_bootstrap_patterns
from art.events import EventBus
from art.gas import GASLoop
from art.model import PixelGPT
from art.trainer import Trainer
from art.utils import setup_device


class OvernightRunner:
    def __init__(
        self,
        config: ArtConfig,
        event_bus: EventBus | None = None,
        use_vlm: bool = False,
        vlm_model: str = "moondream",
    ):
        """Initialize the OvernightRunner with config, device, model, and GASLoop."""
        self.config = config
        self.event_bus = event_bus
        self.device = setup_device()

        # Create and move model to device
        self.model = PixelGPT(config)
        self.model = self.model.to(self.device)

        # Create GASLoop
        self.gas = GASLoop(
            self.model, config, self.device,
            event_bus=event_bus,
            use_vlm=use_vlm,
            vlm_model=vlm_model,
        )

        # Initialize evolution log and bootstrap patterns
        self.evolution_log: list[dict] = []
        self.bootstrap_patterns: Optional[list[np.ndarray]] = None

    def find_latest_generation(self) -> Optional[int]:
        """Scan collections_dir for gen_XXX directories and return highest generation number."""
        if not self.config.collections_dir.exists():
            return None

        latest_gen = None
        for item in self.config.collections_dir.iterdir():
            if item.is_dir() and item.name.startswith("gen_"):
                try:
                    gen_num = int(item.name.split("_")[1])
                    if latest_gen is None or gen_num > latest_gen:
                        latest_gen = gen_num
                except (IndexError, ValueError):
                    continue

        return latest_gen

    def resume(self) -> bool:
        """Resume from latest generation if it exists. Return True if resumed, False if starting fresh."""
        latest_gen = self.find_latest_generation()

        if latest_gen is None:
            return False

        if self.event_bus:
            self.event_bus.emit("init_phase", phase="resume")
            self.event_bus.emit("resume_found", generation=latest_gen)

        # Load checkpoint from latest generation
        checkpoint_path = self.config.collections_dir / f"gen_{latest_gen}" / "checkpoint.pt"
        if checkpoint_path.exists():
            self.gas.trainer.load_checkpoint(checkpoint_path)
            print(f"Loaded checkpoint from generation {latest_gen}")
            if self.event_bus:
                self.event_bus.emit("resume_checkpoint", generation=latest_gen)

        # Set gas generation to latest + 1
        self.gas.generation = latest_gen + 1

        # Load evolution_log.json if exists
        log_path = self.config.collections_dir / "evolution_log.json"
        if log_path.exists():
            with open(log_path, "r") as f:
                self.evolution_log = json.load(f)
            print(f"Loaded evolution log with {len(self.evolution_log)} entries")

        # Load bootstrap patterns if they exist
        if self.config.bootstrap_dir.exists():
            bootstrap_files = sorted(self.config.bootstrap_dir.glob("pattern_*.png"))
            if bootstrap_files:
                from art.tokenizer import PixelTokenizer
                tokenizer = PixelTokenizer(self.config)
                patterns = []
                for img_path in bootstrap_files:
                    img = __import__("PIL").Image.open(img_path)
                    tokens = tokenizer.encode(img)
                    grid = tokenizer.decode_to_grid(tokens)
                    patterns.append(grid)
                self.bootstrap_patterns = patterns
                print(f"Loaded {len(patterns)} bootstrap patterns")

        return True

    def initialize(self):
        """Generate bootstrap patterns and train model on bootstrap data."""
        if self.event_bus:
            self.event_bus.emit("init_phase", phase="bootstrap_gen")
        print("Initializing bootstrap patterns...")

        def on_gen_progress(done, total, category):
            if self.event_bus:
                self.event_bus.emit("bootstrap_progress", done=done, total=total, category=category)

        self.bootstrap_patterns = generate_bootstrap_patterns(self.config, on_progress=on_gen_progress)

        if self.event_bus:
            self.event_bus.emit("init_phase", phase="bootstrap_save")

        def on_save_progress(done, total):
            if self.event_bus:
                self.event_bus.emit("bootstrap_save_progress", done=done, total=total)

        save_bootstrap_patterns(self.bootstrap_patterns, self.config, on_progress=on_save_progress)
        print(f"Generated and saved {len(self.bootstrap_patterns)} bootstrap patterns")
        if self.event_bus:
            self.event_bus.emit("init_bootstrap_done", n_patterns=len(self.bootstrap_patterns))

        if self.event_bus:
            self.event_bus.emit("init_phase", phase="bootstrap_train")
        print("Starting bootstrap training...")
        dataset = PixelDataset(self.bootstrap_patterns, self.config)
        trainer = Trainer(self.model, self.config, self.device, event_bus=self.event_bus)
        trainer.train(dataset, steps=self.config.train_steps)
        print("Bootstrap training complete")

    def save_evolution_log(self):
        """Atomically save evolution_log to JSON file."""
        self.config.collections_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.config.collections_dir / "evolution_log.json"

        # Write to temporary file first
        tmp_path = log_path.parent / f"{log_path.name}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(self.evolution_log, f, indent=2)

        # Atomic replace
        os.replace(tmp_path, log_path)

    def run(self, generations: int, auto_only: bool = True):
        """Run the evolutionary loop for N generations."""
        print(f"Starting evolutionary run for {generations} generations")

        for gen_idx in range(generations):
            print(f"\n--- Generation {self.gas.generation} ---")

            # Run generation
            summary = self.gas.run_generation(
                bootstrap_patterns=self.bootstrap_patterns,
                human_picks=[] if auto_only else None
            )

            # Append to evolution log
            self.evolution_log.append(summary)

            if self.event_bus:
                self.event_bus.emit("evolution_step", summary=summary, log=self.evolution_log)

            # Save evolution log atomically
            self.save_evolution_log()

            # Clear cache if using MPS device
            if self.device.type == "mps":
                torch.mps.empty_cache()
                if self.event_bus:
                    self.event_bus.emit("mps_cache_cleared")

            # Print generation summary
            mean_score = summary.get("mean_score", 0)
            max_score = summary.get("max_score", 0)
            temperature = summary.get("temperature", 0)

            print(f"Gen {self.gas.generation}: "
                  f"mean_score={mean_score:.4f}, "
                  f"max_score={max_score:.4f}, "
                  f"temperature={temperature:.4f}")

        print(f"\nEvolution complete. {generations} generations processed.")
