#!/usr/bin/env python3
"""Evaluate pixel art images in a directory using ArtCritic."""

import argparse
import json
from pathlib import Path

import numpy as np
import PIL.Image

from art.config import ArtConfig
from art.critic import ArtCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a directory of PNG images with ArtCritic")
    parser.add_argument("--dir", required=True, type=Path, help="Directory containing PNG files")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save scores JSON")
    return parser.parse_args()


def load_pngs(directory: Path) -> tuple[list[str], list[np.ndarray]]:
    paths = sorted(directory.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG files found in {directory}")
    names: list[str] = []
    grids: list[np.ndarray] = []
    for p in paths:
        img = PIL.Image.open(p).convert("L")
        arr = np.array(img)
        binary = (arr >= 128).astype(np.uint8)
        names.append(p.name)
        grids.append(binary)
    return names, grids


def main() -> None:
    args = parse_args()

    config = ArtConfig()
    critic = ArtCritic(config)

    print(f"Loading images from {args.dir} ...")
    names, grids = load_pngs(args.dir)
    print(f"Loaded {len(grids)} images. Scoring ...")

    scores = critic.score_batch(grids)

    results = [
        {"name": name, **score}
        for name, score in zip(names, scores)
    ]
    results_sorted = sorted(results, key=lambda r: r["composite"], reverse=True)

    def fmt(r: dict) -> str:
        return (
            f"  {r['name']:<30s} composite={r['composite']:.4f}"
            f"  sym={r['symmetry']:.3f}"
            f"  cplx={r['complexity']:.3f}"
            f"  aes={r['aesthetics']:.3f}"
            f"  div={r['diversity']:.3f}"
        )

    top_n = min(10, len(results_sorted))
    bot_n = min(10, len(results_sorted))

    print(f"\n--- Top {top_n} ---")
    for r in results_sorted[:top_n]:
        print(fmt(r))

    if len(results_sorted) > top_n:
        print(f"\n--- Bottom {bot_n} ---")
        for r in results_sorted[-bot_n:]:
            print(fmt(r))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results_sorted, f, indent=2)
        print(f"\nScores saved to {args.output}")


if __name__ == "__main__":
    main()
