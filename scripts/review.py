import argparse
import json
from pathlib import Path

import numpy as np
import PIL.Image

from art.config import ArtConfig
from art.gallery import Gallery
from art.tokenizer import PixelTokenizer
from art.utils import load_image


def main():
    parser = argparse.ArgumentParser(description="Review and select generated images")
    parser.add_argument(
        "--gen-dir",
        type=Path,
        required=True,
        help="Path to generation directory (e.g., data/collections/gen_000)",
    )
    args = parser.parse_args()

    gen_dir = args.gen_dir
    pieces_dir = gen_dir / "pieces"
    scores_file = gen_dir / "scores.json"
    selections_file = gen_dir / "selections.json"

    # Verify directories exist
    if not pieces_dir.exists():
        print(f"Error: {pieces_dir} does not exist")
        return
    if not scores_file.exists():
        print(f"Error: {scores_file} does not exist")
        return

    # Load images
    image_files = sorted(pieces_dir.glob("*.png"))
    if not image_files:
        print(f"No PNG images found in {pieces_dir}")
        return

    images = [load_image(f) for f in image_files]
    print(f"Loaded {len(images)} images")

    # Load scores
    with open(scores_file, "r") as f:
        all_scores = json.load(f)

    # Handle both list and dict formats for scores
    if isinstance(all_scores, list):
        scores = all_scores
    elif isinstance(all_scores, dict):
        # If it's a dict, extract the scores for each image
        scores = [all_scores.get(str(i), {}) for i in range(len(images))]
    else:
        scores = [{}] * len(images)

    # Create gallery and generate review grid
    config = ArtConfig()
    gallery = Gallery(config)

    grid_path = gen_dir / "review_grid.png"
    gallery.create_review_grid(images, scores, grid_path)
    print(f"Created review grid at {grid_path}")

    # Open preview
    gallery.open_preview(grid_path)

    # Prompt for selections
    human_selections = gallery.prompt_selections()

    # Get auto selections from critic (top selections based on composite score)
    auto_selections = []
    if scores and all("composite" in s for s in scores):
        # Get indices sorted by composite score
        indexed_scores = [
            (i, s.get("composite", 0.0))
            for i, s in enumerate(scores)
        ]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        auto_selections = [idx for idx, _ in indexed_scores[: config.select_top]]

    print(f"\nHuman selections: {human_selections}")
    print(f"Auto selections (top {len(auto_selections)}): {auto_selections}")

    # Save selections
    gallery.save_selections(human_selections, auto_selections, selections_file)
    print(f"Saved selections to {selections_file}")


if __name__ == "__main__":
    main()
