"""Export best pieces from the latest generation as a single grid PNG for review."""
import sys
import json
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from art.config import ArtConfig, PALETTE_16


def find_latest_gen(config: ArtConfig) -> int | None:
    if not config.collections_dir.exists():
        return None
    gens = []
    for item in config.collections_dir.iterdir():
        if item.is_dir() and item.name.startswith("gen_"):
            try:
                gens.append(int(item.name.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return max(gens) if gens else None


def export_grid(gen: int | None = None, top_n: int = 8, scale: int = 16):
    config = ArtConfig()

    if gen is None:
        gen = find_latest_gen(config)
        if gen is None:
            print("No generations found.")
            return

    gen_dir = config.collections_dir / f"gen_{gen:03d}"
    scores_path = gen_dir / "scores.json"
    pieces_dir = gen_dir / "pieces"

    if not scores_path.exists():
        print(f"No scores.json in {gen_dir}")
        return

    with open(scores_path) as f:
        scores = json.load(f)

    # Load selections to map score indices to piece file indices
    selections_path = gen_dir / "selections.json"
    if selections_path.exists():
        with open(selections_path) as f:
            selections = json.load(f)
    else:
        # Legacy: scores indexed by piece number directly
        selections = list(range(len(scores)))

    # Rank by composite score (scores are already only for selected pieces)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i].get("composite", 0), reverse=True)
    best = ranked[:top_n]

    images = []
    labels = []
    for rank_idx in best:
        piece_idx = selections[rank_idx] if rank_idx < len(selections) else rank_idx
        img_path = pieces_dir / f"piece_{piece_idx:04d}.png"
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            score = scores[rank_idx].get("composite", 0)
            labels.append(f"#{piece_idx} ({score:.3f})")

    if not images:
        print("No piece images found.")
        return

    # Create grid
    cols = min(4, len(images))
    rows = (len(images) + cols - 1) // cols
    piece_size = images[0].size[0]
    scaled = piece_size * scale
    padding = 4

    grid_w = cols * scaled + (cols + 1) * padding
    grid_h = rows * scaled + (rows + 1) * padding

    canvas = Image.new("RGB", (grid_w, grid_h), color=(10, 10, 15))

    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        x = padding + c * (scaled + padding)
        y = padding + r * (scaled + padding)
        scaled_img = img.resize((scaled, scaled), Image.NEAREST)
        canvas.paste(scaled_img, (x, y))

    out_path = Path("images/best_pieces.png")
    out_path.parent.mkdir(exist_ok=True)
    canvas.save(out_path)

    print(f"Exported top {len(images)} pieces from gen {gen} to {out_path}")
    print(f"Scores: {', '.join(labels)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", type=int, default=None, help="Generation number (default: latest)")
    parser.add_argument("--top", type=int, default=8, help="Number of best pieces to export")
    parser.add_argument("--scale", type=int, default=16, help="Scale factor for each piece")
    args = parser.parse_args()
    export_grid(gen=args.gen, top_n=args.top, scale=args.scale)
