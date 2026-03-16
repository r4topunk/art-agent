#!/usr/bin/env python3
"""Export the last N generated pieces as a static JSON for web deployment.

Usage:
    python scripts/export_pieces.py [--count 3333] [--out art/web/public/data/pieces.json]

Reads PNGs from data/collections/all/, converts RGB→palette index,
outputs a JSON file with the format the frontend expects:
    { "pieces": [ [[idx,...], ...16 rows], ...N pieces ] }
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# PICO-8 palette (must match art/config.py PALETTE)
PALETTE = [
    (0, 0, 0),
    (255, 241, 232),
    (255, 0, 77),
    (255, 163, 0),
    (255, 236, 39),
    (0, 228, 54),
    (41, 173, 255),
    (255, 119, 168),
]

# Build a lookup from RGB tuple → palette index
_RGB_TO_IDX = {rgb: i for i, rgb in enumerate(PALETTE)}


def rgb_to_palette(img_array: np.ndarray) -> list[list[int]]:
    """Convert a 16x16x3 uint8 array to a 16x16 list of palette indices."""
    rows, cols = img_array.shape[:2]
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            rgb = tuple(img_array[r, c, :3])
            # Exact match first, then nearest if noisy
            idx = _RGB_TO_IDX.get(rgb)
            if idx is None:
                # Find closest palette color by Euclidean distance
                dists = [sum((a - b) ** 2 for a, b in zip(rgb, p)) for p in PALETTE]
                idx = dists.index(min(dists))
            row.append(idx)
        grid.append(row)
    return grid


def main():
    parser = argparse.ArgumentParser(description="Export pieces as static JSON")
    parser.add_argument("--count", type=int, default=3333, help="Number of most recent pieces to export")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    all_dir = root / "data" / "collections" / "all"
    if not all_dir.exists():
        print(f"Error: {all_dir} not found", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else root / "art" / "web" / "public" / "data" / "pieces.json"

    # Get sorted PNGs, take last N
    pngs = sorted(all_dir.glob("*.png"))
    total = len(pngs)
    selected = pngs[-args.count:] if total >= args.count else pngs
    print(f"Found {total} pieces, exporting last {len(selected)}")

    pieces = []
    for i, png_path in enumerate(selected):
        img = Image.open(png_path).convert("RGB")
        arr = np.array(img)
        grid = rgb_to_palette(arr)
        pieces.append(grid)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(selected)} converted...")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"pieces": pieces}, f, separators=(",", ":"))

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Exported {len(pieces)} pieces → {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
