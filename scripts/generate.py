#!/usr/bin/env python3
"""Generate images from a trained checkpoint and save a grid PNG."""

import argparse
from pathlib import Path

import numpy as np
import PIL.Image
import torch

from art.config import ArtConfig
from art.model import PixelGPT
from art.tokenizer import PixelTokenizer
from art.utils import create_grid, setup_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pixel art images from a checkpoint")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to checkpoint .pt file")
    parser.add_argument("--n", type=int, default=64, help="Number of images to generate (default: 64)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling; 0 disables (default: 0)")
    parser.add_argument("--output", type=Path, default=Path("generated_grid.png"), help="Output PNG path (default: generated_grid.png)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = setup_device()
    config = ArtConfig()
    model = PixelGPT(config).to(device)
    tokenizer = PixelTokenizer(config)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    extra = checkpoint.get("extra", {})
    gen_num = extra.get("generation", "?")
    print(f"Loaded checkpoint (generation={gen_num}) from {args.checkpoint}")
    print(f"Generating {args.n} images at temperature={args.temperature}, top_k={args.top_k} ...")

    with torch.no_grad():
        tokens = model.generate(
            batch_size=args.n,
            temperature=args.temperature,
            top_k=args.top_k,
            device=str(device),
        )

    images: list[PIL.Image.Image] = []
    for i in range(tokens.shape[0]):
        token_list = tokens[i].tolist()
        img = tokenizer.decode(token_list)
        images.append(img)

    cols = min(8, args.n)
    grid = create_grid(images, cols=cols)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)
    print(f"Saved grid to {args.output}")


if __name__ == "__main__":
    main()
