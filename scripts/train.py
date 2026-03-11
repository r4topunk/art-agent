"""Standalone bootstrap training script for ArtAgent.

Run with:
    uv run python scripts/train.py [--steps N] [--output-dir PATH]
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path when invoked directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from art.config import ArtConfig
from art.data import PixelDataset, generate_bootstrap_patterns, save_bootstrap_patterns
from art.model import PixelGPT
from art.tokenizer import PixelTokenizer
from art.trainer import Trainer
from art.utils import create_grid, ensure_dirs, save_image, setup_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PixelGPT on bootstrap patterns.")
    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Number of training steps (default: 5000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help='Output directory for checkpoints and generated images (default: "data")',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Config & device
    config = ArtConfig(
        data_dir=Path(args.output_dir),
        bootstrap_dir=Path(args.output_dir) / "bootstrap",
        collections_dir=Path(args.output_dir) / "collections",
        train_steps=args.steps,
    )
    device = setup_device()
    print(f"Using device: {device}")

    ensure_dirs(config)

    # Bootstrap patterns
    print("Generating bootstrap patterns...")
    patterns = generate_bootstrap_patterns(config)
    print(f"Generated {len(patterns)} bootstrap patterns")

    print(f"Saving bootstrap patterns to {config.bootstrap_dir}...")
    save_bootstrap_patterns(patterns, config)

    # Dataset
    dataset = PixelDataset(patterns, config)
    print(f"Dataset size: {len(dataset)} sequences")

    # Model
    model = PixelGPT(config).to(device)
    param_count = model.count_parameters()
    print(f"PixelGPT parameters: {param_count:,}")

    # Training
    trainer = Trainer(model, config, device)
    print(f"Starting training for {args.steps} steps...")
    losses = trainer.train(dataset, steps=args.steps)

    final_loss = losses[-1] if losses else float("nan")
    print(f"\nTraining complete. Final loss: {final_loss:.4f}")

    # Generate 64 images and save grid
    print("Generating 64 sample images...")
    model.eval()
    tokenizer = PixelTokenizer(config)

    generated_tokens = model.generate(
        batch_size=64,
        temperature=1.0,
        device=str(device),
    )

    images = []
    for i in range(generated_tokens.shape[0]):
        token_list = generated_tokens[i].tolist()
        img = tokenizer.decode(token_list)
        images.append(img)

    grid = create_grid(images, cols=8, cell_size=64)
    grid_path = Path(args.output_dir) / "bootstrap_results.png"
    save_image(grid, grid_path)
    print(f"Sample grid saved to {grid_path}")

    # Save checkpoint
    checkpoint_path = Path(args.output_dir) / "bootstrap_checkpoint.pt"
    trainer.save_checkpoint(
        checkpoint_path,
        extra={
            "steps": args.steps,
            "final_loss": final_loss,
            "param_count": param_count,
        },
    )
    print(f"Checkpoint saved to {checkpoint_path}")

    # Summary
    print("\n--- Summary ---")
    print(f"Final loss:       {final_loss:.4f}")
    print(f"Images saved to:  {grid_path.resolve()}")
    print(f"Checkpoint:       {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
