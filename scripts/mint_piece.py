#!/usr/bin/env python3
"""Mint a single piece as an NFT.

Usage:
    python scripts/mint_piece.py --generation 42 --piece 7
    python scripts/mint_piece.py --image path/to/image.png --generation 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from art.nft.ipfs import build_metadata, pin_image, pin_metadata
from art.nft.mint import Minter


def main():
    parser = argparse.ArgumentParser(description="Mint an ArtAgent piece as NFT")
    parser.add_argument("--generation", type=int, required=True, help="Generation number")
    parser.add_argument("--piece", type=int, help="Piece index within generation")
    parser.add_argument("--image", type=str, help="Direct path to image (overrides --piece)")
    parser.add_argument("--to", type=str, help="Recipient address (default: deployer)")
    parser.add_argument("--dry-run", action="store_true", help="Upload to IPFS but skip minting")
    args = parser.parse_args()

    # Resolve image path
    if args.image:
        image_path = Path(args.image)
    elif args.piece is not None:
        image_path = Path(f"data/collections/gen_{args.generation:03d}/pieces/piece_{args.piece:04d}.png")
    else:
        parser.error("Provide either --piece or --image")
        return

    if not image_path.exists():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Load scores if available
    scores = None
    scores_path = Path(f"data/collections/gen_{args.generation:03d}/scores.json")
    if scores_path.exists() and args.piece is not None:
        with open(scores_path) as f:
            all_scores = json.load(f)
        if args.piece < len(all_scores):
            scores = all_scores[args.piece]

    # 1. Pin image to IPFS
    print(f"Pinning image to IPFS: {image_path}")
    image_cid = pin_image(image_path, name=f"artagent_gen{args.generation}")
    print(f"  Image CID: {image_cid}")
    print(f"  URL: ipfs://{image_cid}")

    # 2. Build and pin metadata
    metadata = build_metadata(
        generation=args.generation,
        image_cid=image_cid,
        scores=scores,
        piece_index=args.piece,
    )
    print(f"Pinning metadata to IPFS...")
    metadata_cid = pin_metadata(metadata, name=f"artagent_gen{args.generation}_metadata")
    metadata_uri = f"ipfs://{metadata_cid}"
    print(f"  Metadata CID: {metadata_cid}")
    print(f"  URI: {metadata_uri}")

    if args.dry_run:
        print("\n[dry-run] Skipping on-chain mint.")
        print(json.dumps(metadata, indent=2))
        return

    # 3. Mint on-chain
    print(f"Minting on-chain...")
    minter = Minter()
    tx_hash = minter.mint(
        to=args.to,
        generation=args.generation,
        metadata_uri=metadata_uri,
    )
    print(f"  TX: {tx_hash}")

    print("Waiting for confirmation...")
    receipt = minter.wait_for_receipt(tx_hash)
    print(f"  Confirmed in block {receipt['blockNumber']}")
    print(f"  Gas used: {receipt['gasUsed']}")


if __name__ == "__main__":
    main()
