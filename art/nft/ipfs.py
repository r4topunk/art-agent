"""Upload images and metadata to IPFS via Pinata."""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests

PINATA_API = "https://api.pinata.cloud"


def _headers() -> dict[str, str]:
    return {
        "pinata_api_key": os.environ["PINATA_API_KEY"],
        "pinata_secret_api_key": os.environ["PINATA_SECRET_KEY"],
    }


def pin_image(image_path: Path, name: str | None = None) -> str:
    """Pin an image file to IPFS via Pinata.

    Returns the IPFS CID (e.g. ``Qm...``).
    """
    image_path = Path(image_path)
    fname = name or image_path.stem

    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{PINATA_API}/pinning/pinFileToIPFS",
            files={"file": (image_path.name, f)},
            data={
                "pinataMetadata": json.dumps({"name": fname}),
                "pinataOptions": json.dumps({"cidVersion": 1}),
            },
            headers=_headers(),
            timeout=120,
        )
    resp.raise_for_status()
    return resp.json()["IpfsHash"]


def pin_metadata(metadata: dict, name: str) -> str:
    """Pin a JSON metadata object to IPFS via Pinata.

    Returns the IPFS CID.
    """
    resp = requests.post(
        f"{PINATA_API}/pinning/pinJSONToIPFS",
        json={
            "pinataContent": metadata,
            "pinataMetadata": {"name": name},
            "pinataOptions": {"cidVersion": 1},
        },
        headers=_headers(),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["IpfsHash"]


def build_metadata(
    *,
    generation: int,
    image_cid: str,
    scores: dict | None = None,
    piece_index: int | None = None,
) -> dict:
    """Build ERC-721 compliant metadata for an ArtAgent piece."""
    attributes = [
        {"trait_type": "Generation", "value": generation},
    ]
    if piece_index is not None:
        attributes.append({"trait_type": "Piece Index", "value": piece_index})

    if scores:
        for key in ("symmetry", "complexity", "structure", "aesthetics", "composite"):
            if key in scores:
                attributes.append(
                    {
                        "trait_type": key.capitalize(),
                        "display_type": "number",
                        "value": round(scores[key], 3),
                    }
                )

    return {
        "name": f"ArtAgent #{generation}",
        "description": (
            f"Autonomous pixel art from generation {generation}. "
            "Evolved by PixelGPT through genetic art selection."
        ),
        "image": f"ipfs://{image_cid}",
        "attributes": attributes,
    }
