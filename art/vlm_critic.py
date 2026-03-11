"""VLM-based critic using a local Ollama vision model."""
from __future__ import annotations

import base64
import io
import json
import re
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image


# Keep it short — small VLMs choke on long prompts
DESCRIBE_PROMPT = "What do you see in this pixel art?"

QUALITY_PROMPTS = [
    ("Has clear structure or recognizable shapes (not random noise)?", "structure"),
    ("Is this visually interesting or appealing?", "interest"),
    ("Does this look like intentional art rather than a glitch?", "intentional"),
]


def _image_to_base64(grid: np.ndarray, scale: int = 16) -> str:
    """Convert a color-indexed grid to a base64-encoded RGB PNG scaled up for the VLM."""
    from art.config import PALETTE_16
    h, w = grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for ci in range(16):
        mask = grid == ci
        rgb[mask] = PALETTE_16[ci]
    img = Image.fromarray(rgb)
    img = img.resize((w * scale, h * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _call_vlm(
    b64_image: str,
    prompt: str,
    model: str = "moondream",
    base_url: str = "http://localhost:11434",
) -> str | None:
    """Call Ollama chat API and return the response text."""
    payload = json.dumps({
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64_image],
            }
        ],
        "stream": False,
    }).encode("utf-8")

    req = Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("message", {}).get("content", "")
    except Exception:
        return None


def _description_to_score(description: str) -> dict[str, float]:
    """Extract quality signals from a free-text description."""
    desc = description.lower()

    # Positive signals
    structure_words = ["shape", "pattern", "square", "circle", "line", "cross",
                       "diamond", "triangle", "grid", "border", "frame",
                       "symmetric", "geometric", "design", "art", "pixel"]
    interest_words = ["interesting", "complex", "detailed", "intricate",
                      "colorful", "contrast", "unique", "striking",
                      "beautiful", "creative", "abstract"]
    # Negative signals
    boring_words = ["blank", "empty", "nothing", "black", "plain",
                    "simple dot", "single pixel", "noise", "random",
                    "static", "just a"]

    structure_hits = sum(1 for w in structure_words if w in desc)
    interest_hits = sum(1 for w in interest_words if w in desc)
    boring_hits = sum(1 for w in boring_words if w in desc)

    # Convert to 0-1 scores
    structure = min(1.0, structure_hits / 3.0)
    interest = min(1.0, interest_hits / 2.0)
    penalty = min(0.5, boring_hits * 0.15)

    # Description length as a weak proxy for complexity
    word_count = len(desc.split())
    verbosity_bonus = min(0.3, word_count / 50.0)

    return {
        "interest": max(0, interest + verbosity_bonus - penalty),
        "composition": max(0, structure - penalty),
        "creativity": max(0, (interest + structure) / 2.0 + verbosity_bonus - penalty),
    }


def score_with_vlm(
    grid: np.ndarray,
    model: str = "moondream",
    base_url: str = "http://localhost:11434",
) -> dict[str, float] | None:
    """Score a single piece using a local VLM via Ollama API.

    Uses description-based scoring since small VLMs struggle with
    structured numeric output.
    """
    b64 = _image_to_base64(grid)

    description = _call_vlm(b64, DESCRIBE_PROMPT, model=model, base_url=base_url)
    if not description:
        return None

    scores = _description_to_score(description)

    scores["vlm_description"] = description.strip()
    scores["vlm_composite"] = (
        0.4 * scores["interest"]
        + 0.3 * scores["composition"]
        + 0.3 * scores["creativity"]
    )
    return scores


def score_batch_with_vlm(
    grids: list[np.ndarray],
    model: str = "moondream",
    base_url: str = "http://localhost:11434",
    on_progress=None,
) -> list[dict[str, float] | None]:
    """Score a batch of pieces. Returns list aligned with input (None for failures)."""
    results = []
    for i, grid in enumerate(grids):
        result = score_with_vlm(grid, model=model, base_url=base_url)
        results.append(result)
        if on_progress:
            on_progress(i + 1, len(grids), result)
    return results
