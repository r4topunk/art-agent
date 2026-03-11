"""VLM-based critic using a local Ollama vision model."""
from __future__ import annotations

import base64
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image


# Ask for direct numeric scores — avoids unreliable keyword parsing
SCORE_PROMPT = (
    "Rate this pixel art image on three qualities from 0 to 10:\n"
    "1. Appeal: how visually pleasing or interesting is it?\n"
    "2. Composition: does it have clear structure, shapes, or patterns?\n"
    "3. Intent: does it look deliberate rather than random noise?\n\n"
    "Reply with exactly three integers separated by commas. Example: 7,5,8"
)

# Fallback describe prompt (used if numeric parsing fails)
DESCRIBE_PROMPT = (
    "Describe this pixel art briefly: what shapes or patterns do you see?"
)


def _image_to_base64(grid: np.ndarray, scale: int = 8) -> str:
    """Convert a color-indexed grid to a base64-encoded RGB PNG scaled up for the VLM."""
    from art.config import PALETTE
    h, w = grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for ci in range(len(PALETTE)):
        mask = grid == ci
        rgb[mask] = PALETTE[ci]
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


def _parse_numeric_scores(response: str) -> dict[str, float] | None:
    """Parse three comma-separated integers from a VLM response into 0-1 scores."""
    numbers = re.findall(r'\b(\d+)\b', response)
    if len(numbers) < 3:
        return None
    try:
        appeal, composition, intent = [min(10, max(0, int(n))) for n in numbers[:3]]
        return {
            "interest": appeal / 10.0,
            "composition": composition / 10.0,
            "creativity": intent / 10.0,
        }
    except (ValueError, TypeError):
        return None


def score_with_vlm(
    grid: np.ndarray,
    model: str = "moondream",
    base_url: str = "http://localhost:11434",
) -> dict[str, float] | None:
    """Score a single piece using a local VLM via Ollama API."""
    b64 = _image_to_base64(grid)

    response = _call_vlm(b64, SCORE_PROMPT, model=model, base_url=base_url)
    if not response:
        return None

    scores = _parse_numeric_scores(response)
    if scores is None:
        # Fallback: try description-based if numeric parsing fails
        desc = _call_vlm(b64, DESCRIBE_PROMPT, model=model, base_url=base_url)
        if not desc:
            return None
        # Minimal fallback scoring from description length
        words = len(desc.split())
        base = min(0.6, words / 40.0)
        scores = {"interest": base, "composition": base, "creativity": base}
        response = desc

    scores["vlm_description"] = response.strip()
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
    max_workers: int = 2,
) -> list[dict[str, float] | None]:
    """Score a batch of pieces with concurrent requests for pipelining."""
    results: list[dict[str, float] | None] = [None] * len(grids)
    done_count = 0

    def _score_one(idx: int) -> tuple[int, dict[str, float] | None]:
        return idx, score_with_vlm(grids[idx], model=model, base_url=base_url)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_score_one, i): i for i in range(len(grids))}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            done_count += 1
            if on_progress:
                on_progress(done_count, len(grids), result)

    return results


def benchmark_models(
    grid: np.ndarray,
    models: list[str],
    base_url: str = "http://localhost:11434",
    runs: int = 3,
) -> dict[str, dict]:
    """Benchmark multiple VLM models on a single image. Returns timing and score info."""
    b64 = _image_to_base64(grid)
    results = {}

    for model in models:
        # Warm up: load model into memory
        _call_vlm(b64, "hi", model=model, base_url=base_url)

        times = []
        descriptions = []
        for _ in range(runs):
            t0 = time.perf_counter()
            desc = _call_vlm(b64, DESCRIBE_PROMPT, model=model, base_url=base_url)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            if desc:
                descriptions.append(desc)

        avg_time = sum(times) / len(times) if times else 0
        scores = _parse_numeric_scores(descriptions[-1]) if descriptions else {}

        results[model] = {
            "avg_time_s": round(avg_time, 2),
            "min_time_s": round(min(times), 2) if times else 0,
            "max_time_s": round(max(times), 2) if times else 0,
            "sample_description": descriptions[-1].strip() if descriptions else "FAILED",
            "scores": scores,
        }

    return results
