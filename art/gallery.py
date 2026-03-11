import json
import subprocess
from datetime import datetime
from pathlib import Path

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from art.config import ArtConfig
from art.critic import ArtCritic
from art.tokenizer import PixelTokenizer


class Gallery:
    def __init__(self, config: ArtConfig):
        self.config = config
        self.tokenizer = PixelTokenizer(config)
        self.critic = ArtCritic(config)

    def create_review_grid(
        self, images: list[PIL.Image.Image], scores: list[dict], path: Path
    ) -> Path:
        """Create a labeled grid image with scaled images and scores."""
        path.parent.mkdir(parents=True, exist_ok=True)

        cell_size = 64
        padding = 2
        cols = 8

        n_images = len(images)
        rows = (n_images + cols - 1) // cols

        grid_width = cols * (cell_size + padding) + padding
        grid_height = rows * (cell_size + padding) + padding

        grid = PIL.Image.new("L", (grid_width, grid_height), color=255)
        draw = PIL.ImageDraw.Draw(grid)
        font = PIL.ImageFont.load_default()

        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols

            scaled = img.resize((cell_size, cell_size), PIL.Image.Resampling.LANCZOS)

            x = col * (cell_size + padding) + padding
            y = row * (cell_size + padding) + padding

            grid.paste(scaled, (x, y))

            # Draw index number
            draw.text((x + 2, y + 2), str(idx), fill=0, font=font)

            # Draw composite score
            score = scores[idx]["composite"]
            score_text = f"{score:.2f}"
            draw.text((x + 2, y + cell_size - 10), score_text, fill=0, font=font)

        grid.save(path)
        return path

    def open_preview(self, path: Path) -> None:
        """Open image in Preview.app."""
        subprocess.run(["open", str(path)])

    def prompt_selections(self) -> list[int]:
        """Prompt user for favorite image indices."""
        print("Enter indices of favorites (comma-separated), or 'auto' for critic-only:")
        user_input = input().strip()

        if user_input.lower() == "auto" or user_input == "":
            return []

        try:
            selections = [int(x.strip()) for x in user_input.split(",")]
            return selections
        except ValueError:
            print("Invalid input. Returning empty list.")
            return []

    def save_selections(
        self, selections: list[int], auto_selections: list[int], path: Path
    ) -> None:
        """Save selections to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "human": selections,
            "auto": auto_selections,
            "timestamp": datetime.now().isoformat(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_selections(self, path: Path) -> dict:
        """Load selections from JSON file."""
        with open(path, "r") as f:
            return json.load(f)
