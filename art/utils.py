from pathlib import Path
import PIL.Image
import torch
from art.config import ArtConfig


def save_image(image: PIL.Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def load_image(path: Path) -> PIL.Image.Image:
    img = PIL.Image.open(path)
    return img.convert("L")


def create_grid(images: list[PIL.Image.Image], cols: int = 8, cell_size: int = 64) -> PIL.Image.Image:
    n_images = len(images)
    rows = (n_images + cols - 1) // cols

    padding = 2
    grid_width = cols * (cell_size + padding) + padding
    grid_height = rows * (cell_size + padding) + padding

    grid = PIL.Image.new("L", (grid_width, grid_height), color=255)

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols

        scaled = img.resize((cell_size, cell_size), PIL.Image.Resampling.LANCZOS)

        x = col * (cell_size + padding) + padding
        y = row * (cell_size + padding) + padding

        grid.paste(scaled, (x, y))

    return grid


def setup_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dirs(config: ArtConfig) -> None:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.bootstrap_dir.mkdir(parents=True, exist_ok=True)
    config.collections_dir.mkdir(parents=True, exist_ok=True)
