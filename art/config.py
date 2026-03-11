from dataclasses import dataclass, field
from pathlib import Path


# 8-color palette (RGB tuples) — max perceptual contrast
PALETTE = [
    (0, 0, 0),        # 0  black
    (255, 255, 255),  # 1  white
    (255, 0, 0),      # 2  red
    (0, 0, 255),      # 3  blue
    (0, 180, 0),      # 4  green
    (255, 220, 0),    # 5  yellow
    (255, 0, 255),    # 6  magenta
    (0, 220, 220),    # 7  cyan
]

# Back-compat alias used across the codebase
PALETTE_16 = PALETTE

# Terminal color names for Rich styling (matched to PALETTE indices)
PALETTE_TERM = [
    "black",           # 0
    "white",           # 1
    "red",             # 2
    "blue",            # 3
    "green",           # 4
    "yellow",          # 5
    "magenta",         # 6
    "cyan",            # 7
]


@dataclass
class ArtConfig:
    # Model architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024

    # Grid
    grid_size: int = 16

    # Training
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 20
    train_steps: int = 80

    # GAS
    images_per_gen: int = 12
    select_top: int = 3
    finetune_steps: int = 30
    finetune_lr: float = 1e-4

    # GAS extras
    bootstrap_mix_ratio: float = 0.5
    bootstrap_mix_interval: int = 1
    temp_start: float = 1.0
    temp_end: float = 0.8
    temp_generations: int = 50
    temp_diversity_floor: float = 0.15  # spike temp when batch diversity drops below this

    # Vocab — 8 color tokens (0-7) + special tokens
    n_colors: int = 8
    BOS: int = 8
    EOS: int = 9
    PAD: int = 10
    vocab_size: int = 11

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    bootstrap_dir: Path = field(default_factory=lambda: Path("data/bootstrap"))
    collections_dir: Path = field(default_factory=lambda: Path("data/collections"))

    @property
    def seq_length(self) -> int:
        return self.grid_size * self.grid_size + 2
