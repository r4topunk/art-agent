from dataclasses import dataclass, field
from pathlib import Path


# 16-color palette (RGB tuples) — CGA/terminal-inspired
PALETTE_16 = [
    (0, 0, 0),        # 0  black
    (128, 0, 0),      # 1  dark red
    (0, 128, 0),      # 2  dark green
    (128, 128, 0),    # 3  dark yellow / olive
    (0, 0, 128),      # 4  dark blue
    (128, 0, 128),    # 5  dark magenta
    (0, 128, 128),    # 6  dark cyan
    (192, 192, 192),  # 7  light gray
    (128, 128, 128),  # 8  dark gray
    (255, 0, 0),      # 9  red
    (0, 255, 0),      # 10 green
    (255, 255, 0),    # 11 yellow
    (0, 0, 255),      # 12 blue
    (255, 0, 255),    # 13 magenta
    (0, 255, 255),    # 14 cyan
    (255, 255, 255),  # 15 white
]

# Terminal color names for Rich styling (matched to PALETTE_16 indices)
PALETTE_TERM = [
    "black",           # 0
    "dark_red",        # 1
    "green4",          # 2
    "yellow4",         # 3
    "dark_blue",       # 4
    "dark_magenta",    # 5
    "dark_cyan",       # 6
    "grey74",          # 7
    "grey50",          # 8
    "red",             # 9
    "green",           # 10
    "yellow",          # 11
    "blue",            # 12
    "magenta",         # 13
    "cyan",            # 14
    "white",           # 15
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
    images_per_gen: int = 32
    select_top: int = 12
    finetune_steps: int = 80
    finetune_lr: float = 1e-4

    # GAS extras
    bootstrap_mix_ratio: float = 0.2
    bootstrap_mix_interval: int = 5
    temp_start: float = 1.0
    temp_end: float = 0.7
    temp_generations: int = 50

    # Vocab — 16 color tokens (0-15) + special tokens
    n_colors: int = 16
    BOS: int = 16
    EOS: int = 17
    PAD: int = 18
    vocab_size: int = 19

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    bootstrap_dir: Path = field(default_factory=lambda: Path("data/bootstrap"))
    collections_dir: Path = field(default_factory=lambda: Path("data/collections"))

    @property
    def seq_length(self) -> int:
        return self.grid_size * self.grid_size + 2
