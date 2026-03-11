from dataclasses import dataclass, field
from pathlib import Path


# 8-color palette (RGB tuples) — PICO-8 bright/light colors
PALETTE = [
    (0, 0, 0),          # 0  black
    (255, 241, 232),    # 1  warm white/cream  #FFF1E8
    (255, 0, 77),       # 2  bright red        #FF004D
    (255, 163, 0),      # 3  orange            #FFA300
    (255, 236, 39),     # 4  yellow            #FFEC27
    (0, 228, 54),       # 5  bright green      #00E436
    (41, 173, 255),     # 6  sky blue          #29ADFF
    (255, 119, 168),    # 7  pink              #FF77A8
]

# Back-compat alias used across the codebase
PALETTE_16 = PALETTE

# Hex color strings for Rich styling (matched to PALETTE indices)
PALETTE_TERM = [
    "#000000",   # 0  black
    "#fff1e8",   # 1  cream
    "#ff004d",   # 2  red
    "#ffa300",   # 3  orange
    "#ffec27",   # 4  yellow
    "#00e436",   # 5  green
    "#29adff",   # 6  blue
    "#ff77a8",   # 7  pink
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
    train_steps: int = 200
    dropout: float = 0.1

    # GAS
    images_per_gen: int = 36
    select_top: int = 5
    finetune_steps: int = 30
    finetune_lr: float = 1e-4

    # GAS extras
    bootstrap_mix_ratio: float = 0.5          # kept for back-compat; overridden by start/end when decay is active
    bootstrap_mix_ratio_start: float = 0.6    # bootstrap ratio at generation 0
    bootstrap_mix_ratio_end: float = 0.15     # bootstrap ratio at bootstrap_decay_generations
    bootstrap_decay_generations: int = 40     # how many generations to decay over
    bootstrap_mix_interval: int = 1
    temp_start: float = 1.0
    temp_end: float = 0.8
    temp_generations: int = 50
    temp_diversity_floor: float = 0.15  # spike temp when batch diversity drops below this

    # Generation sampling
    top_p: float = 1.0   # nucleus sampling threshold (1.0 = disabled)

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
