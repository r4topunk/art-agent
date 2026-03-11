from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ArtConfig:
    # Model architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 1024

    # Grid
    grid_size: int = 16

    # Training
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    train_steps: int = 500

    # GAS
    images_per_gen: int = 32
    select_top: int = 12
    finetune_steps: int = 100
    finetune_lr: float = 1e-4

    # GAS extras
    bootstrap_mix_ratio: float = 0.2
    bootstrap_mix_interval: int = 5
    temp_start: float = 1.0
    temp_end: float = 0.7
    temp_generations: int = 50

    # Vocab
    BLACK: int = 0
    WHITE: int = 1
    BOS: int = 2
    EOS: int = 3
    PAD: int = 4
    vocab_size: int = 5

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    bootstrap_dir: Path = field(default_factory=lambda: Path("data/bootstrap"))
    collections_dir: Path = field(default_factory=lambda: Path("data/collections"))

    @property
    def seq_length(self) -> int:
        return self.grid_size * self.grid_size + 2
