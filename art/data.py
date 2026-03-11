import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from art.config import ArtConfig


def generate_bootstrap_patterns(config: ArtConfig) -> list[np.ndarray]:
    rng = np.random.RandomState(42)
    size = config.grid_size
    patterns: list[np.ndarray] = []

    def empty() -> np.ndarray:
        return np.zeros((size, size), dtype=np.uint8)

    # Horizontal lines ~100
    for spacing in range(1, 5):
        for thickness in range(1, 3):
            for offset in range(spacing):
                grid = empty()
                for row in range(size):
                    if (row - offset) % spacing < thickness:
                        grid[row, :] = 1
                patterns.append(grid.copy())
                if len(patterns) >= 100:
                    break

    # Vertical lines ~100
    vert_count = 0
    for spacing in range(1, 5):
        for thickness in range(1, 3):
            for offset in range(spacing):
                grid = empty()
                for col in range(size):
                    if (col - offset) % spacing < thickness:
                        grid[:, col] = 1
                patterns.append(grid.copy())
                vert_count += 1
                if vert_count >= 100:
                    break
            if vert_count >= 100:
                break
        if vert_count >= 100:
            break

    # Diagonal lines both directions ~100
    diag_count = 0
    for spacing in range(1, 5):
        for offset in range(spacing):
            # top-left to bottom-right
            grid = empty()
            for r in range(size):
                for c in range(size):
                    if (r + c - offset) % spacing == 0:
                        grid[r, c] = 1
            patterns.append(grid.copy())
            diag_count += 1
            if diag_count >= 100:
                break

            # top-right to bottom-left
            grid = empty()
            for r in range(size):
                for c in range(size):
                    if (r - c - offset) % spacing == 0:
                        grid[r, c] = 1
            patterns.append(grid.copy())
            diag_count += 1
            if diag_count >= 100:
                break
        if diag_count >= 100:
            break

    # Horizontal symmetry ~500
    for _ in range(500):
        grid = empty()
        half = rng.randint(0, 2, size=(size, size // 2), dtype=np.uint8)
        grid[:, : size // 2] = half
        grid[:, size // 2 :] = half[:, ::-1]
        patterns.append(grid)

    # Vertical symmetry ~500
    for _ in range(500):
        grid = empty()
        half = rng.randint(0, 2, size=(size // 2, size), dtype=np.uint8)
        grid[: size // 2, :] = half
        grid[size // 2 :, :] = half[::-1, :]
        patterns.append(grid)

    # 4-fold symmetry ~500
    for _ in range(500):
        grid = empty()
        q = rng.randint(0, 2, size=(size // 2, size // 2), dtype=np.uint8)
        grid[: size // 2, : size // 2] = q
        grid[: size // 2, size // 2 :] = q[:, ::-1]
        grid[size // 2 :, : size // 2] = q[::-1, :]
        grid[size // 2 :, size // 2 :] = q[::-1, ::-1]
        patterns.append(grid)

    # Checkerboard variants ~20
    for cell_size in range(1, 5):
        for phase in range(2):
            grid = empty()
            for r in range(size):
                for c in range(size):
                    if ((r // cell_size) + (c // cell_size) + phase) % 2 == 0:
                        grid[r, c] = 1
            patterns.append(grid.copy())
            if len([p for p in patterns if p is not None]) >= 1820:
                pass

    # Filled rectangles ~300
    for _ in range(300):
        grid = empty()
        r1 = rng.randint(0, size - 2)
        c1 = rng.randint(0, size - 2)
        r2 = rng.randint(r1 + 1, size)
        c2 = rng.randint(c1 + 1, size)
        grid[r1:r2, c1:c2] = 1
        patterns.append(grid)

    # Crosses ~200
    for _ in range(200):
        grid = empty()
        cx = rng.randint(2, size - 2)
        cy = rng.randint(2, size - 2)
        arm = rng.randint(1, 4)
        thickness = rng.randint(1, 3)
        r1 = max(0, cx - arm)
        r2 = min(size, cx + arm + 1)
        c1 = max(0, cy - arm)
        c2 = min(size, cy + arm + 1)
        t_half = thickness // 2
        grid[r1:r2, max(0, cy - t_half): min(size, cy + t_half + 1)] = 1
        grid[max(0, cx - t_half): min(size, cx + t_half + 1), c1:c2] = 1
        patterns.append(grid)

    # Diamonds ~200
    for _ in range(200):
        grid = empty()
        cx = rng.randint(0, size)
        cy = rng.randint(0, size)
        radius = rng.randint(2, size // 2 + 1)
        filled = rng.choice([True, False])
        for r in range(size):
            for c in range(size):
                dist = abs(r - cx) + abs(c - cy)
                if filled:
                    if dist <= radius:
                        grid[r, c] = 1
                else:
                    if dist == radius:
                        grid[r, c] = 1
        patterns.append(grid)

    # Borders ~100
    for thickness in range(1, 4):
        for _ in range(33):
            grid = empty()
            grid[:thickness, :] = 1
            grid[-thickness:, :] = 1
            grid[:, :thickness] = 1
            grid[:, -thickness:] = 1
            # vary inner border sometimes
            if rng.random() > 0.5:
                t2 = rng.randint(1, 3)
                inner_start = thickness + t2
                if inner_start < size // 2:
                    grid[inner_start: size - inner_start, inner_start: size - inner_start] = 0
                    grid[inner_start, inner_start: size - inner_start] = 1
                    grid[size - inner_start - 1, inner_start: size - inner_start] = 1
                    grid[inner_start: size - inner_start, inner_start] = 1
                    grid[inner_start: size - inner_start, size - inner_start - 1] = 1
            patterns.append(grid.copy())

    # Random noise with varied densities ~500 (50 per density)
    for density in np.arange(0.1, 1.0, 0.1):
        for _ in range(50):
            grid = (rng.random((size, size)) < density).astype(np.uint8)
            patterns.append(grid)

    # Concentric squares
    for _ in range(100):
        grid = empty()
        step = rng.randint(1, 4)
        for i in range(0, size // 2, step):
            if (i // step) % 2 == 0:
                grid[i: size - i, i: size - i] = 1
                if i + step <= size // 2:
                    grid[i + 1: size - i - 1, i + 1: size - i - 1] = 0
        patterns.append(grid)

    # Stripes at random angles via rotation
    for _ in range(100):
        grid = empty()
        spacing = rng.randint(2, 6)
        offset = rng.randint(0, spacing)
        for r in range(size):
            for c in range(size):
                val = rng.randint(1, 4)
                if (r * val + c - offset) % spacing == 0:
                    grid[r, c] = 1
        patterns.append(grid)

    # Spiral-like patterns
    for _ in range(100):
        grid = empty()
        cx, cy = size // 2, size // 2
        max_r = rng.randint(3, size // 2 + 1)
        for r in range(size):
            for c in range(size):
                dist = max(abs(r - cx), abs(c - cy))
                if dist <= max_r and dist % 2 == 0:
                    grid[r, c] = 1
        patterns.append(grid)

    # Half-filled patterns with random boundary
    for _ in range(100):
        grid = empty()
        split = rng.randint(2, size - 2)
        axis = rng.randint(0, 2)
        if axis == 0:
            grid[:split, :] = 1
        else:
            grid[:, :split] = 1
        patterns.append(grid)

    # Dot grids
    for spacing in range(2, 6):
        for _ in range(25):
            grid = empty()
            offset_r = rng.randint(0, spacing)
            offset_c = rng.randint(0, spacing)
            for r in range(size):
                for c in range(size):
                    if r % spacing == offset_r and c % spacing == offset_c:
                        grid[r, c] = 1
            patterns.append(grid)

    # XOR patterns
    for _ in range(100):
        grid = empty()
        a = rng.randint(1, size)
        b = rng.randint(1, size)
        for r in range(size):
            for c in range(size):
                if (r % a) ^ (c % b):
                    grid[r, c] = 1
        patterns.append(grid)

    # Fill remaining to ~5000 with random symmetric patterns
    target = 5000
    while len(patterns) < target:
        mode = rng.randint(0, 4)
        grid = empty()
        if mode == 0:
            half = rng.randint(0, 2, size=(size, size // 2), dtype=np.uint8)
            grid[:, : size // 2] = half
            grid[:, size // 2 :] = half[:, ::-1]
        elif mode == 1:
            half = rng.randint(0, 2, size=(size // 2, size), dtype=np.uint8)
            grid[: size // 2, :] = half
            grid[size // 2 :, :] = half[::-1, :]
        elif mode == 2:
            q = rng.randint(0, 2, size=(size // 2, size // 2), dtype=np.uint8)
            grid[: size // 2, : size // 2] = q
            grid[: size // 2, size // 2 :] = q[:, ::-1]
            grid[size // 2 :, : size // 2] = q[::-1, :]
            grid[size // 2 :, size // 2 :] = q[::-1, ::-1]
        else:
            density = rng.uniform(0.2, 0.8)
            grid = (rng.random((size, size)) < density).astype(np.uint8)
        patterns.append(grid)

    return patterns[:target]


def save_bootstrap_patterns(patterns: list[np.ndarray], config: ArtConfig) -> None:
    config.bootstrap_dir.mkdir(parents=True, exist_ok=True)
    for i, pattern in enumerate(patterns):
        img_array = (pattern * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode="L")
        img.save(config.bootstrap_dir / f"pattern_{i:04d}.png")


class PixelDataset(Dataset):
    def __init__(self, patterns: list[np.ndarray], config: ArtConfig) -> None:
        self.sequences: list[torch.Tensor] = []
        bos = config.BOS
        eos = config.EOS
        for pattern in patterns:
            pixels = [config.WHITE if p else config.BLACK for p in pattern.flatten().tolist()]
            tokens = [bos] + pixels + [eos]
            self.sequences.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]
