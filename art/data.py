import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from art.config import ArtConfig, PALETTE_16


def generate_bootstrap_patterns(config: ArtConfig, on_progress=None) -> list[np.ndarray]:
    """Generate bootstrap patterns as grids of color indices (0..n_colors-1)."""
    rng = np.random.RandomState(42)
    size = config.grid_size
    nc = config.n_colors
    patterns: list[np.ndarray] = []
    _last_report = [0]

    def _maybe_report(category: str = ""):
        n = len(patterns)
        if on_progress and n - _last_report[0] >= 50:
            on_progress(n, 5000, category)
            _last_report[0] = n

    def empty() -> np.ndarray:
        return np.zeros((size, size), dtype=np.uint8)

    def rand_fg() -> int:
        return rng.randint(1, nc)  # any color except black

    def rand_color() -> int:
        return rng.randint(0, nc)

    def rand_pair() -> tuple[int, int]:
        """Return (bg, fg) pair of different colors."""
        bg = rand_color()
        fg = rand_color()
        while fg == bg:
            fg = rand_color()
        return bg, fg

    # --- Geometric patterns with 2 colors ---

    # Horizontal lines ~80
    for spacing in range(1, 5):
        for thickness in range(1, 3):
            for offset in range(spacing):
                bg, fg = rand_pair()
                grid = np.full((size, size), bg, dtype=np.uint8)
                for row in range(size):
                    if (row - offset) % spacing < thickness:
                        grid[row, :] = fg
                patterns.append(grid.copy())
                if len(patterns) >= 80:
                    break

    # Vertical lines ~80
    count = 0
    for spacing in range(1, 5):
        for thickness in range(1, 3):
            for offset in range(spacing):
                bg, fg = rand_pair()
                grid = np.full((size, size), bg, dtype=np.uint8)
                for col in range(size):
                    if (col - offset) % spacing < thickness:
                        grid[:, col] = fg
                patterns.append(grid.copy())
                count += 1
                if count >= 80:
                    break
            if count >= 80:
                break
        if count >= 80:
            break

    # Diagonal lines ~80
    count = 0
    for spacing in range(1, 5):
        for offset in range(spacing):
            bg, fg = rand_pair()
            grid = np.full((size, size), bg, dtype=np.uint8)
            for r in range(size):
                for c in range(size):
                    if (r + c - offset) % spacing == 0:
                        grid[r, c] = fg
            patterns.append(grid.copy())
            count += 1
            if count >= 80:
                break

            bg, fg = rand_pair()
            grid = np.full((size, size), bg, dtype=np.uint8)
            for r in range(size):
                for c in range(size):
                    if (r - c - offset) % spacing == 0:
                        grid[r, c] = fg
            patterns.append(grid.copy())
            count += 1
            if count >= 80:
                break
        if count >= 80:
            break

    _maybe_report("lines")

    # --- Symmetry patterns with multiple colors ---

    # Horizontal symmetry ~400
    for _ in range(400):
        n_used = rng.randint(2, min(6, nc + 1))
        colors = rng.choice(nc, size=n_used, replace=False)
        half = colors[rng.randint(0, n_used, size=(size, size // 2))]
        grid = np.zeros((size, size), dtype=np.uint8)
        grid[:, :size // 2] = half
        grid[:, size // 2:] = half[:, ::-1]
        patterns.append(grid)

    # Vertical symmetry ~400
    for _ in range(400):
        n_used = rng.randint(2, min(6, nc + 1))
        colors = rng.choice(nc, size=n_used, replace=False)
        half = colors[rng.randint(0, n_used, size=(size // 2, size))]
        grid = np.zeros((size, size), dtype=np.uint8)
        grid[:size // 2, :] = half
        grid[size // 2:, :] = half[::-1, :]
        patterns.append(grid)

    # 4-fold symmetry ~400
    for _ in range(400):
        n_used = rng.randint(2, min(6, nc + 1))
        colors = rng.choice(nc, size=n_used, replace=False)
        q = colors[rng.randint(0, n_used, size=(size // 2, size // 2))]
        grid = np.zeros((size, size), dtype=np.uint8)
        grid[:size // 2, :size // 2] = q
        grid[:size // 2, size // 2:] = q[:, ::-1]
        grid[size // 2:, :size // 2] = q[::-1, :]
        grid[size // 2:, size // 2:] = q[::-1, ::-1]
        patterns.append(grid)

    _maybe_report("symmetry")

    # Checkerboard variants ~20
    for cell_size in range(1, 5):
        for phase in range(2):
            bg, fg = rand_pair()
            grid = np.full((size, size), bg, dtype=np.uint8)
            for r in range(size):
                for c in range(size):
                    if ((r // cell_size) + (c // cell_size) + phase) % 2 == 0:
                        grid[r, c] = fg
            patterns.append(grid.copy())

    # Multi-color checkerboard ~40
    for _ in range(40):
        cell_size = rng.randint(1, 5)
        n_cols = rng.randint(2, 5)
        colors = rng.choice(nc, size=n_cols, replace=False)
        grid = empty()
        for r in range(size):
            for c in range(size):
                grid[r, c] = colors[((r // cell_size) + (c // cell_size)) % n_cols]
        patterns.append(grid)

    _maybe_report("checkerboards")

    # Filled rectangles with random colors ~200
    for _ in range(200):
        bg, fg = rand_pair()
        grid = np.full((size, size), bg, dtype=np.uint8)
        r1 = rng.randint(0, size - 2)
        c1 = rng.randint(0, size - 2)
        r2 = rng.randint(r1 + 1, size)
        c2 = rng.randint(c1 + 1, size)
        grid[r1:r2, c1:c2] = fg
        patterns.append(grid)

    # Crosses ~150
    for _ in range(150):
        bg, fg = rand_pair()
        grid = np.full((size, size), bg, dtype=np.uint8)
        cx = rng.randint(2, size - 2)
        cy = rng.randint(2, size - 2)
        arm = rng.randint(1, 4)
        thickness = rng.randint(1, 3)
        r1, r2 = max(0, cx - arm), min(size, cx + arm + 1)
        c1, c2 = max(0, cy - arm), min(size, cy + arm + 1)
        t = thickness // 2
        grid[r1:r2, max(0, cy - t):min(size, cy + t + 1)] = fg
        grid[max(0, cx - t):min(size, cx + t + 1), c1:c2] = fg
        patterns.append(grid)

    _maybe_report("shapes")

    # Diamonds ~150
    for _ in range(150):
        bg, fg = rand_pair()
        grid = np.full((size, size), bg, dtype=np.uint8)
        cx, cy = rng.randint(0, size), rng.randint(0, size)
        radius = rng.randint(2, size // 2 + 1)
        filled = rng.choice([True, False])
        for r in range(size):
            for c in range(size):
                dist = abs(r - cx) + abs(c - cy)
                if filled and dist <= radius:
                    grid[r, c] = fg
                elif not filled and dist == radius:
                    grid[r, c] = fg
        patterns.append(grid)

    # Borders ~80
    for thickness in range(1, 4):
        for _ in range(27):
            bg, fg = rand_pair()
            grid = np.full((size, size), bg, dtype=np.uint8)
            grid[:thickness, :] = fg
            grid[-thickness:, :] = fg
            grid[:, :thickness] = fg
            grid[:, -thickness:] = fg
            patterns.append(grid.copy())

    _maybe_report("diamonds")

    # Concentric squares with alternating colors ~100
    for _ in range(100):
        colors = rng.choice(nc, size=rng.randint(2, 5), replace=False)
        grid = np.full((size, size), colors[0], dtype=np.uint8)
        step = rng.randint(1, 4)
        for i in range(0, size // 2, step):
            c = colors[(i // step) % len(colors)]
            grid[i:size - i, i:size - i] = c
        patterns.append(grid)

    _maybe_report("concentric")

    # Gradient-like patterns (color ramps) ~200
    for _ in range(200):
        n_cols = rng.randint(3, min(8, nc + 1))
        colors = rng.choice(nc, size=n_cols, replace=False)
        axis = rng.randint(0, 2)
        grid = empty()
        for i in range(size):
            ci = int(i / size * n_cols)
            ci = min(ci, n_cols - 1)
            if axis == 0:
                grid[i, :] = colors[ci]
            else:
                grid[:, i] = colors[ci]
        patterns.append(grid)

    _maybe_report("gradients")

    # Random noise with varied number of colors ~400
    for n_used in range(2, 8):
        for _ in range(60):
            colors = rng.choice(nc, size=n_used, replace=False)
            grid = colors[rng.randint(0, n_used, size=(size, size))].astype(np.uint8)
            patterns.append(grid)

    _maybe_report("noise")

    # Stripes with 3+ colors ~100
    for _ in range(100):
        n_cols = rng.randint(2, 5)
        colors = rng.choice(nc, size=n_cols, replace=False)
        spacing = rng.randint(1, 5)
        axis = rng.randint(0, 2)
        grid = empty()
        for i in range(size):
            c = colors[(i // spacing) % n_cols]
            if axis == 0:
                grid[i, :] = c
            else:
                grid[:, i] = c
        patterns.append(grid)

    # XOR patterns with colors ~100
    for _ in range(100):
        n_cols = rng.randint(2, 5)
        colors = rng.choice(nc, size=n_cols, replace=False)
        a = rng.randint(1, size)
        b = rng.randint(1, size)
        grid = empty()
        for r in range(size):
            for c in range(size):
                grid[r, c] = colors[((r % a) ^ (c % b)) % n_cols]
        patterns.append(grid)

    # Dot grids ~80
    for spacing in range(2, 6):
        for _ in range(20):
            bg, fg = rand_pair()
            grid = np.full((size, size), bg, dtype=np.uint8)
            off_r, off_c = rng.randint(0, spacing), rng.randint(0, spacing)
            for r in range(size):
                for c in range(size):
                    if r % spacing == off_r and c % spacing == off_c:
                        grid[r, c] = fg
            patterns.append(grid)

    _maybe_report("xor")

    # Fill remaining with random symmetric colored patterns
    target = 5000
    while len(patterns) < target:
        mode = rng.randint(0, 5)
        n_used = rng.randint(2, min(6, nc + 1))
        colors = rng.choice(nc, size=n_used, replace=False)

        if mode == 0:  # H-symmetry
            half = colors[rng.randint(0, n_used, size=(size, size // 2))]
            grid = np.zeros((size, size), dtype=np.uint8)
            grid[:, :size // 2] = half
            grid[:, size // 2:] = half[:, ::-1]
        elif mode == 1:  # V-symmetry
            half = colors[rng.randint(0, n_used, size=(size // 2, size))]
            grid = np.zeros((size, size), dtype=np.uint8)
            grid[:size // 2, :] = half
            grid[size // 2:, :] = half[::-1, :]
        elif mode == 2:  # 4-fold
            q = colors[rng.randint(0, n_used, size=(size // 2, size // 2))]
            grid = np.zeros((size, size), dtype=np.uint8)
            grid[:size // 2, :size // 2] = q
            grid[:size // 2, size // 2:] = q[:, ::-1]
            grid[size // 2:, :size // 2] = q[::-1, :]
            grid[size // 2:, size // 2:] = q[::-1, ::-1]
        elif mode == 3:  # gradient
            n_cols = rng.randint(2, max(3, n_used + 1))
            grid = np.zeros((size, size), dtype=np.uint8)
            for i in range(size):
                ci = min(int(i / size * n_cols), n_cols - 1)
                if rng.random() > 0.5:
                    grid[i, :] = colors[ci]
                else:
                    grid[:, i] = colors[ci]
        else:  # random mix
            grid = colors[rng.randint(0, n_used, size=(size, size))].astype(np.uint8)
        patterns.append(grid)
        _maybe_report("random")

    return patterns[:target]


def save_bootstrap_patterns(patterns: list[np.ndarray], config: ArtConfig, on_progress=None) -> None:
    """Save bootstrap patterns as indexed-color PNGs."""
    config.bootstrap_dir.mkdir(parents=True, exist_ok=True)
    total = len(patterns)
    for i, pattern in enumerate(patterns):
        # Convert color indices to RGB
        h, w = pattern.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for ci in range(config.n_colors):
            mask = pattern == ci
            rgb[mask] = PALETTE_16[ci]
        img = Image.fromarray(rgb, mode="RGB")
        img.save(config.bootstrap_dir / f"pattern_{i:04d}.png")
        if on_progress and (i + 1) % 200 == 0:
            on_progress(i + 1, total)


class PixelDataset(Dataset):
    def __init__(self, patterns: list[np.ndarray], config: ArtConfig) -> None:
        self.sequences: list[torch.Tensor] = []
        bos = config.BOS
        eos = config.EOS
        for pattern in patterns:
            # Pattern is already a grid of color indices (0..n_colors-1)
            pixels = pattern.flatten().tolist()
            # Clamp to valid range
            pixels = [max(0, min(config.n_colors - 1, p)) for p in pixels]
            tokens = [bos] + pixels + [eos]
            self.sequences.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]
