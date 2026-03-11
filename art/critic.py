import numpy as np
from art.config import ArtConfig


def symmetry_score(grid: np.ndarray) -> float:
    g = grid.astype(np.int16)
    h_sym = 1 - np.mean(np.abs(g - np.fliplr(g)))
    v_sym = 1 - np.mean(np.abs(g - np.flipud(g)))
    r180_sym = 1 - np.mean(np.abs(g - np.rot90(g, 2)))
    r90_sym = 1 - np.mean(np.abs(g - np.rot90(g)))
    return float(np.mean([h_sym, v_sym, r180_sym, r90_sym]))


def complexity_score(grid: np.ndarray) -> float:
    g = grid.astype(np.int16)
    size = g.shape[0]
    density = float(np.mean(g))

    # Hard penalty for too-empty or too-full images
    # Peak at 30-70% density, drops to 0 below 15% or above 85%
    if density < 0.15 or density > 0.85:
        density_score = 0.0
    elif density < 0.3:
        density_score = (density - 0.15) / 0.15  # ramp 0→1
    elif density > 0.7:
        density_score = (0.85 - density) / 0.15  # ramp 1→0
    else:
        density_score = 1.0

    edge_count = (
        np.sum(np.abs(np.diff(g, axis=0))) +
        np.sum(np.abs(np.diff(g, axis=1)))
    )
    max_edges = 2 * size * (size - 1)
    edge_score = float(edge_count / max_edges)

    block_size = max(1, size // 4)
    n_blocks = size // block_size
    entropy_vals = []
    for i in range(n_blocks):
        for j in range(n_blocks):
            block = grid[block_size*i:block_size*(i+1), block_size*j:block_size*(j+1)]
            p = float(np.mean(block))
            if p > 0 and p < 1:
                h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
            else:
                h = 0.0
            entropy_vals.append(h)
    entropy_score = float(np.mean(entropy_vals))
    entropy_score = min(entropy_score, 1.0)

    return float(np.mean([density_score, edge_score, entropy_score]))


def structure_score(grid: np.ndarray) -> float:
    """Reward structured patterns: lines, repeating motifs, coherent shapes.
    Penalize random noise and empty space equally."""
    size = grid.shape[0]
    g = grid.astype(np.int16)

    # Row and column coherence: how many rows/cols have a consistent pattern
    row_densities = np.mean(grid, axis=1)
    col_densities = np.mean(grid, axis=0)
    # Reward rows that are not all-same but have structure (not random)
    row_autocorr = 0.0
    col_autocorr = 0.0
    for axis_vals in [row_densities, col_densities]:
        if len(axis_vals) > 1:
            mean_v = np.mean(axis_vals)
            centered = axis_vals - mean_v
            var = np.sum(centered ** 2)
            if var > 0:
                # Lag-1 autocorrelation: adjacent rows/cols should be related
                autocorr = np.sum(centered[:-1] * centered[1:]) / var
                if axis_vals is row_densities:
                    row_autocorr = float(np.clip(autocorr, 0, 1))
                else:
                    col_autocorr = float(np.clip(autocorr, 0, 1))

    # 2x2 block variety: count distinct 2x2 patterns (rewards structure over noise)
    blocks_2x2 = set()
    for r in range(size - 1):
        for c in range(size - 1):
            block = (grid[r, c], grid[r, c+1], grid[r+1, c], grid[r+1, c+1])
            blocks_2x2.add(block)
    # Too few patterns = boring, too many = noisy. Sweet spot: 6-12 out of 16 possible
    n_patterns = len(blocks_2x2)
    if n_patterns <= 3:
        pattern_score = n_patterns / 6.0
    elif n_patterns <= 12:
        pattern_score = 1.0
    else:
        pattern_score = max(0, 1 - (n_patterns - 12) / 4)

    # Contiguous region sizes: reward medium-sized regions (not dust, not blobs)
    visited = np.zeros_like(grid, dtype=bool)
    regions = []

    def flood_fill(r: int, c: int, val: int) -> int:
        count = 0
        stack = [(r, c)]
        while stack:
            r2, c2 = stack.pop()
            if r2 < 0 or r2 >= size or c2 < 0 or c2 >= size:
                continue
            if visited[r2, c2] or grid[r2, c2] != val:
                continue
            visited[r2, c2] = True
            count += 1
            stack.extend([(r2+1, c2), (r2-1, c2), (r2, c2+1), (r2, c2-1)])
        return count

    for i in range(size):
        for j in range(size):
            if not visited[i, j]:
                region_size = flood_fill(i, j, grid[i, j])
                regions.append(region_size)

    # Reward having multiple medium-sized regions (3-40 pixels each)
    white_regions = []
    visited_w = np.zeros_like(grid, dtype=bool)
    for i in range(size):
        for j in range(size):
            if grid[i, j] == 1 and not visited_w[i, j]:
                count = 0
                stack = [(i, j)]
                while stack:
                    r2, c2 = stack.pop()
                    if r2 < 0 or r2 >= size or c2 < 0 or c2 >= size:
                        continue
                    if visited_w[r2, c2] or grid[r2, c2] != 1:
                        continue
                    visited_w[r2, c2] = True
                    count += 1
                    stack.extend([(r2+1, c2), (r2-1, c2), (r2, c2+1), (r2, c2-1)])
                white_regions.append(count)

    if white_regions:
        # Penalize dust (many tiny regions) and blobs (one huge region)
        median_size = float(np.median(white_regions))
        if median_size < 2:
            region_quality = 0.2  # too dusty
        elif median_size > size * size * 0.4:
            region_quality = 0.3  # one big blob
        else:
            region_quality = min(1.0, median_size / 8.0)
    else:
        region_quality = 0.0

    return float(np.mean([row_autocorr, col_autocorr, pattern_score, region_quality]))


def aesthetics_score(grid: np.ndarray) -> float:
    size = grid.shape[0]
    h_mid, w_mid = size // 2, size // 2
    q1 = np.mean(grid[:h_mid, :w_mid])
    q2 = np.mean(grid[:h_mid, w_mid:])
    q3 = np.mean(grid[h_mid:, :w_mid])
    q4 = np.mean(grid[h_mid:, w_mid:])
    balance_score = 1 - np.std([q1, q2, q3, q4]) * 4
    balance_score = float(np.clip(balance_score, 0, 1))

    # Border framing: reward a clear border that frames content inside
    border = np.concatenate([
        grid[0, :], grid[-1, :],
        grid[1:-1, 0], grid[1:-1, -1]
    ])
    interior = grid[1:-1, 1:-1]
    border_density = float(np.mean(border))
    interior_density = float(np.mean(interior))
    # Reward contrast between border and interior (either way)
    framing = abs(border_density - interior_density)
    framing_score = float(np.clip(framing * 2, 0, 1))

    # Connected components: reward 2-8 components (penalize dust and monoliths)
    visited = np.zeros_like(grid, dtype=bool)

    def flood_fill(r: int, c: int) -> None:
        stack = [(r, c)]
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= size or c < 0 or c >= size or visited[r, c]:
                continue
            if grid[r, c] == 0:
                continue
            visited[r, c] = True
            stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])

    n_components = 0
    for i in range(size):
        for j in range(size):
            if grid[i, j] == 1 and not visited[i, j]:
                flood_fill(i, j)
                n_components += 1

    # Sweet spot: 2-8 components
    if n_components == 0:
        connected_score = 0.0
    elif n_components <= 8:
        connected_score = min(1.0, n_components / 3.0)
    else:
        connected_score = max(0, 1 - (n_components - 8) / 12)
    connected_score = float(np.clip(connected_score, 0, 1))

    return float(np.mean([balance_score, framing_score, connected_score]))


def diversity_bonus(grids: list[np.ndarray]) -> float:
    if len(grids) < 2:
        return 0.0

    pairs_to_sample = min(50, len(grids) * (len(grids) - 1) // 2)
    distances = []
    sampled_count = 0

    for i in range(len(grids)):
        for j in range(i + 1, len(grids)):
            if sampled_count >= pairs_to_sample:
                break
            hamming = float(np.sum(grids[i] != grids[j])) / grids[i].size
            distances.append(hamming)
            sampled_count += 1
        if sampled_count >= pairs_to_sample:
            break

    if not distances:
        return 0.0
    return float(np.mean(distances))


class ArtCritic:
    def __init__(
        self,
        config: ArtConfig,
        weights: dict | None = None,
    ) -> None:
        self.config = config
        if weights is None:
            self.weights = {
                'symmetry': 0.20,
                'complexity': 0.30,
                'structure': 0.25,
                'aesthetics': 0.15,
                'diversity': 0.10,
            }
        else:
            self.weights = weights

    def score_single(self, grid: np.ndarray) -> dict:
        density = float(np.mean(grid))

        sym = symmetry_score(grid)
        cplx = complexity_score(grid)
        struct = structure_score(grid)
        aes = aesthetics_score(grid)

        # Gate: images with <15% or >85% density get hard-capped
        if density < 0.15 or density > 0.85:
            gate = 0.3  # can't score above 0.3 if nearly empty/full
        else:
            gate = 1.0

        composite = (
            self.weights['symmetry'] * sym +
            self.weights['complexity'] * cplx +
            self.weights['structure'] * struct +
            self.weights['aesthetics'] * aes
        ) * gate

        return {
            'symmetry': sym,
            'complexity': cplx,
            'structure': struct,
            'aesthetics': aes,
            'diversity': 0.0,
            'composite': composite,
        }

    def score_batch(self, grids: list[np.ndarray]) -> list[dict]:
        div_bonus = diversity_bonus(grids)

        scores = []
        for grid in grids:
            score_dict = self.score_single(grid)
            score_dict['diversity'] = div_bonus
            score_dict['composite'] = (
                score_dict['composite'] +
                self.weights['diversity'] * div_bonus
            )
            scores.append(score_dict)

        return scores

    def rank(
        self,
        grids: list[np.ndarray],
    ) -> list[tuple[int, dict]]:
        scores = self.score_batch(grids)
        ranked = [
            (i, scores[i])
            for i in range(len(grids))
        ]
        ranked.sort(key=lambda x: x[1]['composite'], reverse=True)
        return ranked
