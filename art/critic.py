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
    density_score = 1 - abs(2 * density - 1)

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
    entropy_score = float(np.mean(entropy_vals)) / 1.0
    entropy_score = min(entropy_score, 1.0)

    return float(np.mean([density_score, edge_score, entropy_score]))


def aesthetics_score(grid: np.ndarray) -> float:
    size = grid.shape[0]
    h_mid, w_mid = size // 2, size // 2
    q1 = np.mean(grid[:h_mid, :w_mid])
    q2 = np.mean(grid[:h_mid, w_mid:])
    q3 = np.mean(grid[h_mid:, :w_mid])
    q4 = np.mean(grid[h_mid:, w_mid:])
    balance_score = 1 - np.std([q1, q2, q3, q4]) * 4
    balance_score = float(np.clip(balance_score, 0, 1))

    border = np.concatenate([
        grid[0, :], grid[-1, :],
        grid[1:-1, 0], grid[1:-1, -1]
    ])
    border_clear = float(1 - np.mean(border))

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

    connected_score = 1 - abs(n_components - 5) / 10
    connected_score = float(np.clip(connected_score, 0, 1))

    return float(np.mean([balance_score, border_clear, connected_score]))


def diversity_bonus(grids: list[np.ndarray]) -> float:
    if len(grids) < 2:
        return 0.0

    pairs_to_sample = min(50, len(grids) * (len(grids) - 1) // 2)
    distances = []

    indices = list(range(len(grids)))
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
                'symmetry': 0.3,
                'complexity': 0.3,
                'aesthetics': 0.3,
                'diversity': 0.1,
            }
        else:
            self.weights = weights

    def score_single(self, grid: np.ndarray) -> dict:
        sym = symmetry_score(grid)
        cplx = complexity_score(grid)
        aes = aesthetics_score(grid)

        composite = (
            self.weights['symmetry'] * sym +
            self.weights['complexity'] * cplx +
            self.weights['aesthetics'] * aes
        )

        return {
            'symmetry': sym,
            'complexity': cplx,
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
