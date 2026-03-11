import numpy as np
from art.config import ArtConfig


def symmetry_score(grid: np.ndarray) -> float:
    """Measure symmetry across axes and rotations. Works with any discrete values."""
    g = grid.astype(np.int16)
    total = g.size
    h_sym = 1 - np.sum(g != np.fliplr(g)) / total
    v_sym = 1 - np.sum(g != np.flipud(g)) / total
    r180_sym = 1 - np.sum(g != np.rot90(g, 2)) / total
    r90_sym = 1 - np.sum(g != np.rot90(g)) / total
    return float(np.mean([h_sym, v_sym, r180_sym, r90_sym]))


def complexity_score(grid: np.ndarray, n_colors: int = 8) -> float:
    """Measure visual complexity: color diversity, edges, entropy, transition variety."""
    size = grid.shape[0]

    # Color usage: sweet spot is 4-7 colors
    unique = len(np.unique(grid))
    if unique <= 1:
        color_score = 0.0
    elif unique <= 2:
        color_score = 0.2
    elif unique == 3:
        color_score = 0.5
    elif unique <= 7:
        color_score = 1.0  # 4-7 colors = richest visual range
    else:
        color_score = 0.75  # all 8 can get chaotic but still okay

    # Edge count: transitions between different colors
    h_edges = np.sum(grid[:, :-1] != grid[:, 1:])
    v_edges = np.sum(grid[:-1, :] != grid[1:, :])
    max_edges = 2 * size * (size - 1)
    edge_score = float((h_edges + v_edges) / max_edges)

    # Block entropy: Shannon entropy of color distribution in 4x4 blocks
    block_size = max(1, size // 4)
    n_blocks = size // block_size
    entropy_vals = []
    for i in range(n_blocks):
        for j in range(n_blocks):
            block = grid[block_size * i:block_size * (i + 1),
                         block_size * j:block_size * (j + 1)]
            counts = np.bincount(block.flatten(), minlength=n_colors)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            h = -np.sum(probs * np.log2(probs))
            max_h = np.log2(n_colors)
            entropy_vals.append(h / max_h if max_h > 0 else 0)
    entropy_score = float(np.mean(entropy_vals)) if entropy_vals else 0.0

    # Color transition variety: how many distinct color-pair transitions exist
    # This rewards patterns where many different colors appear next to each other
    h_pairs = set(map(tuple, np.stack([grid[:, :-1].flatten(), grid[:, 1:].flatten()], axis=1)))
    v_pairs = set(map(tuple, np.stack([grid[:-1, :].flatten(), grid[1:, :].flatten()], axis=1)))
    distinct_transitions = len(h_pairs | v_pairs)
    max_transitions = n_colors * (n_colors - 1)  # all distinct color-pair edges
    transition_score = min(1.0, distinct_transitions / max(1, max_transitions * 0.4))

    return float(np.mean([color_score, edge_score, entropy_score, transition_score]))


def structure_score(grid: np.ndarray) -> float:
    """Reward structured patterns: lines, repeating motifs, coherent shapes."""
    size = grid.shape[0]

    # Row and column coherence via autocorrelation
    row_autocorr = 0.0
    col_autocorr = 0.0
    for axis in [0, 1]:
        # For each row/col, compute fraction of adjacent same-color pixels
        if axis == 0:
            same = np.mean(grid[:, :-1] == grid[:, 1:])
        else:
            same = np.mean(grid[:-1, :] == grid[1:, :])
        # Sweet spot: too low = noise, too high = flat. Reward 0.3-0.7
        coherence = float(same)
        if coherence < 0.2 or coherence > 0.9:
            val = 0.2
        elif coherence < 0.4:
            val = (coherence - 0.2) / 0.2
        elif coherence > 0.7:
            val = (0.9 - coherence) / 0.2
        else:
            val = 1.0
        if axis == 0:
            row_autocorr = val
        else:
            col_autocorr = val

    # 2x2 block variety: count distinct 2x2 color patterns
    blocks_2x2 = set()
    for r in range(size - 1):
        for c in range(size - 1):
            block = (grid[r, c], grid[r, c + 1], grid[r + 1, c], grid[r + 1, c + 1])
            blocks_2x2.add(block)
    n_patterns = len(blocks_2x2)
    # Emergent patterns need rich local variety — sweet spot: 20-160
    if n_patterns <= 5:
        pattern_score = n_patterns / 10.0
    elif n_patterns <= 160:
        pattern_score = 1.0
    else:
        pattern_score = max(0.5, 1 - (n_patterns - 160) / 150)

    # Contiguous region analysis (flood fill by color)
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
            stack.extend([(r2 + 1, c2), (r2 - 1, c2), (r2, c2 + 1), (r2, c2 - 1)])
        return count

    for i in range(size):
        for j in range(size):
            if not visited[i, j]:
                region_size = flood_fill(i, j, grid[i, j])
                regions.append(region_size)

    if regions:
        median_size = float(np.median(regions))
        n_regions = len(regions)
        # Emergent patterns: many small-to-medium regions = texture and visual rhythm
        # Sweet spot: 15-100 regions, median size 2-8 pixels
        if n_regions < 4:
            region_quality = 0.1  # almost no regions = flat fill
        elif n_regions > 140:
            region_quality = 0.3  # isolated pixels = noise
        elif median_size < 1.5:
            region_quality = 0.2  # single pixels = pure noise
        elif median_size > size * size * 0.35:
            region_quality = 0.1  # one giant blob = flat
        else:
            # Peak reward at median size 3-6 pixels (pattern-density sweet spot)
            target = 4.5
            region_quality = max(0.2, 1.0 - abs(median_size - target) / 12.0)
    else:
        region_quality = 0.0

    return float(np.mean([row_autocorr, col_autocorr, pattern_score, region_quality]))


def aesthetics_score(grid: np.ndarray, n_colors: int = 8) -> float:
    """Evaluate aesthetic qualities: balance, framing, color harmony."""
    size = grid.shape[0]

    # Quadrant balance: do all 4 quadrants have similar color distributions?
    h_mid, w_mid = size // 2, size // 2
    quadrants = [
        grid[:h_mid, :w_mid], grid[:h_mid, w_mid:],
        grid[h_mid:, :w_mid], grid[h_mid:, w_mid:],
    ]
    # Compare color frequency vectors across quadrants
    freq_vecs = []
    for q in quadrants:
        counts = np.bincount(q.flatten(), minlength=n_colors)[:n_colors].astype(float)
        counts /= counts.sum() + 1e-8
        freq_vecs.append(counts)
    # Balance = low variance of frequency vectors
    freq_arr = np.array(freq_vecs)
    balance = 1.0 - float(np.mean(np.std(freq_arr, axis=0)))
    balance_score = float(np.clip(balance, 0, 1))

    # Border framing: contrast between border and interior
    border = np.concatenate([
        grid[0, :], grid[-1, :],
        grid[1:-1, 0], grid[1:-1, -1]
    ])
    interior = grid[1:-1, 1:-1].flatten()
    # Compare dominant colors
    border_dom = np.argmax(np.bincount(border, minlength=n_colors)[:n_colors])
    interior_dom = np.argmax(np.bincount(interior, minlength=n_colors)[:n_colors])
    framing_score = 1.0 if border_dom != interior_dom else 0.3

    # Color harmony: reward rich palettes, penalize sparse (flat) ones
    unique = len(np.unique(grid))
    if unique <= 1:
        harmony = 0.0
    elif unique <= 2:
        harmony = 0.2  # nearly monochrome = boring
    elif unique == 3:
        harmony = 0.5
    elif unique <= 6:
        harmony = 1.0  # 4-6 colors = visual richness sweet spot
    else:
        harmony = 0.8  # all 7-8 colors still good

    return float(np.mean([balance_score, framing_score, harmony]))


def per_image_diversity(grids: list[np.ndarray]) -> list[float]:
    """Per-image novelty: mean hamming distance to all other images in the batch."""
    n = len(grids)
    if n < 2:
        return [0.0] * n
    scores = []
    for i in range(n):
        dists = [float(np.sum(grids[i] != grids[j])) / grids[i].size for j in range(n) if j != i]
        scores.append(float(np.mean(dists)))
    return scores


class ArtCritic:
    def __init__(
        self,
        config: ArtConfig,
        weights: dict | None = None,
    ) -> None:
        self.config = config
        if weights is None:
            self.weights = {
                'symmetry': 0.10,
                'complexity': 0.30,
                'structure': 0.25,
                'aesthetics': 0.20,
                'diversity': 0.15,
            }
        else:
            self.weights = weights

    def score_single(self, grid: np.ndarray) -> dict:
        n_colors = self.config.n_colors

        # Gate: penalize flat/sparse images
        unique = len(np.unique(grid))
        if unique <= 1:
            gate = 0.2  # solid fill
        elif unique <= 2:
            gate = 0.6  # near-monochrome
        else:
            gate = 1.0

        sym = symmetry_score(grid)
        cplx = complexity_score(grid, n_colors)
        struct = structure_score(grid)
        aes = aesthetics_score(grid, n_colors)

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
            '_gate': gate,
        }

    def score_batch(self, grids: list[np.ndarray], on_progress=None) -> list[dict]:
        div_scores = per_image_diversity(grids)

        scores = []
        for i, grid in enumerate(grids):
            score_dict = self.score_single(grid)
            score_dict['diversity'] = div_scores[i]
            # Apply the same quality gate to diversity so low-quality images
            # can't inflate their composite score purely through novelty
            score_dict['composite'] = (
                score_dict['composite'] +
                score_dict['_gate'] * self.weights['diversity'] * div_scores[i]
            )
            scores.append(score_dict)
            if on_progress and (i + 1) % 4 == 0:
                on_progress(i + 1, len(grids), score_dict)

        return scores

    def rank(self, grids: list[np.ndarray]) -> list[tuple[int, dict]]:
        scores = self.score_batch(grids)
        ranked = [(i, scores[i]) for i in range(len(grids))]
        ranked.sort(key=lambda x: x[1]['composite'], reverse=True)
        return ranked
