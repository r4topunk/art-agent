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

    # Color balance: entropy of color distribution rewards both variety AND balance.
    # A 2-color image gets at most 0.25 regardless of how balanced it is.
    # A 5-color image where one dominates 80% scores lower than a balanced 4-color one.
    counts_c = np.bincount(grid.flatten(), minlength=n_colors)
    n_unique = int(np.sum(counts_c > 0))
    if n_unique <= 1:
        color_score = 0.0
    else:
        probs_c = counts_c / counts_c.sum()
        probs_nz = probs_c[probs_c > 0]
        h = float(-np.sum(probs_nz * np.log2(probs_nz)))
        color_score = h / np.log2(n_colors)  # 0..1, max when all 8 colors balanced
        if n_unique == 2:
            color_score = min(color_score, 0.25)  # hard cap: 2-color is boring

    # Edge density: sweet spot is 30-65% (enough variation without pure checkerboard noise).
    # Penalises near-maximum edge density which indicates pixel-level alternation.
    h_edges = np.sum(grid[:, :-1] != grid[:, 1:])
    v_edges = np.sum(grid[:-1, :] != grid[1:, :])
    max_edges = 2 * size * (size - 1)
    edge_density = float((h_edges + v_edges) / max_edges)
    if edge_density < 0.10:
        edge_score = edge_density * 3.0
    elif edge_density < 0.30:
        edge_score = 0.30 + (edge_density - 0.10) / 0.20 * 0.50
    elif edge_density <= 0.65:
        edge_score = 1.0
    elif edge_density <= 0.85:
        edge_score = 1.0 - (edge_density - 0.65) / 0.20 * 0.60
    else:
        edge_score = 0.20  # pure checkerboard territory

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
        max_region_frac = float(max(regions)) / (size * size)
        # Emergent patterns: many small-to-medium regions = texture and visual rhythm
        # Sweet spot: 15-100 regions, median size 2-8 pixels.
        # Also penalize when ANY single region dominates >30% of the grid —
        # median_size misses this when one huge region coexists with many tiny ones.
        if n_regions < 4:
            region_quality = 0.1  # almost no regions = flat fill
        elif n_regions > 140:
            region_quality = 0.3  # isolated pixels = noise
        elif median_size < 1.5:
            region_quality = 0.2  # single pixels = pure noise
        elif max_region_frac > 0.25:
            # One blob covers >25% of the grid — penalise steeply
            region_quality = max(0.1, 1.0 - 3.5 * (max_region_frac - 0.25))
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

    # Border framing: continuous L1 distance between border and interior color distributions.
    # Binary dominant-color comparison was trivially gamed by noise patterns.
    border = np.concatenate([
        grid[0, :], grid[-1, :],
        grid[1:-1, 0], grid[1:-1, -1]
    ])
    interior = grid[1:-1, 1:-1].flatten()
    border_freq = np.bincount(border, minlength=n_colors)[:n_colors].astype(float)
    border_freq /= border_freq.sum() + 1e-8
    interior_freq = np.bincount(interior, minlength=n_colors)[:n_colors].astype(float)
    interior_freq /= interior_freq.sum() + 1e-8
    # L1 / 2 gives 0..1 (0 = identical distributions, 1 = completely different)
    framing_score = float(np.sum(np.abs(border_freq - interior_freq)) / 2.0)

    # Color harmony: entropy-based, consistent with complexity color_score.
    counts_h = np.bincount(grid.flatten(), minlength=n_colors)
    n_uniq_h = int(np.sum(counts_h > 0))
    if n_uniq_h <= 1:
        harmony = 0.0
    else:
        probs_h = counts_h / counts_h.sum()
        probs_nz_h = probs_h[probs_h > 0]
        h_ent = float(-np.sum(probs_nz_h * np.log2(probs_nz_h)))
        harmony = h_ent / np.log2(n_colors)
        if n_uniq_h == 2:
            harmony = min(harmony, 0.25)

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


def _stripe_fraction(grid: np.ndarray) -> float:
    """Fraction of rows/columns that are near-uniform (≤2 colors)."""
    rows_uniform = sum(1 for r in range(grid.shape[0]) if len(np.unique(grid[r, :])) <= 2)
    cols_uniform = sum(1 for c in range(grid.shape[1]) if len(np.unique(grid[:, c])) <= 2)
    return max(rows_uniform, cols_uniform) / grid.shape[0]


def _row_repeat_fraction(grid: np.ndarray) -> float:
    """Fraction of adjacent rows that are identical (horizontal banding)."""
    size = grid.shape[0]
    if size < 2:
        return 0.0
    same = sum(1 for r in range(size - 1) if np.array_equal(grid[r], grid[r + 1]))
    return same / (size - 1)


def _motif_repeat_score(grid: np.ndarray) -> float:
    """Detect short repeating motifs within rows/columns (e.g. 142142142...).

    Returns 0..1 where 1 = every row/col is a repeating motif of period ≤4.
    """
    size = grid.shape[0]
    motif_rows = 0
    for r in range(size):
        row = grid[r, :]
        for period in range(1, 5):  # check periods 1,2,3,4
            tile = row[:period]
            repeated = np.tile(tile, (size + period - 1) // period)[:size]
            if np.sum(row == repeated) >= size - 2:  # allow 2 mismatches
                motif_rows += 1
                break

    motif_cols = 0
    for c in range(size):
        col = grid[:, c]
        for period in range(1, 5):
            tile = col[:period]
            repeated = np.tile(tile, (size + period - 1) // period)[:size]
            if np.sum(col == repeated) >= size - 2:
                motif_cols += 1
                break

    return max(motif_rows, motif_cols) / size


class ArtCritic:
    def __init__(
        self,
        config: ArtConfig,
        weights: dict | None = None,
    ) -> None:
        self.config = config
        if weights is None:
            self.weights = {
                'symmetry': 0.15,
                'complexity': 0.25,
                'structure': 0.25,
                'aesthetics': 0.20,
                'diversity': 0.15,
            }
        else:
            self.weights = weights

    def score_single(self, grid: np.ndarray) -> dict:
        n_colors = self.config.n_colors

        # Gate: penalize flat/sparse images and dominant-color images
        unique = len(np.unique(grid))
        if unique <= 1:
            gate = 0.1  # solid fill — nearly worthless
        elif unique <= 2:
            gate = 0.4  # near-monochrome
        else:
            gate = 1.0

        # Dominance penalty: crush score when one color covers too much.
        # With 8 colors, uniform = 12.5% each; 35% is already 2.8× over-represented.
        # Steeper curve (3.0×) ensures 50% dominant → gate 0.55, 65%+ → floor.
        _, counts = np.unique(grid, return_counts=True)
        dominance = float(counts.max()) / grid.size  # 0..1
        if dominance > 0.60:
            gate *= 0.05
        elif dominance > 0.30:
            gate *= max(0.1, 1.0 - 5.0 * (dominance - 0.30))

        # Large-region gate: if any contiguous same-color region covers >20%
        # of the grid, multiply gate down.  structure_score also penalises
        # this in its region_quality sub-component, but that gets diluted by
        # averaging with 3 other sub-scores — this gate is multiplicative and
        # cannot be washed out.
        size = grid.shape[0]
        visited = np.zeros_like(grid, dtype=bool)
        max_region = 0
        for ri in range(size):
            for ci in range(size):
                if not visited[ri, ci]:
                    stack = [(ri, ci)]
                    color = int(grid[ri, ci])
                    count = 0
                    while stack:
                        r2, c2 = stack.pop()
                        if 0 <= r2 < size and 0 <= c2 < size and not visited[r2, c2] and grid[r2, c2] == color:
                            visited[r2, c2] = True
                            count += 1
                            stack.extend([(r2 + 1, c2), (r2 - 1, c2), (r2, c2 + 1), (r2, c2 - 1)])
                    max_region = max(max_region, count)
        max_region_frac = max_region / grid.size
        if max_region_frac > 0.15:
            gate *= max(0.1, 1.0 - 4.0 * (max_region_frac - 0.15))

        # Stripe penalty: penalize grids with many near-uniform rows/columns
        stripe_frac = _stripe_fraction(grid)
        if stripe_frac > 0.30:
            gate *= max(0.05, 1.0 - 3.0 * (stripe_frac - 0.30))

        # Row/column repetition: adjacent identical rows = horizontal bands
        row_rep = _row_repeat_fraction(grid)
        col_rep = _row_repeat_fraction(grid.T)
        band_frac = max(row_rep, col_rep)
        if band_frac > 0.20:
            gate *= max(0.05, 1.0 - 3.0 * (band_frac - 0.20))

        # Motif repetition: short repeating patterns within rows (e.g. 142142142...)
        motif_frac = _motif_repeat_score(grid)
        if motif_frac > 0.25:
            gate *= max(0.10, 1.0 - 2.0 * (motif_frac - 0.25))

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
