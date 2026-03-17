# autoresearch — pixel art edition

This is an experiment to have the LLM do its own research on autonomous pixel art generation.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar16`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, bootstrap data generation, tokenizer, dataloader, ArtCritic evaluation. Do not modify.
   - `train.py` — the file you modify. PixelGPT model, optimizer, training loop, GAS evolution loop.
4. **Verify data exists**: Check that `data/bootstrap/patterns.npz` exists. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Context

This project generates 16×16 pixel art using an 8-color PICO-8 palette. A small GPT (PixelGPT) is trained autoregressively on bootstrap patterns, then evolved through a Genetic Art Selection (GAS) loop:

```
Bootstrap patterns → train PixelGPT → generate batch → score with ArtCritic
                                          ↑                      ↓
                                     finetune               select top-N
                                          ↑                      ↓
                                       repeat ←── next generation
```

The model runs on Apple Silicon (MPS) or CUDA. No H100 required — this is a small model (~6M params).

## Experimentation

Each experiment runs on a single device. The training script runs for a **fixed time budget of 5 minutes** (wall clock). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, GAS parameters, generation temperature, batch size, number of layers, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, scoring, and training constants (time budget, grid size, palette, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_composite` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest mean_composite score.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the GAS parameters, the training/GAS time split. The only constraint is that the code runs without crashing and finishes within the time budget.

**Memory** is a soft constraint. Some increase is acceptable for meaningful composite gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Metric

The key metric is `mean_composite`, which is the average composite score across 5 evaluation generations × 36 images each = 180 images. The composite score combines:

| Component | Weight | Measures |
|-----------|--------|----------|
| **Symmetry** | 0.15 | Horizontal, vertical, rotational similarity |
| **Complexity** | 0.25 | Color diversity, edges, entropy, transitions |
| **Structure** | 0.25 | Row/column coherence, 2×2 patterns, region analysis |
| **Aesthetics** | 0.20 | Quadrant balance, border framing, color harmony |
| **Diversity** | 0.15 | Hamming distance to other images in batch |

Quality gates penalize: monochrome images, dominant single colors, large uniform regions, stripes, row/column banding, and repeating motifs.

Higher is better. Range is approximately 0.0 to 0.7 in practice.

## Output format

Once the script finishes it prints a summary like this:

```
---
mean_composite:     0.234567
max_composite:      0.456789
mean_symmetry:      0.345678
mean_complexity:    0.234567
mean_structure:     0.345678
mean_aesthetics:    0.234567
mean_diversity:     0.456789
training_seconds:   298.5
total_seconds:      315.2
bootstrap_steps:    450
gas_generations:    12
num_params:         6,432,011
n_eval_images:      180
device:             mps
```

You can extract the key metric from the log file:

```
grep "^mean_composite:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	mean_composite	max_composite	status	description
```

1. git commit hash (short, 7 chars)
2. mean_composite achieved (e.g. 0.234567) — use 0.000000 for crashes
3. max_composite achieved (e.g. 0.456789) — use 0.000000 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	mean_composite	max_composite	status	description
a1b2c3d	0.234567	0.456789	keep	baseline
b2c3d4e	0.256789	0.478901	keep	increase d_model to 384
c3d4e5f	0.210000	0.420000	discard	switch to ReLU activation
d4e5f6g	0.000000	0.000000	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar16`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^mean_composite:\|^max_composite:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If mean_composite improved (higher), you "advance" the branch, keeping the git commit
9. If mean_composite is equal or worse, you git reset back to where you started

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval). If a run exceeds 10 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (OOM, or a bug, etc.), use your judgment: quick fix → re-run. Fundamentally broken → skip.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder — try architectural changes, different optimizers, schedule tweaks, GAS parameter tuning, temperature strategies, generation tricks. The loop runs until the human interrupts you, period.

## Ideas to explore

Here are some directions worth trying (you are NOT limited to these):

- **Architecture**: deeper/wider model, different d_ff ratios, RMS norm vs LayerNorm, different attention patterns
- **Optimizer**: different LR, warmup/cooldown schedules, Adam vs AdamW betas, gradient accumulation
- **GAS loop**: time split between bootstrap and GAS, images_per_gen, select_top, finetune_steps, bootstrap mix ratio
- **Temperature**: different start/end, adaptive temperature based on diversity, top-k/top-p sampling
- **Training**: entropy regularization coefficient, different loss functions, data augmentation (rotations, flips)
- **Regularization**: dropout rate, weight decay, gradient clipping threshold
- **Novel ideas**: 2D positional embeddings (row + col), color-aware attention, multi-scale generation
