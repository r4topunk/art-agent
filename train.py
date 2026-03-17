"""
Pixel art autoresearch training script. Single-device, single-file.
Adapted from karpathy/autoresearch for pixel art generation.

This is the ONLY file the AI agent modifies.
Usage: uv run train.py
"""

import gc
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    GRID_SIZE, N_COLORS, VOCAB_SIZE, BOS, EOS, PAD, SEQ_LENGTH,
    TIME_BUDGET, BOOTSTRAP_DIR,
    generate_bootstrap_patterns, make_dataloader,
    decode_to_grid, score_batch, evaluate_composite,
)

# ---------------------------------------------------------------------------
# PixelGPT Model
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))

    def forward_cached(self, x, kv_cache):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        if kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=2)
            v = torch.cat([kv_cache["v"], v], dim=2)
        new_cache = {"k": k, "v": v}
        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), new_cache


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

    def forward_cached(self, x, kv_cache):
        normed = self.ln1(x)
        attn_out, new_cache = self.attn.forward_cached(normed, kv_cache)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, new_cache


class PixelGPT(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(SEQ_LENGTH, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def forward(self, x, targets=None, reduction='mean'):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(positions)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        logits = self.head(h)

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=PAD,
                reduction=reduction,
            )
            return loss
        return logits

    @torch.no_grad()
    def generate(self, batch_size, temperature=1.0, top_k=0, top_p=1.0, device="cpu"):
        gen_device = torch.device(device)
        model_device = next(self.parameters()).device
        if model_device != gen_device:
            self.to(gen_device)

        self.eval()
        seq = torch.full((batch_size, SEQ_LENGTH), PAD, dtype=torch.long, device=gen_device)
        seq[:, 0] = BOS
        kv_caches = [None] * len(self.blocks)

        pos_buf = torch.zeros(1, 1, dtype=torch.long, device=gen_device)
        inv_temp = 1.0 / temperature

        # Prime cache with BOS
        bos = seq[:, :1]
        h = self.tok_emb(bos) + self.pos_emb(pos_buf)
        new_caches = [None] * len(self.blocks)
        for i, block in enumerate(self.blocks):
            h, new_caches[i] = block.forward_cached(h, kv_caches[i])
        kv_caches = new_caches

        for t in range(1, SEQ_LENGTH):
            cur_tok = seq[:, t - 1:t]
            pos_buf.fill_(t - 1)
            h = self.tok_emb(cur_tok) + self.pos_emb(pos_buf)
            for i, block in enumerate(self.blocks):
                h, kv_caches[i] = block.forward_cached(h, kv_caches[i])
            logits = self.head(self.ln_f(h))[:, 0, :]
            logits.mul_(inv_temp)
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                logits.masked_fill_(logits < values[:, -1:], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                remove = (cumulative - sorted_probs) > top_p
                sorted_probs[remove] = 0.0
                probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(probs, num_samples=1)
            seq[:, t] = next_token.squeeze(-1)

        if model_device != gen_device:
            self.to(model_device)
        return seq

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

# Model architecture
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 6
D_FF = 1024
DROPOUT = 0.1

# Training
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 20
ENTROPY_REG = 0.05     # entropy regularization coefficient

# GAS loop
IMAGES_PER_GEN = 64
SELECT_TOP = 12
FINETUNE_STEPS = 50
FINETUNE_LR = 1e-4
BOOTSTRAP_MIX_RATIO = 0.3
GEN_TEMPERATURE_START = 1.1
GEN_TEMPERATURE_END = 0.85
TEMP_DECAY_GENS = 30

# Eval
EVAL_TEMPERATURE = 0.9

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Device selection: MPS > CUDA > CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# Load or generate bootstrap patterns
bootstrap_path = BOOTSTRAP_DIR / "patterns.npz"
if bootstrap_path.exists():
    data = np.load(bootstrap_path)
    bootstrap_patterns = list(data["patterns"])
    print(f"Loaded {len(bootstrap_patterns)} bootstrap patterns from cache")
else:
    print("Generating bootstrap patterns...")
    bootstrap_patterns = generate_bootstrap_patterns(5000)
    BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(bootstrap_path, patterns=np.stack(bootstrap_patterns))
    print(f"Generated and cached {len(bootstrap_patterns)} patterns")

# Build model
model = PixelGPT(D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT).to(device)
num_params = model.count_parameters()
print(f"Model parameters: {num_params:,}")
print(f"Time budget: {TIME_BUDGET}s")

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step, total_steps, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def get_temperature(generation):
    if generation >= TEMP_DECAY_GENS:
        return GEN_TEMPERATURE_END
    t = generation / max(1, TEMP_DECAY_GENS)
    return GEN_TEMPERATURE_START + t * (GEN_TEMPERATURE_END - GEN_TEMPERATURE_START)


# ---------------------------------------------------------------------------
# Phase 1: Bootstrap training (fixed time budget)
# ---------------------------------------------------------------------------

print("\n=== Phase 1: Bootstrap Training ===")
t_train_start = time.time()

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
dataloader = make_dataloader(bootstrap_patterns, BATCH_SIZE)
use_amp = device.type == "mps"

# Estimate steps from time budget (70% for bootstrap, 30% for GAS)
BOOTSTRAP_TIME = int(TIME_BUDGET * 0.20)
GAS_TIME = int(TIME_BUDGET * 0.80)

model.train()
step = 0
total_training_time = 0.0
smooth_loss = 0.0
data_iter = iter(dataloader)

while True:
    t0 = time.time()

    tokens = next(data_iter, None)
    if tokens is None:
        data_iter = iter(dataloader)
        tokens = next(data_iter)
    tokens = tokens.to(device)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]

    with torch.autocast(device.type, enabled=use_amp):
        logits = model(inputs)
        ce_loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        probs = F.softmax(logits.float(), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
        loss = ce_loss - ENTROPY_REG * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # LR schedule
    lr = get_lr(step, 500, LEARNING_RATE, WARMUP_STEPS)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    t1 = time.time()
    dt = t1 - t0
    if step > 5:
        total_training_time += dt

    loss_val = loss.item()
    ema = 0.9
    smooth_loss = ema * smooth_loss + (1 - ema) * loss_val
    debiased = smooth_loss / (1 - ema ** (step + 1))

    if step % 50 == 0:
        remaining = max(0, BOOTSTRAP_TIME - total_training_time)
        print(f"\rstep {step:04d} | loss: {debiased:.4f} | lr: {lr:.6f} | remaining: {remaining:.0f}s    ", end="", flush=True)

    step += 1

    if step > 5 and total_training_time >= BOOTSTRAP_TIME:
        break

    if device.type == "mps" and step % 500 == 0:
        torch.mps.empty_cache()

print(f"\nBootstrap done: {step} steps in {total_training_time:.1f}s")

# ---------------------------------------------------------------------------
# Phase 2: GAS Loop (Generate → Score → Select → Finetune)
# ---------------------------------------------------------------------------

print("\n=== Phase 2: GAS Evolution ===")
t_gas_start = time.time()
gas_time_elapsed = 0.0
generation = 0
archive: list[np.ndarray] = []
best_ever_composite = 0.0

while gas_time_elapsed < GAS_TIME:
    t_gen_start = time.time()

    temperature = get_temperature(generation)
    print(f"\n--- Generation {generation} | temp={temperature:.3f} ---")

    # Generate
    model.eval()
    tokens = model.generate(
        batch_size=IMAGES_PER_GEN,
        temperature=temperature,
        device="cpu",
    )
    grids = [decode_to_grid(tokens[i].tolist()) for i in range(tokens.shape[0])]

    # Score
    scores = score_batch(grids)
    composites = [s['composite'] for s in scores]
    mean_comp = float(np.mean(composites))
    max_comp = float(np.max(composites))
    best_ever_composite = max(best_ever_composite, max_comp)

    print(f"  scores: mean={mean_comp:.4f} max={max_comp:.4f} best_ever={best_ever_composite:.4f}")

    # Select top-N
    ranked = sorted(range(len(grids)), key=lambda i: composites[i], reverse=True)
    selected_indices = ranked[:SELECT_TOP]
    selected = [grids[i] for i in selected_indices]

    # Augment selected with flips/rotations
    augmented = []
    for g in selected:
        augmented.append(g)
        augmented.append(np.fliplr(g))
        augmented.append(np.flipud(g))
        augmented.append(np.rot90(g, 2))

    # Finetune on augmented + bootstrap mix
    training_patterns = list(augmented)
    n_bootstrap = max(1, int(len(selected) * BOOTSTRAP_MIX_RATIO))
    sampled_bootstrap = random.sample(bootstrap_patterns, min(n_bootstrap, len(bootstrap_patterns)))
    training_patterns.extend(sampled_bootstrap)

    finetune_ds = make_dataloader(training_patterns, min(BATCH_SIZE, len(training_patterns)))
    model.train()
    ft_optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)

    ft_iter = iter(finetune_ds)
    for ft_step in range(FINETUNE_STEPS):
        ft_tokens = next(ft_iter, None)
        if ft_tokens is None:
            ft_iter = iter(finetune_ds)
            ft_tokens = next(ft_iter)
        ft_tokens = ft_tokens.to(device)
        ft_inputs, ft_targets = ft_tokens[:, :-1], ft_tokens[:, 1:]

        with torch.autocast(device.type, enabled=use_amp):
            ft_logits = model(ft_inputs)
            ft_loss = loss_fn(ft_logits.reshape(-1, ft_logits.size(-1)), ft_targets.reshape(-1))

        ft_optimizer.zero_grad()
        ft_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        ft_optimizer.step()

    archive.extend(selected)

    t_gen_end = time.time()
    gen_time = t_gen_end - t_gen_start
    gas_time_elapsed = t_gen_end - t_gas_start
    remaining = max(0, GAS_TIME - gas_time_elapsed)
    print(f"  gen_time={gen_time:.1f}s | gas_elapsed={gas_time_elapsed:.1f}s | remaining={remaining:.0f}s")

    generation += 1

    if device.type == "mps":
        torch.mps.empty_cache()

print(f"\nGAS done: {generation} generations in {gas_time_elapsed:.1f}s")

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

print("\n=== Final Evaluation ===")
model.eval()
eval_results = evaluate_composite(model, temperature=EVAL_TEMPERATURE)

t_end = time.time()
total_time = t_end - t_start

# ---------------------------------------------------------------------------
# Save visual output for inspection
# ---------------------------------------------------------------------------

from PIL import Image as PILImage

VISUAL_DIR = Path("run_visuals")
VISUAL_DIR.mkdir(exist_ok=True)

# Generate a final batch for visual inspection
model.eval()
with torch.no_grad():
    vis_tokens = model.generate(batch_size=36, temperature=EVAL_TEMPERATURE, device="cpu")

vis_grids = [decode_to_grid(vis_tokens[i].tolist()) for i in range(vis_tokens.shape[0])]
vis_scores = score_batch(vis_grids)

# Save individual top-9 pieces (upscaled 64x64)
ranked_vis = sorted(range(len(vis_grids)), key=lambda i: vis_scores[i]['composite'], reverse=True)
for rank, idx in enumerate(ranked_vis[:9]):
    g = vis_grids[idx]
    rgb = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
    for ci, color in enumerate([(0,0,0),(255,241,232),(255,0,77),(255,163,0),(255,236,39),(0,228,54),(41,173,255),(255,119,168)]):
        rgb[g == ci] = color
    img = PILImage.fromarray(rgb).resize((64, 64), PILImage.NEAREST)
    img.save(VISUAL_DIR / f"top{rank+1}_score{vis_scores[idx]['composite']:.3f}.png")

# Save a 6x6 contact sheet (all 36 pieces, each 64x64)
sheet = PILImage.new("RGB", (6 * 64, 6 * 64))
for i, g in enumerate(vis_grids):
    rgb = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
    for ci, color in enumerate([(0,0,0),(255,241,232),(255,0,77),(255,163,0),(255,236,39),(0,228,54),(41,173,255),(255,119,168)]):
        rgb[g == ci] = color
    cell = PILImage.fromarray(rgb).resize((64, 64), PILImage.NEAREST)
    row, col = divmod(i, 6)
    sheet.paste(cell, (col * 64, row * 64))
sheet.save(VISUAL_DIR / "contact_sheet.png")

print(f"Visual output saved to {VISUAL_DIR}/")
print(f"  contact_sheet.png  — all 36 generated pieces")
print(f"  top1..top9         — best pieces by composite score")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n---")
print(f"mean_composite:     {eval_results['mean_composite']:.6f}")
print(f"max_composite:      {eval_results['max_composite']:.6f}")
print(f"mean_symmetry:      {eval_results['mean_symmetry']:.6f}")
print(f"mean_complexity:    {eval_results['mean_complexity']:.6f}")
print(f"mean_structure:     {eval_results['mean_structure']:.6f}")
print(f"mean_aesthetics:    {eval_results['mean_aesthetics']:.6f}")
print(f"mean_diversity:     {eval_results['mean_diversity']:.6f}")
print(f"training_seconds:   {total_training_time + gas_time_elapsed:.1f}")
print(f"total_seconds:      {total_time:.1f}")
print(f"bootstrap_steps:    {step}")
print(f"gas_generations:    {generation}")
print(f"num_params:         {num_params:,}")
print(f"n_eval_images:      {eval_results['n_images']}")
print(f"device:             {device}")
