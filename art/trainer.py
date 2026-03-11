from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from art.config import ArtConfig
from art.data import PixelDataset
from art.events import EventBus
from art.model import PixelGPT


class Trainer:
    def __init__(
        self,
        model: PixelGPT,
        config: ArtConfig,
        device: torch.device,
        event_bus: EventBus | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.event_bus = event_bus
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=config.PAD)

    def train(
        self,
        dataset: PixelDataset,
        steps: int | None = None,
        lr: float | None = None,
    ) -> list[float]:
        if steps is None:
            steps = self.config.train_steps
        if lr is None:
            lr = self.config.lr

        # Fresh optimizer with the given lr
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.config.weight_decay,
        )

        warmup_steps = self.config.warmup_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            # Cosine decay from 1.0 to 0.0 after warmup
            progress = (current_step - warmup_steps) / max(1, steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        effective_bs = min(self.config.batch_size, len(dataset))
        dataloader = DataLoader(
            dataset,
            batch_size=effective_bs,
            shuffle=True,
            drop_last=len(dataset) > effective_bs,
        )

        self.model.train()
        losses: list[float] = []
        data_iter = iter(dataloader)
        use_amp = self.device.type == "mps"

        # Preview interval: fire early so gallery isn't empty
        preview_interval = min(500, max(10, steps // 4))

        if self.event_bus:
            self.event_bus.emit("train_start", total_steps=steps, phase="train")

        for step in range(1, steps + 1):
            # Cycle through dataloader if exhausted
            tokens = next(data_iter, None)
            if tokens is None:
                data_iter = iter(dataloader)
                tokens = next(data_iter, None)
                if tokens is None:
                    break

            tokens = tokens.to(self.device)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            with torch.autocast(self.device.type, enabled=use_amp):
                logits = self.model(inputs)
                ce_loss = self.loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )
                # Entropy regularization: penalize overconfident predictions
                probs = F.softmax(logits.float(), dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                loss = ce_loss - 0.02 * entropy

            if self.event_bus and (step == 1 or step % 10 == 0):
                with torch.no_grad():
                    per_token = F.cross_entropy(
                        logits.float().reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        ignore_index=self.config.PAD,
                        reduction='none',
                    )
                    per_pos = per_token.view(targets.shape).mean(dim=0)
                    self.event_bus.emit("token_difficulty", difficulties=per_pos.cpu().tolist())

                    # Capture neural activity: layer activations + embedding similarity
                    self.model.eval()
                    _, activations = self.model.forward_with_activations(inputs[:1])
                    self.model.train()

                    gs = self.config.grid_size
                    n_pixels = gs * gs
                    layer_maps = []
                    for act in activations:
                        # act: (1, T, d_model) — take pixel positions, compute mean abs
                        pixel_act = act[0, 1:n_pixels + 1, :].float().abs().mean(dim=-1)
                        grid = pixel_act.reshape(gs, gs).cpu().numpy()
                        layer_maps.append(grid)

                    # Color embedding similarity (how the model sees color relationships)
                    nc = self.config.n_colors
                    emb = self.model.tok_emb.weight[:nc].detach().float()
                    emb_norm = F.normalize(emb, dim=1)
                    sim = (emb_norm @ emb_norm.T).cpu().numpy()

                    # Per-layer weight energy
                    weight_norms = []
                    for block in self.model.blocks:
                        norm = sum(p.data.float().abs().mean().item() for p in block.parameters())
                        weight_norms.append(norm)

                    self.event_bus.emit(
                        "neural_activity",
                        layer_maps=layer_maps,
                        embedding_sim=sim,
                        weight_norms=weight_norms,
                        step=step,
                        total_steps=steps,
                    )

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            losses.append(loss_val)

            current_lr = self.optimizer.param_groups[0]["lr"]

            if self.event_bus:
                self.event_bus.emit("train_step", step=step, loss=loss_val, lr=current_lr, grad_norm=float(grad_norm))

            # Periodically generate preview samples so the TUI can show the model learning
            if self.event_bus and (step == 1 or step % preview_interval == 0):
                self.model.eval()
                with torch.no_grad():
                    from art.tokenizer import PixelTokenizer
                    tok = PixelTokenizer(self.config)
                    preview_tokens = self.model.generate(
                        batch_size=8, temperature=0.9, device="cpu"
                    )
                    grids = [tok.decode_to_grid(preview_tokens[i].tolist()) for i in range(preview_tokens.shape[0])]
                self.model.train()
                self.event_bus.emit("train_preview", grids=grids, step=step, total_steps=steps)

            if step % 100 == 0:
                print(f"Step {step}/{steps} | Loss: {loss_val:.4f} | LR: {current_lr:.6f}")

            if step % 500 == 0 and self.device.type == "mps":
                torch.mps.empty_cache()

        if self.event_bus:
            self.event_bus.emit("train_end", losses=losses)

        return losses

    def save_checkpoint(self, path: Path, extra: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "extra": extra or {},
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> dict:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("extra", {})
