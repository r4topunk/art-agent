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
                loss = self.loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )

            if self.event_bus and step % 50 == 0:
                with torch.no_grad():
                    per_token = F.cross_entropy(
                        logits.float().reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        ignore_index=self.config.PAD,
                        reduction='none',
                    )
                    per_pos = per_token.view(targets.shape).mean(dim=0)
                    self.event_bus.emit("token_difficulty", difficulties=per_pos.cpu().tolist())

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
