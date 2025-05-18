#!/usr/bin/env python
"""Generic DiLoCo/FSDP training framework.

This module provides the :class:`DilocoFSDPTrainer` class which implements a
training loop for decentralized fine-tuning using Fully Sharded Data Parallel
(FSDP) and the DiLoCo outer optimization step. The original
``finetune_qwen_fsdp.py`` script is now a thin wrapper around this framework.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from typing import Iterable, Tuple, Optional

import torch
from datasets import load_dataset, IterableDataset
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from torch.distributed import barrier
from torch.distributed.device_mesh import DeviceMesh
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm


@dataclass
class TrainerConfig:
    """Configuration options for :class:`DilocoFSDPTrainer`."""

    model_name: str
    dataset_name: str
    subset: str
    output_dir: str
    seq_len: int = 1024
    batch_size: int = 2
    grad_accum: int = 8
    lr: float = 2e-5
    max_steps: int = 10_000
    diloco_loops: int = 1
    outer_lr: float = 1e-3
    outer_momentum: float = 0.0
    text_field: Optional[str] = None


class DilocoFSDPTrainer:
    """Trainer implementing DiLoCo + FSDP for generic models and datasets."""

    def __init__(self, config: TrainerConfig, accelerator: Optional[Accelerator] = None):
        self.config = config
        self.accelerator = accelerator or Accelerator(
            gradient_accumulation_steps=config.grad_accum
        )
        self.device = self.accelerator.device
        self.is_main = self.accelerator.is_main_process

        # --- Tokenizer & model ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, use_fast=False, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16
            if self.accelerator.mixed_precision == "bf16"
            else torch.float16,
            trust_remote_code=True,
        )
        self.model.gradient_checkpointing_enable()

        # --- Dataset streaming ---
        self.raw_dataset, self.text_field = self._get_dataset()
        self.collate_fn = partial(
            self._tokenize_batch, text_field=self.text_field, seq_len=config.seq_len
        )
        self.dataloader = DataLoader(
            self.raw_dataset, batch_size=config.batch_size, collate_fn=self.collate_fn
        )

        # --- Optimizer & scheduler ---
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr, weight_decay=0.01)
        self.lr_scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=100,
            num_training_steps=config.max_steps,
        )

        (
            self.model,
            self.optimizer,
            self.dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(self.model, self.optimizer, self.dataloader, self.lr_scheduler)

        self.outer_optimizer = SGD(
            self.model.parameters(), lr=config.outer_lr, momentum=config.outer_momentum
        )

        if self.accelerator.num_processes > 1:
            device_ids = list(range(self.accelerator.num_processes))
            self.device_mesh = DeviceMesh("cuda", device_ids)
        else:
            self.device_mesh = None

        self.prev_params = [p.data.detach().cpu().clone() for p in self.model.parameters()]
        if config.outer_momentum > 0:
            self.momentum_buffers = [torch.zeros_like(p, device="cpu") for p in self.model.parameters()]
        else:
            self.momentum_buffers = []

    # ------------------------------------------------------------------
    def _get_dataset(self) -> Tuple[IterableDataset, str]:
        ds = load_dataset(
            self.config.dataset_name,
            name=self.config.subset,
            split="train",
            streaming=True,
        )
        first = next(iter(ds.take(1)))
        if self.config.text_field:
            if self.config.text_field not in first:
                raise ValueError(f"Dataset has no field '{self.config.text_field}'")
            field = self.config.text_field
        else:
            if "text" in first:
                field = "text"
            elif "content" in first:
                field = "content"
            else:
                raise ValueError(
                    "Dataset must contain a 'text' or 'content' field; use text_field param"
                )
        return ds, field

    # ------------------------------------------------------------------
    def _tokenize_batch(self, examples, text_field: str, seq_len: int):
        texts = [ex[text_field] for ex in examples]
        tokens = self.tokenizer(
            texts,
            max_length=seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tokens["labels"] = tokens["input_ids"].clone()
        return tokens

    # ------------------------------------------------------------------
    def train(self):
        cfg = self.config
        progress_bar = tqdm(
            range(cfg.max_steps * cfg.diloco_loops), disable=not self.is_main
        )
        self.model.train()
        for _ in range(cfg.diloco_loops):
            step = 0
            for batch in self.dataloader:
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        step += 1
                        progress_bar.update(1)
                        if self.is_main and step % 20 == 0:
                            progress_bar.set_description(
                                f"loss {loss.item():.4f}"
                            )
                        if step >= cfg.max_steps:
                            break
            if self.device_mesh is not None:
                barrier()
            with torch.no_grad():
                for i, (p, prev) in enumerate(zip(self.model.parameters(), self.prev_params)):
                    grad = p.data.detach().cpu() - prev
                    if cfg.outer_momentum > 0:
                        buf = self.momentum_buffers[i]
                        buf.mul_(cfg.outer_momentum).add_(grad)
                        grad = buf
                    p.grad = grad.to(p.device)
                    prev.copy_(p.data.detach().cpu())
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
            for p in self.model.parameters():
                p.grad = None
        self._save_final()

    # ------------------------------------------------------------------
    def _save_final(self):
        self.accelerator.wait_for_everyone()
        unwrapped = self.accelerator.unwrap_model(self.model)
        state_dict = self.accelerator.get_state_dict(self.model)
        if self.is_main:
            os.makedirs(self.config.output_dir, exist_ok=True)
            unwrapped.save_pretrained(self.config.output_dir, state_dict=state_dict)
            self.tokenizer.save_pretrained(self.config.output_dir)
            print(f"\n✅  Model saved to: {self.config.output_dir}")


__all__ = ["DilocoFSDPTrainer", "TrainerConfig"]
