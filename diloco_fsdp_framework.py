#!/usr/bin/env python
"""Generic DiLoCo/FSDP training framework.

This module provides the :class:`DilocoFSDPTrainer` class which implements a
training loop for decentralized fine-tuning using Fully Sharded Data Parallel
(FSDP) and the DiLoCo outer optimization step. It now uses a
``DeviceMesh`` to synchronize the outer gradients across ranks so that
each worker applies the same update. The original
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
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin
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
    log_with: Optional[str] = None
    log_dir: Optional[str] = None
    eval_batches: int = 0


class DilocoFSDPTrainer:
    """Trainer implementing DiLoCo + FSDP for generic models and datasets."""

    def __init__(self, config: TrainerConfig, accelerator: Optional[Accelerator] = None):
        self.config = config

        if accelerator is None:
            self._fsdp_plugin = FullyShardedDataParallelPlugin()
            self.accelerator = Accelerator(
                gradient_accumulation_steps=config.grad_accum,
                fsdp_plugin=self._fsdp_plugin,
            )
        else:
            self.accelerator = accelerator
            self._fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)

        self.device = self.accelerator.device
        self.is_main = self.accelerator.is_main_process

        self.writer = None
        if config.log_with == "tensorboard" and self.is_main:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = config.log_dir or os.path.join(config.output_dir, "logs")
            self.writer = SummaryWriter(log_dir)
        elif config.log_with == "wandb" and self.is_main:
            try:
                import wandb  # type: ignore

                wandb.init(project="diloco_fsdp", dir=config.output_dir, config=vars(config))
                self.writer = wandb
            except Exception:
                print("wandb logging requested but not available")

        self.global_step = 0

        # Create a per-node process group so FSDP only shards within each node
        self.fsdp_process_group = None
        if self.accelerator.num_processes > 1 and dist.is_initialized():
            local_world = int(os.environ.get("LOCAL_WORLD_SIZE", self.accelerator.num_processes))
            rank = dist.get_rank()
            start = (rank // local_world) * local_world
            ranks = list(range(start, start + local_world))
            self.fsdp_process_group = dist.new_group(ranks=ranks)
            if self._fsdp_plugin is not None:
                self._fsdp_plugin.fsdp_process_group = self.fsdp_process_group

        # --- Tokenizer & model ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, use_fast=False, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16
            if self.accelerator.mixed_precision == "bf16"
            else torch.float16,
            trust_remote_code=True,
        )
        self.model.config.use_cache = False
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

        # Store a compressed snapshot of the parameters to reduce memory
        # overhead.  Using float16 provides a reasonable trade-off between
        # precision and size and avoids cloning the full parameter tensors.
        with torch.no_grad():
            params = [p.detach().cpu().to(torch.float16) for p in self.model.parameters()]
            self.prev_vector = torch.nn.utils.parameters_to_vector(params)
        if config.outer_momentum > 0:
            self.momentum_buffer = torch.zeros_like(self.prev_vector)
        else:
            self.momentum_buffer = None

        self.grad_params = [
            torch.nn.Parameter(torch.zeros_like(p), requires_grad=False)
            for p in self.model.parameters()
        ]

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
    def _log(self, metrics: dict, step: int) -> None:
        """Log metrics to the configured backend."""
        if self.writer is None:
            return
        if self.config.log_with == "tensorboard":
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, step)
        elif self.config.log_with == "wandb":
            self.writer.log(metrics, step=step)

    # ------------------------------------------------------------------
    def _grad_norm(self) -> float:
        norms = [p.grad.detach().float().norm() for p in self.model.parameters() if p.grad is not None]
        if norms:
            return torch.norm(torch.stack(norms)).item()
        return 0.0

    # ------------------------------------------------------------------
    def _validate(self, num_batches: int) -> float:
        """Run a simple validation loop and return mean loss."""
        self.model.eval()
        losses = []
        for i, batch in enumerate(self.dataloader):
            if i >= num_batches:
                break
            with torch.no_grad():
                outputs = self.model(**batch)
                losses.append(outputs.loss.item())
        self.model.train()
        if not losses:
            return float("nan")
        return sum(losses) / len(losses)

    # ------------------------------------------------------------------
    def train(self):
        cfg = self.config
        if not hasattr(self, "global_step"):
            self.global_step = 0
        if not hasattr(self, "writer"):
            self.writer = None
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
                        grad_norm = self._grad_norm()
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        step += 1
                        self.global_step += 1
                        progress_bar.update(1)
                        if self.is_main:
                            lr = self.lr_scheduler.get_last_lr()[0] if hasattr(self.lr_scheduler, "get_last_lr") else self.optimizer.param_groups[0]["lr"]
                            metrics = {
                                "train/loss": loss.item(),
                                "lr": lr,
                                "grad_norm": grad_norm,
                            }
                            self._log(metrics, self.global_step)
                            if step % 20 == 0:
                                progress_bar.set_description(
                                    f"loss {loss.item():.4f}"
                                )
                        if step >= cfg.max_steps:
                            break
            if self.device_mesh is not None:
                barrier()
            with torch.no_grad():
                curr_vector = torch.nn.utils.parameters_to_vector(
                    [p.detach().cpu() for p in self.model.parameters()]
                ).to(self.prev_vector.dtype)
                delta = curr_vector - self.prev_vector
                if cfg.outer_momentum > 0:
                    self.momentum_buffer.mul_(cfg.outer_momentum).add_(delta)
                    delta = self.momentum_buffer
                delta_gpu = delta.to(self.device, dtype=self.grad_params[0].dtype)
                if self.device_mesh is not None:
                    self.device_mesh.all_reduce(delta_gpu)
                    delta_gpu.div_(self.accelerator.num_processes)
                torch.nn.utils.vector_to_parameters(delta_gpu, self.grad_params)
                for p, g in zip(self.model.parameters(), self.grad_params):
                    p.grad = g.data
                self.prev_vector.copy_(curr_vector)
            delta_norm = delta.norm().item()
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
            if self.is_main:
                self._log({"outer/delta_norm": delta_norm}, self.global_step)
                if cfg.eval_batches > 0:
                    val_loss = self._validate(cfg.eval_batches)
                    self._log({"val/loss": val_loss}, self.global_step)
            for p in self.model.parameters():
                p.grad = None
        self._save_final()
        if self.writer is not None and self.config.log_with == "tensorboard":
            self.writer.close()

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
