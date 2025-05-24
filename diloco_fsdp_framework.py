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
from dataclasses import dataclass, field
from functools import partial
from typing import Iterable, Tuple, Optional
from copy import deepcopy

import torch
from datasets import load_dataset, IterableDataset
import torch.optim as optim
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
    outer_lr_schedule: Optional[str] = None
    outer_grad_clip: Optional[float] = None
    text_field: Optional[str] = None
    checkpoint_steps: int = 0
    resume_from: Optional[str] = None
    log_with: Optional[str] = None
    log_dir: Optional[str] = None
    eval_batches: int = 0
    inner_opt: str = "AdamW"
    inner_opt_kwargs: dict = field(default_factory=dict)
    outer_opt: str = "SGD"
    outer_opt_kwargs: dict = field(default_factory=dict)
    num_workers: int = 0
    dynamic_batch: bool = False
    seed: int = 0
    shuffle_buffer: int = 10_000

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.diloco_loops <= 0:
            raise ValueError("diloco_loops must be > 0")
        if not 0.0 <= self.outer_momentum <= 1.0:
            raise ValueError("outer_momentum must be between 0.0 and 1.0")
        if self.lr <= 0 or self.outer_lr <= 0:
            raise ValueError("learning rate values must be positive")


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
            self._tokenize_batch,
            text_field=self.text_field,
            seq_len=config.seq_len,
            dynamic_batch=config.dynamic_batch,
        )
        self.dataloader = DataLoader(
            self.raw_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=self.collate_fn,
        )

        # --- Optimizer & scheduler ---
        try:
            inner_cls = getattr(optim, config.inner_opt)
        except AttributeError as e:
            raise ValueError(f"Unknown optimizer: {config.inner_opt}") from e

        inner_kwargs = {"lr": config.lr, "weight_decay": 0.01}
        inner_kwargs.update(config.inner_opt_kwargs)
        self.optimizer = inner_cls(self.model.parameters(), **inner_kwargs)
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

        try:
            outer_cls = getattr(optim, config.outer_opt)
        except AttributeError as e:
            raise ValueError(f"Unknown optimizer: {config.outer_opt}") from e

        outer_kwargs = {"lr": config.outer_lr}
        if "momentum" not in config.outer_opt_kwargs:
            outer_kwargs["momentum"] = config.outer_momentum
        outer_kwargs.update(config.outer_opt_kwargs)
        self.outer_optimizer = outer_cls(self.model.parameters(), **outer_kwargs)
        if config.outer_lr_schedule:
            self.outer_lr_scheduler = get_scheduler(
                config.outer_lr_schedule,
                optimizer=self.outer_optimizer,
                num_warmup_steps=0,
                num_training_steps=config.diloco_loops,
            )
        else:
            self.outer_lr_scheduler = None

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

        self.start_step = 0
        if config.resume_from:
            self.start_step = self.load_checkpoint(config.resume_from)

    # ------------------------------------------------------------------
    def _get_dataset(self) -> Tuple[IterableDataset, str]:
        ds = load_dataset(
            self.config.dataset_name,
            name=self.config.subset,
            split="train",
            streaming=True,
        )
        ds = ds.shuffle(self.config.shuffle_buffer, seed=self.config.seed)
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
    def _tokenize_batch(self, examples, text_field: str, seq_len: int, dynamic_batch: bool):
        texts = [ex[text_field] for ex in examples]
        tokens = self.tokenizer(
            texts,
            max_length=seq_len,
            truncation=True,
            padding=True if dynamic_batch else "max_length",
            return_tensors="pt",
        )
        tokens["labels"] = tokens["input_ids"].clone()
        return tokens

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str, step: int) -> None:
        """Save model and optimizer states."""
        self.accelerator.wait_for_everyone()
        state = {
            "model": self.accelerator.get_state_dict(self.model),
            "optimizer": deepcopy(self.optimizer.state_dict()),
            "outer_optimizer": deepcopy(self.outer_optimizer.state_dict()),
            "momentum_buffer": self.momentum_buffer.clone() if self.momentum_buffer is not None else None,
            "step": step,
        }
        if self.is_main:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state, path)

    # ------------------------------------------------------------------
    def load_checkpoint(self, path: str) -> int:
        """Load model and optimizer states and return the stored step."""
        ckpt = torch.load(path, map_location="cpu")
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.outer_optimizer.load_state_dict(ckpt["outer_optimizer"])
        if self.momentum_buffer is not None and ckpt.get("momentum_buffer") is not None:
            self.momentum_buffer.copy_(ckpt["momentum_buffer"])
        return int(ckpt.get("step", 0))

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
        total_steps = cfg.max_steps * cfg.diloco_loops
        progress_bar = tqdm(total=total_steps, disable=not self.is_main)
        start = getattr(self, "start_step", 0)
        if start:
            progress_bar.update(start)
        self.model.train()
        step = start
        try:
          for _ in range(cfg.diloco_loops):
              inner = 0
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
                          inner += 1
                          self.global_step += 1
                          progress_bar.update(1)
                          if cfg.checkpoint_steps and step % cfg.checkpoint_steps == 0:
                              ckpt = os.path.join(cfg.output_dir, f"checkpoint_{step}.pt")
                              self.save_checkpoint(ckpt, step)
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
                          if inner >= cfg.max_steps:
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
                  handle = None
                  if self.device_mesh is not None:
                      pg = self.device_mesh.get_group()
                      handle = dist.all_reduce(delta_gpu, group=pg, async_op=True)
                  if handle is not None:
                      handle.wait()
                  torch.nn.utils.vector_to_parameters(delta_gpu, self.grad_params)
                  for p, g in zip(self.model.parameters(), self.grad_params):
                      p.grad = g.data
                  self.prev_vector.copy_(curr_vector)
              delta_norm = delta.norm().item()
              if cfg.outer_grad_clip:
                  torch.nn.utils.clip_grad_norm_(
                      self.model.parameters(), cfg.outer_grad_clip
                  )
              self.outer_optimizer.step()
              scheduler = getattr(self, "outer_lr_scheduler", None)
              if scheduler is not None:
                  scheduler.step()
              self.outer_optimizer.zero_grad()
              if self.is_main:
                  self._log({"outer/delta_norm": delta_norm}, self.global_step)
                  if cfg.eval_batches > 0:
                      val_loss = self._validate(cfg.eval_batches)
                      self._log({"val/loss": val_loss}, self.global_step)
              for p in self.model.parameters():
                  p.grad = None
        except KeyboardInterrupt:
            if self.is_main:
                print("\n⚠️  Training interrupted. Saving checkpoint...")
            self._save_final()
            return
        except RuntimeError as e:
            if self.is_main:
                print(f"\n❌  RuntimeError during training: {e}")
                print("Attempting graceful shutdown...")
            self.accelerator.wait_for_everyone()
            return

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
