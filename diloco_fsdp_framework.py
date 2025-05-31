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
from typing import Optional, Union, Callable
from copy import deepcopy

import torch
from datasets import IterableDataset
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributed.device_mesh import DeviceMesh
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin
from tqdm.auto import tqdm

from dataset_utils import DatasetLoader
from outer_optimizer import OuterOptimizer


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
    
    # FSDP2-specific configuration
    fsdp_reshard_after_forward: bool = True
    fsdp_auto_wrap_policy: Optional[Union[Callable, str]] = "transformer_based_wrap"
    fsdp_cpu_offload: bool = False
    fsdp_mixed_precision: Optional[str] = "bf16"
    fsdp_transformer_layer_cls_to_wrap: Optional[list[str]] = None
    fsdp_min_num_params: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.diloco_loops <= 0:
            raise ValueError("diloco_loops must be > 0")
        if not 0.0 <= self.outer_momentum <= 1.0:
            raise ValueError("outer_momentum must be between 0.0 and 1.0")
        if self.lr <= 0 or self.outer_lr <= 0:
            raise ValueError("learning rate values must be positive")
        if self.fsdp_auto_wrap_policy not in [None, "transformer_based_wrap", "size_based_wrap", "no_wrap"] and not callable(self.fsdp_auto_wrap_policy):
            raise ValueError("fsdp_auto_wrap_policy must be a callable or one of: transformer_based_wrap, size_based_wrap, no_wrap")


class DilocoFSDPTrainer:
    """Trainer implementing DiLoCo + FSDP for generic models and datasets."""

    def __init__(self, config: TrainerConfig, accelerator: Optional[Accelerator] = None):
        self.config = config

        if accelerator is None:
            # Set up mixed precision policy for FSDP2
            mixed_precision_policy = None
            if config.fsdp_mixed_precision:
                dtype_map = {
                    "fp16": torch.float16,
                    "bf16": torch.bfloat16,
                }
                if config.fsdp_mixed_precision in dtype_map:
                    mixed_precision_policy = {
                        "param_dtype": dtype_map[config.fsdp_mixed_precision],
                        "reduce_dtype": dtype_map[config.fsdp_mixed_precision],
                        "buffer_dtype": dtype_map[config.fsdp_mixed_precision],
                    }

            # Configure FSDP plugin with FSDP2 options
            fsdp_kwargs = {
                "fsdp_version": 2,
                "reshard_after_forward": config.fsdp_reshard_after_forward,
                "auto_wrap_policy": config.fsdp_auto_wrap_policy,
                "cpu_offload": config.fsdp_cpu_offload,
            }

            if mixed_precision_policy:
                fsdp_kwargs["mixed_precision_policy"] = mixed_precision_policy

            if config.fsdp_transformer_layer_cls_to_wrap:
                fsdp_kwargs["transformer_cls_names_to_wrap"] = config.fsdp_transformer_layer_cls_to_wrap

            if config.fsdp_min_num_params is not None:
                fsdp_kwargs["min_num_params"] = config.fsdp_min_num_params

            self._fsdp_plugin = FullyShardedDataParallelPlugin(**fsdp_kwargs)

            self.accelerator = Accelerator(
                gradient_accumulation_steps=config.grad_accum,
                fsdp_plugin=self._fsdp_plugin,
                mixed_precision=config.fsdp_mixed_precision,
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
        self.dataset_loader = DatasetLoader(
            config.dataset_name,
            config.subset,
            text_field=config.text_field,
            shuffle_buffer=config.shuffle_buffer,
            seed=config.seed,
        )
        self.raw_dataset, self.text_field = self.dataset_loader.load()
        self.collate_fn = partial(
            DatasetLoader.tokenize_batch,
            self.tokenizer,
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
        base_outer = outer_cls(self.model.parameters(), **outer_kwargs)
        self.outer_optimizer = base_outer
        if config.outer_lr_schedule:
            outer_sched = get_scheduler(
                config.outer_lr_schedule,
                optimizer=base_outer,
                num_warmup_steps=0,
                num_training_steps=config.diloco_loops,
            )
        else:
            outer_sched = None

        if self.accelerator.num_processes > 1:
            device_ids = list(range(self.accelerator.num_processes))
            self.device_mesh = DeviceMesh("cuda", device_ids)
        else:
            self.device_mesh = None

        self.outer_opt = OuterOptimizer(
            self.model,
            base_outer,
            device=self.device,
            device_mesh=self.device_mesh,
            momentum=config.outer_momentum,
            grad_clip=config.outer_grad_clip,
            lr_scheduler=outer_sched,
        )

        self.start_step = 0
        if config.resume_from:
            self.start_step = self.load_checkpoint(config.resume_from)

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str, step: int) -> None:
        """Save model and optimizer states."""
        self.accelerator.wait_for_everyone()
        outer_state = self.outer_opt.state_dict()
        state = {
            "model": self.accelerator.get_state_dict(self.model),
            "optimizer": deepcopy(self.optimizer.state_dict()),
            "outer_optimizer": outer_state.optimizer,
            "prev_vector": outer_state.prev_vector,
            "momentum_buffer": outer_state.momentum_buffer,
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
        self.outer_opt.optimizer.load_state_dict(ckpt["outer_optimizer"])
        self.outer_opt.prev_vector.copy_(ckpt["prev_vector"])
        if self.outer_opt.momentum_buffer is not None and ckpt.get("momentum_buffer") is not None:
            self.outer_opt.momentum_buffer.copy_(ckpt["momentum_buffer"])
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
              delta_norm = self.outer_opt.step()
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
