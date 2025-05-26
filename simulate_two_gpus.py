#!/usr/bin/env python
"""Launch a two-process FSDP run on a single GPU.

This utility spawns two processes that each use the same GPU so that the
DilocoFSDPTrainer can be tested locally without multiple devices. It runs a very
small training loop similar to ``sanity_check_framework.py``.
"""

from __future__ import annotations

import argparse
import os
from torch.multiprocessing import spawn
import torch

from diloco_fsdp_framework import DilocoFSDPTrainer, TrainerConfig


def run(rank: int, args: argparse.Namespace) -> None:
    """Worker function executed in each process."""
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "2"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        backend = "nccl"
    else:
        backend = "gloo"

    torch.distributed.init_process_group(backend=backend, rank=rank, world_size=2)

    cfg = TrainerConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        subset=args.subset,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        grad_accum=1,
        lr=1e-4,
        max_steps=args.steps,
        diloco_loops=1,
    )
    trainer = DilocoFSDPTrainer(cfg)
    trainer.train()

    torch.distributed.destroy_process_group()


def parse_args() -> argparse.Namespace:
    """Return CLI arguments."""
    parser = argparse.ArgumentParser(description="Simulate two GPUs for FSDP")
    parser.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset_name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", default="sample-10BT")
    parser.add_argument("--output_dir", default="sim-two-gpu")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spawn(run, args=(args,), nprocs=2, join=True)


if __name__ == "__main__":
    main()
