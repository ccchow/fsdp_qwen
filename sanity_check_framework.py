#!/usr/bin/env python
"""Quick sanity check for the DilocoFSDPTrainer."""

import argparse

from diloco_fsdp_framework import DilocoFSDPTrainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tiny training loop to verify DilocoFSDPTrainer"
    )
    parser.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset_name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", default="sample-10BT")
    parser.add_argument("--output_dir", default="sanity-check-out")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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


if __name__ == "__main__":
    main()
