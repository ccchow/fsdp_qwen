#!/usr/bin/env python
"""Example fine-tuning script using the generic DiLoCo/FSDP framework."""

import argparse

from diloco_fsdp_framework import DilocoFSDPTrainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a model with DiLoCo/FSDP"
    )
    parser.add_argument("--model_name", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset_name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", default="sample-10BT")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument(
        "--diloco_loops",
        type=int,
        default=1,
        help="Number of outer Diloco loops to run before exit",
    )
    parser.add_argument(
        "--outer_lr",
        type=float,
        default=1e-3,
        help="Learning rate for the outer Diloco SGD optimizer",
    )
    parser.add_argument(
        "--outer_momentum",
        type=float,
        default=0.0,
        help="Momentum used by the outer Diloco SGD optimizer",
    )
    parser.add_argument(
        "--outer_lr_schedule",
        default=None,
        help="Scheduler type for the outer optimizer (e.g. linear, cosine)",
    )
    parser.add_argument(
        "--outer_grad_clip",
        type=float,
        default=None,
        help="Max norm for outer gradients (clip before outer step)",
    )
    parser.add_argument(
        "--text_field",
        default=None,
        help="Dataset field containing the text (defaults to auto-detect)",
    )
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
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_steps=args.max_steps,
        diloco_loops=args.diloco_loops,
        outer_lr=args.outer_lr,
        outer_momentum=args.outer_momentum,
        outer_lr_schedule=args.outer_lr_schedule,
        outer_grad_clip=args.outer_grad_clip,
        text_field=args.text_field,
    )
    trainer = DilocoFSDPTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
