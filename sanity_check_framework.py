#!/usr/bin/env python
"""Quick sanity check for the DilocoFSDPTrainer."""

import argparse
import math
import time
import torch
from torch.utils.data import DataLoader

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
    parser.add_argument(
        "--eval_batches",
        type=int,
        default=0,
        help="Number of batches to run for evaluation after training",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Print training throughput information",
    )
    return parser.parse_args()


def evaluate(trainer: DilocoFSDPTrainer, num_batches: int) -> None:
    """Run a few batches in eval mode and print perplexity.

    A fresh dataloader is constructed so evaluation starts from the beginning of
    the streaming dataset. Any non-finite losses are skipped. If no valid losses
    are found, a warning is printed instead of raising an error.
    """
    trainer.model.eval()

    eval_loader = DataLoader(
        trainer.raw_dataset,
        batch_size=trainer.config.batch_size,
        num_workers=trainer.config.num_workers,
        collate_fn=trainer.collate_fn,
    )

    losses: list[float] = []
    for i, batch in enumerate(eval_loader):
        if i >= num_batches:
            break
        with torch.no_grad():
            outputs = trainer.model(**batch)
            loss = outputs.loss.item()
            if math.isfinite(loss):
                losses.append(loss)

    trainer.model.train()

    if not losses:
        print("\nEval produced NaN/Inf loss.")
        return

    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(mean_loss)
    print(f"\nEval perplexity over {len(losses)} batches: {ppl:.2f}")


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
    start = time.time()
    trainer.train()
    end = time.time()
    if args.benchmark:
        tokens = args.seq_len * args.batch_size * args.steps
        throughput = tokens / (end - start)
        print(f"\nTrained {tokens} tokens in {end - start:.2f}s ({throughput:.2f} tokens/s)")
    if args.eval_batches > 0:
        evaluate(trainer, args.eval_batches)


if __name__ == "__main__":
    main()
