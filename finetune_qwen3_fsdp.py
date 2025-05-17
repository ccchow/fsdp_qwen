"""
Fine-tune Qwen-3 0.6B with Fully Sharded Data Parallel (FSDP) via Hugging Face Accelerate.

This script demonstrates core FSDP integration on a small subset of FineWeb-Edu:
  1. Stream a limited number of examples
  2. Tokenize and build a DataLoader
  3. Wrap model, optimizer, and DataLoader with Accelerate (reads FSDP config)
  4. Run manual training loop with gradient accumulation
  5. Save the resulting model and tokenizer (FSDP-aware)

Usage:
  # First configure accelerate for FSDP via `accelerate config`
  accelerate launch python finetune_qwen3_fsdp.py \
      --dataset_name HuggingFaceFW/fineweb-edu \
      --subset_name sample-10BT \
      --max_samples 128 \
      --output_dir fsdp-qwen3-tiny
"""
import argparse
import itertools
import math
import os

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from accelerate import Accelerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen-3 0.6B with FSDP via Accelerate"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu",
        help="HF dataset repo"
    )
    parser.add_argument(
        "--subset_name", type=str, default="sample-10BT",
        help="Dataset subset name"
    )
    parser.add_argument(
        "--max_samples", type=int, default=128,
        help="Max number of samples to load"
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3-0.6B",
        help="Pretrained model"
    )
    parser.add_argument(
        "--output_dir", type=str, default="fsdp-qwen3-tiny",
        help="Where to save the fine-tuned model"
    )
    parser.add_argument(
        "--seq_length", type=int, default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=8,
        help="Gradient accumulation steps"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Initialize Accelerator (reads fsdp config from `accelerate config`)
    accelerator = Accelerator()

    # Stream and collect samples
    ds_stream = load_dataset(
        args.dataset_name,
        name=args.subset_name,
        split="train",
        streaming=True,
    )
    samples = list(itertools.islice(ds_stream, args.max_samples))
    if not samples:
        raise RuntimeError("No samples loaded; check dataset_name/subset_name.")
    # Identify text field
    keys = samples[0].keys()
    text_key = "text" if "text" in keys else "content" if "content" in keys else None
    if text_key is None:
        raise KeyError(f"No text/content field in example: {keys}")
    texts = [ex[text_key] for ex in samples]
    ds = Dataset.from_dict({"text": texts})

    # Model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # Tokenization
    def tokenize_fn(ex):
        return tokenizer(
            ex["text"], truncation=True,
            max_length=args.seq_length,
            padding="max_length",
        )

    tokenized = ds.map(
        tokenize_fn, batched=False, remove_columns=["text"]
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_dataloader = DataLoader(
        tokenized,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate
    )

    # Prepare everything with Accelerator
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # Training loop
    total_steps = math.ceil(
        len(train_dataloader) * args.epochs / args.gradient_accumulation_steps
    )
    if accelerator.is_local_main_process:
        print(f"Beginning training for {args.epochs} epochs ({total_steps} updates)")

    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
    # Save model and tokenizer (FSDP-aware)
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_local_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        unwrapped_model.save_pretrained(
            args.output_dir, save_function=accelerator.save
        )
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
