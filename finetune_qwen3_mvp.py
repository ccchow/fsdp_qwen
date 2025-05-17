"""
Minimal script to fine-tune the Qwen-3 0.6B causal language model on a tiny subset of the FineWeb-Edu dataset.

This serves as a minimal viable product (MVP) demonstrating the core steps:
  1. Stream a small number of examples from FineWeb-Edu via Hugging Face Datasets.
  2. Tokenize and batch the data for causal LM training.
  3. Fine-tune using Hugging Face Trainer API.
  4. Save the resulting model and tokenizer.

Requirements:
  pip install torch transformers datasets

Usage:
  python finetune_qwen3_mvp.py \
      --dataset_name HuggingFaceFW/fineweb-edu \
      --subset_name sample-10BT \
      --max_samples 64 \
      --output_dir fine-tuned-qwen3-tiny
"""
import argparse
import itertools

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen-3 0.6B on a tiny subset of FineWeb-Edu"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu",
        help="Hugging Face dataset repository (e.g., HuggingFaceFW/fineweb-edu)"
    )
    parser.add_argument(
        "--subset_name", type=str, default="sample-10BT",
        help="Subset name of FineWeb-Edu to stream (e.g., sample-10BT)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=64,
        help="Maximum number of examples to load for fine-tuning (MVP)"
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3-0.6B",
        help="Pretrained model identifier"
    )
    parser.add_argument(
        "--output_dir", type=str, default="fine-tuned-qwen3-tiny",
        help="Directory to save fine-tuned model and tokenizer"
    )
    parser.add_argument(
        "--seq_length", type=int, default=512,
        help="Maximum sequence length (tokens)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Per-device training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of training epochs"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading streaming dataset {args.dataset_name}, subset {args.subset_name}...")
    ds_stream = load_dataset(
        args.dataset_name,
        name=args.subset_name,
        split="train",
        streaming=True,
    )
    # Take only a small number of examples for MVP
    samples = list(itertools.islice(ds_stream, args.max_samples))
    if not samples:
        raise RuntimeError("No samples loaded; check dataset_name and subset_name.")
    # Determine text field
    sample0 = samples[0]
    if "text" in sample0:
        text_key = "text"
    elif "content" in sample0:
        text_key = "content"
    else:
        raise KeyError(f"Cannot find 'text' or 'content' field in dataset example: {list(sample0.keys())}")
    texts = [ex[text_key] for ex in samples]
    # Build an in-memory Dataset for Trainer
    ds = Dataset.from_dict({"text": texts})

    print(f"Loading tokenizer and model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    # Set padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    # Enable gradient checkpointing to reduce memory
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # Tokenization function
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.seq_length,
            padding="max_length",
        )

    print("Tokenizing dataset...")
    tokenized = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=500,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving fine-tuned model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
