#!/usr/bin/env python
"""
Fine-tune Qwen-0.5B on FineWeb-Edu (sample-10BT) using Hugging Face Accelerate
with FSDP + CPU-offload on a single RTX 3090.  About 12 GB GPU RAM at
batch-2×1024.

Usage (single machine):
  accelerate launch --config_file fsdp_single_gpu.yaml finetune_qwen_fsdp.py \
      --output_dir ./qwen-0.5B-fineweb-edu \
      --max_steps 5000               # stop after N steps (optional)

Distributed launch (two machines with four GPUs each):
  accelerate launch --config_file fsdp_multi_gpu.yaml \
      --machine_rank 0 --main_process_ip <node0-ip> \
      finetune_qwen_fsdp.py --output_dir ./qwen-0.5B-fineweb-edu
  # Run again on the second machine with --machine_rank 1
"""

import argparse
import os
import torch
from functools import partial
from datasets import load_dataset, IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from torch.optim import AdamW
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.distributed import barrier
from torch.distributed.device_mesh import DeviceMesh
from tqdm.auto import tqdm

# ----------------------- CLI arguments ----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="Qwen/Qwen3-8B")
parser.add_argument("--dataset_name", default="HuggingFaceFW/fineweb-edu")
parser.add_argument("--subset",      default="sample-10BT")   # subset in HF repo
parser.add_argument("--output_dir",  required=True)
parser.add_argument("--seq_len",     type=int, default=1024)
parser.add_argument("--batch_size",  type=int, default=2)
parser.add_argument("--grad_accum",  type=int, default=8)
parser.add_argument("--lr",          type=float, default=2e-5)
parser.add_argument("--max_steps",   type=int, default=10_000)
parser.add_argument(
    "--diloco_loops",
    type=int,
    default=1,
    help="Number of outer Diloco loops to run before exit",
)
parser.add_argument(
    "--outer_momentum",
    type=float,
    default=0.0,
    help="Nesterov momentum applied at each outer Diloco cycle",
)
parser.add_argument(
    "--text_field",
    default=None,
    help="Dataset field containing the text (defaults to auto-detect)",
)
args = parser.parse_args()

# ----------------------- Accelerator & precision ------------------------------
accelerator = Accelerator()  # reads fsdp_single_gpu.yaml automatically
device      = accelerator.device
is_main     = accelerator.is_main_process

# ----------------------- Tokenizer & model ------------------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float16,
    trust_remote_code=True,
)
model.gradient_checkpointing_enable()   # extra memory savings

# ----------------------- Dataset streaming ------------------------------------
def get_dataset():
    ds = load_dataset(
        args.dataset_name,
        name=args.subset,
        split="train",
        streaming=True,  # no local download
    )
    first = next(iter(ds.take(1)))
    if args.text_field:
        if args.text_field not in first:
            raise ValueError(f"Dataset has no field '{args.text_field}'")
        field = args.text_field
    else:
        if "text" in first:
            field = "text"
        elif "content" in first:
            field = "content"
        else:
            raise ValueError(
                "Dataset must contain a 'text' or 'content' field; use --text_field"
            )
    return ds, field

def tokenize_batch(examples, text_field, seq_len):
    texts  = [ex[text_field] for ex in examples]
    tokens = tokenizer(
        texts,
        max_length=seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    tokens["labels"] = tokens["input_ids"].clone()
    return tokens

raw_dataset, text_field = get_dataset()                       # IterableDataset
collate_fn    = partial(tokenize_batch, text_field=text_field, seq_len=args.seq_len)
dataloader    = DataLoader(raw_dataset, batch_size=args.batch_size,
                            collate_fn=collate_fn)

# ----------------------- Optimizer & scheduler --------------------------------
optimizer     = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
lr_scheduler  = get_scheduler(
    "cosine", optimizer=optimizer,
    num_warmup_steps=100, num_training_steps=args.max_steps,
)

# ----------------------- Prepare (FSDP wrap, DDP, etc.) -----------------------
model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, dataloader, lr_scheduler
)

# Create a DeviceMesh for the outer Diloco loop when running distributed
if accelerator.num_processes > 1:
    device_ids = list(range(accelerator.num_processes))
    device_mesh = DeviceMesh("cuda", device_ids)
else:
    device_mesh = None

# Initialize momentum buffers for outer loops if enabled
if args.outer_momentum > 0:
    momentum_buffers = [torch.zeros_like(p) for p in model.parameters()]
    prev_params = [p.data.clone().detach() for p in model.parameters()]
else:
    momentum_buffers = []
    prev_params = []

# ----------------------- Training loop ----------------------------------------
progress_bar  = tqdm(range(args.max_steps), disable=not is_main)
model.train()
step = 0
outer = 0
while step < args.max_steps and outer < args.diloco_loops:
    for batch in dataloader:
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss    = outputs.loss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                step += 1
                progress_bar.update(1)
                if is_main and step % 20 == 0:
                    progress_bar.set_description(f"loss {loss.item():.4f}")
                if step >= args.max_steps:
                    break
    # Outer step synchronization
    if device_mesh is not None:
        barrier()
    if args.outer_momentum > 0:
        with torch.no_grad():
            for p, v, prev in zip(model.parameters(), momentum_buffers, prev_params):
                delta = p.data - prev
                v.mul_(args.outer_momentum).add_(delta)
                p.data.add_(args.outer_momentum * v)
                prev.copy_(p.data)
    elif prev_params:
        # Keep previous parameters updated even if momentum is disabled
        for prev, p in zip(prev_params, model.parameters()):
            prev.copy_(p.data)
    outer += 1

# ----------------------- Save final checkpoint --------------------------------
accelerator.wait_for_everyone()
unwrapped = accelerator.unwrap_model(model)
state_dict = accelerator.get_state_dict(model)
if is_main:
    os.makedirs(args.output_dir, exist_ok=True)
    unwrapped.save_pretrained(args.output_dir, state_dict=state_dict)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅  Model saved to: {args.output_dir}")
