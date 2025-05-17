#!/usr/bin/env bash
set -e
################################################################################
# One-time setup for Qwen fine-tuning on a single NVIDIA 3090 (24 GB)          #
################################################################################

# ---------- 1.  Python environment ----------
# Comment-out whichever method you don’t use.

## (a) Conda -------------------------------------------------------------------
conda create -y -n qwen-finetune python=3.10
conda activate qwen-finetune

## (b) Or plain venv -----------------------------------------------------------
# python -m venv qwen-finetune
# source qwen-finetune/bin/activate

# ---------- 2.  Packages ------------------------------------------------------
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# HF + PEFT
pip install transformers accelerate datasets peft tqdm

# (Optional) flash-attention-2 for faster long-ctx training
# pip install flash-attn --no-build-isolation

# ---------- 3.  Hugging Face login (optional) ---------------------------------
# huggingface-cli login

# ---------- 4.  Accelerate config --------------------------------------------
# Use the provided `fsdp_single_gpu.yaml` as a starting point or run
# `accelerate config` to create your own configuration.

echo "✅  Environment ready.  To fine-tune, run:"
echo "   accelerate launch --config_file=fsdp_single_gpu.yaml finetune_qwen_fsdp.py"
