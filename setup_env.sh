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

# ---------- 4.  Create non-interactive Accelerate config ----------------------
cat <<'YAML' > fsdp_single_gpu.yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
mixed_precision: bf16      # or fp16 if your driver/PyTorch lacks bf16
num_processes: 1
num_machines: 1
machine_rank: 0
gpu_ids: 0
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_offload_params: true
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: Qwen2DecoderLayer
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_use_orig_params: false
  fsdp_sync_module_states: true
YAML

echo "✅  Environment ready.  To fine-tune, run:"
echo "   accelerate launch --config_file=fsdp_single_gpu.yaml finetune_qwen_fsdp.py"
