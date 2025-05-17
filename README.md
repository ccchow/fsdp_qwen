# Qwen FSDP Fine-Tuning Examples

This repository contains simple scripts for fine-tuning Qwen models with [Hugging Face Accelerate](https://github.com/huggingface/accelerate) using Fully Sharded Data Parallel (FSDP). The examples target a single-GPU workstation and are based on the walkthrough in `instruction.md`.

## Environment Setup

1. Create and activate a Python environment (e.g. conda or `venv`).
2. Install PyTorch with CUDA support and the required libraries:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   pip install transformers accelerate datasets peft tqdm
   ```
3. (Optional) Log in to Hugging Face with `huggingface-cli login`.
4. Copy `fsdp_single_gpu.yaml` or run `accelerate config` to create your own Accelerate configuration.

A helper script `setup_env.sh` demonstrates these steps.

## Running a Fine-Tuning Job

After the environment is ready, launch training with:

```bash
accelerate launch --config_file fsdp_single_gpu.yaml finetune_qwen_fsdp.py --output_dir ./qwen-output
```

See `instruction.md` for a complete walkthrough and more details on dataset preparation.

## License

This project is licensed under the terms of the MIT License. See [LICENSE](LICENSE) for details.

