# Qwen FSDP Fine-Tuning Examples

This repository contains simple scripts for fine-tuning Qwen models with [Hugging Face Accelerate](https://github.com/huggingface/accelerate) using Fully Sharded Data Parallel (FSDP). The examples target a single-GPU workstation and are based on the walkthrough in `instruction.md`.
The ``DilocoFSDPTrainer`` uses a ``DeviceMesh`` to synchronize the outer
optimization step across ranks when running with multiple GPUs.

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

After the environment is ready, launch training with the example script
`finetune_qwen_fsdp.py`, which uses the generic `DilocoFSDPTrainer` framework:

```bash
accelerate launch --config_file fsdp_single_gpu.yaml finetune_qwen_fsdp.py --output_dir ./qwen-output
```

See `instruction.md` for a complete walkthrough and more details on dataset preparation.

### Multi-Node Launch

When running on multiple machines, start the script on each node with its
`machine_rank` and the total number of nodes. The trainer builds a per-node
process group so that FSDP shards only among GPUs on the same host.

```bash
# node 0
accelerate launch --num_machines 2 --machine_rank 0 \
    --config_file fsdp_multi_gpu.yaml finetune_qwen_fsdp.py --output_dir /path/to/out

# node 1
accelerate launch --num_machines 2 --machine_rank 1 \
    --config_file fsdp_multi_gpu.yaml finetune_qwen_fsdp.py --output_dir /path/to/out
```

`fsdp_multi_gpu.yaml` should describe the GPU layout for a single node (e.g. 4
processes) and the address of node 0. With this setup, model shards are kept
local to each node while gradients for the outer loop are synchronized across
all nodes.

## Sanity Check

To verify that your environment and dependencies are working, you can run a very
short training loop with `sanity_check_framework.py`:

```bash
python sanity_check_framework.py --output_dir ./tmp-check
```

This will instantiate the `DilocoFSDPTrainer` and perform a single training
step using the sample FineWeb-Edu subset.

## License

This project is licensed under the terms of the MIT License. See [LICENSE](LICENSE) for details.

