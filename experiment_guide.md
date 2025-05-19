# Experiment Guide for `diloco_fsdp_framework.py`

This guide explains how to run experiments and troubleshoot the `DilocoFSDPTrainer` defined in `diloco_fsdp_framework.py`. It assumes that you have already prepared a Python environment with PyTorch, Transformers, and the other dependencies listed in `requirements.txt`.

## 1. Verify the Environment

1. Activate your Python environment and install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Optionally, run `setup_env.sh` for a scripted setup.
3. Check that `pytest` is available for running the unit tests.

## 2. Run the Unit Tests

Use the built-in tests to confirm that key helper methods work as expected:

```bash
PYTHONPATH=. pytest -q
```

All tests should pass before running larger experiments. The suite in `tests/test_diloco_fsdp_framework.py` exercises dataset loading, tokenization, and final checkpoint saving.

## 3. Quick Sanity Check

Before launching lengthy jobs, run the sample training loop to ensure the framework functions correctly with your environment:

```bash
python sanity_check_framework.py --output_dir ./tmp-check
```

This performs a single training step on the FineWeb-Edu subset and writes a small checkpoint in `tmp-check`. If this script fails, troubleshoot your CUDA and dependency setup before continuing.

## 4. Launching Training Jobs

Use the wrapper script `finetune_qwen_fsdp.py` to start a real fine-tuning run. Customize the command below with your output directory and model name:

```bash
accelerate launch --config_file fsdp_single_gpu.yaml \
    finetune_qwen_fsdp.py --model_name Qwen/Qwen1.5-0.5B --output_dir ./qwen-output
```

For multi-GPU setups, edit `fsdp_multi_gpu.yaml` and use `--num_machines`/`--machine_rank` as needed. The trainer automatically builds a per-node process group so FSDP shards only within each machine.

## 5. Troubleshooting Tips

### Dataset Issues
- If `_get_dataset` raises an error about missing fields, specify the correct column with `--text_field`.
- Streaming datasets rely on an internet connection. If you are offline, pre-download the dataset or modify the loader to read local files.

### Gradient Synchronization
- Ensure that all processes can see each other via NCCL when using multiple GPUs. Mismatched NCCL environment variables can lead to hangs.
- The trainer uses a `DeviceMesh` to reduce outer gradients across ranks. If gradients appear inconsistent, check that the mesh spans all participating GPUs and that `LOCAL_WORLD_SIZE` is set correctly.

### Checkpoint Saving
- `_save_final` is called at the end of training. If no files appear in the output directory, verify that the main process has write permission and that the directory path is correct.

## 6. Experiment Variations

- **Sequence length**: Adjust `--seq_len` to study memory usage and training stability.
- **Outer loop iterations**: Increase `--diloco_loops` to apply multiple outer updates per inner loop pass.
- **Optimizer choice**: Edit `diloco_fsdp_framework.py` to swap `AdamW` for a different optimizer if needed.

Keep notes of your changes and results to track what settings lead to successful training.

## 7. Cleaning Up

Large checkpoints can consume significant disk space. Delete old experiment directories once you have collected the metrics you need.

---

For more details about the framework internals, read through `diloco_fsdp_framework.py` and the unit tests in `tests/test_diloco_fsdp_framework.py`.
