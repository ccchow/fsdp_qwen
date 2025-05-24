# TrainerConfig Parameter Guide

This document describes the fields of the `TrainerConfig` dataclass in `diloco_fsdp_framework.py`. Use these options to control training when instantiating `DilocoFSDPTrainer`.

- **model_name**: Name or path of the Hugging Face model to load.
- **dataset_name**: Dataset identifier passed to `datasets.load_dataset`.
- **subset**: Dataset subset or configuration name.
- **output_dir**: Directory where checkpoints and the final model are saved.
- **seq_len**: Maximum sequence length for tokenization (default `1024`).
- **batch_size**: Per-device batch size fed to the DataLoader.
- **grad_accum**: Number of gradient accumulation steps.
- **lr**: Learning rate for the inner optimizer.
- **max_steps**: Inner-loop steps per outer iteration.
- **diloco_loops**: Number of outer optimization loops to run.
- **outer_lr**: Learning rate for the outer optimizer.
- **outer_momentum**: Momentum factor for the outer optimizer (0 to 1).
- **outer_lr_schedule**: Optional scheduler name for the outer optimizer (e.g. `"cosine"`).
- **outer_grad_clip**: If set, clip outer gradients to this norm.
- **text_field**: Name of the dataset field containing text (auto-detected if omitted).
- **checkpoint_steps**: Save a checkpoint every N inner steps (0 disables).
- **resume_from**: Path to a checkpoint file to resume training.
- **log_with**: Logging backend, either `"tensorboard"` or `"wandb"`.
- **log_dir**: Directory for log files when using TensorBoard.
- **eval_batches**: Number of batches for validation at the end of each outer loop.
- **inner_opt**: Class name of the inner optimizer from `torch.optim` (default `"AdamW"`).
- **inner_opt_kwargs**: Extra keyword arguments for the inner optimizer.
- **outer_opt**: Class name of the outer optimizer (default `"SGD"`).
- **outer_opt_kwargs**: Extra keyword arguments for the outer optimizer.
- **num_workers**: Number of worker processes for the DataLoader.
- **dynamic_batch**: Enable dynamic batch lengths instead of padding to `seq_len`.
- **seed**: Random seed used when shuffling the dataset.
- **shuffle_buffer**: Buffer size for streaming dataset shuffling.

These parameters correspond to the dataclass definition in `diloco_fsdp_framework.py` and may be extended in future versions.
