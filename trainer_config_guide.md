# TrainerConfig Parameter Guide

## Introduction

The `TrainerConfig` dataclass in `diloco_fsdp_framework.py` provides comprehensive control over the DiLoCo (Distributed Local SGD with Compression) training process combined with Fully Sharded Data Parallel (FSDP). This configuration system enables efficient fine-tuning of large language models across single or multiple GPUs.

### Key Concepts

- **Inner Loop**: Standard gradient descent steps using chosen optimizer (e.g., AdamW)
- **Outer Loop**: DiLoCo synchronization steps that coordinate updates across distributed workers
- **FSDP**: Memory-efficient model sharding that enables training larger models than would fit on a single GPU

### Quick Start Configuration

For most use cases, start with these recommended settings and adjust based on your hardware and dataset:

```python
config = TrainerConfig(
    model_name="Qwen/Qwen3-8B",           # or your preferred model
    dataset_name="HuggingFaceFW/fineweb-edu",
    subset="sample-10BT",
    output_dir="./training_output",
    seq_len=2048,                         # 2K context for modern models
    batch_size=1,                         # Start small, increase if memory allows
    grad_accum=16,                        # Effective batch size = 16
    lr=5e-6,                             # Conservative for fine-tuning
    max_steps=1000,                       # Reasonable for experimentation
    diloco_loops=10,                      # Multiple outer loops
    outer_lr=0.7,                        # Standard DiLoCo outer learning rate
    outer_momentum=0.9,                   # Enable momentum for stability
    checkpoint_steps=250,                 # Save every 250 steps
    log_with="tensorboard",               # Enable logging
    eval_batches=50                       # Validation at each outer loop
)
```

## Parameter Reference

This section describes all available fields with their recommended values and use cases.

### Core Model and Data Parameters

- **model_name**: Name or path of the Hugging Face model to load.
  - *Best Practice*: Use official model names like `"Qwen/Qwen3-8B"`, `"microsoft/DialoGPT-medium"`, or local paths
  - *Examples*: `"Qwen/Qwen3-8B"`, `"meta-llama/Llama-2-7b-hf"`, `"./local-model"`

- **dataset_name**: Dataset identifier passed to `datasets.load_dataset`.
  - *Best Practice*: Use well-curated datasets; streaming datasets work well for large corpora
  - *Examples*: `"HuggingFaceFW/fineweb-edu"`, `"OpenAssistant/oasst1"`, `"squad"`

- **subset**: Dataset subset or configuration name.
  - *Best Practice*: Start with smaller subsets for experimentation
  - *Examples*: `"sample-10BT"`, `"train"`, `"english"`

- **output_dir**: Directory where checkpoints and the final model are saved.
  - *Best Practice*: Use descriptive names with timestamps for organization
  - *Examples*: `"./qwen-finance-2024-05-24"`, `"/shared/models/experiment-001"`

- **text_field**: Name of the dataset field containing text (auto-detected if omitted).
  - *Best Practice*: Specify explicitly for clarity, especially with custom datasets
  - *Examples*: `"text"`, `"content"`, `"instruction"`

### Sequence and Batch Configuration

- **seq_len**: Maximum sequence length for tokenization (default `1024`).
  - *Best Practice*: Use 2048-4096 for modern models, 1024 for experimentation
  - *Memory Impact*: Quadratic memory usage - reduce if hitting OOM errors
  - *Recommended*: `2048` for most use cases, `4096` for high-memory setups

- **batch_size**: Per-device batch size fed to the DataLoader.
  - *Best Practice*: Start with 1-2, increase gradually while monitoring memory
  - *Recommended*: `1` (safe), `2-4` (if memory allows), `8+` (high-memory GPUs)

- **grad_accum**: Number of gradient accumulation steps.
  - *Best Practice*: Effective batch size = batch_size × grad_accum × num_gpus
  - *Recommended*: `8-32` for most setups, targeting effective batch size of 32-128

- **dynamic_batch**: Enable dynamic batch lengths instead of padding to `seq_len`.
  - *Best Practice*: Enable for efficiency with variable-length sequences
  - *Recommended*: `True` for most datasets, `False` for uniform sequence lengths

### Learning Rate Configuration

- **lr**: Learning rate for the inner optimizer.
  - *Best Practice*: Use lower rates for fine-tuning, higher for pre-training
  - *Recommended*: `5e-6` to `1e-5` (fine-tuning), `1e-4` to `3e-4` (pre-training)

- **outer_lr**: Learning rate for the outer optimizer.
  - *Best Practice*: DiLoCo typically uses 0.7-1.0 for good convergence
  - *Recommended*: `0.7` (conservative), `1.0` (standard), `1.5` (aggressive)

- **outer_momentum**: Momentum factor for the outer optimizer (0 to 1).
  - *Best Practice*: Use momentum for stability, especially with noisy gradients
  - *Recommended*: `0.9` (standard), `0.95` (high momentum), `0.0` (no momentum)

- **outer_lr_schedule**: Optional scheduler name for the outer optimizer (e.g. `"cosine"`).
  - *Best Practice*: Use cosine decay for long training runs
  - *Options*: `"cosine"`, `"linear"`, `"polynomial"`, `None`

### Training Loop Configuration

- **max_steps**: Inner-loop steps per outer iteration.
  - *Best Practice*: Balance between local optimization and communication overhead
  - *Recommended*: `500-2000` (single GPU), `1000-5000` (multi-GPU)

- **diloco_loops**: Number of outer optimization loops to run.
  - *Best Practice*: More loops = longer training, better convergence
  - *Recommended*: `10-50` (experimentation), `100-500` (full training)

### Optimization Configuration

- **inner_opt**: Class name of the inner optimizer from `torch.optim` (default `"AdamW"`).
  - *Best Practice*: AdamW for most cases, SGD for specific research needs
  - *Options*: `"AdamW"` (recommended), `"Adam"`, `"SGD"`, `"RMSprop"`

- **inner_opt_kwargs**: Extra keyword arguments for the inner optimizer.
  - *Best Practice*: Set weight_decay for regularization
  - *Example*: `{"weight_decay": 0.01, "eps": 1e-8}`

- **outer_opt**: Class name of the outer optimizer (default `"SGD"`).
  - *Best Practice*: SGD is standard for DiLoCo, matches research implementations
  - *Recommended*: Keep as `"SGD"` unless experimenting

- **outer_opt_kwargs**: Extra keyword arguments for the outer optimizer.
  - *Example*: `{"nesterov": True}` for Nesterov momentum

- **outer_grad_clip**: If set, clip outer gradients to this norm.
  - *Best Practice*: Use when experiencing training instability
  - *Recommended*: `1.0` (conservative), `5.0` (standard), `None` (no clipping)

### Checkpointing and Logging

- **checkpoint_steps**: Save a checkpoint every N inner steps (0 disables).
  - *Best Practice*: Balance storage space with recovery needs
  - *Recommended*: `250-1000` (frequent), `2000-5000` (less frequent)

- **resume_from**: Path to a checkpoint file to resume training.
  - *Best Practice*: Use absolute paths for reliability
  - *Example*: `"/path/to/checkpoint/step-1000"`

- **log_with**: Logging backend, either `"tensorboard"` or `"wandb"`.
  - *Best Practice*: TensorBoard for local development, W&B for team collaboration
  - *Recommended*: `"tensorboard"` (simple), `"wandb"` (advanced features)

- **log_dir**: Directory for log files when using TensorBoard.
  - *Best Practice*: Organize logs with meaningful names
  - *Default*: `{output_dir}/logs`

### Evaluation and Data Loading

- **eval_batches**: Number of batches for validation at the end of each outer loop.
  - *Best Practice*: Enough batches for stable metrics, not too many to slow training
  - *Recommended*: `50-200` batches, `0` to disable

- **num_workers**: Number of worker processes for the DataLoader.
  - *Best Practice*: Match to CPU cores, but avoid oversubscription
  - *Recommended*: `4-8` (most systems), `0` (debugging), `16+` (high-CPU systems)

- **seed**: Random seed used when shuffling the dataset.
  - *Best Practice*: Set for reproducibility, vary for different runs
  - *Recommended*: `42`, `2024`, or any consistent value

- **shuffle_buffer**: Buffer size for streaming dataset shuffling.
  - *Best Practice*: Larger buffers = better randomization but more memory
  - *Recommended*: `10000` (default), `50000` (large datasets), `1000` (memory-constrained)

## Configuration Examples

### Small-Scale Experimentation (Single GPU, 8GB VRAM)
```python
config = TrainerConfig(
    model_name="microsoft/DialoGPT-small",
    dataset_name="squad",
    subset="train[:1000]",  # Small subset
    output_dir="./experiment",
    seq_len=512,
    batch_size=1,
    grad_accum=8,
    lr=1e-5,
    max_steps=100,
    diloco_loops=5,
    outer_lr=0.7,
    checkpoint_steps=50,
    log_with="tensorboard"
)
```

### Production Fine-Tuning (Multiple GPUs, 24GB+ VRAM)
```python
config = TrainerConfig(
    model_name="Qwen/Qwen3-8B",
    dataset_name="HuggingFaceFW/fineweb-edu",
    subset="sample-10BT",
    output_dir="./qwen-production-run",
    seq_len=4096,
    batch_size=4,
    grad_accum=16,
    lr=5e-6,
    max_steps=2000,
    diloco_loops=50,
    outer_lr=1.0,
    outer_momentum=0.9,
    outer_lr_schedule="cosine",
    checkpoint_steps=500,
    log_with="wandb",
    eval_batches=100,
    num_workers=8
)
```

### Memory-Efficient Large Model Training
```python
config = TrainerConfig(
    model_name="meta-llama/Llama-2-13b-hf",
    dataset_name="OpenAssistant/oasst1",
    subset="train",
    output_dir="./llama-efficient",
    seq_len=2048,
    batch_size=1,          # Minimal batch size
    grad_accum=32,         # Large accumulation
    lr=3e-6,              # Conservative LR for large model
    max_steps=1500,
    diloco_loops=20,
    outer_lr=0.5,         # Lower outer LR for stability
    dynamic_batch=True,    # Save memory on padding
    checkpoint_steps=300,
    outer_grad_clip=1.0   # Gradient clipping for stability
)
```

## Performance Optimization Tips

### Memory Management
- Start with `batch_size=1` and increase gradually
- Use `dynamic_batch=True` for variable-length sequences  
- Enable gradient checkpointing in your FSDP config for large models
- Monitor GPU memory usage with `nvidia-smi`

### Training Efficiency
- Balance `max_steps` vs `diloco_loops`: fewer, longer inner loops reduce communication overhead
- Use `num_workers=4-8` for faster data loading on multi-core systems
- Set `shuffle_buffer` based on available RAM (larger = better randomization)

### Convergence and Stability
- Use `outer_momentum=0.9` for smoother outer optimization
- Enable `outer_grad_clip=1.0` if seeing loss spikes
- Start with conservative learning rates and increase gradually
- Use `eval_batches > 0` to monitor validation loss

### Debugging
- Set `num_workers=0` to simplify debugging data loading issues
- Use smaller `seq_len` values during development
- Enable frequent checkpointing (`checkpoint_steps=100`) during experimentation

## Troubleshooting Common Issues

**Out of Memory Errors:**
- Reduce `batch_size`, `seq_len`, or increase `grad_accum`
- Enable `dynamic_batch=True`
- Check FSDP configuration for gradient checkpointing

**Slow Training:**
- Increase `num_workers` for faster data loading
- Use larger `batch_size` if memory allows
- Reduce `eval_batches` frequency

**Poor Convergence:**
- Lower learning rates (`lr`, `outer_lr`)
- Enable `outer_momentum`
- Add `outer_grad_clip=1.0`
- Increase `max_steps` for more local optimization

These parameters correspond to the dataclass definition in `diloco_fsdp_framework.py` and may be extended in future versions.
