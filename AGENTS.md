# Agent Instructions

This repository contains example training scripts for Qwen models using Fully Sharded Data Parallel (FSDP). The `DilocoFSDPTrainer` in `diloco_fsdp_framework.py` provides the core functionality, while small wrappers such as `finetune_qwen_fsdp.py` launch training. Unit tests live in `tests/`.

## Development Guidelines

* Follow standard Python style (PEP 8) with 4‑space indents. Keep lines under 120 characters when practical.
* Public functions and classes should include short docstrings.
* Keep implementations simple and similar to the existing code base.

## Programmatic Checks

After modifying any Python code or documentation, run the unit tests:

```bash
PYTHONPATH=. pytest -q
```

All tests should pass before committing changes.
