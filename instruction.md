Great, I’ll prepare a step-by-step guide for fine-tuning the Qwen 0.5B model on the FineWeb-Edu dataset using Hugging Face Accelerate with Fully Sharded Data Parallel (FSDP) enabled on a local machine with a single NVIDIA 3090 GPU.
I’ll walk you through environment setup, model preparation, dataset integration, Accelerate configuration, and training.
I’ll let you know as soon as it’s ready.


# Fine-Tuning Qwen-0.5B on FineWeb-Edu with Accelerate (FSDP on a Single 3090 GPU)

## Step 1: Environment Setup

To get started, set up a Python environment with the necessary libraries and ensure CUDA is configured for your NVIDIA RTX 3090 GPU. You can use a virtual environment or Conda environment for isolation. For example:

* **Python version**: Use Python 3.8+ (Python 3.10 is recommended).
* **PyTorch with CUDA**: Install PyTorch built for CUDA 11.x or 12.x (compatible with the 3090). For instance, on Linux: `pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118`. (Adjust the CUDA version to match your system's, e.g. `cu117` or use Conda’s defaults.)
* **Transformers and Accelerate**: Install the latest Hugging Face Transformers and Accelerate, along with the Datasets library and PEFT (for LoRA) if needed:

  ```bash
  pip install transformers accelerate datasets peft
  ```

  Make sure your `transformers` version is ≥4.37, since Qwen2 models are supported only in recent versions.
* **CUDA Toolkit/Drivers**: Ensure NVIDIA drivers are installed and `nvidia-smi` recognizes your 3090. No separate CUDA toolkit installation is needed if using PyTorch’s wheels, but your driver must support the CUDA version (e.g., CUDA 11.8).

Verify the installation by importing the libraries and checking that the GPU is accessible in PyTorch:

```python
import torch
print(torch.cuda.is_available())        # Should be True
print(torch.cuda.get_device_name(0))    # Should show "NVIDIA GeForce RTX 3090"
```

Also, log in to Hugging Face if needed (for dataset/model access) using `huggingface-cli login`. With the environment ready, you can proceed to data preparation.

## Step 2: Preparing the FineWeb-Edu Dataset

&#x20;**FineWeb-Edu** is a massive high-quality text dataset (1.3 trillion tokens) distilled from CommonCrawl, focusing on educational content. It consists of web pages filtered by an educational quality classifier (a LLaMA3-70B model was used to rank pages, and only top-quality pages were kept). Using this dataset will effectively continue pre-training or fine-tuning the Qwen model on educational web text.

**Downloading/Accessing the Data:** The dataset is available on Hugging Face Hub (repository: `HuggingFaceFW/fineweb-edu`). **Do not attempt to download the entire 1.3T-token dataset to disk on a single machine** – it’s enormous (the full dataset is many terabytes). Instead, take advantage of Hugging Face’s streaming or use the provided smaller samples. The FineWeb-Edu repo provides sample subsets of various sizes (e.g., \~10B, 100B, 350B tokens) for easier use. For example, a *10 billion token* sample (`sample-10BT`) is available (still large but manageable for demonstration).

**Using Hugging Face Datasets:** You can load FineWeb-Edu with the `datasets` library. Here’s how to load a smaller sample via streaming:

```python
from datasets import load_dataset

# Stream the 10B-token sample to avoid full download
dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu", 
    name="sample-10BT",   # choose a subset; e.g. "sample-10BT" for ~10B tokens
    split="train", 
    streaming=True
)
```

This will create an **iterable dataset** that yields examples on-the-fly. Each example in FineWeb-Edu is a web page (or document) with multiple fields. The key field we need is the **text content**. For FineWeb-Edu, the text content is likely under a `"text"` or `"content"` field (the dataset includes metadata like URLs, language code, and quality scores, but we only need the actual text for language modeling). You can verify the field names by examining one sample:

```python
sample = next(iter(dataset))
print(sample.keys())
# e.g., dict_keys(['text', 'url', 'language', 'score', ...])
print(sample["text"][:500])  # preview first 500 characters of text
```

If the field is not named `"text"`, adjust accordingly (e.g., some versions might use `"content"`). Assuming it’s `"text"`, we will use that for training.

**Tokenization and Formatting:** Fine-tuning a causal language model like Qwen on this data requires converting the text into token sequences. Use Qwen’s tokenizer to encode the text. First, load the Qwen tokenizer:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
```

Qwen’s tokenizer has a large vocabulary (\~152k tokens), covering multiple languages and code. We should decide on a maximum sequence length for training (for example, 512 or 1024 tokens) to balance memory and the ability to capture long contexts. The Qwen model supports up to 16k or more tokens context, but a 3090 GPU may not handle very long sequences in training, so we might choose 1024 for fine-tuning.

There are two approaches to prepare batches:

* **On-the-fly tokenization**: Stream the data and tokenize each batch dynamically in the data collator. This avoids storing the tokenized dataset in memory, at the cost of repeated tokenization (which is fine if you have CPU headroom).
* **Pre-tokenization**: Tokenize and chunk the dataset into fixed-length sequences ahead of time (e.g., using `dataset.map`). This can be faster during training but may require significant memory/disk if done for billions of tokens, so it’s only feasible for a small subset.

For simplicity, we’ll outline on-the-fly tokenization using a custom collate function. This way you can stream the dataset and only load small chunks into RAM/GPU at a time. We will truncate or split texts longer than our `max_length` (1024) to fit the model’s input size. Shorter texts will be padded.

```python
from functools import partial
import torch

max_length = 1024

def tokenize_batch(batch):
    # batch is a list of examples, each a dict with 'text'
    texts = [ex["text"] for ex in batch]
    tokens = tokenizer(
        texts, 
        max_length=max_length, 
        truncation=True, 
        padding="max_length",  # pad to max_length for efficiency
        return_tensors="pt"
    )
    # For causal LM, set labels = input_ids (the model will shift internally)
    tokens["labels"] = tokens["input_ids"].clone()
    return tokens

# Wrap tokenize_batch in partial to use with DataLoader
collate_fn = partial(tokenize_batch)
```

Here we set `padding="max_length"` to get fixed-size sequences (1024 tokens) per example. This will pad shorter texts to length 1024. If you prefer dynamic padding to the longest in batch, you can use `padding=True` and pad later, but fixed padding simplifies collation. The labels are set equal to `input_ids` (for each token the model will be trained to predict the next token). The Qwen model’s forward function will handle the causal shift when `labels` are provided.

**DataLoader creation:** Once we have the streaming dataset and collate function, we can create a PyTorch DataLoader. If using streaming (which returns an `IterableDataset`), we should use `datasets.iterable_dataset.iterable_dataset` to shuffle or buffer if needed. For large-scale training, you might not shuffle the entire corpus (since it’s huge and randomly sampled already), but small scale fine-tuning could shuffle. We’ll skip shuffling for simplicity and just stream sequentially. For example:

```python
from torch.utils.data import DataLoader, IterableDataset

# If using streaming (IterableDataset), we can directly use it in DataLoader
train_dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
```

Here we chose a **batch\_size of 2** sequences per step as a conservative default. You can adjust based on memory usage; with a 24GB GPU, batch size 2 with 1024-length should be very safe. You might be able to increase to 4 or 8; monitor GPU memory with `nvidia-smi` and adjust accordingly. We will also use **gradient accumulation** to effectively increase the batch size without storing more in GPU at once (explained in Step 5).

*Note:* If you instead loaded a **small subset into memory** (not streaming), you could use `dataset.map` to tokenize all texts and then use `datasets.DataCollatorForLanguageModeling` (with `mlm=False`) to collate batches. However, for the sake of memory and given the dataset size, streaming + custom collate is more practical here.

## Step 3: Configuring Hugging Face Accelerate for FSDP (Single GPU)

Hugging Face **Accelerate** makes it easy to use distributed training techniques like Fully Sharded Data Parallel (FSDP). FSDP is a PyTorch feature that shards a model’s parameters, gradients, and optimizer states across multiple GPUs to reduce memory usage. In multi-GPU scenarios, each GPU holds only a fraction of the model and only gathers full weights when needed, enabling training of very large models (in fact, FSDP has been shown to scale to models with trillions of parameters). It can also optionally offload model shards to CPU RAM when not in use, further saving GPU memory at the cost of speed.

However, on a **single GPU**, there are no other GPUs to shard with. This means **FSDP won’t actually shard the model** – the single process ends up holding the full model (one shard = 100%). In other words, with one GPU, FSDP behaves essentially like normal training with some overhead. The main benefit we can still gain from FSDP in this setup is the **CPU offload** capability: FSDP can automatically move weights/gradients to CPU memory when they’re not actively being used, potentially enabling us to train models or batches that wouldn’t otherwise fit in 24GB VRAM. We will leverage this offload to avoid out-of-memory errors.

**Setting up Accelerate config:** Use the `accelerate` CLI to configure your environment for FSDP. Run `accelerate config` in your terminal and answer the prompts. For a single-machine single-GPU scenario with FSDP, you might answer as follows:

* **Compute environment**: `This machine` (since we are training on a local machine).
* **Distributed training**: Choose `multi-GPU` (even though we have one GPU, select this to unlock distributed training features in Accelerate).
* **Number of machines**: `1`.
* **Number of processes**: `1` (one process corresponding to our single GPU).
* **GPU/CPU**: `No` to CPU (we will use the GPU).
* **DeepSpeed**: `No` (we’re using FSDP instead).
* **FullyShardedDataParallel**: `Yes` (enable FSDP).
* **Sharding strategy**: `FULL_SHARD` (this is FSDP’s sharding strategy 1, meaning shard params, grads, optimizer states across ranks – with 1 rank it’s trivial, but required for offload to function properly).
* **CPU offload**: `Yes` (offload parameters/gradients to CPU RAM when possible).
* **Auto-wrap policy**: `TRANSFORMER_BASED_WRAP` (wrap each Transformer block as an FSDP unit).
* **Use `_no_split_modules`**: `Yes`. The Qwen model implements a `_no_split_modules` attribute (e.g., `["Qwen2DecoderLayer"]` for Qwen2) which tells FSDP how to wrap the model’s layers. By answering yes, Accelerate will wrap each `Qwen2DecoderLayer` (the transformer block) in its own FSDP shard and keep certain parts (like the embedding layer) outside shards to avoid splitting shared weights.
* **Backward prefetch**: `BACKWARD_PRE` (default backward prefetching policy).
* **Forward prefetch**: `False` (you can say No when asked; forward prefetch is an optimization that isn’t necessary for correctness).
* **State checkpointing**: Choose `SHARDED_STATE_DICT` for saving checkpoints (recommended for FSDP). This will save model weights in a sharded format by default. (We will show how to save and load in Step 7.)
* **`fsdp_use_orig_params`**: When asked about “use\_orig\_params” (a feature to allow some params frozen), you can answer `no` (unless you plan to freeze parts of the model or use LoRA, see Step 6 tips).
* **Mixed precision**: It will ask if you want FP16 or BF16. For a 3090, you can use **bf16** (bfloat16) if you have PyTorch 2.x – Ampere GPUs support bfloat16 training and it’s recommended for stability (less overflow issues than fp16). If bf16 isn’t working, use fp16. So answer: `bf16`.

After answering the questions, Accelerate will generate a config file (by default at `~/.cache/huggingface/accelerate/default_config.yaml`). **Review the config** to ensure it captured everything correctly. It should contain something like:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
num_processes: 1
num_machines: 1
mixed_precision: bf16
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_offload_params: true
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: Qwen2DecoderLayer   # (May be auto-set via _no_split_modules)
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: false
```

Key points in this config:

* **`distributed_type: FSDP`** activates Fully Sharded Data Parallel.
* **`num_processes: 1`** ensures only one GPU/process is used.
* Under `fsdp_config`:

  * `fsdp_offload_params: true` enables CPU offloading of params/gradients.
  * `fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP` and the class to wrap (`Qwen2DecoderLayer`) ensure each transformer block is treated as a unit for sharding. This prevents splitting shared layers like the embedding across shards.
  * `fsdp_sharding_strategy: FULL_SHARD` corresponds to PyTorch FSDP “shard everything” (like ZeRO Stage-3) – again, with one process this doesn’t actually shard across GPUs, but it’s needed for the offload logic to shard between GPU and CPU memory.
  * `mixed_precision: bf16` will use automatic mixed precision (AMP) in bfloat16 for faster training and lower memory. Bfloat16 uses 2 bytes per parameter like fp16, so memory usage is roughly halved compared to fp32, and training on Ampere GPUs benefits from BF16 speed.

Once this config is set, the Accelerate launcher will read it and set up FSDP automatically in our training script.

**How FSDP behaves (visual overview):** In multi-GPU training, FSDP shards the model so each GPU holds only a part of the model, greatly reducing per-GPU memory usage. The diagram below (from Meta AI) compares standard Data Parallel (which replicates the full model on each GPU) to Fully Sharded Data Parallel (which shards and only all-gathers weights when needed):

&#x20;*Standard data-parallel vs. fully sharded data-parallel training.* In FSDP, model layers are sharded across GPUs, and weights are gathered only when executing a layer, then shards are reduced/scattered during backward. On a single GPU, we don’t get this across-GPU sharding benefit (there’s only one shard), but we still can offload shards to CPU memory between uses.

As noted, **on one GPU FSDP yields no true sharding**. We use it here primarily for its CPU memory leverage. Keep in mind that this means training speed may be slower than pure GPU training (due to overhead of offloading and wrapping), but it enables us to train more aggressively without OOM. If you find that the model and batch size fit easily in 24GB without offload, you could disable offloading to train faster (set `fsdp_offload_params: false`). In our case with a large dataset and potentially large batch/sequence, we’ll keep it on to be safe.

## Step 4: Creating the Fine-Tuning Script (Transformers + Accelerate)

Now we’ll write the fine-tuning script that ties everything together: loading the model, setting up the optimizer, and training loop. We will use Hugging Face Transformers to get the model, and Accelerate to handle FSDP wrapping and distributed aspects. Let’s outline the script step by step (this could be a Python script `finetune_qwen_fsdp.py` or a Jupyter notebook):

**4.1 Load Model and Tokenizer:** We already loaded the tokenizer in Step 2. Now load the Qwen-0.5B model:

```python
from transformers import AutoModelForCausalLM

model_name = "Qwen/Qwen2-0.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16  # use BF16 precision for model weights
    # device_map="auto" is not needed here because Accelerate will handle device placement
)
```

This downloads the Qwen-0.5B model weights. At 0.5 billion parameters, the model is roughly 2 GB in fp32, or 1 GB in bf16. Loading in bf16 saves memory (the model will initially be on CPU in bf16 dtype). We avoid setting `.to(device)` here because we want Accelerate to put the model onto the GPU (or shard) when preparing.

**(Optional) Enable Gradient Checkpointing:** To further reduce memory, enable gradient checkpointing on the model. This is a technique that trades computation for memory by not storing intermediate activations and recomputing them during backpropagation. Hugging Face models usually have a `gradient_checkpointing_enable()` method:

```python
model.gradient_checkpointing_enable()
```

Gradient checkpointing can significantly cut down activation memory usage (often by \~50% or more), allowing larger batch size or sequence length at the cost of some extra compute during training (this is usually worth it for large models on limited GPU memory).

**(Optional) Apply LoRA adapters:** If you want to use **Low-Rank Adapters (LoRA)** for more memory-efficient fine-tuning, you can integrate the PEFT library here. LoRA will freeze the original model weights and insert small trainable weight matrices (rank decomposition of the original weight updates). This drastically reduces the number of trainable parameters (and gradient memory). For Qwen-0.5B, LoRA is not strictly necessary to fit in 24GB, but it can help if you want to push batch size or context length higher. It also allows faster training since less data needs to be updated.

For example, using PEFT:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,             # rank of LoRA decomposition (choose 8 or 16; 16 for more capacity)
    lora_alpha=32,    # scaling alpha
    lora_dropout=0.05, # dropout on LoRA updates
    target_modules=["Wq", "Wk", "Wv", "o_proj", "down_proj", "up_proj"],  # specify which submodules to apply LoRA to
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
```

In the above, `target_modules` should list the names of the Linear layers in Qwen that we want to apply LoRA to. Qwen’s architecture (being a GPT-like decoder) will have linear layers in the self-attention (queries, keys, values, output projection) and in the MLP (up and down projections). The names may differ; you might inspect `model.named_modules()` to find linear layer names. The example targets commonly named ones (`Wq`, `Wk`, `Wv` might be the query/key/value weight names, etc., adjust as needed). By default, `get_peft_model` will freeze the rest of the model’s weights (so only LoRA layers have `requires_grad=True`). This means our GPU memory for gradients and optimizer will be very small. If using LoRA, you should also set `fsdp_use_orig_params: true` in the accelerate config (to allow some params frozen without issue).

*Note:* If you use LoRA, you might *not* need FSDP offload at all, because the memory usage is so much lower. You could in fact run with just standard accelerate (or even Trainer) in that case. But we include it for completeness. It’s possible to combine LoRA and FSDP (Accelerate’s docs demonstrate fine-tuning LLaMA-70B with LoRA + FSDP on multiple GPUs). If combining, ensure you prepare the model **after** wrapping with LoRA (as shown above).

**4.2 Initialize Optimizer:** Choose an optimizer for fine-tuning. A common choice is AdamW. Given the dataset is essentially for language modeling, a learning rate on the order of 1e-4 to 2e-4 is often used for continuing pre-training, but since Qwen-0.5B is relatively small and we might not train for billions of tokens, we could use a slightly lower LR (e.g., 2e-5 to 1e-4) to avoid overshooting. We’ll also use weight decay:

```python
from torch.optim import AdamW

learning_rate = 2e-5
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
```

If using LoRA, you can actually set a higher learning rate for the LoRA parameters (since they are new and small). But for simplicity, we keep one optimizer for all trainable params.

**4.3 Prepare with Accelerate:** Now comes the critical part – let Accelerate handle moving the model and optimizer to the GPU, and wrap them for FSDP. We’ll use `Accelerate` to prepare the model, optimizer, and dataloader:

```python
from accelerate import Accelerator

# Initialize accelerator (will read the config for FSDP settings)
accelerator = Accelerator()
# Prepare everything
model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
```

When `accelerator.prepare` is called, under the hood it will:

* Move the model to GPU (or appropriate FSDP shards). Since we enabled FSDP in the config, it will wrap the model with `FSDP` (transformer blocks wrapped as specified).
* Shard or place optimizer parameters as needed.
* Handle any DDP or other distributed setup (not much here with one GPU).
* The `train_dataloader` will be wrapped if needed (in this case, on one GPU, it may just return as is).

At this point, the model is ready for training. If `fsdp_offload_params=True`, some of the model’s layers may initially reside in CPU memory and get moved to GPU on-the-fly during forward pass. This is handled by FSDP transparently.

**4.4 Training Loop:** We can now iterate over epochs and batches and train the model. Because our dataset is effectively endless (trillions of tokens), you might not really do “epochs” in the traditional sense (one epoch over 1.3T tokens is infeasible). Instead, you might decide on a certain number of steps or tokens to train on. For demonstration, let’s assume we iterate through, say, 1% of the sample dataset or run for a fixed number of steps. We’ll use a simple loop and manual loss logging:

```python
from tqdm import tqdm

num_epochs = 1   # you can treat one pass through the sample as one epoch
gradient_accumulation_steps = 8  # accumulate gradients for 8 batches before stepping
optimizer.zero_grad()

for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss  # Transformers models return loss when labels are provided
        # Accumulate loss (normalize if doing accumulation)
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        
        if (step + 1) % gradient_accumulation_steps == 0:
            # Gradient accumulated, now step optimizer
            optimizer.step()
            optimizer.zero_grad()
            # (Optional) Learning rate scheduler step can be done here if using one
        
        # (Optional) log loss
        if step % 20 == 0 and accelerator.is_local_main_process:
            print(f"Epoch {epoch}, step {step}, loss: {loss.item() * gradient_accumulation_steps:.4f}")
```

A few things to note in this loop:

* We use `accelerator.backward(loss)` instead of `loss.backward()`. This ensures proper handling with FSDP (and mixed precision scaling if it were fp16).
* We divide the loss by `gradient_accumulation_steps` to average gradients over the accumulation window. We accumulate 8 mini-batches before each optimizer step, which effectively makes the training batch 8× larger without increasing memory per step. This is useful to stabilize training since batch size 2 is very small – with accumulate of 8, our effective batch is 16 sequences.
* After calling `optimizer.step()`, Accelerate’s FSDP will sync parameters as needed. We zero grads and continue.
* We only print loss every 20 steps and only on the main process (in this case, just one process, but we check `is_local_main_process` for generality).

You may also integrate a learning rate scheduler (for example, warmup and cosine decay, etc.) via `transformers.get_scheduler` and step it each iteration or epoch. Given this is a demo, we omitted for brevity.

During training, with `fsdp_offload_params=True`, you might observe CPU RAM being used significantly (the entire model weights and optimizer states may reside in CPU memory when not in use). Ensure your system has enough RAM (for a 0.5B model, a few GB of RAM is sufficient; for larger models this becomes more important). The 3090’s GPU utilization might not be 100% due to offloading overhead, but that’s expected.

## Step 5: Running the Fine-Tuning Process

To launch the training, we will use the `accelerate launch` command. Assuming our script is named `finetune_qwen_fsdp.py`, run:

```bash
accelerate launch finetune_qwen_fsdp.py
```

Accelerate will read the config we set up in Step 3 and initialize the distributed environment (even though it’s just one GPU, it will initialize torch.distributed and FSDP). Then it runs the script. You should see the model begin training, and loss values printing (if you included logging).

Monitor GPU memory with `nvidia-smi` to see that usage. With our settings (bf16, gradient checkpointing, offload, batch 2, seq 1024), GPU memory should stay well within 24 GB. If you try larger batches or sequence lengths, the offloading and checkpointing should help prevent OOM. If an OOM still occurs, see Step 6 for further tips.

Depending on how long you train (how many steps), the model’s weights will update based on the FineWeb-Edu data. Because FineWeb-Edu is so large, a single epoch (pass) is not really meaningful – you might train for a certain number of tokens. For instance, you could aim to train on, say, 50 billion tokens (\~5% of the dataset) which for a 0.5B model might be a reasonable continued pre-training budget. That would be on the order of 50e9 / (batch\_size \* seq\_len) steps. With effective batch 16 and seq 1024, that is \~3 million steps – which is likely too much to do on a single 3090 in a reasonable time. You will probably train on a tiny fraction of the data unless you have a lot of time, and treat it more like a proof-of-concept fine-tuning.

If you just want to validate the process, you can stop training after a few hundred or thousand steps and still save the model to verify everything works.

## Step 6: Avoiding Out-of-Memory Errors – Tips and Tricks

Even though Qwen-0.5B is relatively small by modern standards, training on long sequences or large batches can still exceed 24 GB, especially when storing gradients. We have already used several strategies to mitigate OOM risk, but let’s summarize and add any additional tips:

* **FSDP with CPU Offload**: By setting `fsdp_offload_params=True`, we allowed FSDP to offload parameters and gradients to CPU memory when not in use. This drastically lowers GPU memory usage, effectively using system RAM as extension. The trade-off is speed (data transfer between CPU and GPU). Use this only if needed – if you find GPU memory is comfortable, you can disable offload to train faster. (On a single GPU, if the model fits, there’s “no reason to use FSDP” without offload overhead.)
* **Mixed Precision**: We used `bf16` (or you can use `fp16`) for model weights and computation. This cuts memory by half for model weights and speeds up training on Tensor Cores. **Mixed precision is crucial** to fit larger models on 24GB. Bfloat16 is preferred for stability (no loss of dynamic range), as noted by many experts (training in int8 or fp8 is generally not stable yet).
* **Gradient Checkpointing**: We enabled `model.gradient_checkpointing_enable()`. This saved a lot of activation memory during forward by not storing everything for backward. The downside is slightly slower training (the model has to recompute the dropped activations in the backward pass). This is usually worth the memory savings for large models. It’s a simple toggle and we see it being used even in 70B model fine-tuning setups (the config shows `gradient_checkpointing True`).
* **Gradient Accumulation**: Instead of trying to fit a huge batch in memory, we accumulate smaller batches. In our example, we accumulate 8 batches of size 2 to mimic a batch of 16 without using memory for 16 at once. This allows better GPU utilization and a more stable gradient estimation without OOM. Accelerate provides a context manager `accelerator.accumulate(model)` which can automate gradient accumulation, or you can do it manually as we did. Just be mindful that with FSDP, if you manually accumulate, you might want to use `accelerator.no_sync()` when not at the accumulation step, but `Accelerate.accumulate` handles that internally.
* **Low-Rank Adapters (LoRA)**: If supported for your use-case, LoRA is a game-changer for memory. By freezing the majority of model weights and training only a few small added matrices, you reduce gradient and optimizer memory dramatically. For Qwen-0.5B, training full model is feasible, but for much larger models, LoRA is almost required on a single GPU. We demonstrated how to wrap the model with LoRA using PEFT. With LoRA, you could even consider turning off FSDP offload because the memory usage might drop enough. (If you do still use FSDP with LoRA, set `fsdp_use_orig_params=True` to allow having non-trainable original params.)
* **4-bit Quantization (QLoRA)**: Another advanced trick is 4-bit quantization of model weights during fine-tuning (used in **QLoRA**). This compresses the model to 4-bit precision for storage (while using 16-bit for compute) and is how people fine-tune 65B models on a single GPU. QLoRA requires the `bitsandbytes` library. In our case, Qwen-0.5B doesn’t need this, but if you were curious, Qwen’s documentation even shows an example of loading Qwen-7B in 4-bit precision with NF4 quantization. Combining 4-bit quantization + LoRA (which is exactly QLoRA) would make memory a non-issue here. We didn’t explicitly do this above, but it’s an option if you really want to minimize memory usage.
* **Flash Attention**: Use optimized kernels for attention if available. The Qwen model supports FlashAttention v2 – as shown in the model code snippet, they pass `attn_implementation="flash_attention_2"` when loading. If you `pip install flash-attn` (the package from Tri Dao et al.), the model can use it to compute attention faster and with lower memory overhead for long sequences. This is recommended if you go to very long sequence lengths (e.g. 2K tokens or more).
* **Monitoring and adjusting**: Always keep an eye on memory. If you see GPU memory is only half-used, you might increase batch size a bit or sequence length. If you see it nearly full or swapping a lot, consider reducing batch or seq length. Also watch CPU RAM when using offload – if you run out of RAM, the OS will start swapping to disk which will dramatically slow down training or even crash.
* **DeepSpeed Zero as alternative**: For single GPU, DeepSpeed Zero-Infinity (stage 3 with CPU offload) is conceptually similar to FSDP with offload. Accelerate can also configure DeepSpeed, but since we already use FSDP effectively, there’s no need to switch. Both achieve moving data to CPU to train models larger than VRAM.

In summary, by using the above techniques, a **24GB GPU is sufficient** to fine-tune Qwen-0.5B on reasonably long sequences. In fact, without offload, Qwen-0.5B in 16-bit uses \~2–4GB during training, leaving plenty of space for activations. The main memory consumers might be the dataset (if not streaming) and any very large batch or sequence. With offloading and gradient checkpointing, even much larger models (like 3B or 7B) could be trained on a 3090, though slowly. Keep in mind FSDP offload will slow training due to CPU-GPU transfer – use it only as needed. If you find the training unbearably slow with offload, and memory usage is comfortable, try disabling offload to see if it speeds up.

## Step 7: Saving and Loading Checkpoints

After training for your desired duration, you’ll want to save the fine-tuned model weights. When using FSDP, saving can be a bit tricky because the model is sharded across processes (if multi-GPU) or offloaded. Accelerate provides utilities to help with checkpointing:

**Saving the model**: The simplest way is to use the `Accelerator` to gather the full state dict and then use the regular Hugging Face `save_pretrained`. For example at the end of training:

```python
# Make sure all processes have finished (relevant in multi-GPU cases)
accelerator.wait_for_everyone()
# Get the unwrapped original model (inside FSDP)
unwrapped_model = accelerator.unwrap_model(model)
# Retrieve state dict (on CPU) for saving
state_dict = accelerator.get_state_dict(model)
# Save to disk (only on process 0)
if accelerator.is_main_process:
    unwrapped_model.save_pretrained("output/qwen-finetuned", state_dict=state_dict)
    tokenizer.save_pretrained("output/qwen-finetuned")
```

What this does:

* `unwrap_model` gives us the underlying `AutoModelForCausalLM` (without the FSDP wrapper).
* `accelerator.get_state_dict(model)` will automatically gather the weights from all FSDP shards (if there were multiple) and move them to CPU in a consolidated state dict. It uses `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)` internally to get a complete model on CPU memory (only on rank 0).
* Then we call `save_pretrained` on the unwrapped model, supplying that state\_dict. The `accelerator.is_main_process` check ensures only one process writes the files (to avoid conflicts).
* We also save the tokenizer to the same directory for completeness (so we have `tokenizer.json`, vocab files, etc., needed to use the model later).

This will create the `output/qwen-finetuned/` directory with `pytorch_model.bin` (or `pytorch_model.bin.index.json` plus shard files if still sharded – but `get_state_dict` with full config gives a full state by default, so likely one file), and the model config. Since Qwen was loaded from the hub, it will also copy the config JSON. Now you have a local fine-tuned model!

**Alternative saving**: You could also let Accelerate save shards directly by calling `accelerator.save_state("my_checkpoint_dir")`. This would save each FSDP shard separately (in our single-GPU case, it would just save one shard which is essentially the whole model) along with optimizer states. This is useful for resuming training exactly where you left off (including optimizer momentum, etc.). If you go this route, you’d later load via `accelerator.load_state("my_checkpoint_dir")` to resume. However, the saved format is a bit less convenient for inference because it’s in shards. You can merge shards later with `accelerate/utils.merge_fsdp_weights` utility if needed. For simplicity, we chose `save_pretrained` which produces the standard HuggingFace model files ready for inference.

**Loading the model for inference/use**: Once saved, you can load the model like any other:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("output/qwen-finetuned", torch_dtype=torch.float16).cuda()
tokenizer = AutoTokenizer.from_pretrained("output/qwen-finetuned")
```

Here we load in half precision (fp16) and put on GPU for generation. Since 0.5B is small, this is easy on a 3090. You can then generate text from it to see how it learned from the FineWeb-Edu data. (If you used LoRA and want to merge the LoRA weights, you’d have to either use `peft.merge_and_unload` or keep the LoRA wrappers for inference. But if you saved with LoRA already merged via `get_state_dict`, then it’s a full model.)

Make sure to test the model output on some prompts to verify it behaves as expected (perhaps it should have a style or knowledge reflective of educational content if FineWeb-Edu data influenced it).

## Step 8: Caveats and Limitations of Using FSDP on a Single GPU

Before concluding, let’s emphasize a few **caveats**:

* **No Speedup from FSDP with one GPU**: As highlighted earlier, FSDP won’t shard the model across devices when only one GPU is present. This means you shouldn’t expect a reduction in GPU memory from sharding (only from offloading) and you won’t get a training speed increase (if anything, training is a bit slower due to overhead). The reason we used FSDP here is mainly to demonstrate how to configure it and to utilize its CPU offload for memory extension. If your model fits in GPU memory comfortably, you could fine-tune without FSDP for better performance.
* **CPU Offload overhead**: Offloading introduces significant communication overhead. Every forward pass, FSDP will move layer weights from CPU to GPU, and after backward it will move gradients back to CPU. This can slow down training iterations. The trade-off is worthwhile only if without offload you’d get OOM or be forced to a much smaller batch/seq. In our case, Qwen-0.5B probably could train without offload on 24GB. For bigger models, offload makes the impossible possible (e.g., training a 6B model on a 3090), but expect iterations to be much slower (possibly seconds per step, depending on model size).
* **Gradient Accumulation with Offload**: If you manually accumulate gradients across steps, note that PyTorch FSDP has some limitations with offload – it expects you to synchronize shards properly. Using `Accelerator.accumulate` or the pattern we did (dividing loss and accumulating) is fine. Just avoid calling `.backward()` multiple times without optimizer step in between unless you manage `no_sync()` context. Our code dividing the loss by `gradient_accumulation_steps` and accumulating is effectively doing the right thing.
* **Single GPU vs Multi GPU**: The guide focused on single GPU. If in the future you use multiple GPUs, the same accelerate FSDP config can be reused with `num_processes` set to the number of GPUs. Then each GPU will truly hold a shard of Qwen (e.g., with 2 GPUs, each gets \~0.25B params) and training can scale. Just remember to increase `per_device_train_batch_size` accordingly and remove or adjust gradient accumulation for the new global batch. Multi-GPU would significantly speed up training on a large dataset like FineWeb-Edu (almost linearly, ideally).
* **Dataset size**: FineWeb-Edu is extremely large. On a single 3090, you will not be able to iterate through the entire dataset in a reasonable time. Treat this fine-tuning as *continual pre-training* on a subset of FineWeb-Edu. You might take, for example, the 10B-token sample and train 1-2 epochs on it (which itself could be many iterations). That alone is a substantial amount of training. Don’t feel compelled to use all 1.3T tokens – that would require far more compute (on the order of what it took to originally pretrain big models). Focus on a manageable subset or number of steps and ensure the training is learning (monitor the loss curve).
* **Evaluation**: Since FineWeb-Edu is unlabeled, you may want to evaluate the model on some benchmarks or sample prompts to see the effect of fine-tuning. For instance, test if the model improved in knowledge or writing quality on educational topics. Because Qwen-0.5B is a base model, its outputs might be raw; you might consider using Qwen-0.5B-Chat and fine-tuning that on a curated instruct dataset if your end goal is a chatbot. FineWeb-Edu fine-tuning is more like additional pretraining.

By following the steps above, you should be able to fine-tune the Qwen 0.5B model on the FineWeb-Edu dataset using Hugging Face Accelerate with FSDP on your single-GPU machine. The key configurations (FSDP offload, gradient checkpointing, etc.) help fit the model and training process within 24GB of VRAM. While FSDP on one GPU doesn’t give the full benefit of sharding, it provides a pathway to train larger models than the GPU would normally allow by utilizing CPU memory.

Finally, remember to **cite or credit** the FineWeb-Edu dataset and Qwen model appropriately in any research or project report. Happy fine-tuning!

**Sources:**

* Hugging Face FineWeb-Edu Dataset Card (dataset description and subset info)
* Qwen-0.5B Model Card (Alibaba Qwen2 models)
* Hugging Face Accelerate Documentation on FSDP
* PyTorch Forums – FSDP single GPU explanation
* Hugging Face PEFT/Accelerate integration example (FSDP + LoRA)
* Meta AI Blog on FSDP (for conceptual overview)
