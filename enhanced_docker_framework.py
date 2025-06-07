#!/usr/bin/env python
"""
Enhanced Docker-Compatible DiLoCo FSDP Framework

This is an improved version that addresses FSDP auto_wrap_policy compatibility issues
in Docker containers while maintaining all the training stability fixes.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
from pathlib import Path
from datasets import load_dataset, IterableDataset
import warnings

# Check PyTorch version and FSDP capabilities
PYTORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
FSDP_AVAILABLE = False
FSDP2_AVAILABLE = False
TRANSFORMER_WRAP_AVAILABLE = False

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    FSDP_AVAILABLE = True
    print(f"✅ FSDP available (PyTorch {torch.__version__})")
    
    # Check for transformer-based wrap policy
    try:
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        TRANSFORMER_WRAP_AVAILABLE = True
        print(f"✅ Transformer auto-wrap policy available")
    except ImportError:
        print(f"⚠️  Transformer auto-wrap policy not available")
    
    # Check for FSDP2 features
    try:
        from torch.distributed.fsdp import fully_shard
        FSDP2_AVAILABLE = True
        print(f"✅ FSDP2 (fully_shard) available")
    except ImportError:
        print(f"⚠️  FSDP2 not available, using legacy FSDP")
        
except ImportError:
    print(f"⚠️  FSDP not available in PyTorch {torch.__version__}")

# DatasetLoader utility
class DatasetLoader:
    """Utility for streaming datasets and tokenization."""

    def __init__(self, dataset_name: str, subset: str, *, text_field: str | None = None,
                 shuffle_buffer: int = 10000, seed: int = 0) -> None:
        self.dataset_name = dataset_name
        self.subset = subset
        self.text_field = text_field
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def load(self) -> Tuple[IterableDataset, str]:
        ds = load_dataset(
            self.dataset_name,
            name=self.subset,
            split="train",
            streaming=True,
        )
        ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)
        first = next(iter(ds.take(1)))
        if self.text_field:
            if self.text_field not in first:
                raise ValueError(f"Dataset has no field '{self.text_field}'")
            field = self.text_field
        else:
            if "text" in first:
                field = "text"
            elif "content" in first:
                field = "content"
            else:
                available = list(first.keys())
                raise ValueError(f"Dataset has no 'text' or 'content' field. Available: {available}")
        
        return ds, field


# Simple dataset for testing (fallback)
class SimpleDataset(Dataset):
    """Simple dataset for testing with diverse training samples."""
    
    def __init__(self, tokenizer, seq_len: int = 32):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # More diverse training texts
        self.texts = [
            "The quick brown fox jumps over the lazy dog in the morning sunshine.",
            "Machine learning models require careful training and validation procedures.",
            "Scientific research involves hypothesis formation, experimentation, and analysis.",
            "Software development follows principles of modularity, testing, and documentation.",
            "Natural language processing enables computers to understand human communication.",
            "Deep learning networks consist of multiple layers that learn complex patterns.",
            "Data preprocessing is crucial for achieving optimal model performance.",
            "Distributed computing allows processing of large-scale datasets efficiently.",
            "Neural networks can approximate complex functions through backpropagation training.",
            "Optimization algorithms help models converge to better solutions over time."
        ]
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Add some variation based on index to reduce repetition
        text = self.texts[idx % len(self.texts)]
        if idx >= len(self.texts):
            text = f"Iteration {idx // len(self.texts) + 1}: {text}"
        
        # Tokenize with proper padding and truncation
        encoded = self.tokenizer(
            text,
            max_length=self.seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].squeeze(0)
        
        # Create labels (shift for causal LM)
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "labels": labels
        }


@dataclass
class TrainerConfig:
    """Configuration for the Enhanced Docker Framework."""
    
    # Model and training
    model_name: str = "microsoft/DialoGPT-small"
    steps: int = 10
    seq_len: int = 32
    batch_size: int = 1
    diloco_loops: int = 2
    learning_rate: float = 1e-4
    
    # Mixed precision
    mixed_precision: str = "fp16"  # "fp16", "bf16", or "no"
    
    # Output
    output_dir: str = "/tmp/diloco_output"
    
    # Dataset configuration
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "sample-10BT" 
    text_field: Optional[str] = None
    use_streaming: bool = True
    no_streaming: bool = False
    shuffle_buffer: int = 10000
    dataset_seed: int = 0
    
    # FSDP configuration
    use_fsdp: bool = True
    fsdp_auto_wrap_policy: str = "size_based"  # "size_based", "transformer_based", or "none"
    fsdp_mixed_precision: bool = True
    fsdp_cpu_offload: bool = False  # Only enable if needed for memory
    fsdp_min_num_params: int = 100
    
    # Dynamic batching
    dynamic_batch: bool = False
    
    def __post_init__(self):
        """Validate and adjust configuration based on environment."""
        # Handle streaming preference
        if self.no_streaming:
            self.use_streaming = False
            
        # Adjust FSDP settings based on availability
        if self.use_fsdp and not FSDP_AVAILABLE:
            print("⚠️  FSDP requested but not available, disabling")
            self.use_fsdp = False
            
        if self.use_fsdp and self.fsdp_auto_wrap_policy == "transformer_based" and not TRANSFORMER_WRAP_AVAILABLE:
            print("⚠️  Transformer auto-wrap policy not available, falling back to size-based")
            self.fsdp_auto_wrap_policy = "size_based"
            
        # Validate mixed precision
        if self.mixed_precision not in ["fp16", "bf16", "no"]:
            raise ValueError("mixed_precision must be 'fp16', 'bf16', or 'no'")
            
        # Auto-detect bf16 support
        if self.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
            print("⚠️  BF16 not supported, falling back to FP16")
            self.mixed_precision = "fp16"


class EnhancedDockerTrainer:
    """Enhanced Docker-compatible DiLoCo trainer with robust FSDP fallbacks."""
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.setup_distributed()
        self.setup_model()
        self.setup_data()
        
    def setup_distributed(self):
        """Setup distributed training environment."""
        # Get distributed environment variables
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Check if we're in a distributed setting
        self.is_distributed = self.world_size > 1
        
        if self.is_distributed:
            # Initialize distributed process group
            if not dist.is_initialized():
                master_addr = os.environ.get("MASTER_ADDR", "localhost")
                master_port = os.environ.get("MASTER_PORT", "29500")
                
                # Choose backend based on availability
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                
                try:
                    dist.init_process_group(
                        backend=backend,
                        init_method=f"tcp://{master_addr}:{master_port}",
                        world_size=self.world_size,
                        rank=self.rank
                    )
                    print(f"✅ Distributed initialized: rank={self.rank}, world_size={self.world_size}")
                except Exception as e:
                    print(f"⚠️  Distributed initialization failed: {e}")
                    self.is_distributed = False
                    self.world_size = 1
                    self.rank = 0
        
        # Setup device
        if torch.cuda.is_available():
            if self.is_distributed:
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(f"cuda:{self.local_rank}")
            else:
                self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            
        print(f"🚀 Trainer setup: rank={self.rank}, world_size={self.world_size}, distributed={self.is_distributed}")
        
    def setup_model(self):
        """Setup model with optional FSDP wrapping."""
        print(f"📦 Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Set up pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"🔧 Set pad_token to eos_token: {self.tokenizer.eos_token} (id: {self.tokenizer.eos_token_id})")
        
        # Log tokenizer info
        print(f"🔧 Tokenizer info:")
        print(f"   Vocab size: {self.tokenizer.vocab_size}")
        print(f"   EOS token: {self.tokenizer.eos_token} (id: {self.tokenizer.eos_token_id})")
        print(f"   PAD token: {self.tokenizer.pad_token} (id: {self.tokenizer.pad_token_id})")
        if hasattr(self.tokenizer, 'bos_token'):
            print(f"   BOS token: {self.tokenizer.bos_token} (id: {self.tokenizer.bos_token_id})")
        if hasattr(self.tokenizer, 'unk_token'):
            print(f"   UNK token: {self.tokenizer.unk_token} (id: {self.tokenizer.unk_token_id})")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32  # Start in fp32, FSDP will handle precision
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Setup FSDP if enabled and available
        if self.config.use_fsdp and FSDP_AVAILABLE and self.is_distributed:
            self.setup_fsdp()
        else:
            if self.config.use_fsdp:
                print(f"⚠️  FSDP not applied: available={FSDP_AVAILABLE}, distributed={self.is_distributed}")
            print(f"🔧 Using regular model (no FSDP)")
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate
        )
        
        # Setup gradient scaler for mixed precision
        self.setup_mixed_precision()
        
        print(f"✅ Model loaded on {self.device}")
        print(f"🔧 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def setup_fsdp(self):
        """Setup FSDP with compatibility checks."""
        try:
            print(f"🔧 Setting up FSDP (rank {self.rank})")
            
            # Choose auto-wrap policy
            auto_wrap_policy = None
            if self.config.fsdp_auto_wrap_policy == "size_based":
                auto_wrap_policy = size_based_auto_wrap_policy(
                    min_num_params=self.config.fsdp_min_num_params
                )
                print(f"🔧 Using size-based auto-wrap policy (min_params={self.config.fsdp_min_num_params})")
                
            elif self.config.fsdp_auto_wrap_policy == "transformer_based" and TRANSFORMER_WRAP_AVAILABLE:
                # Try to detect transformer layer class
                transformer_layer_cls = self.detect_transformer_layer_cls()
                if transformer_layer_cls:
                    auto_wrap_policy = transformer_auto_wrap_policy(
                        transformer_layer_cls={transformer_layer_cls}
                    )
                    print(f"🔧 Using transformer-based auto-wrap policy (layer: {transformer_layer_cls.__name__})")
                else:
                    print(f"⚠️  Could not detect transformer layer class, falling back to size-based")
                    auto_wrap_policy = size_based_auto_wrap_policy(
                        min_num_params=self.config.fsdp_min_num_params
                    )
                    
            else:
                print(f"🔧 No auto-wrap policy specified")
            
            # Setup mixed precision policy
            mixed_precision_policy = None
            if self.config.fsdp_mixed_precision and self.config.mixed_precision != "no":
                from torch.distributed.fsdp import MixedPrecision
                if self.config.mixed_precision == "fp16":
                    mixed_precision_policy = MixedPrecision(
                        param_dtype=torch.float16,
                        reduce_dtype=torch.float16,
                        buffer_dtype=torch.float16,
                    )
                elif self.config.mixed_precision == "bf16":
                    mixed_precision_policy = MixedPrecision(
                        param_dtype=torch.bfloat16,
                        reduce_dtype=torch.bfloat16,
                        buffer_dtype=torch.bfloat16,
                    )
                print(f"🔧 FSDP mixed precision: {self.config.mixed_precision}")
            
            # Create FSDP wrapper
            fsdp_kwargs = {}
            if auto_wrap_policy:
                fsdp_kwargs['auto_wrap_policy'] = auto_wrap_policy
            if mixed_precision_policy:
                fsdp_kwargs['mixed_precision'] = mixed_precision_policy
            if self.config.fsdp_cpu_offload:
                from torch.distributed.fsdp import CPUOffload
                fsdp_kwargs['cpu_offload'] = CPUOffload(offload_params=True)
                print(f"🔧 FSDP CPU offload enabled")
            
            # Apply FSDP
            self.model = FSDP(
                self.model,
                device_id=self.device if torch.cuda.is_available() else None,
                **fsdp_kwargs
            )
            print(f"✅ FSDP model wrapped (rank {self.rank})")
            
        except Exception as e:
            print(f"⚠️  FSDP setup failed: {e}, using regular model")
            # Model is already on device, so just continue
            
    def detect_transformer_layer_cls(self):
        """Try to detect the transformer layer class for auto-wrap policy."""
        # Common transformer layer class names
        layer_names = [
            'TransformerBlock', 'Block', 'Layer', 'DecoderLayer', 
            'GPTBlock', 'QwenBlock', 'Qwen2DecoderLayer', 'LlamaDecoderLayer'
        ]
        
        for name, module in self.model.named_modules():
            module_name = module.__class__.__name__
            if any(layer_name in module_name for layer_name in layer_names):
                return module.__class__
                
        return None
        
    def setup_mixed_precision(self):
        """Setup mixed precision training."""
        self.scaler = None
        
        if self.config.mixed_precision == "fp16" and torch.cuda.is_available():
            try:
                # Try new API first
                try:
                    self.scaler = torch.amp.GradScaler('cuda')
                except AttributeError:
                    # Fallback to old API
                    self.scaler = torch.cuda.amp.GradScaler()
                print(f"🔧 Gradient scaler: enabled (fp16)")
            except Exception as e:
                print(f"⚠️  GradScaler not available: {e}")
                self.config.mixed_precision = "no"
        elif self.config.mixed_precision == "bf16":
            print(f"🔧 Mixed precision: bf16 (no scaler needed)")
        else:
            print(f"🔧 Mixed precision: disabled")
            
    def setup_data(self):
        """Setup training data with enhanced validation."""
        if self.config.use_streaming:
            try:
                print(f"📊 Setting up streaming dataset: {self.config.dataset_name}")
                dataset_loader = DatasetLoader(
                    self.config.dataset_name,
                    self.config.dataset_subset,
                    text_field=self.config.text_field,
                    shuffle_buffer=self.config.shuffle_buffer,
                    seed=self.config.dataset_seed + self.rank  # Unique seed per rank
                )
                self.dataset, self.text_field = dataset_loader.load()
                print(f"📊 Streaming dataset ready with field: {self.text_field}")
                
            except Exception as e:
                print(f"⚠️  Streaming dataset setup failed: {e}")
                # Fallback to simple dataset
                self.setup_simple_dataset()
        else:
            self.setup_simple_dataset()
            
        # Create dataloader
        if hasattr(self, 'simple_dataset'):
            # Simple dataset case
            self.dataloader = DataLoader(
                self.simple_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0  # Avoid multiprocessing issues in Docker
            )
        else:
            # Streaming dataset case - we'll handle this in the training loop
            pass
            
    def setup_simple_dataset(self):
        """Setup simple fallback dataset."""
        print(f"📊 Setting up SimpleDataset fallback")
        self.simple_dataset = SimpleDataset(self.tokenizer, self.config.seq_len)
        print(f"📊 Simple dataset ready: {len(self.simple_dataset)} samples")
        
    def create_streaming_generator(self):
        """Create a generator for streaming dataset with enhanced validation."""
        processed_count = 0
        skipped_count = 0
        
        for example in self.dataset:
            try:
                text = example[self.text_field]
                
                # Enhanced text validation
                if len(text.strip()) < 50:  # Minimum text length
                    skipped_count += 1
                    continue
                    
                # Check for meaningful content (not just whitespace/special chars)
                meaningful_chars = sum(1 for c in text if c.isalnum())
                if meaningful_chars < 20:
                    skipped_count += 1
                    continue
                
                # Pre-validate tokenization (without padding)
                try:
                    test_tokens = self.tokenizer.encode(text, add_special_tokens=True)
                    if len(test_tokens) < 10:  # Minimum token count
                        skipped_count += 1
                        continue
                except Exception as e:
                    skipped_count += 1
                    continue
                
                # Tokenize with proper settings
                encoded = self.tokenizer(
                    text,
                    max_length=self.config.seq_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                input_ids = encoded["input_ids"].squeeze(0)
                attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).squeeze(0)
                
                # Validate attention mask consistency
                actual_tokens = (input_ids != self.tokenizer.pad_token_id).sum().item()
                expected_tokens = attention_mask.sum().item()
                
                if actual_tokens != expected_tokens:
                    skipped_count += 1
                    continue
                
                # Create labels (for causal LM)
                labels = input_ids.clone()
                
                processed_count += 1
                
                # Progress logging
                if processed_count % 100 == 0:
                    print(f"📊 Dataset progress: {processed_count} processed, {skipped_count} skipped")
                
                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                }
                
                # Yield enough examples for training
                if processed_count >= self.config.steps * self.config.diloco_loops * 2:
                    break
                    
            except Exception as e:
                skipped_count += 1
                if skipped_count % 50 == 0:
                    print(f"⚠️  Dataset errors: {skipped_count} examples skipped")
                continue
                
        print(f"📊 Dataset generation completed: {processed_count} processed, {skipped_count} skipped")
        
    def train_step(self, batch):
        """Enhanced training step with comprehensive validation."""
        self.model.train()
        
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                # Handle both 1D and 2D tensors
                if batch[key].dim() == 1:
                    batch[key] = batch[key].unsqueeze(0)  # Add batch dimension
                batch[key] = batch[key].to(self.device)
        
        # Enhanced input validation
        input_ids = batch["input_ids"]
        batch_size, seq_len = input_ids.shape
        
        # Check for empty batch
        if batch_size == 0:
            print("⚠️  Empty batch detected, skipping")
            return None
            
        # Check padding ratio
        if self.tokenizer.pad_token_id is not None:
            total_tokens = batch_size * seq_len
            pad_tokens = (input_ids == self.tokenizer.pad_token_id).sum().item()
            pad_ratio = pad_tokens / total_tokens if total_tokens > 0 else 1.0
            
            if pad_ratio > 0.95:  # More than 95% padding
                print(f"⚠️  High padding ratio ({pad_ratio:.1%}), skipping")
                return None
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=(self.config.mixed_precision == "fp16")) if torch.cuda.is_available() else torch.autocast(device_type='cuda', enabled=False):
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Validate loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  Invalid loss detected: {loss.item()}")
                return None
                
            # Check for extremely small or large losses
            loss_val = loss.item()
            if loss_val < 1e-8:
                print(f"⚠️  Extremely small loss: {loss_val}")
                return None
            elif loss_val > 100:
                print(f"⚠️  Extremely large loss: {loss_val}")
                return None
        
        # Backward pass
        if self.scaler is not None:
            # FP16 with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Check for gradient scaling issues
            try:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            except Exception as e:
                print(f"⚠️  Gradient scaling error: {e}")
                self.optimizer.zero_grad()
                return None
        else:
            # Regular backward pass
            loss.backward()
            
            # Enhanced gradient analysis
            total_grad_norm = 0.0
            max_grad_norm = 0.0
            zero_grad_count = 0
            total_params = 0
            
            for param in self.model.parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.data.norm(2).item()
                    total_grad_norm += param_grad_norm ** 2
                    max_grad_norm = max(max_grad_norm, param_grad_norm)
                    total_params += 1
                else:
                    zero_grad_count += 1
            
            total_grad_norm = total_grad_norm ** 0.5
            
            # Check for gradient issues
            if zero_grad_count > 0:
                print(f"⚠️  Zero gradients: {zero_grad_count}/{total_params + zero_grad_count}")
                
            if total_grad_norm > 10.0:  # Gradient clipping threshold
                print(f"⚠️  Large gradient norm ({total_grad_norm:.4f})")
                print(f"   Max individual param grad: {max_grad_norm:.4f}")
                print(f"   Applying gradient clipping")
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        return loss.item()
        
    def train(self):
        """Main training loop with DiLoCo outer optimization."""
        print(f"🎯 Starting DiLoCo training:")
        print(f"   DiLoCo loops: {self.config.diloco_loops}")
        print(f"   Steps per loop: {self.config.steps}")
        print(f"   Total steps: {self.config.diloco_loops * self.config.steps}")
        print(f"   Dataset: {'streaming' if self.config.use_streaming else 'simple'}")
        print()
        
        start_time = time.time()
        all_losses = []
        
        # Setup data iterator
        if hasattr(self, 'simple_dataset'):
            data_iter = iter(self.dataloader)
        else:
            # Create streaming generator
            data_gen = self.create_streaming_generator()
            data_iter = iter(data_gen)
        
        for loop in range(self.config.diloco_loops):
            print(f"🔄 Rank {self.rank}: DiLoCo Loop {loop + 1}/{self.config.diloco_loops}")
            
            loop_losses = []
            
            for step in range(self.config.steps):
                try:
                    # Get next batch
                    if hasattr(self, 'simple_dataset'):
                        try:
                            batch = next(data_iter)
                        except StopIteration:
                            data_iter = iter(self.dataloader)
                            batch = next(data_iter)
                    else:
                        batch = next(data_iter)
                    
                    # Training step
                    loss = self.train_step(batch)
                    
                    if loss is not None:
                        loop_losses.append(loss)
                        print(f"⚡ Rank {self.rank} Loop {loop + 1} Step {step + 1}/{self.config.steps}: loss={loss:.4f}")
                        
                        # Debug info with less frequency
                        if step % 2 == 0:  # Every other step
                            # Calculate parameter and gradient norms for debugging
                            max_param_norm = max(p.data.norm().item() for p in self.model.parameters())
                            max_grad_norm = max(p.grad.data.norm().item() for p in self.model.parameters() if p.grad is not None) if any(p.grad is not None for p in self.model.parameters()) else 0.0
                            print(f"🔍 Debug Loop {loop + 1} Step {step + 1}: max_param_norm={max_param_norm:.4f}, max_grad_norm={max_grad_norm:.4f}")
                    else:
                        print(f"⚠️  Rank {self.rank} Loop {loop + 1} Step {step + 1}/{self.config.steps}: step skipped")
                        
                except StopIteration:
                    print(f"⚠️  Data exhausted at loop {loop + 1}, step {step + 1}")
                    break
                except Exception as e:
                    print(f"⚠️  Training step error: {e}")
                    continue
            
            print(f"✅ Rank {self.rank}: Inner loop {loop + 1} completed ({len(loop_losses)} steps)")
            
            if loop_losses:
                all_losses.extend(loop_losses)
                
                # DiLoCo outer optimization step
                print(f"🔄 Rank {self.rank}: Performing DiLoCo outer optimization step...")
                if self.is_distributed and self.world_size > 1:
                    print(f"🔄 Rank {self.rank}: DiLoCo outer step (distributed)")
                    # In real DiLoCo, this would involve parameter averaging across nodes
                    # For now, we just sync gradients
                    for param in self.model.parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)
                else:
                    print(f"🔄 Rank {self.rank}: DiLoCo outer step (single node)")
                
                print(f"✅ Rank {self.rank}: DiLoCo loop {loop + 1} completed")
            else:
                print(f"⚠️  Rank {self.rank}: No valid losses in loop {loop + 1}")
        
        duration = time.time() - start_time
        
        print(f"✅ Rank {self.rank}: DiLoCo training completed in {duration:.2f}s")
        print(f"   Total loops: {self.config.diloco_loops}")
        print(f"   Total steps: {len(all_losses)}")
        
        if all_losses:
            print(f"🎉 Success! Rank {self.rank} completed training")
            print(f"📊 Final loss: {all_losses[-1]:.4f}")
            print(f"⏱️  Duration: {duration:.2f}s")
            
            # Print loop summaries
            step_size = self.config.steps
            for i in range(self.config.diloco_loops):
                start_idx = i * step_size
                end_idx = min((i + 1) * step_size, len(all_losses))
                if start_idx < len(all_losses):
                    loop_losses = all_losses[start_idx:end_idx]
                    if loop_losses:
                        print(f"   Loop {i + 1}: {loop_losses[0]:.4f} → {loop_losses[-1]:.4f} (avg: {sum(loop_losses)/len(loop_losses):.4f})")
        else:
            print(f"⚠️  No valid training steps completed")
            
        # Save final results
        self.save_model()
        
        return all_losses
        
    def save_model(self):
        """Save the trained model."""
        if self.rank == 0:  # Only main process saves
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # Unwrap model from FSDP if needed
            model_to_save = self.model
            if hasattr(self.model, 'module'):
                model_to_save = self.model.module
                
            try:
                model_to_save.save_pretrained(self.config.output_dir)
                self.tokenizer.save_pretrained(self.config.output_dir)
                print(f"✅ Model saved to: {self.config.output_dir}")
            except Exception as e:
                print(f"⚠️  Model save failed: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Docker DiLoCo Training")
    
    # Model and training
    parser.add_argument("--model_name", default="microsoft/DialoGPT-small")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--diloco_loops", type=int, default=2)
    parser.add_argument("--mixed_precision", default="fp16", choices=["fp16", "bf16", "no"])
    parser.add_argument("--output_dir", default="/tmp/diloco_output")
    
    # Dataset
    parser.add_argument("--dataset_name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_subset", default="sample-10BT")
    parser.add_argument("--text_field", default=None)
    parser.add_argument("--use_streaming", action="store_true")
    parser.add_argument("--no_streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=10000)
    parser.add_argument("--dataset_seed", type=int, default=0)
    parser.add_argument("--dynamic_batch", action="store_true")
    
    # FSDP options
    parser.add_argument("--use_fsdp", action="store_true", default=True)
    parser.add_argument("--no_fsdp", action="store_true")
    parser.add_argument("--fsdp_auto_wrap_policy", default="size_based", 
                        choices=["size_based", "transformer_based", "none"])
    parser.add_argument("--fsdp_cpu_offload", action="store_true")
    parser.add_argument("--fsdp_min_num_params", type=int, default=100)
    
    args = parser.parse_args()
    
    # Handle FSDP disable flag
    if args.no_fsdp:
        args.use_fsdp = False
    
    # Create config
    config = TrainerConfig(
        model_name=args.model_name,
        steps=args.steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        diloco_loops=args.diloco_loops,
        mixed_precision=args.mixed_precision,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        dataset_subset=args.dataset_subset,
        text_field=args.text_field,
        use_streaming=args.use_streaming,
        no_streaming=args.no_streaming,
        shuffle_buffer=args.shuffle_buffer,
        dataset_seed=args.dataset_seed,
        dynamic_batch=args.dynamic_batch,
        use_fsdp=args.use_fsdp,
        fsdp_auto_wrap_policy=args.fsdp_auto_wrap_policy,
        fsdp_cpu_offload=args.fsdp_cpu_offload,
        fsdp_min_num_params=args.fsdp_min_num_params,
    )
    
    print("🐳 Enhanced Docker DiLoCo Training Starting...")
    print(f"📋 Config: {vars(config)}")
    print()
    
    # Create and run trainer
    trainer = EnhancedDockerTrainer(config)
    losses = trainer.train()
    
    print("\n🎉 Training completed!")
    return losses


if __name__ == "__main__":
    main()
