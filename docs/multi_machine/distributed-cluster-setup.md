# Distributed GPU Cluster Setup: Multi-Node LLM Training

## System Overview

Your three-machine cluster:

| Machine | GPU | VRAM | CPU | RAM | Role |
|---------|-----|------|-----|-----|------|
| **Desktop (Main)** | RTX 5070 Ti | 12 GB | Ryzen 7 9800X3D | 32 GB | Master Node (Rank 0) |
| **Laptop** | RTX 4070 | 8 GB | Intel Ultra 9 185H | 16 GB | Worker Node (Rank 1) |
| **Desktop 2** | RTX 3070 Ti | 8 GB | Ryzen 7 3700X | 32 GB | Worker Node (Rank 2) |

**Total GPU VRAM:** 28 GB  
**Total GPU Memory:** ~45 GB effective (with gradient checkpointing)  
**Network:** Same LAN (Trondheim, local network)

---

## Architecture: Multi-Node Distributed Data Parallel (DDP)

### Why DDP?

✅ **Best for your setup:**
- Model fits on each GPU separately ✓
- Simple code modifications ✓
- Near-linear scaling ✓
- Lower network overhead than FSDP for your setup

**Your heterogeneous scaling:**
- Main desktop: 1 GPU
- Laptop: 1 GPU  
- Desktop 2: 1 GPU
- **World Size = 3 (3 total processes)**

---

## PHASE 1: Network & SSH Setup

### 1.1 Assign Static IPs to Each Machine

On each machine, assign static IPs on your local network. Example:

```
Main Desktop:    192.168.1.100
Laptop:          192.168.1.101
Desktop 2:       192.168.1.102
```

**Ubuntu 24.04 - Set Static IP (Netplan):**

```bash
sudo nano /etc/netplan/00-installer-config.yaml
```

Replace with:

```yaml
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: no
      addresses:
        - 192.168.1.100/24  # Change .100 to .101, .102 for other machines
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
```

Apply changes:

```bash
sudo netplan apply
ip addr show  # Verify new IP
```

### 1.2 SSH Passwordless Access (Master → Workers)

On **Master Desktop (192.168.1.100)**, generate SSH keys:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
```

On each **Worker Node**, add master's public key:

```bash
# SSH into worker
ssh user@192.168.1.101

# Add master's public key
mkdir -p ~/.ssh
echo "PASTE_MASTER_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh

# Test from master
exit
ssh user@192.168.1.101 "echo Connection successful"
```

Repeat for both workers (192.168.1.101 and 192.168.1.102).

### 1.3 Hostname Configuration (Optional but recommended)

Make it easier to reference machines. On each:

```bash
# Main Desktop
sudo hostnamectl set-hostname gpu-master

# Laptop
sudo hostnamectl set-hostname gpu-worker-1

# Desktop 2
sudo hostnamectl set-hostname gpu-worker-2

# Add to /etc/hosts on all machines
sudo nano /etc/hosts
```

Add:
```
192.168.1.100  gpu-master
192.168.1.101  gpu-worker-1
192.168.1.102  gpu-worker-2
```

Test:
```bash
ping gpu-worker-1
```

---

## PHASE 2: Environment Setup

### 2.1 NVIDIA GPU Drivers & CUDA (All Machines)

Check current setup:

```bash
nvidia-smi
nvcc --version
```

If missing, install:

```bash
sudo apt update
sudo apt install nvidia-driver-570  # Or latest available
sudo apt install nvidia-cuda-toolkit

# Verify installation
nvidia-smi
nvcc --version
```

### 2.2 PyTorch with CUDA Support (All Machines)

Install matching PyTorch versions on ALL machines (consistency is critical):

```bash
# Create conda environment on each machine
conda create -n distributed-training python=3.11 -y
conda activate distributed-training

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2.3 Additional Dependencies (All Machines)

```bash
# Core dependencies
pip install transformers datasets accelerate peft

# Unsloth (optimized LLM training)
pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git

# Monitoring & utilities
pip install tensorboard wandb

# Development tools
pip install jupyterlab ipython
```

### 2.4 Network Communication Backend

Verify NCCL (comes with PyTorch):

```bash
python -c "import torch; print(torch.cuda.nccl.version())"
```

NCCL enables GPU-to-GPU communication across the network.

---

## PHASE 3: Project Structure

Create on **Master Node** (this will be mirrored to workers):

```bash
mkdir -p ~/ai-cluster/{data,models,scripts,checkpoints,logs}
cd ~/ai-cluster

# Create this directory structure:
ai-cluster/
├── data/                    # Training datasets
│   └── processed/
├── models/                  # Base models, checkpoints
│   ├── base/
│   └── checkpoints/
├── scripts/
│   ├── train_distributed.py      # Main training script
│   ├── train_single_test.py       # Single-GPU test
│   └── launch_cluster.sh          # Launch script
├── logs/                    # TensorBoard logs, training logs
│   └── runs/
├── config/
│   ├── training_config.yaml
│   └── cluster_config.yaml
├── results/                 # Final models, evaluation results
└── README.md
```

---

## PHASE 4: Training Script Setup

### 4.1 Distributed Training Script Template

Create `~/ai-cluster/scripts/train_distributed.py`:

```python
"""
Multi-Node Distributed LLM Fine-tuning with Unsloth
Supports PyTorch DDP across multiple machines
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    TrainingArguments,
)
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from peft import LoraConfig, get_peft_model
import argparse
from datetime import datetime


def setup_distributed():
    """Initialize distributed training."""
    if not dist.is_available():
        raise RuntimeError("Requires distributed package to be available")
    
    dist.init_process_group("nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    if rank == 0:
        print(f"=== Distributed Setup ===")
        print(f"Rank: {rank}/{world_size}")
        print(f"Local Rank: {local_rank}")
        print(f"World Size: {world_size}")
        print(f"Backend: NCCL")
        print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def load_model_and_tokenizer(model_name: str, rank: int):
    """Load model with LoRA using Unsloth."""
    
    max_seq_length = 2048
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    
    # Add LoRA adapter
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    if rank == 0:
        model.print_trainable_parameters()
    
    return model, tokenizer, max_seq_length


def prepare_data(dataset_name: str, tokenizer, max_seq_length: int, rank: int):
    """Prepare dataset with distributed sampling."""
    
    if rank == 0:
        print(f"Loading dataset: {dataset_name}")
    
    # Load dataset
    dataset = load_dataset(dataset_name, split="train")
    
    def formatting_func(examples):
        output_texts = []
        for i in range(len(examples["text"])):
            output_texts.append(examples["text"][i])
        return output_texts
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            formatting_func(examples),
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    # Use DistributedSampler for multi-node training
    sampler = DistributedSampler(
        tokenized_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
    )
    
    train_loader = DataLoader(
        tokenized_dataset,
        sampler=sampler,
        batch_size=1,  # Per-GPU batch size (recommended for Unsloth DDP)
    )
    
    if rank == 0:
        print(f"Dataset size: {len(tokenized_dataset)}")
        print(f"Batches per epoch: {len(train_loader)}")
    
    return train_loader, sampler


def train(args):
    """Main training loop."""
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    
    # Load model
    model, tokenizer, max_seq_length = load_model_and_tokenizer(
        args.model_name, rank
    )
    
    # Move to GPU
    model = model.to(local_rank)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    # Prepare data
    train_loader, sampler = prepare_data(
        args.dataset_name, tokenizer, max_seq_length, rank
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    
    # Learning rate scheduler
    num_training_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )
    
    # Training loop
    model.train()
    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)  # Shuffle different data per epoch
        
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Move batch to GPU
            input_ids = batch["input_ids"].to(local_rank)
            attention_mask = batch["attention_mask"].to(local_rank)
            labels = batch["input_ids"].clone().to(local_rank)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Logging (only rank 0)
            if rank == 0 and batch_idx % args.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(
                    f"Epoch [{epoch+1}/{args.num_epochs}] "
                    f"Step [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {avg_loss:.4f}"
                )
        
        # Save checkpoint (only rank 0)
        if rank == 0:
            checkpoint_path = f"../checkpoints/epoch_{epoch+1}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": total_loss / len(train_loader),
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved: {checkpoint_path}")
    
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Multi-Node Distributed LLM Training")
    parser.add_argument("--model-name", default="meta-llama/Llama-2-7b-hf", type=str)
    parser.add_argument("--dataset-name", default="wikitext", type=str)
    parser.add_argument("--learning-rate", default=2e-4, type=float)
    parser.add_argument("--num-epochs", default=3, type=int)
    parser.add_argument("--log-interval", default=10, type=int)
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
```

### 4.2 Single-GPU Test Script

Create `~/ai-cluster/scripts/train_single_test.py`:

```python
"""Test script to verify environment on a single GPU."""

import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

def test_environment():
    print("=== Environment Test ===")
    
    # Check CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Check PyTorch
    print(f"\nPyTorch version: {torch.__version__}")
    
    # Check Unsloth
    print(f"\nTesting Unsloth model loading...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/tinyllama-bnb-4bit",
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
        )
        print("✓ Unsloth loaded successfully")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        
        # Quick forward pass
        test_input = tokenizer("Hello, how are you?", return_tensors="pt").to(0)
        with torch.no_grad():
            output = model(**test_input)
        print("✓ Forward pass successful")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    print("\n=== All tests passed! ===")
    return True

if __name__ == "__main__":
    test_environment()
```

Run on each machine to verify setup:

```bash
cd ~/ai-cluster/scripts
python train_single_test.py
```

---

## PHASE 5: Launching Distributed Training

### 5.1 Multi-Node Launch Script

Create `~/ai-cluster/scripts/launch_cluster.sh`:

```bash
#!/bin/bash

# Distributed training launcher for multi-node setup

MASTER_ADDR="192.168.1.100"
MASTER_PORT="29500"
NNODES=3
NPROC_PER_NODE=1  # 1 GPU per machine

# Node rank (0 for master, 1+ for workers)
NODE_RANK=${1:-0}

# Training script arguments
MODEL_NAME="meta-llama/Llama-2-7b-hf"
DATASET_NAME="wikitext"
EPOCHS=3
LR=2e-4

echo "================================"
echo "Distributed Training Launcher"
echo "================================"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Total Nodes: $NNODES"
echo "Processes per Node: $NPROC_PER_NODE"
echo "Node Rank: $NODE_RANK"
echo "================================"

# Launch with torchrun
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --node_rank=$NODE_RANK \
    train_distributed.py \
    --model-name $MODEL_NAME \
    --dataset-name $DATASET_NAME \
    --num-epochs $EPOCHS \
    --learning-rate $LR

```

Make executable:

```bash
chmod +x ~/ai-cluster/scripts/launch_cluster.sh
```

### 5.2 Launching Training

**On Master Desktop (192.168.1.100):**

```bash
cd ~/ai-cluster/scripts
./launch_cluster.sh 0  # NODE_RANK=0 (master)
```

**On Laptop (192.168.1.101):**

Open SSH connection and run:

```bash
ssh user@192.168.1.101
cd ~/ai-cluster/scripts
./launch_cluster.sh 1  # NODE_RANK=1 (worker)
```

**On Desktop 2 (192.168.1.102):**

```bash
ssh user@192.168.1.102
cd ~/ai-cluster/scripts
./launch_cluster.sh 2  # NODE_RANK=2 (worker)
```

**Order:** Start from any node; torchrun handles synchronization.

---

## PHASE 6: Advanced Configurations

### 6.1 Model Parallelism (If single GPU insufficient)

If model doesn't fit on single GPU, use `device_map="balanced"` in Unsloth:

```python
model = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-2-70b-hf",
    device_map="balanced",  # Splits model across GPUs on same machine
    max_seq_length=2048,
)
```

### 6.2 Gradient Checkpointing (Memory Optimization)

Already enabled in training script with `use_gradient_checkpointing="unsloth"`.

Additional memory optimization:

```python
# Reduce batch size
batch_size = 1

# Enable gradient accumulation
gradient_accumulation_steps = 8

# Use mixed precision
torch.autocast(device_type="cuda", dtype=torch.float16)
```

### 6.3 Monitoring with TensorBoard

On Master:

```bash
tensorboard --logdir=~/ai-cluster/logs/runs/
```

Access at `http://localhost:6006`

---

## PHASE 7: Troubleshooting

### Common Issues

**1. Connection refused error:**
```
RuntimeError: connect() call failed
```
Solution: Check firewall, ensure all nodes can reach master on port 29500:
```bash
telnet 192.168.1.100 29500
```

**2. GPU out of memory:**
```
RuntimeError: CUDA out of memory
```
Solutions:
- Reduce batch size to 1
- Enable gradient checkpointing
- Use 8-bit quantization (`load_in_4bit=False` → `load_in_8bit=True`)

**3. Process hangs during training:**
Ensure environment variables are identical on all nodes:
```bash
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
```

**4. Different PyTorch versions:**
All nodes MUST have identical PyTorch/CUDA versions:
```bash
python -c "import torch; print(torch.__version__)"
```

### Debug Commands

On any node:

```bash
# Check GPU
nvidia-smi

# Check connectivity
ssh -v user@gpu-worker-1 "nvidia-smi"

# Check PyTorch distributed
python -c "import torch.distributed as dist; print('OK')"

# Monitor network
iftop  # Install: sudo apt install iftop
```

---

## PHASE 8: Performance Considerations

### Expected Throughput

With your setup (3 GPUs, DDP):

| Task | Single GPU | 3-GPU DDP | Theoretical |
|------|-----------|----------|------------|
| Training speed | 1x | ~2.7x | 3x |
| Memory per GPU | 12GB | ~9GB | Same |
| Time per epoch | 2h | ~45min | Scales |

**Why not 3x?** Network communication overhead (~10-15% slowdown on same LAN).

### Network Optimization

Your local network is fast (1 Gbps+), so most bottleneck is GPU-GPU syncing within DDP. For large scale:

- Keep batch size small (per-GPU)
- Gradient accumulation for larger effective batch
- Monitor network with `iftop` during training

---

## Next Steps

1. **Phase 1:** Setup network and SSH ✓
2. **Phase 2:** Install environments ✓
3. **Phase 3:** Create project structure ✓
4. **Phase 4:** Create training scripts ✓
5. **Phase 5:** Test on single GPU (all nodes)
6. **Phase 6:** Launch distributed training
7. **Phase 7:** Monitor and optimize

---

## Quick Reference Commands

```bash
# Test single GPU
python scripts/train_single_test.py

# Start master node
cd scripts && ./launch_cluster.sh 0

# Start worker node
ssh user@192.168.1.101
cd ~/ai-cluster/scripts && ./launch_cluster.sh 1

# Monitor training
tensorboard --logdir=logs/runs/

# Kill training on all nodes
pkill -f torchrun
```

---

## Resources

- PyTorch DDP: https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html
- Unsloth Multi-GPU: https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth
- torchrun: https://pytorch.org/docs/stable/elastic/run.html
- NCCL: https://docs.nvidia.com/deeplearning/nccl/user-guide/
