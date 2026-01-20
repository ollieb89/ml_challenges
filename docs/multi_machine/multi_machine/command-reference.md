# Command Reference Sheet

A quick copy-paste guide for common commands during setup and training.

---

## PHASE 1: Network Setup

### Set Static IP on Ubuntu 24.04

```bash
# Edit network configuration
sudo nano /etc/netplan/00-installer-config.yaml

# For main desktop (192.168.1.100)
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: no
      addresses:
        - 192.168.1.100/24
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]

# For laptop (192.168.1.101)
# Change .100 to .101 in addresses

# Apply changes
sudo netplan apply
ip addr show  # Verify
```

### Verify Network Connectivity

```bash
# From main desktop
ping -c 5 192.168.1.101
ping -c 5 192.168.1.102

# Test SSH connectivity
ssh -v ob@192.168.1.101
```

---

## PHASE 2: SSH Setup (Master Only)

### Generate SSH Keys

```bash
# Generate on main desktop
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""

# View public key
cat ~/.ssh/id_ed25519.pub
```

### Copy SSH Key to Workers

```bash
# Copy to laptop
ssh-copy-id -i ~/.ssh/id_ed25519.pub ob@192.168.1.101

# Copy to desktop 2
ssh-copy-id -i ~/.ssh/id_ed25519.pub ollie@192.168.1.102

# Test passwordless access
ssh ob@192.168.1.101 "echo Connection successful"
ssh ollie@192.168.1.102 "echo Connection successful"
```

---

## PHASE 3: Environment Setup (All Machines)

### Create Conda Environment

```bash
# Create environment
conda create -n dist-train python=3.11 -y

# Activate environment
conda activate dist-train

# Verify Python version
python --version
```

### Install PyTorch with CUDA

```bash
# Install PyTorch 2.1 with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

### Install ML Stack

```bash
# Core dependencies
pip install transformers datasets accelerate peft

# Unsloth (optimized LLM training)
pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git

# Monitoring
pip install tensorboard wandb

# Development
pip install jupyterlab ipython
```

### Verify Installation

```bash
# Check all imports
python -c "
import torch
import transformers
import unsloth
import peft
import datasets
print('✓ All imports successful')
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

---

## PHASE 4: Project Structure (Master Only)

### Create Directory Structure

```bash
# Create all directories
mkdir -p ~/Tools/ai-ml-pipeline/ai-cluster/{scripts,data,checkpoints,logs,results,config}

# Verify structure
tree ~/Tools/ai-ml-pipeline/ai-cluster
```

### Copy Training Scripts

```bash
# Copy from these docs to ~/ai-cluster/scripts/:
# - train_production.py
# - train_single_test.py

# Make executable
    chmod +x ~/Tools/ai-ml-pipeline/ai-cluster/scripts/train_production.py
    chmod +x ~/Tools/ai-ml-pipeline/ai-cluster/scripts/train_single_test.py
```

---

## PHASE 5: Pre-Training Tests

### Single GPU Test (On Each Machine)

```bash
# On main desktop
cd ~/Tools/ai-ml-pipeline/ai-cluster/scripts
python -c "
import torch
from unsloth import FastLanguageModel

print('Loading TinyLlama...')
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='unsloth/tinyllama-bnb-4bit',
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)
print(f'✓ Model loaded')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"
```

### Environment Verification

```bash
# Check each machine matches
python --version
python -c "import torch; print(torch.__version__)"
nvcc --version
nvidia-smi
```

---

## PHASE 6: Launch Distributed Training

### Master Node (192.168.1.100) - Terminal 1

```bash
cd ~/Tools/ai-ml-pipeline/ai-cluster/scripts
torchrun \
    --nnodes=3 \
    --nproc_per_node=1 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=192.168.1.100:29500 \
    --node_rank=0 \
    train_production.py \
    --model-name unsloth/tinyllama-bnb-4bit \
    --dataset-name wikitext \
    --num-epochs 3 \
    --learning-rate 2e-4
```

### Worker 1 (192.168.1.101) - Terminal 2

```bash
ssh ob@192.168.1.101

cd ~/Tools/ai-ml-pipeline/ai-cluster/scripts
torchrun \
    --nnodes=3 \
    --nproc_per_node=1 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=192.168.1.100:29500 \
    --node_rank=1 \
    train_production.py \
    --model-name unsloth/tinyllama-bnb-4bit \
    --dataset-name wikitext \
    --num-epochs 3 \
    --learning-rate 2e-4
```

### Worker 2 (192.168.1.102) - Terminal 3

```bash
ssh user@192.168.1.102

cd ~/Tools/ai-ml-pipeline/ai-cluster/scripts
torchrun \
    --nnodes=3 \
    --nproc_per_node=1 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=192.168.1.100:29500 \
    --node_rank=2 \
    train_production.py \
    --model-name unsloth/tinyllama-bnb-4bit \
    --dataset-name wikitext \
    --num-epochs 3 \
    --learning-rate 2e-4
```

---

## PHASE 7: Monitoring During Training

### GPU Monitoring - Terminal 4

```bash
# Live GPU monitoring (updates every 1 second)
watch -n 1 nvidia-smi

# Or static view
nvidia-smi

# With memory only
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
```

### TensorBoard - Terminal 5

```bash
# Start TensorBoard
tensorboard --logdir ~/Tools/ai-ml-pipeline/ai-cluster/logs/runs/

# Access at http://localhost:6006
# Open in browser: http://127.0.0.1:6006
```

### Log Viewing - Terminal 6

```bash
# View logs (rank 0 is main)
tail -f ~/Tools/ai-ml-pipeline/ai-cluster/logs/training_rank0_*.log

# Or all logs
tail -f ~/Tools/ai-ml-pipeline/ai-cluster/logs/training_rank*.log

# Search for errors
grep ERROR ~/ai-cluster/logs/training_rank*.log
```

---

## EMERGENCY COMMANDS

### Kill All Training

```bash
# On each machine
pkill -f torchrun

# Or be specific
killall torchrun
```

### Free GPU Memory

```bash
# Reset GPU
nvidia-smi --gpu-reset

# Or restart NVIDIA daemon
sudo systemctl restart nvidia-persistenced
```

### Check Connectivity

```bash
# Is port 29500 open?
telnet 192.168.1.100 29500

# Network latency
ping -c 10 192.168.1.101

# SSH connectivity
ssh -v user@192.168.1.101
```

### Debug Distributed

```bash
# Enable NCCL debug logging
export NCCL_DEBUG=INFO
torchrun ... train_production.py

# Check NCCL version
python -c "import torch; print(torch.cuda.nccl.version())"
```

---

## Useful One-Liners

### Check All Machines Are Ready

```bash
# Run from main desktop
for i in 100 101 102; do
  echo "=== 192.168.1.$i ==="
  ssh user@192.168.1.$i "nvidia-smi | head -5"
done
```

### Get All GPU Names

```bash
for i in 100 101 102; do
  echo -n "192.168.1.$i: "
  ssh user@192.168.1.$i "nvidia-smi --query-gpu=name --format=csv,noheader"
done
```

### Check Disk Space

```bash
# On each machine
df -h ~/Tools/ai-ml-pipeline/ai-cluster/

# Remote check
ssh ob@192.168.1.101 "df -h ~/Tools/ai-ml-pipeline/ai-cluster/"
```

### Monitor Network During Training

```bash
# Install if not present
sudo apt install iftop

# Monitor in real-time
sudo iftop -n
```

### Quick Training Sanity Check

```bash
# Single machine, single epoch, tiny dataset
python -c "
import torch
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained('unsloth/tinyllama-bnb-4bit', max_seq_length=512, dtype=torch.float16, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=8, lora_alpha=16, target_modules=['q_proj', 'v_proj'], lora_dropout=0.05, bias='none', use_gradient_checkpointing='unsloth', random_state=42)

# Mock training
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
for i in range(5):
    inp = tokenizer('Hello world', return_tensors='pt').to(0)
    out = model(**inp, labels=inp['input_ids'])
    out.loss.backward()
    opt.step()
    opt.zero_grad()
    print(f'Step {i+1}: Loss = {out.loss.item():.4f}')
"
```

---

## Configuration Adjustment Commands

### Change Model Size

```bash
# TinyLlama (1.1B - fast, for testing)
--model-name unsloth/tinyllama-bnb-4bit

# Llama-2-7B (recommended)
--model-name meta-llama/Llama-2-7b-hf

# Llama-2-13B (requires optimization)
--model-name meta-llama/Llama-2-13b-hf
```

### Change Dataset

```bash
# WikiText (recommended for testing)
--dataset-name wikitext

# Open Web Text (large, production)
--dataset-name openwebtext

# Custom dataset
# See train_production.py for custom implementation
```

### Change Batch Configuration

```bash
# Smaller (if OOM)
--per-device-train-batch-size=1
--gradient-accumulation-steps=4

# Larger (if memory available)
--per-device-train-batch-size=2
--gradient-accumulation-steps=16
```

### Change Learning Rate

```bash
# Lower (more stable, slower convergence)
--learning-rate 1e-4

# Higher (faster, may diverge)
--learning-rate 5e-4

# Standard for LoRA
--learning-rate 2e-4
```

---

## Post-Training Commands

### List Checkpoints

```bash
# Show all checkpoints
ls -lh ~/Tools/ai-ml-pipeline/ai-cluster/checkpoints/

# Show latest checkpoint
ls -lht ~/Tools/ai-ml-pipeline/ai-cluster/checkpoints/ | head -5
```

### Load and Test Trained Model

```python
from unsloth import FastLanguageModel
import torch

# Load base model + LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-2-7b-hf",
    adapter_name="./lora-weights",
)

# Generate
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(0)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### Save Model for Inference

```bash
# Copy checkpoint for deployment
cp ~/Tools/ai-ml-pipeline/ai-cluster/checkpoints/epoch_3_loss_0.4521.pt ~/my-trained-model.pt

# Export to HuggingFace format
# See train_production.py for save_pretrained() usage
```

---

## Troubleshooting Quick Commands

### Port Already in Use

```bash
# Find process using port 29500
sudo lsof -i :29500

# Kill the process
sudo kill -9 <PID>
```

### CUDA Out of Memory

```bash
# Check GPU memory
nvidia-smi

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Or restart nvidia-driver
sudo systemctl restart nvidia-persistenced
```

### SSH Key Issues

```bash
# Regenerate key
rm ~/.ssh/id_ed25519*
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""

# Copy to workers again
ssh-copy-id -i ~/.ssh/id_ed25519.pub user@192.168.1.101
```

### Network Timeouts

```bash
# Check network connectivity
ping -c 10 192.168.1.101

# Check latency (should be < 5ms)
# If > 10ms, network is slow

# Set NCCL timeout higher
export NCCL_SOCKET_TIMEOUT=3600
```

---

## Success Indicators

Watch for these during training:

```
✓ All 3 processes show in nvidia-smi
✓ GPU memory increases to 8-10 GB per GPU
✓ GPU utilization 70-90%
✓ Training loss decreases each step
✓ TensorBoard shows 3 GPU curves
✓ Logs show all ranks synchronized
✓ No CUDA out of memory errors
✓ No timeout errors
✓ Checkpoints saved at each epoch
```

---

## Cheat Sheet: Copy All Commands

```bash
# Quick setup recap (run these in order)

# 1. Setup network (on each machine)
sudo nano /etc/netplan/00-installer-config.yaml
sudo netplan apply
ip addr show

# 2. SSH setup (on master)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
ssh-copy-id -i ~/.ssh/id_ed25519.pub user@192.168.1.101
ssh-copy-id -i ~/.ssh/id_ed25519.pub user@192.168.1.102

# 3. Environment (on each machine)
conda create -n dist-train python=3.11 -y
conda activate dist-train
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate peft unsloth[colab-new] tensorboard

# 4. Create structure (on master)
mkdir -p ~/Tools/ai-ml-pipeline/ai-cluster/{scripts,data,checkpoints,logs,results,config}

# 5. Launch training (on each terminal with different node_rank)
# See PHASE 6 above for full commands
```

---

## Keep This Sheet Handy

Print or bookmark this file when running distributed training!
