# Distributed Training: Quick Start Guide

## 5-Minute Setup Summary

### Prerequisites
- ✅ All 3 machines on same network
- ✅ Ubuntu 24.04 with NVIDIA drivers installed
- ✅ SSH access between machines

### Step 1: Network Setup (10 min)

**Assign static IPs:**
```bash
# On each machine, edit netplan
sudo nano /etc/netplan/00-installer-config.yaml

# Main Desktop: 192.168.1.100
# Laptop: 192.168.1.101
# Desktop 2: 192.168.1.102

sudo netplan apply
```

**Test connectivity:**
```bash
# From main desktop
ping 192.168.1.101
ping 192.168.1.102
```

### Step 2: SSH Passwordless (5 min)

**On Main Desktop:**
```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
ssh-copy-id -i ~/.ssh/id_ed25519.pub user@192.168.1.101
ssh-copy-id -i ~/.ssh/id_ed25519.pub user@192.168.1.102

# Test
ssh user@192.168.1.101 "echo OK"
ssh user@192.168.1.102 "echo OK"
```

### Step 3: Install Dependencies (15 min, run on ALL machines)

```bash
# Create environment
conda create -n dist-train python=3.11 -y
conda activate dist-train

# PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ML Stack
pip install transformers datasets accelerate peft unsloth[colab-new] tensorboard

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 4: Project Structure (5 min)

**On Main Desktop:**
```bash
mkdir -p ~/Development/Projects/ai-ml-pipeline/ai-cluster/{scripts,data,checkpoints,logs}
cd ~/Development/Projects/ai-ml-pipeline/ai-cluster

# Copy training scripts to scripts/
# (See attached train_distributed.py)
```

### Step 5: Launch Training (1 min per machine)

**Terminal 1 - Main Desktop:**
```bash
cd ~/Development/Projects/ai-ml-pipeline/ai-cluster/scripts
torchrun \
    --nnodes=3 \
    --nproc_per_node=1 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=192.168.1.100:29500 \
    --node_rank=0 \
    train_distributed.py
```

**Terminal 2 - Laptop (SSH):**
```bash
ssh user@192.168.1.101
cd ~/Development/Projects/ai-ml-pipeline/ai-cluster/scripts
torchrun \
    --nnodes=3 \
    --nproc_per_node=1 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=192.168.1.100:29500 \
    --node_rank=1 \
    train_distributed.py
```

**Terminal 3 - Desktop 2 (SSH):**
```bash
ssh user@192.168.1.102
cd ~/Development/Projects/ai-ml-pipeline/ai-cluster/scripts
torchrun \
    --nnodes=3 \
    --nproc_per_node=1 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=192.168.1.100:29500 \
    --node_rank=2 \
    train_distributed.py
```

**Expected Output (on rank 0):**
```
=== Distributed Setup ===
Rank: 0/3
Local Rank: 0
World Size: 3
Backend: NCCL
GPU: NVIDIA GeForce RTX 5070 Ti
```

---

## Configuration Reference

### Training Parameters (in train_distributed.py)

```python
# Edit these for your needs
MODEL_NAME = "meta-llama/Llama-2-7b-hf"        # Model to train
DATASET_NAME = "wikitext"                       # Dataset
EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 1  # Per GPU (recommended for DDP + Unsloth)
GRADIENT_ACCUMULATION = 8  # For larger effective batch
```

### Performance Tips

| Scenario | Solution |
|----------|----------|
| Out of Memory | Reduce batch size, enable gradient checkpointing |
| Slow training | Check network latency: `ping -c 10 worker1` |
| One node slow | Monitor CPU/GPU: `watch -n 1 nvidia-smi` |
| Uneven training speed | Different GPU architectures expected, DDP handles |

---

## File Locations

After setup, your structure should be:

```
~/Development/Projects/ai-ml-pipeline/ai-cluster/
├── scripts/
│   ├── train_distributed.py       # Main training script
│   ├── train_single_test.py       # Single GPU test
│   └── launch_cluster.sh          # Easy launcher
├── data/
│   └── processed/                 # Training data
├── checkpoints/
│   └── epoch_1.pt                 # Saved models
├── logs/
│   └── runs/                      # TensorBoard logs
└── README.md
```

---

## Monitoring Training

### Real-time Metrics

While training:

```bash
# On main desktop, in new terminal
tensorboard --logdir ~/Development/Projects/ai-ml-pipeline/ai-cluster/logs/runs/

# Access at http://localhost:6006
```

### GPU Usage

```bash
# On any machine
watch -n 1 nvidia-smi
```

Should show:
- 3 Python processes (1 per machine, if DDP)
- GPU memory increasing during warmup
- GPU utilization ~70-90% during training

---

## Troubleshooting

### "connect() call failed"

```bash
# Check master is reachable
ping 192.168.1.100

# Check port open
telnet 192.168.1.100 29500

# Solution: Ensure no firewall blocking
sudo ufw allow 29500
```

### "CUDA out of memory"

```python
# In train_distributed.py, reduce batch size
BATCH_SIZE = 1  # Or even 1
GRADIENT_ACCUMULATION = 16  # Increase this instead
```

### Processes hang after starting

Common cause: torch.distributed timeout or rank mismatch

```bash
# Kill everything
pkill -f torchrun

# Check NCCL debug
export NCCL_DEBUG=INFO
# Re-run training
```

### One worker much slower

Expected with heterogeneous GPUs (5070 Ti vs 4070 vs 3070 Ti). DDP is bottlenecked by slowest GPU. This is normal and expected.

---

## Next: Scale Beyond 3 Machines

This setup works for 3+ machines. To add more:

1. Assign new static IP (e.g., 192.168.1.103)
2. Install conda environment identical to others
3. Setup SSH passwordless from master
4. Update `--nnodes=N` in training command
5. Run torchrun on new machine with `--node_rank=3` (and so on)

---

## Performance Expectations

Training Llama-2-7B with your setup:

| Metric | Single GPU | 3-GPU DDP |
|--------|-----------|----------|
| Throughput | 1x | ~2.5-2.7x |
| Memory per GPU | 12GB (5070 Ti) | ~8-9GB |
| Time per epoch | 2h | ~45 min |
| Network overhead | N/A | ~10-15% |

---

## Key Commands Cheatsheet

```bash
# Single GPU test (do on each machine)
cd ~/ai-cluster/scripts
python train_single_test.py

# Start training (run on each machine with different node_rank)
torchrun --nnodes=3 --nproc_per_node=1 --rdzv_id=100 \
  --rdzv_backend=c10d --rdzv_endpoint=192.168.1.100:29500 \
  --node_rank=0 train_distributed.py

# Monitor
tensorboard --logdir logs/runs/

# Kill all training
pkill -f torchrun

# Check NCCL connectivity
python -c "import torch.distributed as dist; print('OK')"

# View GPU
nvidia-smi
```

---

## Important Notes

⚠️ **Critical Requirements:**

1. **Identical environments** on all machines
   - Same PyTorch version
   - Same CUDA version
   - Same Python version

2. **Network connectivity** must be stable
   - Same LAN recommended
   - Firewall must allow port 29500
   - SSH passwordless access required

3. **Synchronization**
   - Master node (rank 0) must stay running
   - All workers must use same training script
   - All must start within ~30 seconds of each other

⚠️ **Data Handling:**

- Training data can live on master, or
- Copy dataset to each machine for local I/O, or
- Use NFS mount (advanced, not needed for your setup)

---

## Success Indicators

After starting training, you should see:

✅ All 3 processes show in `nvidia-smi`  
✅ GPU memory increases to ~8-9GB per GPU  
✅ Training loss decreasing each step  
✅ All nodes stay synchronized (no rank warnings)  
✅ TensorBoard shows 3 GPU utilizations trending up  

---

## Next Steps

1. Follow "5-Minute Setup" above
2. Run `train_single_test.py` on each machine
3. Start distributed training with torchrun commands
4. Monitor with TensorBoard
5. Adjust hyperparameters based on loss/speed
6. Scale to larger models once comfortable

Good luck! 🚀
