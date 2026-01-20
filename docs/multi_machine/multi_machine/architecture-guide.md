# Architecture & Decision Guide

## Your Multi-Node GPU Cluster Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    LOCAL AREA NETWORK (LAN)                     │
│              192.168.1.0/24 (Gigabit Ethernet)                  │
└─────────────────────────────────────────────────────────────────┘
                  ▲                ▲                 ▲
                  │                │                 │
    ┌─────────────┴────────┐  ┌────┴─────────┐  ┌───┴──────────┐
    │   MASTER NODE        │  │ WORKER NODE 1│  │WORKER NODE 2 │
    │   (192.168.1.100)    │  │(192.168.1.101)  │(192.168.1.102) │
    │                      │  │              │  │              │
    │ ┌──────────────────┐ │  │┌────────────┐│  │┌────────────┐│
    │ │ RTX 5070 Ti 12GB │ │  ││RTX 4070 8GB││  ││RTX 3070 Ti ││
    │ │ VRAM: 12 GB      │ │  ││VRAM: 8 GB  ││  ││VRAM: 8 GB  ││
    │ │ CUDA cores: 5888 │ │  ││CUDA: 5888  ││  ││CUDA: 5888  ││
    │ └──────────────────┘ │  │└────────────┘│  │└────────────┘│
    │                      │  │              │  │              │
    │ CPU: 9800X3D (16c)   │  │Core Ultra 9  │  │Ryzen 7 3700X │
    │ RAM: 32 GB           │  │185H (22c)    │  │(16c) 32GB    │
    │ Ubuntu 24.04         │  │16 GB RAM     │  │Ubuntu 24.04  │
    │ CUDA 12.1            │  │Ubuntu 24.04  │  │CUDA 12.1     │
    │ PyTorch 2.x          │  │CUDA 12.1     │  │PyTorch 2.x   │
    │                      │  │PyTorch 2.x   │  │              │
    └──────────────────────┘  └──────────────┘  └──────────────┘
         Rank 0 (Master)         Rank 1         Rank 2 (Worker)

    NCCL over Ethernet (PyTorch Distributed Data Parallel)
          Inter-GPU Communication Backbone
```

---

## Data Flow During Training

```
Training Data
    │
    ├─► GPU 0 (Master)
    │   - Loads batch slice
    │   - Forward pass
    │   - Compute loss & gradients
    │
    ├─► GPU 1 (Worker 1)
    │   - Loads batch slice
    │   - Forward pass
    │   - Compute loss & gradients
    │
    └─► GPU 2 (Worker 2)
        - Loads batch slice
        - Forward pass
        - Compute loss & gradients

        │
        ▼
    NCCL AllReduce
    (Synchronize gradients across all GPUs)
        │
        ▼
    Optimizer Step
    (Update model weights on all GPUs simultaneously)
        │
        ▼
    Next Training Step
```

---

## Memory Distribution

### Per-GPU Memory Breakdown (Llama-2-7B with LoRA)

```
┌─────────────────────────────────────────┐
│ GPU Memory (12 GB for RTX 5070 Ti)      │
├─────────────────────────────────────────┤
│ Model Weights (4-bit)       │ ~2.8 GB   │  ← Quantized LLM
│ LoRA Weights               │ ~0.5 GB   │  ← Adapter weights
│ Optimizer State (AdamW)     │ ~2.0 GB   │  ← Adam momentum + variance
│ Gradient Buffer             │ ~2.0 GB   │  ← Backward pass gradients
│ Activations/Cache           │ ~2.5 GB   │  ← Forward pass cache
│ Free/Overhead               │ ~0.2 GB   │  ← Padding & system
├─────────────────────────────────────────┤
│ TOTAL                       │ ~10.0 GB  │  ← Safe margin to 12 GB
└─────────────────────────────────────────┘

Result: Fits comfortably on RTX 5070 Ti (12 GB)
Result: Tight on RTX 4070 (8 GB) - needs careful tuning
Result: Tight on RTX 3070 Ti (8 GB) - needs careful tuning
```

### Heterogeneous GPU Scaling

With your mismatched GPUs:

```
Master (12 GB)   ████████████ Capacity: 10 GB usable
Worker 1 (8 GB)  ████████     Capacity: 6.5 GB usable
Worker 2 (8 GB)  ████████     Capacity: 6.5 GB usable

DDP Bottleneck: SLOWEST GPU (8 GB workers)
→ Training speed limited by Worker 1 & 2
→ Expected speedup: ~2.5x (not 3x) due to slowest link
```

---

## Decision Tree: Which Configuration?

```
┌─ START: Choose Training Approach
│
├─ Do you want to train Llama-7B?
│  ├─ YES → Continue below
│  └─ NO → Adjust model size accordingly
│
├─ Is model fitting in single GPU?
│  ├─ YES ─→ Use DDP (Distributed Data Parallel) ✓
│  │         (Your setup, 3 separate GPUs)
│  │
│  └─ NO ──→ Check total VRAM
│      ├─ Total VRAM > Model size?
│      │  ├─ YES → Use FSDP or device_map="balanced"
│      │  └─ NO → Quantize more aggressively
│
├─ Do you want to use Unsloth?
│  ├─ YES → Setup DDP with Unsloth (Recommended)
│  │       - Use torchrun launcher
│  │       - Per-GPU batch size = 1
│  │       - Enable gradient accumulation
│  │
│  └─ NO → Use standard transformers DDP
│         - More flexibility
│         - Slightly slower
│
├─ Network fast enough? (LAN > 100 Mbps)
│  ├─ YES → All good, proceed ✓
│  └─ NO → Consider NFS data mount
│          (Reduces per-step communication)
│
└─ All 3 nodes have identical environment?
   ├─ YES → Launch training ✓✓✓
   └─ NO → Sync environments first
```

---

## Training Scenarios & Recommendations

### Scenario 1: Quick Test (Start here)

**Goal:** Verify distributed setup works

```
Model:       TinyLlama-1.1B (small, fast)
Batch Size:  1 per GPU
Epochs:      1
Dataset:     WikiText (small)
Command:     torchrun with --node_rank for each machine
Expected:    ~5 min total training time
```

**Then scale up after confirming:**
- All ranks initialize ✓
- GPU memory is safe ✓
- Data loads correctly ✓

---

### Scenario 2: Standard Production (Recommended)

**Goal:** Fine-tune Llama-2-7B efficiently

```
Model:       Llama-2-7B (good balance)
Batch Size:  1 per GPU
Gradient Accumulation: 8
Epochs:      3
Dataset:     Custom domain data
Settings:    4-bit quantization + LoRA
Command:     Use train_production.py
Expected:    ~2-3 hours total for 3 epochs
```

**Memory Usage:**
- RTX 5070 Ti: ~10 GB / 12 GB ✓ Safe
- RTX 4070:    ~8 GB / 8 GB   ⚠️  Tight
- RTX 3070 Ti: ~8 GB / 8 GB   ⚠️  Tight

---

### Scenario 3: Advanced (After Scenario 2)

**Goal:** Fine-tune larger model with mixed precision

```
Model:       Llama-2-13B or Code Llama-13B
Batch Size:  1 per GPU
Gradient Accumulation: 12
Epochs:      1-2 (test)
Quantization: 8-bit or 4-bit + FSDP
Settings:    device_map="balanced" on workers
Expected:    ~4-5 hours for 1 epoch
```

**Requires:** model_parallelism within each node

---

## Performance Expectations

### Training Speed Comparison

```
Model: Llama-2-7B | Dataset: WikiText | Seq Len: 2048

Single GPU (Best):
  - GPU: RTX 5070 Ti
  - Throughput: ~150 tokens/sec
  - Time/epoch: 2 hours

Distributed 3-GPU (Expected):
  - Setup: 3x DDP
  - Throughput: ~380-400 tokens/sec (2.5x speedup)
  - Time/epoch: ~45-50 minutes
  - Efficiency: ~85% (15% overhead from network)

Why not 3x?
  - Network sync overhead (~10-15%)
  - Heterogeneous GPU speeds (8 GB workers slower)
  - Gradient communication latency
```

### Network Bandwidth Impact

```
Gradient Exchange per step (Llama-7B):
  - Gradient size: ~1.6 GB (per GPU)
  - All-Reduce overhead: ~100ms (over 1 Gbps Ethernet)

Total Training Time Breakdown:
  - Compute: 85%
  - Communication: 12%
  - I/O: 3%

Optimization: Gradient compression (not implemented yet)
```

---

## Scaling Beyond 3 Machines

### To add 4th machine (RTX 4090, RTX 3090):

```yaml
Changes:
  1. Add machine to LAN (static IP 192.168.1.103)
  2. Install identical conda environment
  3. Copy ~/ai-cluster to new machine
  4. Update torchrun command:
     --nnodes=4  # Was 3
     --node_rank=3  # For 4th machine
  5. Launch new machine's training process

Expected speedup:
  - Adding similar GPU: ~0.8x additional (4.0x total)
  - Adding faster GPU: ~1.0x additional (4.2x+ total)
```

### Maximum practical scaling (same LAN):

```
Machines: Up to 8-10 on same LAN ✓
Bandwidth: 1 Gbps Ethernet adequate for < 10 GPUs
Network overhead becomes >15% at 10+ GPUs
→ Consider 10 Gbps Ethernet for >10 GPUs
```

---

## Failure Modes & Recovery

### Mode 1: One Worker Dies During Training

```
What happens:
  - torchrun detects missing rank within 30 seconds
  - All processes exit cleanly
  - Training stops

Recovery:
  1. Check checkpoint (saved every epoch)
  2. Fix worker issue (reboot, check logs)
  3. Restart all 3 machines from last checkpoint
  4. Resume training from saved epoch
```

### Mode 2: Network Interruption

```
What happens:
  - NCCL timeout after 30 seconds
  - All processes hang or error
  - GPU memory not freed

Recovery:
  1. Kill all processes: pkill -f torchrun
  2. Free GPU memory: nvidia-smi --query-gpu=memory.free --format=csv
  3. Check network: ping all machines
  4. Restart training
```

### Mode 3: Memory Overflow on Worker

```
What happens:
  - CUDA OOM on slowest GPU (8 GB)
  - Worker process exits
  - Other GPUs hang waiting for sync

Recovery:
  1. Reduce batch size: --per-device-train-batch-size=1
  2. Reduce sequence length: max_seq_length=1024
  3. Increase gradient accumulation: --gradient_accumulation_steps=16
  4. Restart training
```

---

## Optimization Checklist

Before launching training:

### Infrastructure (✓ Do once)
- [ ] All 3 machines have static IP
- [ ] SSH passwordless auth working
- [ ] Firewall allows port 29500
- [ ] Network latency < 10ms: `ping -c 10 192.168.1.101`

### Environment (✓ Verify with test script)
- [ ] PyTorch version identical: `python -c "import torch; print(torch.__version__)"`
- [ ] CUDA version identical: `nvcc --version`
- [ ] Python version identical: `python --version`
- [ ] Unsloth installed: `python -c "from unsloth import FastLanguageModel"`
- [ ] Model loads: Run `train_single_test.py`

### Data (✓ Before large training)
- [ ] Dataset fits in free disk: `df -h`
- [ ] Dataset loads correctly: Test with small split
- [ ] Tokenization is fast: Profile on single GPU

### Training Config (✓ Adjust for your GPUs)
- [ ] Per-GPU batch size: 1 (for Unsloth DDP)
- [ ] Gradient accumulation: 8-12 (for effective batch 8-12)
- [ ] Learning rate: 2e-4 (standard for LoRA)
- [ ] Max steps or epochs defined
- [ ] Checkpoint dir exists and writable

### Monitoring (✓ Setup before launch)
- [ ] TensorBoard logdir created
- [ ] Terminal for logs from each rank
- [ ] `nvidia-smi` monitoring command ready
- [ ] Network monitoring tool ready: `iftop`

---

## Key Configuration Parameters

### For Your Llama-7B Training:

```python
# Batch Configuration
per_device_train_batch_size = 1  # Per GPU (required for Unsloth)
gradient_accumulation_steps = 8  # Effective batch = 1 * 3 GPUs * 8 = 24

# Learning Rate
learning_rate = 2e-4            # Standard for LoRA
warmup_steps = 500              # Warmup schedule
max_steps = -1                  # Use epochs instead
num_train_epochs = 3            # 3 full passes through data

# Model Config
max_seq_length = 2048           # Sequence length (reduce to 1024 if OOM)
load_in_4bit = True             # 4-bit quantization (saves VRAM)

# LoRA Config
lora_r = 16                     # LoRA rank
lora_alpha = 32                 # LoRA alpha
lora_dropout = 0.05             # LoRA dropout
target_modules = ["q_proj", "v_proj"]  # Which layers to adapt

# Training Config
gradient_checkpointing = True   # Unsloth handles this
optim = "paged_adamw_8bit"     # Memory efficient optimizer
```

---

## Troubleshooting Matrix

| Symptom | Cause | Solution |
|---------|-------|----------|
| "connect() call failed" | Network issue | Check firewall, ping all machines |
| "CUDA out of memory" | Model too large | Reduce batch size, enable checkpointing |
| Hangs after "initialized" | Rank mismatch | Check node_rank, ensure all processes start |
| One GPU slower than others | Expected | Normal with heterogeneous hardware, DDP waits for slowest |
| High network overhead | Network congestion | Reduce logging frequency, batch network operations |
| Gradient NaN | Learning rate too high | Reduce learning_rate or enable gradient clipping |
| Training diverging | Unstable training | Reduce learning_rate, check data quality |

---

## Next: Inference After Training

After training completes, save and use your model:

```python
# Save LoRA weights
model.save_pretrained("./lora-weights")

# Inference: Load base model + LoRA
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-2-7b-hf",
    adapter_name="./lora-weights",  # Your fine-tuned LoRA
)

# Generate
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

## Architecture Summary

✓ **Your Setup: Distributed Data Parallel (DDP)**
- 3 separate GPUs (heterogeneous)
- Replicated model on each GPU
- Batch data sharded across GPUs
- Gradient synchronization via NCCL

✓ **Why DDP for you:**
- Model fits on each GPU
- Simple to implement (1 line: `DDP(model)`)
- Near-linear scaling
- Lower network overhead

✗ **Not needed yet:**
- FSDP (would use if model > 24 GB)
- Tensor parallelism (would use if model > 24 GB)
- Pipeline parallelism (would use if model > 24 GB)

**Upgrade path:** If you get Llama-70B or larger, then revisit FSDP.
