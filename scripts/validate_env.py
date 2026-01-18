import torch
import sys

print("🔍 Environment Validation")
print("=" * 50)

# PyTorch
print(f"✓ PyTorch: {torch.__version__}")

# CUDA
if torch.cuda.is_available():
    print(f"✓ CUDA Available: True")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  - GPU {i}: {props.name} ({props.total_memory/1e9:.1f}GB)")
else:
    print("⚠️ CUDA not available, using CPU")

# Dependencies
deps = ["mediapipe", "ultralytics", "fastapi", "numpy", "pandas"]
for dep in deps:
    try:
        __import__(dep)
        print(f"✓ {dep}")
    except ImportError:
        print(f"✗ {dep} MISSING")
        sys.exit(1)

print("=" * 50)
print("✅ All validations passed!")
