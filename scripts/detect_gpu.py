#!/usr/bin/env python3
"""GPU Detection and Validation Script for AI/ML Pipeline."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_cuda_availability():
    """Check if CUDA is available via PyTorch."""
    try:
        import torch
        
        print("\n" + "=" * 60)
        print("🔍 GPU DETECTION REPORT")
        print("=" * 60)
        
        print(f"\n📦 PyTorch Version: {torch.__version__}")
        print(f"🔧 CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA Version: {torch.version.cuda}")
            print(f"🎮 GPU Count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\n🎯 GPU {i}: {props.name}")
                print(f"   💾 Total Memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"   🔢 Compute Capability: {props.major}.{props.minor}")
                print(f"   🧮 Multiprocessors: {props.multi_processor_count}")
            
            # Test GPU computation
            print("\n🧪 Testing GPU computation...")
            try:
                x = torch.randn(1000, 1000, device='cuda')
                y = torch.matmul(x, x)
                print("✅ GPU computation test passed!")
            except Exception as e:
                print(f"❌ GPU computation test failed: {e}")
        else:
            print("\n⚠️  No CUDA-capable GPU detected")
            print("💡 Running in CPU mode")
            
            # Check if CUDA was built into PyTorch
            print(f"\n🔧 CUDA Built: {torch.backends.cuda.is_built()}")
            print(f"🔧 cuDNN Available: {torch.backends.cudnn.is_available()}")
        
        print("\n" + "=" * 60)
        return torch.cuda.is_available()
        
    except ImportError as e:
        print(f"❌ Error: PyTorch not installed - {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def check_other_frameworks():
    """Check GPU availability in other frameworks."""
    print("\n📚 Checking other frameworks...")
    
    # TensorFlow (if installed)
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        print(f"🔷 TensorFlow GPUs: {len(gpus)}")
        for gpu in gpus:
            print(f"   - {gpu.name}")
    except ImportError:
        print("⚪ TensorFlow not installed (optional)")
    except Exception as e:
        print(f"⚠️  TensorFlow GPU check failed: {e}")
    
    # Check NVIDIA driver
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            print("\n✅ NVIDIA Driver detected:")
            print(result.stdout)
        else:
            print("\n⚠️  nvidia-smi command failed")
    except FileNotFoundError:
        print("\n⚠️  nvidia-smi not found (NVIDIA drivers may not be installed)")
    except Exception as e:
        print(f"\n⚠️  Could not check NVIDIA driver: {e}")


def main():
    """Main entry point."""
    print("\n🚀 Starting GPU Detection...")
    
    has_gpu = check_cuda_availability()
    check_other_frameworks()
    
    print("\n" + "=" * 60)
    if has_gpu:
        print("✅ GPU ENVIRONMENT READY")
        print("💡 Use: pixi run -e cuda <command>")
    else:
        print("ℹ️  CPU ENVIRONMENT ACTIVE")
        print("💡 Use: pixi run -e cpu <command>")
    print("=" * 60 + "\n")
    
    return 0 if has_gpu else 1


if __name__ == "__main__":
    sys.exit(main())
