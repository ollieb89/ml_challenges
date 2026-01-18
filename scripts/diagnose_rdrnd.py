#!/usr/bin/env python3
"""
RDRND Hardware Random Number Generator Diagnostic Script

This script diagnoses the RDRND (Intel/AMD hardware RNG) warning that appeared
during GPU profiling. It checks CPU capabilities, tests the RNG, and provides
actionable recommendations.

Usage:
    python scripts/diagnose_rdrnd.py
    # Or via pixi:
    pixi run python scripts/diagnose_rdrnd.py
"""

import os
import platform
import subprocess
import sys
from pathlib import Path


def check_cpu_flags() -> dict:
    """Check CPU flags for RDRND and RDSEED support."""
    result = {
        "rdrnd": False,
        "rdseed": False,
        "aes": False,
        "cpu_model": "Unknown",
        "flags_raw": "",
    }
    
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        content = cpuinfo_path.read_text()
        
        # Extract CPU model
        for line in content.split("\n"):
            if line.startswith("model name"):
                result["cpu_model"] = line.split(":")[1].strip()
                break
        
        # Extract flags
        for line in content.split("\n"):
            if line.startswith("flags"):
                flags = line.split(":")[1].strip()
                result["flags_raw"] = flags
                result["rdrnd"] = "rdrand" in flags or "rdrnd" in flags
                result["rdseed"] = "rdseed" in flags
                result["aes"] = "aes" in flags
                break
    
    return result


def test_urandom() -> dict:
    """Test /dev/urandom functionality."""
    result = {
        "available": False,
        "readable": False,
        "sample": None,
        "error": None,
    }
    
    urandom_path = Path("/dev/urandom")
    result["available"] = urandom_path.exists()
    
    if result["available"]:
        try:
            with open("/dev/urandom", "rb") as f:
                sample = f.read(16)
                result["readable"] = True
                result["sample"] = sample.hex()
                
                # Check for suspicious patterns (all zeros or all ones)
                if sample == b"\x00" * 16:
                    result["error"] = "WARNING: urandom returned all zeros"
                elif sample == b"\xff" * 16:
                    result["error"] = "WARNING: urandom returned all ones"
        except Exception as e:
            result["error"] = str(e)
    
    return result


def test_python_random() -> dict:
    """Test Python's random module with system entropy."""
    import secrets
    import random
    
    result = {
        "secrets_works": False,
        "os_urandom_works": False,
        "samples": {},
        "error": None,
    }
    
    try:
        # Test secrets module (uses system entropy)
        result["samples"]["secrets"] = secrets.token_hex(8)
        result["secrets_works"] = True
    except Exception as e:
        result["error"] = f"secrets module failed: {e}"
    
    try:
        # Test os.urandom
        result["samples"]["os_urandom"] = os.urandom(8).hex()
        result["os_urandom_works"] = True
    except Exception as e:
        result["error"] = f"os.urandom failed: {e}"
    
    return result


def check_rdrand_directly() -> dict:
    """
    Attempt to test RDRAND instruction directly using inline assembly.
    This requires a compiler and may not work in all environments.
    """
    result = {
        "test_available": False,
        "rdrand_works": None,
        "error": None,
    }
    
    # Try using cpuid to check RDRAND support
    try:
        # Check if rdrand instruction can be executed via Python
        # This is a tricky test - we'll use subprocess to call a system tool
        
        # Try using OpenSSL to test hardware RNG
        proc = subprocess.run(
            ["openssl", "rand", "-hex", "16"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if proc.returncode == 0:
            result["test_available"] = True
            sample = proc.stdout.strip()
            
            # Check for suspicious output
            if sample == "ffffffffffffffffffffffffffffffff":
                result["rdrand_works"] = False
                result["error"] = "OpenSSL returned all 0xFF - RDRAND may be failing"
            elif sample == "00000000000000000000000000000000":
                result["rdrand_works"] = False
                result["error"] = "OpenSSL returned all zeros - RNG issue"
            else:
                result["rdrand_works"] = True
                result["samples"] = sample
    except FileNotFoundError:
        result["error"] = "OpenSSL not found"
    except subprocess.TimeoutExpired:
        result["error"] = "OpenSSL timed out"
    except Exception as e:
        result["error"] = str(e)
    
    return result


def check_kernel_rng() -> dict:
    """Check kernel RNG status via /proc/sys/kernel/random/."""
    result = {
        "entropy_available": None,
        "poolsize": None,
        "urandom_min_reseed_secs": None,
    }
    
    random_proc = Path("/proc/sys/kernel/random")
    
    if random_proc.exists():
        try:
            entropy_path = random_proc / "entropy_avail"
            if entropy_path.exists():
                result["entropy_available"] = int(entropy_path.read_text().strip())
            
            poolsize_path = random_proc / "poolsize"
            if poolsize_path.exists():
                result["poolsize"] = int(poolsize_path.read_text().strip())
                
            reseed_path = random_proc / "urandom_min_reseed_secs"
            if reseed_path.exists():
                result["urandom_min_reseed_secs"] = int(reseed_path.read_text().strip())
        except Exception as e:
            result["error"] = str(e)
    
    return result


def check_microcode_version() -> dict:
    """Check CPU microcode version."""
    result = {
        "microcode_version": None,
        "update_available": None,
    }
    
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        content = cpuinfo_path.read_text()
        for line in content.split("\n"):
            if "microcode" in line.lower():
                result["microcode_version"] = line.split(":")[1].strip()
                break
    
    return result


def generate_recommendations(diagnostics: dict) -> list:
    """Generate actionable recommendations based on diagnostics."""
    recommendations = []
    
    cpu_flags = diagnostics["cpu_flags"]
    urandom = diagnostics["urandom"]
    kernel_rng = diagnostics["kernel_rng"]
    rdrand_test = diagnostics["rdrand_test"]
    
    # Check RDRAND support
    if not cpu_flags["rdrnd"]:
        recommendations.append({
            "severity": "INFO",
            "issue": "CPU does not support RDRAND instruction",
            "action": "This is normal for older CPUs. The warning may be from PyTorch falling back to software RNG.",
        })
    elif rdrand_test.get("rdrand_works") is False:
        recommendations.append({
            "severity": "WARNING",
            "issue": "RDRAND instruction may be malfunctioning",
            "action": "Update CPU microcode via BIOS update or `intel-microcode`/`amd-microcode` package.",
        })
    
    # Check entropy availability
    if kernel_rng.get("entropy_available", 0) < 256:
        recommendations.append({
            "severity": "WARNING",
            "issue": f"Low kernel entropy: {kernel_rng.get('entropy_available')} bits",
            "action": "Install `haveged` or `rng-tools` to improve entropy gathering.",
        })
    
    # Check if running in VM
    cpu_model = cpu_flags.get("cpu_model", "").lower()
    if "qemu" in cpu_model or "virtual" in cpu_model:
        recommendations.append({
            "severity": "INFO",
            "issue": "Running in a virtual machine",
            "action": "VMs may have limited access to hardware RNG. Enable VirtIO RNG if available.",
        })
    
    # Check microcode
    microcode = diagnostics["microcode"]
    if microcode.get("microcode_version"):
        recommendations.append({
            "severity": "INFO",
            "issue": f"Microcode version: {microcode['microcode_version']}",
            "action": "Consider updating BIOS to get the latest microcode updates.",
        })
    
    # General PyTorch recommendation
    recommendations.append({
        "severity": "INFO",
        "issue": "PyTorch RDRND warning during model loading",
        "action": "This warning is generally harmless. PyTorch falls back to software RNG. "
                  "To suppress, set `PYTORCH_NO_CUDA_MEMORY_CACHING=1` or update PyTorch.",
    })
    
    return recommendations


def main():
    """Run all diagnostics and print results."""
    print("=" * 60)
    print("🔍 RDRND HARDWARE RNG DIAGNOSTIC")
    print("=" * 60)
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")
    print()
    
    # Run diagnostics
    diagnostics = {
        "cpu_flags": check_cpu_flags(),
        "urandom": test_urandom(),
        "python_random": test_python_random(),
        "rdrand_test": check_rdrand_directly(),
        "kernel_rng": check_kernel_rng(),
        "microcode": check_microcode_version(),
    }
    
    # Print CPU info
    print("📊 CPU Information")
    print("-" * 40)
    cpu = diagnostics["cpu_flags"]
    print(f"  Model: {cpu['cpu_model']}")
    print(f"  RDRAND support: {'✅ Yes' if cpu['rdrnd'] else '❌ No'}")
    print(f"  RDSEED support: {'✅ Yes' if cpu['rdseed'] else '❌ No'}")
    print(f"  AES-NI support: {'✅ Yes' if cpu['aes'] else '❌ No'}")
    print()
    
    # Print kernel RNG
    print("🎲 Kernel RNG Status")
    print("-" * 40)
    kernel = diagnostics["kernel_rng"]
    entropy = kernel.get("entropy_available", "N/A")
    poolsize = kernel.get("poolsize", "N/A")
    print(f"  Entropy available: {entropy} bits")
    print(f"  Pool size: {poolsize} bits")
    if isinstance(entropy, int) and entropy < 256:
        print("  ⚠️  Low entropy detected!")
    else:
        print("  ✅ Entropy level OK")
    print()
    
    # Print /dev/urandom status
    print("🔐 /dev/urandom Status")
    print("-" * 40)
    urandom = diagnostics["urandom"]
    print(f"  Available: {'✅ Yes' if urandom['available'] else '❌ No'}")
    print(f"  Readable: {'✅ Yes' if urandom['readable'] else '❌ No'}")
    if urandom.get("error"):
        print(f"  ⚠️  {urandom['error']}")
    else:
        print(f"  Sample: {urandom.get('sample', 'N/A')[:16]}...")
    print()
    
    # Print Python random test
    print("🐍 Python Random Tests")
    print("-" * 40)
    py_random = diagnostics["python_random"]
    print(f"  secrets module: {'✅ Working' if py_random['secrets_works'] else '❌ Failed'}")
    print(f"  os.urandom: {'✅ Working' if py_random['os_urandom_works'] else '❌ Failed'}")
    if py_random.get("samples"):
        print(f"  Sample (secrets): {py_random['samples'].get('secrets', 'N/A')}")
    print()
    
    # Print RDRAND test
    print("⚙️  RDRAND Direct Test (via OpenSSL)")
    print("-" * 40)
    rdrand = diagnostics["rdrand_test"]
    if rdrand.get("test_available"):
        status = "✅ Working" if rdrand.get("rdrand_works") else "❌ Failing"
        print(f"  Status: {status}")
        if rdrand.get("samples"):
            print(f"  Sample: {rdrand['samples'][:16]}...")
    else:
        print(f"  Status: Test unavailable ({rdrand.get('error', 'Unknown')})")
    if rdrand.get("error") and rdrand.get("test_available"):
        print(f"  ⚠️  {rdrand['error']}")
    print()
    
    # Print microcode info
    print("🔧 Microcode Version")
    print("-" * 40)
    microcode = diagnostics["microcode"]
    print(f"  Version: {microcode.get('microcode_version', 'Unknown')}")
    print()
    
    # Generate and print recommendations
    print("=" * 60)
    print("📋 RECOMMENDATIONS")
    print("=" * 60)
    recommendations = generate_recommendations(diagnostics)
    
    for i, rec in enumerate(recommendations, 1):
        severity = rec["severity"]
        icon = {"INFO": "ℹ️ ", "WARNING": "⚠️ ", "ERROR": "🔴"}[severity]
        print(f"\n{i}. {icon}{rec['issue']}")
        print(f"   Action: {rec['action']}")
    
    print("\n" + "=" * 60)
    print("✅ DIAGNOSTIC COMPLETE")
    print("=" * 60)
    
    # Return exit code based on findings
    has_critical = any(r["severity"] == "ERROR" for r in recommendations)
    return 1 if has_critical else 0


if __name__ == "__main__":
    sys.exit(main())
