"""
Diagnostic script to check why NCCL P2P and IB need to be disabled.

Run this to understand your GPU topology and network setup.
"""

import os
import torch
import subprocess
import sys


def check_gpu_topology():
    """Check GPU topology and interconnects."""
    print("=" * 60)
    print("GPU Topology Check")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    
    # Check GPU model
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
    
    # Check NVLink (if available)
    print("\n" + "=" * 60)
    print("NVLink Check")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("nvidia-smi topo failed. Trying nvidia-smi nvlink...")
            result = subprocess.run(
                ["nvidia-smi", "nvlink", "--status"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print(result.stdout)
            else:
                print("Could not check NVLink status")
    except Exception as e:
        print(f"Error checking topology: {e}")
        print("\nNote: A40 GPUs typically don't have NVLink between them.")
        print("They communicate via PCIe, which is why P2P might fail.")


def check_network_interfaces():
    """Check network interfaces for InfiniBand."""
    print("\n" + "=" * 60)
    print("Network Interfaces Check")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            ["ip", "addr", "show"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'inet' in line or 'state' in line or 'ib' in line.lower():
                    print(line)
        else:
            print("Could not list network interfaces")
    except Exception as e:
        print(f"Error checking interfaces: {e}")
    
    # Check for InfiniBand specifically
    print("\nChecking for InfiniBand devices...")
    try:
        result = subprocess.run(
            ["ibstat", "-l"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            print("InfiniBand devices found:")
            print(result.stdout)
        else:
            print("No InfiniBand devices found (or ibstat not available)")
            print("This is normal for cloud instances without IB.")
    except FileNotFoundError:
        print("ibstat not found - InfiniBand likely not configured")
    except Exception as e:
        print(f"Error checking IB: {e}")


def check_nccl_env():
    """Check NCCL environment variables."""
    print("\n" + "=" * 60)
    print("NCCL Environment Variables")
    print("=" * 60)
    
    nccl_vars = [k for k in os.environ.keys() if 'NCCL' in k.upper()]
    
    if nccl_vars:
        for var in sorted(nccl_vars):
            print(f"{var} = {os.environ[var]}")
    else:
        print("No NCCL environment variables set")
    
    print("\nRecommended settings for A40s without NVLink/IB:")
    print("  NCCL_P2P_DISABLE=1  # Disable P2P (no NVLink)")
    print("  NCCL_IB_DISABLE=1    # Disable InfiniBand (not available)")
    print("  NCCL_SOCKET_IFNAME=eth0  # Use Ethernet interface")


def check_pytorch_nccl():
    """Check PyTorch NCCL backend."""
    print("\n" + "=" * 60)
    print("PyTorch NCCL Backend Check")
    print("=" * 60)
    
    if not torch.distributed.is_nccl_available():
        print("WARNING: NCCL not available in PyTorch!")
        return
    
    print("NCCL is available in PyTorch")
    print(f"PyTorch version: {torch.__version__}")
    
    # Try to get NCCL version
    try:
        import torch.distributed as dist
        if hasattr(dist, 'get_nccl_version'):
            version = dist.get_nccl_version()
            print(f"NCCL version: {version}")
    except:
        pass


def test_simple_comm():
    """Test simple NCCL communication."""
    print("\n" + "=" * 60)
    print("Simple NCCL Communication Test")
    print("=" * 60)
    print("Run this with: torchrun --nproc_per_node=2 diagnose_nccl.py")
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        try:
            import torch.distributed as dist
            
            dist.init_process_group(backend="nccl")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            
            print(f"Rank {rank}/{world_size} initialized")
            
            # Simple all-reduce test
            tensor = torch.ones(1, device=f"cuda:{rank}")
            print(f"Rank {rank}: Before all_reduce: {tensor.item()}")
            
            dist.all_reduce(tensor)
            
            print(f"Rank {rank}: After all_reduce: {tensor.item()} (should be {world_size})")
            
            dist.destroy_process_group()
            print(f"Rank {rank}: Test passed!")
            
        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Run all diagnostics."""
    print("NCCL Diagnostic Tool")
    print("=" * 60)
    print("This script helps diagnose why P2P and IB need to be disabled.\n")
    
    check_gpu_topology()
    check_network_interfaces()
    check_nccl_env()
    check_pytorch_nccl()
    test_simple_comm()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
For A40 GPUs on RunPod (typical cloud setup):
- A40s don't have NVLink between GPUs → P2P will fail
- No InfiniBand → IB will fail
- GPUs communicate via PCIe → slower but works

Solution: Keep using:
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1

This forces NCCL to use:
- Shared memory for same-node communication
- Ethernet for multi-node (if applicable)

Performance impact: ~10-20% slower than NVLink, but stable.
""")


# if __name__ == "__main__":
#     if len(sys.argv) > 1 and sys.argv[1] == "test":
#         test_simple_comm()
#     else:
#         main()

