"""
NCU (Nsight Compute) Profiler Script for DeltaNet v2 Training Pass
Measures hardware floating-point operations (FLOPs), Tensor Core utilization,
and exact hardware TFLOP/s on NVIDIA GPUs (such as Tesla T4 on Google Colab).

Usage from Bash / Colab:
    ncu --csv --metrics sm__sass_thread_inst_executed_op_hfma_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,gpu__time_duration.sum \\
        python /content/Deltanet/tests/profile_ncu.py --batch-size 4 --seq-len 2048 --iters 5

Or automated measurement:
    python /content/Deltanet/tests/profile_ncu.py --run-ncu --batch-size 4 --seq-len 2048
"""

import argparse
import os
import subprocess
import sys
from typing import Optional

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from deltanet_v2.model import DeltaBlock
from tests.utils import (
    compute_deltanet_training_flops,
    get_device,
    print_hardware_info,
    print_section,
)


def run_workload(
    d_model: int = 768,
    num_q_heads: int = 8,
    num_kv_heads: int = 8,
    d_k: int = 64,
    d_v: int = 64,
    chunk_size: int = 64,
    batch_size: int = 4,
    seq_len: int = 2048,
    warmup: int = 3,
    iters: int = 5,
    device: Optional[torch.device] = None,
):
    if device is None:
        device = get_device()
    if device.type != "cuda":
        print("[!] Error: NCU profiling requires a CUDA-capable GPU.")
        sys.exit(1)

    torch.manual_seed(42)
    block = DeltaBlock(
        d_model=d_model,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        d_k=d_k,
        d_v=d_v,
        chunk_size=chunk_size,
        hidden_dim=None,
    ).to(device=device, dtype=torch.float16)
    block.train()

    total_tokens = batch_size * seq_len
    x_batched = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16, requires_grad=True)
    x_packed = x_batched.detach().reshape(total_tokens, d_model).clone().requires_grad_(True)
    cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=device)

    # Warmup
    for _ in range(warmup):
        block.zero_grad(set_to_none=True)
        out, _ = block(x_packed, cu_seqlens=cu_seqlens, use_kernel=True)
        loss = out.sum()
        loss.backward()
    torch.cuda.synchronize(device)

    # Profile range
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iters):
        block.zero_grad(set_to_none=True)
        out, _ = block(x_packed, cu_seqlens=cu_seqlens, use_kernel=True)
        loss = out.sum()
        loss.backward()
    torch.cuda.synchronize(device)
    torch.cuda.cudart().cudaProfilerStop()


def run_with_ncu(args):
    """Run NCU profiler on this script and parse hardware FLOP counters."""
    script_path = os.path.abspath(__file__)
    metrics = [
        "sm__sass_thread_inst_executed_op_hfma_pred_on.sum",
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
        "gpu__time_duration.sum",
    ]
    cmd = [
        "ncu",
        "--csv",
        "--metrics", ",".join(metrics),
        sys.executable,
        script_path,
        "--dim", str(args.dim),
        "--q-heads", str(args.q_heads),
        "--kv-heads", str(args.kv_heads),
        "--d-k", str(args.d_k),
        "--d-v", str(args.d_v),
        "--chunk-size", str(args.chunk_size),
        "--batch-size", str(args.batch_size),
        "--seq-len", str(args.seq_len),
        "--warmup", str(args.warmup),
        "--iters", str(args.iters),
    ]

    print(f"Executing NCU Profiler Command:\n{' '.join(cmd)}\n")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if proc.returncode != 0:
        print("[!] NCU Profiler Error:\n", proc.stderr)
        return

    # Theoretical calculations
    theo_flops = compute_deltanet_training_flops(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.dim,
        num_q_heads=args.q_heads,
        num_kv_heads=args.kv_heads,
        d_k=args.d_k,
        d_v=args.d_v,
        chunk_size=args.chunk_size,
    )
    theo_gflops = theo_flops / 1e9

    print("=" * 72)
    print("NCU HARDWARE PROFILER RESULTS (DeltaNet v2 Training Pass)")
    print("=" * 72)
    print(f"Batch Size (B)       : {args.batch_size}")
    print(f"Sequence Length (L)  : {args.seq_len}")
    print(f"Total Tokens (B * L) : {args.batch_size * args.seq_len:,}")
    print(f"Theoretical GFLOPs   : {theo_gflops:.2f} GFLOPs / step")
    print("-" * 72)
    print("NCU Raw Output Summary:")
    for line in proc.stdout.splitlines():
        if "sm__" in line or "gpu__time" in line:
            print(" ", line)
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="NCU Profiler for DeltaNet v2")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--d-k", type=int, default=64)
    parser.add_argument("--d-v", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--run-ncu", action="store_true", help="Launch NCU profiler automatically")
    args = parser.parse_args()

    if args.run_ncu:
        run_with_ncu(args)
    else:
        run_workload(
            d_model=args.dim,
            num_q_heads=args.q_heads,
            num_kv_heads=args.kv_heads,
            d_k=args.d_k,
            d_v=args.d_v,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            warmup=args.warmup,
            iters=args.iters,
        )


if __name__ == "__main__":
    main()
