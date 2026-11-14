"""
NCU (Nsight Compute) Hardware Profiler for DeltaNet v2
Directly measures silicon-level performance counters on NVIDIA GPUs (e.g., Tesla T4 on Google Colab):
- Hardware Floating-Point Operations (HFMA / FFMA SASS instructions)
- Hardware Memory Traffic (DRAM read/write bytes & throughput)
- Hardware GPU Execution Duration

Compares exact hardware-measured metrics directly against theoretical model values.

Usage from Bash / Colab:
    # 1. Profile Autoregressive Inference (L = 1):
    python /content/Deltanet/tests/profile_ncu.py --run-ncu --mode inference --batch-size 64

    # 2. Profile Training Step (Forward + Backward, L = 2048):
    python /content/Deltanet/tests/profile_ncu.py --run-ncu --mode training --batch-size 4 --seq-len 2048
"""

import argparse
import csv
import io
import os
import subprocess
import sys
from typing import Dict, Optional

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from deltanet_v2.model import DeltaBlock
from tests.utils import (
    compute_deltanet_inference_flops,
    compute_deltanet_inference_memory_bytes,
    compute_deltanet_training_flops,
    format_markdown_table,
    get_device,
    print_hardware_info,
    print_section,
)


def run_target_workload(
    mode: str = "inference",
    d_model: int = 768,
    num_q_heads: int = 8,
    num_kv_heads: int = 8,
    d_k: int = 64,
    d_v: int = 64,
    chunk_size: int = 64,
    batch_size: int = 64,
    seq_len: int = 1,
    warmup: int = 3,
    iters: int = 5,
    device: Optional[torch.device] = None,
):
    """Executes target workload scoped strictly within cudaProfilerStart/Stop for NCU."""
    if device is None:
        device = get_device()
    if device.type != "cuda":
        print("[!] Error: NCU profiling requires an active CUDA GPU.")
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

    if mode == "inference":
        block.eval()
        x = torch.randn(batch_size, 1, d_model, device=device, dtype=torch.float16)
        past_state = torch.zeros(batch_size, num_kv_heads, d_v, d_k, device=device, dtype=torch.float16)

        # Warmup (untimed, JIT initialization)
        for _ in range(warmup):
            with torch.no_grad():
                _ = block(x, past_states=past_state, use_kernel=True)
        torch.cuda.synchronize(device)

        # Profiled iterations
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(iters):
            with torch.no_grad():
                _ = block(x, past_states=past_state, use_kernel=True)
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStop()

    else:
        # Training pass (Forward + Backward)
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

        # Profiled iterations
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(iters):
            block.zero_grad(set_to_none=True)
            out, _ = block(x_packed, cu_seqlens=cu_seqlens, use_kernel=True)
            loss = out.sum()
            loss.backward()
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStop()


def run_with_ncu(args):
    """Invokes ncu CLI subprocess, parses CSV hardware counters, and displays comparison."""
    script_path = os.path.abspath(__file__)
    metrics = [
        "sm__sass_thread_inst_executed_op_hfma_pred_on.sum",
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
        "gpu__time_duration.sum",
        "dram__bytes.sum",
        "dram__bytes.sum.per_second",
    ]

    cmd = [
        "ncu",
        "--csv",
        "--metrics", ",".join(metrics),
        sys.executable,
        script_path,
        "--mode", str(args.mode),
        "--dim", str(args.dim),
        "--q-heads", str(args.q_heads),
        "--kv-heads", str(args.kv_heads),
        "--d-k", str(args.d_k),
        "--d-v", str(args.d_v),
        "--chunk-size", str(args.chunk_size),
        "--batch-size", str(args.batch_size),
        "--seq-len", str(args.seq_len if args.mode == "training" else 1),
        "--warmup", str(args.warmup),
        "--iters", str(args.iters),
    ]

    print_section(f"RUNNING NVIDIA NSIGHT COMPUTE (NCU) PROFILER: {args.mode.upper()}")
    print(f"Command:\n{' '.join(cmd)}\n", flush=True)

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if proc.returncode != 0:
        print("[!] NCU Profiler Error / Warning:")
        if "ERR_NVGPUCTRPERM" in proc.stderr or "ERR_NVGPUCTRPERM" in proc.stdout:
            print("    Permission denied accessing GPU performance counters.")
            print("    Run this in Colab to grant profiling permissions:")
            print("    !sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0\n")
        print(proc.stderr if proc.stderr else proc.stdout)
        return

    # Parse CSV output from NCU
    csv_lines = []
    capture = False
    for line in proc.stdout.splitlines():
        if "ID" in line and "Metric Name" in line:
            capture = True
        if capture:
            csv_lines.append(line)

    metric_sums = {
        "hfma": 0.0,
        "ffma": 0.0,
        "time_ns": 0.0,
        "dram_bytes": 0.0,
        "dram_bw_list": [],
    }

    if csv_lines:
        reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))
        for row in reader:
            m_name = row.get("Metric Name", "")
            m_val_str = row.get("Metric Value", "0").replace(",", "")
            try:
                val = float(m_val_str)
            except ValueError:
                val = 0.0

            if "op_hfma" in m_name:
                metric_sums["hfma"] += val
            elif "op_ffma" in m_name:
                metric_sums["ffma"] += val
            elif "time_duration" in m_name:
                metric_sums["time_ns"] += val
            elif m_name == "dram__bytes.sum":
                metric_sums["dram_bytes"] += val
            elif "per_second" in m_name:
                metric_sums["dram_bw_list"].append(val)

    iters = max(args.iters, 1)

    # 1 HFMA = 2 FLOPs (Fused Multiply-Add FP16), 1 FFMA = 2 FLOPs (FP32)
    hw_flops_total = 2.0 * (metric_sums["hfma"] + metric_sums["ffma"])
    hw_flops_per_step = hw_flops_total / iters
    hw_gflops = hw_flops_per_step / 1e9

    hw_lat_ms = (metric_sums["time_ns"] / iters) / 1e6
    hw_tflops = (hw_gflops / hw_lat_ms) if hw_lat_ms > 0 else 0.0

    avg_dram_bw_gbs = (
        (sum(metric_sums["dram_bw_list"]) / len(metric_sums["dram_bw_list"])) / 1e9
        if metric_sums["dram_bw_list"]
        else (metric_sums["dram_bytes"] / (metric_sums["time_ns"] if metric_sums["time_ns"] > 0 else 1.0))
    )

    # Theoretical values for comparison
    if args.mode == "inference":
        theo_flops = compute_deltanet_inference_flops(
            batch_size=args.batch_size,
            d_model=args.dim,
            num_q_heads=args.q_heads,
            num_kv_heads=args.kv_heads,
            d_k=args.d_k,
            d_v=args.d_v,
        )
        theo_bytes = compute_deltanet_inference_memory_bytes(
            batch_size=args.batch_size,
            d_model=args.dim,
            num_q_heads=args.q_heads,
            num_kv_heads=args.kv_heads,
            d_k=args.d_k,
            d_v=args.d_v,
            dtype_bytes=2,
            state_bytes=2,
        )
    else:
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
        theo_bytes = 0.0

    theo_gflops = theo_flops / 1e9
    theo_tflops = (theo_gflops / hw_lat_ms) if hw_lat_ms > 0 else 0.0
    theo_mem_bw_gbs = (theo_bytes / (hw_lat_ms * 1e6)) if (hw_lat_ms > 0 and theo_bytes > 0) else 0.0

    # Print summary table
    headers = [
        "Metric Parameter",
        "NCU Hardware Measured (Silicon)",
        "Theoretical Algorithmic Model",
        "Hardware / Theoretical Ratio",
    ]
    rows = [
        ["Mode", args.mode.upper(), args.mode.upper(), "-"],
        ["Batch Size (B)", args.batch_size, args.batch_size, "-"],
        ["Sequence Length (L)", args.seq_len if args.mode == "training" else 1, args.seq_len if args.mode == "training" else 1, "-"],
        ["Kernel Latency (ms)", f"{hw_lat_ms:.3f}", f"{hw_lat_ms:.3f}", "1.00x"],
        ["Operation Count (GFLOPs)", f"{hw_gflops:.4f}", f"{theo_gflops:.4f}", f"{(hw_gflops / theo_gflops):.2f}x" if theo_gflops > 0 else "-"],
        ["Compute Throughput (TFLOP/s)", f"{hw_tflops:.2f}", f"{theo_tflops:.2f}", f"{(hw_tflops / theo_tflops):.2f}x" if theo_tflops > 0 else "-"],
        ["DRAM Memory Bandwidth (GB/s)", f"{avg_dram_bw_gbs:.1f}", f"{theo_mem_bw_gbs:.1f}" if theo_mem_bw_gbs > 0 else "N/A", "-"],
    ]

    print("=" * 76)
    print(f"NSIGHT COMPUTE (NCU) HARDWARE PROFILER RESULTS: {args.mode.upper()}")
    print("=" * 76)
    print(format_markdown_table(headers, rows))
    print("=" * 76)
    print(f"  • FP16 SASS FMA Instructions (hfma) : {metric_sums['hfma']:,.0f}")
    print(f"  • FP32 SASS FMA Instructions (ffma) : {metric_sums['ffma']:,.0f}")
    print(f"  • Total DRAM Traffic Transferred     : {metric_sums['dram_bytes'] / (1024**2):.2f} MB")
    print(f"  • Total GPU Execution Time Scoped    : {metric_sums['time_ns'] / 1e6:.3f} ms ({iters} iterations)")
    print("=" * 76 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="NCU Hardware Profiler for DeltaNet v2")
    parser.add_argument("--mode", type=str, default="inference", choices=["inference", "training"], help="Profile inference or training step")
    parser.add_argument("--dim", type=int, default=768, help="Hidden dimension (d_model)")
    parser.add_argument("--q-heads", type=int, default=8, help="Query heads")
    parser.add_argument("--kv-heads", type=int, default=8, help="Key/Value heads")
    parser.add_argument("--d-k", type=int, default=64, help="Key head dimension")
    parser.add_argument("--d-v", type=int, default=64, help="Value head dimension")
    parser.add_argument("--chunk-size", type=int, default=64, help="Chunk size C")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size B")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length L (for training)")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=5, help="Profiled iterations")
    parser.add_argument("--run-ncu", action="store_true", help="Launch NCU profiler automatically")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.run_ncu:
        run_with_ncu(args)
    else:
        run_target_workload(
            mode=args.mode,
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
