"""
Benchmark: Training Pass (Forward + Backward)
Evaluates end-to-end training latency (ms/step), training throughput (tokens/sec), and peak VRAM.
Compares DeltaNet v2 (SM75 CuTe fused kernel) against NVLabs GatedDeltaNet-2, FLA GatedDeltaNet-2, FLA GatedDeltaNet (v1), and PyTorch reference.

Benchmarks strictly the Gated DeltaNet token-mixer block (without MLP / FFW / SWA)
to isolate and fairly evaluate the recurrence kernels.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# Add project root to sys.path so scripts can run standalone from anywhere
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from deltanet_v2.model import DeltaBlock
from tests.utils import (
    check_baseline_availability,
    format_markdown_table,
    get_device,
    measure_performance,
    print_baseline_status,
    print_hardware_info,
    print_section,
    try_import_fla_gdn,
    try_import_fla_gdn2,
    try_import_nvlabs_gdn2,
)


def run_training_benchmark(
    d_model: int = 768,
    num_q_heads: int = 8,
    num_kv_heads: int = 8,
    d_k: int = 64,
    d_v: int = 64,
    chunk_size: int = 64,
    batch_size: int = 4,
    seq_lens: Optional[List[int]] = None,
    num_warmup: int = 10,
    num_iters: int = 30,
    use_short_conv: bool = False,
    dtype: torch.dtype = torch.float16,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """
    Benchmark training step (Forward + Backward) across sequence lengths.
    """
    if device is None:
        device = get_device()
    if seq_lens is None:
        seq_lens = [512, 1024, 2048, 4096]

    print_section("BENCHMARK: TRAINING PASS (FORWARD + BACKWARD)")
    print(f"Config: d_model={d_model}, num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, "
          f"d_k={d_k}, d_v={d_v}, chunk_size={chunk_size}, batch_size={batch_size}, "
          f"use_short_conv={use_short_conv}, dtype={dtype}, device={device}")
    print(f"Sequence lengths to evaluate: {seq_lens}\n")

    # 1. DeltaNet v2 Block (Pure Gated DeltaNet: hidden_dim=None excludes FFW/MLP)
    print("Initializing DeltaNet v2...")
    our_block = DeltaBlock(
        d_model=d_model,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        d_k=d_k,
        d_v=d_v,
        chunk_size=chunk_size,
        hidden_dim=None,
    ).to(device=device, dtype=dtype)
    our_block.train()

    # 2. Check and load Baselines
    nvlabs_gdn2_cls = try_import_nvlabs_gdn2()
    fla_gdn2_cls = try_import_fla_gdn2()
    fla_gdn_cls = try_import_fla_gdn()

    # NVLabs GatedDeltaNet-2 (Official NVIDIA implementation)
    nvlabs_gdn2_block = None
    if nvlabs_gdn2_cls is not None:
        try:
            print("Initializing NVLabs GatedDeltaNet-2 (Official NVIDIA chunkwise parallel)...")
            nvlabs_gdn2_block = nvlabs_gdn2_cls(
                hidden_size=d_model,
                num_heads=num_kv_heads,
                head_dim=d_k,
                expand_v=float(d_v) / float(d_k),
                mode="chunk",
                use_short_conv=use_short_conv,
                layer_idx=0,
            ).to(device=device, dtype=dtype)
            nvlabs_gdn2_block.train()
        except Exception as e:
            print(f"[!] Warning: Could not instantiate NVLabs GatedDeltaNet-2 with mode='chunk' ({e}). Retrying default.")
            try:
                nvlabs_gdn2_block = nvlabs_gdn2_cls(
                    hidden_size=d_model,
                    num_heads=num_kv_heads,
                    head_dim=d_k,
                    expand_v=float(d_v) / float(d_k),
                    use_short_conv=use_short_conv,
                    layer_idx=0,
                ).to(device=device, dtype=dtype)
                nvlabs_gdn2_block.train()
            except Exception as e2:
                print(f"[!] Error instantiating NVLabs GatedDeltaNet-2: {e2}")
                nvlabs_gdn2_block = None

    # FLA GatedDeltaNet-2 (Flash Linear Attention library)
    fla_gdn2_block = None
    if fla_gdn2_cls is not None:
        try:
            print("Initializing FLA GatedDeltaNet-2 (Flash Linear Attention chunkwise parallel)...")
            fla_gdn2_block = fla_gdn2_cls(
                hidden_size=d_model,
                num_heads=num_kv_heads,
                head_dim=d_k,
                expand_v=float(d_v) / float(d_k),
                mode="chunk",
                use_short_conv=use_short_conv,
                layer_idx=0,
            ).to(device=device, dtype=dtype)
            fla_gdn2_block.train()
        except Exception as e:
            try:
                fla_gdn2_block = fla_gdn2_cls(
                    hidden_size=d_model,
                    num_heads=num_kv_heads,
                    head_dim=d_k,
                    expand_v=float(d_v) / float(d_k),
                    use_short_conv=use_short_conv,
                    layer_idx=0,
                ).to(device=device, dtype=dtype)
                fla_gdn2_block.train()
            except Exception as e2:
                print(f"[!] Error instantiating FLA GatedDeltaNet-2: {e2}")
                fla_gdn2_block = None

    # FLA GatedDeltaNet (v1)
    fla_gdn_block = None
    if fla_gdn_cls is not None:
        try:
            print("Initializing FLA GatedDeltaNet (v1 chunkwise parallel)...")
            fla_gdn_block = fla_gdn_cls(
                hidden_size=d_model,
                num_heads=num_kv_heads,
                head_dim=d_k,
                expand_v=float(d_v) / float(d_k),
                mode="chunk",
                use_short_conv=use_short_conv,
                layer_idx=0,
            ).to(device=device, dtype=dtype)
            fla_gdn_block.train()
        except Exception as e:
            try:
                fla_gdn_block = fla_gdn_cls(
                    hidden_size=d_model,
                    num_heads=num_kv_heads,
                    head_dim=d_k,
                    expand_v=float(d_v) / float(d_k),
                    use_short_conv=use_short_conv,
                    layer_idx=0,
                ).to(device=device, dtype=dtype)
                fla_gdn_block.train()
            except Exception as e2:
                print(f"[!] Error instantiating FLA GatedDeltaNet: {e2}")
                fla_gdn_block = None

    all_results = []

    headers = [
        "SeqLen (L)",
        "Tokens (B*L)",
        "Method / Implementation",
        "Fwd (ms)",
        "Fwd+Bwd (ms)",
        "Throughput (tok/s)",
        "Peak VRAM (MB)",
        "Speedup",
    ]
    rows = []

    B = batch_size
    for L in seq_lens:
        total_tokens = B * L
        print(f"\n---> Benchmarking B = {B}, L = {L} (Total Tokens: {total_tokens:,}) ...")

        # Tensors
        x_batched = torch.randn(B, L, d_model, device=device, dtype=dtype, requires_grad=True)
        x_packed = x_batched.detach().reshape(total_tokens, d_model).clone().requires_grad_(True)
        cu_seqlens = torch.arange(0, (B + 1) * L, L, dtype=torch.int32, device=device)

        # Baseline A: PyTorch Reference (Unfused autograd)
        def run_ref_fwd():
            with torch.no_grad():
                return our_block(x_batched, use_kernel=False)

        def run_ref_step():
            our_block.zero_grad(set_to_none=True)
            if x_batched.grad is not None:
                x_batched.grad = None
            out, _ = our_block(x_batched, use_kernel=False)
            loss = out.sum()
            loss.backward()

        res_ref_fwd = measure_performance(run_ref_fwd, num_warmup, num_iters, device)
        res_ref_step = measure_performance(run_ref_step, num_warmup, num_iters, device)
        ref_step_lat = res_ref_step["latency_ms"]

        if res_ref_step["status"] == "OK":
            tok_s_ref = (total_tokens * 1000.0) / ref_step_lat
            rows.append([
                L,
                f"{total_tokens:,}",
                "PyTorch Reference (Unfused)",
                f"{res_ref_fwd['latency_ms']:.2f}",
                f"{ref_step_lat:.2f}",
                f"{tok_s_ref:,.0f}",
                f"{res_ref_step['peak_alloc_mb']:.1f}",
                "1.00x",
            ])
        else:
            print(f"  [!] PyTorch Reference failed at L={L}: {res_ref_step.get('error_msg')}")
            rows.append([L, f"{total_tokens:,}", "PyTorch Reference (Unfused)", "-", res_ref_step["status"], "-", "-", "-"])

        # Baseline B: FLA GatedDeltaNet (v1)
        if fla_gdn_block is not None:
            def run_fla_gdn_fwd():
                with torch.no_grad():
                    return fla_gdn_block(x_batched)

            def run_fla_gdn_step():
                fla_gdn_block.zero_grad(set_to_none=True)
                if x_batched.grad is not None:
                    x_batched.grad = None
                out, *_ = fla_gdn_block(x_batched)
                loss = out.sum()
                loss.backward()

            res_fla_gdn_fwd = measure_performance(run_fla_gdn_fwd, num_warmup, num_iters, device)
            res_fla_gdn_step = measure_performance(run_fla_gdn_step, num_warmup, num_iters, device)

            if res_fla_gdn_step["status"] == "OK":
                lat = res_fla_gdn_step["latency_ms"]
                tok_s = (total_tokens * 1000.0) / lat
                sp = f"{ref_step_lat / lat:.2f}x" if (ref_step_lat > 0 and lat > 0) else "-"
                rows.append([
                    L,
                    f"{total_tokens:,}",
                    "FLA GatedDeltaNet (v1)",
                    f"{res_fla_gdn_fwd['latency_ms']:.2f}",
                    f"{lat:.2f}",
                    f"{tok_s:,.0f}",
                    f"{res_fla_gdn_step['peak_alloc_mb']:.1f}",
                    sp,
                ])
            else:
                print(f"  [!] FLA GatedDeltaNet (v1) notice at L={L}: {res_fla_gdn_step.get('error_msg')}")
                rows.append([L, f"{total_tokens:,}", "FLA GatedDeltaNet (v1)", "-", res_fla_gdn_step["status"], "-", "-", "-"])

        # Baseline C: FLA GatedDeltaNet-2
        if fla_gdn2_block is not None:
            def run_fla_gdn2_fwd():
                with torch.no_grad():
                    return fla_gdn2_block(x_batched)

            def run_fla_gdn2_step():
                fla_gdn2_block.zero_grad(set_to_none=True)
                if x_batched.grad is not None:
                    x_batched.grad = None
                out, *_ = fla_gdn2_block(x_batched)
                loss = out.sum()
                loss.backward()

            res_fla_gdn2_fwd = measure_performance(run_fla_gdn2_fwd, num_warmup, num_iters, device)
            res_fla_gdn2_step = measure_performance(run_fla_gdn2_step, num_warmup, num_iters, device)

            if res_fla_gdn2_step["status"] == "OK":
                lat = res_fla_gdn2_step["latency_ms"]
                tok_s = (total_tokens * 1000.0) / lat
                sp = f"{ref_step_lat / lat:.2f}x" if (ref_step_lat > 0 and lat > 0) else "-"
                rows.append([
                    L,
                    f"{total_tokens:,}",
                    "FLA GatedDeltaNet-2",
                    f"{res_fla_gdn2_fwd['latency_ms']:.2f}",
                    f"{lat:.2f}",
                    f"{tok_s:,.0f}",
                    f"{res_fla_gdn2_step['peak_alloc_mb']:.1f}",
                    sp,
                ])
            else:
                print(f"  [!] FLA GatedDeltaNet-2 notice at L={L}: {res_fla_gdn2_step.get('error_msg')}")
                rows.append([L, f"{total_tokens:,}", "FLA GatedDeltaNet-2", "-", res_fla_gdn2_step["status"], "-", "-", "-"])

        # Baseline D: NVLabs GatedDeltaNet-2 (Official NVIDIA)
        if nvlabs_gdn2_block is not None:
            def run_nvlabs_gdn2_fwd():
                with torch.no_grad():
                    return nvlabs_gdn2_block(x_batched)

            def run_nvlabs_gdn2_step():
                nvlabs_gdn2_block.zero_grad(set_to_none=True)
                if x_batched.grad is not None:
                    x_batched.grad = None
                out, *_ = nvlabs_gdn2_block(x_batched)
                loss = out.sum()
                loss.backward()

            res_nvlabs_gdn2_fwd = measure_performance(run_nvlabs_gdn2_fwd, num_warmup, num_iters, device)
            res_nvlabs_gdn2_step = measure_performance(run_nvlabs_gdn2_step, num_warmup, num_iters, device)

            if res_nvlabs_gdn2_step["status"] == "OK":
                lat = res_nvlabs_gdn2_step["latency_ms"]
                tok_s = (total_tokens * 1000.0) / lat
                sp = f"{ref_step_lat / lat:.2f}x" if (ref_step_lat > 0 and lat > 0) else "-"
                rows.append([
                    L,
                    f"{total_tokens:,}",
                    "NVLabs GatedDeltaNet-2",
                    f"{res_nvlabs_gdn2_fwd['latency_ms']:.2f}",
                    f"{lat:.2f}",
                    f"{tok_s:,.0f}",
                    f"{res_nvlabs_gdn2_step['peak_alloc_mb']:.1f}",
                    sp,
                ])
            else:
                print(f"  [!] NVLabs GatedDeltaNet-2 notice at L={L}: {res_nvlabs_gdn2_step.get('error_msg')}")
                rows.append([L, f"{total_tokens:,}", "NVLabs GatedDeltaNet-2", "-", res_nvlabs_gdn2_step["status"], "-", "-", "-"])

        # Method: DeltaNet v2 (Ours - SM75 Fused CUDA Kernel)
        def run_ours_fwd():
            with torch.no_grad():
                return our_block(x_packed, cu_seqlens=cu_seqlens, use_kernel=True)

        def run_ours_step():
            our_block.zero_grad(set_to_none=True)
            if x_packed.grad is not None:
                x_packed.grad = None
            out, _ = our_block(x_packed, cu_seqlens=cu_seqlens, use_kernel=True)
            loss = out.sum()
            loss.backward()

        res_ours_fwd = measure_performance(run_ours_fwd, num_warmup, num_iters, device)
        res_ours_step = measure_performance(run_ours_step, num_warmup, num_iters, device)
        ours_step_lat = res_ours_step["latency_ms"]

        if res_ours_step["status"] == "OK":
            tok_s_ours = (total_tokens * 1000.0) / ours_step_lat
            sp = f"{ref_step_lat / ours_step_lat:.2f}x" if (ref_step_lat > 0 and ours_step_lat > 0) else "-"
            rows.append([
                L,
                f"{total_tokens:,}",
                "**DeltaNet v2 (Ours - SM75 CuTe)**",
                f"**{res_ours_fwd['latency_ms']:.2f}**",
                f"**{ours_step_lat:.2f}**",
                f"**{tok_s_ours:,.0f}**",
                f"**{res_ours_step['peak_alloc_mb']:.1f}**",
                f"**{sp}**",
            ])
        else:
            print(f"  [!] DeltaNet v2 (Ours) failed at L={L}: {res_ours_step.get('error_msg')}")
            rows.append([L, f"{total_tokens:,}", "**DeltaNet v2 (Ours - SM75 CuTe)**", "-", res_ours_step["status"], "-", "-", "-"])

        all_results.append({
            "L": L,
            "total_tokens": total_tokens,
            "ref": res_ref_step,
            "ours": res_ours_step,
        })

    table_md = format_markdown_table(headers, rows)
    print("\n" + "=" * 72)
    print("TRAINING PASS (FORWARD + BACKWARD) BENCHMARK RESULTS")
    print("=" * 72)
    print(table_md)
    print("=" * 72 + "\n")
    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="DeltaNet v2 Training Step Benchmark")
    parser.add_argument("--dim", type=int, default=768, help="Hidden dimension (d_model)")
    parser.add_argument("--q-heads", type=int, default=8, help="Number of Query heads")
    parser.add_argument("--kv-heads", type=int, default=8, help="Number of Key/Value heads")
    parser.add_argument("--d-k", type=int, default=64, help="Head dimension for Keys")
    parser.add_argument("--d-v", type=int, default=64, help="Head dimension for Values")
    parser.add_argument("--chunk-size", type=int, default=64, help="Chunk size for WY representation")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (B)")
    parser.add_argument("--seq-lens", type=str, default="512,1024,2048,4096", help="Comma-separated sequence lengths")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=30, help="Number of timed iterations")
    parser.add_argument("--use-conv", action="store_true", help="Enable ShortConvolution in baselines (default: False)")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 instead of float16")
    parser.add_argument("--save-md", type=str, default=None, help="Save markdown table to file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print_hardware_info()
    print_baseline_status()

    l_list = [int(s.strip()) for s in args.seq_lens.split(",") if s.strip()]
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    results = run_training_benchmark(
        d_model=args.dim,
        num_q_heads=args.q_heads,
        num_kv_heads=args.kv_heads,
        d_k=args.d_k,
        d_v=args.d_v,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        seq_lens=l_list,
        num_warmup=args.warmup,
        num_iters=args.iters,
        use_short_conv=args.use_conv,
        dtype=dtype,
    )
