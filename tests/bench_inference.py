"""
Benchmark: Single-Step Autoregressive Inference (L = 1)
Evaluates decoding latency (ms/step), generation throughput (tokens/sec), and peak VRAM.
Compares DeltaNet v2 fused CUDA kernel against NVLabs GatedDeltaNet-2, FLA GatedDeltaNet-2, FLA GatedDeltaNet (v1), and PyTorch reference.

Benchmarks strictly the Gated DeltaNet token-mixer block (without MLP / FFW / SWA)
to ensure direct and fair comparison of the recurrence kernels.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        if iterable is not None:
            return iterable
        class DummyPBar:
            def update(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return DummyPBar()

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
    try_import_fla_cache,
    try_import_fla_gdn,
    try_import_fla_gdn2,
    try_import_nvlabs_gdn2,
)


def run_inference_benchmark(
    d_model: int = 768,
    num_q_heads: int = 8,
    num_kv_heads: int = 8,
    d_k: int = 64,
    d_v: int = 64,
    batch_sizes: Optional[List[int]] = None,
    num_warmup: int = 15,
    num_iters: int = 50,
    use_short_conv: bool = False,
    dtype: torch.dtype = torch.float16,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """
    Benchmark autoregressive generation step (L = 1) across batch sizes.
    """
    if device is None:
        device = get_device()
    if batch_sizes is None:
        batch_sizes = [1, 16, 64, 256, 1024, 2048]

    print_section("BENCHMARK: SINGLE-STEP AUTOREGRESSIVE INFERENCE (L = 1)")
    print(f"Config: d_model={d_model}, num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, "
          f"d_k={d_k}, d_v={d_v}, use_short_conv={use_short_conv}, dtype={dtype}, device={device}")
    print(f"Batch sizes to evaluate: {batch_sizes}\n")

    # 1. DeltaNet v2 Block (Pure Gated DeltaNet: hidden_dim=None excludes FFW/MLP)
    print("Initializing DeltaNet v2...")
    our_block = DeltaBlock(
        d_model=d_model,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        d_k=d_k,
        d_v=d_v,
        hidden_dim=None,
    ).to(device=device, dtype=dtype)
    our_block.eval()

    # 2. Check and load Baselines
    nvlabs_gdn2_cls = try_import_nvlabs_gdn2()
    fla_gdn2_cls = try_import_fla_gdn2()
    fla_gdn_cls = try_import_fla_gdn()
    fla_cache_cls = try_import_fla_cache()

    # NVLabs GatedDeltaNet-2 (Official NVIDIA implementation)
    nvlabs_gdn2_block = None
    if nvlabs_gdn2_cls is not None:
        try:
            print("Initializing NVLabs GatedDeltaNet-2 (Official NVIDIA reference)...")
            nvlabs_gdn2_block = nvlabs_gdn2_cls(
                hidden_size=d_model,
                num_heads=num_kv_heads,
                head_dim=d_k,
                expand_v=float(d_v) / float(d_k),
                mode="fused_recurrent",
                use_short_conv=use_short_conv,
                layer_idx=0,
            ).to(device=device, dtype=dtype)
            nvlabs_gdn2_block.eval()
        except Exception as e:
            print(f"[!] Warning: Could not instantiate NVLabs GatedDeltaNet-2 with mode='fused_recurrent' ({e}). Retrying default.")
            try:
                nvlabs_gdn2_block = nvlabs_gdn2_cls(
                    hidden_size=d_model,
                    num_heads=num_kv_heads,
                    head_dim=d_k,
                    expand_v=float(d_v) / float(d_k),
                    use_short_conv=use_short_conv,
                    layer_idx=0,
                ).to(device=device, dtype=dtype)
                nvlabs_gdn2_block.eval()
            except Exception as e2:
                print(f"[!] Error instantiating NVLabs GatedDeltaNet-2: {e2}")
                nvlabs_gdn2_block = None

    # FLA GatedDeltaNet-2 (Flash Linear Attention library)
    fla_gdn2_block = None
    if fla_gdn2_cls is not None:
        try:
            print("Initializing FLA GatedDeltaNet-2 (Flash Linear Attention)...")
            fla_gdn2_block = fla_gdn2_cls(
                hidden_size=d_model,
                num_heads=num_kv_heads,
                head_dim=d_k,
                expand_v=float(d_v) / float(d_k),
                mode="fused_recurrent",
                use_short_conv=use_short_conv,
                layer_idx=0,
            ).to(device=device, dtype=dtype)
            fla_gdn2_block.eval()
        except Exception as e:
            print(f"[!] Warning: Could not instantiate FLA GatedDeltaNet-2 with mode='fused_recurrent' ({e}). Retrying default.")
            try:
                fla_gdn2_block = fla_gdn2_cls(
                    hidden_size=d_model,
                    num_heads=num_kv_heads,
                    head_dim=d_k,
                    expand_v=float(d_v) / float(d_k),
                    use_short_conv=use_short_conv,
                    layer_idx=0,
                ).to(device=device, dtype=dtype)
                fla_gdn2_block.eval()
            except Exception as e2:
                print(f"[!] Error instantiating FLA GatedDeltaNet-2: {e2}")
                fla_gdn2_block = None

    # FLA GatedDeltaNet (v1)
    fla_gdn_block = None
    if fla_gdn_cls is not None:
        try:
            print("Initializing FLA GatedDeltaNet (v1)...")
            fla_gdn_block = fla_gdn_cls(
                hidden_size=d_model,
                num_heads=num_kv_heads,
                head_dim=d_k,
                expand_v=float(d_v) / float(d_k),
                mode="fused_recurrent",
                use_short_conv=use_short_conv,
                layer_idx=0,
            ).to(device=device, dtype=dtype)
            fla_gdn_block.eval()
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
                fla_gdn_block.eval()
            except Exception as e2:
                print(f"[!] Error instantiating FLA GatedDeltaNet (v1): {e2}")
                fla_gdn_block = None

    all_results = []

    # Table columns
    headers = [
        "Batch (B)",
        "Method / Implementation",
        "Latency (ms)",
        "Throughput (tok/s)",
        "Peak VRAM (MB)",
        "Speedup (vs PyTorch)",
    ]
    rows = []

    for B in tqdm(batch_sizes, desc="Inference Benchmark (Batch)", unit="batch"):
        print(f"\n---> Benchmarking Batch Size B = {B} (L = 1) ...")
        x_step = torch.randn(B, 1, d_model, device=device, dtype=dtype)
        past_state = torch.zeros(B, num_kv_heads, d_v, d_k, device=device, dtype=torch.float16)

        # Baseline A: PyTorch Reference (Unfused)
        def run_ref():
            with torch.no_grad():
                return our_block(x_step, past_states=past_state, use_kernel=False)

        res_ref = measure_performance(run_ref, num_warmup, num_iters, device)
        ref_lat = res_ref["latency_ms"]

        if res_ref["status"] == "OK":
            tok_s_ref = (B * 1000.0) / ref_lat
            rows.append([
                B,
                "PyTorch Reference (Unfused)",
                f"{ref_lat:.3f}",
                f"{tok_s_ref:,.0f}",
                f"{res_ref['peak_alloc_mb']:.1f}",
                "1.00x",
            ])
        else:
            print(f"  [!] PyTorch Reference failed at B={B}: {res_ref.get('error_msg')}")
            rows.append([B, "PyTorch Reference (Unfused)", res_ref["status"], "-", "-", "-"])

        # Baseline B: FLA GatedDeltaNet (v1)
        if fla_gdn_block is not None:
            cache_gdn = fla_cache_cls() if fla_cache_cls is not None else None

            def run_fla_gdn():
                with torch.no_grad():
                    return fla_gdn_block(x_step, past_key_values=cache_gdn, use_cache=True)

            res_fla_gdn = measure_performance(run_fla_gdn, num_warmup, num_iters, device)
            if res_fla_gdn["status"] == "OK":
                lat = res_fla_gdn["latency_ms"]
                tok_s = (B * 1000.0) / lat
                sp = f"{ref_lat / lat:.2f}x" if (ref_lat > 0 and lat > 0) else "-"
                rows.append([
                    B,
                    "FLA GatedDeltaNet (v1)",
                    f"{lat:.3f}",
                    f"{tok_s:,.0f}",
                    f"{res_fla_gdn['peak_alloc_mb']:.1f}",
                    sp,
                ])
            else:
                print(f"  [!] FLA GatedDeltaNet (v1) notice at B={B}: {res_fla_gdn.get('error_msg')}")
                rows.append([B, "FLA GatedDeltaNet (v1)", res_fla_gdn["status"], "-", "-", "-"])

        # Baseline C: FLA GatedDeltaNet-2
        if fla_gdn2_block is not None:
            cache_fla_gdn2 = fla_cache_cls() if fla_cache_cls is not None else None

            def run_fla_gdn2():
                with torch.no_grad():
                    return fla_gdn2_block(x_step, past_key_values=cache_fla_gdn2, use_cache=True)

            res_fla_gdn2 = measure_performance(run_fla_gdn2, num_warmup, num_iters, device)
            if res_fla_gdn2["status"] == "OK":
                lat = res_fla_gdn2["latency_ms"]
                tok_s = (B * 1000.0) / lat
                sp = f"{ref_lat / lat:.2f}x" if (ref_lat > 0 and lat > 0) else "-"
                rows.append([
                    B,
                    "FLA GatedDeltaNet-2",
                    f"{lat:.3f}",
                    f"{tok_s:,.0f}",
                    f"{res_fla_gdn2['peak_alloc_mb']:.1f}",
                    sp,
                ])
            else:
                print(f"  [!] FLA GatedDeltaNet-2 notice at B={B}: {res_fla_gdn2.get('error_msg')}")
                rows.append([B, "FLA GatedDeltaNet-2", res_fla_gdn2["status"], "-", "-", "-"])

        # Baseline D: NVLabs GatedDeltaNet-2 (Official NVIDIA)
        if nvlabs_gdn2_block is not None:
            cache_nvlabs_gdn2 = fla_cache_cls() if fla_cache_cls is not None else None

            def run_nvlabs_gdn2():
                with torch.no_grad():
                    return nvlabs_gdn2_block(x_step, past_key_values=cache_nvlabs_gdn2, use_cache=True)

            res_nvlabs_gdn2 = measure_performance(run_nvlabs_gdn2, num_warmup, num_iters, device)
            if res_nvlabs_gdn2["status"] == "OK":
                lat = res_nvlabs_gdn2["latency_ms"]
                tok_s = (B * 1000.0) / lat
                sp = f"{ref_lat / lat:.2f}x" if (ref_lat > 0 and lat > 0) else "-"
                rows.append([
                    B,
                    "NVLabs GatedDeltaNet-2",
                    f"{lat:.3f}",
                    f"{tok_s:,.0f}",
                    f"{res_nvlabs_gdn2['peak_alloc_mb']:.1f}",
                    sp,
                ])
            else:
                print(f"  [!] NVLabs GatedDeltaNet-2 notice at B={B}: {res_nvlabs_gdn2.get('error_msg')}")
                rows.append([B, "NVLabs GatedDeltaNet-2", res_nvlabs_gdn2["status"], "-", "-", "-"])

        # Method: DeltaNet v2 (Ours - Fused CUDA Kernel)
        def run_ours():
            with torch.no_grad():
                return our_block(x_step, past_states=past_state, use_kernel=True)

        res_ours = measure_performance(run_ours, num_warmup, num_iters, device)
        if res_ours["status"] == "OK":
            lat = res_ours["latency_ms"]
            tok_s = (B * 1000.0) / lat
            sp = f"{ref_lat / lat:.2f}x" if (ref_lat > 0 and lat > 0) else "-"
            rows.append([
                B,
                "**DeltaNet v2 (Ours)**",
                f"**{lat:.3f}**",
                f"**{tok_s:,.0f}**",
                f"**{res_ours['peak_alloc_mb']:.1f}**",
                f"**{sp}**",
            ])
        else:
            print(f"  [!] DeltaNet v2 (Ours) failed at B={B}: {res_ours.get('error_msg')}")
            rows.append([B, "**DeltaNet v2 (Ours)**", res_ours["status"], "-", "-", "-"])

        all_results.append({
            "B": B,
            "ref": res_ref,
            "ours": res_ours,
        })

    table_md = format_markdown_table(headers, rows)
    print("\n" + "=" * 72)
    print("INFERENCE BENCHMARK RESULTS (L = 1)")
    print("=" * 72)
    print(table_md)
    print("=" * 72 + "\n")
    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="DeltaNet v2 Autoregressive Inference Benchmark")
    parser.add_argument("--dim", type=int, default=768, help="Hidden dimension (d_model)")
    parser.add_argument("--q-heads", type=int, default=8, help="Number of Query heads")
    parser.add_argument("--kv-heads", type=int, default=8, help="Number of Key/Value heads")
    parser.add_argument("--d-k", type=int, default=64, help="Head dimension for Keys")
    parser.add_argument("--d-v", type=int, default=64, help="Head dimension for Values")
    parser.add_argument("--batches", type=str, default="1,16,64,256,1024,2048", help="Comma-separated batch sizes")
    parser.add_argument("--warmup", type=int, default=15, help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=50, help="Number of timed iterations")
    parser.add_argument("--use-conv", action="store_true", help="Enable ShortConvolution in baselines (default: False)")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 instead of float16")
    parser.add_argument("--save-md", type=str, default=None, help="Save markdown table to file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print_hardware_info()
    print_baseline_status()

    b_list = [int(b.strip()) for b in args.batches.split(",") if b.strip()]
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    results = run_inference_benchmark(
        d_model=args.dim,
        num_q_heads=args.q_heads,
        num_kv_heads=args.kv_heads,
        d_k=args.d_k,
        d_v=args.d_v,
        batch_sizes=b_list,
        num_warmup=args.warmup,
        num_iters=args.iters,
        use_short_conv=args.use_conv,
        dtype=dtype,
    )
