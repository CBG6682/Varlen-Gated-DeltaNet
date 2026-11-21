"""
Benchmark: Autoregressive Inference (L = 1) & End-to-End Prefill+Generation
Evaluates:
  - Part A: Pure Gated DeltaNet token-mixer decode step (L = 1) vs Open-Source Baselines
            (PyTorch Reference, FLA v1, FLA GatedDeltaNet-2, NVLabs GatedDeltaNet-2,
             DeltaNet v2 Eager & torch.compile).
  - Part B: End-to-End DeltaBlock (with MLP/FFW) Decode L=1 and Prefill + Multi-Step
            Generation across torch-eager, torch-compiled, kernel-eager, kernel-compiled.
Supports --save-plot, --save-md, and --save-json.
"""

import argparse
import gc
import json
import os
import sys
import warnings
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore", message=".*ShortConvolution is crucial.*")

import torch

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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from deltanet_v2.model import DeltaBlock
from tests.utils import (
    compute_deltanet_inference_flops,
    compute_deltanet_inference_memory_bytes,
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


def _try_init_infer_baseline(cls: Any, label: str, d_model: int, num_kv_heads: int,
                             d_k: int, d_v: int, use_short_conv: bool,
                             device: torch.device, dtype: torch.dtype):
    """Instantiate an inference baseline block, falling back across constructor signatures."""
    if cls is None:
        return None
    for extra in ({"mode": "fused_recurrent"}, {}):
        try:
            blk = cls(
                hidden_size=d_model,
                num_heads=num_kv_heads,
                head_dim=d_k,
                expand_v=float(d_v) / float(d_k),
                use_short_conv=use_short_conv,
                layer_idx=0,
                **extra,
            ).to(device=device, dtype=dtype)
            blk.eval()
            return blk
        except Exception:
            continue
    print(f"[!] Warning: Could not instantiate {label} for inference.")
    return None


def _compile_or_none(target: Any, label: str):
    """Compile module with TorchInductor, returning None cleanly on failure."""
    try:
        return torch.compile(target, backend="inductor")
    except Exception as exc:
        print(f"[!] torch.compile({label}) skipped: {exc}")
        return None


def run_inference_benchmark(
    d_model: int = 768,
    num_q_heads: int = 8,
    num_kv_heads: int = 8,
    d_k: int = 64,
    d_v: int = 64,
    batch_sizes: Optional[List[int]] = None,
    num_warmup: int = 5,
    num_iters: int = 20,
    use_short_conv: bool = False,
    include_compile: bool = True,
    dtype: torch.dtype = torch.float16,
    device: Optional[torch.device] = None,
):
    """Part A: Benchmark pure token-mixer decode step (L = 1) across batch sizes."""
    device = device or get_device()
    batch_sizes = batch_sizes or [1, 16, 64, 256, 1024, 2048]

    print_section("PART A: AUTOREGRESSIVE DECODE (L = 1, PURE TOKEN MIXER VS BASELINES)")
    print(
        f"Config: d_model={d_model}, H_q={num_q_heads}, H_kv={num_kv_heads}, "
        f"d_k={d_k}, d_v={d_v}, dtype={dtype}, device={device}"
    )

    our_block = DeltaBlock(
        d_model=d_model, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        d_k=d_k, d_v=d_v, hidden_dim=None,
    ).to(device=device, dtype=dtype).eval()
    our_compiled = _compile_or_none(our_block, "DeltaBlock-infer") if include_compile else None

    fla_cache_cls = try_import_fla_cache()
    baselines = [
        ("FLA GatedDeltaNet (v1)", _try_init_infer_baseline(
            try_import_fla_gdn(), "FLA GatedDeltaNet (v1)",
            d_model, num_kv_heads, d_k, d_v, use_short_conv, device, dtype,
        )),
        ("FLA GatedDeltaNet-2", _try_init_infer_baseline(
            try_import_fla_gdn2(), "FLA GatedDeltaNet-2",
            d_model, num_kv_heads, d_k, d_v, use_short_conv, device, dtype,
        )),
        ("NVLabs GatedDeltaNet-2", _try_init_infer_baseline(
            try_import_nvlabs_gdn2(), "NVLabs GatedDeltaNet-2",
            d_model, num_kv_heads, d_k, d_v, use_short_conv, device, dtype,
        )),
    ]

    headers = [
        "Batch (B)", "Method / Implementation", "Latency (ms)", "Throughput (tok/s)",
        "GFLOPs", "TFLOP/s", "Memory (GB/s)", "Peak VRAM (MB)", "Speedup",
    ]
    rows: List[List[Any]] = []
    records: List[Dict[str, Any]] = []

    dtype_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4

    for B in tqdm(batch_sizes, desc="Inference (Part A)", unit="batch"):
        print(f"\n---> Benchmarking Batch Size B = {B} (L = 1) ...", flush=True)
        gflops = compute_deltanet_inference_flops(
            batch_size=B, d_model=d_model, num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads, d_k=d_k, d_v=d_v,
        ) / 1e9
        total_bytes = compute_deltanet_inference_memory_bytes(
            batch_size=B, d_model=d_model, num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads, d_k=d_k, d_v=d_v,
            dtype_bytes=dtype_bytes, state_bytes=2,
        )

        x_step = torch.randn(B, 1, d_model, device=device, dtype=dtype)
        past_state = torch.zeros(B, num_kv_heads, d_v, d_k, device=device, dtype=torch.float16)

        def _run_ref():
            with torch.no_grad():
                return our_block(x_step, past_states=past_state, use_kernel=False)

        res_ref = measure_performance(_run_ref, num_warmup, num_iters, device)
        ref_lat = res_ref["latency_ms"] if res_ref["status"] == "OK" else 0.0

        def _append_result(label: str, res: Dict[str, Any], bold: bool = False):
            status = res["status"]
            if status == "OK":
                lat = res["latency_ms"]
                tok_s = (B * 1000.0) / lat if lat > 0 else 0.0
                tflops = gflops / lat if lat > 0 else 0.0
                mem_bw = total_bytes / (lat * 1e6) if lat > 0 else 0.0
                sp = f"{ref_lat / lat:.2f}x" if (ref_lat > 0 and lat > 0) else "-"
                if label == "PyTorch Reference (Unfused)":
                    sp = "1.00x"
                peak_mb = res["peak_alloc_mb"]
                print(f"  ✓ {label}: Lat={lat:.3f}ms | {tok_s:,.0f} tok/s | {mem_bw:.1f} GB/s", flush=True)
                wrap = (lambda s: f"**{s}**") if bold else (lambda s: s)
                rows.append([
                    B, wrap(label), wrap(f"{lat:.3f}"), wrap(f"{tok_s:,.0f}"),
                    wrap(f"{gflops:.3f}"), wrap(f"{tflops:.2f}"),
                    wrap(f"{mem_bw:.1f}"), wrap(f"{peak_mb:.1f}"), wrap(sp),
                ])
                records.append({
                    "batch_size": B, "method": label, "latency_ms": lat,
                    "throughput_tok_s": tok_s, "mem_bw_gbps": mem_bw,
                    "peak_mb": peak_mb, "status": "OK",
                })
            else:
                print(f"  ✗ {label} failed at B={B}: {res.get('error_msg')}", flush=True)
                rows.append([B, label, status, "-", "-", "-", "-", "-", "-"])
                records.append({
                    "batch_size": B, "method": label, "latency_ms": None,
                    "throughput_tok_s": None, "mem_bw_gbps": None,
                    "peak_mb": None, "status": status,
                })

        _append_result("PyTorch Reference (Unfused)", res_ref)

        for bname, bmod in baselines:
            if bmod is None:
                continue
            cache = fla_cache_cls() if fla_cache_cls is not None else None

            def _run_base(m=bmod, c=cache):
                with torch.no_grad():
                    return m(x_step, past_key_values=c, use_cache=True)

            _append_result(bname, measure_performance(_run_base, num_warmup, num_iters, device))

        def _run_ours():
            with torch.no_grad():
                return our_block(x_step, past_states=past_state, use_kernel=True)

        _append_result("DeltaNet v2 (Ours)", measure_performance(_run_ours, num_warmup, num_iters, device), bold=True)

        if our_compiled is not None:
            def _run_ours_comp():
                with torch.no_grad():
                    return our_compiled(x_step, past_states=past_state, use_kernel=True)

            _append_result(
                "DeltaNet v2 (Compiled)",
                measure_performance(_run_ours_comp, num_warmup, num_iters, device),
                bold=True,
            )

        del x_step, past_state
        gc.collect()
        torch.cuda.empty_cache()

    table_md = format_markdown_table(headers, rows)
    print("\n" + "=" * 78)
    print("PART A: INFERENCE BENCHMARK RESULTS (L = 1, PURE TOKEN MIXER)")
    print("=" * 78)
    print(table_md)
    print("=" * 78 + "\n")
    return records, table_md


def run_e2e_inference_benchmark(
    d_model: int = 768,
    num_q_heads: int = 8,
    num_kv_heads: int = 4,
    d_k: int = 64,
    d_v: int = 64,
    chunk_size: int = 64,
    decode_batches: Optional[List[int]] = None,
    prefill_len: int = 512,
    gen_steps: int = 32,
    num_warmup: int = 4,
    num_iters: int = 12,
    dtype: torch.dtype = torch.float16,
    device: Optional[torch.device] = None,
):
    """Part B: End-to-End DeltaBlock (with MLP) decode L=1 and prefill+decode rollout."""
    device = device or get_device()
    decode_batches = decode_batches or [16, 64, 256, 1024]
    hidden_dim = d_model * 4

    print_section("PART B: END-TO-END DELTABLOCK INFERENCE (WITH MLP + torch.compile)")
    print(
        f"Config: d_model={d_model}, hidden_dim={hidden_dim}, H_q={num_q_heads}, H_kv={num_kv_heads}, "
        f"decode_batches={decode_batches}, prefill_len={prefill_len}, gen_steps={gen_steps}"
    )

    headers = ["Batch (B)", "Mode", "Decode L=1 (ms)", "Throughput (tok/s)", "E2E Prefill+Gen (ms)", "Peak VRAM (MB)"]
    rows: List[List[Any]] = []
    records: List[Dict[str, Any]] = []

    modes = [
        ("torch-eager", False, False),
        ("torch-compiled", False, True),
        ("kernel-eager", True, False),
        ("kernel-compiled", True, True),
    ]

    for B in decode_batches:
        for mname, use_k, do_comp in modes:
            gc.collect()
            torch.cuda.empty_cache()
            blk = DeltaBlock(
                d_model=d_model, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
                d_k=d_k, d_v=d_v, chunk_size=chunk_size, hidden_dim=hidden_dim,
            ).to(device=device, dtype=dtype).eval()
            fn = _compile_or_none(blk, mname) if do_comp else blk
            if fn is None:
                rows.append([B, mname, "COMPILE_ERR", "-", "-", "-"])
                records.append({"B": B, "mode": mname, "dec_ms": None, "tok_s": None, "e2e_ms": None, "peak_mb": None, "status": "COMPILE_ERR"})
                del blk
                continue

            x_dec = torch.randn(B, 1, d_model, device=device, dtype=dtype)
            past = torch.zeros(B, num_kv_heads, d_v, d_k, device=device, dtype=torch.float16)

            def _dec_step():
                with torch.no_grad():
                    return fn(x_dec, past_states=past, use_kernel=use_k)

            res_dec = measure_performance(_dec_step, num_warmup, num_iters, device)

            # E2E Prefill + gen_steps rollout on compact batch <= 256 to avoid excessive Colab OOM
            res_e2e_ms: Optional[float] = None
            peak_e2e_mb = res_dec["peak_alloc_mb"]
            if B <= 256 and res_dec["status"] == "OK":
                x_pre_var = torch.randn(B * prefill_len, d_model, device=device, dtype=dtype)
                cu_pre = torch.arange(0, (B + 1) * prefill_len, prefill_len, dtype=torch.int32, device=device)
                x_pre_pad = x_pre_var.view(B, prefill_len, d_model)
                x_roll = torch.randn(B, gen_steps, 1, d_model, device=device, dtype=dtype)

                def _e2e_run():
                    with torch.no_grad():
                        if use_k:
                            _, st = fn(x_pre_var, cu_seqlens=cu_pre, use_kernel=True)
                        else:
                            _, st = fn(x_pre_pad, use_kernel=False)
                        for t in range(gen_steps):
                            _, st = fn(x_roll[:, t], past_states=st, use_kernel=use_k)
                        return st

                r_e2e = measure_performance(_e2e_run, max(2, num_warmup // 2), max(4, num_iters // 3), device)
                if r_e2e["status"] == "OK":
                    res_e2e_ms = r_e2e["latency_ms"]
                    peak_e2e_mb = max(peak_e2e_mb, r_e2e["peak_alloc_mb"])
                del x_pre_var, cu_pre, x_pre_pad, x_roll

            if res_dec["status"] == "OK":
                dec_ms = res_dec["latency_ms"]
                tok_s = (B * 1000.0) / dec_ms if dec_ms > 0 else 0.0
                e2e_str = f"{res_e2e_ms:.2f}" if res_e2e_ms is not None else "-"
                print(f"  [B={B}] {mname:<15}: decode={dec_ms:.3f}ms ({tok_s:,.0f} tok/s) | E2E={e2e_str}ms | VRAM={peak_e2e_mb:.0f}MB")
                bold = (mname == "kernel-compiled")
                w = (lambda s: f"**{s}**") if bold else (lambda s: s)
                rows.append([B, w(mname), w(f"{dec_ms:.3f}"), w(f"{tok_s:,.0f}"), w(e2e_str), w(f"{peak_e2e_mb:.1f}")])
                records.append({
                    "B": B, "mode": mname, "dec_ms": dec_ms, "tok_s": tok_s,
                    "e2e_ms": res_e2e_ms, "peak_mb": peak_e2e_mb, "status": "OK",
                })
            else:
                print(f"  [B={B}] {mname:<15}: {res_dec['status']}")
                rows.append([B, mname, res_dec["status"], "-", "-", "-"])
                records.append({
                    "B": B, "mode": mname, "dec_ms": None, "tok_s": None,
                    "e2e_ms": None, "peak_mb": None, "status": res_dec["status"],
                })

            del x_dec, past, blk, fn
            gc.collect()
            torch.cuda.empty_cache()

    table_md = format_markdown_table(headers, rows)
    print("\n" + "=" * 78)
    print(f"PART B: E2E DELTABLOCK INFERENCE RESULTS (Prefill={prefill_len}, Gen={gen_steps})")
    print("=" * 78)
    print(table_md)
    print("=" * 78 + "\n")
    return records, table_md


def _save_inference_plot(
    part_a: List[Dict[str, Any]], part_b: Optional[List[Dict[str, Any]]], save_path: str
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[!] matplotlib/numpy not installed; skipping inference plot.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    methods = [
        ("PyTorch Reference (Unfused)", "#6c757d"),
        ("FLA GatedDeltaNet (v1)", "#3b82f6"),
        ("FLA GatedDeltaNet-2", "#8b5cf6"),
        ("NVLabs GatedDeltaNet-2", "#f59e0b"),
        ("DeltaNet v2 (Ours)", "#10b981"),
        ("DeltaNet v2 (Compiled)", "#047857"),
    ]
    active_methods = [(m, c) for m, c in methods if any(r["method"] == m and r["status"] == "OK" for r in part_a)]
    batches_a = sorted({r["batch_size"] for r in part_a})

    ncols = 2
    nrows = 2 if part_b else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5.2 * nrows))
    axes_2d = axes.reshape(nrows, ncols)
    fig.suptitle("DeltaNet v2 Inference Benchmark (NVIDIA Turing SM75, FP16)", fontsize=13, fontweight="bold")

    ax1 = axes_2d[0, 0]
    ax2 = axes_2d[0, 1]
    bx_labels = [str(b) for b in batches_a]
    for m, color in active_methods:
        lw = 2.4 if "Ours" in m or "Compiled" in m else 1.5
        toks = [next((r["throughput_tok_s"] for r in part_a if r["batch_size"] == b and r["method"] == m), None) for b in batches_a]
        lats = [next((r["latency_ms"] for r in part_a if r["batch_size"] == b and r["method"] == m), None) for b in batches_a]
        if any(v is not None for v in toks):
            ax1.plot(bx_labels, [v or 0.0 for v in toks], marker="o", linewidth=lw, label=m, color=color)
            ax2.plot(bx_labels, [v or 0.0 for v in lats], marker="o", linewidth=lw, label=m, color=color)

    ax1.set_title("Part A: Decode (L=1) Throughput vs Baselines (Higher is Better)")
    ax1.set_xlabel("Batch Size (B)")
    ax1.set_ylabel("Throughput (tokens/s)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_title("Part A: Decode (L=1) Latency vs Baselines (Lower is Better)")
    ax2.set_xlabel("Batch Size (B)")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    if part_b:
        batches_b = sorted({r["B"] for r in part_b})
        e2e_modes = [
            ("torch-eager", "#6c757d"),
            ("torch-compiled", "#3b82f6"),
            ("kernel-eager", "#f59e0b"),
            ("kernel-compiled", "#10b981"),
        ]
        ax3 = axes_2d[1, 0]
        ax4 = axes_2d[1, 1]

        for m, color in e2e_modes:
            toks = [next((r["tok_s"] or 0.0 for r in part_b if r["B"] == b and r["mode"] == m), 0.0) for b in batches_b]
            ax3.plot([str(b) for b in batches_b], toks, marker="o", linewidth=2.2 if "kernel" in m else 1.5, label=m, color=color)

        ax3.set_title("Part B: E2E DeltaBlock Decode Throughput (L=1)")
        ax3.set_xlabel("Batch Size (B)")
        ax3.set_ylabel("Throughput (tokens/s)")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)

        e2e_batches = [b for b in batches_b if any(r["B"] == b and r.get("e2e_ms") is not None for r in part_b)]
        if e2e_batches:
            xb = np.arange(len(e2e_batches))
            wb = 0.19
            for idx, (m, color) in enumerate(e2e_modes):
                offs = xb + (idx - 1.5) * wb
                e2e_vals = [next((r["e2e_ms"] or 0.0 for r in part_b if r["B"] == b and r["mode"] == m), 0.0) for b in e2e_batches]
                ax4.bar(offs, e2e_vals, width=wb, label=m, color=color)
            ax4.set_xticks(xb)
            ax4.set_xticklabels([f"B={b}" for b in e2e_batches])
            ax4.set_title("Part B: E2E Prefill + Generation Latency (Lower is Better)")
            ax4.set_ylabel("Total Latency (ms)")
            ax4.grid(True, axis="y", alpha=0.3)
            ax4.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved inference benchmark plot -> {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="DeltaNet v2 Autoregressive Inference Benchmark")
    parser.add_argument("--dim", type=int, default=768, help="Hidden dimension (d_model)")
    parser.add_argument("--q-heads", type=int, default=8, help="Number of Query heads")
    parser.add_argument("--kv-heads", type=int, default=8, help="Number of Key/Value heads")
    parser.add_argument("--d-k", type=int, default=64, help="Head dimension for Keys")
    parser.add_argument("--d-v", type=int, default=64, help="Head dimension for Values")
    parser.add_argument("--batches", type=str, default="1,16,64,256,1024,2048", help="Comma-separated batch sizes (Part A)")
    parser.add_argument("--e2e-batches", type=str, default="16,64,256,1024", help="Comma-separated batch sizes (Part B E2E)")
    parser.add_argument("--prefill-len", type=int, default=512, help="Prefill length for Part B E2E rollout")
    parser.add_argument("--gen-steps", type=int, default=32, help="Generation steps for Part B E2E rollout")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Number of timed iterations")
    parser.add_argument("--use-conv", action="store_true", help="Enable ShortConvolution in baselines")
    parser.add_argument("--skip-e2e", action="store_true", help="Skip Part B E2E DeltaBlock benchmark")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile variants")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 instead of float16")
    parser.add_argument("--save-md", type=str, default=None, help="Save markdown tables to file")
    parser.add_argument("--save-json", type=str, default=None, help="Save benchmark results as JSON")
    parser.add_argument("--save-plot", type=str, default=None, help="Save Matplotlib benchmark PNG")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print_hardware_info()
    print_baseline_status()

    b_list = [int(b.strip()) for b in args.batches.split(",") if b.strip()]
    e2e_b_list = [int(b.strip()) for b in args.e2e_batches.split(",") if b.strip()]
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    part_a_rec, part_a_md = run_inference_benchmark(
        d_model=args.dim, num_q_heads=args.q_heads, num_kv_heads=args.kv_heads,
        d_k=args.d_k, d_v=args.d_v, batch_sizes=b_list, num_warmup=args.warmup,
        num_iters=args.iters, use_short_conv=args.use_conv,
        include_compile=not args.no_compile, dtype=dtype,
    )

    part_b_rec, part_b_md = None, ""
    if not args.skip_e2e:
        e2e_hkv = 4 if (args.q_heads % 4 == 0) else args.kv_heads
        part_b_rec, part_b_md = run_e2e_inference_benchmark(
            d_model=args.dim, num_q_heads=args.q_heads, num_kv_heads=e2e_hkv,
            d_k=args.d_k, d_v=args.d_v, decode_batches=e2e_b_list,
            prefill_len=args.prefill_len, gen_steps=args.gen_steps,
            num_warmup=args.warmup, num_iters=max(8, args.iters), dtype=dtype,
        )

    if args.save_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_md)), exist_ok=True)
        with open(args.save_md, "w", encoding="utf-8") as f:
            f.write("### Part A: Token-Mixer Decode Baselines\n\n" + part_a_md + "\n")
            if part_b_md:
                f.write("\n### Part B: E2E DeltaBlock Decode & Prefill+Gen (`torch.compile`)\n\n" + part_b_md + "\n")

    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump({"part_a": part_a_rec, "part_b": part_b_rec}, f, indent=2)

    if args.save_plot:
        _save_inference_plot(part_a_rec, part_b_rec, args.save_plot)
