"""
Benchmark: Training Pass (Forward + Backward)
Evaluates:
  - Part A: Pure Gated DeltaNet token-mixer kernel vs Open-Source Baselines
            (PyTorch Reference, FLA v1, FLA GatedDeltaNet-2, NVLabs GatedDeltaNet-2,
             DeltaNet v2 Eager & torch.compile).
  - Part B: End-to-End DeltaBlock (with MLP/FFW) comparing torch-eager, torch-compiled,
            kernel-eager, and kernel-compiled (Throughput & Peak VRAM).
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
    compute_deltanet_training_flops,
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


def _try_init_train_baseline(cls: Any, label: str, d_model: int, num_kv_heads: int,
                            d_k: int, d_v: int, use_short_conv: bool,
                            device: torch.device, dtype: torch.dtype):
    """Instantiate a training baseline block, falling back across constructor signatures."""
    if cls is None:
        return None
    for extra in ({"mode": "chunk"}, {}):
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
            blk.train()
            return blk
        except Exception:
            continue
    print(f"[!] Warning: Could not instantiate {label} for training.")
    return None


def _compile_or_none(target: Any, label: str):
    """Compile module with TorchInductor, returning None cleanly on failure."""
    try:
        return torch.compile(target, backend="inductor")
    except Exception as exc:
        print(f"[!] torch.compile({label}) skipped: {exc}")
        return None


def run_training_benchmark(
    d_model: int = 768,
    num_q_heads: int = 8,
    num_kv_heads: int = 8,
    d_k: int = 64,
    d_v: int = 64,
    chunk_size: int = 64,
    batch_size: int = 4,
    seq_lens: Optional[List[int]] = None,
    num_warmup: int = 3,
    num_iters: int = 10,
    use_short_conv: bool = False,
    include_compile: bool = True,
    dtype: torch.dtype = torch.float16,
    device: Optional[torch.device] = None,
):
    """Part A: Benchmark pure token-mixer training step across sequence lengths."""
    device = device or get_device()
    seq_lens = seq_lens or [512, 1024, 2048, 4096]

    print_section("PART A: TRAINING BENCHMARK (PURE TOKEN MIXER VS OPEN-SOURCE BASELINES)")
    print(
        f"Config: d_model={d_model}, H_q={num_q_heads}, H_kv={num_kv_heads}, "
        f"d_k={d_k}, d_v={d_v}, chunk={chunk_size}, B={batch_size}, dtype={dtype}"
    )

    our_block = DeltaBlock(
        d_model=d_model, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        d_k=d_k, d_v=d_v, chunk_size=chunk_size, hidden_dim=None,
    ).to(device=device, dtype=dtype).train()
    our_compiled = _compile_or_none(our_block, "DeltaBlock-kernel") if include_compile else None

    baselines = [
        ("FLA GatedDeltaNet (v1)", _try_init_train_baseline(
            try_import_fla_gdn(), "FLA GatedDeltaNet (v1)",
            d_model, num_kv_heads, d_k, d_v, use_short_conv, device, dtype,
        )),
        ("FLA GatedDeltaNet-2", _try_init_train_baseline(
            try_import_fla_gdn2(), "FLA GatedDeltaNet-2",
            d_model, num_kv_heads, d_k, d_v, use_short_conv, device, dtype,
        )),
        ("NVLabs GatedDeltaNet-2", _try_init_train_baseline(
            try_import_nvlabs_gdn2(), "NVLabs GatedDeltaNet-2",
            d_model, num_kv_heads, d_k, d_v, use_short_conv, device, dtype,
        )),
    ]

    headers = [
        "SeqLen (L)", "Tokens (B*L)", "Method / Implementation",
        "Fwd (ms)", "Fwd+Bwd (ms)", "GFLOPs", "TFLOP/s", "Peak VRAM (MB)", "Speedup",
    ]
    rows: List[List[Any]] = []
    records: List[Dict[str, Any]] = []

    B = batch_size
    for L in tqdm(seq_lens, desc="Training (Part A)", unit="len"):
        total_tokens = B * L
        print(f"\n---> Benchmarking B={B}, L={L} (Total Tokens: {total_tokens:,}) ...", flush=True)
        gflops = compute_deltanet_training_flops(
            batch_size=B, seq_len=L, d_model=d_model,
            num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
            d_k=d_k, d_v=d_v, chunk_size=chunk_size,
        ) / 1e9

        x_batched = torch.randn(B, L, d_model, device=device, dtype=dtype, requires_grad=True)
        x_packed = x_batched.detach().reshape(total_tokens, d_model).clone().requires_grad_(True)
        cu_seqlens = torch.arange(0, (B + 1) * L, L, dtype=torch.int32, device=device)

        def _measure_pair(label: str, fn_fwd, fn_step):
            rf = measure_performance(fn_fwd, num_warmup, num_iters, device)
            rs = measure_performance(fn_step, num_warmup, num_iters, device)
            return rf, rs

        # 1. PyTorch Reference
        def _ref_fwd():
            with torch.no_grad():
                return our_block(x_batched, use_kernel=False)

        def _ref_step():
            our_block.zero_grad(set_to_none=True)
            x_batched.grad = None
            out, _ = our_block(x_batched, use_kernel=False)
            out.sum().backward()

        rf_ref, rs_ref = _measure_pair("PyTorch Reference", _ref_fwd, _ref_step)
        ref_ms = rs_ref["latency_ms"] if rs_ref["status"] == "OK" else 0.0

        def _append_result(label: str, rf: Dict[str, Any], rs: Dict[str, Any], bold: bool = False):
            status = rs["status"]
            if status == "OK":
                fwd_ms = rf["latency_ms"] if rf["status"] == "OK" else 0.0
                step_ms = rs["latency_ms"]
                tflops = gflops / step_ms if step_ms > 0 else 0.0
                sp = f"{ref_ms / step_ms:.2f}x" if (ref_ms > 0 and step_ms > 0) else "-"
                if label == "PyTorch Reference (Unfused)":
                    sp = "1.00x"
                peak_mb = rs["peak_alloc_mb"]
                print(f"  ✓ {label}: Fwd={fwd_ms:.2f}ms | Fwd+Bwd={step_ms:.2f}ms | {tflops:.2f} TFLOP/s", flush=True)
                wrap = (lambda s: f"**{s}**") if bold else (lambda s: s)
                rows.append([
                    L, f"{total_tokens:,}", wrap(label), wrap(f"{fwd_ms:.2f}"),
                    wrap(f"{step_ms:.2f}"), wrap(f"{gflops:.1f}"),
                    wrap(f"{tflops:.2f}"), wrap(f"{peak_mb:.1f}"), wrap(sp),
                ])
                records.append({
                    "seq_len": L, "total_tokens": total_tokens, "method": label,
                    "fwd_ms": fwd_ms, "fwd_bwd_ms": step_ms, "tflops": tflops,
                    "peak_mb": peak_mb, "status": "OK",
                })
            else:
                print(f"  ✗ {label} failed at L={L}: {rs.get('error_msg')}", flush=True)
                rows.append([L, f"{total_tokens:,}", label, "-", status, "-", "-", "-", "-"])
                records.append({
                    "seq_len": L, "total_tokens": total_tokens, "method": label,
                    "fwd_ms": None, "fwd_bwd_ms": None, "tflops": None,
                    "peak_mb": None, "status": status,
                })

        _append_result("PyTorch Reference (Unfused)", rf_ref, rs_ref)

        # 2. External baselines
        for bname, bmod in baselines:
            if bmod is None:
                continue

            def _b_fwd(m=bmod):
                with torch.no_grad():
                    return m(x_batched)

            def _b_step(m=bmod):
                m.zero_grad(set_to_none=True)
                x_batched.grad = None
                out, *_ = m(x_batched)
                out.sum().backward()

            rf_b, rs_b = _measure_pair(bname, _b_fwd, _b_step)
            _append_result(bname, rf_b, rs_b)

        # 3. DeltaNet v2 (Ours - Eager)
        def _ours_fwd():
            with torch.no_grad():
                return our_block(x_packed, cu_seqlens=cu_seqlens, use_kernel=True)

        def _ours_step():
            our_block.zero_grad(set_to_none=True)
            x_packed.grad = None
            out, _ = our_block(x_packed, cu_seqlens=cu_seqlens, use_kernel=True)
            out.sum().backward()

        rf_o, rs_o = _measure_pair("DeltaNet v2 (Ours)", _ours_fwd, _ours_step)
        _append_result("DeltaNet v2 (Ours)", rf_o, rs_o, bold=True)

        # 4. DeltaNet v2 (Ours - torch.compile)
        if our_compiled is not None:
            def _ours_c_fwd():
                with torch.no_grad():
                    return our_compiled(x_packed, cu_seqlens=cu_seqlens, use_kernel=True)

            def _ours_c_step():
                our_block.zero_grad(set_to_none=True)
                x_packed.grad = None
                out, _ = our_compiled(x_packed, cu_seqlens=cu_seqlens, use_kernel=True)
                out.sum().backward()

            rf_oc, rs_oc = _measure_pair("DeltaNet v2 (Compiled)", _ours_c_fwd, _ours_c_step)
            _append_result("DeltaNet v2 (Compiled)", rf_oc, rs_oc, bold=True)

        del x_batched, x_packed, cu_seqlens
        gc.collect()
        torch.cuda.empty_cache()

    table_md = format_markdown_table(headers, rows)
    print("\n" + "=" * 78)
    print("PART A: TRAINING BENCHMARK RESULTS (PURE TOKEN MIXER)")
    print("=" * 78)
    print(table_md)
    print("=" * 78 + "\n")
    return records, table_md


def run_e2e_training_benchmark(
    d_model: int = 768,
    num_q_heads: int = 8,
    num_kv_heads: int = 4,
    d_k: int = 64,
    d_v: int = 64,
    chunk_size: int = 64,
    seq_len: int = 1024,
    batch_sizes: Optional[List[int]] = None,
    num_warmup: int = 3,
    num_iters: int = 8,
    dtype: torch.dtype = torch.float16,
    device: Optional[torch.device] = None,
):
    """Part B: End-to-End DeltaBlock (with MLP) training vs torch.compile."""
    device = device or get_device()
    batch_sizes = batch_sizes or [4, 8, 16]
    hidden_dim = d_model * 4

    print_section("PART B: END-TO-END DELTABLOCK TRAINING (WITH MLP + torch.compile)")
    print(
        f"Config: d_model={d_model}, hidden_dim={hidden_dim}, H_q={num_q_heads}, "
        f"H_kv={num_kv_heads}, L={seq_len}, batches={batch_sizes}"
    )

    headers = ["Batch (B)", "Mode", "Step (ms)", "Throughput (tok/s)", "Peak VRAM (GB)", "Speedup"]
    rows: List[List[Any]] = []
    records: List[Dict[str, Any]] = []

    modes = [
        ("torch-eager", False, False),
        ("torch-compiled", False, True),
        ("kernel-eager", True, False),
        ("kernel-compiled", True, True),
    ]

    for B in batch_sizes:
        total_tokens = B * seq_len
        cu = torch.arange(0, (B + 1) * seq_len, seq_len, dtype=torch.int32, device=device)
        eager_ms = 0.0
        for mname, use_k, do_comp in modes:
            gc.collect()
            torch.cuda.empty_cache()
            blk = DeltaBlock(
                d_model=d_model, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
                d_k=d_k, d_v=d_v, chunk_size=chunk_size, hidden_dim=hidden_dim,
            ).to(device=device, dtype=dtype).train()
            fn = _compile_or_none(blk, mname) if do_comp else blk
            if fn is None:
                rows.append([B, mname, "COMPILE_ERR", "-", "-", "-"])
                records.append({"B": B, "mode": mname, "step_ms": None, "tok_s": None, "peak_gb": None, "status": "COMPILE_ERR"})
                del blk
                continue

            x = (
                torch.randn(total_tokens, d_model, device=device, dtype=dtype, requires_grad=True)
                if use_k
                else torch.randn(B, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
            )

            def _step():
                blk.zero_grad(set_to_none=True)
                x.grad = None
                out, _ = fn(x, cu_seqlens=cu, use_kernel=True) if use_k else fn(x, use_kernel=False)
                out.sum().backward()

            res = measure_performance(_step, num_warmup, num_iters, device)
            if res["status"] == "OK":
                lat = res["latency_ms"]
                tok_s = (total_tokens * 1000.0) / lat if lat > 0 else 0.0
                peak_gb = res["peak_alloc_mb"] / 1024.0
                if mname == "torch-eager":
                    eager_ms = lat
                sp = f"{eager_ms / lat:.2f}x" if (eager_ms > 0 and lat > 0) else "-"
                print(f"  [B={B}] {mname:<15}: {lat:6.1f} ms | {tok_s:8,.0f} tok/s | {peak_gb:.2f} GB ({sp})")
                bold = (mname == "kernel-compiled")
                w = (lambda s: f"**{s}**") if bold else (lambda s: s)
                rows.append([B, w(mname), w(f"{lat:.1f}"), w(f"{tok_s:,.0f}"), w(f"{peak_gb:.2f}"), w(sp)])
                records.append({"B": B, "mode": mname, "step_ms": lat, "tok_s": tok_s, "peak_gb": peak_gb, "status": "OK"})
            else:
                print(f"  [B={B}] {mname:<15}: {res['status']}")
                rows.append([B, mname, res["status"], "-", "-", "-"])
                records.append({"B": B, "mode": mname, "step_ms": None, "tok_s": None, "peak_gb": None, "status": res["status"]})

            del x, blk, fn
            gc.collect()
            torch.cuda.empty_cache()

    table_md = format_markdown_table(headers, rows)
    print("\n" + "=" * 78)
    print(f"PART B: E2E DELTABLOCK TRAINING RESULTS (L = {seq_len})")
    print("=" * 78)
    print(table_md)
    print("=" * 78 + "\n")
    return records, table_md


def _save_training_plot(
    part_a: List[Dict[str, Any]], part_b: Optional[List[Dict[str, Any]]], save_path: str
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[!] matplotlib/numpy not installed; skipping training plot.")
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
    seq_lens = sorted({r["seq_len"] for r in part_a})

    ncols = 2
    nrows = 2 if part_b else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5.2 * nrows))
    axes_2d = axes.reshape(nrows, ncols)
    fig.suptitle("DeltaNet v2 Training Benchmark (NVIDIA Turing SM75, FP16)", fontsize=13, fontweight="bold")

    x = np.arange(len(seq_lens))
    n_m = max(len(active_methods), 1)
    width = 0.78 / n_m

    # Panel 1: Fwd+Bwd Latency vs Open-Source Baselines
    ax1 = axes_2d[0, 0]
    ax2 = axes_2d[0, 1]
    for idx, (m, color) in enumerate(active_methods):
        offs = x + (idx - (n_m - 1) / 2.0) * width
        lats = [next((r["fwd_bwd_ms"] or 0.0 for r in part_a if r["seq_len"] == L and r["method"] == m), 0.0) for L in seq_lens]
        mems = [next((r["peak_mb"] or 0.0 for r in part_a if r["seq_len"] == L and r["method"] == m), 0.0) for L in seq_lens]
        ax1.bar(offs, lats, width=width, label=m, color=color)
        ax2.bar(offs, mems, width=width, label=m, color=color)

    ax1.set_title("Part A: Token-Mixer Training Latency (Fwd+Bwd, Lower is Better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"L={L}" for L in seq_lens])
    ax1.set_ylabel("Step Time (ms)")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_title("Part A: Token-Mixer Peak VRAM (Lower is Better)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"L={L}" for L in seq_lens])
    ax2.set_ylabel("Peak Allocated VRAM (MB)")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(fontsize=8)

    # Panel 3 & 4: Part B E2E DeltaBlock + torch.compile
    if part_b:
        batches = sorted({r["B"] for r in part_b})
        xb = np.arange(len(batches))
        e2e_modes = [
            ("torch-eager", "#6c757d"),
            ("torch-compiled", "#3b82f6"),
            ("kernel-eager", "#f59e0b"),
            ("kernel-compiled", "#10b981"),
        ]
        wb = 0.19
        ax3 = axes_2d[1, 0]
        ax4 = axes_2d[1, 1]
        for idx, (m, color) in enumerate(e2e_modes):
            offs = xb + (idx - 1.5) * wb
            toks = [next((r["tok_s"] or 0.0 for r in part_b if r["B"] == b and r["mode"] == m), 0.0) for b in batches]
            gbs = [next((r["peak_gb"] or 0.0 for r in part_b if r["B"] == b and r["mode"] == m), 0.0) for b in batches]
            ax3.bar(offs, toks, width=wb, label=m, color=color)
            ax4.bar(offs, gbs, width=wb, label=m, color=color)

        ax3.set_title("Part B: E2E DeltaBlock Training Throughput (Higher is Better)")
        ax3.set_xticks(xb)
        ax3.set_xticklabels([f"B={b}" for b in batches])
        ax3.set_ylabel("Throughput (tokens/s)")
        ax3.grid(True, axis="y", alpha=0.3)
        ax3.legend(fontsize=8)

        ax4.set_title("Part B: E2E DeltaBlock Peak VRAM (Lower is Better)")
        ax4.set_xticks(xb)
        ax4.set_xticklabels([f"B={b}" for b in batches])
        ax4.set_ylabel("Peak Allocated VRAM (GB)")
        ax4.grid(True, axis="y", alpha=0.3)
        ax4.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved training benchmark plot -> {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="DeltaNet v2 Training Step Benchmark")
    parser.add_argument("--dim", type=int, default=768, help="Hidden dimension (d_model)")
    parser.add_argument("--q-heads", type=int, default=8, help="Number of Query heads")
    parser.add_argument("--kv-heads", type=int, default=8, help="Number of Key/Value heads")
    parser.add_argument("--d-k", type=int, default=64, help="Head dimension for Keys")
    parser.add_argument("--d-v", type=int, default=64, help="Head dimension for Values")
    parser.add_argument("--chunk-size", type=int, default=64, help="Chunk size for WY representation")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for Part A (B)")
    parser.add_argument("--seq-lens", type=str, default="512,1024,2048,4096", help="Comma-separated sequence lengths")
    parser.add_argument("--e2e-batches", type=str, default="4,8,16", help="Comma-separated batch sizes for Part B E2E")
    parser.add_argument("--e2e-seq-len", type=int, default=1024, help="Sequence length for Part B E2E (default: 1024)")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations (default: 3)")
    parser.add_argument("--iters", type=int, default=10, help="Number of timed iterations (default: 10)")
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

    l_list = [int(s.strip()) for s in args.seq_lens.split(",") if s.strip()]
    e2e_b_list = [int(b.strip()) for b in args.e2e_batches.split(",") if b.strip()]
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    part_a_rec, part_a_md = run_training_benchmark(
        d_model=args.dim, num_q_heads=args.q_heads, num_kv_heads=args.kv_heads,
        d_k=args.d_k, d_v=args.d_v, chunk_size=args.chunk_size,
        batch_size=args.batch_size, seq_lens=l_list, num_warmup=args.warmup,
        num_iters=args.iters, use_short_conv=args.use_conv,
        include_compile=not args.no_compile, dtype=dtype,
    )

    part_b_rec, part_b_md = None, ""
    if not args.skip_e2e:
        e2e_hkv = 4 if (args.q_heads % 4 == 0) else args.kv_heads
        part_b_rec, part_b_md = run_e2e_training_benchmark(
            d_model=args.dim, num_q_heads=args.q_heads, num_kv_heads=e2e_hkv,
            d_k=args.d_k, d_v=args.d_v, chunk_size=args.chunk_size,
            seq_len=args.e2e_seq_len, batch_sizes=e2e_b_list,
            num_warmup=args.warmup, num_iters=max(5, args.iters), dtype=dtype,
        )

    if args.save_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_md)), exist_ok=True)
        with open(args.save_md, "w", encoding="utf-8") as f:
            f.write("### Part A: Token-Mixer Baselines\n\n" + part_a_md + "\n")
            if part_b_md:
                f.write("\n### Part B: E2E DeltaBlock (`torch.compile`)\n\n" + part_b_md + "\n")

    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump({"part_a": part_a_rec, "part_b": part_b_rec}, f, indent=2)

    if args.save_plot:
        _save_training_plot(part_a_rec, part_b_rec, args.save_plot)
