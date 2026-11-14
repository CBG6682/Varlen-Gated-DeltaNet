"""
Benchmarking and profiling utilities for DeltaNet v2 and baseline linear attention models.
Provides precise CUDA event timing, peak VRAM memory tracking, and dynamic baseline loaders.
"""

import gc
import os
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def get_device() -> torch.device:
    """Return default CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def print_hardware_info(device: Optional[torch.device] = None) -> None:
    """Print CUDA device name, compute capability, and total memory."""
    if device is None:
        device = get_device()
    print("=" * 72)
    print("HARDWARE & RUNTIME ENVIRONMENT")
    print("=" * 72)
    print(f"PyTorch Version  : {torch.__version__}")
    print(f"CUDA Available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        dev_idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(dev_idx)
        print(f"Device Name      : {props.name}")
        print(f"Compute Arch     : SM{props.major}.{props.minor}")
        print(f"Total VRAM       : {props.total_memory / (1024 ** 3):.2f} GB")
        print(f"Multi-Processors : {props.multi_processor_count}")
    else:
        print("Running on CPU. CUDA acceleration not available.")
    print("=" * 72)


def measure_performance(
    fn: Callable[[], Any],
    num_warmup: int = 3,
    num_iters: int = 10,
    device: Optional[torch.device] = None,
    pbar: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Measure average latency (ms) and peak VRAM usage (MB) using CUDA events and memory tracking.

    Args:
        fn: Zero-argument callable executing the target forward or forward+backward workload.
        num_warmup: Number of untimed warmup iterations to initialize JIT, kernels, and caches.
        num_iters: Number of timed iterations to compute average latency.
        device: Target CUDA device.
        pbar: Optional tqdm progress bar to advance upon completion of iterations.

    Returns:
        dict with keys:
            'latency_ms': float (average ms per iteration)
            'peak_alloc_mb': float (maximum memory allocated in MB)
            'peak_resv_mb': float (maximum memory reserved by caching allocator in MB)
            'status': str ('OK', 'OOM', or 'FAILED')
            'error_msg': Optional[str]
    """
    if device is None:
        device = get_device()

    # Clear memory prior to measurement
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    try:
        # 1. Warm-up
        if pbar is not None and num_warmup > 0:
            pbar.set_postfix_str("Compiling / Warmup...")
            pbar.refresh()

        for i in range(num_warmup):
            fn()
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix_str(f"Warmup {i + 1}/{num_warmup}")

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # Reset peak stats after warmup to measure steady-state operational peak
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # 2. Timing
        if pbar is not None and num_iters > 0:
            pbar.set_postfix_str("Benchmarking...")
            pbar.refresh()

        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for it in range(num_iters):
                fn()
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix_str(f"Timing {it + 1}/{num_iters}")
            end_event.record()
            torch.cuda.synchronize(device)

            total_ms = start_event.elapsed_time(end_event)
            latency_ms = total_ms / max(num_iters, 1)
            peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_resv = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        else:
            import time
            t0 = time.perf_counter()
            for it in range(num_iters):
                fn()
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix_str(f"Timing {it + 1}/{num_iters}")
            total_ms = (time.perf_counter() - t0) * 1000.0
            latency_ms = total_ms / max(num_iters, 1)
            peak_alloc = 0.0
            peak_resv = 0.0

        if pbar is not None:
            pbar.set_postfix_str(f"{latency_ms:.2f} ms")
            pbar.refresh()

        return {
            "latency_ms": latency_ms,
            "peak_alloc_mb": peak_alloc,
            "peak_resv_mb": peak_resv,
            "status": "OK",
            "error_msg": None,
        }

    except torch.cuda.OutOfMemoryError as e:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if pbar is not None:
            remaining = getattr(pbar, "total", 0) - getattr(pbar, "n", 0)
            if remaining > 0:
                pbar.update(remaining)
            pbar.set_postfix_str("OOM")
            pbar.refresh()
        return {
            "latency_ms": float("nan"),
            "peak_alloc_mb": float("nan"),
            "peak_resv_mb": float("nan"),
            "status": "OOM",
            "error_msg": "CUDA Out Of Memory",
        }
    except Exception as e:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if pbar is not None:
            remaining = getattr(pbar, "total", 0) - getattr(pbar, "n", 0)
            if remaining > 0:
                pbar.update(remaining)
            pbar.set_postfix_str("FAILED")
            pbar.refresh()
        return {
            "latency_ms": float("nan"),
            "peak_alloc_mb": float("nan"),
            "peak_resv_mb": float("nan"),
            "status": "FAILED",
            "error_msg": str(e),
        }


# =========================================================================
# Dynamic Baseline Model Loaders
# =========================================================================

def try_import_nvlabs_gdn2():
    """
    Attempt to import GatedDeltaNet2 from official NVIDIA NVlabs repository.
    Searches installed lit_gpt package as well as clone locations.
    Bypasses lit_gpt/__init__.py to prevent lightning/flash_attn dependency failures.
    """
    if "lit_gpt.gdn2" in sys.modules:
        return getattr(sys.modules["lit_gpt.gdn2"], "GatedDeltaNet2", None)

    candidates = [
        "/content/gateddeltanet-2",
        "/content/GatedDeltaNet-2",
        "/tmp/gateddeltanet-2",
        "/tmp/GatedDeltaNet-2",
        os.path.join(ROOT_DIR, "gateddeltanet-2"),
        os.path.join(ROOT_DIR, "..", "gateddeltanet-2"),
    ]

    import types

    for cand in candidates:
        lit_gpt_dir = os.path.join(cand, "lit_gpt")
        gdn2_file = os.path.join(lit_gpt_dir, "gdn2.py")
        if os.path.isdir(lit_gpt_dir) and os.path.isfile(gdn2_file):
            if cand not in sys.path:
                sys.path.insert(0, cand)

            # Pre-register dummy lit_gpt package to prevent __init__.py execution
            # (which requires lightning>=2.1.0 and flash_attn)
            if "lit_gpt" not in sys.modules:
                pkg = types.ModuleType("lit_gpt")
                pkg.__path__ = [lit_gpt_dir]
                pkg.__file__ = os.path.join(lit_gpt_dir, "__init__.py")
                sys.modules["lit_gpt"] = pkg

            # Register dummy lit_gpt.gdn2_ops package for relative imports
            ops_dir = os.path.join(lit_gpt_dir, "gdn2_ops")
            if os.path.isdir(ops_dir) and "lit_gpt.gdn2_ops" not in sys.modules:
                ops_pkg = types.ModuleType("lit_gpt.gdn2_ops")
                ops_pkg.__path__ = [ops_dir]
                ops_pkg.__file__ = os.path.join(ops_dir, "__init__.py")
                sys.modules["lit_gpt.gdn2_ops"] = ops_pkg

            try:
                import lit_gpt.gdn2 as gdn2_mod
                cls = getattr(gdn2_mod, "GatedDeltaNet2", None)
                if cls is not None:
                    return cls
            except Exception:
                pass

    try:
        from lit_gpt.gdn2 import GatedDeltaNet2
        return GatedDeltaNet2
    except ImportError:
        pass

    return None


def try_import_fla_gdn2():
    """
    Attempt to import GatedDeltaNet2 from Flash Linear Attention (FLA) library.
    Returns class or None.
    """
    try:
        from fla.layers.gdn2 import GatedDeltaNet2
        return GatedDeltaNet2
    except ImportError:
        pass
    try:
        from fla.layers import GatedDeltaNet2
        return GatedDeltaNet2
    except ImportError:
        pass
    return None


def try_import_fla_gdn():
    """
    Attempt to import GatedDeltaNet (v1 architecture) from Flash Linear Attention.
    Returns class or None.
    """
    try:
        from fla.layers.gated_deltanet import GatedDeltaNet
        return GatedDeltaNet
    except ImportError:
        pass
    try:
        from fla.layers import GatedDeltaNet
        return GatedDeltaNet
    except ImportError:
        pass
    return None


def try_import_fla_cache():
    """Attempt to import Cache object from FLA for stateful autoregressive generation."""
    try:
        from fla.models.cache_utils import Cache
        return Cache
    except ImportError:
        pass
    try:
        from fla.models.utils import Cache
        return Cache
    except ImportError:
        pass
    return None


def check_baseline_availability() -> Dict[str, bool]:
    """Check availability of external baseline packages."""
    return {
        "nvlabs_gdn2": try_import_nvlabs_gdn2() is not None,
        "fla_gdn2": try_import_fla_gdn2() is not None,
        "fla_gdn": try_import_fla_gdn() is not None,
        "fla_cache": try_import_fla_cache() is not None,
    }


def print_baseline_status() -> None:
    """Print detected baselines and installation commands for missing packages."""
    avail = check_baseline_availability()
    print("\n" + "=" * 72)
    print("BASELINE AVAILABILITY")
    print("=" * 72)
    print(f"[-] DeltaNet v2 (Ours)                 : INSTALLED")
    print(f"[-] DeltaNet v2 (PyTorch Reference)    : INSTALLED")
    print(f"[-] NVLabs GatedDeltaNet-2 (Official)  : {'INSTALLED' if avail['nvlabs_gdn2'] else 'NOT FOUND'}")
    print(f"[-] FLA GatedDeltaNet-2 (FLA Library)  : {'INSTALLED' if avail['fla_gdn2'] else 'NOT FOUND'}")
    print(f"[-] FLA GatedDeltaNet (v1)             : {'INSTALLED' if avail['fla_gdn'] else 'NOT FOUND'}")

    if not (avail['nvlabs_gdn2'] and avail['fla_gdn2']):
        print("\n[!] Notice: External baselines can be installed via:")
        print("    rm -rf /content/gateddeltanet-2")
        print("    git clone --depth 1 https://github.com/NVlabs/GatedDeltaNet-2.git /content/gateddeltanet-2")
        print("    echo \"\" > /content/gateddeltanet-2/lit_gpt/__init__.py")
        print("    pip install git+https://github.com/sustcsonglin/flash-linear-attention.git --no-build-isolation")
    print("=" * 72 + "\n")


# =========================================================================
# FLOP Counting Utilities
# =========================================================================

def compute_deltanet_training_flops(
    batch_size: int,
    seq_len: int,
    d_model: int = 768,
    num_q_heads: int = 8,
    num_kv_heads: int = 8,
    d_k: int = 64,
    d_v: int = 64,
    chunk_size: int = 64,
) -> float:
    """
    Theoretical floating-point operations (FLOPs) for a full Forward + Backward
    training step of the pure Gated DeltaNet token mixer (excluding MLP/FFW).

    FLOPs_total = 3 * FLOPs_fwd (Forward + Backward gradients w.r.t inputs & weights).
    """
    T = batch_size * seq_len
    num_chunks = max(seq_len // chunk_size, 1)

    # Input projection: [T, d_model] -> [T, d_proj_out]
    # q, k, v, w, b, alpha, gamma
    d_proj_out = num_q_heads * d_k + num_kv_heads * (d_k + 2 * d_v + d_k + 2)

    fwd_proj = 2.0 * T * d_model * d_proj_out
    fwd_gate = 2.0 * T * d_model * (num_q_heads * d_v)
    fwd_out_proj = 2.0 * T * (num_q_heads * d_v) * d_model

    # Chunk WY recurrence
    fwd_wy = 2.0 * T * num_kv_heads * (d_k * d_k + d_v * d_k)
    fwd_scan = 2.0 * batch_size * num_chunks * num_kv_heads * d_v * d_k * d_k
    fwd_intra_attn = 2.0 * T * num_kv_heads * chunk_size * (d_k + d_v)
    fwd_inter_attn = 2.0 * T * num_q_heads * d_k * d_v

    fwd_elementwise = 10.0 * T * d_model
    fwd_flops = (
        fwd_proj + fwd_gate + fwd_out_proj
        + fwd_wy + fwd_scan + fwd_intra_attn + fwd_inter_attn
        + fwd_elementwise
    )
    return 3.0 * fwd_flops


# =========================================================================
# Table Formatting Utilities
# =========================================================================

def format_markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Format headers and row data into a GitHub Flavored Markdown table string."""
    str_rows = [[str(cell) for cell in row] for row in rows]
    col_widths = [len(h) for h in headers]
    for row in str_rows:
        for idx, cell in enumerate(row):
            if idx < len(col_widths):
                col_widths[idx] = max(col_widths[idx], len(cell))
            else:
                col_widths.append(len(cell))

    def make_row(cells):
        padded = [c.ljust(col_widths[i]) for i, c in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    def make_sep():
        bars = ["-" * w for w in col_widths]
        return "| " + " | ".join(bars) + " |"

    lines = [make_row(headers), make_sep()]
    for row in str_rows:
        lines.append(make_row(row))
    return "\n".join(lines)


def print_section(title: str) -> None:
    """Print an eye-catching section divider."""
    print("\n" + "#" * 72)
    print(f"# {title}")
    print("#" * 72 + "\n")
