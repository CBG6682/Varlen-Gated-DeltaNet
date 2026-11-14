"""
Benchmarking and profiling utilities for DeltaNet v2 and baseline linear attention models.
Provides precise CUDA event timing, peak VRAM memory tracking, and dynamic baseline loaders.
"""

import gc
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


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
    num_warmup: int = 10,
    num_iters: int = 30,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Measure average latency (ms) and peak VRAM usage (MB) using CUDA events and memory tracking.

    Args:
        fn: Zero-argument callable executing the target forward or forward+backward workload.
        num_warmup: Number of untimed warmup iterations to initialize JIT, kernels, and caches.
        num_iters: Number of timed iterations to compute average latency.
        device: Target CUDA device.

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
        for _ in range(num_warmup):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # Reset peak stats after warmup to measure steady-state operational peak
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # 2. Timing
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(num_iters):
                fn()
            end_event.record()
            torch.cuda.synchronize(device)

            total_ms = start_event.elapsed_time(end_event)
            latency_ms = total_ms / max(num_iters, 1)
            peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_resv = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        else:
            import time
            t0 = time.perf_counter()
            for _ in range(num_iters):
                fn()
            total_ms = (time.perf_counter() - t0) * 1000.0
            latency_ms = total_ms / max(num_iters, 1)
            peak_alloc = 0.0
            peak_resv = 0.0

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

def try_import_fla_gdn2():
    """
    Attempt to import GatedDeltaNet2 (official NVIDIA GDN2 architecture in FLA).
    Returns class or None.
    """
    try:
        from fla.layers import GatedDeltaNet2
        return GatedDeltaNet2
    except ImportError:
        pass
    try:
        from fla.layers.gdn2 import GatedDeltaNet2
        return GatedDeltaNet2
    except ImportError:
        pass
    return None


def try_import_fla_gdn():
    """
    Attempt to import GatedDeltaNet (v1 architecture in FLA).
    Returns class or None.
    """
    try:
        from fla.layers import GatedDeltaNet
        return GatedDeltaNet
    except ImportError:
        pass
    try:
        from fla.layers.gated_deltanet import GatedDeltaNet
        return GatedDeltaNet
    except ImportError:
        pass
    return None


def try_import_fla_cache():
    """Attempt to import Cache object from FLA for stateful autoregressive generation."""
    try:
        from fla.models.utils import Cache
        return Cache
    except ImportError:
        pass
    return None


def check_baseline_availability() -> Dict[str, bool]:
    """Check availability of external baseline packages."""
    return {
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
    print(f"[-] DeltaNet v2 (Ours - SM75 CuTe)   : INSTALLED")
    print(f"[-] DeltaNet v2 (PyTorch Reference)  : INSTALLED")
    print(f"[-] FLA GatedDeltaNet-2 (NVLabs)     : {'INSTALLED' if avail['fla_gdn2'] else 'NOT FOUND'}")
    print(f"[-] FLA GatedDeltaNet (v1)           : {'INSTALLED' if avail['fla_gdn'] else 'NOT FOUND'}")

    if not (avail['fla_gdn2'] or avail['fla_gdn']):
        print("\n[!] Notice: Baselines from Flash Linear Attention are not installed.")
        print("    To benchmark against NVLabs GatedDeltaNet-2 and FLA, run:")
        print("      pip install flash-linear-attention")
        print("      pip install git+https://github.com/nvlabs/gateddeltanet-2.git --no-deps")
    print("=" * 72 + "\n")


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
