"""
Benchmarking and profiling utilities for DeltaNet v2 and baseline linear attention models.
Provides precise CUDA event timing, peak VRAM memory tracking, and dynamic baseline loaders.
"""

import gc
import os
import random
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


def compute_deltanet_inference_flops(
    batch_size: int,
    d_model: int = 768,
    num_q_heads: int = 8,
    num_kv_heads: int = 8,
    d_k: int = 64,
    d_v: int = 64,
) -> float:
    """
    Theoretical floating-point operations (FLOPs) for single-step autoregressive decode (L = 1)
    of the pure Gated DeltaNet token mixer block (excluding MLP/FFW).
    """
    B = batch_size
    Hq = num_q_heads
    H = num_kv_heads
    G = max(Hq // H, 1)
    d_out_total = Hq * d_v

    # Projections:
    # W_proj: q (Hq*d_k), k (H*d_k), v (H*d_v), b (H*d_k), w (H*d_v), alpha (H), gamma (H)
    d_proj_out = Hq * d_k + H * (d_k + 2 * d_v + d_k + 2)
    flops_proj = 2.0 * B * d_model * d_proj_out
    flops_gate = 2.0 * B * d_model * d_out_total
    flops_out = 2.0 * B * d_out_total * d_model

    # Recurrence update per step:
    # S * q^T: [B, H, G, d_v]
    flops_q_S = 2.0 * B * Hq * d_v * d_k
    # S * u: [B, H, d_v]
    flops_S_u = 2.0 * B * H * d_v * d_k
    # q * k^T: [B, H, G]
    flops_q_k = 2.0 * B * Hq * d_k
    # attn * v_new: [B, H, G, d_v]
    flops_attn_v = 2.0 * B * Hq * d_v
    # State update: alpha * S + (v_new - S_u) * k^T
    flops_state_update = 2.0 * B * H * d_v * d_k + 2.0 * B * H * d_v * d_k

    # Elementwise ops (RMSNorm, silu, activations)
    flops_elem = 10.0 * B * d_model

    return (
        flops_proj + flops_gate + flops_out
        + flops_q_S + flops_S_u + flops_q_k + flops_attn_v + flops_state_update
        + flops_elem
    )


def compute_deltanet_inference_memory_bytes(
    batch_size: int,
    d_model: int = 768,
    num_q_heads: int = 8,
    num_kv_heads: int = 8,
    d_k: int = 64,
    d_v: int = 64,
    dtype_bytes: int = 2,
    state_bytes: int = 2,
) -> float:
    """
    Theoretical minimum bytes transferred from DRAM (weights, activations, recurrent state)
    during a single autoregressive decode step (L = 1).
    """
    B = batch_size
    Hq = num_q_heads
    H = num_kv_heads
    d_out_total = Hq * d_v
    d_proj_out = Hq * d_k + H * (d_k + 2 * d_v + d_k + 2)

    # 1. Static weights (read once per step across all batch items)
    # W_proj, W_gate, W_out, rms_norm
    weights_elems = (
        d_model * d_proj_out       # W_proj
        + d_model * d_out_total    # W_gate
        + d_out_total * d_model    # W_out
        + d_out_total              # RMSNorm weight
    )
    weights_bytes = weights_elems * dtype_bytes

    # 2. Input and output activations
    # Read x, write y
    act_bytes = 2.0 * B * d_model * dtype_bytes

    # 3. Recurrent state memory traffic (KV cache equivalent)
    # Read past_states [B, H, d_v, d_k], write next_states [B, H, d_v, d_k]
    state_elems = 2.0 * B * H * d_v * d_k
    state_traffic_bytes = state_elems * state_bytes

    return float(weights_bytes + act_bytes + state_traffic_bytes)


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


# ---------------------------------------------------------------------------
# Correctness & Varlen Helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42, disable_tf32: bool = True) -> None:
    """Seed Python, PyTorch, and CUDA RNGs for reproducible correctness tests."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if disable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False


def make_cu_seqlens(
    lengths: List[int], device: torch.device
) -> Tuple[torch.Tensor, int]:
    """Build cumulative sequence lengths tensor (int32) and max sequence length."""
    lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(lens, dim=0).to(torch.int32)
    max_len = int(lens.max().item()) if len(lengths) > 0 else 0
    return cu_seqlens.contiguous(), max_len


def make_varlen_inputs(
    seqlens: List[int],
    d_model: int,
    scale: float = 1.0,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate padded + packed varlen inputs, padding mask, and cu_seqlens."""
    B = len(seqlens)
    cu_seqlens, Lmax = make_cu_seqlens(seqlens, device)
    lens = torch.tensor(seqlens, dtype=torch.int32, device=device)
    mask = torch.arange(Lmax, device=device).unsqueeze(0) < lens.unsqueeze(1)
    x_pad = (torch.randn(B, Lmax, d_model, device=device, dtype=torch.float32) * scale).to(dtype)
    x_pad = (x_pad * mask.unsqueeze(-1)).contiguous()
    x_var = x_pad[mask].contiguous()
    return x_pad, x_var, mask.contiguous(), cu_seqlens


def compute_tensor_metrics(
    actual: torch.Tensor, expected: torch.Tensor
) -> Tuple[float, float, float]:
    """Return (rel_l2, cos_sim, max_abs) in float64."""
    a = actual.detach().double().reshape(-1)
    b = expected.detach().double().reshape(-1)
    if a.numel() == 0:
        return 0.0, 1.0, 0.0
    diff = a - b
    diff_norm = diff.norm().item()
    b_norm = b.norm().item()
    rel_l2 = diff_norm / (b_norm + 1e-12)
    max_abs = diff.abs().max().item()
    a_norm = a.norm().item()
    if a_norm == 0.0 and b_norm == 0.0:
        cos_sim = 1.0
    elif a_norm == 0.0 or b_norm == 0.0:
        cos_sim = 0.0
    else:
        cos_sim = float(torch.dot(a, b).item() / (a_norm * b_norm + 1e-12))
    return rel_l2, cos_sim, max_abs


def worst_group_rel(
    actual: torch.Tensor, expected: torch.Tensor, lead_dims: int
) -> float:
    """Compute max per-group L2 error normalized by mean group truth norm.

    Dividing by the mean truth norm avoids exploding relative errors when an
    individual token or (seq, head) slice happens to have near-zero norm.
    """
    if actual.numel() == 0:
        return 0.0
    a = actual.detach().double().flatten(0, lead_dims - 1).flatten(1)
    b = expected.detach().double().flatten(0, lead_dims - 1).flatten(1)
    d = (a - b).norm(dim=1)
    nb = b.norm(dim=1)
    return float((d.max() / (nb.mean() + 1e-30)).item())


def check_allclose(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    verbose: bool = True,
) -> Tuple[bool, float, float]:
    """Compare tensors in float64 against (atol, rtol) and finite check."""
    a = actual.detach().double()
    b = expected.detach().double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        if verbose:
            print(f"    [FAIL] {name:<26} non-finite values detected!")
        return False, float("inf"), float("inf")
    if a.numel() == 0:
        if verbose:
            print(f"    [PASS] {name:<26} empty tensor match")
        return True, 0.0, 0.0
    diff = (a - b).abs()
    max_abs = float(diff.max().item())
    max_rel = float((diff / b.abs().clamp_min(1e-6)).max().item())
    passed = bool(torch.allclose(a, b, atol=atol, rtol=rtol))
    if verbose:
        status = "PASS" if passed else "FAIL"
        rel_l2, cos_sim, _ = compute_tensor_metrics(a, b)
        print(
            f"    [{status}] {name:<26} rel_l2={rel_l2:.3e} cos={cos_sim:.6f} "
            f"max_abs={max_abs:.3e}"
        )
    return passed, max_abs, max_rel


def check_dual_precision(
    name: str,
    kernel_val: torch.Tensor,
    baseline_fp16_val: Optional[torch.Tensor],
    truth_val: torch.Tensor,
    abs_cap: float = 1e-1,
    abs_cap_group: float = 1e-1,
    lead_dims: Optional[int] = None,
    use_ratio: bool = True,
    ratio_fail: float = 8.0,
    ratio_warn: float = 4.0,
    err_floor: float = 3e-4,
    verbose: bool = True,
) -> Tuple[bool, float, float]:
    """Dual-criterion check comparing kernel and FP16 baseline against FP32/FP64 truth."""
    if not torch.isfinite(kernel_val).all():
        if verbose:
            print(f"    [FAIL] {name:<24} non-finite kernel values!")
        return False, float("inf"), float("inf")

    kr, kc, km = compute_tensor_metrics(kernel_val, truth_val)
    kw = (
        worst_group_rel(kernel_val, truth_val, lead_dims)
        if lead_dims is not None
        else None
    )

    if baseline_fp16_val is not None:
        br, bc, bm = compute_tensor_metrics(baseline_fp16_val, truth_val)
        bw = (
            worst_group_rel(baseline_fp16_val, truth_val, lead_dims)
            if lead_dims is not None
            else None
        )
    else:
        br, bc, bm, bw = float("nan"), float("nan"), float("nan"), None

    def _eval_one(k_err: float, b_err: Optional[float], cap: float) -> Tuple[str, Optional[float]]:
        if k_err != k_err or k_err > cap:
            return "FAIL", None
        if use_ratio and b_err is not None:
            base = err_floor if b_err != b_err else max(b_err, err_floor)
            ratio = k_err / base
            if ratio > ratio_fail:
                return "FAIL", ratio
            return "PASS", ratio
        return "PASS", None

    tag, r_main = _eval_one(kr, br, abs_cap)
    r_grp = None
    if kw is not None:
        tag_w, r_grp = _eval_one(kw, bw, abs_cap_group)
        if tag_w == "FAIL":
            tag = "FAIL"

    passed = tag != "FAIL"
    if verbose:
        base_str = (
            f"{br:.2e} {bc:.6f} {bm:.1e}"
            if baseline_fp16_val is not None
            else f"{'N/A':>24}"
        )
        grp_str = (
            f" | grp {kw:.2e} {bw:.2e}"
            if kw is not None and bw is not None
            else (f" | grp {kw:.2e}" if kw is not None else "")
        )
        max_r = max((r for r in (r_main, r_grp) if r is not None), default=0.0)
        note = f" (~{max_r:.1f}x base)" if max_r > ratio_warn else ""
        print(
            f"    [{tag:<4}] {name:<20} | {kr:.2e} {kc:.6f} {km:.1e} "
            f"| {base_str}{grp_str}{note}"
        )
    return passed, kr, br

