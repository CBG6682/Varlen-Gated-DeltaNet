import glob
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# =========================================================================
# 1. CUDA Extension / Shared Library Loading
# =========================================================================
_LIB_LOADED = False
_LOAD_ERROR = None


def _try_load_library():
    global _LIB_LOADED, _LOAD_ERROR
    if _LIB_LOADED:
        return True

    # 1. Check if already registered in PyTorch dispatcher
    if hasattr(torch.ops, "deltanet") and hasattr(torch.ops.deltanet, "chunk_wy_fwd"):
        _LIB_LOADED = True
        _LOAD_ERROR = None
        return True

    # 2. Try importing compiled C extension module (deltanet_v2._C)
    try:
        from deltanet_v2 import _C
        _LIB_LOADED = True
        _LOAD_ERROR = None
        return True
    except Exception as e:
        _LOAD_ERROR = e

    # 3. Search for compiled .so files in package directory, site-packages, or build folders
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []

    # Local package directory
    if os.path.isdir(pkg_dir):
        candidates += [
            os.path.join(pkg_dir, f) for f in os.listdir(pkg_dir) if f.endswith(".so")
        ]

    # Build directories
    candidates += glob.glob(os.path.join(pkg_dir, "..", "build", "*", "*.so"))
    candidates += glob.glob(os.path.join(pkg_dir, "..", "build", "*", "*", "*.so"))
    candidates += glob.glob(os.path.join(pkg_dir, "..", "*.so"))

    # Installed site-packages / dist-packages
    try:
        import site
        sp_dirs = site.getsitepackages() if hasattr(site, "getsitepackages") else []
        if hasattr(site, "getusersitepackages"):
            sp_dirs = list(sp_dirs) + [site.getusersitepackages()]
        for sp in sp_dirs:
            candidates += glob.glob(os.path.join(sp, "deltanet_v2", "*.so"))
            candidates += glob.glob(os.path.join(sp, "*deltanet_v2*.so"))
    except Exception:
        pass

    for so_path in candidates:
        if os.path.isfile(so_path):
            try:
                torch.ops.load_library(os.path.abspath(so_path))
                if hasattr(torch.ops, "deltanet") and hasattr(torch.ops.deltanet, "chunk_wy_fwd"):
                    _LIB_LOADED = True
                    _LOAD_ERROR = None
                    return True
            except Exception as e:
                _LOAD_ERROR = e
                continue

    return _LIB_LOADED


_try_load_library()


# =========================================================================
# 2. PyTorch Fake Tensor / Meta Registrations (for torch.compile / Dynamo)
# =========================================================================
if (
    _LIB_LOADED
    and hasattr(torch.ops, "deltanet")
    and hasattr(torch.ops.deltanet, "chunk_wy_fwd")
    and hasattr(torch, "library")
    and hasattr(torch.library, "register_fake")
):

    @torch.library.register_fake("deltanet::chunk_wy_fwd")
    def _chunk_wy_fwd_fake(k, v, b, w, gamma, a, cu_seq_lens, chunk_size=64):
        total_L = k.shape[1]
        num_kv_heads = k.shape[0]
        d_k = k.shape[2]
        d_v = v.shape[2]
        num_chunks = torch.library.get_ctx().new_dynamic_size()
        W = k.new_empty(num_kv_heads, num_chunks, d_k, d_k)
        Y = k.new_empty(num_kv_heads, num_chunks, d_v, d_k)
        return W, Y

    @torch.library.register_fake("deltanet::chunk_scan_fwd")
    def _chunk_scan_fwd_fake(W, Y, state_old, cu_seq_lens, chunk_size=64):
        num_kv_heads = W.shape[0]
        num_chunks = W.shape[1]
        d_k = W.shape[2]
        d_v = Y.shape[2]
        batch = cu_seq_lens.shape[0] - 1
        S = W.new_empty(num_kv_heads, num_chunks, d_v, d_k)
        final_states = W.new_empty(batch, num_kv_heads, d_v, d_k)
        return S, final_states

    @torch.library.register_fake("deltanet::output_combine_fwd")
    def _output_combine_fwd_fake(
        q, k, v, b, w, gamma, a, S, cu_seq_lens,
        num_q_heads, num_kv_heads, d_k=64, d_v=64, chunk_size=64,
    ):
        total_L = q.shape[1]
        return q.new_empty(num_q_heads, total_L, d_v)

    @torch.library.register_fake("deltanet::deltanet_fwd")
    def _deltanet_fwd_fake(
        past_states, q, k, v, w, b, a, gamma, cu_seq_lens,
        num_q_heads, num_kv_heads, d_k=64, d_v=64, chunk_size=64, forward_only=False,
    ):
        total_L = q.shape[1]
        batch = cu_seq_lens.shape[0] - 1
        y = q.new_empty(num_q_heads, total_L, d_v)
        final_states = q.new_empty(batch, num_kv_heads, d_v, d_k)
        if forward_only:
            W = q.new_empty(0)
            S = q.new_empty(0)
        else:
            num_chunks = torch.library.get_ctx().new_dynamic_size()
            W = q.new_empty(num_kv_heads, num_chunks, d_k, d_k)
            S = q.new_empty(num_kv_heads, num_chunks, d_v, d_k)
        return y, final_states, W, S

    @torch.library.register_fake("deltanet::deltanet_bwd")
    def _deltanet_bwd_fake(
        grad_output, grad_final_states, past_states, q, k, v, w, b, a, gamma,
        cu_seq_lens, W, S, num_q_heads, num_kv_heads,
        d_k=64, d_v=64, chunk_size=64, has_past_states=False,
    ):
        total_L = q.shape[1]
        batch = cu_seq_lens.shape[0] - 1
        dq = q.new_empty(num_q_heads, total_L, d_k)
        dk = q.new_empty(num_kv_heads, total_L, d_k)
        dv = q.new_empty(num_kv_heads, total_L, d_v)
        dw = q.new_empty(num_kv_heads, total_L, d_v)
        db = q.new_empty(num_kv_heads, total_L, d_k)
        da = q.new_empty(num_kv_heads, total_L, 1)
        dgamma = q.new_empty(num_kv_heads, total_L, 1)
        if has_past_states:
            dpast = q.new_empty(batch, num_kv_heads, d_v, d_k)
        else:
            dpast = q.new_empty(0)
        return dq, dk, dv, dw, db, da, dgamma, dpast

    @torch.library.register_fake("deltanet::deltanet_infer")
    def _deltanet_infer_fake(
        past_state, q, k, v, b, w, a, gamma,
        num_q_heads, num_kv_heads, d_k=64, d_v=64,
    ):
        B = q.shape[0]
        out = q.new_empty(B, num_q_heads, d_v)
        final_states = torch.empty_like(past_state)
        return out, final_states


class _CustomOpsProxy:
    """Proxy object accessing torch.ops.deltanet or compiled extension functions."""

    def __getattr__(self, name: str):
        if not _LIB_LOADED:
            _try_load_library()
        if hasattr(torch.ops, "deltanet") and hasattr(torch.ops.deltanet, name):
            return getattr(torch.ops.deltanet, name)
        try:
            from deltanet_v2 import _C
            if hasattr(_C, name):
                return getattr(_C, name)
        except Exception:
            pass
        err_msg = (
            f"DeltaNet op '{name}' not found. Please ensure the CUDA extension is compiled "
            f"via 'pip install .' or 'python setup.py build_ext --inplace'."
        )
        if _LOAD_ERROR is not None:
            err_msg += f" (Underlying loader error: {_LOAD_ERROR})"
        raise RuntimeError(err_msg)


custom_ops = _CustomOpsProxy()


# =========================================================================
# 3. Autograd Function & Input Verification
# =========================================================================
def _check_kernel_inputs(tensors, cu_seqlens=None):
    """Metadata-only check without GPU synchronization or host copies."""
    for name, t in tensors.items():
        if t is None:
            continue
        assert t.dtype == torch.float16, f"{name}: expected float16, got {t.dtype}"
        assert t.is_contiguous(), f"{name}: expected contiguous tensor, stride={t.stride()}"
    if cu_seqlens is not None:
        assert cu_seqlens.dtype == torch.int32 and cu_seqlens.is_contiguous(), (
            "cu_seqlens must be int32 and contiguous"
        )


def _headfirst(t: torch.Tensor, L: int, h: int, d: int) -> torch.Tensor:
    """Token-major (L, h*d) -> contiguous head-first (h, L, d). Single copy at producer."""
    return t.view(L, h, d).transpose(0, 1).contiguous()


class DeltaBlockKernelFn(torch.autograd.Function):
    """PyTorch Autograd Function wrapping deltanet_fwd and deltanet_bwd."""

    @staticmethod
    def forward(
        ctx, past_states, q, k, v, w, b, a, gamma, cu_seqlens,
        num_q_heads, num_kv_heads, d_k=64, d_v=64, chunk_size=64, forward_only=False,
    ):
        y, final_states, W, S = custom_ops.deltanet_fwd(
            past_states, q, k, v, w, b, a, gamma, cu_seqlens,
            num_q_heads, num_kv_heads, d_k, d_v, chunk_size, forward_only,
        )

        if not forward_only:
            ctx.save_for_backward(past_states, q, k, v, w, b, a, gamma, cu_seqlens, W, S)
            ctx.cfg = (num_q_heads, num_kv_heads, d_k, d_v, chunk_size)
            ctx.has_past_states = past_states is not None and past_states.numel() > 0
            ctx.set_materialize_grads(False)

        return y, final_states

    @staticmethod
    def backward(ctx, grad_y, grad_final_states):
        past_states, q, k, v, w, b, a, gamma, cu_seqlens, W, S = ctx.saved_tensors

        if grad_y is not None:
            grad_y = grad_y.contiguous()
        if grad_final_states is not None:
            grad_final_states = grad_final_states.contiguous()

        dq, dk, dv, dw, db, da, dgamma, grad_past = custom_ops.deltanet_bwd(
            grad_y, grad_final_states, past_states, q, k, v, w, b, a, gamma,
            cu_seqlens, W, S, *ctx.cfg, ctx.has_past_states,
        )

        if not ctx.has_past_states or (grad_past is not None and grad_past.numel() == 0):
            grad_past = None

        return (
            grad_past, dq, dk, dv, dw, db, da, dgamma,
            None, None, None, None, None, None, None,
        )


def apply_deltablock(
    past_states: Optional[torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    gamma: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    d_k: int = 64,
    d_v: int = 64,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Execute forward training pass with automatic backward pass recording."""
    tensors = (past_states, q, k, v, w, b, a, gamma)
    forward_only = not (
        torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors)
    )
    _check_kernel_inputs(
        dict(past=past_states, q=q, k=k, v=v, w=w, b=b, a=a, gamma=gamma), cu_seqlens
    )
    return DeltaBlockKernelFn.apply(
        past_states, q, k, v, w, b, a, gamma, cu_seqlens,
        num_q_heads, num_kv_heads, d_k, d_v, chunk_size, forward_only,
    )


def apply_deltablock_infer(
    past_state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    gamma: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    d_k: int = 64,
    d_v: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Execute single autoregressive decode step (L = 1)."""
    _check_kernel_inputs(
        dict(past=past_state, q=q, k=k, v=v, b=b, w=w, a=a, gamma=gamma)
    )
    return custom_ops.deltanet_infer(
        past_state, q, k, v, b, w, a, gamma,
        num_q_heads, num_kv_heads, d_k, d_v,
    )
