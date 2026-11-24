import glob
import os
from typing import Optional, Tuple

import torch

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
    if hasattr(torch.ops, "deltanet") and hasattr(torch.ops.deltanet, "deltablock_fwd"):
        _LIB_LOADED = True
        _LOAD_ERROR = None
        return True

    # 2. Try importing compiled C extension module (deltanet_v2._C)
    try:
        from deltanet_v2 import _C
        if hasattr(_C, "deltablock_fwd"):
            _LIB_LOADED = True
            _LOAD_ERROR = None
            return True
    except Exception as e:
        _LOAD_ERROR = e

    # 3. Search for compiled .so files in package directory, site-packages, or build folders
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []

    if os.path.isdir(pkg_dir):
        candidates += [
            os.path.join(pkg_dir, f) for f in os.listdir(pkg_dir) if f.endswith(".so")
        ]

    candidates += glob.glob(os.path.join(pkg_dir, "..", "build", "*", "*.so"))
    candidates += glob.glob(os.path.join(pkg_dir, "..", "build", "*", "*", "*.so"))
    candidates += glob.glob(os.path.join(pkg_dir, "..", "*.so"))

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
                if hasattr(torch.ops, "deltanet") and hasattr(torch.ops.deltanet, "deltablock_fwd"):
                    _LIB_LOADED = True
                    _LOAD_ERROR = None
                    return True
            except Exception as e:
                _LOAD_ERROR = e
                continue

    return _LIB_LOADED


_try_load_library()


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
# 2. Input Verification & Autograd Function
# =========================================================================
def _check_tensors_dtype(tensors, expected_dtype: torch.dtype):
    """Metadata-only check without GPU synchronization or host copies."""
    for name, t in tensors.items():
        if t is None:
            continue
        assert t.dtype == expected_dtype, f"{name}: expected {expected_dtype}, got {t.dtype}"
        assert t.is_contiguous(), f"{name}: expected contiguous tensor, stride={t.stride()}"


class FullDeltaBlockFn(torch.autograd.Function):
    """PyTorch Autograd Function wrapping full DeltaBlock deltablock_fwd and deltablock_bwd."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        past_states: Optional[torch.Tensor],
        past_conv_state: Optional[torch.Tensor],
        W_proj: torch.Tensor,
        conv_weight: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        o_norm_weight: torch.Tensor,
        W_out: torch.Tensor,
        norm1_weight: Optional[torch.Tensor],
        norm2_weight: Optional[torch.Tensor],
        mlp_W_gate_up: Optional[torch.Tensor],
        mlp_W_down: Optional[torch.Tensor],
        num_q_heads: int,
        num_kv_heads: int,
        d_k: int = 64,
        d_v: int = 64,
        conv_size: int = 4,
        chunk_size: int = 64,
        scale: float = 0.125,
        eps: float = 1e-5,
        allow_neg_eigval: bool = False,
        beta_guard: bool = True,
        is_training: bool = True,
        output_conv_state: bool = True,
    ):
        res = custom_ops.deltablock_fwd(
            x, cu_seqlens, past_states, past_conv_state,
            W_proj, conv_weight, A_log, dt_bias, o_norm_weight, W_out,
            norm1_weight, norm2_weight, mlp_W_gate_up, mlp_W_down,
            num_q_heads, num_kv_heads, d_k, d_v, conv_size, chunk_size,
            scale, eps, allow_neg_eigval, beta_guard, is_training, output_conv_state,
        )
        y, final_states, final_conv_state = res[0], res[1], res[2]
        ctx.mark_non_differentiable(final_conv_state)

        if is_training:
            saved_intermediates = res[3:]
            ctx.save_for_backward(
                x, cu_seqlens, past_states, past_conv_state,
                W_proj, conv_weight, A_log, dt_bias, o_norm_weight, W_out,
                norm1_weight, norm2_weight, mlp_W_gate_up, mlp_W_down,
                *saved_intermediates,
            )
            ctx.cfg = (num_q_heads, num_kv_heads, d_k, d_v, conv_size, chunk_size, scale, allow_neg_eigval)
            ctx.has_past_states = past_states is not None and past_states.numel() > 0
            ctx.has_past_conv_state = past_conv_state is not None and past_conv_state.numel() > 0
            ctx.has_mlp = mlp_W_gate_up is not None
            ctx.set_materialize_grads(False)

        return y, final_states, final_conv_state

    @staticmethod
    def backward(ctx, grad_y, grad_final_states, grad_final_conv_state=None):
        saved = ctx.saved_tensors
        (
            x, cu_seqlens, past_states, past_conv_state,
            W_proj, conv_weight, A_log, dt_bias, o_norm_weight, W_out,
            norm1_weight, norm2_weight, mlp_W_gate_up, mlp_W_down,
        ) = saved[:14]
        intermediates = saved[14:]

        if grad_y is None:
            grad_y = torch.zeros_like(x)
        else:
            grad_y = grad_y.contiguous()

        if grad_final_states is not None:
            grad_final_states = grad_final_states.contiguous()

        num_q_heads, num_kv_heads, d_k, d_v, conv_size, chunk_size, scale, allow_neg_eigval = ctx.cfg

        grads = custom_ops.deltablock_bwd(
            grad_y, grad_final_states,
            x, cu_seqlens, past_states, past_conv_state,
            W_proj, conv_weight, A_log, dt_bias, o_norm_weight, W_out,
            norm1_weight, norm2_weight, mlp_W_gate_up, mlp_W_down,
            *intermediates,
            num_q_heads, num_kv_heads, d_k, d_v, conv_size, chunk_size,
            scale, allow_neg_eigval, ctx.has_past_states, ctx.has_past_conv_state,
        )
        (
            grad_x, grad_past_states, grad_past_conv_state,
            dW_proj, d_conv_weight, d_A_log, d_dt_bias, d_o_norm_weight, dW_out,
            d_norm1_weight, d_norm2_weight, d_mlp_W_gate_up, d_mlp_W_down,
        ) = grads

        if not ctx.has_past_states or grad_past_states.numel() == 0:
            grad_past_states = None
        if not ctx.has_past_conv_state or grad_past_conv_state.numel() == 0:
            grad_past_conv_state = None
        if not ctx.has_mlp:
            d_norm1_weight = None
            d_norm2_weight = None
            d_mlp_W_gate_up = None
            d_mlp_W_down = None

        return (
            grad_x, None, grad_past_states, grad_past_conv_state,
            dW_proj, d_conv_weight, d_A_log, d_dt_bias, d_o_norm_weight, dW_out,
            d_norm1_weight, d_norm2_weight, d_mlp_W_gate_up, d_mlp_W_down,
            None, None, None, None, None, None, None, None, None, None, None, None,
        )


def apply_deltablock(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    past_states: Optional[torch.Tensor],
    past_conv_state: Optional[torch.Tensor],
    W_proj: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    o_norm_weight: torch.Tensor,
    W_out: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    norm1_weight: Optional[torch.Tensor] = None,
    norm2_weight: Optional[torch.Tensor] = None,
    mlp_W_gate_up: Optional[torch.Tensor] = None,
    mlp_W_down: Optional[torch.Tensor] = None,
    d_k: int = 64,
    d_v: int = 64,
    conv_size: int = 4,
    chunk_size: int = 64,
    scale: Optional[float] = None,
    eps: float = 1e-5,
    allow_neg_eigval: bool = False,
    beta_guard: bool = True,
    output_conv_state: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute full DeltaBlock forward (varlen training or prefill) returning (y, final_states, final_conv_state)."""
    if scale is None:
        scale = d_k ** -0.5

    _check_tensors_dtype(
        dict(
            x=x,
            past_conv_state=past_conv_state,
            W_proj=W_proj,
            conv_weight=conv_weight,
            o_norm_weight=o_norm_weight,
            W_out=W_out,
            norm1_weight=norm1_weight,
            norm2_weight=norm2_weight,
            mlp_W_gate_up=mlp_W_gate_up,
            mlp_W_down=mlp_W_down,
        ),
        torch.float16,
    )
    _check_tensors_dtype(dict(past_states=past_states, A_log=A_log, dt_bias=dt_bias), torch.float32)
    assert cu_seqlens.dtype in (torch.int32, torch.int64) and cu_seqlens.is_contiguous(), (
        "cu_seqlens must be contiguous int32/int64"
    )

    grad_tensors = (
        x, past_states, past_conv_state, W_proj, conv_weight, A_log, dt_bias,
        o_norm_weight, W_out, norm1_weight, norm2_weight, mlp_W_gate_up, mlp_W_down,
    )
    is_training = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in grad_tensors)

    return FullDeltaBlockFn.apply(
        x, cu_seqlens, past_states, past_conv_state,
        W_proj, conv_weight, A_log, dt_bias, o_norm_weight, W_out,
        norm1_weight, norm2_weight, mlp_W_gate_up, mlp_W_down,
        num_q_heads, num_kv_heads, d_k, d_v, conv_size, chunk_size,
        float(scale), float(eps), allow_neg_eigval, beta_guard, is_training, output_conv_state,
    )


def apply_deltablock_infer(
    x: torch.Tensor,
    past_states: torch.Tensor,
    conv_state: torch.Tensor,
    W_proj: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    o_norm_weight: torch.Tensor,
    W_out: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    norm1_weight: Optional[torch.Tensor] = None,
    norm2_weight: Optional[torch.Tensor] = None,
    mlp_W_gate_up: Optional[torch.Tensor] = None,
    mlp_W_down: Optional[torch.Tensor] = None,
    d_k: int = 64,
    d_v: int = 64,
    conv_size: int = 4,
    scale: Optional[float] = None,
    eps: float = 1e-5,
    allow_neg_eigval: bool = False,
    beta_guard: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute single autoregressive decode step (L = 1) updating past_states [FP32] and conv_state [FP16] in-place."""
    if scale is None:
        scale = d_k ** -0.5

    _check_tensors_dtype(
        dict(
            x=x,
            conv_state=conv_state,
            W_proj=W_proj,
            conv_weight=conv_weight,
            o_norm_weight=o_norm_weight,
            W_out=W_out,
            norm1_weight=norm1_weight,
            norm2_weight=norm2_weight,
            mlp_W_gate_up=mlp_W_gate_up,
            mlp_W_down=mlp_W_down,
        ),
        torch.float16,
    )
    _check_tensors_dtype(dict(past_states=past_states, A_log=A_log, dt_bias=dt_bias), torch.float32)

    return custom_ops.deltablock_infer(
        x, past_states, conv_state,
        W_proj, conv_weight, A_log, dt_bias, o_norm_weight, W_out,
        norm1_weight, norm2_weight, mlp_W_gate_up, mlp_W_down,
        num_q_heads, num_kv_heads, d_k, d_v, conv_size,
        float(scale), float(eps), allow_neg_eigval, beta_guard,
    )
