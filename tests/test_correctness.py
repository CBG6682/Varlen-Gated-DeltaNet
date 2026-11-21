"""
Comprehensive correctness and regression test suite for DeltaNet v2 CUDA kernels.

Covers four verification suites modeled after standard CUDA/CuTe operator validation:
  1. Varlen Forward & Backward Kernels (raw custom_ops / apply_deltablock + edge cases)
  2. Inference Decode Kernel (L = 1 raw kernel, stress/edge cases, ablations, invariance, canary, long rollout)
  3. End-to-End DeltaBlock Modules (varlen fwd+bwd, decode rollout, prefill->decode handoff, CUDA graphs/streams, API guards)
  4. torch.compile (Inductor) Compatibility (training fwd+bwd & decode step)

Usage:
  python tests/test_correctness.py
  python tests/test_correctness.py --quick
  python tests/test_correctness.py --suite infer --seed 123
"""

import argparse
import copy
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from deltanet_v2 import DeltaBlock, apply_deltablock, apply_deltablock_infer, custom_ops
from deltanet_v2.ops import _LIB_LOADED, _LOAD_ERROR, _try_load_library
from tests.utils import (
    check_allclose,
    check_dual_precision,
    compute_tensor_metrics,
    make_cu_seqlens,
    make_varlen_inputs,
    print_section,
    seed_everything,
)

# ---------------------------------------------------------------------------
# Tolerance Constants
# ---------------------------------------------------------------------------
ABS_CAP = 1.0e-1
ABS_CAP_GROUP = 1.0e-1
RATIO_FAIL = 8.0
RATIO_WARN = 4.0
ERR_FLOOR = 3.0e-4
REF_CONSIST_TOL = 1.0e-5
INV_TOL = 1.0e-3
MODEL_INV_TOL = 2.0e-3


# ---------------------------------------------------------------------------
# Environment / Result Helpers
# ---------------------------------------------------------------------------
def _require_gpu_and_ext() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to run DeltaNet v2 correctness tests.")
    if not _LIB_LOADED:
        _try_load_library()
    if not _LIB_LOADED:
        raise RuntimeError(
            "DeltaNet v2 CUDA extension (_C) is not loaded. Build it first via "
            f"`pip install . --no-build-isolation` or `python setup.py build_ext --inplace` "
            f"(error: {_LOAD_ERROR})."
        )
    return torch.device("cuda")


def _record(
    failures: List[str],
    tag: str,
    name: str,
    passed: bool,
    detail: str = "",
) -> None:
    status = "PASS" if passed else "FAIL"
    suffix = f" {detail}" if detail else ""
    print(f"    [{status:<4}] {name:<30}{suffix}")
    if not passed:
        failures.append(f"{tag} :: {name}{suffix}")


def _dual_check(
    failures: List[str],
    tag: str,
    name: str,
    kernel_val: torch.Tensor,
    baseline_fp16_val: Optional[torch.Tensor],
    truth_val: torch.Tensor,
    lead_dims: Optional[int] = None,
    use_ratio: bool = True,
) -> None:
    passed, kr, br = check_dual_precision(
        name=name,
        kernel_val=kernel_val,
        baseline_fp16_val=baseline_fp16_val,
        truth_val=truth_val,
        abs_cap=ABS_CAP,
        abs_cap_group=ABS_CAP_GROUP,
        lead_dims=lead_dims,
        use_ratio=use_ratio,
        ratio_fail=RATIO_FAIL,
        ratio_warn=RATIO_WARN,
        err_floor=ERR_FLOOR,
        verbose=True,
    )
    if not passed:
        failures.append(f"{tag} :: {name} (kernel_rel={kr:.2e}, base_rel={br:.2e})")


def _print_table_header(ref_name: str = "fp32") -> None:
    print(
        f"    {'status':<6} {'tensor':<20} | {f'kernel vs {ref_name}':<24} "
        f"| {f'base16 vs {ref_name}':<24} | grp(worst) k b"
    )


def _summarize_failures(suite_name: str, failures: List[str]) -> None:
    if failures:
        print(f"\n[FAILED] {suite_name}: {len(failures)} check(s) failed:")
        for msg in failures[:20]:
            print(f"  - {msg}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
    else:
        print(f"\n[OK] {suite_name}: all checks passed.")


# ---------------------------------------------------------------------------
# Pure-PyTorch Reference Implementations
# ---------------------------------------------------------------------------
def _ref_core_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    gamma: torch.Tensor,
    S: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-step DeltaNet state update and readout (pure PyTorch in caller dtype)."""
    B, Hq, dk = q.shape
    H = k.shape[1]
    dv = v.shape[-1]
    G = Hq // H
    if S is None:
        S = torch.zeros(B, H, dv, dk, device=k.device, dtype=k.dtype)

    kn = F.normalize(k, p=2, dim=-1, eps=1e-6)
    sg = F.softplus(gamma).sqrt()
    ks = kn * sg
    vn = torch.sigmoid(w) * (v * sg)
    u = torch.sigmoid(b) * ks
    alpha = torch.sigmoid(a).unsqueeze(-1)  # (B, H, 1, 1)
    q4 = q.reshape(B, H, G, dk)

    Su = torch.einsum("bhvk,bhk->bhv", S, u)
    S_new = S * alpha + (vn - Su).unsqueeze(-1) * ks.unsqueeze(-2)

    o_inter = torch.einsum("bhvk,bhgk->bhgv", S, q4) * alpha
    attn = torch.einsum("bhk,bhgk->bhg", ks, q4)
    o_intra = attn.unsqueeze(-1) * vn.unsqueeze(2)
    o = (o_inter + o_intra).reshape(B, Hq, dv)
    return o.contiguous(), S_new.contiguous()


def _ref_decode_block_step(
    model: DeltaBlock, x_2d: torch.Tensor, S: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-step DeltaBlock decode reference matching DeltaBlock architecture."""
    B = x_2d.shape[0]
    H, Hq, dk, dv = model.num_kv_heads, model.num_q_heads, model.d_k, model.d_v
    q_r, k_r, v_r, a_r, b_r, w_r, g_r = torch.split(
        model.W_proj(x_2d), model.split_sizes, dim=-1
    )
    o, S_new = _ref_core_step(
        q_r.reshape(B, Hq, dk),
        k_r.reshape(B, H, dk),
        v_r.reshape(B, H, dv),
        a_r.reshape(B, H, 1),
        b_r.reshape(B, H, dk),
        w_r.reshape(B, H, dv),
        g_r.reshape(B, H, 1),
        S,
    )
    out = model.rms_norm(o.reshape(B, model.d_out_total)) * F.silu(model.W_gate(x_2d))
    y = model.W_out(out)
    if model.hidden_dim is not None:
        xn = model.norm1(x_2d + y)
        ffn = model.mlp_W_down(F.silu(model.mlp_W_gate(xn)) * model.mlp_W_up(xn))
        y = model.norm2(xn + ffn)
    return y.contiguous(), S_new


def _ref_varlen_raw_chunked(
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
    """Autograd-capable chunked recurrence reference on packed head-first raw tensors."""
    device = q.device
    dtype = q.dtype
    Hq, H, C = num_q_heads, num_kv_heads, chunk_size
    G = Hq // H
    B = cu_seqlens.numel() - 1
    eye = torch.eye(d_k, device=device, dtype=torch.float32).view(1, 1, d_k, d_k)

    if q.shape[1] == 0 or B == 0:
        empty_y = torch.empty(Hq, 0, d_v, device=device, dtype=dtype)
        if past_states is not None:
            fs = past_states.clone()
        else:
            fs = torch.zeros(B, H, d_v, d_k, device=device, dtype=dtype)
        return empty_y, fs

    cu_list = cu_seqlens.tolist()
    y_parts = []
    state_parts = []

    for i in range(B):
        start, end = cu_list[i], cu_list[i + 1]
        if past_states is not None:
            S_curr = past_states[i].float()
        else:
            S_curr = torch.zeros(H, d_v, d_k, device=device, dtype=torch.float32)

        L_i = end - start
        if L_i <= 0:
            state_parts.append(S_curr.to(dtype))
            continue

        # Convert slices from head-first (h, L_i, d) -> token-first (L_i, h, d)
        qi = q[:, start:end, :].transpose(0, 1)
        ki = k[:, start:end, :].transpose(0, 1)
        vi = v[:, start:end, :].transpose(0, 1)
        wi = w[:, start:end, :].transpose(0, 1)
        bi = b[:, start:end, :].transpose(0, 1)
        ai = a[:, start:end, :].transpose(0, 1)
        gi = gamma[:, start:end, :].transpose(0, 1)

        pad_len = (-L_i) % C
        if pad_len:
            qi = F.pad(qi, (0, 0, 0, 0, 0, pad_len))
            ki = F.pad(ki, (0, 0, 0, 0, 0, pad_len))
            vi = F.pad(vi, (0, 0, 0, 0, 0, pad_len))
            wi = F.pad(wi, (0, 0, 0, 0, 0, pad_len))
            bi = F.pad(bi, (0, 0, 0, 0, 0, pad_len))
            ai = F.pad(ai, (0, 0, 0, 0, 0, pad_len))
            gi = F.pad(gi, (0, 0, 0, 0, 0, pad_len))
            mask = torch.arange(L_i + pad_len, device=device) < L_i
            m = mask.view(-1, 1, 1)
        else:
            m = None

        Lp = L_i + pad_len
        num_chunks = Lp // C

        log_alpha = F.logsigmoid(ai.float())
        b_gate = torch.sigmoid(bi)
        w_gate = torch.sigmoid(wi)
        k_norm = F.normalize(ki.float(), p=2, dim=-1, eps=1e-6).to(dtype)
        sqrt_g = F.softplus(gi).sqrt()

        if m is not None:
            log_alpha = log_alpha.masked_fill(~m, 0.0)
            b_gate = b_gate * m
            w_gate = w_gate * m
            k_norm = k_norm * m

        k_scaled = k_norm * sqrt_g
        v_scaled = vi * sqrt_g
        u = b_gate * k_scaled
        v_new = w_gate * v_scaled

        q_c = qi.view(num_chunks, C, H, G, d_k)
        k_c = k_scaled.view(num_chunks, C, H, d_k)
        v_c = v_new.view(num_chunks, C, H, d_v)
        u_c = u.view(num_chunks, C, H, d_k)
        la_c = log_alpha.view(num_chunks, C, H, 1)

        cum = torch.cumsum(la_c, dim=1)
        alpha_cumprod = cum.exp()
        cum_last = cum[:, -1:]
        chunk_decay = cum_last.exp()
        k_decayed = k_c.float() * torch.exp(cum_last - cum)

        erase_m = torch.einsum("c t h k, c t h m -> c h k m", u_c.float(), k_decayed)
        W_chunks = eye * chunk_decay.squeeze(1).unsqueeze(-1) - erase_m
        Y_chunks = torch.einsum("c t h v, c t h k -> c h v k", v_c.float(), k_decayed)

        S_list = []
        for c in range(num_chunks):
            S_list.append(S_curr)
            S_curr = torch.matmul(S_curr, W_chunks[c]) + Y_chunks[c]
        S_all = torch.stack(S_list, dim=0)

        attn = torch.einsum("c i h g d, c j h d -> c h g i j", q_c, k_c)
        causal = torch.tril(torch.ones(C, C, device=device, dtype=torch.bool))
        a_perm = cum.squeeze(-1).permute(0, 2, 1)  # (c, H, C)
        diff = (a_perm.unsqueeze(-1) - a_perm.unsqueeze(-2)).masked_fill(~causal, float("-inf"))
        attn = attn * diff.exp().unsqueeze(2).to(attn.dtype)
        out_intra = torch.einsum("c h g i j, c j h v -> c i h g v", attn, v_c)

        q_S = torch.einsum("c t h g k, c h v k -> c t h g v", q_c.float(), S_all)
        out_inter = q_S * alpha_cumprod.unsqueeze(-2)
        out_i = (out_intra + out_inter.to(out_intra.dtype)).reshape(Lp, Hq, d_v)[:L_i]
        y_parts.append(out_i.transpose(0, 1))
        state_parts.append(S_curr.to(dtype))

    y_all = torch.cat(y_parts, dim=1).contiguous()
    fs_all = torch.stack(state_parts, dim=0).contiguous()
    return y_all, fs_all


# ===========================================================================
# SUITE 1: Raw Varlen Forward & Backward Kernels
# ===========================================================================
def test_varlen_fwd_bwd_kernels(
    quick: bool = False, device: Optional[torch.device] = None
) -> List[str]:
    print_section("SUITE 1: Varlen Forward & Backward Kernels (Raw Ops & Edge Cases)")
    device = device or _require_gpu_and_ext()
    failures: List[str] = []

    configs = (
        [(8, 4, [64, 128, 65, 1]), (8, 2, [1, 17, 63, 64, 65])]
        if quick
        else [
            (8, 4, [64, 128, 65, 1]),
            (8, 8, [1, 17, 63, 64, 65, 129]),
            (8, 2, [300, 129, 64, 257]),
            (16, 4, [512, 127, 65]),
        ]
    )

    for hq, hkv, seqlens in configs:
        B = len(seqlens)
        L = sum(seqlens)
        cu, _ = make_cu_seqlens(seqlens, device)
        for use_past in (False, True):
            tag = f"[train/raw] hq={hq} hkv={hkv} seqlens={seqlens} past={use_past}"
            print(f"\n  {tag}")
            _print_table_header("fp32")

            q0 = (torch.randn(hq, L, 64, device=device) * 0.5).half().contiguous()
            k0 = (torch.randn(hkv, L, 64, device=device) * 0.5).half().contiguous()
            v0 = (torch.randn(hkv, L, 64, device=device) * 0.5).half().contiguous()
            w0 = torch.randn(hkv, L, 64, device=device).half().contiguous()
            b0 = torch.randn(hkv, L, 64, device=device).half().contiguous()
            a0 = torch.randn(hkv, L, 1, device=device).half().contiguous()
            g0 = torch.randn(hkv, L, 1, device=device).half().contiguous()
            p0 = (torch.randn(B, hkv, 64, 64, device=device) * 0.1).half().contiguous() if use_past else None

            gy = torch.randn(hq, L, 64, device=device, dtype=torch.float32)
            gfs = torch.randn(B, hkv, 64, 64, device=device, dtype=torch.float32)

            def _run_raw(mode: str):
                dtype = torch.float32 if mode == "truth" else torch.float16
                q = q0.to(dtype).detach().clone().contiguous().requires_grad_(True)
                k = k0.to(dtype).detach().clone().contiguous().requires_grad_(True)
                v = v0.to(dtype).detach().clone().contiguous().requires_grad_(True)
                w = w0.to(dtype).detach().clone().contiguous().requires_grad_(True)
                b = b0.to(dtype).detach().clone().contiguous().requires_grad_(True)
                a = a0.to(dtype).detach().clone().contiguous().requires_grad_(True)
                g = g0.to(dtype).detach().clone().contiguous().requires_grad_(True)
                p = p0.to(dtype).detach().clone().contiguous().requires_grad_(True) if p0 is not None else None
                if mode == "kernel":
                    y, fs = apply_deltablock(p, q, k, v, w, b, a, g, cu, hq, hkv, 64, 64, 64)
                else:
                    y, fs = _ref_varlen_raw_chunked(p, q, k, v, w, b, a, g, cu, hq, hkv, 64, 64, 64)
                loss = (y.float() * gy).sum() + (fs.float() * gfs).sum()
                loss.backward()
                out = {
                    "y": y.detach(),
                    "final_states": fs.detach(),
                    "grad_q": q.grad.detach(),
                    "grad_k": k.grad.detach(),
                    "grad_v": v.grad.detach(),
                    "grad_w": w.grad.detach(),
                    "grad_b": b.grad.detach(),
                    "grad_a": a.grad.detach(),
                    "grad_gamma": g.grad.detach(),
                }
                if p is not None:
                    out["grad_past"] = p.grad.detach()
                return out

            res_t = _run_raw("truth")
            res_b = _run_raw("base16")
            res_k = _run_raw("kernel")

            for name in res_t:
                lead = 2 if name in ("y", "final_states") or name.startswith("grad_") else None
                use_ratio = not name.startswith("grad_")
                _dual_check(
                    failures, tag, name, res_k[name], res_b[name], res_t[name],
                    lead_dims=lead, use_ratio=use_ratio,
                )

    # 1B: Direct deltanet_fwd (forward_only=True/False) & deltanet_bwd consistency
    tag_direct = "[train/direct-ops] hq=8 hkv=4 seqlens=[64, 65, 1]"
    print(f"\n  {tag_direct}")
    cu_d, _ = make_cu_seqlens([64, 65, 1], device)
    L_d, B_d, hq, hkv = 130, 3, 8, 4
    q_d = (torch.randn(hq, L_d, 64, device=device) * 0.5).half().contiguous().requires_grad_(True)
    k_d = (torch.randn(hkv, L_d, 64, device=device) * 0.5).half().contiguous().requires_grad_(True)
    v_d = (torch.randn(hkv, L_d, 64, device=device) * 0.5).half().contiguous().requires_grad_(True)
    w_d = torch.randn(hkv, L_d, 64, device=device).half().contiguous().requires_grad_(True)
    b_d = torch.randn(hkv, L_d, 64, device=device).half().contiguous().requires_grad_(True)
    a_d = torch.randn(hkv, L_d, 1, device=device).half().contiguous().requires_grad_(True)
    g_d = torch.randn(hkv, L_d, 1, device=device).half().contiguous().requires_grad_(True)
    p_d = (torch.randn(B_d, hkv, 64, 64, device=device) * 0.1).half().contiguous().requires_grad_(True)

    with torch.no_grad():
        y_fo, fs_fo, W_fo, S_fo = custom_ops.deltanet_fwd(
            p_d, q_d, k_d, v_d, w_d, b_d, a_d, g_d, cu_d, hq, hkv, 64, 64, 64, True
        )
        y_tr, fs_tr, W_tr, S_tr = custom_ops.deltanet_fwd(
            p_d, q_d, k_d, v_d, w_d, b_d, a_d, g_d, cu_d, hq, hkv, 64, 64, 64, False
        )
    _record(
        failures, tag_direct, "forward_only == train fwd",
        torch.equal(y_fo, y_tr) and torch.equal(fs_fo, fs_tr)
        and W_fo.numel() == 0 and S_fo.numel() == 0 and W_tr.numel() > 0 and S_tr.numel() > 0,
    )

    gy_d = torch.randn_like(y_tr)
    y_ag, fs_ag = apply_deltablock(p_d, q_d, k_d, v_d, w_d, b_d, a_d, g_d, cu_d, hq, hkv, 64, 64, 64)
    (y_ag * gy_d).sum().backward(retain_graph=False)
    # Compare explicit deltanet_bwd with gy_d and zero grad_final_states
    gfs_zero = torch.zeros_like(fs_tr)
    direct_grads = custom_ops.deltanet_bwd(
        gy_d, gfs_zero, p_d.detach(), q_d.detach(), k_d.detach(), v_d.detach(),
        w_d.detach(), b_d.detach(), a_d.detach(), g_d.detach(), cu_d, W_tr, S_tr,
        hq, hkv, 64, 64, 64, True,
    )
    ag_grads = (q_d.grad, k_d.grad, v_d.grad, w_d.grad, b_d.grad, a_d.grad, g_d.grad, p_d.grad)
    direct_match = all(
        check_allclose(f"g{i} (direct vs autograd)", dm, ag, atol=1e-4, rtol=1e-4, verbose=False)[0]
        for i, (dm, ag) in enumerate(zip(direct_grads, ag_grads))
    )
    _record(failures, tag_direct, "deltanet_bwd == autograd", direct_match)

    # 1C: Zero-length sequence & empty batch edge cases
    tag_edge = "[train/zero-len] seqlens=[0, 17, 0, 65]"
    print(f"\n  {tag_edge}")
    cu_z, _ = make_cu_seqlens([0, 17, 0, 65], device)
    Lz, Bz, hq_z, hkv_z = 82, 4, 8, 4
    qz = (torch.randn(hq_z, Lz, 64, device=device) * 0.5).half().contiguous()
    kz = (torch.randn(hkv_z, Lz, 64, device=device) * 0.5).half().contiguous()
    vz = (torch.randn(hkv_z, Lz, 64, device=device) * 0.5).half().contiguous()
    wz = torch.randn(hkv_z, Lz, 64, device=device).half().contiguous()
    bz = torch.randn(hkv_z, Lz, 64, device=device).half().contiguous()
    az = torch.randn(hkv_z, Lz, 1, device=device).half().contiguous()
    gz = torch.randn(hkv_z, Lz, 1, device=device).half().contiguous()
    pz = (torch.randn(Bz, hkv_z, 64, 64, device=device) * 0.1).half().contiguous()

    yz_ref, fsz_ref = _ref_varlen_raw_chunked(
        pz.float(), qz.float(), kz.float(), vz.float(), wz.float(), bz.float(),
        az.float(), gz.float(), cu_z, hq_z, hkv_z, 64, 64, 64,
    )
    pz_in = pz.clone().requires_grad_(True)
    qz_in = qz.clone().requires_grad_(True)
    yz_k, fsz_k = apply_deltablock(
        pz_in, qz_in,
        kz.clone().requires_grad_(True), vz.clone().requires_grad_(True),
        wz.clone().requires_grad_(True), bz.clone().requires_grad_(True),
        az.clone().requires_grad_(True), gz.clone().requires_grad_(True),
        cu_z, hq_z, hkv_z, 64, 64, 64,
    )
    yz_k.sum().backward()
    _dual_check(failures, tag_edge, "y (zero-len mixed)", yz_k, None, yz_ref, lead_dims=2, use_ratio=False)
    _dual_check(failures, tag_edge, "fs (zero-len mixed)", fsz_k, None, fsz_ref, lead_dims=2, use_ratio=False)
    # Empty seqs 0 and 2 must carry past_states unchanged and receive zero past grad
    carry_ok = (
        torch.allclose(fsz_k[0].float(), pz[0].float(), atol=1e-4)
        and torch.allclose(fsz_k[2].float(), pz[2].float(), atol=1e-4)
        and bool(torch.isfinite(qz_in.grad).all())
        and bool(torch.isfinite(pz_in.grad).all())
        and torch.equal(pz_in.grad[0], torch.zeros_like(pz_in.grad[0]))
        and torch.equal(pz_in.grad[2], torch.zeros_like(pz_in.grad[2]))
    )
    _record(failures, tag_edge, "empty-seq state & grad pass", carry_ok)

    # All-zero total length
    tag_empty = "[train/empty-batch] seqlens=[0, 0]"
    cu_e, _ = make_cu_seqlens([0, 0], device)
    qe = torch.empty(hq_z, 0, 64, device=device, dtype=torch.float16)
    ke = torch.empty(hkv_z, 0, 64, device=device, dtype=torch.float16)
    ve = torch.empty(hkv_z, 0, 64, device=device, dtype=torch.float16)
    we = torch.empty(hkv_z, 0, 64, device=device, dtype=torch.float16)
    be = torch.empty(hkv_z, 0, 64, device=device, dtype=torch.float16)
    ae = torch.empty(hkv_z, 0, 1, device=device, dtype=torch.float16)
    ge = torch.empty(hkv_z, 0, 1, device=device, dtype=torch.float16)
    pe = (torch.randn(2, hkv_z, 64, 64, device=device) * 0.1).half().contiguous()
    ye, fse, _, _ = custom_ops.deltanet_fwd(pe, qe, ke, ve, we, be, ae, ge, cu_e, hq_z, hkv_z, 64, 64, 64, True)
    _record(
        failures, tag_empty, "total_L=0 shapes & finite",
        tuple(ye.shape) == (hq_z, 0, 64) and tuple(fse.shape) == (2, hkv_z, 64, 64)
        and bool(torch.isfinite(fse).all()),
    )

    _summarize_failures("Suite 1 (Varlen Fwd/Bwd Kernels)", failures)
    return failures


# ===========================================================================
# SUITE 2: Raw Inference Decode Kernel (L = 1)
# ===========================================================================
def _make_infer_inputs(
    B: int, hq: int, hkv: int, device: torch.device, scale: float = 1.0, a_off: float = 0.0
) -> Dict[str, torch.Tensor]:
    def _r(*shape, s=1.0, off=0.0):
        return (torch.randn(*shape, device=device) * s + off).half().contiguous()

    return {
        "q": _r(B, hq, 64, s=scale),
        "k": _r(B, hkv, 64, s=scale),
        "v": _r(B, hkv, 64, s=scale),
        "a": _r(B, hkv, 1, off=a_off),
        "b": _r(B, hkv, 64),
        "w": _r(B, hkv, 64),
        "g": _r(B, hkv, 1),
    }


def _make_infer_state(B: int, hkv: int, device: torch.device, mode: str) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros(B, hkv, 64, 64, device=device, dtype=torch.float16)
    S = (torch.randn(B, hkv, 64, 64, device=device) * 0.1).half().contiguous()
    if mode == "mixed":
        S[::2] = 0
    return S


def _make_edge_infer_inputs(
    B: int, hq: int, hkv: int, device: torch.device, case: str
) -> Dict[str, torch.Tensor]:
    inp = _make_infer_inputs(B, hq, hkv, device)
    if case == "gate_sat":
        for name in ("a", "b", "w"):
            sign = (torch.rand_like(inp[name].float()) > 0.5).float() * 2.0 - 1.0
            inp[name] = (sign * 20.0).half().contiguous()
    elif case == "gamma_hi":
        inp["g"] = torch.full_like(inp["g"], 10.0)
    elif case == "gamma_lo":
        inp["g"] = torch.full_like(inp["g"], -15.0)
    elif case == "k_tiny":
        inp["k"] = (inp["k"].float() * 1e-3).half().contiguous()
    elif case == "k_big":
        inp["k"] = (inp["k"].float() * 50.0).half().contiguous()
    elif case == "k_zero":
        inp["k"] = torch.zeros_like(inp["k"])
    elif case == "qv_big":
        inp["q"] = (inp["q"].float() * 30.0).half().contiguous()
        inp["v"] = (inp["v"].float() * 30.0).half().contiguous()
    else:
        raise ValueError(f"Unknown edge case: {case}")
    return inp


def _run_raw_infer(
    inp: Dict[str, torch.Tensor], S: torch.Tensor, hq: int, hkv: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    Sk = S.clone().contiguous()
    o, s = apply_deltablock_infer(
        Sk, inp["q"], inp["k"], inp["v"], inp["b"], inp["w"], inp["a"], inp["g"],
        hq, hkv, 64, 64,
    )
    return o.reshape(S.shape[0], hq, 64), s


def _ref_raw_infer(
    inp: Dict[str, torch.Tensor], S: torch.Tensor, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _ref_core_step(
        inp["q"].to(dtype), inp["k"].to(dtype), inp["v"].to(dtype),
        inp["a"].to(dtype), inp["b"].to(dtype), inp["w"].to(dtype),
        inp["g"].to(dtype), S.to(dtype),
    )


def _check_drift(
    failures: List[str],
    tag: str,
    err_k: torch.Tensor,
    err_b: torch.Tensor,
) -> None:
    n = err_k.numel()
    w = max(1, n // 4)
    hk, tk = err_k[:w].mean().item(), err_k[-w:].mean().item()
    hb, tb = err_b[:w].mean().item(), err_b[-w:].mean().item()
    gk, gb = tk / (hk + 1e-30), tb / (hb + 1e-30)
    marks = sorted({0, n // 4, n // 2, n - 1})
    print(
        "    step drift | "
        + "  ".join(f"t={m}: k={err_k[m].item():.1e}/b={err_b[m].item():.1e}" for m in marks)
    )
    _record(
        failures, tag, "drift growth rate", gk <= max(3.0, 2.0 * gb),
        f"(kernel {gk:.2f}x | base16 {gb:.2f}x)",
    )
    _record(
        failures, tag, "drift tail level", tk <= ABS_CAP,
        f"(kernel {tk:.1e} | base16 {tb:.1e} | cap {ABS_CAP:.0e})",
    )


@torch.no_grad()
def test_infer_decode_kernels(
    quick: bool = False, device: Optional[torch.device] = None
) -> List[str]:
    print_section("SUITE 2: Raw Inference Decode Kernel (L = 1)")
    device = device or _require_gpu_and_ext()
    failures: List[str] = []

    # 2A: Batch sizes & state modes across GQA configs
    batches = [1, 17, 64] if quick else [1, 16, 17, 33, 64, 129, 256, 513]
    head_cfgs = [(8, 4)] if quick else [(8, 4), (8, 2)]
    for hq, hkv in head_cfgs:
        for B in batches:
            for mode in ("zero", "mixed", "rand"):
                tag = f"[core/batch] hq={hq} hkv={hkv} B={B} state={mode}"
                inp = _make_infer_inputs(B, hq, hkv, device, scale=0.5)
                S = _make_infer_state(B, hkv, device, mode)
                o_k, s_k = _run_raw_infer(inp, S, hq, hkv)
                o_b, s_b = _ref_raw_infer(inp, S, torch.float16)
                o_t, s_t = _ref_raw_infer(inp, S, torch.float32)
                print(f"\n  {tag}")
                _print_table_header("fp32")
                _dual_check(failures, tag, "o_h", o_k, o_b, o_t, lead_dims=2)
                _dual_check(failures, tag, "state", s_k, s_b, s_t, lead_dims=2)

    # 2B: Stress sweeps & numerical edge cases
    stress_batches = (17,) if quick else (17, 513)
    stress_pairs = [(0.1, 0.0), (10.0, 0.0), (1.0, 5.0), (3.0, 7.0)]
    if not quick:
        stress_pairs.extend([(3.0, 0.0), (1.0, 7.0)])
    edge_cases = ["gate_sat", "gamma_hi", "gamma_lo", "k_tiny", "k_big", "k_zero", "qv_big"]

    for B in stress_batches:
        for scale, a_off in stress_pairs:
            tag = f"[core/stress] B={B} scale={scale} alpha_bias={a_off}"
            print(f"\n  {tag}")
            _print_table_header("fp32")
            inp = _make_infer_inputs(B, 8, 4, device, scale=scale, a_off=a_off)
            S = _make_infer_state(B, 4, device, "rand")
            o_k, s_k = _run_raw_infer(inp, S, 8, 4)
            o_b, s_b = _ref_raw_infer(inp, S, torch.float16)
            o_t, s_t = _ref_raw_infer(inp, S, torch.float32)
            _dual_check(failures, tag, "o_h", o_k, o_b, o_t, lead_dims=2)
            _dual_check(failures, tag, "state", s_k, s_b, s_t, lead_dims=2)

        for case in edge_cases:
            tag = f"[core/edge:{case}] B={B}"
            print(f"\n  {tag}")
            _print_table_header("fp32")
            inp = _make_edge_infer_inputs(B, 8, 4, device, case)
            S = _make_infer_state(B, 4, device, "rand")
            o_k, s_k = _run_raw_infer(inp, S, 8, 4)
            o_b, s_b = _ref_raw_infer(inp, S, torch.float16)
            o_t, s_t = _ref_raw_infer(inp, S, torch.float32)
            _dual_check(failures, tag, "o_h", o_k, o_b, o_t, lead_dims=2)
            _dual_check(failures, tag, "state", s_k, s_b, s_t, lead_dims=2)

    # 2C: Recurrence ablations (isolate each term cleanly)
    ablations = {
        "full": {},
        "no_erase (b=-30)": {"b": -30.0},
        "no_write (w=-30)": {"w": -30.0},
        "alpha=1 (a=+30)": {"a": 30.0},
        "alpha=0 (a=-30)": {"a": -30.0},
        "readout_only": {"a": 30.0, "b": -30.0, "w": -30.0},
    }
    S_ab = _make_infer_state(64, 4, device, "rand")
    for ab_name, overrides in ablations.items():
        tag = f"[core/ablation:{ab_name}] B=64"
        print(f"\n  {tag}")
        _print_table_header("fp32")
        inp = _make_infer_inputs(64, 8, 4, device, scale=0.5)
        for field, val in overrides.items():
            inp[field] = torch.full_like(inp[field], val)
        o_k, s_k = _run_raw_infer(inp, S_ab, 8, 4)
        o_b, s_b = _ref_raw_infer(inp, S_ab, torch.float16)
        o_t, s_t = _ref_raw_infer(inp, S_ab, torch.float32)
        _dual_check(failures, tag, "o_h", o_k, o_b, o_t, lead_dims=2)
        _dual_check(failures, tag, "state", s_k, s_b, s_t, lead_dims=2)

    # 2D: Batch permutation / split invariance & sentinel canary guard
    B_inv = 256 if quick else 1024
    tag_inv = f"[core/invariance] B={B_inv}"
    print(f"\n  {tag_inv}")
    inp_inv = _make_infer_inputs(B_inv, 8, 4, device, scale=0.5)
    S_inv = _make_infer_state(B_inv, 4, device, "mixed")
    o0, s0 = _run_raw_infer(inp_inv, S_inv, 8, 4)

    perm = torch.randperm(B_inv, device=device)
    inp_p = {k: val[perm].contiguous() for k, val in inp_inv.items()}
    o1, s1 = _run_raw_infer(inp_p, S_inv[perm].contiguous(), 8, 4)
    ro = compute_tensor_metrics(o1, o0[perm])[0]
    rs = compute_tensor_metrics(s1, s0[perm])[0]
    _record(
        failures, tag_inv, "row permutation invariance",
        ro < INV_TOL and rs < INV_TOL, f"(rel out={ro:.1e}, state={rs:.1e})",
    )

    chunk = 16
    parts = [
        _run_raw_infer(
            {k: val[i : i + chunk].contiguous() for k, val in inp_inv.items()},
            S_inv[i : i + chunk].contiguous(), 8, 4,
        )
        for i in range(0, B_inv, chunk)
    ]
    o2 = torch.cat([p[0] for p in parts], dim=0)
    s2 = torch.cat([p[1] for p in parts], dim=0)
    ro2 = compute_tensor_metrics(o2, o0)[0]
    rs2 = compute_tensor_metrics(s2, s0)[0]
    _record(
        failures, tag_inv, f"B={B_inv} == {B_inv // chunk}x(B={chunk})",
        ro2 < INV_TOL and rs2 < INV_TOL, f"(rel out={ro2:.1e}, state={rs2:.1e})",
    )

    tag_can = "[core/canary] B=64 pad=4096"
    print(f"\n  {tag_can}")
    PAD, SENT = 4096, 1234.0
    inp_c = _make_infer_inputs(64, 8, 4, device, scale=0.5)
    S_c = _make_infer_state(64, 4, device, "rand")
    o_ref, s_ref = _run_raw_infer(inp_c, S_c, 8, 4)
    all_t = dict(inp_c, S=S_c)
    bufs, views = {}, {}
    for k, val in all_t.items():
        buf = torch.full((val.numel() + 2 * PAD,), SENT, dtype=torch.float16, device=device)
        view = buf[PAD : PAD + val.numel()].view_as(val)
        view.copy_(val)
        bufs[k], views[k] = buf, view
    o_c, s_c = apply_deltablock_infer(
        views["S"], views["q"], views["k"], views["v"], views["b"], views["w"],
        views["a"], views["g"], 8, 4, 64, 64,
    )
    torch.cuda.synchronize()
    canary_bad = []
    for k, val in all_t.items():
        buf, n_el = bufs[k], val.numel()
        if not bool((buf[:PAD] == SENT).all()) or not bool((buf[PAD + n_el :] == SENT).all()):
            canary_bad.append(f"{k}:oob_write")
        if k != "S" and not torch.equal(buf[PAD : PAD + n_el].view_as(val), val):
            canary_bad.append(f"{k}:input_mutated")
    if not (torch.equal(o_c.reshape(64, 8, 64), o_ref) and torch.equal(s_c, s_ref)):
        canary_bad.append("output_mismatch")
    _record(failures, tag_can, "sentinel guard & read-only", not canary_bad, ", ".join(canary_bad))

    # 2E: Long autoregressive rollout across retention regimes
    T_long = 256 if quick else 1024
    long_batches = (16,) if quick else (16, 129)
    alpha_biases = (0.0, 5.0) if quick else (0.0, 5.0, 7.0)
    for B_l in long_batches:
        for a_bias in alpha_biases:
            alpha_approx = torch.sigmoid(torch.tensor(a_bias)).item()
            tag_l = f"[core/long-rollout T={T_long}] B={B_l} alpha~{alpha_approx:.4f}"
            print(f"\n  {tag_l}")
            q = (torch.randn(B_l, T_long, 8, 64, device=device) * 0.5).half().contiguous()
            k = (torch.randn(B_l, T_long, 4, 64, device=device) * 0.5).half().contiguous()
            v = (torch.randn(B_l, T_long, 4, 64, device=device) * 0.5).half().contiguous()
            a = (torch.randn(B_l, T_long, 4, 1, device=device) * 0.5 + a_bias).half().contiguous()
            b = torch.randn(B_l, T_long, 4, 64, device=device).half().contiguous()
            w = torch.randn(B_l, T_long, 4, 64, device=device).half().contiguous()
            g = (torch.randn(B_l, T_long, 4, 1, device=device) * 0.5).half().contiguous()

            s_k = torch.zeros(B_l, 4, 64, 64, device=device, dtype=torch.float16)
            s_b = s_k.clone()
            s_t = torch.zeros(B_l, 4, 64, 64, device=device, dtype=torch.float32)
            os_k, os_b, os_t = [], [], []
            for t in range(T_long):
                st = dict(
                    q=q[:, t].contiguous(), k=k[:, t].contiguous(), v=v[:, t].contiguous(),
                    a=a[:, t].contiguous(), b=b[:, t].contiguous(), w=w[:, t].contiguous(),
                    g=g[:, t].contiguous(),
                )
                ok_, s_k = apply_deltablock_infer(
                    s_k, st["q"], st["k"], st["v"], st["b"], st["w"], st["a"], st["g"],
                    8, 4, 64, 64,
                )
                ob, s_b = _ref_raw_infer(st, s_b, torch.float16)
                ot, s_t = _ref_raw_infer(st, s_t, torch.float32)
                os_k.append(ok_.reshape(B_l, 8 * 64))
                os_b.append(ob.reshape(B_l, 8 * 64))
                os_t.append(ot.reshape(B_l, 8 * 64))

            o_k = torch.stack(os_k, dim=1)
            o_b = torch.stack(os_b, dim=1)
            o_t = torch.stack(os_t, dim=1)
            _print_table_header("fp32")
            _dual_check(failures, tag_l, "o_h (all steps)", o_k, o_b, o_t, lead_dims=2)
            _dual_check(failures, tag_l, "final_states", s_k, s_b, s_t, lead_dims=2)
            ek = ((o_k.double() - o_t.double()).norm(dim=-1) / (o_t.double().norm(dim=-1) + 1e-30)).mean(0)
            eb = ((o_b.double() - o_t.double()).norm(dim=-1) / (o_t.double().norm(dim=-1) + 1e-30)).mean(0)
            _check_drift(failures, tag_l, ek, eb)

    _summarize_failures("Suite 2 (Inference Decode Kernel)", failures)
    return failures


# ===========================================================================
# SUITE 3: End-to-End DeltaBlock Modules & Runtime Environment
# ===========================================================================
def _build_model_pair(
    d_model: int,
    num_q_heads: int,
    num_kv_heads: int,
    hidden_dim: Optional[int],
    device: torch.device,
) -> Tuple[DeltaBlock, DeltaBlock]:
    m16 = DeltaBlock(
        d_model=d_model, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        d_k=64, d_v=64, chunk_size=64, hidden_dim=hidden_dim, drop_path_rate=0.0,
    ).to(device).half()
    m32 = copy.deepcopy(m16).float()
    return m16, m32


def test_end_to_end_modules(
    quick: bool = False, device: Optional[torch.device] = None
) -> List[str]:
    print_section("SUITE 3: End-to-End DeltaBlock (Training, Handoff, Rollout & Runtime Guards)")
    device = device or _require_gpu_and_ext()
    failures: List[str] = []

    d_model, nq, nkv, hidden_dim = 512, 8, 4, 1024
    m16, m32 = _build_model_pair(d_model, nq, nkv, hidden_dim, device)

    # 3A: End-to-end DeltaBlock Training (varlen kernel vs _forward_batch)
    m16.train()
    m32.train()
    train_configs = (
        [[64, 128, 65, 1]]
        if quick
        else [[64, 128, 65, 1], [1041, 756], [518, 1988, 700, 300]]
    )

    for seqlens in train_configs:
        B = len(seqlens)
        for use_past in (False, True):
            tag = f"[e2e/train] seqlens={seqlens} past={use_past}"
            print(f"\n  {tag}")
            _print_table_header("fp32")
            x_pad, x_var, mask, cu = make_varlen_inputs(seqlens, d_model, scale=0.5, device=device)
            past16 = (torch.randn(B, nkv, 64, 64, device=device) * 0.1).half().contiguous() if use_past else None
            g_out = torch.randn(x_var.shape, device=device, dtype=torch.float32)

            def _run_blk(mod: DeltaBlock, use_k: bool, dtype: torch.dtype):
                mod.zero_grad(set_to_none=True)
                if use_k:
                    xi = x_var.to(dtype).detach().clone().contiguous().requires_grad_(True)
                    pi = past16.to(dtype).detach().clone().contiguous().requires_grad_(True) if past16 is not None else None
                    out, fs = mod(xi, cu_seqlens=cu, past_states=pi, use_kernel=True)
                    out_v = out
                else:
                    xi = x_pad.to(dtype).detach().clone().contiguous().requires_grad_(True)
                    pi = past16.to(dtype).detach().clone().contiguous().requires_grad_(True) if past16 is not None else None
                    out, fs = mod._forward_batch(xi, padding_mask=mask, past_states=pi)
                    out_v = out[mask].contiguous()
                (out_v.float() * g_out).sum().backward()
                res = {
                    "out": out_v.detach(),
                    "final_states": fs.detach(),
                    "grad(x)": (xi.grad if use_k else xi.grad[mask]).detach(),
                    "grad(W_proj)": mod.W_proj.weight.grad.detach(),
                }
                if pi is not None:
                    res["grad(past_states)"] = pi.grad.detach()
                return res

            truth = _run_blk(m32, False, torch.float32)
            base16 = _run_blk(m16, False, torch.float16)
            kern16 = _run_blk(m16, True, torch.float16)

            lead_map = {"out": 1, "grad(x)": 1, "final_states": 2}
            for name in truth:
                _dual_check(
                    failures, tag, name, kern16[name], base16[name], truth[name],
                    lead_dims=lead_map.get(name), use_ratio=not name.startswith("grad("),
                )

    # forward_only (no_grad) vs autograd consistency on DeltaBlock
    tag_fo = "[e2e/forward_only] seqlens=[64, 128, 65, 1]"
    print(f"\n  {tag_fo}")
    _, x_fo, _, cu_fo = make_varlen_inputs([64, 128, 65, 1], d_model, scale=0.5, device=device)
    with torch.no_grad():
        y0, s0 = m16(x_fo, cu_seqlens=cu_fo, past_states=None, use_kernel=True)
    y1, s1 = m16(x_fo, cu_seqlens=cu_fo, past_states=None, use_kernel=True)
    ry = compute_tensor_metrics(y0, y1)[0]
    rs = compute_tensor_metrics(s0, s1)[0]
    _record(
        failures, tag_fo, "no_grad == autograd",
        ry < INV_TOL and rs < INV_TOL, f"(rel out={ry:.1e}, state={rs:.1e})",
    )

    # 3B: Decode single-step, rollout, x-scale sweep, and Prefill -> Decode handoff
    m16.eval()
    m32.eval()
    with torch.no_grad():
        # Reference baseline consistency (ref_decode_block_step vs _forward_batch L=1)
        tag_ref = "[e2e/ref-consistency] L=1"
        print(f"\n  {tag_ref}")
        for use_past in (False, True):
            xr = (torch.randn(16, 1, d_model, device=device) * 0.5).float().contiguous()
            pr = (torch.randn(16, nkv, 64, 64, device=device) * 0.1).float().contiguous() if use_past else None
            yr, sr = _ref_decode_block_step(m32, xr[:, 0].contiguous(), pr)
            yb, sb = m32._forward_batch(xr, past_states=pr)
            r1 = compute_tensor_metrics(yr, yb[:, 0])[0]
            r2 = compute_tensor_metrics(sr, sb)[0]
            _record(
                failures, tag_ref, f"ref_step==batch(past={use_past})",
                r1 < REF_CONSIST_TOL and r2 < REF_CONSIST_TOL, f"(out={r1:.1e}, state={r2:.1e})",
            )

        # Single-step L=1 across batch sizes & 2D vs 3D input shapes
        infer_batches = [1, 17, 64] if quick else [1, 16, 17, 64, 129, 256, 513]
        for B in infer_batches:
            x_dims = (3, 2) if B in (1, 17, 64) else (3,)
            for use_past in (False, True):
                for x_dim in x_dims:
                    tag_1s = f"[e2e/infer-1step] B={B} past={use_past} x_dim={x_dim}"
                    print(f"\n  {tag_1s}")
                    _print_table_header("fp32")
                    x3 = (torch.randn(B, 1, d_model, device=device) * 0.5).half().contiguous()
                    x_in = x3 if x_dim == 3 else x3[:, 0].clone().contiguous()
                    x2 = x3[:, 0].clone().contiguous()
                    past16 = (torch.randn(B, nkv, 64, 64, device=device) * 0.1).half().contiguous() if use_past else None
                    pk = past16.clone().contiguous() if past16 is not None else None

                    yk, sk = m16(x_in, past_states=pk, use_kernel=True)
                    yb, sb = _ref_decode_block_step(m16, x2, past16)
                    yt, st = _ref_decode_block_step(m32, x2.float(), past16.float() if past16 is not None else None)

                    inplace_ok = (pk is None) or (sk.data_ptr() == pk.data_ptr() and torch.equal(sk, pk))
                    shape_ok = (yk.shape == x_in.shape) and (tuple(sk.shape) == (B, nkv, 64, 64))
                    _record(failures, tag_1s, "shape/inplace", inplace_ok and shape_ok)
                    _dual_check(failures, tag_1s, "out", yk.reshape(B, d_model), yb, yt, lead_dims=1)
                    _dual_check(failures, tag_1s, "final_states", sk, sb, st, lead_dims=2)

        # X-scale sweep
        for scale in ((0.05, 3.0) if quick else (0.05, 0.25, 1.0, 3.0)):
            tag_xs = f"[e2e/x-scale] B=17 scale={scale}"
            print(f"\n  {tag_xs}")
            _print_table_header("fp32")
            x3 = (torch.randn(17, 1, d_model, device=device) * scale).half().contiguous()
            x2 = x3[:, 0].clone().contiguous()
            past = (torch.randn(17, nkv, 64, 64, device=device) * 0.1).half().contiguous()
            yk, sk = m16(x3, past_states=past.clone(), use_kernel=True)
            yb, sb = _ref_decode_block_step(m16, x2, past)
            yt, st = _ref_decode_block_step(m32, x2.float(), past.float())
            _dual_check(failures, tag_xs, "out", yk.reshape(17, d_model), yb, yt, lead_dims=1)
            _dual_check(failures, tag_xs, "final_states", sk, sb, st, lead_dims=2)

        # Prefill (varlen kernel) -> Decode rollout handoff
        T_hand = 24 if quick else 64
        hand_configs = (
            [[300, 129, 64, 1, 65, 63, 128, 257]]
            if quick
            else [
                [300, 129, 64, 1, 65, 63, 128, 257, 2, 17, 511, 100, 64, 64, 33, 1000],
                torch.randint(1, 300, (64,), generator=torch.Generator().manual_seed(49)).tolist(),
            ]
        )
        for seqlens in hand_configs:
            B = len(seqlens)
            tag_h = f"[e2e/handoff] B={B} prefill_toks={sum(seqlens)} -> decode T={T_hand}"
            print(f"\n  {tag_h}")
            _print_table_header("fp32")
            x_pad, x_var, mask, cu = make_varlen_inputs(seqlens, d_model, scale=0.5, device=device)
            x_new = (torch.randn(B, T_hand, d_model, device=device) * 0.5).half().contiguous()

            m16.train()
            _, s_pre_k = m16(x_var, cu_seqlens=cu, past_states=None, use_kernel=True)
            m16.eval()
            _, s_pre_t = m32._forward_batch(x_pad.float(), padding_mask=mask)
            _, s_pre_b = m16._forward_batch(x_pad, padding_mask=mask)
            _dual_check(failures, tag_h, "prefill final_states", s_pre_k, s_pre_b, s_pre_t, lead_dims=2)

            sk, sb, st = s_pre_k.clone(), s_pre_b.clone(), s_pre_t.clone()
            ysk, ysb, yst = [], [], []
            for t in range(T_hand):
                xt3 = x_new[:, t : t + 1].contiguous()
                xt2 = x_new[:, t].contiguous()
                yk, sk = m16(xt3, past_states=sk, use_kernel=True)
                yb, sb = _ref_decode_block_step(m16, xt2, sb)
                yt, st = _ref_decode_block_step(m32, xt2.float(), st)
                ysk.append(yk[:, 0])
                ysb.append(yb)
                yst.append(yt)
            y_k = torch.stack(ysk, dim=1)
            y_b = torch.stack(ysb, dim=1)
            y_t = torch.stack(yst, dim=1)
            _dual_check(failures, tag_h, "decode out", y_k, y_b, y_t, lead_dims=2)
            _dual_check(failures, tag_h, "decode final_states", sk, sb, st, lead_dims=2)
            ek = ((y_k.double() - y_t.double()).norm(dim=-1) / (y_t.double().norm(dim=-1) + 1e-30)).mean(0)
            eb = ((y_b.double() - y_t.double()).norm(dim=-1) / (y_t.double().norm(dim=-1) + 1e-30)).mean(0)
            _check_drift(failures, tag_h, ek, eb)

        # 3C: Runtime Environment (In-place repeatability, Side stream, CUDA Graph, Multi-GPU, API guards)
        repeats = 15 if quick else 40
        for B_r in ((16, 513) if quick else (16, 513, 1024)):
            tag_race = f"[e2e/inplace-race] B={B_r}"
            print(f"\n  {tag_race}")
            xr = (torch.randn(B_r, 1, d_model, device=device) * 0.5).half().contiguous()
            pr = (torch.randn(B_r, nkv, 64, 64, device=device) * 0.1).half().contiguous()
            y_ref, s_ref = m16(xr, past_states=pr.clone(), use_kernel=True)
            diff_cnt = 0
            for _ in range(repeats):
                yi, si = m16(xr, past_states=pr.clone(), use_kernel=True)
                diff_cnt += int(not (torch.equal(yi, y_ref) and torch.equal(si, s_ref)))
            _record(
                failures, tag_race, f"bitwise repeat x{repeats}",
                diff_cnt == 0, f"({diff_cnt}/{repeats} mismatches)",
            )

        tag_rt = "[e2e/runtime] B=64 stream & CUDAGraph"
        print(f"\n  {tag_rt}")
        B_rt = 64
        x_rt = (torch.randn(B_rt, 1, d_model, device=device) * 0.5).half().contiguous()
        p_rt = (torch.randn(B_rt, nkv, 64, 64, device=device) * 0.1).half().contiguous()
        y0_rt, s0_rt = m16(x_rt, past_states=p_rt.clone(), use_kernel=True)
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        bad_stream = 0
        for _ in range(10):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                junk = torch.randn(1024, 1024, device=device)
                junk = torch.tanh(junk @ junk)
                xs, ps = x_rt.clone(), p_rt.clone()
                ys, ss = m16(xs, past_states=ps, use_kernel=True)
            torch.cuda.synchronize()
            bad_stream += int(not (torch.equal(ys, y0_rt) and torch.equal(ss, s0_rt)))
        _record(failures, tag_rt, "side CUDA stream", bad_stream == 0)

        try:
            sx, sp = x_rt.clone(), p_rt.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    m16(sx, past_states=sp, use_kernel=True)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                yg, sg = m16(sx, past_states=sp, use_kernel=True)

            sx.copy_(x_rt)
            sp.copy_(p_rt)
            graph.replay()
            torch.cuda.synchronize()
            ok1 = torch.equal(yg, y0_rt) and torch.equal(sg, s0_rt)

            x2 = (torch.randn(B_rt, 1, d_model, device=device) * 0.5).half().contiguous()
            p2 = (torch.randn(B_rt, nkv, 64, 64, device=device) * 0.1).half().contiguous()
            ye, se = m16(x2, past_states=p2.clone(), use_kernel=True)
            sx.copy_(x2)
            sp.copy_(p2)
            graph.replay()
            torch.cuda.synchronize()
            ok2 = torch.equal(yg, ye) and torch.equal(sg, se)
            _record(failures, tag_rt, "CUDA graph replay (new inputs)", ok1 and ok2)
        except Exception as exc:  # noqa: BLE001
            _record(failures, tag_rt, "CUDA graph replay (new inputs)", False, f"({exc})")

        if torch.cuda.device_count() >= 2:
            dev1 = torch.device("cuda:1")
            m_dev1 = copy.deepcopy(m16).to(dev1)
            x1, p1 = x_rt.to(dev1), p_rt.to(dev1).clone()
            with torch.cuda.device(0):
                y1, s1 = m_dev1(x1, past_states=p1, use_kernel=True)
            torch.cuda.synchronize(dev1)
            ry = compute_tensor_metrics(y1.to(device), y0_rt)[0]
            rs = compute_tensor_metrics(s1.to(device), s0_rt)[0]
            _record(failures, tag_rt, "non-default device cuda:1", ry < INV_TOL and rs < INV_TOL)

        # Input validation & alignment guards (run last)
        tag_misc = "[e2e/api-guards] B=32"
        print(f"\n  {tag_misc}")
        B_m = 32
        xm = (torch.randn(B_m, 1, d_model, device=device) * 0.5).half().contiguous()
        pm = (torch.randn(B_m, nkv, 64, 64, device=device) * 0.1).half().contiguous()
        y_m0, s_m0 = m16(xm, past_states=pm.clone(), use_kernel=True)
        y_m1, s_m1 = m16(xm, past_states=pm.clone(), use_kernel=True)
        _record(failures, tag_misc, "deterministic bitwise", torch.equal(y_m0, y_m1) and torch.equal(s_m0, s_m1))

        y_2d, s_2d = m16(xm[:, 0].clone().contiguous(), past_states=pm.clone(), use_kernel=True)
        _record(failures, tag_misc, "2D == 3D input", torch.equal(y_m0[:, 0], y_2d) and torch.equal(s_m0, s_2d))

        pf32 = pm.float().contiguous()
        y_f, s_f = m16(xm, past_states=pf32, use_kernel=True)
        _record(
            failures, tag_misc, "past fp32 auto-cast & immutable",
            torch.equal(y_m0, y_f) and torch.equal(s_m0, s_f) and torch.equal(pf32, pm.float()),
        )

        p_nc = torch.randn(nkv, B_m, 64, 64, device=device).half().mul_(0.1).transpose(0, 1)
        y_nc, s_nc = m16(xm, past_states=p_nc, use_kernel=True)
        y_cc, s_cc = m16(xm, past_states=p_nc.contiguous().clone(), use_kernel=True)
        _record(failures, tag_misc, "past non-contiguous", torch.equal(y_nc, y_cc) and torch.equal(s_nc, s_cc))

        x_2tok = (torch.randn(B_m, 2, d_model, device=device) * 0.5).half().contiguous()
        x_nc = x_2tok[:, :1]
        y_xnc, s_xnc = m16(x_nc, past_states=pm.clone(), use_kernel=True)
        y_xcc, s_xcc = m16(x_nc.contiguous(), past_states=pm.clone(), use_kernel=True)
        _record(failures, tag_misc, "x non-contiguous", torch.equal(y_xnc, y_xcc) and torch.equal(s_xnc, s_xcc))

        l2_rejected = False
        try:
            m16(x_2tok, past_states=pm.clone(), use_kernel=True)
        except AssertionError:
            l2_rejected = True
        _record(failures, tag_misc, "eval L=2 rejected cleanly", l2_rejected)

        yz, sz = m16(xm, past_states=None, use_kernel=True)
        yz2, sz2 = m16(xm, past_states=torch.zeros_like(pm), use_kernel=True)
        _record(failures, tag_misc, "past=None == zeros", torch.equal(yz, yz2) and torch.equal(sz, sz2))

        half_B = B_m // 2
        y_sub, s_sub = m16(xm[:half_B].clone(), past_states=pm[:half_B].clone(), use_kernel=True)
        r_y = compute_tensor_metrics(y_sub, y_m0[:half_B])[0]
        r_s = compute_tensor_metrics(s_sub, s_m0[:half_B])[0]
        _record(
            failures, tag_misc, "batch slice independence",
            r_y < MODEL_INV_TOL and r_s < MODEL_INV_TOL, f"(out={r_y:.1e}, state={r_s:.1e})",
        )

        # Misaligned pointer check (kept last)
        buf = torch.empty(pm.numel() + 8, device=device, dtype=torch.float16)
        p_mis = buf[1 : 1 + pm.numel()].view_as(pm)
        p_mis.copy_(pm)
        assert p_mis.is_contiguous() and p_mis.data_ptr() % 16 != 0
        try:
            ym, sm = m16(xm, past_states=p_mis, use_kernel=True)
            _record(
                failures, tag_misc, "misaligned past handling",
                torch.equal(y_m0, ym) and torch.equal(s_m0, sm), "(aligned copy match)",
            )
        except (AssertionError, RuntimeError) as exc:
            msg = str(exc)
            clean = ("align" in msg.lower()) and ("CUDA error" not in msg)
            _record(
                failures, tag_misc, "misaligned past handling", clean,
                "(clean rejection)" if clean else f"(unexpected error: {msg[:80]})",
            )

    _summarize_failures("Suite 3 (End-to-End DeltaBlock)", failures)
    return failures


# ===========================================================================
# SUITE 4: torch.compile (Inductor) Compatibility
# ===========================================================================
def test_torch_compile_inductor(
    quick: bool = False, device: Optional[torch.device] = None
) -> List[str]:
    print_section("SUITE 4: torch.compile (Inductor) Training & Decode")
    device = device or _require_gpu_and_ext()
    failures: List[str] = []

    d_model, nq, nkv = 512, 8, 4
    seqlens = [64, 65] if quick else [64, 128, 65, 1]
    B = len(seqlens)

    # Reset dynamo cache before compiling
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()

    # 4A: Compiled Training Forward + Backward on DeltaBlock
    tag_ct = f"[compile/train] seqlens={seqlens}"
    print(f"\n  {tag_ct}")
    m_eager, _ = _build_model_pair(d_model, nq, nkv, hidden_dim=1024, device=device)
    m_comp = copy.deepcopy(m_eager)
    m_eager.train()
    m_comp.train()

    try:
        compiled_train = torch.compile(m_comp, backend="inductor")
        _, x_var, _, cu = make_varlen_inputs(seqlens, d_model, scale=0.5, device=device)
        past = (torch.randn(B, nkv, 64, 64, device=device) * 0.1).half().contiguous()
        gy = torch.randn_like(x_var)
        gp = torch.randn_like(past)

        def _step(fn, detach_clone: bool = True):
            xi = x_var.detach().clone().requires_grad_(True) if detach_clone else x_var
            pi = past.detach().clone().requires_grad_(True) if detach_clone else past
            y, fs = fn(xi, cu_seqlens=cu, past_states=pi, use_kernel=True)
            loss = (y * gy).sum() + (fs * gp).sum()
            loss.backward()
            return y.detach(), fs.detach(), xi.grad.detach(), pi.grad.detach()

        m_eager.zero_grad(set_to_none=True)
        ye, fse, gxe, gpe = _step(m_eager)
        m_comp.zero_grad(set_to_none=True)
        yc, fsc, gxc, gpc = _step(compiled_train)

        ok_fwd = (
            check_allclose("compile train y", yc, ye, atol=2e-2, rtol=2e-2, verbose=False)[0]
            and check_allclose("compile train fs", fsc, fse, atol=2e-2, rtol=2e-2, verbose=False)[0]
        )
        ok_bwd = (
            check_allclose("compile train dx", gxc, gxe, atol=5e-2, rtol=5e-2, verbose=False)[0]
            and check_allclose("compile train dpast", gpc, gpe, atol=5e-2, rtol=5e-2, verbose=False)[0]
            and check_allclose(
                "compile train dW_proj",
                m_comp.W_proj.weight.grad, m_eager.W_proj.weight.grad,
                atol=5e-2, rtol=5e-2, verbose=False,
            )[0]
        )
        _record(failures, tag_ct, "compiled DeltaBlock train fwd", ok_fwd)
        _record(failures, tag_ct, "compiled DeltaBlock train bwd", ok_bwd)
    except Exception as exc:  # noqa: BLE001
        _record(failures, tag_ct, "torch.compile train execution", False, f"({type(exc).__name__}: {exc})")

    # 4B: Compiled Decode Step (DeltaBlock eval + raw apply_deltablock_infer)
    tag_ci = "[compile/infer] B=16 L=1"
    print(f"\n  {tag_ci}")
    try:
        m_eager.eval()
        m_comp_eval = copy.deepcopy(m_eager).eval()
        compiled_eval = torch.compile(m_comp_eval, backend="inductor")

        x_dec = (torch.randn(16, 1, d_model, device=device) * 0.5).half().contiguous()
        p_dec = (torch.randn(16, nkv, 64, 64, device=device) * 0.1).half().contiguous()
        with torch.no_grad():
            ye_i, se_i = m_eager(x_dec, past_states=p_dec.clone(), use_kernel=True)
            yc_i, sc_i = compiled_eval(x_dec, past_states=p_dec.clone(), use_kernel=True)

        ok_blk_dec = (
            check_allclose("compile eval y", yc_i, ye_i, atol=2e-2, rtol=2e-2, verbose=False)[0]
            and check_allclose("compile eval state", sc_i, se_i, atol=2e-2, rtol=2e-2, verbose=False)[0]
        )
        _record(failures, tag_ci, "compiled DeltaBlock eval decode", ok_blk_dec)

        # Raw apply_deltablock_infer under inductor
        inp_r = _make_infer_inputs(16, nq, nkv, device, scale=0.5)
        S_r = _make_infer_state(16, nkv, device, "rand")

        def _raw_step(s_in, q, k, v, b, w, a, g):
            return apply_deltablock_infer(s_in.clone(), q, k, v, b, w, a, g, nq, nkv, 64, 64)

        compiled_raw = torch.compile(_raw_step, backend="inductor")
        with torch.no_grad():
            or_e, sr_e = _raw_step(S_r, inp_r["q"], inp_r["k"], inp_r["v"], inp_r["b"], inp_r["w"], inp_r["a"], inp_r["g"])
            or_c, sr_c = compiled_raw(S_r, inp_r["q"], inp_r["k"], inp_r["v"], inp_r["b"], inp_r["w"], inp_r["a"], inp_r["g"])
        _record(
            failures, tag_ci, "compiled raw deltanet_infer",
            torch.equal(or_c, or_e) and torch.equal(sr_c, sr_e),
        )
    except Exception as exc:  # noqa: BLE001
        _record(failures, tag_ci, "torch.compile infer execution", False, f"({type(exc).__name__}: {exc})")

    _summarize_failures("Suite 4 (torch.compile Inductor)", failures)
    return failures


# ===========================================================================
# Runner & CLI
# ===========================================================================
def run_all_tests(
    quick: bool = False, seed: int = 42, suite: str = "all"
) -> None:
    seed_everything(seed, disable_tf32=True)
    device = _require_gpu_and_ext()

    all_failures: List[str] = []
    if suite in ("all", "train"):
        all_failures.extend(test_varlen_fwd_bwd_kernels(quick=quick, device=device))
    if suite in ("all", "infer"):
        all_failures.extend(test_infer_decode_kernels(quick=quick, device=device))
    if suite in ("all", "e2e"):
        all_failures.extend(test_end_to_end_modules(quick=quick, device=device))
    if suite in ("all", "compile"):
        all_failures.extend(test_torch_compile_inductor(quick=quick, device=device))

    print("\n" + "#" * 72)
    if all_failures:
        print(f"# CORRECTNESS SUMMARY: FAILED ({len(all_failures)} failure(s))")
        print("#" * 72)
        for msg in all_failures[:30]:
            print(f"  - {msg}")
        raise AssertionError(f"{len(all_failures)} correctness test(s) failed.")
    print("# CORRECTNESS SUMMARY: ALL SUITES PASSED")
    print("#" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run comprehensive DeltaNet v2 correctness suites."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a compact subset of shapes/steps for fast smoke verification.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible test generation (default: 42).",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["all", "train", "infer", "e2e", "compile"],
        help="Which suite to run (default: all).",
    )
    args = parser.parse_args()
    run_all_tests(quick=args.quick, seed=args.seed, suite=args.suite)


if __name__ == "__main__":
    main()
