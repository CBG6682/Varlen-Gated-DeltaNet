"""
Comprehensive correctness and regression test suite for Gated DeltaNet-2 CUDA kernels.

Covers three verification suites aligned with the GDN-2 Turing (SM75) kernel stack:
  1. Varlen Forward & Backward Kernels:
     - Raw Block Ops (RMSNorm, Residual+RMSNorm, SwiGLU, ResidualAdd) & `is_training=False == True`
     - Raw `proj_conv_prep_fwd/bwd`, `chunk_wy`, `chunk_scan`, `output_combine` & `deltablock_fwd/bwd`
     - Mandatory Cross-Check: Chunkwise (CUDA & PyTorch) == Exact Token-by-Token Recurrence (FP32)
     - Full `apply_deltablock` across GQA ratios, Option 5A PreNorm SwiGLU (`hidden_dim=None` vs `!=None`),
       `beta_guard`, `allow_neg_eigval`, FP32 state + FP16 conv_state gradients, zero-len & tiny seqs
  2. Inference Decode Kernel (`L = 1` Super-Fused `apply_deltablock_infer`):
     - Single-step decode across batch sizes, GQA ratios, `hidden_dim=None` & `!=None`,
       FP32 `past_states` + FP16 `conv_state` in-place updates
     - Extreme stress/edge cases (`x`-scale, `dt_bias`/`A_log` extremes, `beta_guard`, `allow_neg_eigval`, near-zero `k`)
     - Structural ablations, slice invariance, Canary exact orthogonal-key check, long rollout drift
  3. End-to-End `DeltaBlock` Modules & Runtime Environment:
     - Varlen Fwd+Bwd on `DeltaBlock` (with and without SwiGLU MLP)
     - Single-step decode (2D & 3D `x`) and Prefill -> Decode handoff carrying BOTH `past_states` [FP32] & `conv_state` [FP16]
     - Bitwise repeatability, side CUDA streams, CUDA Graph capture/replay, multi-GPU, API/alignment guards

Usage:
  python tests/test_correctness.py
  python tests/test_correctness.py --quick
  python tests/test_correctness.py --suite infer --seed 123
"""

import argparse
import copy
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from deltanet_v2 import DeltaBlock, apply_deltablock, apply_deltablock_infer, custom_ops
from deltanet_v2.model import _beta_guard_project_ref, _rmsnorm_ref
from deltanet_v2.ops import _LIB_LOADED, _LOAD_ERROR, _try_load_library
from tests.utils import (
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
REF_CONSIST_TOL = 2.0e-5
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
    print(f"    [{status:<4}] {name:<32}{suffix}")
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
    abs_cap: float = ABS_CAP,
) -> None:
    passed, kr, br = check_dual_precision(
        name=name,
        kernel_val=kernel_val,
        baseline_fp16_val=baseline_fp16_val,
        truth_val=truth_val,
        abs_cap=abs_cap,
        abs_cap_group=ABS_CAP_GROUP,
        lead_dims=lead_dims,
        use_ratio=use_ratio,
        ratio_fail=RATIO_FAIL,
        ratio_warn=RATIO_WARN,
        err_floor=ERR_FLOOR,
        verbose=True,
    )
    if not passed:
        failures.append(f"{tag} :: {name} (k={kr:.2e}, b={br:.2e})")


def _print_table_header(truth_label: str = "fp32") -> None:
    print(
        f"    {'status':<6} {'tensor':<20} | {f'kernel vs {truth_label}':<24} "
        f"| {f'base16 vs {truth_label}':<24} | grp(worst) k b"
    )


def _summarize_failures(suite_name: str, failures: List[str]) -> None:
    if failures:
        print(f"\n[FAIL] {suite_name}: {len(failures)} check(s) failed:")
        for msg in failures[:20]:
            print(f"  - {msg}")
    else:
        print(f"\n[OK] {suite_name}: all checks passed.")


def _check_drift(
    failures: List[str],
    tag: str,
    ek: torch.Tensor,
    eb: torch.Tensor,
    tail_cap: float = ABS_CAP,
) -> None:
    T = ek.numel()
    marks = sorted({0, T // 4, T // 2, T - 1})
    print(
        "    step drift | "
        + "  ".join(f"t={t}: k={ek[t]:.1e}/b={eb[t]:.1e}" for t in marks)
    )
    q = max(1, T // 4)
    gk = float((ek[-q:].mean() / ek[:q].mean().clamp_min(1e-12)).item())
    gb = float((eb[-q:].mean() / eb[:q].mean().clamp_min(1e-12)).item())
    tail_k = float(ek[-q:].mean().item())
    tail_b = float(eb[-q:].mean().item())
    _record(
        failures,
        tag,
        "drift growth rate",
        gk <= max(2.5, 2.0 * gb),
        f"(kernel {gk:.2f}x | base16 {gb:.2f}x)",
    )
    _record(
        failures,
        tag,
        "drift tail level",
        tail_k <= max(tail_cap, 2.0 * tail_b),
        f"(kernel {tail_k:.1e} | base16 {tail_b:.1e} | cap {tail_cap:.0e})",
    )


def _build_model_pair(
    d_model: int,
    num_q_heads: int,
    num_kv_heads: int,
    hidden_dim: Optional[int],
    device: torch.device,
    allow_neg_eigval: bool = False,
    beta_guard: bool = True,
) -> Tuple[DeltaBlock, DeltaBlock]:
    m16 = (
        DeltaBlock(
            d_model=d_model,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            d_k=64,
            d_v=64,
            chunk_size=64,
            conv_size=4,
            hidden_dim=hidden_dim,
            allow_neg_eigval=allow_neg_eigval,
            beta_guard=beta_guard,
        )
        .to(device)
        .half()
    )
    # Give conv_weight / dt_bias / A_log non-trivial values so every branch is exercised
    with torch.no_grad():
        m16.conv_weight.copy_((torch.randn_like(m16.conv_weight) * 0.25).half())
        m16.A_log.copy_(torch.randn_like(m16.A_log) * 0.5)
        m16.dt_bias.copy_(torch.randn_like(m16.dt_bias) * 0.5)
        m16.o_norm_weight.copy_((1.0 + torch.randn_like(m16.o_norm_weight) * 0.1).half())
    m32 = copy.deepcopy(m16).float()
    return m16, m32


# ---------------------------------------------------------------------------
# Pure-PyTorch Sub-Op References (Block Ops & Proj-Conv Prep)
# ---------------------------------------------------------------------------
def _ref_proj_conv_prep(
    proj: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    past_conv_state: Optional[torch.Tensor],
    cu: List[int],
    num_q_heads: int,
    num_kv_heads: int,
    d_k: int = 64,
    d_v: int = 64,
    conv_size: int = 4,
    chunk_size: int = 64,
    scale: float = 0.125,
    allow_neg_eigval: bool = False,
    beta_guard: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference matching proj_conv_prep_fwd (returns head-first tensors)."""
    device, dtype = proj.device, proj.dtype
    L_tot = proj.shape[0]
    B = len(cu) - 1
    tap_cache = conv_size - 1
    d_q = num_q_heads * d_k
    d_kt = num_kv_heads * d_k
    d_vt = num_kv_heads * d_v
    d_qkv = d_q + d_kt + d_vt
    split_sizes = [d_q, d_kt, d_vt, d_kt, d_vt, d_kt, num_q_heads * d_v]

    q_chunks, k_chunks, v_chunks, b_chunks, w_chunks, g_chunks = [], [], [], [], [], []
    final_convs = []

    for i in range(B):
        s, e = int(cu[i]), int(cu[i + 1])
        cs = (
            past_conv_state[i].to(dtype)
            if past_conv_state is not None
            else torch.zeros(tap_cache, d_qkv, device=device, dtype=dtype)
        )
        if s >= e:
            final_convs.append(cs)
            continue
        p_seq = proj[s:e]
        Li = e - s
        q_r, k_r, v_r, b_r, w_r, f_r, _ = torch.split(p_seq, split_sizes, dim=-1)
        x_qkv = torch.cat([q_r, k_r, v_r], dim=-1)
        conv_seq = torch.cat([cs, x_qkv], dim=0)
        final_convs.append(conv_seq[-tap_cache:, :])

        conv_raw = sum(
            conv_seq[tap : tap + Li, :].float() * conv_weight[tap].float().view(1, -1)
            for tap in range(conv_size)
        )
        conv_act = F.silu(conv_raw)
        qc, kc, vc = torch.split(conv_act, [d_q, d_kt, d_vt], dim=-1)
        qc = qc.view(Li, num_q_heads, d_k)
        kc = kc.view(Li, num_kv_heads, d_k)
        vc = vc.view(Li, num_kv_heads, d_v)

        qn = (qc * torch.rsqrt(qc.pow(2).sum(dim=-1, keepdim=True) + 1e-12)) * scale
        kn = kc * torch.rsqrt(kc.pow(2).sum(dim=-1, keepdim=True) + 1e-12)
        bi = torch.sigmoid(b_r.float()).view(Li, num_kv_heads, d_k) * (2.0 if allow_neg_eigval else 1.0)
        wi = torch.sigmoid(w_r.float()).view(Li, num_kv_heads, d_v)
        gi = -torch.exp(A_log.float()).view(1, num_kv_heads, 1) * F.softplus(
            f_r.float().view(Li, num_kv_heads, d_k) + dt_bias.float().view(1, num_kv_heads, d_k)
        )
        if beta_guard:
            bi = _beta_guard_project_ref(kn, bi, gi)

        # Chunk-local cumulative sum in log2 domain (matching kernel G_log2_h)
        g_log2 = gi / math.log(2.0)
        n_c = (Li + chunk_size - 1) // chunk_size
        Lp = n_c * chunk_size
        if Lp > Li:
            pad = torch.zeros(Lp - Li, num_kv_heads, d_k, device=device, dtype=torch.float32)
            g_pad = torch.cat([g_log2, pad], dim=0)
        else:
            g_pad = g_log2
        g_cs = g_pad.view(n_c, chunk_size, num_kv_heads, d_k).cumsum(dim=1).view(Lp, num_kv_heads, d_k)[:Li]

        q_chunks.append(qn.to(dtype).transpose(0, 1))
        k_chunks.append(kn.to(dtype).transpose(0, 1))
        v_chunks.append(vc.to(dtype).transpose(0, 1))
        b_chunks.append(bi.to(dtype).transpose(0, 1))
        w_chunks.append(wi.to(dtype).transpose(0, 1))
        g_chunks.append(g_cs.transpose(0, 1))

    if L_tot == 0:
        q_h = torch.empty(num_q_heads, 0, d_k, device=device, dtype=dtype)
        k_h = torch.empty(num_kv_heads, 0, d_k, device=device, dtype=dtype)
        v_h = torch.empty(num_kv_heads, 0, d_v, device=device, dtype=dtype)
        b_h = torch.empty(num_kv_heads, 0, d_k, device=device, dtype=dtype)
        w_h = torch.empty(num_kv_heads, 0, d_v, device=device, dtype=dtype)
        G_h = torch.empty(num_kv_heads, 0, d_k, device=device, dtype=torch.float32)
    else:
        q_h = torch.cat(q_chunks, dim=1).contiguous()
        k_h = torch.cat(k_chunks, dim=1).contiguous()
        v_h = torch.cat(v_chunks, dim=1).contiguous()
        b_h = torch.cat(b_chunks, dim=1).contiguous()
        w_h = torch.cat(w_chunks, dim=1).contiguous()
        G_h = torch.cat(g_chunks, dim=1).contiguous()

    cs_out = (
        torch.stack(final_convs, dim=0)
        if B > 0
        else torch.empty(0, tap_cache, d_qkv, device=device, dtype=dtype)
    )
    return q_h, k_h, v_h, b_h, w_h, G_h, cs_out


# ===========================================================================
# SUITE 1: Varlen Forward & Backward Kernels
# ===========================================================================
def test_varlen_fwd_bwd_kernels(
    quick: bool = False, device: Optional[torch.device] = None
) -> List[str]:
    print_section("SUITE 1: Varlen Forward & Backward Kernels (Sub-Ops, Cross-Check & Full Block)")
    device = device or _require_gpu_and_ext()
    failures: List[str] = []

    # -----------------------------------------------------------------------
    # 1A: Raw Block Ops (RMSNorm, Residual+RMSNorm, SwiGLU, ResidualAdd)
    # -----------------------------------------------------------------------
    tag_b = "[train/block_ops] L=193 D=768"
    print(f"\n  {tag_b}")
    _print_table_header("fp32")
    L_b, D_b, H_ffn = 193, 768, 1536
    x_b = (torch.randn(L_b, D_b, device=device) * 0.5).half().contiguous()
    res_b = (torch.randn(L_b, D_b, device=device) * 0.5).half().contiguous()
    w_b = (1.0 + torch.randn(D_b, device=device) * 0.1).half().contiguous()
    gu_b = (torch.randn(L_b, 2 * H_ffn, device=device) * 0.5).half().contiguous()
    g_norm = torch.randn(L_b, D_b, device=device, dtype=torch.float32)
    g_h = torch.randn(L_b, D_b, device=device, dtype=torch.float32)
    g_act = torch.randn(L_b, H_ffn, device=device, dtype=torch.float32)

    # RMSNorm Fwd + Bwd + is_training=False == True
    rn_out_t, rn_rstd = custom_ops.fused_rmsnorm_fwd(x_b, w_b, 1e-5, True)
    rn_out_e, _ = custom_ops.fused_rmsnorm_fwd(x_b, w_b, 1e-5, False)
    _record(failures, tag_b, "rmsnorm eval==train", torch.equal(rn_out_t, rn_out_e))
    dx_rn_k, dw_rn_k = custom_ops.fused_rmsnorm_bwd(g_norm.half(), x_b, w_b, rn_rstd)

    def _run_rn_ref(dt: torch.dtype):
        xi = x_b.to(dt).detach().requires_grad_(True)
        wi = w_b.to(dt).detach().requires_grad_(True)
        yi = _rmsnorm_ref(xi, wi, 1e-5)
        (yi.float() * g_norm).sum().backward()
        return yi.detach(), xi.grad.detach(), wi.grad.detach()

    rn_t, dx_rn_t, dw_rn_t = _run_rn_ref(torch.float32)
    rn_b, dx_rn_b, dw_rn_b = _run_rn_ref(torch.float16)
    _dual_check(failures, tag_b, "rmsnorm out", rn_out_t, rn_b, rn_t, lead_dims=1)
    _dual_check(failures, tag_b, "rmsnorm dx", dx_rn_k, dx_rn_b, dx_rn_t, use_ratio=False)
    _dual_check(failures, tag_b, "rmsnorm dw", dw_rn_k, dw_rn_b, dw_rn_t, use_ratio=False)

    # Residual + RMSNorm Fwd + Bwd + eval==train
    h_k, xn_k, rr_rstd = custom_ops.fused_residual_rmsnorm_fwd(x_b, res_b, w_b, 1e-5, True)
    h_e, xn_e, _ = custom_ops.fused_residual_rmsnorm_fwd(x_b, res_b, w_b, 1e-5, False)
    _record(
        failures, tag_b, "res_rmsnorm eval==train",
        torch.equal(h_k, h_e) and torch.equal(xn_k, xn_e),
    )
    dh_k, dw_rr_k = custom_ops.fused_residual_rmsnorm_bwd(
        g_norm.half(), g_h.half(), h_k, w_b, rr_rstd
    )

    def _run_rr_ref(dt: torch.dtype):
        hi = (x_b.to(dt) + res_b.to(dt)).detach().requires_grad_(True)
        wi = w_b.to(dt).detach().requires_grad_(True)
        yi = _rmsnorm_ref(hi, wi, 1e-5)
        ((yi.float() * g_norm).sum() + (hi.float() * g_h).sum()).backward()
        return hi.detach(), yi.detach(), hi.grad.detach(), wi.grad.detach()

    ht_r, ynt_r, dht_r, dwt_r = _run_rr_ref(torch.float32)
    hb_r, ynb_r, dhb_r, dwb_r = _run_rr_ref(torch.float16)
    _dual_check(failures, tag_b, "res_rmsnorm x_norm", xn_k, ynb_r, ynt_r, lead_dims=1)
    _dual_check(failures, tag_b, "res_rmsnorm dh", dh_k, dhb_r, dht_r, use_ratio=False)
    _dual_check(failures, tag_b, "res_rmsnorm dw", dw_rr_k, dwb_r, dwt_r, use_ratio=False)

    # SwiGLU Fwd + Bwd & ResidualAdd
    act_k = custom_ops.fused_swiglu_fwd(gu_b)
    dgu_k = custom_ops.fused_swiglu_bwd(g_act.half(), gu_b)
    add_k = custom_ops.fused_residual_add_fwd(x_b, res_b)

    def _run_swiglu_ref(dt: torch.dtype):
        gui = gu_b.to(dt).detach().requires_grad_(True)
        g_p, u_p = gui.chunk(2, dim=-1)
        ai = F.silu(g_p) * u_p
        (ai.float() * g_act).sum().backward()
        return ai.detach(), gui.grad.detach()

    at_s, dgut_s = _run_swiglu_ref(torch.float32)
    ab_s, dgub_s = _run_swiglu_ref(torch.float16)
    _dual_check(failures, tag_b, "swiglu act", act_k, ab_s, at_s, lead_dims=1)
    _dual_check(failures, tag_b, "swiglu dgate_up", dgu_k, dgub_s, dgut_s, use_ratio=False)
    _dual_check(failures, tag_b, "residual_add", add_k, x_b + res_b, x_b.float() + res_b.float(), lead_dims=1)

    # -----------------------------------------------------------------------
    # 1B: Raw `proj_conv_prep_fwd/bwd` & Core Sub-Op Chaining + is_training=False == True
    # -----------------------------------------------------------------------
    tag_pc = "[train/proj_conv_prep] seqlens=[1, 65, 129]"
    print(f"\n  {tag_pc}")
    _print_table_header("fp32")
    hq_p, hkv_p = 8, 4
    seqlens_p = [1, 65, 129]
    Bp, Lp = len(seqlens_p), sum(seqlens_p)
    cu_p, _ = make_cu_seqlens(seqlens_p, device)
    scale_p = 1.0 / math.sqrt(64)
    d_qkv_p = (hq_p + 2 * hkv_p) * 64
    total_proj_p = (hq_p + 3 * hkv_p) * 64 + (2 * hkv_p + hq_p) * 64

    for neg_eig, bguard in ((False, True), (True, False)):
        proj_p = (torch.randn(Lp, total_proj_p, device=device) * 0.4).half().contiguous()
        cw_p = (torch.randn(4, d_qkv_p, device=device) * 0.3).half().contiguous()
        Alog_p = (torch.randn(hkv_p, device=device) * 0.5).float().contiguous()
        dt_p = (torch.randn(hkv_p, 64, device=device) * 0.5).float().contiguous()
        pcs_p = (torch.randn(Bp, 3, d_qkv_p, device=device) * 0.2).half().contiguous()

        qk, kk, vk, bk, wk, Gk, csk, qinv, kinv = custom_ops.proj_conv_prep_fwd(
            proj_p, cw_p, Alog_p, dt_p, pcs_p, cu_p,
            hq_p, hkv_p, 64, 64, 4, 64, scale_p, neg_eig, bguard, True, True,
        )
        qe, ke, ve, be, we, Ge, cse, _, _ = custom_ops.proj_conv_prep_fwd(
            proj_p, cw_p, Alog_p, dt_p, pcs_p, cu_p,
            hq_p, hkv_p, 64, 64, 4, 64, scale_p, neg_eig, bguard, False, True,
        )
        eval_ok = (
            torch.equal(qk, qe) and torch.equal(kk, ke) and torch.equal(vk, ve)
            and torch.equal(bk, be) and torch.equal(wk, we) and torch.equal(Gk, Ge)
            and torch.equal(csk, cse)
        )
        _record(failures, tag_pc, f"eval==train (neg={neg_eig},bg={bguard})", eval_ok)

        qt, kt, vt, bt, wt, Gt, cst = _ref_proj_conv_prep(
            proj_p.float(), cw_p.float(), Alog_p, dt_p, pcs_p.float(), seqlens_to_cu(seqlens_p),
            hq_p, hkv_p, 64, 64, 4, 64, scale_p, neg_eig, bguard,
        )
        qb, kb, vb, bb, wb, Gb, csb = _ref_proj_conv_prep(
            proj_p, cw_p, Alog_p, dt_p, pcs_p, seqlens_to_cu(seqlens_p),
            hq_p, hkv_p, 64, 64, 4, 64, scale_p, neg_eig, bguard,
        )
        _dual_check(failures, tag_pc, f"q_h(neg={neg_eig})", qk, qb, qt, lead_dims=2)
        _dual_check(failures, tag_pc, f"b_eff(bg={bguard})", bk, bb, bt, lead_dims=2)
        _dual_check(failures, tag_pc, f"G_log2(neg={neg_eig})", Gk, Gb, Gt, lead_dims=2)
        _dual_check(failures, tag_pc, f"conv_s(bg={bguard})", csk, csb, cst, lead_dims=2)

    # Regression: mixed beta_guard violations inside the same 4-row warp (SM75 warp-convergence safety)
    cu_bg, _ = make_cu_seqlens([4], device)
    proj_bg = torch.zeros(4, total_proj_p, device=device, dtype=torch.float16)
    proj_bg[:, hq_p * 64 : d_qkv_p] = 0.5
    off_b_bg = d_qkv_p
    off_f_bg = d_qkv_p + 2 * hkv_p * 64
    proj_bg[:, off_b_bg : off_b_bg + hkv_p * 64].view(4, hkv_p, 64)[..., :32] = 6.0
    proj_bg[:, off_b_bg : off_b_bg + hkv_p * 64].view(4, hkv_p, 64)[..., 32:] = -6.0
    proj_bg[0, off_f_bg : off_f_bg + hkv_p * 64] = -12.0   # row 0 triggers beta_guard
    proj_bg[1:, off_f_bg : off_f_bg + hkv_p * 64] = 6.0    # rows 1..3 in same warp do not
    cw_bg = torch.zeros(4, d_qkv_p, device=device, dtype=torch.float16)
    cw_bg[3, :] = 1.0
    Alog_bg = torch.zeros(hkv_p, device=device, dtype=torch.float32)
    dt_bg = torch.zeros(hkv_p, 64, device=device, dtype=torch.float32)
    _, _, _, bk_div, _, _, _, _, _ = custom_ops.proj_conv_prep_fwd(
        proj_bg, cw_bg, Alog_bg, dt_bg, None, cu_bg,
        hq_p, hkv_p, 64, 64, 4, 64, scale_p, False, True, False, False,
    )
    _, _, _, bt_div, _, _, _ = _ref_proj_conv_prep(
        proj_bg.float(), cw_bg.float(), Alog_bg, dt_bg, None, [0, 4],
        hq_p, hkv_p, 64, 64, 4, 64, scale_p, False, True,
    )
    _, _, _, bb_div, _, _, _ = _ref_proj_conv_prep(
        proj_bg, cw_bg, Alog_bg, dt_bg, None, [0, 4],
        hq_p, hkv_p, 64, 64, 4, 64, scale_p, False, True,
    )
    _dual_check(failures, tag_pc, "b_eff(mixed-warp div)", bk_div, bb_div, bt_div, lead_dims=2)

    # Check proj_conv_prep_bwd when beta_guard=False (exact PyTorch autograd match)
    dq_g = torch.randn_like(qk).float()
    dk_g = torch.randn_like(kk).float()
    dv_g = torch.randn_like(vk).float()
    db_g = torch.randn_like(bk).float()
    dw_g = torch.randn_like(wk).float()
    dG_g = torch.randn_like(Gk).float()

    qk0, kk0, vk0, bk0, wk0, Gk0, _, qinv0, kinv0 = custom_ops.proj_conv_prep_fwd(
        proj_p, cw_p, Alog_p, dt_p, pcs_p, cu_p,
        hq_p, hkv_p, 64, 64, 4, 64, scale_p, False, False, True, True,
    )
    d_proj_init = torch.zeros_like(proj_p)
    dp_k, dcw_k, dA_k, ddt_k, dpcs_k = custom_ops.proj_conv_prep_bwd(
        dq_g.half(), dk_g.half(), dv_g.half(), db_g.half(), dw_g.half(), dG_g,
        proj_p, cw_p, Alog_p, dt_p, pcs_p, qinv0, kinv0, cu_p, d_proj_init,
        hq_p, hkv_p, 64, 64, 4, 64, scale_p, False, True,
    )

    def _run_pcp_bwd_ref(dt: torch.dtype):
        pi = proj_p.to(dt).detach().requires_grad_(True)
        cwi = cw_p.to(dt).detach().requires_grad_(True)
        Ai = Alog_p.detach().clone().requires_grad_(True)
        dti = dt_p.detach().clone().requires_grad_(True)
        pcsi = pcs_p.to(dt).detach().requires_grad_(True)
        qr, kr, vr, br, wr, Gr, _ = _ref_proj_conv_prep(
            pi, cwi, Ai, dti, pcsi, seqlens_to_cu(seqlens_p),
            hq_p, hkv_p, 64, 64, 4, 64, scale_p, False, False,
        )
        loss = (
            (qr.float() * dq_g).sum()
            + (kr.float() * dk_g).sum()
            + (vr.float() * dv_g).sum()
            + (br.float() * db_g).sum()
            + (wr.float() * dw_g).sum()
            + ((Gr.float() * math.log(2.0)) * dG_g).sum()
        )
        loss.backward()
        return pi.grad.detach(), cwi.grad.detach(), Ai.grad.detach(), dti.grad.detach(), pcsi.grad.detach()

    dpt, dcwt, dAt, ddtt, dpcst = _run_pcp_bwd_ref(torch.float32)
    dpb, dcwb, dAb, ddtb, dpcsb = _run_pcp_bwd_ref(torch.float16)
    _dual_check(failures, tag_pc, "d_proj", dp_k, dpb, dpt, use_ratio=False)
    _dual_check(failures, tag_pc, "d_conv_weight", dcw_k, dcwb, dcwt, use_ratio=False)
    _dual_check(failures, tag_pc, "d_A_log", dA_k, dAb, dAt, use_ratio=False)
    _dual_check(failures, tag_pc, "d_dt_bias", ddt_k, ddtb, ddtt, use_ratio=False)
    _dual_check(failures, tag_pc, "d_past_conv_state", dpcs_k, dpcsb, dpcst, use_ratio=False)

    # -----------------------------------------------------------------------
    # 1C: MANDATORY CROSS-CHECK: Chunkwise (CUDA & PyTorch) == Token-by-Token Recurrence (FP32)
    # -----------------------------------------------------------------------
    cross_seqlens = [1, 17, 63, 64, 65, 129]
    for hidden_cfg in (None, 512):
        tag_cc = f"[train/cross-check] chunkwise == recurrence (hidden_dim={hidden_cfg})"
        print(f"\n  {tag_cc}")
        _print_table_header("recur-fp32")
        m16_c, m32_c = _build_model_pair(256, 8, 4, hidden_cfg, device)
        m16_c.eval()
        m32_c.eval()
        x_pad, x_var, mask_c, cu_c = make_varlen_inputs(cross_seqlens, 256, scale=0.4, device=device)
        B_c = len(cross_seqlens)
        past32_c = (torch.randn(B_c, 4, 64, 64, device=device) * 0.1).float().contiguous()
        conv16_c = (torch.randn(B_c, 3, (8 + 2 * 4) * 64, device=device) * 0.2).half().contiguous()

        with torch.no_grad():
            # 1) Exact Token-by-Token Recurrence (FP32 truth via step-by-step _forward_infer_ref)
            y_rec_list, s_rec_list, cs_rec_list = [], [], []
            for b_idx, Lb in enumerate(cross_seqlens):
                sb = past32_c[b_idx : b_idx + 1].clone()
                csb = conv16_c[b_idx : b_idx + 1].float().clone()
                for t_idx in range(Lb):
                    xt = x_pad[b_idx : b_idx + 1, t_idx, :].float().contiguous()
                    yt, sb, csb = m32_c._forward_infer_ref(xt, past_states=sb, conv_state=csb)
                    y_rec_list.append(yt[0])
                s_rec_list.append(sb[0])
                cs_rec_list.append(csb[0])
            y_recur_t = torch.stack(y_rec_list, dim=0)
            s_recur_t = torch.stack(s_rec_list, dim=0)
            cs_recur_t = torch.stack(cs_rec_list, dim=0)

            # 2) PyTorch Chunkwise (_forward_batch FP32) == Token-by-Token Recurrence
            y_batch_t, s_batch_t, cs_batch_t = m32_c._forward_batch(
                x_pad.float(), padding_mask=mask_c, past_states=past32_c.clone(), conv_state=conv16_c.float().clone()
            )
            y_batch_v = y_batch_t[mask_c].contiguous()
            ry_cb = compute_tensor_metrics(y_batch_v, y_recur_t)[0]
            rs_cb = compute_tensor_metrics(s_batch_t, s_recur_t)[0]
            rc_cb = compute_tensor_metrics(cs_batch_t, cs_recur_t)[0]
            _record(
                failures, tag_cc, "pytorch_chunk == pytorch_recur",
                max(ry_cb, rs_cb, rc_cb) < REF_CONSIST_TOL,
                f"(y={ry_cb:.1e}, s={rs_cb:.1e}, cs={rc_cb:.1e})",
            )

            # 3) PyTorch Varlen (_forward_varlen FP16 baseline) & CUDA Chunkwise Kernel (FP16) vs Recurrence Truth
            y_var_b, s_var_b, cs_var_b = m16_c._forward_varlen(
                x_var, cu_seqlens=cu_c, past_states=past32_c.clone(), conv_state=conv16_c.clone()
            )
            y_kern, s_kern, cs_kern = m16_c(
                x_var, cu_seqlens=cu_c, past_states=past32_c.clone(), conv_state=conv16_c.clone(), use_kernel=True
            )
            _dual_check(failures, tag_cc, "chunk_y vs recur", y_kern, y_var_b, y_recur_t, lead_dims=1)
            _dual_check(failures, tag_cc, "chunk_state vs recur", s_kern, s_var_b, s_recur_t, lead_dims=2)
            _dual_check(failures, tag_cc, "chunk_conv vs recur", cs_kern, cs_var_b, cs_recur_t, lead_dims=2)

    # -----------------------------------------------------------------------
    # 1D: Full `apply_deltablock` Varlen Fwd + Bwd across GQA, Option 5A MLP, Toggles
    # -----------------------------------------------------------------------
    raw_configs = (
        [
            (256, 8, 4, None, [64, 128, 65, 1], False, True),
            (256, 8, 4, 512, [1, 17, 63, 64, 65], True, False),
        ]
        if quick
        else [
            (256, 8, 4, None, [64, 128, 65, 1], False, True),
            (256, 8, 4, 512, [1, 17, 63, 64, 65], False, True),
            (256, 8, 8, 512, [1, 17, 63, 64, 65, 129], True, True),
            (256, 8, 2, None, [300, 129, 64, 257], False, False),
            (512, 16, 4, 1024, [512, 127, 65], False, True),
        ]
    )

    for d_model, hq, hkv, hidden_dim, seqlens, neg_eig, bguard in raw_configs:
        B = len(seqlens)
        m16, m32 = _build_model_pair(
            d_model, hq, hkv, hidden_dim, device, allow_neg_eigval=neg_eig, beta_guard=bguard
        )
        d_qkv = (hq + 2 * hkv) * 64
        for use_past in (False, True):
            tag = (
                f"[train/deltablock] hq={hq} hkv={hkv} mlp={hidden_dim} "
                f"seqlens={seqlens} past={use_past} neg={neg_eig} bg={bguard}"
            )
            print(f"\n  {tag}")
            _print_table_header("fp32")

            _, x_var, _, cu = make_varlen_inputs(seqlens, d_model, scale=0.45, device=device)
            past32 = (torch.randn(B, hkv, 64, 64, device=device) * 0.1).float().contiguous() if use_past else None
            conv16 = (torch.randn(B, 3, d_qkv, device=device) * 0.2).half().contiguous() if use_past else None

            gy = torch.randn_like(x_var).float()
            gfs = (torch.randn(B, hkv, 64, 64, device=device) * 0.25).float()

            def _run_varlen_full(mod: DeltaBlock, dt: torch.dtype, use_k: bool) -> Dict[str, torch.Tensor]:
                mod.zero_grad(set_to_none=True)
                xi = x_var.to(dt).detach().clone().contiguous().requires_grad_(True)
                pi = past32.detach().clone().contiguous().requires_grad_(True) if past32 is not None else None
                csi = conv16.to(dt).detach().clone().contiguous().requires_grad_(True) if conv16 is not None else None

                if use_k:
                    n1w, n2w, mgu, mdn = mod._mlp_params()
                    y, fs, fcs = apply_deltablock(
                        x=xi, cu_seqlens=cu, past_states=pi, past_conv_state=csi,
                        W_proj=mod.W_proj.weight, conv_weight=mod.conv_weight,
                        A_log=mod.A_log, dt_bias=mod.dt_bias,
                        o_norm_weight=mod.o_norm_weight, W_out=mod.W_out.weight,
                        num_q_heads=hq, num_kv_heads=hkv,
                        norm1_weight=n1w, norm2_weight=n2w, mlp_W_gate_up=mgu, mlp_W_down=mdn,
                        d_k=64, d_v=64, conv_size=4, chunk_size=64,
                        scale=mod.scale, eps=mod.eps,
                        allow_neg_eigval=mod.allow_neg_eigval, beta_guard=mod.beta_guard,
                        output_conv_state=True,
                    )
                else:
                    y, fs, fcs = mod._forward_varlen(xi, cu_seqlens=cu, past_states=pi, conv_state=csi)

                assert fs.dtype == torch.float32, f"Expected FP32 final_states, got {fs.dtype}"
                assert fcs.dtype == xi.dtype, f"Expected conv_state dtype {xi.dtype}, got {fcs.dtype}"

                ((y.float() * gy).sum() + (fs.float() * gfs).sum()).backward()
                out_map = {
                    "y": y.detach(),
                    "final_states": fs.detach(),
                    "final_conv_state": fcs.detach(),
                    "grad_x": xi.grad.detach(),
                    "grad_W_proj": mod.W_proj.weight.grad.detach(),
                    "grad_conv_w": mod.conv_weight.grad.detach(),
                    "grad_A_log": mod.A_log.grad.detach(),
                    "grad_dt_bias": mod.dt_bias.grad.detach(),
                    "grad_o_norm_w": mod.o_norm_weight.grad.detach(),
                    "grad_W_out": mod.W_out.weight.grad.detach(),
                }
                if pi is not None:
                    out_map["grad_past_states"] = pi.grad.detach()
                if csi is not None:
                    out_map["grad_past_conv"] = csi.grad.detach()
                if mod.hidden_dim is not None:
                    out_map["grad_norm1_w"] = mod.norm1_weight.grad.detach()
                    out_map["grad_norm2_w"] = mod.norm2_weight.grad.detach()
                    out_map["grad_mlp_gu"] = mod.mlp_W_gate_up.weight.grad.detach()
                    out_map["grad_mlp_dn"] = mod.mlp_W_down.weight.grad.detach()
                return out_map

            # Note: beta_guard=True is piecewise-differentiable projection only on kernel backward;
            # PyTorch truth handles exact autograd smoothly everywhere else.
            truth = _run_varlen_full(m32, torch.float32, False)
            base16 = _run_varlen_full(m16, torch.float16, False)
            kern16 = _run_varlen_full(m16, torch.float16, True)

            lead_map = {"y": 1, "final_states": 2, "final_conv_state": 2}
            for name in truth:
                _dual_check(
                    failures, tag, name,
                    kern16[name], base16[name], truth[name],
                    lead_dims=lead_map.get(name),
                    use_ratio=not name.startswith("grad_"),
                )

    # -----------------------------------------------------------------------
    # 1E: Zero-length, all-empty, and deltablock_fwd is_training=False == True
    # -----------------------------------------------------------------------
    tag_z = "[train/zero-len & forward_only] seqlens=[0, 17, 0, 65]"
    print(f"\n  {tag_z}")
    m16_z, m32_z = _build_model_pair(256, 8, 4, 512, device)
    seqlens_z = [0, 17, 0, 65]
    Bz = len(seqlens_z)
    _, x_z, _, cu_z = make_varlen_inputs(seqlens_z, 256, scale=0.5, device=device)
    past_z = (torch.randn(Bz, 4, 64, 64, device=device) * 0.1).float().contiguous()
    conv_z = (torch.randn(Bz, 3, (8 + 2 * 4) * 64, device=device) * 0.2).half().contiguous()

    with torch.no_grad():
        yk_z, fsk_z, csk_z = m16_z(
            x_z, cu_seqlens=cu_z, past_states=past_z.clone(), conv_state=conv_z.clone(), use_kernel=True
        )
        yt_z, fst_z, cst_z = m32_z._forward_varlen(
            x_z.float(), cu_seqlens=cu_z, past_states=past_z.clone(), conv_state=conv_z.float().clone()
        )
    _dual_check(failures, tag_z, "y (zero-len mixed)", yk_z, None, yt_z, lead_dims=1)
    _dual_check(failures, tag_z, "fs (zero-len mixed)", fsk_z, None, fst_z, lead_dims=2)
    _dual_check(failures, tag_z, "cs (zero-len mixed)", csk_z, None, cst_z, lead_dims=2)
    empty_ok = (
        torch.equal(fsk_z[0], past_z[0])
        and torch.equal(fsk_z[2], past_z[2])
        and torch.equal(csk_z[0], conv_z[0])
        and torch.equal(csk_z[2], conv_z[2])
    )
    _record(failures, tag_z, "empty-seq state & conv passthrough", empty_ok)

    # is_training=False == True on deltablock_fwd
    n1w, n2w, mgu, mdn = m16_z._mlp_params()
    out_tr = custom_ops.deltablock_fwd(
        x_z, cu_z, past_z.clone(), conv_z.clone(),
        m16_z.W_proj.weight, m16_z.conv_weight, m16_z.A_log, m16_z.dt_bias,
        m16_z.o_norm_weight, m16_z.W_out.weight, n1w, n2w, mgu, mdn,
        8, 4, 64, 64, 4, 64, m16_z.scale, m16_z.eps, False, True, True, True,
    )
    out_ev = custom_ops.deltablock_fwd(
        x_z, cu_z, past_z.clone(), conv_z.clone(),
        m16_z.W_proj.weight, m16_z.conv_weight, m16_z.A_log, m16_z.dt_bias,
        m16_z.o_norm_weight, m16_z.W_out.weight, n1w, n2w, mgu, mdn,
        8, 4, 64, 64, 4, 64, m16_z.scale, m16_z.eps, False, True, False, True,
    )
    _record(
        failures, tag_z, "deltablock is_training=False==True",
        torch.equal(out_tr[0], out_ev[0]) and torch.equal(out_tr[1], out_ev[1]) and torch.equal(out_tr[2], out_ev[2]),
    )

    # All-empty total_L == 0
    cu_e, _ = make_cu_seqlens([0, 0], device)
    x_e = torch.empty(0, 256, device=device, dtype=torch.float16)
    p_e = (torch.randn(2, 4, 64, 64, device=device) * 0.1).float().contiguous()
    c_e = (torch.randn(2, 3, (8 + 2 * 4) * 64, device=device) * 0.2).half().contiguous()
    ye_k, fse_k, cse_k = m16_z(x_e, cu_seqlens=cu_e, past_states=p_e.clone(), conv_state=c_e.clone(), use_kernel=True)
    _record(
        failures,
        tag_z,
        "total_L=0 shapes & finite",
        ye_k.shape == (0, 256)
        and torch.equal(fse_k, p_e)
        and torch.equal(cse_k, c_e),
    )

    _summarize_failures("Suite 1 (Varlen Fwd/Bwd Kernels)", failures)
    return failures


def seqlens_to_cu(seqlens: List[int]) -> List[int]:
    cu = [0]
    for s in seqlens:
        cu.append(cu[-1] + int(s))
    return cu


# ===========================================================================
# SUITE 2: Inference Decode Kernel (L = 1 Super-Fused `apply_deltablock_infer`)
# ===========================================================================
def test_infer_decode_kernels(
    quick: bool = False, device: Optional[torch.device] = None
) -> List[str]:
    print_section("SUITE 2: Inference Decode Kernel (L = 1 Super-Fused)")
    device = device or _require_gpu_and_ext()
    failures: List[str] = []

    def _run_step_compare(
        m16: DeltaBlock,
        m32: DeltaBlock,
        x16: torch.Tensor,
        past32: torch.Tensor,
        conv16: torch.Tensor,
    ):
        n1w, n2w, mgu, mdn = m16._mlp_params()
        pk = past32.clone().contiguous()
        ck = conv16.clone().contiguous()
        yk, sk, csk = apply_deltablock_infer(
            x=x16, past_states=pk, conv_state=ck,
            W_proj=m16.W_proj.weight, conv_weight=m16.conv_weight,
            A_log=m16.A_log, dt_bias=m16.dt_bias,
            o_norm_weight=m16.o_norm_weight, W_out=m16.W_out.weight,
            num_q_heads=m16.num_q_heads, num_kv_heads=m16.num_kv_heads,
            norm1_weight=n1w, norm2_weight=n2w, mlp_W_gate_up=mgu, mlp_W_down=mdn,
            d_k=64, d_v=64, conv_size=4, scale=m16.scale, eps=m16.eps,
            allow_neg_eigval=m16.allow_neg_eigval, beta_guard=m16.beta_guard,
        )
        yb, sb, csb = m16._forward_infer_ref(x16, past_states=past32.clone(), conv_state=conv16.clone())
        yt, st, cst = m32._forward_infer_ref(x16.float(), past_states=past32.clone(), conv_state=conv16.float().clone())
        return (yk, sk, csk, pk, ck), (yb, sb, csb), (yt, st, cst)

    # -----------------------------------------------------------------------
    # 2A: Batch & GQA Sweep (hidden_dim=None and !=None)
    # -----------------------------------------------------------------------
    batches = [1, 17, 64] if quick else [1, 16, 17, 33, 64, 129, 256, 513]
    gqa_cfgs = [(256, 8, 4, None), (256, 8, 2, 512)] if quick else [(256, 8, 4, None), (256, 8, 2, 512), (512, 8, 8, 1024)]

    with torch.no_grad():
        for d_model, hq, hkv, hidden_dim in gqa_cfgs:
            m16, m32 = _build_model_pair(d_model, hq, hkv, hidden_dim, device)
            m16.eval()
            m32.eval()
            d_qkv = (hq + 2 * hkv) * 64
            for B in batches:
                state_modes = ("zero", "rand") if quick else ("zero", "mixed", "rand")
                for smode in state_modes:
                    tag = f"[infer/batch] hq={hq} hkv={hkv} mlp={hidden_dim} B={B} st={smode}"
                    print(f"\n  {tag}")
                    _print_table_header("fp32")
                    x16 = (torch.randn(B, d_model, device=device) * 0.5).half().contiguous()
                    if smode == "zero":
                        past32 = torch.zeros(B, hkv, 64, 64, device=device, dtype=torch.float32)
                        conv16 = torch.zeros(B, 3, d_qkv, device=device, dtype=torch.float16)
                    elif smode == "mixed":
                        past32 = (torch.randn(B, hkv, 64, 64, device=device) * 0.1).float()
                        past32[::2] = 0.0
                        past32 = past32.contiguous()
                        conv16 = (torch.randn(B, 3, d_qkv, device=device) * 0.2).half().contiguous()
                    else:
                        past32 = (torch.randn(B, hkv, 64, 64, device=device) * 0.1).float().contiguous()
                        conv16 = (torch.randn(B, 3, d_qkv, device=device) * 0.2).half().contiguous()

                    (yk, sk, csk, pk, ck), (yb, sb, csb), (yt, st, cst) = _run_step_compare(
                        m16, m32, x16, past32, conv16
                    )
                    inplace_ok = (
                        sk.data_ptr() == pk.data_ptr()
                        and csk.data_ptr() == ck.data_ptr()
                        and sk.dtype == torch.float32
                        and csk.dtype == torch.float16
                    )
                    _record(failures, tag, "inplace fp32 state & fp16 conv", inplace_ok)
                    _dual_check(failures, tag, "y", yk, yb, yt, lead_dims=1)
                    _dual_check(failures, tag, "state", sk, sb, st, lead_dims=2)
                    _dual_check(failures, tag, "conv_state", csk, csb, cst, lead_dims=2)

        # -------------------------------------------------------------------
        # 2B: Extreme Stress & Toggles (x-scale, dt_bias/A_log, beta_guard, allow_neg_eigval)
        # -------------------------------------------------------------------
        m16_s, m32_s = _build_model_pair(256, 8, 4, 512, device)
        m16_s.eval()
        m32_s.eval()
        B_s = 17
        d_qkv_s = (8 + 2 * 4) * 64
        stress_cases = (
            [(0.1, 0.0, False, True), (5.0, 5.0, True, False)]
            if quick
            else [
                (0.1, 0.0, False, True),
                (5.0, 0.0, False, True),
                (1.0, 7.0, False, True),
                (1.0, -7.0, False, True),
                (2.5, 4.0, True, True),
                (2.5, -4.0, True, False),
            ]
        )
        for scale, dt_off, neg_eig, bguard in stress_cases:
            tag_s = f"[infer/stress] scale={scale} dt={dt_off} neg={neg_eig} bg={bguard}"
            print(f"\n  {tag_s}")
            _print_table_header("fp32")
            ms16 = copy.deepcopy(m16_s)
            ms16.allow_neg_eigval = neg_eig
            ms16.beta_guard = bguard
            ms16.dt_bias.add_(dt_off)
            ms32 = copy.deepcopy(ms16).float()

            x16 = (torch.randn(B_s, 256, device=device) * scale).half().contiguous()
            past32 = (torch.randn(B_s, 4, 64, 64, device=device) * 0.1).float().contiguous()
            conv16 = (torch.randn(B_s, 3, d_qkv_s, device=device) * 0.2).half().contiguous()

            (yk, sk, csk, _, _), (yb, sb, csb), (yt, st, cst) = _run_step_compare(
                ms16, ms32, x16, past32, conv16
            )
            _dual_check(failures, tag_s, "y", yk, yb, yt, lead_dims=1)
            _dual_check(failures, tag_s, "state", sk, sb, st, lead_dims=2)
            _dual_check(failures, tag_s, "conv_state", csk, csb, cst, lead_dims=2)

        # Edge cases: zero k/v inputs & tiny norm stability
        tag_ed = "[infer/edge:zero_input] B=17"
        print(f"\n  {tag_ed}")
        _print_table_header("fp32")
        x_z = torch.zeros(B_s, 256, device=device, dtype=torch.float16)
        past_z = (torch.randn(B_s, 4, 64, 64, device=device) * 0.1).float().contiguous()
        conv_z = torch.zeros(B_s, 3, d_qkv_s, device=device, dtype=torch.float16)
        (yk, sk, csk, _, _), (yb, sb, csb), (yt, st, cst) = _run_step_compare(
            m16_s, m32_s, x_z, past_z, conv_z
        )
        _dual_check(failures, tag_ed, "y (zero x/conv)", yk, yb, yt, lead_dims=1)
        _dual_check(failures, tag_ed, "state (zero x/conv)", sk, sb, st, lead_dims=2)

        # -------------------------------------------------------------------
        # 2C: Ablations (verify every component affects outputs as expected)
        # -------------------------------------------------------------------
        tag_ab = "[infer/ablations] B=16"
        print(f"\n  {tag_ab}")
        x_a = (torch.randn(16, 256, device=device) * 0.5).half().contiguous()
        p_a = (torch.randn(16, 4, 64, 64, device=device) * 0.1).float().contiguous()
        c_a = (torch.randn(16, 3, d_qkv_s, device=device) * 0.25).half().contiguous()
        base_y, base_s, base_c = m16_s(x_a, past_states=p_a.clone(), conv_state=c_a.clone(), use_kernel=True)

        # 1) conv_state sensitivity (oldest tap shifts out; newest tap shifts into slot 1)
        c_mod = c_a.clone()
        c_mod[:, 0] += 0.5
        yc, _, cc = m16_s(x_a, past_states=p_a.clone(), conv_state=c_mod, use_kernel=True)
        c_mod2 = c_a.clone()
        c_mod2[:, 2] += 0.5
        yc2, _, cc2 = m16_s(x_a, past_states=p_a.clone(), conv_state=c_mod2, use_kernel=True)
        _record(
            failures, tag_ab, "conv_state sensitivity",
            (yc - base_y).abs().max().item() > 1e-3
            and torch.equal(cc, base_c)
            and (yc2 - base_y).abs().max().item() > 1e-3
            and (cc2 - base_c).abs().max().item() > 1e-3,
        )

        # 2) channel-wise dt_bias sensitivity
        m_dt = copy.deepcopy(m16_s)
        m_dt.dt_bias[:, 0::2] += 2.0
        ydt, sdt, _ = m_dt(x_a, past_states=p_a.clone(), conv_state=c_a.clone(), use_kernel=True)
        _record(
            failures, tag_ab, "channel-wise dt_bias active",
            (ydt - base_y).abs().max().item() > 1e-3 and (sdt - base_s).abs().max().item() > 1e-3,
        )

        # 3) allow_neg_eigval toggle effect
        m_neg = copy.deepcopy(m16_s)
        m_neg.allow_neg_eigval = True
        yng, sng, _ = m_neg(x_a, past_states=p_a.clone(), conv_state=c_a.clone(), use_kernel=True)
        _record(
            failures, tag_ab, "allow_neg_eigval active",
            (yng - base_y).abs().max().item() > 1e-3 and (sng - base_s).abs().max().item() > 1e-3,
        )

        # -------------------------------------------------------------------
        # 2D: Slice Invariance, Canary & Long Decode Rollout Drift
        # -------------------------------------------------------------------
        tag_inv = "[infer/invariance] B=32"
        print(f"\n  {tag_inv}")
        B_i = 32
        x_i = (torch.randn(B_i, 256, device=device) * 0.5).half().contiguous()
        p_i = (torch.randn(B_i, 4, 64, 64, device=device) * 0.1).float().contiguous()
        c_i = (torch.randn(B_i, 3, d_qkv_s, device=device) * 0.2).half().contiguous()
        y_full, s_full, c_full = m16_s(x_i, past_states=p_i.clone(), conv_state=c_i.clone(), use_kernel=True)
        y_sub, s_sub, c_sub = m16_s(x_i[:16], past_states=p_i[:16].clone(), conv_state=c_i[:16].clone(), use_kernel=True)
        _record(
            failures, tag_inv, "batch slice independence",
            compute_tensor_metrics(y_sub, y_full[:16])[0] < MODEL_INV_TOL
            and compute_tensor_metrics(s_sub, s_full[:16])[0] < MODEL_INV_TOL
            and torch.equal(c_sub, c_full[:16]),
        )

        # Canary: zero state + zero conv_state + unit residual/o_norm identity check
        tag_can = "[infer/canary] B=1 deterministic step"
        print(f"\n  {tag_can}")
        x_can = (torch.randn(1, 256, device=device) * 0.5).half().contiguous()
        p_can = torch.zeros(1, 4, 64, 64, device=device, dtype=torch.float32)
        c_can = torch.zeros(1, 3, d_qkv_s, device=device, dtype=torch.float16)
        yk_c, sk_c, ck_c = m16_s(x_can, past_states=p_can.clone(), conv_state=c_can.clone(), use_kernel=True)
        yt_c, st_c, ct_c = m32_s._forward_infer_ref(x_can.float(), past_states=p_can.clone(), conv_state=c_can.float().clone())
        _record(
            failures, tag_can, "canary match fp32 truth",
            compute_tensor_metrics(yk_c, yt_c)[0] < 3e-3
            and compute_tensor_metrics(sk_c, st_c)[0] < 3e-3
            and compute_tensor_metrics(ck_c, ct_c)[0] < 1e-3,
        )

        # Long autoregressive decode rollout (T = 24 / 64 steps)
        T_roll = 24 if quick else 64
        for B_r in ([16] if quick else [1, 32]):
            tag_r = f"[infer/rollout] B={B_r} T={T_roll}"
            print(f"\n  {tag_r}")
            x_seq = (torch.randn(B_r, T_roll, 256, device=device) * 0.4).half().contiguous()
            sk = (torch.randn(B_r, 4, 64, 64, device=device) * 0.05).float().contiguous()
            sb = sk.clone()
            st = sk.clone()
            ck = (torch.randn(B_r, 3, d_qkv_s, device=device) * 0.1).half().contiguous()
            cb = ck.clone()
            ct = ck.float().clone()
            ysk, ysb, yst = [], [], []
            for t in range(T_roll):
                xt = x_seq[:, t, :].contiguous()
                yk, sk, ck = m16_s(xt, past_states=sk, conv_state=ck, use_kernel=True)
                yb, sb, cb = m16_s._forward_infer_ref(xt, past_states=sb, conv_state=cb)
                yt, st, ct = m32_s._forward_infer_ref(xt.float(), past_states=st, conv_state=ct)
                ysk.append(yk)
                ysb.append(yb)
                yst.append(yt)

            y_k = torch.stack(ysk, dim=1)
            y_b = torch.stack(ysb, dim=1)
            y_t = torch.stack(yst, dim=1)
            _print_table_header("fp32")
            _dual_check(failures, tag_r, "y (all steps)", y_k, y_b, y_t, lead_dims=2)
            _dual_check(failures, tag_r, "final_states", sk, sb, st, lead_dims=2)
            _dual_check(failures, tag_r, "final_conv_state", ck, cb, ct, lead_dims=2)
            ek = ((y_k.double() - y_t.double()).norm(dim=-1) / (y_t.double().norm(dim=-1) + 1e-30)).mean(0)
            eb = ((y_b.double() - y_t.double()).norm(dim=-1) / (y_t.double().norm(dim=-1) + 1e-30)).mean(0)
            _check_drift(failures, tag_r, ek, eb)

    _summarize_failures("Suite 2 (Inference Decode Kernel)", failures)
    return failures


# ===========================================================================
# SUITE 3: End-to-End DeltaBlock Modules & Runtime Environment
# ===========================================================================
def test_end_to_end_modules(
    quick: bool = False, device: Optional[torch.device] = None
) -> List[str]:
    print_section("SUITE 3: End-to-End DeltaBlock (Training, Handoff, Rollout & Runtime Guards)")
    device = device or _require_gpu_and_ext()
    failures: List[str] = []

    d_model, nq, nkv = 512, 8, 4
    d_qkv = (nq + 2 * nkv) * 64

    # 3A: End-to-end DeltaBlock Training (hidden_dim=None and !=None)
    mlp_cfgs = [1024] if quick else [None, 1024]
    train_configs = [[1], [64, 128, 65, 1]] if quick else [[1], [64, 128, 65, 1], [518, 1988, 700, 300]]

    for hidden_dim in mlp_cfgs:
        m16, m32 = _build_model_pair(d_model, nq, nkv, hidden_dim, device)
        m16.train()
        m32.train()
        for seqlens in train_configs:
            B = len(seqlens)
            for use_past in (False, True):
                tag = f"[e2e/train] mlp={hidden_dim} seqlens={seqlens} past={use_past}"
                print(f"\n  {tag}")
                _print_table_header("fp32")
                x_pad, x_var, mask, cu = make_varlen_inputs(seqlens, d_model, scale=0.4, device=device)
                past32 = (torch.randn(B, nkv, 64, 64, device=device) * 0.1).float().contiguous() if use_past else None
                conv16 = (torch.randn(B, 3, d_qkv, device=device) * 0.2).half().contiguous() if use_past else None
                g_out = torch.randn(x_var.shape, device=device, dtype=torch.float32)

                def _run_blk(mod: DeltaBlock, use_k: bool, dtype: torch.dtype):
                    mod.zero_grad(set_to_none=True)
                    pi = past32.detach().clone().contiguous().requires_grad_(True) if past32 is not None else None
                    csi = conv16.to(dtype).detach().clone().contiguous().requires_grad_(True) if conv16 is not None else None
                    if use_k:
                        xi = x_var.to(dtype).detach().clone().contiguous().requires_grad_(True)
                        out, fs, fcs = mod(xi, cu_seqlens=cu, past_states=pi, conv_state=csi, use_kernel=True)
                        out_v = out
                    else:
                        xi = x_pad.to(dtype).detach().clone().contiguous().requires_grad_(True)
                        out, fs, fcs = mod._forward_batch(xi, padding_mask=mask, past_states=pi, conv_state=csi)
                        out_v = out[mask].contiguous()
                    (out_v.float() * g_out).sum().backward()
                    res = {
                        "out": out_v.detach(),
                        "final_states": fs.detach(),
                        "final_conv_state": fcs.detach(),
                        "grad(x)": (xi.grad if use_k else xi.grad[mask]).detach(),
                        "grad(W_proj)": mod.W_proj.weight.grad.detach(),
                        "grad(conv_w)": mod.conv_weight.grad.detach(),
                        "grad(dt_bias)": mod.dt_bias.grad.detach(),
                        "grad(o_norm_w)": mod.o_norm_weight.grad.detach(),
                    }
                    if pi is not None:
                        res["grad(past_states)"] = pi.grad.detach()
                    if csi is not None:
                        res["grad(past_conv)"] = csi.grad.detach()
                    return res

                truth = _run_blk(m32, False, torch.float32)
                base16 = _run_blk(m16, False, torch.float16)
                kern16 = _run_blk(m16, True, torch.float16)

                lead_map = {"out": 1, "grad(x)": 1, "final_states": 2, "final_conv_state": 2}
                for name in truth:
                    _dual_check(
                        failures, tag, name, kern16[name], base16[name], truth[name],
                        lead_dims=lead_map.get(name), use_ratio=not name.startswith("grad("),
                    )

    # 3B: no_grad vs autograd consistency + Single-step 2D/3D + Prefill -> Decode Handoff
    m16, m32 = _build_model_pair(d_model, nq, nkv, 1024, device)
    tag_fo = "[e2e/forward_only] seqlens=[64, 128, 65, 1]"
    print(f"\n  {tag_fo}")
    _, x_fo, _, cu_fo = make_varlen_inputs([64, 128, 65, 1], d_model, scale=0.4, device=device)
    with torch.no_grad():
        y0, s0, c0 = m16(x_fo, cu_seqlens=cu_fo, use_kernel=True)
    m16.train()
    y1, s1, c1 = m16(x_fo, cu_seqlens=cu_fo, use_kernel=True)
    ry = compute_tensor_metrics(y0, y1)[0]
    rs = compute_tensor_metrics(s0, s1)[0]
    rc = compute_tensor_metrics(c0, c1)[0]
    _record(
        failures, tag_fo, "no_grad == autograd",
        max(ry, rs, rc) < INV_TOL, f"(out={ry:.1e}, state={rs:.1e}, conv={rc:.1e})",
    )

    m16.eval()
    m32.eval()
    with torch.no_grad():
        # Single-step L=1 2D vs 3D
        for B in ([1, 17, 64] if quick else [1, 16, 17, 64, 256]):
            for x_dim in (3, 2):
                tag_1s = f"[e2e/infer-1step] B={B} x_dim={x_dim}"
                print(f"\n  {tag_1s}")
                _print_table_header("fp32")
                x3 = (torch.randn(B, 1, d_model, device=device) * 0.4).half().contiguous()
                x_in = x3 if x_dim == 3 else x3[:, 0].clone().contiguous()
                x2 = x3[:, 0].clone().contiguous()
                p32 = (torch.randn(B, nkv, 64, 64, device=device) * 0.1).float().contiguous()
                c16 = (torch.randn(B, 3, d_qkv, device=device) * 0.2).half().contiguous()
                pk, ck = p32.clone(), c16.clone()

                yk, sk, csk = m16(x_in, past_states=pk, conv_state=ck, use_kernel=True)
                yb, sb, csb = m16._forward_infer_ref(x2, past_states=p32.clone(), conv_state=c16.clone())
                yt, st, cst = m32._forward_infer_ref(x2.float(), past_states=p32.clone(), conv_state=c16.float().clone())

                inplace_ok = (
                    sk.data_ptr() == pk.data_ptr()
                    and csk.data_ptr() == ck.data_ptr()
                    and yk.shape == x_in.shape
                )
                _record(failures, tag_1s, "shape & inplace buffers", inplace_ok)
                _dual_check(failures, tag_1s, "out", yk.reshape(B, d_model), yb, yt, lead_dims=1)
                _dual_check(failures, tag_1s, "final_states", sk, sb, st, lead_dims=2)
                _dual_check(failures, tag_1s, "final_conv_state", csk, csb, cst, lead_dims=2)

        # Prefill (varlen kernel) -> Decode rollout handoff carrying BOTH past_states & conv_state
        T_hand = 24 if quick else 64
        hand_configs = (
            [[300, 129, 64, 1, 65, 63, 128, 257]]
            if quick
            else [
                [300, 129, 64, 1, 65, 63, 128, 257, 2, 17, 511, 100, 64, 64, 33, 1000],
                torch.randint(1, 300, (48,), generator=torch.Generator().manual_seed(49)).tolist(),
            ]
        )
        for seqlens in hand_configs:
            B = len(seqlens)
            tag_h = f"[e2e/handoff] B={B} prefill={sum(seqlens)} -> decode T={T_hand}"
            print(f"\n  {tag_h}")
            _print_table_header("fp32")
            x_pad, x_var, mask, cu = make_varlen_inputs(seqlens, d_model, scale=0.4, device=device)
            x_new = (torch.randn(B, T_hand, d_model, device=device) * 0.4).half().contiguous()

            m16.train()
            _, s_pre_k, c_pre_k = m16(x_var, cu_seqlens=cu, use_kernel=True)
            m16.eval()
            _, s_pre_t, c_pre_t = m32._forward_batch(x_pad.float(), padding_mask=mask)
            _, s_pre_b, c_pre_b = m16._forward_batch(x_pad, padding_mask=mask)
            _dual_check(failures, tag_h, "prefill final_states", s_pre_k, s_pre_b, s_pre_t, lead_dims=2)
            _dual_check(failures, tag_h, "prefill conv_state", c_pre_k, c_pre_b, c_pre_t, lead_dims=2)

            sk, sb, st = s_pre_k.clone(), s_pre_b.clone(), s_pre_t.clone()
            ck, cb, ct = c_pre_k.clone(), c_pre_b.clone(), c_pre_t.clone()
            ysk, ysb, yst = [], [], []
            for t in range(T_hand):
                xt3 = x_new[:, t : t + 1].contiguous()
                xt2 = x_new[:, t].contiguous()
                yk, sk, ck = m16(xt3, past_states=sk, conv_state=ck, use_kernel=True)
                yb, sb, cb = m16._forward_infer_ref(xt2, past_states=sb, conv_state=cb)
                yt, st, ct = m32._forward_infer_ref(xt2.float(), past_states=st, conv_state=ct)
                ysk.append(yk[:, 0])
                ysb.append(yb)
                yst.append(yt)
            y_k = torch.stack(ysk, dim=1)
            y_b = torch.stack(ysb, dim=1)
            y_t = torch.stack(yst, dim=1)
            _dual_check(failures, tag_h, "decode out", y_k, y_b, y_t, lead_dims=2)
            _dual_check(failures, tag_h, "decode final_states", sk, sb, st, lead_dims=2)
            _dual_check(failures, tag_h, "decode conv_state", ck, cb, ct, lead_dims=2)
            ek = ((y_k.double() - y_t.double()).norm(dim=-1) / (y_t.double().norm(dim=-1) + 1e-30)).mean(0)
            eb = ((y_b.double() - y_t.double()).norm(dim=-1) / (y_t.double().norm(dim=-1) + 1e-30)).mean(0)
            _check_drift(failures, tag_h, ek, eb)

        # -------------------------------------------------------------------
        # 3C: Runtime Environment (Bitwise repeatability, Side Stream, CUDA Graph, API Guards)
        # -------------------------------------------------------------------
        repeats = 15 if quick else 40
        for B_r in ((16, 513) if quick else (16, 513, 1024)):
            tag_race = f"[e2e/inplace-race] B={B_r}"
            print(f"\n  {tag_race}")
            xr = (torch.randn(B_r, 1, d_model, device=device) * 0.4).half().contiguous()
            pr = (torch.randn(B_r, nkv, 64, 64, device=device) * 0.1).float().contiguous()
            cr = (torch.randn(B_r, 3, d_qkv, device=device) * 0.2).half().contiguous()
            y_ref, s_ref, c_ref = m16(xr, past_states=pr.clone(), conv_state=cr.clone(), use_kernel=True)
            diff_cnt = 0
            for _ in range(repeats):
                yi, si, ci = m16(xr, past_states=pr.clone(), conv_state=cr.clone(), use_kernel=True)
                diff_cnt += int(not (torch.equal(yi, y_ref) and torch.equal(si, s_ref) and torch.equal(ci, c_ref)))
            _record(
                failures, tag_race, f"bitwise repeat x{repeats}",
                diff_cnt == 0, f"({diff_cnt}/{repeats} mismatches)",
            )

        tag_rt = "[e2e/runtime] B=64 stream & CUDAGraph"
        print(f"\n  {tag_rt}")
        B_rt = 64
        x_rt = (torch.randn(B_rt, 1, d_model, device=device) * 0.4).half().contiguous()
        p_rt = (torch.randn(B_rt, nkv, 64, 64, device=device) * 0.1).float().contiguous()
        c_rt = (torch.randn(B_rt, 3, d_qkv, device=device) * 0.2).half().contiguous()
        y0_rt, s0_rt, c0_rt = m16(x_rt, past_states=p_rt.clone(), conv_state=c_rt.clone(), use_kernel=True)
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        bad_stream = 0
        for _ in range(10):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                junk = torch.randn(1024, 1024, device=device)
                junk = torch.tanh(junk @ junk)
                ys, ss, cs = m16(x_rt.clone(), past_states=p_rt.clone(), conv_state=c_rt.clone(), use_kernel=True)
            torch.cuda.synchronize()
            bad_stream += int(not (torch.equal(ys, y0_rt) and torch.equal(ss, s0_rt) and torch.equal(cs, c0_rt)))
        _record(failures, tag_rt, "side CUDA stream", bad_stream == 0)

        try:
            sx, sp, sc = x_rt.clone(), p_rt.clone(), c_rt.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    m16(sx, past_states=sp, conv_state=sc, use_kernel=True)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                yg, sg, cg = m16(sx, past_states=sp, conv_state=sc, use_kernel=True)

            sx.copy_(x_rt)
            sp.copy_(p_rt)
            sc.copy_(c_rt)
            graph.replay()
            torch.cuda.synchronize()
            ok1 = torch.equal(yg, y0_rt) and torch.equal(sg, s0_rt) and torch.equal(cg, c0_rt)

            x2 = (torch.randn(B_rt, 1, d_model, device=device) * 0.4).half().contiguous()
            p2 = (torch.randn(B_rt, nkv, 64, 64, device=device) * 0.1).float().contiguous()
            c2 = (torch.randn(B_rt, 3, d_qkv, device=device) * 0.2).half().contiguous()
            ye, se, ce = m16(x2, past_states=p2.clone(), conv_state=c2.clone(), use_kernel=True)
            sx.copy_(x2)
            sp.copy_(p2)
            sc.copy_(c2)
            graph.replay()
            torch.cuda.synchronize()
            ok2 = torch.equal(yg, ye) and torch.equal(sg, se) and torch.equal(cg, ce)
            _record(failures, tag_rt, "CUDA graph replay (new inputs)", ok1 and ok2)
        except Exception as exc:  # noqa: BLE001
            _record(failures, tag_rt, "CUDA graph replay (new inputs)", False, f"({exc})")

        if torch.cuda.device_count() >= 2:
            dev1 = torch.device("cuda:1")
            m_dev1 = copy.deepcopy(m16).to(dev1)
            with torch.cuda.device(0):
                y1, s1, c1 = m_dev1(
                    x_rt.to(dev1), past_states=p_rt.to(dev1).clone(), conv_state=c_rt.to(dev1).clone(), use_kernel=True
                )
            torch.cuda.synchronize(dev1)
            ry = compute_tensor_metrics(y1.to(device), y0_rt)[0]
            rs = compute_tensor_metrics(s1.to(device), s0_rt)[0]
            rc = compute_tensor_metrics(c1.to(device), c0_rt)[0]
            _record(failures, tag_rt, "non-default device cuda:1", max(ry, rs, rc) < INV_TOL)

        # API & Alignment Guards
        tag_misc = "[e2e/api-guards] B=32"
        print(f"\n  {tag_misc}")
        B_m = 32
        xm = (torch.randn(B_m, 1, d_model, device=device) * 0.4).half().contiguous()
        pm = (torch.randn(B_m, nkv, 64, 64, device=device) * 0.1).float().contiguous()
        cm = (torch.randn(B_m, 3, d_qkv, device=device) * 0.2).half().contiguous()
        y_m0, s_m0, c_m0 = m16(xm, past_states=pm.clone(), conv_state=cm.clone(), use_kernel=True)
        y_m1, s_m1, c_m1 = m16(xm, past_states=pm.clone(), conv_state=cm.clone(), use_kernel=True)
        _record(
            failures, tag_misc, "deterministic bitwise",
            torch.equal(y_m0, y_m1) and torch.equal(s_m0, s_m1) and torch.equal(c_m0, c_m1),
        )

        y_2d, s_2d, c_2d = m16(xm[:, 0].clone().contiguous(), past_states=pm.clone(), conv_state=cm.clone(), use_kernel=True)
        _record(
            failures, tag_misc, "2D == 3D input",
            torch.equal(y_m0[:, 0], y_2d) and torch.equal(s_m0, s_2d) and torch.equal(c_m0, c_2d),
        )

        # FP16 past_states auto-cast to FP32 & immutable on caller buffer
        pm16 = pm.half().contiguous()
        y_f, s_f, c_f = m16(xm, past_states=pm16, conv_state=cm.clone(), use_kernel=True)
        _record(
            failures, tag_misc, "past fp16 auto-cast & immutable",
            s_f.dtype == torch.float32
            and torch.equal(pm16, pm.half())
            and compute_tensor_metrics(y_m0, y_f)[0] < INV_TOL,
        )

        # Non-contiguous past_states & conv_state & x
        p_nc = torch.randn(nkv, B_m, 64, 64, device=device).float().mul_(0.1).transpose(0, 1)
        c_nc = torch.randn(3, B_m, d_qkv, device=device).half().mul_(0.2).transpose(0, 1)
        y_nc, s_nc, cs_nc = m16(xm, past_states=p_nc, conv_state=c_nc, use_kernel=True)
        y_cc, s_cc, cs_cc = m16(xm, past_states=p_nc.contiguous().clone(), conv_state=c_nc.contiguous().clone(), use_kernel=True)
        _record(
            failures, tag_misc, "non-contiguous states handled",
            torch.equal(y_nc, y_cc) and torch.equal(s_nc, s_cc) and torch.equal(cs_nc, cs_cc),
        )

        x_2tok = (torch.randn(B_m, 2, d_model, device=device) * 0.4).half().contiguous()
        x_nc = x_2tok[:, :1]
        y_xnc, s_xnc, cx_nc = m16(x_nc, past_states=pm.clone(), conv_state=cm.clone(), use_kernel=True)
        y_xcc, s_xcc, cx_cc = m16(x_nc.contiguous(), past_states=pm.clone(), conv_state=cm.clone(), use_kernel=True)
        _record(
            failures, tag_misc, "x non-contiguous",
            torch.equal(y_xnc, y_xcc) and torch.equal(s_xnc, s_xcc) and torch.equal(cx_nc, cx_cc),
        )

        l2_rejected = False
        try:
            m16._forward_infer_kernel(x_2tok, past_states=pm.clone(), conv_state=cm.clone())
        except AssertionError:
            l2_rejected = True
        _record(failures, tag_misc, "infer L=2 rejected cleanly", l2_rejected)

        yz, sz, cz = m16(xm, past_states=None, conv_state=None, use_kernel=True)
        yz2, sz2, cz2 = m16(xm, past_states=torch.zeros_like(pm), conv_state=torch.zeros_like(cm), use_kernel=True)
        _record(
            failures, tag_misc, "past/conv=None == zeros",
            torch.equal(yz, yz2) and torch.equal(sz, sz2) and torch.equal(cz, cz2),
        )

        # Misaligned pointer handling
        buf = torch.empty(pm.numel() + 8, device=device, dtype=torch.float32)
        p_mis = buf[1 : 1 + pm.numel()].view_as(pm)
        p_mis.copy_(pm)
        assert p_mis.is_contiguous() and p_mis.data_ptr() % 16 != 0
        try:
            ym, sm, cm_o = m16(xm, past_states=p_mis, conv_state=cm.clone(), use_kernel=True)
            _record(
                failures, tag_misc, "misaligned past handling",
                torch.equal(y_m0, ym) and torch.equal(s_m0, sm) and torch.equal(c_m0, cm_o),
                "(aligned copy match)",
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
        description="Run comprehensive Gated DeltaNet-2 correctness suites."
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
        choices=["all", "train", "infer", "e2e"],
        help="Which suite to run (default: all).",
    )
    args = parser.parse_args()
    run_all_tests(quick=args.quick, seed=args.seed, suite=args.suite)


if __name__ == "__main__":
    main()
