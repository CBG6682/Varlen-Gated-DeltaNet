import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from deltanet_v2.ops import (
    apply_deltablock,
    apply_deltablock_infer,
)

_RCP_LN2 = 1.4426950408889634
_GUARD_MARGIN = 1.0 / 32.0
_GUARD_QUANT_TOL_MULT = 4.0
_FP16_EPS = 9.765625e-4


def _rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMSNorm matching CUDA FP32 variance accumulation and target dtype output."""
    x32 = x.float()
    rstd = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x32 * rstd).to(weight.dtype) * weight


def _beta_guard_project_ref(k: torch.Tensor, b: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Conditional key-weighted beta-contrast projection with straight-through gradients."""
    k_d = k.detach()
    b_d = b.detach()
    g_d = g.detach()

    g_max = g_d.amax(dim=-1, keepdim=True)
    c2 = torch.exp(-2.0 * g_max)
    k2 = k_d * k_d
    n = k2.sum(dim=-1, keepdim=True)
    a = (b_d * k2).sum(dim=-1, keepdim=True)
    nu = (b_d * b_d * k2).sum(dim=-1, keepdim=True)

    r2 = (n * nu - a * a).clamp(min=0.0)
    r2_crit = ((c2 - 1.0) * (1.0 - (1.0 - a).square() / c2.clamp(min=1e-30))).clamp(min=0.0)
    viol = (n > 1e-20) & (r2 > r2_crit)
    if not torch.any(viol):
        return b

    mu = a / n.clamp(min=1e-30)
    eta = torch.sqrt(((1.0 - _GUARD_MARGIN) * r2_crit) / r2.clamp(min=1e-30)).clamp(0.0, 1.0)
    cand = (mu + eta * (b_d - mu)).to(torch.float16).float()

    a_q = (cand * k2).sum(dim=-1, keepdim=True)
    nu_q = (cand * cand * k2).sum(dim=-1, keepdim=True)
    r2_q = (n * nu_q - a_q * a_q).clamp(min=0.0)
    r2_crit_q = ((c2 - 1.0) * (1.0 - (1.0 - a_q).square() / c2.clamp(min=1e-30))).clamp(min=0.0)
    quant_tol = (_GUARD_QUANT_TOL_MULT * _FP16_EPS) * (n * nu_q + a_q * a_q)

    fallback = r2_q > (r2_crit_q + quant_tol)
    mu_q = (a_q / n.clamp(min=1e-30)).to(torch.float16).float().expand_as(b_d)
    projected = torch.where(fallback, mu_q, cand)
    b_eff_val = torch.where(viol, projected, b_d)
    return b + (b_eff_val - b).detach()


class DeltaBlock(nn.Module):
    """
    Gated DeltaNet-2 Transformer Block (`arXiv:2605.22791`).

    Features:
      - Unified input projection `[q, k, v, b, w, f, o_gate]`
      - Depthwise causal ShortConv1D (`conv_size=4`) + SiLU + L2-norm (`q`, `k`) + Q scale
      - Channel-wise log-decay (`A_log`, `dt_bias`), `allow_neg_eigval`, and `beta_guard`
      - Exact chunkwise WY & sequential FP32 state recurrence with GQA
      - Fused Per-Head RMSNorm * SiLU(`o_gate`) + output projection
      - Optional PreNorm SwiGLU MLP when `hidden_dim is not None`
    """

    def __init__(
        self,
        d_model: int,
        num_q_heads: int,
        num_kv_heads: int,
        d_k: int = 64,
        d_v: int = 64,
        chunk_size: int = 64,
        conv_size: int = 4,
        hidden_dim: Optional[int] = None,
        allow_neg_eigval: bool = False,
        beta_guard: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        assert num_q_heads > 0 and num_kv_heads > 0 and num_q_heads % num_kv_heads == 0, (
            f"num_q_heads ({num_q_heads}) must be positive and divisible by num_kv_heads ({num_kv_heads})"
        )

        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_q_heads // num_kv_heads
        self.d_k = d_k
        self.d_v = d_v
        self.chunk_size = chunk_size
        self.conv_size = conv_size
        self.hidden_dim = hidden_dim
        self.allow_neg_eigval = allow_neg_eigval
        self.beta_guard = beta_guard
        self.eps = eps
        self.scale = d_k ** -0.5

        self.d_q_total = num_q_heads * d_k
        self.d_k_total = num_kv_heads * d_k
        self.d_v_total = num_kv_heads * d_v
        self.d_qkv_total = self.d_q_total + self.d_k_total + self.d_v_total
        self.d_out_total = num_q_heads * d_v

        self.split_sizes = [
            self.d_q_total,    # q
            self.d_k_total,    # k
            self.d_v_total,    # v
            self.d_k_total,    # b
            self.d_v_total,    # w
            self.d_k_total,    # f
            self.d_out_total,  # o_gate
        ]
        self.total_proj_dim = sum(self.split_sizes)

        self.W_proj = nn.Linear(d_model, self.total_proj_dim, bias=False)
        self.conv_weight = nn.Parameter(torch.empty(conv_size, self.d_qkv_total))
        self.A_log = nn.Parameter(torch.empty(num_kv_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(num_kv_heads, d_k, dtype=torch.float32))
        self.o_norm_weight = nn.Parameter(torch.ones(d_v))
        self.W_out = nn.Linear(self.d_out_total, d_model, bias=False)

        if hidden_dim is not None:
            self.norm1_weight = nn.Parameter(torch.ones(d_model))
            self.norm2_weight = nn.Parameter(torch.ones(d_model))
            self.mlp_W_gate_up = nn.Linear(d_model, 2 * hidden_dim, bias=False)
            self.mlp_W_down = nn.Linear(hidden_dim, d_model, bias=False)
        else:
            self.norm1_weight = None
            self.norm2_weight = None
            self.mlp_W_gate_up = None
            self.mlp_W_down = None

        self.reset_parameters()

    def reset_parameters(self):
        linear_gain = 2.0 ** -2.5
        nn.init.xavier_uniform_(self.W_proj.weight, gain=linear_gain)
        nn.init.xavier_uniform_(self.W_out.weight, gain=linear_gain)
        nn.init.uniform_(self.conv_weight, -0.5, 0.5)

        with torch.no_grad():
            a_init = torch.empty_like(self.A_log, dtype=torch.float32).uniform_(1e-2, 16.0)
            self.A_log.copy_(torch.log(a_init))

            dt_min, dt_max = 0.001, 0.1
            u = torch.rand_like(self.dt_bias, dtype=torch.float32)
            dt = torch.exp(u * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
            inv_sp = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias.copy_(inv_sp)

            nn.init.ones_(self.o_norm_weight)
            if self.hidden_dim is not None:
                nn.init.xavier_uniform_(self.mlp_W_gate_up.weight, gain=linear_gain)
                nn.init.xavier_uniform_(self.mlp_W_down.weight, gain=linear_gain)
                nn.init.ones_(self.norm1_weight)
                nn.init.ones_(self.norm2_weight)

    def _apply(self, fn):
        super()._apply(fn)
        # Keep log-decay parameters in FP32 even when the module is cast to FP16/BF16.
        for name in ("A_log", "dt_bias"):
            p = getattr(self, name, None)
            if isinstance(p, nn.Parameter) and p.dtype in (torch.float16, torch.bfloat16):
                p.data = p.data.float()
                if p.grad is not None and p.grad.dtype in (torch.float16, torch.bfloat16):
                    p.grad = p.grad.float()
        return self

    def _mlp_params(self):
        if self.hidden_dim is None:
            return None, None, None, None
        return (
            self.norm1_weight,
            self.norm2_weight,
            self.mlp_W_gate_up.weight,
            self.mlp_W_down.weight,
        )

    # ---------------------------------------------------------------------
    # CUDA Kernel Branch: Varlen Training / Prefill
    # ---------------------------------------------------------------------
    def _forward_kernel(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        past_states: Optional[torch.Tensor] = None,
        conv_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        norm1_w, norm2_w, mlp_gu_w, mlp_down_w = self._mlp_params()
        is_batched_3d = (x.dim() == 3)

        if is_batched_3d:
            B, L, D = x.shape
            if cu_seqlens is None:
                if padding_mask is None:
                    x_packed = x.reshape(B * L, D).contiguous()
                    cu = torch.arange(0, (B + 1) * L, L, device=x.device, dtype=torch.int32)
                    mask_b = None
                else:
                    mask_b = padding_mask.bool()
                    lengths = mask_b.sum(dim=1, dtype=torch.int32)
                    cu = torch.zeros(B + 1, device=x.device, dtype=torch.int32)
                    cu[1:] = torch.cumsum(lengths, dim=0, dtype=torch.int32)
                    x_packed = x[mask_b].contiguous()
            else:
                x_packed = x.reshape(-1, D).contiguous()
                cu = cu_seqlens
                mask_b = None
        else:
            assert cu_seqlens is not None, (
                "use_kernel=True on 2D input requires cu_seqlens (packed varlen)"
            )
            x_packed = x if x.is_contiguous() else x.contiguous()
            cu = cu_seqlens
            mask_b = None

        if x_packed.dtype != torch.float16:
            x_packed = x_packed.to(torch.float16)

        if past_states is not None:
            if past_states.dtype != torch.float32 or not past_states.is_contiguous():
                past_states = past_states.to(torch.float32).contiguous()

        if conv_state is not None:
            if conv_state.dtype != torch.float16 or not conv_state.is_contiguous():
                conv_state = conv_state.to(torch.float16).contiguous()

        y_packed, final_states, final_conv_state = apply_deltablock(
            x=x_packed,
            cu_seqlens=cu,
            past_states=past_states,
            past_conv_state=conv_state,
            W_proj=self.W_proj.weight,
            conv_weight=self.conv_weight,
            A_log=self.A_log.float(),
            dt_bias=self.dt_bias.float(),
            o_norm_weight=self.o_norm_weight,
            W_out=self.W_out.weight,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            norm1_weight=norm1_w,
            norm2_weight=norm2_w,
            mlp_W_gate_up=mlp_gu_w,
            mlp_W_down=mlp_down_w,
            d_k=self.d_k,
            d_v=self.d_v,
            conv_size=self.conv_size,
            chunk_size=self.chunk_size,
            scale=self.scale,
            eps=self.eps,
            allow_neg_eigval=self.allow_neg_eigval,
            beta_guard=self.beta_guard,
            output_conv_state=True,
        )

        if is_batched_3d:
            if mask_b is not None:
                y = y_packed.new_zeros(B, L, D)
                y[mask_b] = y_packed
            else:
                y = y_packed.view(B, L, D)
        else:
            y = y_packed

        return y, final_states, final_conv_state

    # ---------------------------------------------------------------------
    # CUDA Kernel Branch: Autoregressive Decoding (L = 1)
    # ---------------------------------------------------------------------
    def _forward_infer_kernel(
        self,
        x: torch.Tensor,
        past_states: Optional[torch.Tensor] = None,
        conv_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        has_seq_dim = (x.dim() == 3)
        if has_seq_dim:
            B, L, _ = x.shape
            assert L == 1, f"Inference kernel requires L = 1, got L = {L}"
            x_2d = x.squeeze(1).contiguous()
        else:
            B = x.shape[0]
            x_2d = x if x.is_contiguous() else x.contiguous()

        if x_2d.dtype != torch.float16:
            x_2d = x_2d.to(torch.float16)

        if past_states is None:
            past_states = torch.zeros(
                B, self.num_kv_heads, self.d_v, self.d_k, device=x.device, dtype=torch.float32
            )
        elif past_states.dtype != torch.float32 or not past_states.is_contiguous():
            past_states = past_states.to(torch.float32).contiguous()

        if conv_state is None:
            conv_state = torch.zeros(
                B, self.conv_size - 1, self.d_qkv_total, device=x.device, dtype=torch.float16
            )
        elif conv_state.dtype != torch.float16 or not conv_state.is_contiguous():
            conv_state = conv_state.to(torch.float16).contiguous()

        norm1_w, norm2_w, mlp_gu_w, mlp_down_w = self._mlp_params()
        y, past_states, conv_state = apply_deltablock_infer(
            x=x_2d,
            past_states=past_states,
            conv_state=conv_state,
            W_proj=self.W_proj.weight,
            conv_weight=self.conv_weight,
            A_log=self.A_log.float(),
            dt_bias=self.dt_bias.float(),
            o_norm_weight=self.o_norm_weight,
            W_out=self.W_out.weight,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            norm1_weight=norm1_w,
            norm2_weight=norm2_w,
            mlp_W_gate_up=mlp_gu_w,
            mlp_W_down=mlp_down_w,
            d_k=self.d_k,
            d_v=self.d_v,
            conv_size=self.conv_size,
            scale=self.scale,
            eps=self.eps,
            allow_neg_eigval=self.allow_neg_eigval,
            beta_guard=self.beta_guard,
        )

        if has_seq_dim:
            y = y.unsqueeze(1)
        return y, past_states, conv_state

    # ---------------------------------------------------------------------
    # Baseline Batch (Pure PyTorch, Exact Sub-chunk WY + Chunkwise Recurrence)
    # ---------------------------------------------------------------------
    def _forward_batch(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        past_states: Optional[torch.Tensor] = None,
        conv_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        device, dtype = x.device, x.dtype
        C = self.chunk_size
        H, G, Hq = self.num_kv_heads, self.num_groups, self.num_q_heads
        d_k, d_v = self.d_k, self.d_v
        tap_cache = self.conv_size - 1

        if B == 0:
            empty_y = x.new_empty(0, L, self.d_model)
            empty_s = torch.empty(0, H, d_v, d_k, device=device, dtype=torch.float32)
            empty_c = torch.empty(0, tap_cache, self.d_qkv_total, device=device, dtype=dtype)
            return empty_y, empty_s, empty_c

        if padding_mask is not None:
            padding_mask = padding_mask.bool()
            x_masked = x.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
        else:
            x_masked = x

        x_in = _rmsnorm_ref(x_masked, self.norm1_weight, self.eps) if self.hidden_dim is not None else x_masked
        if padding_mask is not None:
            x_in = x_in.masked_fill(~padding_mask.unsqueeze(-1), 0.0)

        proj = self.W_proj(x_in)
        if padding_mask is not None:
            proj = proj.masked_fill(~padding_mask.unsqueeze(-1), 0.0)

        q_r, k_r, v_r, b_r, w_r, f_r, o_gate = torch.split(proj, self.split_sizes, dim=-1)

        if conv_state is None:
            cs_in = x.new_zeros(B, tap_cache, self.d_qkv_total)
        else:
            cs_in = conv_state.to(dtype)

        x_qkv = torch.cat([q_r, k_r, v_r], dim=-1)
        conv_seq = torch.cat([cs_in, x_qkv], dim=1)  # (B, 3 + L, D_qkv)

        if padding_mask is None:
            final_conv_state = conv_seq[:, -tap_cache:, :] if L > 0 else cs_in
        else:
            lengths = padding_mask.long().sum(dim=1)
            fcs = []
            for b_idx in range(B):
                lb = int(lengths[b_idx].item())
                fcs.append(conv_seq[b_idx, lb : lb + tap_cache, :])
            final_conv_state = torch.stack(fcs, dim=0)

        if L > 0:
            conv_raw = sum(
                conv_seq[:, tap : tap + L, :].float() * self.conv_weight[tap].float().view(1, 1, -1)
                for tap in range(self.conv_size)
            )
            conv_act = F.silu(conv_raw)
            if padding_mask is not None:
                conv_act = conv_act * padding_mask.unsqueeze(-1).float()
        else:
            conv_act = x.new_zeros(B, 0, self.d_qkv_total, dtype=torch.float32)

        q_c, k_c, v_c = torch.split(conv_act, [self.d_q_total, self.d_k_total, self.d_v_total], dim=-1)
        q_c = q_c.view(B, L, Hq, d_k)
        k_c = k_c.view(B, L, H, d_k)
        v_c = v_c.view(B, L, H, d_v)

        q = (q_c * torch.rsqrt(q_c.pow(2).sum(dim=-1, keepdim=True) + 1e-12)) * self.scale
        k = k_c * torch.rsqrt(k_c.pow(2).sum(dim=-1, keepdim=True) + 1e-12)
        v = v_c

        b = torch.sigmoid(b_r.float()).view(B, L, H, d_k) * (2.0 if self.allow_neg_eigval else 1.0)
        w = torch.sigmoid(w_r.float()).view(B, L, H, d_v)
        g = -torch.exp(self.A_log.float()).view(1, 1, H, 1) * F.softplus(
            f_r.float().view(B, L, H, d_k) + self.dt_bias.float().view(1, 1, H, d_k)
        )

        if padding_mask is not None:
            m4 = padding_mask.view(B, L, 1, 1).float()
            q = q * m4
            k = k * m4
            v = v * m4
            b = b * m4
            w = w * m4
            g = g * m4

        if self.beta_guard:
            b = _beta_guard_project_ref(k, b, g)

        pad_len = (-L) % C
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
            b = F.pad(b, (0, 0, 0, 0, 0, pad_len))
            w = F.pad(w, (0, 0, 0, 0, 0, pad_len))
            g = F.pad(g, (0, 0, 0, 0, 0, pad_len))

        Lp = L + pad_len
        num_chunks = Lp // C
        S_curr = (
            past_states.float()
            if past_states is not None
            else torch.zeros(B, H, d_v, d_k, device=device, dtype=torch.float32)
        )

        if num_chunks > 0:
            # Reshape to (B, num_chunks, H, C, d)
            q_c = q.view(B, num_chunks, C, Hq, d_k).permute(0, 1, 3, 2, 4)
            k_c = k.view(B, num_chunks, C, H, d_k).permute(0, 1, 3, 2, 4)
            v_c = v.view(B, num_chunks, C, H, d_v).permute(0, 1, 3, 2, 4)
            b_c = b.view(B, num_chunks, C, H, d_k).permute(0, 1, 3, 2, 4)
            w_c = w.view(B, num_chunks, C, H, d_v).permute(0, 1, 3, 2, 4)
            g_c = g.view(B, num_chunks, C, H, d_k).permute(0, 1, 3, 2, 4)

            with torch.autocast(device_type=device.type, enabled=False):
                G_log2 = torch.cumsum(g_c * _RCP_LN2, dim=-2)  # (B, n_c, H, C, d_k)
                BC = 16 if (C % 16 == 0) else C
                n_sub = C // BC

                T = torch.zeros(B, num_chunks, H, C, C, device=device, dtype=torch.float32)
                e_base = b_c * k_c
                sub_tril_strict = torch.tril(torch.ones(BC, BC, device=device, dtype=torch.bool), diagonal=-1)
                sub_tril_incl = torch.tril(torch.ones(BC, BC, device=device, dtype=torch.bool), diagonal=0)

                for i in range(n_sub):
                    r0, r1 = i * BC, (i + 1) * BC
                    G_i = G_log2[..., r0:r1, :]
                    e_i = e_base[..., r0:r1, :]
                    k_i = k_c[..., r0:r1, :]

                    diff_ii = G_i.unsqueeze(-2) - G_i.unsqueeze(-3)
                    diff_ii = diff_ii.masked_fill(~sub_tril_strict.unsqueeze(-1), float("-inf"))
                    T[..., r0:r1, r0:r1] = torch.einsum("...ik,...jk,...ijk->...ij", e_i, k_i, torch.exp2(diff_ii))

                    if i > 0:
                        g_ref = G_i[..., :1, :]
                        e_tilde = e_i * torch.exp2(G_i - g_ref)
                        for j in range(i):
                            c0, c1 = j * BC, (j + 1) * BC
                            G_j = G_log2[..., c0:c1, :]
                            k_j = k_c[..., c0:c1, :]
                            k_tilde = k_j * torch.exp2(g_ref - G_j)
                            T[..., r0:r1, c0:c1] = torch.matmul(e_tilde, k_tilde.transpose(-1, -2))

                I = torch.eye(C, device=device, dtype=torch.float32).expand_as(T)
                A = torch.linalg.solve_triangular(I + T, I, upper=False, unitriangular=True)

                E_bar = torch.exp2(G_log2) * e_base
                Z = w_c * v_c
                Y = torch.matmul(A, E_bar)
                U = torch.matmul(A, Z)

                S_list = []
                R_list = []
                G_last = G_log2[..., -1:, :]
                decay_C = torch.exp2(G_last)
                K_tail = k_c * torch.exp2(G_last - G_log2)
                for c in range(num_chunks):
                    S_list.append(S_curr)
                    R_c = U[:, c] - torch.matmul(Y[:, c], S_curr.transpose(-1, -2))
                    R_list.append(R_c)
                    S_curr = S_curr * decay_C[:, c] + torch.matmul(R_c.transpose(-1, -2), K_tail[:, c])

                S_all = torch.stack(S_list, dim=1)  # (B, n_c, H, d_v, d_k)
                R_all = torch.stack(R_list, dim=1)  # (B, n_c, H, C, d_v)

                S_exp = S_all.unsqueeze(3).expand(B, num_chunks, H, G, d_v, d_k).reshape(B, num_chunks, Hq, d_v, d_k)
                R_exp = R_all.unsqueeze(3).expand(B, num_chunks, H, G, C, d_v).reshape(B, num_chunks, Hq, C, d_v)
                G_exp = G_log2.unsqueeze(3).expand(B, num_chunks, H, G, C, d_k).reshape(B, num_chunks, Hq, C, d_k)
                k_exp = k_c.unsqueeze(3).expand(B, num_chunks, H, G, C, d_k).reshape(B, num_chunks, Hq, C, d_k)

                Q_gamma = q_c * torch.exp2(G_exp)
                O_inter = torch.matmul(Q_gamma, S_exp.transpose(-1, -2))

                A_qk = torch.zeros(B, num_chunks, Hq, C, C, device=device, dtype=torch.float32)
                for i in range(n_sub):
                    r0, r1 = i * BC, (i + 1) * BC
                    G_i = G_exp[..., r0:r1, :]
                    q_i = q_c[..., r0:r1, :]
                    k_i = k_exp[..., r0:r1, :]

                    diff_ii = G_i.unsqueeze(-2) - G_i.unsqueeze(-3)
                    diff_ii = diff_ii.masked_fill(~sub_tril_incl.unsqueeze(-1), float("-inf"))
                    A_qk[..., r0:r1, r0:r1] = torch.einsum("...ik,...jk,...ijk->...ij", q_i, k_i, torch.exp2(diff_ii))

                    if i > 0:
                        g_ref = G_i[..., :1, :]
                        q_tilde = q_i * torch.exp2(G_i - g_ref)
                        for j in range(i):
                            c0, c1 = j * BC, (j + 1) * BC
                            G_j = G_exp[..., c0:c1, :]
                            k_j = k_exp[..., c0:c1, :]
                            k_tilde = k_j * torch.exp2(g_ref - G_j)
                            A_qk[..., r0:r1, c0:c1] = torch.matmul(q_tilde, k_tilde.transpose(-1, -2))

                O_intra = torch.matmul(A_qk, R_exp)
                O_raw = (O_inter + O_intra).permute(0, 1, 3, 2, 4).reshape(B, Lp, Hq, d_v)[:, :L]
        else:
            O_raw = torch.zeros(B, 0, Hq, d_v, device=device, dtype=torch.float32)

        final_states = S_curr  # FP32
        o_norm = _rmsnorm_ref(O_raw, self.o_norm_weight.view(1, 1, 1, d_v), self.eps).reshape(B, L, self.d_out_total)
        out_tok = o_norm.to(dtype) * F.silu(o_gate)
        y_mix = self.W_out(out_tok)

        if self.hidden_dim is not None:
            h = x_masked + y_mix
            x_n2 = _rmsnorm_ref(h, self.norm2_weight, self.eps)
            gate_part, up_part = self.mlp_W_gate_up(x_n2).chunk(2, dim=-1)
            ffn = self.mlp_W_down(F.silu(gate_part) * up_part)
            y = h + ffn
        else:
            y = y_mix

        if padding_mask is not None:
            y = y.masked_fill(~padding_mask.unsqueeze(-1), 0.0)

        return y, final_states, final_conv_state

    # ---------------------------------------------------------------------
    # Baseline Varlen: Exact Token-by-Token Recurrence in Pure PyTorch
    # ---------------------------------------------------------------------
    def _forward_varlen(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        past_states: Optional[torch.Tensor] = None,
        conv_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del padding_mask
        if cu_seqlens is None:
            cu_seqlens = torch.tensor([0, x.shape[0]], dtype=torch.int32, device=x.device)
        cu = cu_seqlens.tolist()
        num_seqs = len(cu) - 1
        H, G, Hq = self.num_kv_heads, self.num_groups, self.num_q_heads
        d_k, d_v = self.d_k, self.d_v
        tap_cache = self.conv_size - 1
        device, dtype = x.device, x.dtype

        res_out = []
        final_states = []
        final_convs = []

        for i in range(num_seqs):
            start, end = int(cu[i]), int(cu[i + 1])
            S_seq = (
                past_states[i].float()
                if past_states is not None
                else torch.zeros(H, d_v, d_k, device=device, dtype=torch.float32)
            )
            cs_seq = (
                conv_state[i].to(dtype)
                if conv_state is not None
                else torch.zeros(tap_cache, self.d_qkv_total, device=device, dtype=dtype)
            )

            if start >= end:
                final_states.append(S_seq)
                final_convs.append(cs_seq)
                continue

            x_seq = x[start:end]
            L = end - start

            x_in = _rmsnorm_ref(x_seq, self.norm1_weight, self.eps) if self.hidden_dim is not None else x_seq
            proj = self.W_proj(x_in)
            q_r, k_r, v_r, b_r, w_r, f_r, o_gate = torch.split(proj, self.split_sizes, dim=-1)

            x_qkv = torch.cat([q_r, k_r, v_r], dim=-1)
            conv_seq = torch.cat([cs_seq, x_qkv], dim=0)  # (3 + L, D_qkv)
            final_convs.append(conv_seq[-tap_cache:, :])

            conv_raw = sum(
                conv_seq[tap : tap + L, :].float() * self.conv_weight[tap].float().view(1, -1)
                for tap in range(self.conv_size)
            )
            conv_act = F.silu(conv_raw)
            q_c, k_c, v_c = torch.split(conv_act, [self.d_q_total, self.d_k_total, self.d_v_total], dim=-1)

            q_c = q_c.view(L, Hq, d_k)
            k_c = k_c.view(L, H, d_k)
            v = v_c.view(L, H, d_v)

            q = (q_c * torch.rsqrt(q_c.pow(2).sum(dim=-1, keepdim=True) + 1e-12)) * self.scale
            k = k_c * torch.rsqrt(k_c.pow(2).sum(dim=-1, keepdim=True) + 1e-12)

            b = torch.sigmoid(b_r.float()).view(L, H, d_k) * (2.0 if self.allow_neg_eigval else 1.0)
            w = torch.sigmoid(w_r.float()).view(L, H, d_v)
            g = -torch.exp(self.A_log.float()).view(1, H, 1) * F.softplus(
                f_r.float().view(L, H, d_k) + self.dt_bias.float().view(1, H, d_k)
            )
            if self.beta_guard:
                b = _beta_guard_project_ref(k, b, g)

            o_tokens = []
            with torch.autocast(device_type=device.type, enabled=False):
                S_curr = S_seq
                for t in range(L):
                    alpha_t = torch.exp(g[t]).unsqueeze(1)
                    S_bar = S_curr * alpha_t
                    u_t = b[t] * k[t]
                    r_t = torch.einsum("hvk,hk->hv", S_bar, u_t)
                    delta_v = w[t] * v[t] - r_t
                    S_curr = S_bar + torch.einsum("hv,hk->hvk", delta_v, k[t])

                    q_t = q[t].view(H, G, d_k)
                    o_t = torch.einsum("hvk,hgk->hgv", S_curr, q_t).reshape(Hq, d_v)
                    o_tokens.append(o_t)

            final_states.append(S_curr)
            o_raw = torch.stack(o_tokens, dim=0)  # (L, Hq, d_v)
            o_norm = _rmsnorm_ref(o_raw, self.o_norm_weight.view(1, 1, d_v), self.eps).reshape(L, self.d_out_total)
            out_tok = o_norm.to(dtype) * F.silu(o_gate)
            y_seq = self.W_out(out_tok)

            if self.hidden_dim is not None:
                h = x_seq + y_seq
                x_n2 = _rmsnorm_ref(h, self.norm2_weight, self.eps)
                gate_part, up_part = self.mlp_W_gate_up(x_n2).chunk(2, dim=-1)
                ffn = self.mlp_W_down(F.silu(gate_part) * up_part)
                y_seq = h + ffn

            res_out.append(y_seq)

        if num_seqs > 0:
            final_states_t = torch.stack(final_states, dim=0)
            final_convs_t = torch.stack(final_convs, dim=0)
        else:
            final_states_t = torch.empty(0, H, d_v, d_k, device=device, dtype=torch.float32)
            final_convs_t = torch.empty(0, tap_cache, self.d_qkv_total, device=device, dtype=dtype)

        if len(res_out) == 0:
            return x.new_empty(0, self.d_model), final_states_t, final_convs_t
        if len(res_out) == 1:
            return res_out[0], final_states_t, final_convs_t
        return torch.cat(res_out, dim=0), final_states_t, final_convs_t

    # ---------------------------------------------------------------------
    # Baseline Inference Reference (Pure PyTorch, L = 1 step)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _forward_infer_ref(
        self,
        x: torch.Tensor,
        past_states: Optional[torch.Tensor] = None,
        conv_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        has_seq_dim = (x.dim() == 3)
        if has_seq_dim:
            B, L, _ = x.shape
            assert L == 1, f"Inference ref requires L = 1, got L = {L}"
            x_2d = x.squeeze(1)
        else:
            B = x.shape[0]
            x_2d = x

        H, G, Hq = self.num_kv_heads, self.num_groups, self.num_q_heads
        d_k, d_v = self.d_k, self.d_v
        tap_cache = self.conv_size - 1
        device, dtype = x.device, x.dtype

        if B == 0:
            out_empty = x.new_empty(0, 1, self.d_model) if has_seq_dim else x.new_empty(0, self.d_model)
            st_empty = torch.empty(0, H, d_v, d_k, device=device, dtype=torch.float32)
            cs_empty = torch.empty(0, tap_cache, self.d_qkv_total, device=device, dtype=dtype)
            return out_empty, st_empty, cs_empty

        x_in = _rmsnorm_ref(x_2d, self.norm1_weight, self.eps) if self.hidden_dim is not None else x_2d
        proj = self.W_proj(x_in)
        q_r, k_r, v_r, b_r, w_r, f_r, o_gate = torch.split(proj, self.split_sizes, dim=-1)

        cs_in = (
            conv_state.to(dtype)
            if conv_state is not None
            else torch.zeros(B, tap_cache, self.d_qkv_total, device=device, dtype=dtype)
        )
        cur_qkv = torch.cat([q_r, k_r, v_r], dim=-1)  # (B, D_qkv)
        conv_in = torch.cat([cs_in, cur_qkv.unsqueeze(1)], dim=1)  # (B, 4, D_qkv)
        next_conv_state = conv_in[:, 1:, :]

        conv_raw = (conv_in.float() * self.conv_weight.float().unsqueeze(0)).sum(dim=1)
        conv_act = F.silu(conv_raw)
        q_c, k_c, v_c = torch.split(conv_act, [self.d_q_total, self.d_k_total, self.d_v_total], dim=-1)

        q_c = q_c.view(B, Hq, d_k)
        k_c = k_c.view(B, H, d_k)
        v = v_c.view(B, H, d_v)

        q = (q_c * torch.rsqrt(q_c.pow(2).sum(dim=-1, keepdim=True) + 1e-12)) * self.scale
        k = k_c * torch.rsqrt(k_c.pow(2).sum(dim=-1, keepdim=True) + 1e-12)

        b = torch.sigmoid(b_r.float()).view(B, H, d_k) * (2.0 if self.allow_neg_eigval else 1.0)
        w = torch.sigmoid(w_r.float()).view(B, H, d_v)
        g = -torch.exp(self.A_log.float()).view(1, H, 1) * F.softplus(
            f_r.float().view(B, H, d_k) + self.dt_bias.float().view(1, H, d_k)
        )
        if self.beta_guard:
            b = _beta_guard_project_ref(k, b, g)

        S = (
            past_states.float()
            if past_states is not None
            else torch.zeros(B, H, d_v, d_k, device=device, dtype=torch.float32)
        )

        alpha = torch.exp(g).unsqueeze(2)  # (B, H, 1, d_k)
        S_bar = S * alpha
        r_v = torch.einsum("bhvk,bhk->bhv", S_bar, b * k)
        delta_v = w * v - r_v
        S_next = S_bar + torch.einsum("bhv,bhk->bhvk", delta_v, k)

        o_raw = torch.einsum("bhvk,bhgk->bhgv", S_next, q.view(B, H, G, d_k)).reshape(B, Hq, d_v)
        o_norm = _rmsnorm_ref(o_raw, self.o_norm_weight.view(1, 1, d_v), self.eps).reshape(B, self.d_out_total)
        out_tok = o_norm.to(dtype) * F.silu(o_gate)
        y_mix = self.W_out(out_tok)

        if self.hidden_dim is not None:
            h = x_2d + y_mix
            x_n2 = _rmsnorm_ref(h, self.norm2_weight, self.eps)
            gate_part, up_part = self.mlp_W_gate_up(x_n2).chunk(2, dim=-1)
            ffn = self.mlp_W_down(F.silu(gate_part) * up_part)
            y = h + ffn
        else:
            y = y_mix

        if has_seq_dim:
            y = y.unsqueeze(1)
        return y, S_next, next_conv_state

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        past_states: Optional[torch.Tensor] = None,
        conv_state: Optional[torch.Tensor] = None,
        use_kernel: bool = False,
        varlen: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of Gated DeltaNet-2 `DeltaBlock`.

        Returns:
            `(y, final_states, final_conv_state)` where `final_states` is FP32 `(B, H_kv, d_v, d_k)`
            and `final_conv_state` is FP16/input dtype `(B, conv_size - 1, D_qkv)`.
        """
        is_step = (x.dim() == 2) or (x.dim() == 3 and x.shape[1] == 1)

        if use_kernel:
            if cu_seqlens is None and not self.training and is_step:
                return self._forward_infer_kernel(x, past_states, conv_state)
            return self._forward_kernel(
                x=x,
                cu_seqlens=cu_seqlens,
                padding_mask=padding_mask,
                past_states=past_states,
                conv_state=conv_state,
            )

        if is_step and cu_seqlens is None and not self.training:
            return self._forward_infer_ref(x, past_states, conv_state)

        if cu_seqlens is not None or varlen:
            return self._forward_varlen(
                x=x,
                cu_seqlens=cu_seqlens,
                padding_mask=padding_mask,
                past_states=past_states,
                conv_state=conv_state,
            )

        return self._forward_batch(
            x=x,
            padding_mask=padding_mask,
            past_states=past_states,
            conv_state=conv_state,
        )
