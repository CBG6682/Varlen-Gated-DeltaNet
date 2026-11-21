from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from deltanet_v2.ops import (
    apply_deltablock,
    apply_deltablock_infer,
    _headfirst,
)


class DeltaBlock(nn.Module):
    """
    DeltaNet v2 Transformer Block with Linear Attention Recurrence.

    Implements the DeltaNet recurrence with chunk-wise WY representation,
    Grouped Query Attention (GQA), decay gate (alpha), erase gate (beta),
    write gate (W), and L2 normalization scaling (gamma).

    Supports:
      1. High-performance fused CUDA kernels for variable-length training (use_kernel=True).
      2. Single-step autoregressive decoding kernel (L = 1) for fast inference.
      3. PyTorch reference implementations (batch chunked and varlen token-by-token).
    """

    def __init__(
        self,
        d_model: int,
        num_q_heads: int,
        num_kv_heads: int,
        d_k: int = 64,
        d_v: int = 64,
        chunk_size: int = 64,
        hidden_dim: Optional[int] = None,
        drop_path_rate: float = 0.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0, (
            f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_q_heads // num_kv_heads
        self.d_k = d_k
        self.d_v = d_v
        self.chunk_size = chunk_size
        self.eps = eps

        self.d_q_total = num_q_heads * d_k
        self.d_k_total = num_kv_heads * d_k
        self.d_v_total = num_kv_heads * d_v
        self.d_out_total = num_q_heads * d_v

        self.split_sizes = [
            self.d_q_total,  # q
            self.d_k_total,  # k
            self.d_v_total,  # v
            self.d_k_total,  # b (erase gate)
            self.d_v_total,  # w (write gate)
            num_kv_heads,    # alpha (decay gate)
            num_kv_heads,    # gamma (L2 scaler)
        ]
        self.total_proj_dim = sum(self.split_sizes)
        self.W_proj = nn.Linear(d_model, self.total_proj_dim, bias=False)

        self.rms_norm = nn.RMSNorm(self.d_out_total, eps=eps)
        self.W_gate = nn.Linear(d_model, self.d_out_total, bias=False)
        self.W_out = nn.Linear(self.d_out_total, d_model, bias=False)

        self.hidden_dim = hidden_dim
        self.drop_path_rate = drop_path_rate
        if hidden_dim is not None:
            self.norm1 = nn.RMSNorm(d_model, eps=eps)
            self.norm2 = nn.RMSNorm(d_model, eps=eps)
            self.mlp_W_gate = nn.Linear(d_model, hidden_dim, bias=False)
            self.mlp_W_up = nn.Linear(d_model, hidden_dim, bias=False)
            self.mlp_W_down = nn.Linear(hidden_dim, d_model, bias=False)

    def _token_drop_scales(
        self, x: torch.Tensor, cu_seqlens: torch.Tensor, seed: Optional[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        L = x.shape[0]
        num_seqs = cu_seqlens.shape[0] - 1
        keep = 1.0 - self.drop_path_rate
        g = None
        if seed is not None:
            g = torch.Generator(device=x.device)
            g.manual_seed(int(seed))
        dp_scale = (
            (torch.rand(2, num_seqs, generator=g, device=x.device) < keep).float() / keep
        ).to(x.dtype)

        bounds = cu_seqlens[1:]
        tok = torch.arange(L, device=x.device, dtype=bounds.dtype)
        seq_id = (tok[:, None] >= bounds[None, :]).sum(-1)
        onehot = (
            seq_id[:, None] == torch.arange(num_seqs, device=x.device)[None, :]
        ).to(x.dtype)
        s1 = (onehot * dp_scale[0][None, :]).sum(-1, keepdim=True)
        s2 = (onehot * dp_scale[1][None, :]).sum(-1, keepdim=True)
        return s1, s2

    # ---------------------------------------------------------------------
    # CUDA Kernel Branch: Training (varlen)
    # ---------------------------------------------------------------------
    def _forward_kernel(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        past_states: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert cu_seqlens is not None, (
            "use_kernel=True in training mode requires cu_seqlens (packed varlen)"
        )
        L = x.shape[0]
        H, Hq, d_k, d_v = self.num_kv_heads, self.num_q_heads, self.d_k, self.d_v

        proj = self.W_proj(x)
        if proj.dtype != torch.float16:
            proj = proj.to(torch.float16)

        q_raw, k_raw, v_raw, b_raw, w_raw, a_raw, g_raw = torch.split(
            proj, self.split_sizes, dim=-1
        )
        q = _headfirst(q_raw, L, Hq, d_k)
        k = _headfirst(k_raw, L, H, d_k)
        v = _headfirst(v_raw, L, H, d_v)
        w = _headfirst(w_raw, L, H, d_v)
        b = _headfirst(b_raw, L, H, d_k)
        a = _headfirst(a_raw, L, H, 1)
        g = _headfirst(g_raw, L, H, 1)
        del proj, q_raw, k_raw, v_raw, a_raw, b_raw, w_raw, g_raw

        y_h, final_states = apply_deltablock(
            past_states, q, k, v, w, b, a, g, cu_seqlens,
            Hq, H, d_k, d_v, self.chunk_size,
        )

        out = y_h.transpose(0, 1).reshape(L, self.d_out_total)
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        out = self.rms_norm(out)
        out = out * F.silu(self.W_gate(x))
        y = self.W_out(out)

        if self.hidden_dim is not None:
            s1 = s2 = 1.0
            if self.training and self.drop_path_rate > 0.0:
                s1, s2 = self._token_drop_scales(x, cu_seqlens, seed)
            x_norm = self.norm1(x + y * s1)
            ffn = self.mlp_W_down(F.silu(self.mlp_W_gate(x_norm)) * self.mlp_W_up(x_norm))
            y = self.norm2(x_norm + ffn * s2)

        return y, final_states

    # ---------------------------------------------------------------------
    # CUDA Kernel Branch: Autoregressive Decoding (L = 1)
    # ---------------------------------------------------------------------
    def _forward_infer_kernel(
        self, x: torch.Tensor, past_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        has_seq_dim = (x.dim() == 3)
        if has_seq_dim:
            B, L, _ = x.shape
            assert L == 1, f"Inference kernel requires L = 1, got L = {L}"
            x_2d = x.squeeze(1)
        else:
            B = x.shape[0]
            x_2d = x

        H, Hq, d_k, d_v = self.num_kv_heads, self.num_q_heads, self.d_k, self.d_v

        if past_states is None:
            past_states = torch.zeros(B, H, d_v, d_k, device=x.device, dtype=torch.float16)
        elif past_states.dtype != torch.float16 or not past_states.is_contiguous():
            past_states = past_states.to(torch.float16).contiguous()

        proj = self.W_proj(x_2d)
        if proj.dtype != torch.float16:
            proj = proj.to(torch.float16)

        q, k, v, b, w, a, g = torch.split(proj, self.split_sizes, dim=-1)

        q = q.view(B, Hq, d_k).contiguous()
        k = k.view(B, H, d_k).contiguous()
        v = v.view(B, H, d_v).contiguous()
        b = b.view(B, H, d_k).contiguous()
        w = w.view(B, H, d_v).contiguous()
        a = a.view(B, H, 1).contiguous()
        g = g.view(B, H, 1).contiguous()

        out_h, final_states = apply_deltablock_infer(
            past_states, q, k, v, b, w, a, g,
            Hq, H, d_k, d_v,
        )

        out = out_h.view(B, self.d_out_total)
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        out = self.rms_norm(out)
        out = out * F.silu(self.W_gate(x_2d))
        y = self.W_out(out)

        if self.hidden_dim is not None:
            x_norm = self.norm1(x_2d + y)
            ffn = self.mlp_W_down(F.silu(self.mlp_W_gate(x_norm)) * self.mlp_W_up(x_norm))
            y = self.norm2(x_norm + ffn)

        if has_seq_dim:
            y = y.unsqueeze(1)

        return y, final_states

    # ---------------------------------------------------------------------
    # Baseline Batch (Pure PyTorch, chunked recurrence)
    # ---------------------------------------------------------------------
    def _forward_batch(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        past_states: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        x_orig = x
        device, dtype = x.device, x.dtype
        C = self.chunk_size
        H, G = self.num_kv_heads, self.num_groups
        d_k, d_v = self.d_k, self.d_v

        if padding_mask is not None:
            padding_mask = padding_mask.bool()

        pad_len = (-L) % C
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))
            if padding_mask is None:
                padding_mask = torch.ones(B, L, dtype=torch.bool, device=device)
            padding_mask = F.pad(padding_mask, (0, pad_len), value=False)

        Lp = L + pad_len
        num_chunks = Lp // C

        has_mask = padding_mask is not None
        if has_mask:
            m = padding_mask[:, :, None, None]
            x = x.masked_fill(~padding_mask.unsqueeze(-1), 0.0)

        q_raw, k_raw, v_raw, b_raw, w_raw, alpha_raw, gamma_raw = torch.split(
            self.W_proj(x), self.split_sizes, dim=-1
        )

        log_alpha = F.logsigmoid(alpha_raw.float()).view(B, Lp, H, 1)
        b = torch.sigmoid(b_raw).view(B, Lp, H, d_k)
        w = torch.sigmoid(w_raw).view(B, Lp, H, d_v)

        if has_mask:
            log_alpha = log_alpha.masked_fill(~m, 0.0)
            b = b * m
            w = w * m

        q = q_raw.view(B, Lp, self.num_q_heads, d_k)
        k = F.normalize(k_raw.view(B, Lp, H, d_k).float(), p=2, dim=-1, eps=1e-6).to(k_raw.dtype)
        if has_mask:
            k = k * m
        v = v_raw.view(B, Lp, H, d_v)
        sqrt_gamma = F.softplus(gamma_raw).view(B, Lp, H, 1).sqrt()

        k_scaled = k * sqrt_gamma
        v_scaled = v * sqrt_gamma

        u = b * k_scaled
        v_new = w * v_scaled

        q_c = q.view(B, num_chunks, C, H, G, d_k)
        k_c = k_scaled.view(B, num_chunks, C, H, d_k)
        v_new_c = v_new.view(B, num_chunks, C, H, d_v)
        u_c = u.view(B, num_chunks, C, H, d_k)
        log_alpha_c = log_alpha.view(B, num_chunks, C, H, 1)

        with torch.autocast(device_type=device.type, enabled=False):
            cum = torch.cumsum(log_alpha_c, dim=2)
            alpha_cumprod = cum.exp()
            cum_last = cum[:, :, -1:]
            chunk_decay = cum_last.exp()
            k_decayed = k_c.float() * torch.exp(cum_last - cum)
            erase_matrix = torch.einsum(
                "b c t h k, b c t h m -> b c h k m", u_c.float(), k_decayed
            )
            I = torch.eye(d_k, device=device, dtype=torch.float32).view(1, 1, 1, d_k, d_k)
            decay_val = chunk_decay.squeeze(2).unsqueeze(-1)
            W_chunks = I * decay_val - erase_matrix
            Y_chunks = torch.einsum(
                "b c t h v, b c t h k -> b c h v k", v_new_c.float(), k_decayed
            )

            if past_states is None:
                S_curr = torch.zeros(B, H, d_v, d_k, device=device, dtype=torch.float32)
            else:
                S_curr = past_states.float()
            S_list = []
            for c in range(num_chunks):
                S_list.append(S_curr)
                S_curr = torch.matmul(S_curr, W_chunks[:, c]) + Y_chunks[:, c]
            S_all = torch.stack(S_list, dim=1)
        final_states = S_curr.to(dtype)

        attn = torch.einsum("b c i h g d, b c j h d -> b c h g i j", q_c, k_c)
        causal_mask = torch.tril(torch.ones(C, C, device=device, dtype=torch.bool))
        a = cum.squeeze(-1).permute(0, 1, 3, 2)
        diff = a.unsqueeze(-1) - a.unsqueeze(-2)
        diff = diff.masked_fill(~causal_mask, float("-inf"))
        attn = attn * diff.exp().unsqueeze(3).to(attn.dtype)
        out_intra = torch.einsum("b c h g i j, b c j h v -> b c i h g v", attn, v_new_c)

        with torch.autocast(device_type=device.type, enabled=False):
            q_S = torch.einsum("b c t h g k, b c h v k -> b c t h g v", q_c.float(), S_all)
            out_inter = q_S * alpha_cumprod.unsqueeze(-2)
        out = (out_intra + out_inter.to(out_intra.dtype)).reshape(B, Lp, self.d_out_total)[:, :L, :]

        out = self.rms_norm(out)
        out = out * F.silu(self.W_gate(x_orig))
        y = self.W_out(out)
        if has_mask:
            y = y.masked_fill(~padding_mask[:, :L].unsqueeze(-1), 0.0)

        if self.hidden_dim is not None:
            if self.training and self.drop_path_rate > 0.0:
                keep = 1.0 - self.drop_path_rate
                g = None
                if seed is not None:
                    g = torch.Generator(device=device)
                    g.manual_seed(int(seed))
                s = ((torch.rand(2, B, generator=g, device=device) < keep).float() / keep).to(dtype)
                s1, s2 = s[0].view(B, 1, 1), s[1].view(B, 1, 1)
            else:
                s1 = s2 = 1.0

            x_norm = self.norm1(x_orig + y * s1)
            ffn = self.mlp_W_down(F.silu(self.mlp_W_gate(x_norm)) * self.mlp_W_up(x_norm))
            y = self.norm2(x_norm + ffn * s2)
            if has_mask:
                y = y.masked_fill(~padding_mask[:, :L].unsqueeze(-1), 0.0)

        return y, final_states

    # ---------------------------------------------------------------------
    # Baseline Varlen: Token-by-token recurrence in pure PyTorch
    # ---------------------------------------------------------------------
    def _forward_varlen(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        past_states: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cu_seqlens is None:
            cu_seqlens = torch.tensor([0, x.shape[0]], dtype=torch.int32, device=x.device)
        cu = cu_seqlens.tolist()
        num_seqs = len(cu) - 1
        H, G = self.num_kv_heads, self.num_groups
        d_k, d_v = self.d_k, self.d_v
        C = self.chunk_size
        device, dtype = x.device, x.dtype

        use_dp = self.hidden_dim is not None and self.training and self.drop_path_rate > 0.0
        if use_dp:
            keep = 1.0 - self.drop_path_rate
            g = None
            if seed is not None:
                g = torch.Generator(device=device)
                g.manual_seed(int(seed))
            dp_scale = ((torch.rand(2, num_seqs, generator=g, device=device) < keep).float() / keep).to(dtype)

        res_out = []
        final_states = []

        for i in range(num_seqs):
            start, end = int(cu[i]), int(cu[i + 1])
            if start == end:
                final_states.append(
                    past_states[i].to(dtype)
                    if past_states is not None
                    else torch.zeros(H, d_v, d_k, device=device, dtype=dtype)
                )
                continue

            x_seq = x[start:end]
            L = end - start

            proj = self.W_proj(x_seq)
            q_raw, k_raw, v_raw, b_raw, w_raw, alpha_raw, gamma_raw = torch.split(
                proj, self.split_sizes, dim=-1
            )

            q = q_raw.view(L, self.num_q_heads, d_k).transpose(0, 1)
            k = F.normalize(k_raw.view(L, H, d_k), p=2, dim=-1)
            sqrt_gamma = F.softplus(gamma_raw).view(L, H, 1).sqrt()
            k_scaled = (k * sqrt_gamma).transpose(0, 1)
            v_scaled = (v_raw.view(L, H, d_v) * sqrt_gamma).transpose(0, 1)

            b = torch.sigmoid(b_raw).view(L, H, d_k).transpose(0, 1)
            w = torch.sigmoid(w_raw).view(L, H, d_v).transpose(0, 1)
            log_alpha = F.logsigmoid(alpha_raw.float()).view(L, H).transpose(0, 1)

            u = b * k_scaled
            v_new = w * v_scaled

            n_full = (L // C) * C
            cum = torch.empty(H, L, device=device, dtype=torch.float32)
            if n_full > 0:
                cum[:, :n_full] = (
                    log_alpha[:, :n_full].reshape(H, n_full // C, C).cumsum(dim=-1).reshape(H, n_full)
                )
            if n_full < L:
                cum[:, n_full:] = log_alpha[:, n_full:].cumsum(dim=-1)
            cum = cum.unsqueeze(-1)
            log_alpha = log_alpha.unsqueeze(-1)

            past_seq = past_states[i] if past_states is not None else None
            S_curr = (
                past_seq.float()
                if past_seq is not None
                else torch.zeros(H, d_v, d_k, device=device, dtype=torch.float32)
            )

            u32, k32, v32, q32 = u.float(), k_scaled.float(), v_new.float(), q.float()
            o_tokens = []
            with torch.autocast(device_type=device.type, enabled=False):
                for t in range(L):
                    alpha_t = log_alpha[:, t, :].exp().unsqueeze(-1)
                    u_t = u32[:, t, :].unsqueeze(-1)
                    k_t = k32[:, t, :].unsqueeze(-1)
                    v_t = v32[:, t, :].unsqueeze(-1)

                    erase = torch.matmul(torch.matmul(S_curr, u_t), k_t.transpose(-1, -2))
                    write = torch.matmul(v_t, k_t.transpose(-1, -2))
                    S_curr = S_curr * alpha_t - erase + write

                    q_t = q32[:, t, :].reshape(H, G, d_k, 1)
                    o_t = torch.matmul(S_curr.unsqueeze(1), q_t).squeeze(-1)
                    o_tokens.append(o_t.reshape(self.num_q_heads, d_v).to(dtype))

            final_states.append(S_curr.to(dtype))
            out_combine = torch.stack(o_tokens, dim=1)
            out = out_combine.transpose(0, 1).reshape(L, self.d_out_total)

            out = self.rms_norm(out)
            out = out * F.silu(self.W_gate(x_seq))
            y_seq = self.W_out(out)

            if self.hidden_dim is not None:
                s1, s2 = (dp_scale[0, i], dp_scale[1, i]) if use_dp else (1.0, 1.0)
                x_norm = self.norm1(x_seq + y_seq * s1)
                ffn = self.mlp_W_down(F.silu(self.mlp_W_gate(x_norm)) * self.mlp_W_up(x_norm))
                y_seq = self.norm2(x_norm + ffn * s2)

            res_out.append(y_seq)

        if num_seqs > 0:
            final_states = torch.stack(final_states, dim=0)
        else:
            final_states = x.new_empty(0, H, d_v, d_k)

        if len(res_out) == 0:
            return x.new_empty(0, self.d_model), final_states
        if len(res_out) == 1:
            return res_out[0], final_states
        return torch.cat(res_out, dim=0), final_states

    # ---------------------------------------------------------------------
    # Baseline Inference Reference (Pure PyTorch, L = 1 step)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _forward_infer_ref(
        self, x: torch.Tensor, past_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        has_seq_dim = (x.dim() == 3)
        if has_seq_dim:
            B, L, _ = x.shape
            assert L == 1, f"Inference ref requires L = 1, got L = {L}"
            x_2d = x.squeeze(1)
        else:
            B = x.shape[0]
            x_2d = x

        H, G = self.num_kv_heads, self.num_groups
        d_k, d_v = self.d_k, self.d_v
        dtype = x.dtype

        proj = self.W_proj(x_2d)
        q_raw, k_raw, v_raw, b_raw, w_raw, alpha_raw, gamma_raw = torch.split(
            proj, self.split_sizes, dim=-1
        )

        alpha      = torch.sigmoid(alpha_raw.float()).view(B, H, 1, 1)
        b          = torch.sigmoid(b_raw.float()).view(B, H, d_k)
        w          = torch.sigmoid(w_raw.float()).view(B, H, d_v)
        sqrt_gamma = F.softplus(gamma_raw.float()).sqrt().view(B, H, 1)

        q = q_raw.float().view(B, H, G, d_k)
        k = F.normalize(k_raw.float().view(B, H, d_k), p=2, dim=-1, eps=1e-6)
        k_scaled = k * sqrt_gamma
        v_scaled = v_raw.float().view(B, H, d_v) * sqrt_gamma
        u     = b * k_scaled
        v_new = w * v_scaled

        if past_states is None:
            S = torch.zeros(B, H, d_v, d_k, device=x.device, dtype=torch.float32)
        else:
            S = past_states.float()

        qu   = torch.cat([q, u.unsqueeze(2)], dim=2)
        S_qu = torch.einsum("bhvk,bhrk->bhrv", S, qu)
        q_S, S_u = S_qu[:, :, :G], S_qu[:, :, G]

        attn      = torch.einsum("bhgk,bhk->bhg", q, k_scaled)
        out_inter = q_S * alpha
        out_intra = attn.unsqueeze(-1) * v_new.unsqueeze(2)
        out_h     = (out_inter + out_intra).reshape(B, self.d_out_total)

        delta_v = v_new - S_u
        S_next  = S * alpha + torch.einsum("bhv,bhk->bhvk", delta_v, k_scaled)
        final_states = S_next.to(dtype)

        out = out_h.to(dtype)
        out = self.rms_norm(out)
        out = out * F.silu(self.W_gate(x_2d))
        y = self.W_out(out)

        if self.hidden_dim is not None:
            x_norm = self.norm1(x_2d + y)
            ffn = self.mlp_W_down(F.silu(self.mlp_W_gate(x_norm)) * self.mlp_W_up(x_norm))
            y = self.norm2(x_norm + ffn)

        if has_seq_dim:
            y = y.unsqueeze(1)
        return y, final_states

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        past_states: Optional[torch.Tensor] = None,
        use_kernel: bool = False,
        varlen: bool = False,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward computation of DeltaBlock.

        Args:
            x: Input tensor. For varlen training: (total_tokens, d_model).
               For batched prefill: (batch, seq_len, d_model).
               For decode: (batch, 1, d_model) or (batch, d_model).
            cu_seqlens: Cumulative sequence lengths (batch_size + 1,) [int32].
            padding_mask: Boolean mask (batch, seq_len) for batched input.
            past_states: Hidden recurrence state (batch, num_kv_heads, d_v, d_k) [float16].
            use_kernel: Whether to dispatch to high-performance CUDA kernels.
            varlen: Whether to run variable-length sequence logic in baseline PyTorch.
            seed: Optional seed for stochastic drop path reproducibility.
        """
        if use_kernel:
            if cu_seqlens is not None or self.training:
                return self._forward_kernel(x, cu_seqlens, past_states, seed)
            return self._forward_infer_kernel(x, past_states)

        is_step = (x.dim() == 2) or (x.dim() == 3 and x.shape[1] == 1)
        if is_step and cu_seqlens is None and not self.training:
            return self._forward_infer_ref(x, past_states)

        if cu_seqlens is not None or varlen:
            return self._forward_varlen(
                x=x, cu_seqlens=cu_seqlens, padding_mask=padding_mask,
                past_states=past_states, seed=seed,
            )

        return self._forward_batch(
            x=x, padding_mask=padding_mask, past_states=past_states, seed=seed,
        )
