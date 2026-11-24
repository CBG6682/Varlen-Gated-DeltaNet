#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "deltanet_common.cuh"
#include "deltanet_ops.cuh"

// ============================================================================
// Part 2: Full DeltaBlock Host Coordinators (deltablock_fwd & deltablock_bwd)
// ============================================================================

// Outputs: [y, final_states, final_conv_state, ...saved_for_backward_if_is_training]
std::vector<at::Tensor>
deltablock_fwd(
    const at::Tensor& x,
    const at::Tensor& cu_seq_lens,
    const std::optional<at::Tensor>& past_states,
    const std::optional<at::Tensor>& past_conv_state,
    const at::Tensor& W_proj,
    const at::Tensor& conv_weight,
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    const at::Tensor& o_norm_weight,
    const at::Tensor& W_out,
    const std::optional<at::Tensor>& norm1_weight,
    const std::optional<at::Tensor>& norm2_weight,
    const std::optional<at::Tensor>& mlp_W_gate_up,
    const std::optional<at::Tensor>& mlp_W_down,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t d_k,
    int64_t d_v,
    int64_t conv_size,
    int64_t chunk_size,
    double scale,
    double eps,
    bool allow_neg_eigval,
    bool beta_guard,
    bool is_training,
    bool output_conv_state
) {
    // 1. Input Validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    const c10::cuda::CUDAGuard device_guard(x.device());

    auto check_t = [&](const at::Tensor& t, const char* name, int64_t expected_dim, at::ScalarType dtype) {
        TORCH_CHECK(t.is_cuda() && t.device() == x.device(), name, " must be on the same CUDA device as x");
        TORCH_CHECK(t.scalar_type() == dtype, name, " dtype mismatch");
        TORCH_CHECK(t.dim() == expected_dim && t.is_contiguous(), name, " must be contiguous with expected rank");
    };

    check_t(x,             "x",             2, at::ScalarType::Half);
    check_t(W_proj,        "W_proj",        2, at::ScalarType::Half);
    check_t(conv_weight,   "conv_weight",   2, at::ScalarType::Half);
    check_t(A_log,         "A_log",         1, at::ScalarType::Float);
    check_t(dt_bias,       "dt_bias",       2, at::ScalarType::Float);
    check_t(o_norm_weight, "o_norm_weight", 1, at::ScalarType::Half);
    check_t(W_out,         "W_out",         2, at::ScalarType::Half);

    TORCH_CHECK(cu_seq_lens.is_cuda() && cu_seq_lens.device() == x.device() &&
                cu_seq_lens.dim() == 1 && cu_seq_lens.is_contiguous() && cu_seq_lens.size(0) >= 1 &&
                (cu_seq_lens.scalar_type() == at::ScalarType::Int || cu_seq_lens.scalar_type() == at::ScalarType::Long),
                "cu_seq_lens must be a contiguous 1D Int32/Int64 CUDA tensor");

    TORCH_CHECK(d_k == 64 && d_v == 64, "d_k and d_v must be 64");
    TORCH_CHECK(conv_size == 4 && chunk_size == 64, "conv_size must be 4 and chunk_size must be 64");
    TORCH_CHECK(num_kv_heads > 0 && num_q_heads > 0 && (num_q_heads % num_kv_heads == 0),
                "num_q_heads must be positive and divisible by num_kv_heads");

    int64_t total_L        = x.size(0);
    int64_t d_model        = x.size(1);
    int64_t batch_size     = cu_seq_lens.size(0) - 1;
    int64_t D_q            = num_q_heads * d_k;
    int64_t D_k            = num_kv_heads * d_k;
    int64_t D_v            = num_kv_heads * d_v;
    int64_t D_qkv          = D_q + D_k + D_v;
    int64_t D_out          = num_q_heads * d_v;
    int64_t total_proj_dim = D_qkv + 2 * D_k + D_v + D_out;

    TORCH_CHECK(d_model > 0 && d_model % 8 == 0, "d_model must be positive and divisible by 8");
    TORCH_CHECK(W_proj.size(0) == total_proj_dim && W_proj.size(1) == d_model, "W_proj shape mismatch");
    TORCH_CHECK(conv_weight.size(0) == conv_size && conv_weight.size(1) == D_qkv, "conv_weight shape mismatch");
    TORCH_CHECK(A_log.size(0) == num_kv_heads, "A_log shape mismatch");
    TORCH_CHECK(dt_bias.size(0) == num_kv_heads && dt_bias.size(1) == d_k, "dt_bias shape mismatch");
    TORCH_CHECK(o_norm_weight.size(0) == d_v, "o_norm_weight shape mismatch");
    TORCH_CHECK(W_out.size(0) == d_model && W_out.size(1) == D_out, "W_out shape mismatch");

    bool has_mlp = mlp_W_gate_up.has_value() && mlp_W_gate_up->defined();
    TORCH_CHECK(has_mlp == (norm1_weight.has_value() && norm1_weight->defined()) &&
                has_mlp == (norm2_weight.has_value() && norm2_weight->defined()) &&
                has_mlp == (mlp_W_down.has_value() && mlp_W_down->defined()),
                "norm1_weight, norm2_weight, mlp_W_gate_up, and mlp_W_down must all be provided or omitted together");

    int64_t hidden_dim = 0;
    if (has_mlp) {
        check_t(*norm1_weight,  "norm1_weight",  1, at::ScalarType::Half);
        check_t(*norm2_weight,  "norm2_weight",  1, at::ScalarType::Half);
        check_t(*mlp_W_gate_up, "mlp_W_gate_up", 2, at::ScalarType::Half);
        check_t(*mlp_W_down,    "mlp_W_down",    2, at::ScalarType::Half);
        TORCH_CHECK(norm1_weight->size(0) == d_model, "norm1_weight shape mismatch");
        TORCH_CHECK(norm2_weight->size(0) == d_model, "norm2_weight shape mismatch");
        TORCH_CHECK(mlp_W_gate_up->size(1) == d_model && mlp_W_gate_up->size(0) > 0 && (mlp_W_gate_up->size(0) % 16 == 0),
                    "mlp_W_gate_up shape invalid");
        hidden_dim = mlp_W_gate_up->size(0) / 2;
        TORCH_CHECK(mlp_W_down->size(0) == d_model && mlp_W_down->size(1) == hidden_dim, "mlp_W_down shape mismatch");
    }

    if (past_states.has_value() && past_states->defined()) {
        check_t(*past_states, "past_states", 4, at::ScalarType::Float);
        TORCH_CHECK(past_states->size(0) == batch_size && past_states->size(1) == num_kv_heads &&
                    past_states->size(2) == d_v && past_states->size(3) == d_k,
                    "past_states shape mismatch");
    }
    if (past_conv_state.has_value() && past_conv_state->defined()) {
        check_t(*past_conv_state, "past_conv_state", 3, at::ScalarType::Half);
        TORCH_CHECK(past_conv_state->size(0) == batch_size && past_conv_state->size(1) == conv_size - 1 &&
                    past_conv_state->size(2) == D_qkv,
                    "past_conv_state shape mismatch");
    }

    // 2. Shared Varlen Metadata & PreNorm 1 + Unified Projection
    VarlenChunkMeta meta = compute_varlen_chunk_metadata(cu_seq_lens, chunk_size);
    auto h_opts = x.options();
    auto f_opts = x.options().dtype(at::kFloat);

    at::Tensor x_in;
    at::Tensor rstd1;
    if (has_mlp) {
        std::tie(x_in, rstd1) = fused_rmsnorm_fwd(x, *norm1_weight, eps, is_training);
    } else {
        x_in  = x;
        rstd1 = is_training ? at::empty({0}, f_opts) : at::Tensor();
    }

    at::Tensor proj = at::mm(x_in, W_proj.t());
    if (!is_training && has_mlp) {
        x_in.reset();
    }

    // 3. Projection Epilogue: ShortConv1D + SiLU + L2Norm + Channel-wise Decay + BetaGuard
    auto [q_h, k_h, v_h, b_eff_h, w_h, G_log2_h, final_conv_state, q_inv_norm, k_inv_norm] =
        proj_conv_prep_fwd_with_meta(
            proj, conv_weight, A_log, dt_bias, past_conv_state, cu_seq_lens, meta,
            num_q_heads, num_kv_heads, d_k, d_v, conv_size, chunk_size,
            scale, allow_neg_eigval, beta_guard, is_training, output_conv_state
        );

    // 4. Chunk WY Forward & Sequential Scan (R_list overwrites U in-place)
    auto [Y, U, A_list] = chunk_wy_fwd_with_meta(
        k_h, v_h, b_eff_h, w_h, G_log2_h, meta, chunk_size, is_training
    );
    if (!is_training) {
        v_h.reset();
        b_eff_h.reset();
        w_h.reset();
    }

    auto [S_list, R_list, final_states] = chunk_scan_fwd_with_meta(
        Y, U, k_h, G_log2_h, past_states, meta, batch_size, chunk_size
    );
    if (!is_training) {
        Y.reset();
    }

    // 5. Output Combine + Per-Head RMSNorm * SiLU(o_gate) + Output Projection
    auto [out_tok, O_raw, o_rstd] = output_combine_fwd_with_meta(
        q_h, k_h, G_log2_h, S_list, R_list, proj, o_norm_weight, meta,
        num_q_heads, num_kv_heads, d_k, d_v, chunk_size, eps, is_training
    );
    if (!is_training) {
        q_h.reset();
        k_h.reset();
        G_log2_h.reset();
        S_list.reset();
        R_list.reset();
        U.reset();
        proj.reset();
    }

    at::Tensor y_mix = at::mm(out_tok, W_out.t());
    if (!is_training) {
        out_tok.reset();
    }

    if (!has_mlp) {
        if (!is_training) {
            return {y_mix, final_states, final_conv_state};
        }
        at::Tensor empty_h  = at::empty({0}, h_opts);
        at::Tensor empty_f  = at::empty({0}, f_opts);
        return {
            y_mix, final_states, final_conv_state,
            x_in, rstd1, proj, q_h, k_h, v_h, b_eff_h, w_h, G_log2_h,
            q_inv_norm, k_inv_norm, A_list, Y, S_list, R_list, O_raw, o_rstd, out_tok,
            empty_h, empty_h, empty_f, empty_h, empty_h
        };
    }

    // 6. Post-Mixer Residual + PreNorm 2 + SwiGLU MLP + Final Residual
    auto [h, x_n2, rstd2] = fused_residual_rmsnorm_fwd(x, y_mix, *norm2_weight, eps, is_training);
    y_mix.reset();

    at::Tensor gate_up = at::mm(x_n2, mlp_W_gate_up->t());
    if (!is_training) {
        x_n2.reset();
    }

    at::Tensor act = fused_swiglu_fwd(gate_up);
    if (!is_training) {
        gate_up.reset();
    }

    at::Tensor ffn = at::mm(act, mlp_W_down->t());
    if (!is_training) {
        act.reset();
    }

    at::Tensor y = fused_residual_add_fwd(h, ffn);
    ffn.reset();
    if (!is_training) {
        h.reset();
        return {y, final_states, final_conv_state};
    }

    return {
        y, final_states, final_conv_state,
        x_in, rstd1, proj, q_h, k_h, v_h, b_eff_h, w_h, G_log2_h,
        q_inv_norm, k_inv_norm, A_list, Y, S_list, R_list, O_raw, o_rstd, out_tok,
        h, x_n2, rstd2, gate_up, act
    };
}

// Outputs: [grad_x, grad_past_states, grad_past_conv_state, dW_proj, d_conv_weight, d_A_log, d_dt_bias, d_o_norm_weight, dW_out, d_norm1_weight, d_norm2_weight, d_mlp_W_gate_up, d_mlp_W_down]
std::vector<at::Tensor>
deltablock_bwd(
    const at::Tensor& grad_y,
    const std::optional<at::Tensor>& grad_final_states,
    const at::Tensor& x,
    const at::Tensor& cu_seq_lens,
    const std::optional<at::Tensor>& past_states,
    const std::optional<at::Tensor>& past_conv_state,
    const at::Tensor& W_proj,
    const at::Tensor& conv_weight,
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    const at::Tensor& o_norm_weight,
    const at::Tensor& W_out,
    const std::optional<at::Tensor>& norm1_weight,
    const std::optional<at::Tensor>& norm2_weight,
    const std::optional<at::Tensor>& mlp_W_gate_up,
    const std::optional<at::Tensor>& mlp_W_down,
    const at::Tensor& x_in,
    const at::Tensor& rstd1,
    const at::Tensor& proj,
    const at::Tensor& q_h,
    const at::Tensor& k_h,
    const at::Tensor& v_h,
    const at::Tensor& b_eff_h,
    const at::Tensor& w_h,
    const at::Tensor& G_log2_h,
    const at::Tensor& q_inv_norm,
    const at::Tensor& k_inv_norm,
    const at::Tensor& A_list,
    const at::Tensor& Y,
    const at::Tensor& S_list,
    const at::Tensor& R_list,
    const at::Tensor& O_raw,
    const at::Tensor& o_rstd,
    const at::Tensor& out_tok,
    const at::Tensor& h,
    const at::Tensor& x_n2,
    const at::Tensor& rstd2,
    const at::Tensor& gate_up,
    const at::Tensor& act,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t d_k,
    int64_t d_v,
    int64_t conv_size,
    int64_t chunk_size,
    double scale,
    bool allow_neg_eigval,
    bool has_past_states,
    bool has_past_conv_state
) {
    // 1. Input Validation
    TORCH_CHECK(grad_y.is_cuda(), "grad_y must be a CUDA tensor");
    const c10::cuda::CUDAGuard device_guard(grad_y.device());

    auto check_t = [&](const at::Tensor& t, const char* name, int64_t expected_dim, at::ScalarType dtype) {
        TORCH_CHECK(t.is_cuda() && t.device() == grad_y.device(), name, " must be on the same CUDA device as grad_y");
        TORCH_CHECK(t.scalar_type() == dtype, name, " dtype mismatch");
        TORCH_CHECK(t.dim() == expected_dim && t.is_contiguous(), name, " must be contiguous with expected rank");
    };

    check_t(grad_y,        "grad_y",        2, at::ScalarType::Half);
    check_t(x,             "x",             2, at::ScalarType::Half);
    check_t(W_proj,        "W_proj",        2, at::ScalarType::Half);
    check_t(conv_weight,   "conv_weight",   2, at::ScalarType::Half);
    check_t(A_log,         "A_log",         1, at::ScalarType::Float);
    check_t(dt_bias,       "dt_bias",       2, at::ScalarType::Float);
    check_t(o_norm_weight, "o_norm_weight", 1, at::ScalarType::Half);
    check_t(W_out,         "W_out",         2, at::ScalarType::Half);

    check_t(x_in,       "x_in",       2, at::ScalarType::Half);
    check_t(proj,       "proj",       2, at::ScalarType::Half);
    check_t(q_h,        "q_h",        3, at::ScalarType::Half);
    check_t(k_h,        "k_h",        3, at::ScalarType::Half);
    check_t(v_h,        "v_h",        3, at::ScalarType::Half);
    check_t(b_eff_h,    "b_eff_h",    3, at::ScalarType::Half);
    check_t(w_h,        "w_h",        3, at::ScalarType::Half);
    check_t(G_log2_h,   "G_log2_h",   3, at::ScalarType::Float);
    check_t(q_inv_norm, "q_inv_norm", 2, at::ScalarType::Float);
    check_t(k_inv_norm, "k_inv_norm", 2, at::ScalarType::Float);
    check_t(A_list,     "A_list",     4, at::ScalarType::Half);
    check_t(Y,          "Y",          4, at::ScalarType::Half);
    check_t(S_list,     "S_list",     4, at::ScalarType::Float);
    check_t(R_list,     "R_list",     4, at::ScalarType::Half);
    check_t(O_raw,      "O_raw",      3, at::ScalarType::Half);
    check_t(o_rstd,     "o_rstd",     2, at::ScalarType::Float);
    check_t(out_tok,    "out_tok",    2, at::ScalarType::Half);

    TORCH_CHECK(cu_seq_lens.is_cuda() && cu_seq_lens.device() == grad_y.device() &&
                cu_seq_lens.dim() == 1 && cu_seq_lens.is_contiguous() && cu_seq_lens.size(0) >= 1 &&
                (cu_seq_lens.scalar_type() == at::ScalarType::Int || cu_seq_lens.scalar_type() == at::ScalarType::Long),
                "cu_seq_lens must be a contiguous 1D Int32/Int64 CUDA tensor");

    TORCH_CHECK(d_k == 64 && d_v == 64, "d_k and d_v must be 64");
    TORCH_CHECK(conv_size == 4 && chunk_size == 64, "conv_size must be 4 and chunk_size must be 64");
    TORCH_CHECK(num_kv_heads > 0 && num_q_heads > 0 && (num_q_heads % num_kv_heads == 0),
                "num_q_heads must be positive and divisible by num_kv_heads");

    int64_t total_L        = x.size(0);
    int64_t d_model        = x.size(1);
    int64_t batch_size     = cu_seq_lens.size(0) - 1;
    int64_t D_q            = num_q_heads * d_k;
    int64_t D_k            = num_kv_heads * d_k;
    int64_t D_v            = num_kv_heads * d_v;
    int64_t D_qkv          = D_q + D_k + D_v;
    int64_t D_out          = num_q_heads * d_v;
    int64_t total_proj_dim = D_qkv + 2 * D_k + D_v + D_out;

    TORCH_CHECK(d_model > 0 && d_model % 8 == 0, "d_model must be positive and divisible by 8");
    TORCH_CHECK(grad_y.size(0) == total_L && grad_y.size(1) == d_model, "grad_y shape mismatch");
    TORCH_CHECK(x_in.size(0) == total_L && x_in.size(1) == d_model, "x_in shape mismatch");
    TORCH_CHECK(proj.size(0) == total_L && proj.size(1) == total_proj_dim, "proj shape mismatch");
    TORCH_CHECK(out_tok.size(0) == total_L && out_tok.size(1) == D_out, "out_tok shape mismatch");

    if (past_states.has_value() && past_states->defined()) {
        check_t(*past_states, "past_states", 4, at::ScalarType::Float);
        TORCH_CHECK(past_states->size(0) == batch_size && past_states->size(1) == num_kv_heads &&
                    past_states->size(2) == d_v && past_states->size(3) == d_k,
                    "past_states shape mismatch");
    }
    if (grad_final_states.has_value() && grad_final_states->defined()) {
        check_t(*grad_final_states, "grad_final_states", 4, at::ScalarType::Float);
        TORCH_CHECK(grad_final_states->size(0) == batch_size && grad_final_states->size(1) == num_kv_heads &&
                    grad_final_states->size(2) == d_v && grad_final_states->size(3) == d_k,
                    "grad_final_states shape mismatch");
    }

    bool has_mlp = mlp_W_gate_up.has_value() && mlp_W_gate_up->defined();
    TORCH_CHECK(has_mlp == (norm1_weight.has_value() && norm1_weight->defined()) &&
                has_mlp == (norm2_weight.has_value() && norm2_weight->defined()) &&
                has_mlp == (mlp_W_down.has_value() && mlp_W_down->defined()),
                "norm1_weight, norm2_weight, mlp_W_gate_up, and mlp_W_down must all be provided or omitted together");

    int64_t hidden_dim = 0;
    if (has_mlp) {
        check_t(*norm1_weight,  "norm1_weight",  1, at::ScalarType::Half);
        check_t(*norm2_weight,  "norm2_weight",  1, at::ScalarType::Half);
        check_t(*mlp_W_gate_up, "mlp_W_gate_up", 2, at::ScalarType::Half);
        check_t(*mlp_W_down,    "mlp_W_down",    2, at::ScalarType::Half);
        check_t(rstd1,   "rstd1",   1, at::ScalarType::Float);
        check_t(h,       "h",       2, at::ScalarType::Half);
        check_t(x_n2,    "x_n2",    2, at::ScalarType::Half);
        check_t(rstd2,   "rstd2",   1, at::ScalarType::Float);
        check_t(gate_up, "gate_up", 2, at::ScalarType::Half);
        check_t(act,     "act",     2, at::ScalarType::Half);

        hidden_dim = mlp_W_gate_up->size(0) / 2;
        TORCH_CHECK(hidden_dim > 0 && hidden_dim % 8 == 0, "hidden_dim must be positive and divisible by 8");
        TORCH_CHECK(rstd1.size(0) == total_L && rstd2.size(0) == total_L, "rstd shape mismatch");
        TORCH_CHECK(h.size(0) == total_L && h.size(1) == d_model, "h shape mismatch");
        TORCH_CHECK(x_n2.size(0) == total_L && x_n2.size(1) == d_model, "x_n2 shape mismatch");
        TORCH_CHECK(gate_up.size(0) == total_L && gate_up.size(1) == 2 * hidden_dim, "gate_up shape mismatch");
        TORCH_CHECK(act.size(0) == total_L && act.size(1) == hidden_dim, "act shape mismatch");
    }

    auto h_opts = grad_y.options();
    auto f_opts = grad_y.options().dtype(at::kFloat);
    VarlenChunkMeta meta = compute_varlen_chunk_metadata(cu_seq_lens, chunk_size);

    // 2. MLP & Post-Mixer PreNorm 2 Backward
    at::Tensor d_h_total;
    at::Tensor grad_y_mix;
    at::Tensor d_norm2_weight;
    at::Tensor d_mlp_W_gate_up;
    at::Tensor d_mlp_W_down;

    if (has_mlp) {
        at::Tensor d_act = at::mm(grad_y, *mlp_W_down);
        d_mlp_W_down     = at::mm(grad_y.t(), act);

        at::Tensor d_gate_up = fused_swiglu_bwd(d_act, gate_up);
        d_act.reset();

        at::Tensor d_x_n2 = at::mm(d_gate_up, *mlp_W_gate_up);
        d_mlp_W_gate_up   = at::mm(d_gate_up.t(), x_n2);
        d_gate_up.reset();

        std::tie(d_h_total, d_norm2_weight) = fused_residual_rmsnorm_bwd(d_x_n2, grad_y, h, *norm2_weight, rstd2);
        d_x_n2.reset();
        grad_y_mix = d_h_total;
    } else {
        grad_y_mix      = grad_y;
        d_norm2_weight  = at::empty({0}, h_opts);
        d_mlp_W_gate_up = at::empty({0}, h_opts);
        d_mlp_W_down    = at::empty({0}, h_opts);
    }

    // 3. Output Projection Backward
    at::Tensor grad_out_tok = at::mm(grad_y_mix, W_out);
    at::Tensor dW_out       = at::mm(grad_y_mix.t(), out_tok);
    if (!has_mlp) {
        grad_y_mix.reset();
    }

    // 4. Core DeltaNet-2 Backward Pipeline
    at::Tensor d_proj = torch::empty({total_L, total_proj_dim}, h_opts);

    auto [dq_h, dk_ext, dG_ext, dR_out, dS_out, d_o_norm_weight, d_proj_after_oc] =
        output_combine_bwd_with_meta(
            grad_out_tok, O_raw, o_rstd, proj, o_norm_weight,
            q_h, k_h, G_log2_h, S_list, R_list, meta, d_proj,
            num_q_heads, num_kv_heads, d_k, d_v, chunk_size
        );
    grad_out_tok.reset();
    (void)d_proj_after_oc;

    auto [dS_next_list, dR_total, grad_past_states] =
        chunk_scan_bwd_with_meta(
            dS_out, dR_out, Y, k_h, G_log2_h, meta, batch_size,
            grad_final_states, has_past_states, chunk_size
        );
    at::Tensor grad_past_states_out = (has_past_states && grad_past_states.defined())
        ? grad_past_states
        : at::empty({0}, f_opts);

    auto [dk_h, dv_h, db_h, dw_h, dG_h] =
        chunk_wy_bwd_with_meta(
            dR_total, R_list, S_list, dS_next_list, A_list,
            k_h, v_h, b_eff_h, w_h, G_log2_h, dk_ext, dG_ext, meta, chunk_size
        );
    dS_next_list.reset();
    dR_total.reset();
    dk_ext.reset();
    dG_ext.reset();

    auto [d_proj_done, d_conv_weight, d_A_log, d_dt_bias, grad_past_conv_state] =
        proj_conv_prep_bwd_with_meta(
            dq_h, dk_h, dv_h, db_h, dw_h, dG_h,
            proj, conv_weight, A_log, dt_bias, past_conv_state,
            q_inv_norm, k_inv_norm, cu_seq_lens, meta, d_proj,
            num_q_heads, num_kv_heads, d_k, d_v, conv_size, chunk_size,
            scale, allow_neg_eigval, has_past_conv_state
        );
    dq_h.reset();
    dk_h.reset();
    dv_h.reset();
    db_h.reset();
    dw_h.reset();
    dG_h.reset();
    (void)d_proj_done;

    // 5. Input Projection & PreNorm 1 Backward
    at::Tensor d_x_in = at::mm(d_proj, W_proj);
    at::Tensor dW_proj = at::mm(d_proj.t(), x_in);
    d_proj.reset();

    at::Tensor grad_x;
    at::Tensor d_norm1_weight;
    if (has_mlp) {
        auto [d_x_norm1, d_n1_w] = fused_rmsnorm_bwd(d_x_in, x, *norm1_weight, rstd1);
        d_x_in.reset();
        d_norm1_weight = d_n1_w;
        grad_x = fused_residual_add_fwd(d_h_total, d_x_norm1);
        d_h_total.reset();
        d_x_norm1.reset();
    } else {
        grad_x         = d_x_in;
        d_norm1_weight = at::empty({0}, h_opts);
    }

    return {
        grad_x, grad_past_states_out, grad_past_conv_state,
        dW_proj, d_conv_weight, d_A_log, d_dt_bias, d_o_norm_weight, dW_out,
        d_norm1_weight, d_norm2_weight, d_mlp_W_gate_up, d_mlp_W_down
    };
}
