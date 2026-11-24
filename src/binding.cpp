#include <torch/extension.h>
#include <torch/library.h>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "deltanet_ops.cuh"

namespace {

// Thin wrappers taking const at::Tensor& for PyTorch dispatcher / pybind11 compatibility
// while still mutating the underlying TensorImpl buffers in-place.

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
proj_conv_prep_bwd_bind(
    const at::Tensor& dq_h,
    const at::Tensor& dk_h,
    const at::Tensor& dv_h,
    const at::Tensor& db_h,
    const at::Tensor& dw_h,
    const at::Tensor& dG_h,
    const at::Tensor& proj,
    const at::Tensor& conv_weight,
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    const std::optional<at::Tensor>& past_conv_state,
    const at::Tensor& q_inv_norm,
    const at::Tensor& k_inv_norm,
    const at::Tensor& cu_seq_lens,
    const at::Tensor& d_proj_inout,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t d_k,
    int64_t d_v,
    int64_t conv_size,
    int64_t chunk_size,
    double scale,
    bool allow_neg_eigval,
    bool has_past_conv_state
) {
    at::Tensor d_proj_mut = d_proj_inout;
    return proj_conv_prep_bwd(
        dq_h, dk_h, dv_h, db_h, dw_h, dG_h,
        proj, conv_weight, A_log, dt_bias, past_conv_state,
        q_inv_norm, k_inv_norm, cu_seq_lens, d_proj_mut,
        num_q_heads, num_kv_heads, d_k, d_v, conv_size, chunk_size,
        scale, allow_neg_eigval, has_past_conv_state
    );
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
chunk_scan_fwd_bind(
    const at::Tensor& Y,
    const at::Tensor& U_R,
    const at::Tensor& k_h,
    const at::Tensor& G_log2_h,
    const std::optional<at::Tensor>& state_old,
    const at::Tensor& cu_seq_lens,
    int64_t chunk_size
) {
    at::Tensor ur_mut = U_R;
    return chunk_scan_fwd(Y, ur_mut, k_h, G_log2_h, state_old, cu_seq_lens, chunk_size);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
chunk_scan_bwd_bind(
    const at::Tensor& dS_inout,
    const at::Tensor& dR_inout,
    const at::Tensor& Y,
    const at::Tensor& k_h,
    const at::Tensor& G_log2_h,
    const at::Tensor& cu_seq_lens,
    const std::optional<at::Tensor>& grad_final_states,
    bool has_state_old,
    int64_t chunk_size
) {
    at::Tensor dS_mut = dS_inout;
    at::Tensor dR_mut = dR_inout;
    return chunk_scan_bwd(
        dS_mut, dR_mut, Y, k_h, G_log2_h, cu_seq_lens, grad_final_states, has_state_old, chunk_size
    );
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
output_combine_bwd_bind(
    const at::Tensor& grad_out_tok,
    const at::Tensor& O_raw,
    const at::Tensor& o_rstd,
    const at::Tensor& proj,
    const at::Tensor& o_norm_weight,
    const at::Tensor& q_h,
    const at::Tensor& k_h,
    const at::Tensor& G_log2_h,
    const at::Tensor& S_list,
    const at::Tensor& R_list,
    const at::Tensor& cu_seq_lens,
    const at::Tensor& d_proj_inout,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t d_k,
    int64_t d_v,
    int64_t chunk_size
) {
    at::Tensor d_proj_mut = d_proj_inout;
    return output_combine_bwd(
        grad_out_tok, O_raw, o_rstd, proj, o_norm_weight,
        q_h, k_h, G_log2_h, S_list, R_list, cu_seq_lens, d_proj_mut,
        num_q_heads, num_kv_heads, d_k, d_v, chunk_size
    );
}

} // namespace

// 1. PyTorch Dispatcher Registration (TORCH_LIBRARY)

TORCH_LIBRARY(deltanet, m) {
    // Block Ops
    m.def("fused_rmsnorm_fwd(Tensor x, Tensor weight, float eps=1e-5, bool is_training=True) -> (Tensor, Tensor)");
    m.def("fused_rmsnorm_bwd(Tensor grad_out, Tensor x, Tensor weight, Tensor rstd) -> (Tensor, Tensor)");
    m.def("fused_residual_rmsnorm_fwd(Tensor x, Tensor residual_in, Tensor weight, float eps=1e-5, bool is_training=True) -> (Tensor, Tensor, Tensor)");
    m.def("fused_residual_rmsnorm_bwd(Tensor grad_norm_out, Tensor grad_h_out, Tensor h, Tensor weight, Tensor rstd) -> (Tensor, Tensor)");
    m.def("fused_swiglu_fwd(Tensor gate_up) -> Tensor");
    m.def("fused_swiglu_bwd(Tensor grad_act, Tensor gate_up) -> Tensor");
    m.def("fused_residual_add_fwd(Tensor a, Tensor b) -> Tensor");

    // Projection Epilogue
    m.def("proj_conv_prep_fwd(Tensor proj, Tensor conv_weight, Tensor A_log, Tensor dt_bias, Tensor? past_conv_state, Tensor cu_seq_lens, int num_q_heads, int num_kv_heads, int d_k=64, int d_v=64, int conv_size=4, int chunk_size=64, float scale=0.125, bool allow_neg_eigval=False, bool beta_guard=True, bool is_training=True, bool output_conv_state=True) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("proj_conv_prep_bwd(Tensor dq_h, Tensor dk_h, Tensor dv_h, Tensor db_h, Tensor dw_h, Tensor dG_h, Tensor proj, Tensor conv_weight, Tensor A_log, Tensor dt_bias, Tensor? past_conv_state, Tensor q_inv_norm, Tensor k_inv_norm, Tensor cu_seq_lens, Tensor d_proj_inout, int num_q_heads, int num_kv_heads, int d_k=64, int d_v=64, int conv_size=4, int chunk_size=64, float scale=0.125, bool allow_neg_eigval=False, bool has_past_conv_state=False) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

    // Core DeltaNet-2 Kernels
    m.def("chunk_wy_fwd(Tensor k_h, Tensor v_h, Tensor b_eff_h, Tensor w_h, Tensor G_log2_h, Tensor cu_seq_lens, int chunk_size=64, bool is_training=True) -> (Tensor, Tensor, Tensor)");
    m.def("chunk_wy_bwd(Tensor dR_total, Tensor R_list, Tensor S_list, Tensor dS_next_list, Tensor A_list, Tensor k_h, Tensor v_h, Tensor b_eff_h, Tensor w_h, Tensor G_log2_h, Tensor dk_ext, Tensor dG_ext, Tensor cu_seq_lens, int chunk_size=64) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("chunk_scan_fwd(Tensor Y, Tensor U_R, Tensor k_h, Tensor G_log2_h, Tensor? state_old, Tensor cu_seq_lens, int chunk_size=64) -> (Tensor, Tensor, Tensor)");
    m.def("chunk_scan_bwd(Tensor dS_inout, Tensor dR_inout, Tensor Y, Tensor k_h, Tensor G_log2_h, Tensor cu_seq_lens, Tensor? grad_final_states=None, bool has_state_old=False, int chunk_size=64) -> (Tensor, Tensor, Tensor)");
    m.def("output_combine_fwd(Tensor q_h, Tensor k_h, Tensor G_log2_h, Tensor S_list, Tensor R_list, Tensor proj, Tensor o_norm_weight, Tensor cu_seq_lens, int num_q_heads, int num_kv_heads, int d_k=64, int d_v=64, int chunk_size=64, float eps=1e-5, bool is_training=True) -> (Tensor, Tensor, Tensor)");
    m.def("output_combine_bwd(Tensor grad_out_tok, Tensor O_raw, Tensor o_rstd, Tensor proj, Tensor o_norm_weight, Tensor q_h, Tensor k_h, Tensor G_log2_h, Tensor S_list, Tensor R_list, Tensor cu_seq_lens, Tensor d_proj_inout, int num_q_heads, int num_kv_heads, int d_k=64, int d_v=64, int chunk_size=64) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

    // Full DeltaBlock Coordinators & Decode
    m.def("deltablock_fwd(Tensor x, Tensor cu_seq_lens, Tensor? past_states, Tensor? past_conv_state, Tensor W_proj, Tensor conv_weight, Tensor A_log, Tensor dt_bias, Tensor o_norm_weight, Tensor W_out, Tensor? norm1_weight, Tensor? norm2_weight, Tensor? mlp_W_gate_up, Tensor? mlp_W_down, int num_q_heads, int num_kv_heads, int d_k=64, int d_v=64, int conv_size=4, int chunk_size=64, float scale=0.125, float eps=1e-5, bool allow_neg_eigval=False, bool beta_guard=True, bool is_training=True, bool output_conv_state=True) -> Tensor[]");
    m.def("deltablock_bwd(Tensor grad_y, Tensor? grad_final_states, Tensor x, Tensor cu_seq_lens, Tensor? past_states, Tensor? past_conv_state, Tensor W_proj, Tensor conv_weight, Tensor A_log, Tensor dt_bias, Tensor o_norm_weight, Tensor W_out, Tensor? norm1_weight, Tensor? norm2_weight, Tensor? mlp_W_gate_up, Tensor? mlp_W_down, Tensor x_in, Tensor rstd1, Tensor proj, Tensor q_h, Tensor k_h, Tensor v_h, Tensor b_eff_h, Tensor w_h, Tensor G_log2_h, Tensor q_inv_norm, Tensor k_inv_norm, Tensor A_list, Tensor Y, Tensor S_list, Tensor R_list, Tensor O_raw, Tensor o_rstd, Tensor out_tok, Tensor h, Tensor x_n2, Tensor rstd2, Tensor gate_up, Tensor act, int num_q_heads, int num_kv_heads, int d_k=64, int d_v=64, int conv_size=4, int chunk_size=64, float scale=0.125, bool allow_neg_eigval=False, bool has_past_states=False, bool has_past_conv_state=False) -> Tensor[]");
    m.def("deltablock_infer(Tensor x, Tensor past_states, Tensor conv_state, Tensor W_proj, Tensor conv_weight, Tensor A_log, Tensor dt_bias, Tensor o_norm_weight, Tensor W_out, Tensor? norm1_weight, Tensor? norm2_weight, Tensor? mlp_W_gate_up, Tensor? mlp_W_down, int num_q_heads, int num_kv_heads, int d_k=64, int d_v=64, int conv_size=4, float scale=0.125, float eps=1e-5, bool allow_neg_eigval=False, bool beta_guard=True) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(deltanet, CUDA, m) {
    m.impl("fused_rmsnorm_fwd",          &fused_rmsnorm_fwd);
    m.impl("fused_rmsnorm_bwd",          &fused_rmsnorm_bwd);
    m.impl("fused_residual_rmsnorm_fwd", &fused_residual_rmsnorm_fwd);
    m.impl("fused_residual_rmsnorm_bwd", &fused_residual_rmsnorm_bwd);
    m.impl("fused_swiglu_fwd",           &fused_swiglu_fwd);
    m.impl("fused_swiglu_bwd",           &fused_swiglu_bwd);
    m.impl("fused_residual_add_fwd",     &fused_residual_add_fwd);

    m.impl("proj_conv_prep_fwd",         &proj_conv_prep_fwd);
    m.impl("proj_conv_prep_bwd",         &proj_conv_prep_bwd_bind);

    m.impl("chunk_wy_fwd",               &chunk_wy_fwd);
    m.impl("chunk_wy_bwd",               &chunk_wy_bwd);
    m.impl("chunk_scan_fwd",             &chunk_scan_fwd_bind);
    m.impl("chunk_scan_bwd",             &chunk_scan_bwd_bind);
    m.impl("output_combine_fwd",         &output_combine_fwd);
    m.impl("output_combine_bwd",         &output_combine_bwd_bind);

    m.impl("deltablock_fwd",             &deltablock_fwd);
    m.impl("deltablock_bwd",             &deltablock_bwd);
    m.impl("deltablock_infer",           &deltablock_infer);
}

// 2. Python C++ Extension Module (PYBIND11_MODULE)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Gated DeltaNet-2 CUDA Extension (Turing SM75+ / CuTe)";

    // Block Ops
    m.def("fused_rmsnorm_fwd", &fused_rmsnorm_fwd,
          "Fused RMSNorm Forward -> (out, rstd)",
          py::arg("x"), py::arg("weight"), py::arg("eps") = 1e-5, py::arg("is_training") = true);

    m.def("fused_rmsnorm_bwd", &fused_rmsnorm_bwd,
          "Fused RMSNorm Backward -> (grad_x, grad_weight)",
          py::arg("grad_out"), py::arg("x"), py::arg("weight"), py::arg("rstd"));

    m.def("fused_residual_rmsnorm_fwd", &fused_residual_rmsnorm_fwd,
          "Fused Residual Add + RMSNorm Forward -> (h, x_norm, rstd)",
          py::arg("x"), py::arg("residual_in"), py::arg("weight"),
          py::arg("eps") = 1e-5, py::arg("is_training") = true);

    m.def("fused_residual_rmsnorm_bwd", &fused_residual_rmsnorm_bwd,
          "Fused Residual Add + RMSNorm Backward -> (grad_h_total, grad_weight)",
          py::arg("grad_norm_out"), py::arg("grad_h_out"), py::arg("h"), py::arg("weight"), py::arg("rstd"));

    m.def("fused_swiglu_fwd", &fused_swiglu_fwd,
          "Fused SwiGLU Forward -> act",
          py::arg("gate_up"));

    m.def("fused_swiglu_bwd", &fused_swiglu_bwd,
          "Fused SwiGLU Backward -> grad_gate_up",
          py::arg("grad_act"), py::arg("gate_up"));

    m.def("fused_residual_add_fwd", &fused_residual_add_fwd,
          "Fused Residual Add Forward -> y",
          py::arg("a"), py::arg("b"));

    // Projection Epilogue
    m.def("proj_conv_prep_fwd", &proj_conv_prep_fwd,
          "Fused ShortConv1D + SiLU + L2Norm + Decay + BetaGuard Forward",
          py::arg("proj"), py::arg("conv_weight"), py::arg("A_log"), py::arg("dt_bias"),
          py::arg("past_conv_state"), py::arg("cu_seq_lens"),
          py::arg("num_q_heads"), py::arg("num_kv_heads"),
          py::arg("d_k") = 64, py::arg("d_v") = 64, py::arg("conv_size") = 4, py::arg("chunk_size") = 64,
          py::arg("scale") = 0.125, py::arg("allow_neg_eigval") = false, py::arg("beta_guard") = true,
          py::arg("is_training") = true, py::arg("output_conv_state") = true);

    m.def("proj_conv_prep_bwd", &proj_conv_prep_bwd_bind,
          "Fused ShortConv1D + SiLU + L2Norm + Decay Backward",
          py::arg("dq_h"), py::arg("dk_h"), py::arg("dv_h"), py::arg("db_h"), py::arg("dw_h"), py::arg("dG_h"),
          py::arg("proj"), py::arg("conv_weight"), py::arg("A_log"), py::arg("dt_bias"),
          py::arg("past_conv_state"), py::arg("q_inv_norm"), py::arg("k_inv_norm"),
          py::arg("cu_seq_lens"), py::arg("d_proj_inout"),
          py::arg("num_q_heads"), py::arg("num_kv_heads"),
          py::arg("d_k") = 64, py::arg("d_v") = 64, py::arg("conv_size") = 4, py::arg("chunk_size") = 64,
          py::arg("scale") = 0.125, py::arg("allow_neg_eigval") = false, py::arg("has_past_conv_state") = false);

    // Core DeltaNet-2 Kernels
    m.def("chunk_wy_fwd", &chunk_wy_fwd,
          "Chunk WY Forward -> (Y, U, A)",
          py::arg("k_h"), py::arg("v_h"), py::arg("b_eff_h"), py::arg("w_h"), py::arg("G_log2_h"),
          py::arg("cu_seq_lens"), py::arg("chunk_size") = 64, py::arg("is_training") = true);

    m.def("chunk_wy_bwd", &chunk_wy_bwd,
          "Chunk WY Backward -> (dk_h, dv_h, db_h, dw_h, dG_h)",
          py::arg("dR_total"), py::arg("R_list"), py::arg("S_list"), py::arg("dS_next_list"), py::arg("A_list"),
          py::arg("k_h"), py::arg("v_h"), py::arg("b_eff_h"), py::arg("w_h"), py::arg("G_log2_h"),
          py::arg("dk_ext"), py::arg("dG_ext"), py::arg("cu_seq_lens"), py::arg("chunk_size") = 64);

    m.def("chunk_scan_fwd", &chunk_scan_fwd_bind,
          "Chunk Scan Forward -> (S_list, R_list, final_states)",
          py::arg("Y"), py::arg("U_R"), py::arg("k_h"), py::arg("G_log2_h"),
          py::arg("state_old"), py::arg("cu_seq_lens"), py::arg("chunk_size") = 64);

    m.def("chunk_scan_bwd", &chunk_scan_bwd_bind,
          "Chunk Scan Backward -> (dS_next_list, dR_total, grad_state_old)",
          py::arg("dS_inout"), py::arg("dR_inout"), py::arg("Y"), py::arg("k_h"), py::arg("G_log2_h"),
          py::arg("cu_seq_lens"), py::arg("grad_final_states") = py::none(),
          py::arg("has_state_old") = false, py::arg("chunk_size") = 64);

    m.def("output_combine_fwd", &output_combine_fwd,
          "Output Combine + Per-Head RMSNorm * SiLU(o_gate) Forward -> (out_tok, O_raw, o_rstd)",
          py::arg("q_h"), py::arg("k_h"), py::arg("G_log2_h"), py::arg("S_list"), py::arg("R_list"),
          py::arg("proj"), py::arg("o_norm_weight"), py::arg("cu_seq_lens"),
          py::arg("num_q_heads"), py::arg("num_kv_heads"),
          py::arg("d_k") = 64, py::arg("d_v") = 64, py::arg("chunk_size") = 64,
          py::arg("eps") = 1e-5, py::arg("is_training") = true);

    m.def("output_combine_bwd", &output_combine_bwd_bind,
          "Output Combine + Per-Head RMSNorm * SiLU(o_gate) Backward -> (dq_h, dk_ext, dG_ext, dR_out, dS_out, d_o_norm_weight, d_proj_inout)",
          py::arg("grad_out_tok"), py::arg("O_raw"), py::arg("o_rstd"), py::arg("proj"), py::arg("o_norm_weight"),
          py::arg("q_h"), py::arg("k_h"), py::arg("G_log2_h"), py::arg("S_list"), py::arg("R_list"),
          py::arg("cu_seq_lens"), py::arg("d_proj_inout"),
          py::arg("num_q_heads"), py::arg("num_kv_heads"),
          py::arg("d_k") = 64, py::arg("d_v") = 64, py::arg("chunk_size") = 64);

    // Full DeltaBlock Coordinators & Decode
    m.def("deltablock_fwd", &deltablock_fwd,
          "Full DeltaBlock Forward -> [y, final_states, final_conv_state, ...saved_if_is_training]",
          py::arg("x"), py::arg("cu_seq_lens"), py::arg("past_states"), py::arg("past_conv_state"),
          py::arg("W_proj"), py::arg("conv_weight"), py::arg("A_log"), py::arg("dt_bias"),
          py::arg("o_norm_weight"), py::arg("W_out"),
          py::arg("norm1_weight"), py::arg("norm2_weight"), py::arg("mlp_W_gate_up"), py::arg("mlp_W_down"),
          py::arg("num_q_heads"), py::arg("num_kv_heads"),
          py::arg("d_k") = 64, py::arg("d_v") = 64, py::arg("conv_size") = 4, py::arg("chunk_size") = 64,
          py::arg("scale") = 0.125, py::arg("eps") = 1e-5,
          py::arg("allow_neg_eigval") = false, py::arg("beta_guard") = true,
          py::arg("is_training") = true, py::arg("output_conv_state") = true);

    m.def("deltablock_bwd", &deltablock_bwd,
          "Full DeltaBlock Backward -> [grad_x, grad_past_states, grad_past_conv_state, dW_proj, d_conv_weight, d_A_log, d_dt_bias, d_o_norm_weight, dW_out, d_norm1_weight, d_norm2_weight, d_mlp_W_gate_up, d_mlp_W_down]",
          py::arg("grad_y"), py::arg("grad_final_states"), py::arg("x"), py::arg("cu_seq_lens"),
          py::arg("past_states"), py::arg("past_conv_state"),
          py::arg("W_proj"), py::arg("conv_weight"), py::arg("A_log"), py::arg("dt_bias"),
          py::arg("o_norm_weight"), py::arg("W_out"),
          py::arg("norm1_weight"), py::arg("norm2_weight"), py::arg("mlp_W_gate_up"), py::arg("mlp_W_down"),
          py::arg("x_in"), py::arg("rstd1"), py::arg("proj"),
          py::arg("q_h"), py::arg("k_h"), py::arg("v_h"), py::arg("b_eff_h"), py::arg("w_h"), py::arg("G_log2_h"),
          py::arg("q_inv_norm"), py::arg("k_inv_norm"),
          py::arg("A_list"), py::arg("Y"), py::arg("S_list"), py::arg("R_list"),
          py::arg("O_raw"), py::arg("o_rstd"), py::arg("out_tok"),
          py::arg("h"), py::arg("x_n2"), py::arg("rstd2"), py::arg("gate_up"), py::arg("act"),
          py::arg("num_q_heads"), py::arg("num_kv_heads"),
          py::arg("d_k") = 64, py::arg("d_v") = 64, py::arg("conv_size") = 4, py::arg("chunk_size") = 64,
          py::arg("scale") = 0.125, py::arg("allow_neg_eigval") = false,
          py::arg("has_past_states") = false, py::arg("has_past_conv_state") = false);

    m.def("deltablock_infer", &deltablock_infer,
          "Full DeltaBlock Decode Step (L = 1) -> (y, past_states, conv_state)",
          py::arg("x"), py::arg("past_states"), py::arg("conv_state"),
          py::arg("W_proj"), py::arg("conv_weight"), py::arg("A_log"), py::arg("dt_bias"),
          py::arg("o_norm_weight"), py::arg("W_out"),
          py::arg("norm1_weight"), py::arg("norm2_weight"), py::arg("mlp_W_gate_up"), py::arg("mlp_W_down"),
          py::arg("num_q_heads"), py::arg("num_kv_heads"),
          py::arg("d_k") = 64, py::arg("d_v") = 64, py::arg("conv_size") = 4,
          py::arg("scale") = 0.125, py::arg("eps") = 1e-5,
          py::arg("allow_neg_eigval") = false, py::arg("beta_guard") = true);
}
