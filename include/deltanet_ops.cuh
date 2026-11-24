#pragma once
#include <torch/types.h>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

struct VarlenChunkMeta {
    at::Tensor chunk_offsets;      // (num_chunks + 1,) [Int32, CUDA]
    at::Tensor chunk_valid_lens;   // (num_chunks,) [Int32, CUDA]
    at::Tensor chunk_seq_ids;      // (num_chunks,) [Int32, CUDA]
    at::Tensor chunk_local_starts; // (num_chunks,) [Int32, CUDA]
    at::Tensor chunk_is_last;      // (num_chunks,) [Int32, CUDA]
    at::Tensor cu_chunk_lens;      // (batch_size + 1,) [Int32, CUDA]
    int64_t num_chunks;
};

// DeltaNet v2 operator declarations (CuTe Tensor / SM75+).
// Packed token-major DeltaBlock API + head-first core kernels with FP32 recurrent state.

// 1. Block Ops (src/block_ops.cu)

// Outputs: (out: (L, D) [FP16], rstd: (L,) [FP32] or empty when !is_training)
std::tuple<at::Tensor, at::Tensor>
fused_rmsnorm_fwd(
    const at::Tensor& x,           // (L, D) [FP16] - Input activations
    const at::Tensor& weight,      // (D,) [FP16] - RMSNorm scale weight
    double eps = 1e-5,             // scalar [double] - Stability epsilon
    bool is_training = true        // scalar [bool] - Save rstd when true
);

// Outputs: (grad_x: (L, D) [FP16], grad_weight: (D,) [FP16])
std::tuple<at::Tensor, at::Tensor>
fused_rmsnorm_bwd(
    const at::Tensor& grad_out,    // (L, D) [FP16] - Upstream gradient
    const at::Tensor& x,           // (L, D) [FP16] - Saved input activations
    const at::Tensor& weight,      // (D,) [FP16] - RMSNorm scale weight
    const at::Tensor& rstd         // (L,) [FP32] - Saved reciprocal std
);

// Outputs: (h: (L, D) [FP16], x_norm: (L, D) [FP16], rstd: (L,) [FP32] or empty when !is_training)
std::tuple<at::Tensor, at::Tensor, at::Tensor>
fused_residual_rmsnorm_fwd(
    const at::Tensor& x,           // (L, D) [FP16] - Residual branch input
    const at::Tensor& residual_in, // (L, D) [FP16] - Branch output to add
    const at::Tensor& weight,      // (D,) [FP16] - RMSNorm scale weight
    double eps = 1e-5,             // scalar [double] - Stability epsilon
    bool is_training = true        // scalar [bool] - Save rstd when true
);

// Outputs: (grad_h_total: (L, D) [FP16], grad_weight: (D,) [FP16])
std::tuple<at::Tensor, at::Tensor>
fused_residual_rmsnorm_bwd(
    const at::Tensor& grad_norm_out, // (L, D) [FP16] - Gradient from normalized branch
    const at::Tensor& grad_h_out,    // (L, D) [FP16] - Gradient from residual skip branch
    const at::Tensor& h,             // (L, D) [FP16] - Saved residual sum h
    const at::Tensor& weight,        // (D,) [FP16] - RMSNorm scale weight
    const at::Tensor& rstd           // (L,) [FP32] - Saved reciprocal std
);

// Output: act (L, H_mlp) [FP16]
at::Tensor
fused_swiglu_fwd(
    const at::Tensor& gate_up      // (L, 2 * H_mlp) [FP16] - Concatenated gate and up projections
);

// Output: grad_gate_up (L, 2 * H_mlp) [FP16]
at::Tensor
fused_swiglu_bwd(
    const at::Tensor& grad_act,    // (L, H_mlp) [FP16] - Upstream gradient
    const at::Tensor& gate_up      // (L, 2 * H_mlp) [FP16] - Saved gate_up tensor
);

// Output: y (L, D) [FP16]
at::Tensor
fused_residual_add_fwd(
    const at::Tensor& a,           // (L, D) [FP16] - First operand
    const at::Tensor& b            // (L, D) [FP16] - Second operand
);

// 2. Fused Projection Epilogue (src/proj_conv_prep.cu)

// Outputs: (q_h, k_h, v_h, b_eff_h, w_h, G_log2_h [FP32], final_conv_state, q_inv_norm, k_inv_norm)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
proj_conv_prep_fwd(
    const at::Tensor& proj,                          // (total_L, total_proj_dim) [FP16] - Unified projection
    const at::Tensor& conv_weight,                   // (conv_size, D_qkv) [FP16] - Depthwise conv weights
    const at::Tensor& A_log,                         // (num_kv_heads,) [FP32] - Log-decay rate
    const at::Tensor& dt_bias,                       // (num_kv_heads, d_k) [FP32] - Decay bias
    const std::optional<at::Tensor>& past_conv_state,// (B, conv_size - 1, D_qkv) [FP16] - Optional conv cache
    const at::Tensor& cu_seq_lens,                   // (B + 1,) [Int32 / Int64] - Cumulative sequence lengths
    int64_t num_q_heads,                             // scalar [int64] - Query heads
    int64_t num_kv_heads,                            // scalar [int64] - KV heads
    int64_t d_k = 64,                                // scalar [int64] - Key dimension
    int64_t d_v = 64,                                // scalar [int64] - Value dimension
    int64_t conv_size = 4,                           // scalar [int64] - Short conv kernel size
    int64_t chunk_size = 64,                         // scalar [int64] - Chunk size
    double scale = 0.125,                            // scalar [double] - Query scale (1/sqrt(d_k))
    bool allow_neg_eigval = false,                   // scalar [bool] - Scale erase gate by 2
    bool beta_guard = true,                          // scalar [bool] - Enable beta contrast guard
    bool is_training = true,                         // scalar [bool] - Save L2 inv norms when true
    bool output_conv_state = true                    // scalar [bool] - Output updated conv_state
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
proj_conv_prep_fwd_with_meta(
    const at::Tensor& proj,
    const at::Tensor& conv_weight,
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    const std::optional<at::Tensor>& past_conv_state,
    const at::Tensor& cu_seq_lens,
    const VarlenChunkMeta& meta,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t d_k,
    int64_t d_v,
    int64_t conv_size,
    int64_t chunk_size,
    double scale,
    bool allow_neg_eigval,
    bool beta_guard,
    bool is_training,
    bool output_conv_state
);

// Outputs: (d_proj_inout, d_conv_weight, d_A_log, d_dt_bias, d_past_conv_state)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
proj_conv_prep_bwd(
    const at::Tensor& dq_h,                          // (num_q_heads, total_L, d_k) [FP16] - Grad q_h
    const at::Tensor& dk_h,                          // (num_kv_heads, total_L, d_k) [FP16] - Grad k_h
    const at::Tensor& dv_h,                          // (num_kv_heads, total_L, d_v) [FP16] - Grad v_h
    const at::Tensor& db_h,                          // (num_kv_heads, total_L, d_k) [FP16] - Grad b_eff_h
    const at::Tensor& dw_h,                          // (num_kv_heads, total_L, d_v) [FP16] - Grad w_h
    const at::Tensor& dG_h,                          // (num_kv_heads, total_L, d_k) [FP32] - Grad G_ln_h (natural-log decay)
    const at::Tensor& proj,                          // (total_L, total_proj_dim) [FP16] - Saved projection
    const at::Tensor& conv_weight,                   // (conv_size, D_qkv) [FP16] - Conv weights
    const at::Tensor& A_log,                         // (num_kv_heads,) [FP32] - Log-decay rate
    const at::Tensor& dt_bias,                       // (num_kv_heads, d_k) [FP32] - Decay bias
    const std::optional<at::Tensor>& past_conv_state,// (B, conv_size - 1, D_qkv) [FP16] - Optional conv cache
    const at::Tensor& q_inv_norm,                    // (num_q_heads, total_L) [FP32] - Saved Q inv norm
    const at::Tensor& k_inv_norm,                    // (num_kv_heads, total_L) [FP32] - Saved K inv norm
    const at::Tensor& cu_seq_lens,                   // (B + 1,) [Int32 / Int64] - Cumulative sequence lengths
    at::Tensor& d_proj_inout,                        // (total_L, total_proj_dim) [FP16] - Grad projection (in-place)
    int64_t num_q_heads,                             // scalar [int64] - Query heads
    int64_t num_kv_heads,                            // scalar [int64] - KV heads
    int64_t d_k = 64,                                // scalar [int64] - Key dimension
    int64_t d_v = 64,                                // scalar [int64] - Value dimension
    int64_t conv_size = 4,                           // scalar [int64] - Short conv kernel size
    int64_t chunk_size = 64,                         // scalar [int64] - Chunk size
    double scale = 0.125,                            // scalar [double] - Query scale
    bool allow_neg_eigval = false,                   // scalar [bool] - Scale erase gate by 2
    bool has_past_conv_state = false                 // scalar [bool] - Emit d_past_conv_state when true
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
proj_conv_prep_bwd_with_meta(
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
    const VarlenChunkMeta& meta,
    at::Tensor& d_proj_inout,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t d_k,
    int64_t d_v,
    int64_t conv_size,
    int64_t chunk_size,
    double scale,
    bool allow_neg_eigval,
    bool has_past_conv_state
);

// 3. Chunk WY Forward (src/chunk_wy_fwd.cu)

// Outputs: (Y: (H_kv, num_chunks, 64, 64) [FP16], U: (H_kv, num_chunks, 64, 64) [FP16], A: (H_kv, num_chunks, 64, 64) [FP16] or empty)
std::tuple<at::Tensor, at::Tensor, at::Tensor>
chunk_wy_fwd(
    const at::Tensor& k_h,           // (num_kv_heads, total_L, d_k) [FP16] - Normalized Key
    const at::Tensor& v_h,           // (num_kv_heads, total_L, d_v) [FP16] - Activated Value
    const at::Tensor& b_eff_h,       // (num_kv_heads, total_L, d_k) [FP16] - Effective erase gate
    const at::Tensor& w_h,           // (num_kv_heads, total_L, d_v) [FP16] - Write gate
    const at::Tensor& G_log2_h,      // (num_kv_heads, total_L, d_k) [FP32] - Chunk cumsum log2 decay
    const at::Tensor& cu_seq_lens,   // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    int64_t chunk_size = 64,         // scalar [int64] - Chunk size
    bool is_training = true          // scalar [bool] - Retain A_list when true
);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
chunk_wy_fwd_with_meta(
    const at::Tensor& k_h,
    const at::Tensor& v_h,
    const at::Tensor& b_eff_h,
    const at::Tensor& w_h,
    const at::Tensor& G_log2_h,
    const VarlenChunkMeta& meta,
    int64_t chunk_size,
    bool is_training
);

// 4. Chunk Scan Forward & Backward (src/chunk_scan_fwd.cu, src/chunk_scan_bwd.cu)

// Outputs: (S_list: (H_kv, num_chunks, d_v, d_k) [FP32], R_list: (H_kv, num_chunks, 64, d_v) [FP16 in-place U_R], final_states: (B, H_kv, d_v, d_k) [FP32])
std::tuple<at::Tensor, at::Tensor, at::Tensor>
chunk_scan_fwd(
    const at::Tensor& Y,                        // (num_kv_heads, num_chunks, 64, d_k) [FP16] - Erase WY auxiliary
    at::Tensor& U_R,                            // (num_kv_heads, num_chunks, 64, d_v) [FP16] - In: U, Out in-place: R
    const at::Tensor& k_h,                      // (num_kv_heads, total_L, d_k) [FP16] - Normalized Key
    const at::Tensor& G_log2_h,                 // (num_kv_heads, total_L, d_k) [FP32] - Chunk cumsum log2 decay
    const std::optional<at::Tensor>& state_old, // (batch_size, num_kv_heads, d_v, d_k) [FP32] - Initial state (optional)
    const at::Tensor& cu_seq_lens,              // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    int64_t chunk_size = 64                     // scalar [int64] - Chunk size
);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
chunk_scan_fwd_with_meta(
    const at::Tensor& Y,
    at::Tensor& U_R,
    const at::Tensor& k_h,
    const at::Tensor& G_log2_h,
    const std::optional<at::Tensor>& state_old,
    const VarlenChunkMeta& meta,
    int64_t batch_size,
    int64_t chunk_size
);

// Outputs: (dS_next_list: (H_kv, num_chunks, d_v, d_k) [FP32 in-place], dR_total: (H_kv, num_chunks, 64, d_v) [FP16 in-place], grad_state_old: (B, H_kv, d_v, d_k) [FP32])
std::tuple<at::Tensor, at::Tensor, at::Tensor>
chunk_scan_bwd(
    at::Tensor& dS_inout,                               // (num_kv_heads, num_chunks, d_v, d_k) [FP32] - In: dS_out, Out: dS_next
    at::Tensor& dR_inout,                               // (num_kv_heads, num_chunks, 64, d_v) [FP16] - In: dR_out, Out: dR_total
    const at::Tensor& Y,                                // (num_kv_heads, num_chunks, 64, d_k) [FP16] - Erase WY auxiliary
    const at::Tensor& k_h,                              // (num_kv_heads, total_L, d_k) [FP16] - Normalized Key
    const at::Tensor& G_log2_h,                         // (num_kv_heads, total_L, d_k) [FP32] - Chunk cumsum log2 decay
    const at::Tensor& cu_seq_lens,                      // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    const std::optional<at::Tensor>& grad_final_states = std::nullopt, // (batch_size, num_kv_heads, d_v, d_k) [FP32]
    bool has_state_old = false,                         // scalar [bool] - Emit grad_state_old when true
    int64_t chunk_size = 64                             // scalar [int64] - Chunk size
);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
chunk_scan_bwd_with_meta(
    at::Tensor& dS_inout,
    at::Tensor& dR_inout,
    const at::Tensor& Y,
    const at::Tensor& k_h,
    const at::Tensor& G_log2_h,
    const VarlenChunkMeta& meta,
    int64_t batch_size,
    const std::optional<at::Tensor>& grad_final_states,
    bool has_state_old,
    int64_t chunk_size
);

// 5. Output Combine Forward & Backward (src/output_combine_fwd.cu, src/output_combine_bwd.cu)

// Outputs: (out_tok: (total_L, H_q * d_v) [FP16], O_raw: (H_q, total_L, d_v) [FP16] or empty, o_rstd: (H_q, total_L) [FP32] or empty)
std::tuple<at::Tensor, at::Tensor, at::Tensor>
output_combine_fwd(
    const at::Tensor& q_h,           // (num_q_heads, total_L, d_k) [FP16] - Scaled normalized Query
    const at::Tensor& k_h,           // (num_kv_heads, total_L, d_k) [FP16] - Normalized Key
    const at::Tensor& G_log2_h,      // (num_kv_heads, total_L, d_k) [FP32] - Chunk cumsum log2 decay
    const at::Tensor& S_list,        // (num_kv_heads, num_chunks, d_v, d_k) [FP32] - Chunk initial states
    const at::Tensor& R_list,        // (num_kv_heads, num_chunks, 64, d_v) [FP16] - Residual matrix R
    const at::Tensor& proj,          // (total_L, total_proj_dim) [FP16] - Unified projection (reads o_gate)
    const at::Tensor& o_norm_weight, // (d_v,) [FP16] - Per-head RMSNorm weight
    const at::Tensor& cu_seq_lens,   // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    int64_t num_q_heads,             // scalar [int64] - Query heads
    int64_t num_kv_heads,            // scalar [int64] - KV heads
    int64_t d_k = 64,                // scalar [int64] - Key dimension
    int64_t d_v = 64,                // scalar [int64] - Value dimension
    int64_t chunk_size = 64,         // scalar [int64] - Chunk size
    double eps = 1e-5,               // scalar [double] - RMSNorm epsilon
    bool is_training = true          // scalar [bool] - Save O_raw and o_rstd when true
);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
output_combine_fwd_with_meta(
    const at::Tensor& q_h,
    const at::Tensor& k_h,
    const at::Tensor& G_log2_h,
    const at::Tensor& S_list,
    const at::Tensor& R_list,
    const at::Tensor& proj,
    const at::Tensor& o_norm_weight,
    const VarlenChunkMeta& meta,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t d_k,
    int64_t d_v,
    int64_t chunk_size,
    double eps,
    bool is_training
);

// Outputs: (dq_h, dk_ext, dG_ext [FP32], dR_out, dS_out [FP32], d_o_norm_weight, d_proj_inout)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
output_combine_bwd(
    const at::Tensor& grad_out_tok,  // (total_L, num_q_heads * d_v) [FP16] - Upstream gradient
    const at::Tensor& O_raw,         // (num_q_heads, total_L, d_v) [FP16] - Saved pre-norm output
    const at::Tensor& o_rstd,        // (num_q_heads, total_L) [FP32] - Saved per-head rstd
    const at::Tensor& proj,          // (total_L, total_proj_dim) [FP16] - Unified projection
    const at::Tensor& o_norm_weight, // (d_v,) [FP16] - Per-head RMSNorm weight
    const at::Tensor& q_h,           // (num_q_heads, total_L, d_k) [FP16] - Scaled normalized Query
    const at::Tensor& k_h,           // (num_kv_heads, total_L, d_k) [FP16] - Normalized Key
    const at::Tensor& G_log2_h,      // (num_kv_heads, total_L, d_k) [FP32] - Chunk cumsum log2 decay
    const at::Tensor& S_list,        // (num_kv_heads, num_chunks, d_v, d_k) [FP32] - Chunk initial states
    const at::Tensor& R_list,        // (num_kv_heads, num_chunks, 64, d_v) [FP16] - Residual matrix R
    const at::Tensor& cu_seq_lens,   // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    at::Tensor& d_proj_inout,        // (total_L, total_proj_dim) [FP16] - Grad projection (writes o_gate slice)
    int64_t num_q_heads,             // scalar [int64] - Query heads
    int64_t num_kv_heads,            // scalar [int64] - KV heads
    int64_t d_k = 64,                // scalar [int64] - Key dimension
    int64_t d_v = 64,                // scalar [int64] - Value dimension
    int64_t chunk_size = 64          // scalar [int64] - Chunk size
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
output_combine_bwd_with_meta(
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
    const VarlenChunkMeta& meta,
    at::Tensor& d_proj_inout,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t d_k,
    int64_t d_v,
    int64_t chunk_size
);

// 6. Chunk WY Backward (src/chunk_wy_bwd.cu)

// Outputs: (dk_h, dv_h, db_h, dw_h, dG_h [FP32])
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
chunk_wy_bwd(
    const at::Tensor& dR_total,      // (num_kv_heads, num_chunks, 64, d_v) [FP16] - Total grad dR (= dU)
    const at::Tensor& R_list,        // (num_kv_heads, num_chunks, 64, d_v) [FP16] - Residual matrix R
    const at::Tensor& S_list,        // (num_kv_heads, num_chunks, d_v, d_k) [FP32] - Chunk initial states
    const at::Tensor& dS_next_list,  // (num_kv_heads, num_chunks, d_v, d_k) [FP32] - Grad w.r.t S_{c+1}
    const at::Tensor& A_list,        // (num_kv_heads, num_chunks, 64, 64) [FP16] - Saved WY inverse A
    const at::Tensor& k_h,           // (num_kv_heads, total_L, d_k) [FP16] - Normalized Key
    const at::Tensor& v_h,           // (num_kv_heads, total_L, d_v) [FP16] - Activated Value
    const at::Tensor& b_eff_h,       // (num_kv_heads, total_L, d_k) [FP16] - Effective erase gate
    const at::Tensor& w_h,           // (num_kv_heads, total_L, d_v) [FP16] - Write gate
    const at::Tensor& G_log2_h,      // (num_kv_heads, total_L, d_k) [FP32] - Chunk cumsum log2 decay
    const at::Tensor& dk_ext,        // (num_kv_heads, total_L, d_k) [FP16] - External Key gradient
    const at::Tensor& dG_ext,        // (num_kv_heads, total_L, d_k) [FP32] - External log-decay gradient
    const at::Tensor& cu_seq_lens,   // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    int64_t chunk_size = 64          // scalar [int64] - Chunk size
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
chunk_wy_bwd_with_meta(
    const at::Tensor& dR_total,
    const at::Tensor& R_list,
    const at::Tensor& S_list,
    const at::Tensor& dS_next_list,
    const at::Tensor& A_list,
    const at::Tensor& k_h,
    const at::Tensor& v_h,
    const at::Tensor& b_eff_h,
    const at::Tensor& w_h,
    const at::Tensor& G_log2_h,
    const at::Tensor& dk_ext,
    const at::Tensor& dG_ext,
    const VarlenChunkMeta& meta,
    int64_t chunk_size
);

// 7. Full DeltaBlock Coordinators (src/deltanet.cu)

// Outputs: [y, final_states, final_conv_state, ...saved_for_backward_if_is_training]
std::vector<at::Tensor>
deltablock_fwd(
    const at::Tensor& x,                                   // (total_L, d_model) [FP16] - Input activations
    const at::Tensor& cu_seq_lens,                         // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    const std::optional<at::Tensor>& past_states,          // (batch_size, num_kv_heads, d_v, d_k) [FP32] - Initial states
    const std::optional<at::Tensor>& past_conv_state,      // (batch_size, conv_size - 1, D_qkv) [FP16] - Initial conv cache
    const at::Tensor& W_proj,                              // (total_proj_dim, d_model) [FP16] - Unified input projection
    const at::Tensor& conv_weight,                         // (conv_size, D_qkv) [FP16] - Depthwise conv weights
    const at::Tensor& A_log,                               // (num_kv_heads,) [FP32] - Log-decay rate
    const at::Tensor& dt_bias,                             // (num_kv_heads, d_k) [FP32] - Decay bias
    const at::Tensor& o_norm_weight,                       // (d_v,) [FP16] - Per-head output RMSNorm weight
    const at::Tensor& W_out,                               // (d_model, num_q_heads * d_v) [FP16] - Output projection
    const std::optional<at::Tensor>& norm1_weight,         // (d_model,) [FP16] - PreNorm 1 weight (optional)
    const std::optional<at::Tensor>& norm2_weight,         // (d_model,) [FP16] - PreNorm 2 weight (optional)
    const std::optional<at::Tensor>& mlp_W_gate_up,        // (2 * hidden_dim, d_model) [FP16] - MLP gate+up (optional)
    const std::optional<at::Tensor>& mlp_W_down,           // (d_model, hidden_dim) [FP16] - MLP down (optional)
    int64_t num_q_heads,                                   // scalar [int64] - Query heads
    int64_t num_kv_heads,                                  // scalar [int64] - KV heads
    int64_t d_k = 64,                                      // scalar [int64] - Key dimension
    int64_t d_v = 64,                                      // scalar [int64] - Value dimension
    int64_t conv_size = 4,                                 // scalar [int64] - Short conv kernel size
    int64_t chunk_size = 64,                               // scalar [int64] - Chunk size
    double scale = 0.125,                                  // scalar [double] - Query scale
    double eps = 1e-5,                                     // scalar [double] - RMSNorm epsilon
    bool allow_neg_eigval = false,                         // scalar [bool] - Scale erase gate by 2
    bool beta_guard = true,                                // scalar [bool] - Enable beta contrast guard
    bool is_training = true,                               // scalar [bool] - Retain backward intermediates when true
    bool output_conv_state = true                          // scalar [bool] - Emit updated conv_state
);

// Outputs: [grad_x, grad_past_states, grad_past_conv_state, dW_proj, d_conv_weight, d_A_log, d_dt_bias, d_o_norm_weight, dW_out, d_norm1_weight, d_norm2_weight, d_mlp_W_gate_up, d_mlp_W_down]
std::vector<at::Tensor>
deltablock_bwd(
    const at::Tensor& grad_y,                              // (total_L, d_model) [FP16] - Upstream gradient
    const std::optional<at::Tensor>& grad_final_states,    // (batch_size, num_kv_heads, d_v, d_k) [FP32] - Final state grad
    const at::Tensor& x,                                   // (total_L, d_model) [FP16] - Saved block input
    const at::Tensor& cu_seq_lens,                         // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    const std::optional<at::Tensor>& past_states,          // (batch_size, num_kv_heads, d_v, d_k) [FP32] - Initial states
    const std::optional<at::Tensor>& past_conv_state,      // (batch_size, conv_size - 1, D_qkv) [FP16] - Initial conv cache
    const at::Tensor& W_proj,                              // (total_proj_dim, d_model) [FP16] - Unified input projection
    const at::Tensor& conv_weight,                         // (conv_size, D_qkv) [FP16] - Depthwise conv weights
    const at::Tensor& A_log,                               // (num_kv_heads,) [FP32] - Log-decay rate
    const at::Tensor& dt_bias,                             // (num_kv_heads, d_k) [FP32] - Decay bias
    const at::Tensor& o_norm_weight,                       // (d_v,) [FP16] - Per-head output RMSNorm weight
    const at::Tensor& W_out,                               // (d_model, num_q_heads * d_v) [FP16] - Output projection
    const std::optional<at::Tensor>& norm1_weight,         // (d_model,) [FP16] - PreNorm 1 weight (optional)
    const std::optional<at::Tensor>& norm2_weight,         // (d_model,) [FP16] - PreNorm 2 weight (optional)
    const std::optional<at::Tensor>& mlp_W_gate_up,        // (2 * hidden_dim, d_model) [FP16] - MLP gate+up (optional)
    const std::optional<at::Tensor>& mlp_W_down,           // (d_model, hidden_dim) [FP16] - MLP down (optional)
    const at::Tensor& x_in,                                // (total_L, d_model) [FP16] - Saved norm1 output
    const at::Tensor& rstd1,                               // (total_L,) [FP32] - Saved norm1 rstd (empty if no MLP)
    const at::Tensor& proj,                                // (total_L, total_proj_dim) [FP16] - Saved projection
    const at::Tensor& q_h,                                 // (num_q_heads, total_L, d_k) [FP16] - Saved q_h
    const at::Tensor& k_h,                                 // (num_kv_heads, total_L, d_k) [FP16] - Saved k_h
    const at::Tensor& v_h,                                 // (num_kv_heads, total_L, d_v) [FP16] - Saved v_h
    const at::Tensor& b_eff_h,                             // (num_kv_heads, total_L, d_k) [FP16] - Saved b_eff_h
    const at::Tensor& w_h,                                 // (num_kv_heads, total_L, d_v) [FP16] - Saved w_h
    const at::Tensor& G_log2_h,                            // (num_kv_heads, total_L, d_k) [FP32] - Saved G_log2_h
    const at::Tensor& q_inv_norm,                          // (num_q_heads, total_L) [FP32] - Saved Q inv norm
    const at::Tensor& k_inv_norm,                          // (num_kv_heads, total_L) [FP32] - Saved K inv norm
    const at::Tensor& A_list,                              // (num_kv_heads, num_chunks, 64, 64) [FP16] - Saved WY inverse
    const at::Tensor& Y,                                   // (num_kv_heads, num_chunks, 64, d_k) [FP16] - Saved WY Y
    const at::Tensor& S_list,                              // (num_kv_heads, num_chunks, d_v, d_k) [FP32] - Saved states
    const at::Tensor& R_list,                              // (num_kv_heads, num_chunks, 64, d_v) [FP16] - Saved R
    const at::Tensor& O_raw,                               // (num_q_heads, total_L, d_v) [FP16] - Saved O_raw
    const at::Tensor& o_rstd,                              // (num_q_heads, total_L) [FP32] - Saved o_rstd
    const at::Tensor& out_tok,                             // (total_L, num_q_heads * d_v) [FP16] - Saved out_tok
    const at::Tensor& h,                                   // (total_L, d_model) [FP16] - Saved residual sum (empty if no MLP)
    const at::Tensor& x_n2,                                // (total_L, d_model) [FP16] - Saved norm2 output (empty if no MLP)
    const at::Tensor& rstd2,                               // (total_L,) [FP32] - Saved norm2 rstd (empty if no MLP)
    const at::Tensor& gate_up,                             // (total_L, 2 * hidden_dim) [FP16] - Saved MLP gate_up (empty if no MLP)
    const at::Tensor& act,                                 // (total_L, hidden_dim) [FP16] - Saved MLP act (empty if no MLP)
    int64_t num_q_heads,                                   // scalar [int64] - Query heads
    int64_t num_kv_heads,                                  // scalar [int64] - KV heads
    int64_t d_k = 64,                                      // scalar [int64] - Key dimension
    int64_t d_v = 64,                                      // scalar [int64] - Value dimension
    int64_t conv_size = 4,                                 // scalar [int64] - Short conv kernel size
    int64_t chunk_size = 64,                               // scalar [int64] - Chunk size
    double scale = 0.125,                                  // scalar [double] - Query scale
    bool allow_neg_eigval = false,                         // scalar [bool] - Scale erase gate by 2
    bool has_past_states = false,                          // scalar [bool] - Emit grad_past_states when true
    bool has_past_conv_state = false                       // scalar [bool] - Emit grad_past_conv_state when true
);

// 8. Full DeltaBlock Inference Decode (Step L = 1) (src/deltanet_infer.cu)

// Outputs: (y: (B, d_model) [FP16], past_states: (B, H_kv, d_v, d_k) [FP32 in-place], conv_state: (B, conv_size - 1, D_qkv) [FP16 in-place])
std::tuple<at::Tensor, at::Tensor, at::Tensor>
deltablock_infer(
    const at::Tensor& x,                                   // (B, d_model) [FP16] - Decode step input
    const at::Tensor& past_states,                         // (B, num_kv_heads, d_v, d_k) [FP32] - Recurrent state (in-place)
    const at::Tensor& conv_state,                          // (B, conv_size - 1, D_qkv) [FP16] - Conv state (in-place)
    const at::Tensor& W_proj,                              // (total_proj_dim, d_model) [FP16] - Unified input projection
    const at::Tensor& conv_weight,                         // (conv_size, D_qkv) [FP16] - Depthwise conv weights
    const at::Tensor& A_log,                               // (num_kv_heads,) [FP32] - Log-decay rate
    const at::Tensor& dt_bias,                             // (num_kv_heads, d_k) [FP32] - Decay bias
    const at::Tensor& o_norm_weight,                       // (d_v,) [FP16] - Per-head output RMSNorm weight
    const at::Tensor& W_out,                               // (d_model, num_q_heads * d_v) [FP16] - Output projection
    const std::optional<at::Tensor>& norm1_weight,         // (d_model,) [FP16] - PreNorm 1 weight (optional)
    const std::optional<at::Tensor>& norm2_weight,         // (d_model,) [FP16] - PreNorm 2 weight (optional)
    const std::optional<at::Tensor>& mlp_W_gate_up,        // (2 * hidden_dim, d_model) [FP16] - MLP gate+up (optional)
    const std::optional<at::Tensor>& mlp_W_down,           // (d_model, hidden_dim) [FP16] - MLP down (optional)
    int64_t num_q_heads,                                   // scalar [int64] - Query heads
    int64_t num_kv_heads,                                  // scalar [int64] - KV heads
    int64_t d_k = 64,                                      // scalar [int64] - Key dimension
    int64_t d_v = 64,                                      // scalar [int64] - Value dimension
    int64_t conv_size = 4,                                 // scalar [int64] - Short conv kernel size
    double scale = 0.125,                                  // scalar [double] - Query scale
    double eps = 1e-5,                                     // scalar [double] - RMSNorm epsilon
    bool allow_neg_eigval = false,                         // scalar [bool] - Scale erase gate by 2
    bool beta_guard = true                                 // scalar [bool] - Enable beta contrast guard
);
