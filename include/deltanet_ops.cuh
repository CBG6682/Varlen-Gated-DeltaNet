#pragma once
#include <torch/types.h>
#include <optional>
#include <tuple>
#include <cstdint>

// =========================================================================================
// DeltaNet v2 Operator Declarations (CuTe Tensor / SM75+ Turing/Ampere/Ada/Hopper)
// =========================================================================================
// Key Design Principles:
// - All kernels consume raw inputs: q, k, v, w, b, a, gamma, cu_seq_lens, and scalar params.
// - Activations are recomputed directly in registers/SMEM rather than storing intermediate tensors.
// - deltanet_fwd and deltanet_bwd serve as the primary host coordinators.
// - Submodule operators are exposed via TORCH_LIBRARY for debugging and standalone benchmarking.
// =========================================================================================

// -----------------------------------------------------------------------------------------
// 1. Chunk WY Forward
// -----------------------------------------------------------------------------------------
// Execution flow:
//   Stage 1: Load raw inputs (k, v, b, w, gamma, a) for 1 chunk, recompute activations
//            (log_alpha_cumsum, sqrt_gamma, k_norm, k_scaled, b_sig, w_sig, u, v_new).
//   Stage 2: Compute k_decay by applying intra-chunk decay factors.
//   Stage 3: Tensor Core GEMM to compute chunk transition matrix W = exp(c_last)*I - U^T @ K_decay.
//   Stage 4: Tensor Core GEMM to compute chunk update matrix Y = V_new^T @ K_decay.
// Outputs:
//   1. W: (num_kv_heads, num_chunks, d_k, d_k) [FP16] - Transition matrix
//   2. Y: (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Update matrix
// -----------------------------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor>
chunk_wy_fwd(
    const at::Tensor& k,               // (num_kv_heads, total_L, d_k) [FP16] - Raw Key
    const at::Tensor& v,               // (num_kv_heads, total_L, d_v) [FP16] - Raw Value
    const at::Tensor& b,               // (num_kv_heads, total_L, d_k) [FP16] - Raw Beta gate
    const at::Tensor& w,               // (num_kv_heads, total_L, d_v) [FP16] - Raw W gate
    const at::Tensor& gamma,           // (num_kv_heads, total_L, 1) [FP16] - Raw Gamma
    const at::Tensor& a,               // (num_kv_heads, total_L, 1) [FP16] - Raw Alpha
    const at::Tensor& cu_seq_lens,     // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    int64_t chunk_size = 64            // scalar [int64] - Chunk size (default 64)
);

// -----------------------------------------------------------------------------------------
// 2. Chunk WY Backward
// -----------------------------------------------------------------------------------------
// Execution flow:
//   Stage 1: Recompute activations (k_scaled, v_new, u, k_decay, log_alpha_cumsum) from raw inputs.
//   Stage 2: GEMM to compute du = -K_decay @ dW^T.
//   Stage 3: GEMM to compute dv_accum = K_decay @ dY^T and accumulate with dv_ext.
//   Stage 4: GEMM to compute dk_decay = V_new @ dY - U @ dW.
//   Stage 5: Recurrence epilogue computing dk_total and da_total, accumulating dk_ext and da_ext.
//   Stage 6: Analytical backward gradient through activations (L2-norm, sigmoid, softplus, cumsum).
// Outputs:
//   1. dk:     (num_kv_heads, total_L, d_k) [FP16] - Gradient w.r.t raw Key
//   2. dv:     (num_kv_heads, total_L, d_v) [FP16] - Gradient w.r.t raw Value
//   3. db:     (num_kv_heads, total_L, d_k) [FP16] - Gradient w.r.t raw Beta gate
//   4. dw:     (num_kv_heads, total_L, d_v) [FP16] - Gradient w.r.t raw W gate
//   5. dgamma: (num_kv_heads, total_L, 1)   [FP16] - Gradient w.r.t raw Gamma
//   6. da:     (num_kv_heads, total_L, 1)   [FP16] - Gradient w.r.t raw Alpha
// -----------------------------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
chunk_wy_bwd(
    const at::Tensor& dW,              // (num_kv_heads, num_chunks, d_k, d_k) [FP16] - Gradient w.r.t W
    const at::Tensor& dY,              // (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Gradient w.r.t Y
    const at::Tensor& k,               // (num_kv_heads, total_L, d_k) [FP16] - Raw Key
    const at::Tensor& v,               // (num_kv_heads, total_L, d_v) [FP16] - Raw Value
    const at::Tensor& b,               // (num_kv_heads, total_L, d_k) [FP16] - Raw Beta
    const at::Tensor& w,               // (num_kv_heads, total_L, d_v) [FP16] - Raw W gate
    const at::Tensor& gamma,           // (num_kv_heads, total_L, 1) [FP16] - Raw Gamma
    const at::Tensor& a,               // (num_kv_heads, total_L, 1) [FP16] - Raw Alpha
    const at::Tensor& cu_seq_lens,     // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    const at::Tensor& dk_ext,          // (num_kv_heads, total_L, d_k) [FP16] - External gradient w.r.t Key from output combine
    const at::Tensor& dv_ext,          // (num_kv_heads, total_L, d_v) [FP16] - External gradient w.r.t Value from output combine
    const at::Tensor& da_ext,          // (num_kv_heads, total_L, 1) [FP32] - External gradient w.r.t log alpha from output combine
    int64_t chunk_size = 64            // scalar [int64] - Chunk size
);

// -----------------------------------------------------------------------------------------
// 3. Chunk Scan Forward
// -----------------------------------------------------------------------------------------
// Execution flow:
//   Sequential scan of recurrent state S across chunks: S_{c+1} = S_c @ W_c + Y_c.
// Outputs:
//   1. S:            (num_kv_heads, num_chunks, d_v, d_k) [FP16] - State at each chunk
//   2. final_states: (batch_size, num_kv_heads, d_v, d_k) [FP16] - Final state at sequence boundary
// -----------------------------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor>
chunk_scan_fwd(
    const at::Tensor& W,                               // (num_kv_heads, num_chunks, d_k, d_k) [FP16] - Transition matrix W
    const at::Tensor& Y,                               // (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Update matrix Y
    const std::optional<at::Tensor>& state_old,        // (batch_size, num_kv_heads, d_v, d_k) [FP16] - Initial state (optional)
    const at::Tensor& cu_seq_lens,                     // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    int64_t chunk_size = 64                            // scalar [int64] - Chunk size
);

// -----------------------------------------------------------------------------------------
// 4. Chunk Scan Backward
// -----------------------------------------------------------------------------------------
// Execution flow:
//   Reverse scan from final chunk to first chunk (BPTT): compute grad_Y = dS, grad_W = S^T @ dS,
//   and propagate adjoint state dS_{prev} = dS @ W^T + grad_S_list.
// Outputs:
//   1. grad_W:         (num_kv_heads, num_chunks, d_k, d_k) [FP16] - Gradient w.r.t W
//   2. grad_Y:         (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Gradient w.r.t Y
//   3. grad_state_old: (batch_size, num_kv_heads, d_v, d_k) [FP16] - Gradient w.r.t state_old (if provided)
// -----------------------------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor, at::Tensor>
chunk_scan_bwd(
    const at::Tensor& grad_S_list,                     // (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Upstream gradient from output combine
    const at::Tensor& W,                               // (num_kv_heads, num_chunks, d_k, d_k) [FP16] - Transition matrix W
    const at::Tensor& S_list,                          // (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Forward recurrent states
    const at::Tensor& cu_seq_lens,                     // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    const std::optional<at::Tensor>& grad_final_states,// (batch_size, num_kv_heads, d_v, d_k) [FP16] - Gradient of final state (optional)
    bool has_state_old = false,                        // scalar [bool] - Flag indicating presence of state_old
    int64_t chunk_size = 64                            // scalar [int64] - Chunk size
);

// -----------------------------------------------------------------------------------------
// 5. Output Combine Forward
// -----------------------------------------------------------------------------------------
// Execution flow:
//   Stage 1: Prefix scan c = log_alpha_cumsum. Cache coordinates. Load K_scaled, V_new, S to SMEM.
//   Stage 2: Loop g = 0..num_groups-1:
//            - GEMM 1: Q_S = Q @ S^T -> epilogue O_inter = (Q @ S^T) * a_t.
//            - GEMM 2: R = Q @ K_scaled^T -> epilogue A = R * E, prefetch Q_next.
//            - GEMM 3: O_total = A @ V_new + O_inter -> Direct store to GMEM via half2.
// Outputs:
//   1. out: (num_q_heads, total_L, d_v) [FP16] - Combined attention output
// -----------------------------------------------------------------------------------------
at::Tensor
output_combine_fwd(
    const at::Tensor& q,                // (num_q_heads, total_L, d_k) [FP16] - Raw Query
    const at::Tensor& k,                // (num_kv_heads, total_L, d_k) [FP16] - Raw Key
    const at::Tensor& v,                // (num_kv_heads, total_L, d_v) [FP16] - Raw Value
    const at::Tensor& b,                // (num_kv_heads, total_L, d_k) [FP16] - Raw Beta
    const at::Tensor& w,                // (num_kv_heads, total_L, d_v) [FP16] - Raw W gate
    const at::Tensor& gamma,            // (num_kv_heads, total_L, 1) [FP16] - Raw Gamma
    const at::Tensor& a,                // (num_kv_heads, total_L, 1) [FP16] - Raw Alpha
    const at::Tensor& S,                // (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Recurrent state from chunk scan
    const at::Tensor& cu_seq_lens,      // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    int64_t num_q_heads,                // scalar [int64] - Number of query heads
    int64_t num_kv_heads,               // scalar [int64] - Number of KV heads (GQA support)
    int64_t d_k = 64,                   // scalar [int64] - Key head dimension (default 64)
    int64_t d_v = 64,                   // scalar [int64] - Value head dimension (default 64)
    int64_t chunk_size = 64             // scalar [int64] - Chunk size (default 64)
);

// -----------------------------------------------------------------------------------------
// 6. Output Combine Backward
// -----------------------------------------------------------------------------------------
// Execution flow:
//   Launches 3 sequential kernels on CUDA stream:
//     - Kernel 1 (dq_dk_dv): Grid (Hkv, num_chunks). Compute dq_intra, dk, dv, and dD^T.
//     - Kernel 2 (dc_inter_dS): Grid (Hkv, num_chunks). Compute dq_total = G @ S + dq_intra, dS, dc_inter.
//     - Kernel 3 (dc_total): Grid (Hkv, num_chunks), 128 threads. Accumulate da_total = dc_intra + dc_inter into da_ext.
// Outputs:
//   1. dq:     (num_q_heads, total_L, d_k)          [FP16] - Gradient w.r.t Query (dq_intra + dq_inter)
//   2. dk_ext: (num_kv_heads, total_L, d_k)         [FP16] - External gradient w.r.t Key
//   3. dv_ext: (num_kv_heads, total_L, d_v)         [FP16] - External gradient w.r.t Value
//   4. da_ext: (num_kv_heads, total_L, 1)           [FP32] - External gradient w.r.t log_alpha_cumsum
//   5. dS:     (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Gradient w.r.t recurrent state S
// -----------------------------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
output_combine_bwd(
    const at::Tensor& grad_output,      // (num_q_heads, total_L, d_v) [FP16] - Upstream gradient dO
    const at::Tensor& q,                // (num_q_heads, total_L, d_k) [FP16] - Raw Query
    const at::Tensor& k,                // (num_kv_heads, total_L, d_k) [FP16] - Raw Key
    const at::Tensor& v,                // (num_kv_heads, total_L, d_v) [FP16] - Raw Value
    const at::Tensor& b,                // (num_kv_heads, total_L, d_k) [FP16] - Raw Beta
    const at::Tensor& w,                // (num_kv_heads, total_L, d_v) [FP16] - Raw W gate
    const at::Tensor& gamma,            // (num_kv_heads, total_L, 1) [FP16] - Raw Gamma
    const at::Tensor& a,                // (num_kv_heads, total_L, 1) [FP16] - Raw Alpha
    const at::Tensor& S,                // (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Recurrent state S
    const at::Tensor& cu_seq_lens,      // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    int64_t num_q_heads,                // scalar [int64] - Number of query heads
    int64_t num_kv_heads,               // scalar [int64] - Number of KV heads
    int64_t d_k = 64,                   // scalar [int64] - Key dimension
    int64_t d_v = 64,                   // scalar [int64] - Value dimension
    int64_t chunk_size = 64             // scalar [int64] - Chunk size
);

// -----------------------------------------------------------------------------------------
// 7. Top-level Coordinator: DeltaNet Forward
// -----------------------------------------------------------------------------------------
// Execution flow:
//   Stage 1: Validate input tensors via TORCH_CHECK (CUDA, contiguous, dtype, shapes).
//   Stage 2: Invoke chunk_wy_fwd to compute W, Y from raw inputs.
//   Stage 3: Invoke chunk_scan_fwd to compute S, final_states from W, Y.
//   Stage 4: Invoke output_combine_fwd to compute output y from q, raw inputs, and state S.
//   Stage 5: Return (y, final_states, W, S). If forward_only, W and S are empty tensors.
// Outputs:
//   1. y:            (num_q_heads, total_L, d_v)          [FP16] - Combined attention output
//   2. final_states: (batch_size, num_kv_heads, d_v, d_k) [FP16] - Final hidden states
//   3. W:            (num_kv_heads, num_chunks, d_k, d_k) [FP16] - Transition matrix (empty if forward_only)
//   4. S:            (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Recurrent states (empty if forward_only)
// -----------------------------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
deltanet_fwd(
    const std::optional<at::Tensor>& past_states,          // (batch_size, num_kv_heads, d_v, d_k) [FP16] - Initial states
    const at::Tensor& q,                                   // (num_q_heads, total_L, d_k) [FP16] - Raw Query
    const at::Tensor& k,                                   // (num_kv_heads, total_L, d_k) [FP16] - Raw Key
    const at::Tensor& v,                                   // (num_kv_heads, total_L, d_v) [FP16] - Raw Value
    const at::Tensor& w,                                   // (num_kv_heads, total_L, d_v) [FP16] - Raw W gate
    const at::Tensor& b,                                   // (num_kv_heads, total_L, d_k) [FP16] - Raw Beta
    const at::Tensor& a,                                   // (num_kv_heads, total_L, 1) [FP16] - Raw Alpha
    const at::Tensor& gamma,                               // (num_kv_heads, total_L, 1) [FP16] - Raw Gamma
    const at::Tensor& cu_seq_lens,                         // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    int64_t num_q_heads,                                   // scalar [int64] - Number of Query heads
    int64_t num_kv_heads,                                  // scalar [int64] - Number of KV heads
    int64_t d_k = 64,                                      // scalar [int64] - Key dimension (default 64)
    int64_t d_v = 64,                                      // scalar [int64] - Value dimension (default 64)
    int64_t chunk_size = 64,                               // scalar [int64] - Chunk size (default 64)
    bool forward_only = false                              // scalar [bool] - Forward-only flag
);

// -----------------------------------------------------------------------------------------
// 8. Top-level Coordinator: DeltaNet Backward
// -----------------------------------------------------------------------------------------
// Execution flow:
//   Stage 1: Validate inputs via TORCH_CHECK.
//   Stage 2: Reuse W and S if provided from forward, or recompute if missing.
//   Stage 3: Invoke output_combine_bwd to compute dq, dk_ext, dv_ext, da_ext, dS.
//   Stage 4: Invoke chunk_scan_bwd to compute dW, dY, grad_past_states from dS.
//   Stage 5: Invoke chunk_wy_bwd to compute dk, dv, db, dw, dgamma, da from dW, dY and external gradients.
//   Stage 6: Return 8-tensor gradient tuple.
// Outputs:
//   (dq, dk, dv, dw, db, da, dgamma, grad_past_states)
// -----------------------------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
deltanet_bwd(
    const at::Tensor& grad_output,                         // (num_q_heads, total_L, d_v) [FP16] - Upstream gradient
    const std::optional<at::Tensor>& grad_final_states,    // (batch_size, num_kv_heads, d_v, d_k) [FP16] - Gradient w.r.t final states
    const std::optional<at::Tensor>& past_states,          // (batch_size, num_kv_heads, d_v, d_k) [FP16] - Initial states
    const at::Tensor& q,                                   // (num_q_heads, total_L, d_k) [FP16] - Raw Query
    const at::Tensor& k,                                   // (num_kv_heads, total_L, d_k) [FP16] - Raw Key
    const at::Tensor& v,                                   // (num_kv_heads, total_L, d_v) [FP16] - Raw Value
    const at::Tensor& w,                                   // (num_kv_heads, total_L, d_v) [FP16] - Raw W gate
    const at::Tensor& b,                                   // (num_kv_heads, total_L, d_k) [FP16] - Raw Beta
    const at::Tensor& a,                                   // (num_kv_heads, total_L, 1) [FP16] - Raw Alpha
    const at::Tensor& gamma,                               // (num_kv_heads, total_L, 1) [FP16] - Raw Gamma
    const at::Tensor& cu_seq_lens,                         // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    const std::optional<at::Tensor>& W = std::nullopt,     // (num_kv_heads, num_chunks, d_k, d_k) [FP16] - W from fwd (if cached)
    const std::optional<at::Tensor>& S = std::nullopt,     // (num_kv_heads, num_chunks, d_v, d_k) [FP16] - S from fwd (if cached)
    int64_t num_q_heads = 1,                               // scalar [int64] - Number of Query heads
    int64_t num_kv_heads = 1,                              // scalar [int64] - Number of KV heads
    int64_t d_k = 64,                                      // scalar [int64] - Key dimension (default 64)
    int64_t d_v = 64,                                      // scalar [int64] - Value dimension (default 64)
    int64_t chunk_size = 64,                               // scalar [int64] - Chunk size (default 64)
    bool has_past_states = false                           // scalar [bool] - Flag indicating presence of past states
);

// -----------------------------------------------------------------------------------------
// 9. Inference Coordinator: DeltaNet Inference (Step L = 1)
// -----------------------------------------------------------------------------------------
// Execution flow:
//   Stage 1: Validate inputs via TORCH_CHECK (CUDA, contiguous, dtype, alignment).
//   Stage 2: Launch kernel_deltanet_infer to update past_state in-place and compute out.
//   Stage 3: Return (out, past_state).
// Outputs:
//   1. out:        (batch_size, num_q_heads, d_v)          [FP16] - Attention output
//   2. past_state: (batch_size, num_kv_heads, d_v, d_k)    [FP16] - In-place updated hidden states
// -----------------------------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor>
deltanet_infer(
    const at::Tensor& past_state,                          // (B, H, dv, dk) [FP16] - In-place update
    const at::Tensor& q,                                   // (B, num_q_heads, dk) [FP16] - Raw Query
    const at::Tensor& k,                                   // (B, H, dk) [FP16] - Raw Key
    const at::Tensor& v,                                   // (B, H, dv) [FP16] - Raw Value
    const at::Tensor& b,                                   // (B, H, dk) [FP16] - Raw Beta
    const at::Tensor& w,                                   // (B, H, dv) [FP16] - Raw W gate
    const at::Tensor& a,                                   // (B, H, 1) or (B, H) [FP16] - Raw Alpha
    const at::Tensor& gamma,                               // (B, H, 1) or (B, H) [FP16] - Raw Gamma
    int64_t num_q_heads,                                   // scalar [int64] - Number of Query heads
    int64_t num_kv_heads,                                  // scalar [int64] - Number of KV heads
    int64_t d_k = 64,                                      // scalar [int64] - Head dimension Key (default 64)
    int64_t d_v = 64                                       // scalar [int64] - Head dimension Value (default 64)
);
