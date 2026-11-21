#pragma once
#include <torch/types.h>
#include <optional>
#include <tuple>
#include <cstdint>

// DeltaNet v2 operator declarations (CuTe Tensor / SM75+).
// Packed head-first layout (H, total_L, D) with GQA support (num_q_heads % num_kv_heads == 0).

// 1. Chunk WY Forward
// Outputs: (W: (H_kv, num_chunks, d_k, d_k) [FP16], Y: (H_kv, num_chunks, d_v, d_k) [FP16])
std::tuple<at::Tensor, at::Tensor>
chunk_wy_fwd(
    const at::Tensor& k,               // (num_kv_heads, total_L, d_k) [FP16] - Raw Key
    const at::Tensor& v,               // (num_kv_heads, total_L, d_v) [FP16] - Raw Value
    const at::Tensor& b,               // (num_kv_heads, total_L, d_k) [FP16] - Raw Beta gate
    const at::Tensor& w,               // (num_kv_heads, total_L, d_v) [FP16] - Raw W gate
    const at::Tensor& gamma,           // (num_kv_heads, total_L, 1) [FP16] - Raw Gamma
    const at::Tensor& a,               // (num_kv_heads, total_L, 1) [FP16] - Raw Alpha
    const at::Tensor& cu_seq_lens,     // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    int64_t chunk_size = 64            // scalar [int64] - Chunk size
);

// 2. Chunk WY Backward
// Outputs: (dk, dv, db, dw, dgamma, da)
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
    const at::Tensor& dk_ext,          // (num_kv_heads, total_L, d_k) [FP16] - External Key gradient
    const at::Tensor& dv_ext,          // (num_kv_heads, total_L, d_v) [FP16] - External Value gradient
    const at::Tensor& da_ext,          // (num_kv_heads, total_L, 1) [FP32] - External log_alpha gradient
    int64_t chunk_size = 64            // scalar [int64] - Chunk size
);

// 3. Chunk Scan Forward
// Outputs: (S: (H_kv, num_chunks, d_v, d_k) [FP16], final_states: (B, H_kv, d_v, d_k) [FP16])
std::tuple<at::Tensor, at::Tensor>
chunk_scan_fwd(
    const at::Tensor& W,                               // (num_kv_heads, num_chunks, d_k, d_k) [FP16] - Transition matrix
    const at::Tensor& Y,                               // (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Update matrix
    const std::optional<at::Tensor>& state_old,        // (batch_size, num_kv_heads, d_v, d_k) [FP16] - Initial state (optional)
    const at::Tensor& cu_seq_lens,                     // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    int64_t chunk_size = 64                            // scalar [int64] - Chunk size
);

// 4. Chunk Scan Backward
// Outputs: (grad_W: (H_kv, num_chunks, d_k, d_k), grad_Y: (H_kv, num_chunks, d_v, d_k), grad_state_old: (B, H_kv, d_v, d_k))
std::tuple<at::Tensor, at::Tensor, at::Tensor>
chunk_scan_bwd(
    const at::Tensor& grad_S_list,                     // (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Upstream state gradient
    const at::Tensor& W,                               // (num_kv_heads, num_chunks, d_k, d_k) [FP16] - Transition matrix
    const at::Tensor& S_list,                          // (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Forward states
    const at::Tensor& cu_seq_lens,                     // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    const std::optional<at::Tensor>& grad_final_states,// (batch_size, num_kv_heads, d_v, d_k) [FP16] - Final state gradient (optional)
    bool has_state_old = false,                        // scalar [bool] - Presence of initial state
    int64_t chunk_size = 64                            // scalar [int64] - Chunk size
);

// 5. Output Combine Forward
// Output: out (num_q_heads, total_L, d_v) [FP16]
at::Tensor
output_combine_fwd(
    const at::Tensor& q,                // (num_q_heads, total_L, d_k) [FP16] - Raw Query
    const at::Tensor& k,                // (num_kv_heads, total_L, d_k) [FP16] - Raw Key
    const at::Tensor& v,                // (num_kv_heads, total_L, d_v) [FP16] - Raw Value
    const at::Tensor& b,                // (num_kv_heads, total_L, d_k) [FP16] - Raw Beta
    const at::Tensor& w,                // (num_kv_heads, total_L, d_v) [FP16] - Raw W gate
    const at::Tensor& gamma,            // (num_kv_heads, total_L, 1) [FP16] - Raw Gamma
    const at::Tensor& a,                // (num_kv_heads, total_L, 1) [FP16] - Raw Alpha
    const at::Tensor& S,                // (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Recurrent states
    const at::Tensor& cu_seq_lens,      // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    int64_t num_q_heads,                // scalar [int64] - Number of query heads
    int64_t num_kv_heads,               // scalar [int64] - Number of KV heads
    int64_t d_k = 64,                   // scalar [int64] - Key head dimension
    int64_t d_v = 64,                   // scalar [int64] - Value head dimension
    int64_t chunk_size = 64             // scalar [int64] - Chunk size
);

// 6. Output Combine Backward
// Outputs: (dq, dk_ext, dv_ext, da_ext [FP32], dS)
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

// 7. DeltaNet Forward Coordinator
// Outputs: (y: (H_q, total_L, d_v), final_states: (B, H_kv, d_v, d_k), W, S)
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
    int64_t d_k = 64,                                      // scalar [int64] - Key dimension
    int64_t d_v = 64,                                      // scalar [int64] - Value dimension
    int64_t chunk_size = 64,                               // scalar [int64] - Chunk size
    bool forward_only = false                              // scalar [bool] - Omit W/S caches when true
);

// 8. DeltaNet Backward Coordinator
// Outputs: (dq, dk, dv, dw, db, da, dgamma, grad_past_states)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
deltanet_bwd(
    const at::Tensor& grad_output,                         // (num_q_heads, total_L, d_v) [FP16] - Upstream gradient
    const std::optional<at::Tensor>& grad_final_states,    // (batch_size, num_kv_heads, d_v, d_k) [FP16] - Final state gradient
    const std::optional<at::Tensor>& past_states,          // (batch_size, num_kv_heads, d_v, d_k) [FP16] - Initial states
    const at::Tensor& q,                                   // (num_q_heads, total_L, d_k) [FP16] - Raw Query
    const at::Tensor& k,                                   // (num_kv_heads, total_L, d_k) [FP16] - Raw Key
    const at::Tensor& v,                                   // (num_kv_heads, total_L, d_v) [FP16] - Raw Value
    const at::Tensor& w,                                   // (num_kv_heads, total_L, d_v) [FP16] - Raw W gate
    const at::Tensor& b,                                   // (num_kv_heads, total_L, d_k) [FP16] - Raw Beta
    const at::Tensor& a,                                   // (num_kv_heads, total_L, 1) [FP16] - Raw Alpha
    const at::Tensor& gamma,                               // (num_kv_heads, total_L, 1) [FP16] - Raw Gamma
    const at::Tensor& cu_seq_lens,                         // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    const std::optional<at::Tensor>& W = std::nullopt,     // (num_kv_heads, num_chunks, d_k, d_k) [FP16] - Cached W (optional)
    const std::optional<at::Tensor>& S = std::nullopt,     // (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Cached S (optional)
    int64_t num_q_heads = 1,                               // scalar [int64] - Number of Query heads
    int64_t num_kv_heads = 1,                              // scalar [int64] - Number of KV heads
    int64_t d_k = 64,                                      // scalar [int64] - Key dimension
    int64_t d_v = 64,                                      // scalar [int64] - Value dimension
    int64_t chunk_size = 64,                               // scalar [int64] - Chunk size
    bool has_past_states = false                           // scalar [bool] - Presence of initial states
);

// 9. DeltaNet Inference (Step L = 1)
// Outputs: (out: (B, num_q_heads, d_v) [FP16], past_state: (B, H_kv, d_v, d_k) [FP16] updated in-place)
std::tuple<at::Tensor, at::Tensor>
deltanet_infer(
    const at::Tensor& past_state,                          // (B, H, dv, dk) [FP16] - State updated in-place
    const at::Tensor& q,                                   // (B, num_q_heads, dk) [FP16] - Raw Query
    const at::Tensor& k,                                   // (B, H, dk) [FP16] - Raw Key
    const at::Tensor& v,                                   // (B, H, dv) [FP16] - Raw Value
    const at::Tensor& b,                                   // (B, H, dk) [FP16] - Raw Beta
    const at::Tensor& w,                                   // (B, H, dv) [FP16] - Raw W gate
    const at::Tensor& a,                                   // (B, H, 1) or (B, H) [FP16] - Raw Alpha
    const at::Tensor& gamma,                               // (B, H, 1) or (B, H) [FP16] - Raw Gamma
    int64_t num_q_heads,                                   // scalar [int64] - Number of Query heads
    int64_t num_kv_heads,                                  // scalar [int64] - Number of KV heads
    int64_t d_k = 64,                                      // scalar [int64] - Head dimension Key
    int64_t d_v = 64                                       // scalar [int64] - Head dimension Value
);
