#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <optional>
#include <tuple>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "deltanet_common.cuh"
#include "deltanet_ops.cuh"

// HOST COORDINATION FUNCTIONS (deltanet_fwd & deltanet_bwd)

// Host coordinator for deltanet_fwd. Outputs: y (H_q, total_L, d_v), final_states (B, H_kv, d_v, d_k), W, S.
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
    int64_t d_k,                                           // scalar [int64] - Key dimension (default 64)
    int64_t d_v,                                           // scalar [int64] - Value dimension (default 64)
    int64_t chunk_size,                                    // scalar [int64] - Chunk size (default 64)
    bool forward_only                                      // scalar [bool] - Forward-only flag
) {
    // Stage 1 - Data Validation
    TORCH_CHECK(q.is_cuda(), "Input tensor 'q' must be a CUDA tensor");
    const c10::cuda::CUDAGuard device_guard(q.device());

    TORCH_CHECK(k.is_cuda(), "Input tensor 'k' must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "Input tensor 'v' must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "Input tensor 'w' must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "Input tensor 'b' must be a CUDA tensor");
    TORCH_CHECK(a.is_cuda(), "Input tensor 'a' must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "Input tensor 'gamma' must be a CUDA tensor");
    TORCH_CHECK(cu_seq_lens.is_cuda(), "Tensor 'cu_seq_lens' must be a CUDA tensor");

    TORCH_CHECK(k.device() == q.device(), "Tensor 'k' must be on same device as 'q'");
    TORCH_CHECK(v.device() == q.device(), "Tensor 'v' must be on same device as 'q'");
    TORCH_CHECK(w.device() == q.device(), "Tensor 'w' must be on same device as 'q'");
    TORCH_CHECK(b.device() == q.device(), "Tensor 'b' must be on same device as 'q'");
    TORCH_CHECK(a.device() == q.device(), "Tensor 'a' must be on same device as 'q'");
    TORCH_CHECK(gamma.device() == q.device(), "Tensor 'gamma' must be on same device as 'q'");
    TORCH_CHECK(cu_seq_lens.device() == q.device(), "Tensor 'cu_seq_lens' must be on same device as 'q'");

    TORCH_CHECK(q.scalar_type() == at::ScalarType::Half, "Tensor 'q' must be FP16 (torch.float16)");
    TORCH_CHECK(k.scalar_type() == at::ScalarType::Half, "Tensor 'k' must be FP16");
    TORCH_CHECK(v.scalar_type() == at::ScalarType::Half, "Tensor 'v' must be FP16");
    TORCH_CHECK(w.scalar_type() == at::ScalarType::Half, "Tensor 'w' must be FP16");
    TORCH_CHECK(b.scalar_type() == at::ScalarType::Half, "Tensor 'b' must be FP16");
    TORCH_CHECK(a.scalar_type() == at::ScalarType::Half, "Tensor 'a' must be FP16");
    TORCH_CHECK(gamma.scalar_type() == at::ScalarType::Half, "Tensor 'gamma' must be FP16");

    TORCH_CHECK(cu_seq_lens.scalar_type() == at::ScalarType::Int || cu_seq_lens.scalar_type() == at::ScalarType::Long,
                "Tensor 'cu_seq_lens' must be Int32 or Int64");

    TORCH_CHECK(q.is_contiguous(), "Tensor 'q' must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "Tensor 'k' must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "Tensor 'v' must be contiguous");
    TORCH_CHECK(w.is_contiguous(), "Tensor 'w' must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Tensor 'b' must be contiguous");
    TORCH_CHECK(a.is_contiguous(), "Tensor 'a' must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "Tensor 'gamma' must be contiguous");
    TORCH_CHECK(cu_seq_lens.is_contiguous(), "Tensor 'cu_seq_lens' must be contiguous");

    int64_t total_L    = q.size(1);
    int64_t batch_size = cu_seq_lens.size(0) - 1;
    TORCH_CHECK(batch_size >= 0, "batch_size must be >= 0");

    TORCH_CHECK(num_kv_heads > 0 && num_q_heads >= num_kv_heads && (num_q_heads % num_kv_heads == 0),
                "num_q_heads must be divisible by num_kv_heads");
    TORCH_CHECK(d_k == 64, "d_k must be 64 on NVIDIA Turing/Ampere/Ada/Hopper");
    TORCH_CHECK(d_v == 64, "d_v must be 64 on NVIDIA Turing/Ampere/Ada/Hopper");
    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64 for optimal SM alignment");

    if (past_states.has_value() && past_states->defined()) {
        const at::Tensor& ps = *past_states;
        TORCH_CHECK(ps.is_cuda() && ps.device() == q.device(), "past_states must be a CUDA tensor on same device");
        TORCH_CHECK(ps.scalar_type() == at::ScalarType::Half, "past_states must be FP16");
        TORCH_CHECK(ps.is_contiguous(), "past_states must be contiguous");
        TORCH_CHECK(ps.dim() == 4 && ps.size(0) == batch_size && ps.size(1) == num_kv_heads &&
                    ps.size(2) == d_v && ps.size(3) == d_k,
                    "past_states shape must be (batch_size, num_kv_heads, d_v, d_k)");
    }

    if (total_L == 0 || batch_size == 0) {
        at::Tensor empty_y = at::empty({num_q_heads, total_L, d_v}, q.options());
        at::Tensor empty_final_states = at::zeros({batch_size, num_kv_heads, d_v, d_k}, q.options());
        at::Tensor empty_W = at::empty({num_kv_heads, 0, d_k, d_k}, q.options());
        at::Tensor empty_S = at::empty({num_kv_heads, 0, d_v, d_k}, q.options());
        return std::make_tuple(empty_y, empty_final_states, empty_W, empty_S);
    }

    // Stage 2 - Execute Forward 3-Stage Pipeline

    // Step 1: Chunk WY Forward
    // Compute W and Y from raw inputs (activation recomputation)
    auto [W, Y] = chunk_wy_fwd(k, v, b, w, gamma, a, cu_seq_lens, chunk_size);

    // Step 2: Chunk Scan Forward
    // Sequentially scan recurrent states S across chunks and compute final_states
    // Note: S aliases/reuses Y memory buffer to minimize peak VRAM
    auto [S, final_states] = chunk_scan_fwd(W, Y, past_states, cu_seq_lens, chunk_size);

    // Step 3: Output Combine Forward
    // Combine intra-chunk attention and inter-chunk state S to generate attention output y
    at::Tensor y = output_combine_fwd(
        q, k, v, b, w, gamma, a, S, cu_seq_lens,
        num_q_heads, num_kv_heads, d_k, d_v, chunk_size
    );

    // Stage 3 - Return Results
    // If forward_only == true: free W and S, return empty tensors
    // If forward_only == false: retain W and S for backward reuse
    at::Tensor W_ret;
    at::Tensor S_ret;
    if (forward_only) {
        W.reset();
        S.reset();
        W_ret = at::Tensor();
        S_ret = at::Tensor();
    } else {
        W_ret = W;
        S_ret = S;
    }

    return std::make_tuple(y, final_states, W_ret, S_ret);
}

// Host coordinator for deltanet_bwd. Outputs: (dq, dk, dv, dw, db, da, dgamma, grad_past_states).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
deltanet_bwd(
    const at::Tensor& grad_output,                         // (num_q_heads, total_L, d_v) [FP16] - Upstream gradient
    const std::optional<at::Tensor>& grad_final_states,    // (batch_size, num_kv_heads, d_v, d_k) [FP16] - Gradient final states
    const std::optional<at::Tensor>& past_states,          // (batch_size, num_kv_heads, d_v, d_k) [FP16] - Initial states
    const at::Tensor& q,                                   // (num_q_heads, total_L, d_k) [FP16] - Raw Query
    const at::Tensor& k,                                   // (num_kv_heads, total_L, d_k) [FP16] - Raw Key
    const at::Tensor& v,                                   // (num_kv_heads, total_L, d_v) [FP16] - Raw Value
    const at::Tensor& w,                                   // (num_kv_heads, total_L, d_v) [FP16] - Raw W gate
    const at::Tensor& b,                                   // (num_kv_heads, total_L, d_k) [FP16] - Raw Beta
    const at::Tensor& a,                                   // (num_kv_heads, total_L, 1) [FP16] - Raw Alpha
    const at::Tensor& gamma,                               // (num_kv_heads, total_L, 1) [FP16] - Raw Gamma
    const at::Tensor& cu_seq_lens,                         // (batch_size + 1,) [Int32 / Int64] - Cumulative sequence lengths
    const std::optional<at::Tensor>& W,                    // (num_kv_heads, num_chunks, d_k, d_k) [FP16] - W from fwd (if cached)
    const std::optional<at::Tensor>& S,                    // (num_kv_heads, num_chunks, d_v, d_k) [FP16] - S from fwd (if cached)
    int64_t num_q_heads,                                   // scalar [int64] - Number of Query heads
    int64_t num_kv_heads,                                  // scalar [int64] - Number of KV heads
    int64_t d_k,                                           // scalar [int64] - Key dimension (default 64)
    int64_t d_v,                                           // scalar [int64] - Value dimension (default 64)
    int64_t chunk_size,                                    // scalar [int64] - Chunk size (default 64)
    bool has_past_states                                   // scalar [bool] - Flag indicating presence of past states
) {
    // Stage 1 - Input Validation
    TORCH_CHECK(grad_output.is_cuda(), "Input 'grad_output' must be a CUDA tensor");
    const c10::cuda::CUDAGuard device_guard(grad_output.device());

    TORCH_CHECK(q.is_cuda(), "Tensor 'q' must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "Tensor 'k' must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "Tensor 'v' must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "Tensor 'w' must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "Tensor 'b' must be a CUDA tensor");
    TORCH_CHECK(a.is_cuda(), "Tensor 'a' must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "Tensor 'gamma' must be a CUDA tensor");
    TORCH_CHECK(cu_seq_lens.is_cuda(), "Tensor 'cu_seq_lens' must be a CUDA tensor");

    TORCH_CHECK(grad_output.scalar_type() == at::ScalarType::Half, "grad_output must be FP16");
    TORCH_CHECK(q.scalar_type() == at::ScalarType::Half, "q must be FP16");
    TORCH_CHECK(k.scalar_type() == at::ScalarType::Half, "k must be FP16");
    TORCH_CHECK(v.scalar_type() == at::ScalarType::Half, "v must be FP16");
    TORCH_CHECK(w.scalar_type() == at::ScalarType::Half, "w must be FP16");
    TORCH_CHECK(b.scalar_type() == at::ScalarType::Half, "b must be FP16");
    TORCH_CHECK(a.scalar_type() == at::ScalarType::Half, "a must be FP16");
    TORCH_CHECK(gamma.scalar_type() == at::ScalarType::Half, "gamma must be FP16");

    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(cu_seq_lens.is_contiguous(), "cu_seq_lens must be contiguous");

    int64_t total_L    = q.size(1);
    int64_t batch_size = cu_seq_lens.size(0) - 1;
    TORCH_CHECK(batch_size >= 0, "batch_size must be >= 0");

    TORCH_CHECK(num_kv_heads > 0 && num_q_heads >= num_kv_heads && (num_q_heads % num_kv_heads == 0),
                "num_q_heads must be divisible by num_kv_heads");
    TORCH_CHECK(d_k == 64, "d_k must be 64 on NVIDIA Turing/Ampere/Ada/Hopper");
    TORCH_CHECK(d_v == 64, "d_v must be 64 on NVIDIA Turing/Ampere/Ada/Hopper");
    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64 for optimal SM alignment");

    if (W.has_value() && W->defined() && W->numel() > 0) {
        const at::Tensor& w_in = *W;
        TORCH_CHECK(w_in.is_cuda() && w_in.device() == q.device(), "Tensor 'W' must be on same device as 'q'");
        TORCH_CHECK(w_in.scalar_type() == at::ScalarType::Half, "Tensor 'W' must be FP16");
        TORCH_CHECK(w_in.is_contiguous(), "Tensor 'W' must be contiguous");
    }
    if (S.has_value() && S->defined() && S->numel() > 0) {
        const at::Tensor& s_in = *S;
        TORCH_CHECK(s_in.is_cuda() && s_in.device() == q.device(), "Tensor 'S' must be on same device as 'q'");
        TORCH_CHECK(s_in.scalar_type() == at::ScalarType::Half, "Tensor 'S' must be FP16");
        TORCH_CHECK(s_in.is_contiguous(), "Tensor 'S' must be contiguous");
    }

    if (total_L == 0 || batch_size == 0) {
        at::Tensor empty_dq     = at::empty({num_q_heads, total_L, d_k}, q.options());
        at::Tensor empty_dk     = at::empty({num_kv_heads, total_L, d_k}, q.options());
        at::Tensor empty_dv     = at::empty({num_kv_heads, total_L, d_v}, q.options());
        at::Tensor empty_dw     = at::empty({num_kv_heads, total_L, d_v}, q.options());
        at::Tensor empty_db     = at::empty({num_kv_heads, total_L, d_k}, q.options());
        at::Tensor empty_da     = at::empty({num_kv_heads, total_L, 1}, q.options());
        at::Tensor empty_dgamma = at::empty({num_kv_heads, total_L, 1}, q.options());
        at::Tensor empty_gps    = at::zeros({batch_size, num_kv_heads, d_v, d_k}, q.options());
        return std::make_tuple(empty_dq, empty_dk, empty_dv, empty_dw, empty_db, empty_da, empty_dgamma, empty_gps);
    }

    // Stage 2 - Manage W_use and Recurrent States S_list (Reuse from forward if cached)
    at::Tensor W_use;
    at::Tensor S_list;
    bool recomputed_w = false;
    bool recomputed_s = false;

    bool has_w = (W.has_value() && W->defined() && W->numel() > 0);
    bool has_s = (S.has_value() && S->defined() && S->numel() > 0);

    if (has_w && has_s) {
        // Full reuse: Both W and S were preserved from forward, avoid recomputation entirely
        W_use = *W;
        S_list = *S;
    } else if (has_w && !has_s) {
        // W is present, S is missing: only recompute Y then scan to recover S
        W_use = *W;
        auto [W_dummy, Y_temp] = chunk_wy_fwd(k, v, b, w, gamma, a, cu_seq_lens, chunk_size);
        auto [S_recomp, final_states_dummy] = chunk_scan_fwd(W_use, Y_temp, past_states, cu_seq_lens, chunk_size);
        S_list = S_recomp;
        recomputed_s = true;
    } else if (!has_w && has_s) {
        // S is present, W is missing: recompute W
        auto [W_recomp, Y_temp] = chunk_wy_fwd(k, v, b, w, gamma, a, cu_seq_lens, chunk_size);
        W_use = W_recomp;
        S_list = *S;
        recomputed_w = true;
        Y_temp.reset();
    } else {
        // Both missing: recompute W, Y then scan to compute S
        auto [W_recomp, Y_temp] = chunk_wy_fwd(k, v, b, w, gamma, a, cu_seq_lens, chunk_size);
        W_use = W_recomp;
        auto [S_recomp, final_states_dummy] = chunk_scan_fwd(W_use, Y_temp, past_states, cu_seq_lens, chunk_size);
        S_list = S_recomp;
        recomputed_w = true;
        recomputed_s = true;
    }

    // Stage 3 - Execute Backward Pipeline across 3 stages

    // Step 1: Output Combine Backward
    // Compute dq, dk_ext, dv_ext, da_ext, dS from grad_output, q, raw inputs, and S
    auto [dq, dk_ext, dv_ext, da_ext, dS] = output_combine_bwd(
        grad_output, q, k, v, b, w, gamma, a, S_list, cu_seq_lens,
        num_q_heads, num_kv_heads, d_k, d_v, chunk_size
    );

    // Step 2: Chunk Scan Backward
    // Reverse scan over S_list and W_use to compute grad_W, grad_Y, grad_past_states
    auto [grad_W, grad_Y, grad_past_states] = chunk_scan_bwd(
        dS, W_use, S_list, cu_seq_lens, grad_final_states, has_past_states, chunk_size
    );

    // Release intermediate buffers after step 2
    if (recomputed_s) {
        S_list.reset();
    }
    dS.reset();

    // Step 3: Chunk WY Backward
    // Consume grad_W, grad_Y and external gradients (dk_ext, dv_ext, da_ext)
    // Compute complete gradients for all raw inputs
    auto [dk, dv, db, dw, dgamma, da] = chunk_wy_bwd(
        grad_W, grad_Y, k, v, b, w, gamma, a, cu_seq_lens,
        dk_ext, dv_ext, da_ext, chunk_size
    );

    // Release temporary tensors
    grad_W.reset();
    grad_Y.reset();
    dk_ext.reset();
    dv_ext.reset();
    da_ext.reset();
    if (recomputed_w) {
        W_use.reset();
    }

    // Stage 4 - Return Results
    return std::make_tuple(dq, dk, dv, dw, db, da, dgamma, grad_past_states);
}
