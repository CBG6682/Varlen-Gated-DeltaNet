#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <optional>
#include <tuple>

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "deltanet_common.cuh"
#include "deltanet_ops.cuh"

using namespace cute;

// ============================================================================
// Part 1: Device Helpers & Super-Fused Decode Kernel (L = 1)
// ============================================================================

__device__ __forceinline__ float stable_sigmoid_infer_f(float x) {
    float xc = fmaxf(fminf(x, 80.0f), -80.0f);
    return 1.0f / (1.0f + expf(-xc));
}

template <
    typename T,
    int NUM_THREADS,
    int D_K,
    int D_V,
    bool ALLOW_NEG_EIGVAL,
    bool BETA_GUARD,
    typename TiledCopyHalf,
    typename TiledCopyFloat
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_deltanet_infer(
    float*         __restrict__ gPastState,   // (B, H_kv, 64, 64) [FP32] - Recurrent state (in-place)
    T*             __restrict__ gConvState,   // (B, 3, D_qkv) [FP16] - ShortConv1D cache (in-place)
    const T*       __restrict__ gProj,        // (B, total_proj_dim) [FP16] - Unified input projection
    const T*       __restrict__ gConvW,       // (4, D_qkv) [FP16] - Depthwise conv weights
    const float*   __restrict__ gA_log,       // (H_kv,) [FP32] - Log-decay rate
    const float*   __restrict__ gDtBias,      // (H_kv, 64) [FP32] - Decay bias
    const T*       __restrict__ gONormW,      // (64,) [FP16] - Per-head RMSNorm weight
    T*             __restrict__ gOutTok,      // (B, H_q * 64) [FP16] - Output before W_out (head-last)
    int64_t num_q_heads,                      // scalar [int64] - Total Query heads
    int64_t num_kv_heads,                     // scalar [int64] - Total KV heads
    int64_t num_groups,                       // scalar [int64] - Query heads per KV group
    int64_t D_qkv,                            // scalar [int64] - Total Q+K+V channels
    int64_t total_proj_dim,                   // scalar [int64] - Stride of gProj
    int64_t offset_b,                         // scalar [int64] - Start col of beta gate in gProj
    int64_t offset_w,                         // scalar [int64] - Start col of write gate in gProj
    int64_t offset_f,                         // scalar [int64] - Start col of decay gate in gProj
    int64_t offset_ogate,                     // scalar [int64] - Start col of o_gate in gProj
    float   scale,                            // scalar [float] - Query scale (1/sqrt(d_k))
    float   eps                               // scalar [float] - Per-head RMSNorm epsilon
) {
    // 1. Setup & Partitioning (16 rows x 8 cols = 128 threads, < 4 KB SMEM)
    int head_idx  = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid       = threadIdx.x;
    if (head_idx >= num_kv_heads) return;

    int row_id   = tid / 8;
    int col_id   = tid % 8;
    int col_base = col_id * 8;

    __shared__ alignas(16) float sVec[4][D_K]; // 1 KB SMEM for cross-row r_v, delta_v, o_raw

    TiledCopyHalf  tiled_copy_h;
    TiledCopyFloat tiled_copy_f32;
    auto thr_copy_h   = tiled_copy_h.get_slice(tid);
    auto thr_copy_f32 = tiled_copy_f32.get_slice(tid);

    auto bcast_layout  = make_layout(make_shape(Int<16>{}, Int<D_K>{}), make_stride(Int<0>{}, Int<1>{}));
    auto cs_mat_layout = make_layout(make_shape(Int<16>{}, Int<D_K>{}), make_stride(D_qkv, Int<1>{}));
    auto state_layout  = make_layout(make_shape(Int<D_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{}));
    auto out_layout    = make_layout(make_shape(Int<16>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{}));

    int64_t state_off    = (static_cast<int64_t>(batch_idx) * num_kv_heads + head_idx) * (D_V * D_K);
    int64_t cs_batch_off = static_cast<int64_t>(batch_idx) * 3 * D_qkv;
    int64_t proj_row_off = static_cast<int64_t>(batch_idx) * total_proj_dim;

    auto gS = make_tensor(make_gmem_ptr(gPastState + state_off), state_layout);
    auto tSgS = thr_copy_f32.partition_S(gS);
    auto rS   = make_fragment_like(tSgS);

    auto gDummyHalf = make_tensor(make_gmem_ptr(gProj), bcast_layout);
    auto tXgDummy   = thr_copy_h.partition_S(gDummyHalf);
    auto r_h0       = make_fragment_like(tXgDummy);
    auto r_h1       = make_fragment_like(tXgDummy);
    auto r_h2       = make_fragment_like(tXgDummy);
    auto r_h3       = make_fragment_like(tXgDummy);
    auto r_w0       = make_fragment_like(tXgDummy);
    auto r_w1       = make_fragment_like(tXgDummy);
    auto r_w2       = make_fragment_like(tXgDummy);
    auto r_w3       = make_fragment_like(tXgDummy);
    auto r_cs_new   = make_fragment_like(tXgDummy);

    auto gDtB_dummy = make_tensor(make_gmem_ptr(gDtBias), bcast_layout);
    auto r_f32_row  = make_fragment_like(thr_copy_f32.partition_S(gDtB_dummy));

    auto cTile16   = make_identity_tensor(make_shape(Int<16>{}, Int<D_K>{}));
    auto tXc16     = thr_copy_h.partition_S(cTile16);
    auto tPred3    = make_fragment_like<bool>(r_h0);
    auto tPred1    = make_fragment_like<bool>(r_h0);
    bool is_r_lt3  = (get<0>(tXc16(Int<0>{}, Int<0>{}, Int<0>{})) < 3);
    bool is_r_eq0  = (get<0>(tXc16(Int<0>{}, Int<0>{}, Int<0>{})) == 0);
    #pragma unroll
    for (int i = 0; i < size(tPred3); ++i) {
        tPred3(i) = is_r_lt3;
        tPred1(i) = is_r_eq0;
    }

    auto conv1d_silu_step = [&](int64_t col_off, float out_f[8]) {
        auto gCS0 = make_tensor(make_gmem_ptr(gConvState + cs_batch_off + 0 * D_qkv + col_off), bcast_layout);
        auto gCS1 = make_tensor(make_gmem_ptr(gConvState + cs_batch_off + 1 * D_qkv + col_off), bcast_layout);
        auto gCS2 = make_tensor(make_gmem_ptr(gConvState + cs_batch_off + 2 * D_qkv + col_off), bcast_layout);
        auto gCur = make_tensor(make_gmem_ptr(gProj      + proj_row_off + col_off),             bcast_layout);

        auto gCW0 = make_tensor(make_gmem_ptr(gConvW + 0 * D_qkv + col_off), bcast_layout);
        auto gCW1 = make_tensor(make_gmem_ptr(gConvW + 1 * D_qkv + col_off), bcast_layout);
        auto gCW2 = make_tensor(make_gmem_ptr(gConvW + 2 * D_qkv + col_off), bcast_layout);
        auto gCW3 = make_tensor(make_gmem_ptr(gConvW + 3 * D_qkv + col_off), bcast_layout);

        cute::copy(tiled_copy_h, thr_copy_h.partition_S(gCS0), r_h0);
        cute::copy(tiled_copy_h, thr_copy_h.partition_S(gCS1), r_h1);
        cute::copy(tiled_copy_h, thr_copy_h.partition_S(gCS2), r_h2);
        cute::copy(tiled_copy_h, thr_copy_h.partition_S(gCur), r_h3);

        cute::copy(tiled_copy_h, thr_copy_h.partition_S(gCW0), r_w0);
        cute::copy(tiled_copy_h, thr_copy_h.partition_S(gCW1), r_w1);
        cute::copy(tiled_copy_h, thr_copy_h.partition_S(gCW2), r_w2);
        cute::copy(tiled_copy_h, thr_copy_h.partition_S(gCW3), r_w3);

        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            float conv_val =
                half_to_float(r_h0(c, Int<0>{}, Int<0>{})) * half_to_float(r_w0(c, Int<0>{}, Int<0>{})) +
                half_to_float(r_h1(c, Int<0>{}, Int<0>{})) * half_to_float(r_w1(c, Int<0>{}, Int<0>{})) +
                half_to_float(r_h2(c, Int<0>{}, Int<0>{})) * half_to_float(r_w2(c, Int<0>{}, Int<0>{})) +
                half_to_float(r_h3(c, Int<0>{}, Int<0>{})) * half_to_float(r_w3(c, Int<0>{}, Int<0>{}));
            out_f[c] = conv_val * stable_sigmoid_infer_f(conv_val);

            r_cs_new(c, Int<0>{}, Int<0>{}) = (row_id == 0)
                ? r_h1(c, Int<0>{}, Int<0>{})
                : ((row_id == 1) ? r_h2(c, Int<0>{}, Int<0>{}) : r_h3(c, Int<0>{}, Int<0>{}));
        }

        // Wait for all warps to finish reading old conv_state before leader rows overwrite it
        __syncthreads();
        auto gCS_mat = make_tensor(make_gmem_ptr(gConvState + cs_batch_off + col_off), cs_mat_layout);
        cute::copy_if(tiled_copy_h, tPred3, r_cs_new, thr_copy_h.partition_D(gCS_mat));
    };

    // 2. ShortConv1D + SiLU + L2-norm(k) and gates/decay preparation
    int64_t D_q   = num_q_heads * D_K;
    int64_t D_k   = num_kv_heads * D_K;
    int64_t col_k = D_q + static_cast<int64_t>(head_idx) * D_K;
    int64_t col_v = D_q + D_k + static_cast<int64_t>(head_idx) * D_V;

    float k_f[8];
    float v_f[8];
    conv1d_silu_step(col_k, k_f);
    conv1d_silu_step(col_v, v_f);

    float local_sq_k = 0.0f;
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        local_sq_k += k_f[c] * k_f[c];
    }
    float inv_norm_k = rsqrtf(octet_reduce_sum(local_sq_k) + 1e-12f);
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        k_f[c] *= inv_norm_k;
    }

    int64_t kv_head_col = static_cast<int64_t>(head_idx) * D_K;
    auto gB_cur = make_tensor(make_gmem_ptr(gProj   + proj_row_off + offset_b + kv_head_col), bcast_layout);
    auto gW_cur = make_tensor(make_gmem_ptr(gProj   + proj_row_off + offset_w + kv_head_col), bcast_layout);
    auto gF_cur = make_tensor(make_gmem_ptr(gProj   + proj_row_off + offset_f + kv_head_col), bcast_layout);
    auto gDtB   = make_tensor(make_gmem_ptr(gDtBias + kv_head_col),                           bcast_layout);

    cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gB_cur),   r_h0);
    cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gW_cur),   r_h1);
    cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gF_cur),   r_h2);
    cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gDtB),   r_f32_row);

    // Load FP32 state S[64, 64] into 32 registers (4 rows x 8 cols per thread)
    cute::copy(tiled_copy_f32, tSgS, rS);

    float exp_a = expf(gA_log[head_idx]);
    float b_f[8];
    float w_f[8];
    float g_f[8];
    float alpha_f[8];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int c0 = 2 * i;
        int c1 = 2 * i + 1;

        __half2 b_sig = fast_sigmoid_h2(__halves2half2(to_half_val(r_h0(c0, Int<0>{}, Int<0>{})),
                                                       to_half_val(r_h0(c1, Int<0>{}, Int<0>{}))));
        __half2 w_sig = fast_sigmoid_h2(__halves2half2(to_half_val(r_h1(c0, Int<0>{}, Int<0>{})),
                                                       to_half_val(r_h1(c1, Int<0>{}, Int<0>{}))));

        float b_scale = ALLOW_NEG_EIGVAL ? 2.0f : 1.0f;
        b_f[c0] = __half2float(__low2half(b_sig))  * b_scale;
        b_f[c1] = __half2float(__high2half(b_sig)) * b_scale;
        w_f[c0] = __half2float(__low2half(w_sig));
        w_f[c1] = __half2float(__high2half(w_sig));
    }

    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        float x_decay = half_to_float(r_h2(c, Int<0>{}, Int<0>{})) + r_f32_row(c, Int<0>{}, Int<0>{});
        g_f[c]        = -exp_a * stable_softplus_f(x_decay);
        alpha_f[c]    = expf(g_f[c]);
    }

    if constexpr (BETA_GUARD) {
        beta_guard_project_octet(k_f, b_f, g_f);
    }

    // 3. Exact Gated Delta Rule-2 Update in FP32 (decay -> erase -> write)
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            rS(c, m, Int<0>{}) *= alpha_f[c];
        }
    }

    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        float local_su = 0.0f;
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            local_su += rS(c, m, Int<0>{}) * (b_f[c] * k_f[c]);
        }
        float su = octet_reduce_sum(local_su);
        if (col_id == 0) {
            sVec[0][m * 16 + row_id] = su;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        float dv = w_f[c] * v_f[c] - sVec[0][col_base + c];
        if (row_id == 0) {
            sVec[1][col_base + c] = dv;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        float dv_row = sVec[1][m * 16 + row_id];
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            rS(c, m, Int<0>{}) += dv_row * k_f[c];
        }
    }

    cute::copy(tiled_copy_f32, rS, thr_copy_f32.partition_D(gS));

    // 4. Exact Readout + Fused Per-Head RMSNorm * SiLU(o_gate) across GQA group
    auto gOW = make_tensor(make_gmem_ptr(gONormW), bcast_layout);
    cute::copy(tiled_copy_h, thr_copy_h.partition_S(gOW), r_w0);
    float r_ow[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        r_ow[c] = half_to_float(r_w0(c, Int<0>{}, Int<0>{}));
    }

    for (int64_t g = 0; g < num_groups; ++g) {
        int64_t h_q    = static_cast<int64_t>(head_idx) * num_groups + g;
        int64_t col_q  = h_q * D_K;
        int64_t col_og = offset_ogate + h_q * D_V;

        float q_f[8];
        conv1d_silu_step(col_q, q_f);

        float local_sq_q = 0.0f;
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            local_sq_q += q_f[c] * q_f[c];
        }
        float q_mult = rsqrtf(octet_reduce_sum(local_sq_q) + 1e-12f) * scale;
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            q_f[c] *= q_mult;
        }

        // Compute o_raw = S_new @ q_g directly from updated FP32 registers rS
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            float local_qs = 0.0f;
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                local_qs += rS(c, m, Int<0>{}) * q_f[c];
            }
            float qs = octet_reduce_sum(local_qs);
            if (col_id == 0) {
                sVec[2][m * 16 + row_id] = qs;
            }
        }
        __syncthreads();

        float o_raw_f[8];
        float local_sq_o = 0.0f;
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            float ov   = sVec[2][col_base + c];
            o_raw_f[c] = ov;
            local_sq_o += ov * ov;
        }
        float rstd_o = rsqrtf(octet_reduce_sum(local_sq_o) / static_cast<float>(D_V) + eps);

        auto gOG = make_tensor(make_gmem_ptr(gProj + proj_row_off + col_og), bcast_layout);
        cute::copy(tiled_copy_h, thr_copy_h.partition_S(gOG), r_w1);

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int c0 = 2 * i;
            int c1 = 2 * i + 1;
            __half2 og_act = fast_silu_h2(__halves2half2(to_half_val(r_w1(c0, Int<0>{}, Int<0>{})),
                                                         to_half_val(r_w1(c1, Int<0>{}, Int<0>{}))));
            float y0 = (o_raw_f[c0] * rstd_o * r_ow[c0]) * __half2float(__low2half(og_act));
            float y1 = (o_raw_f[c1] * rstd_o * r_ow[c1]) * __half2float(__high2half(og_act));
            r_h0(c0, Int<0>{}, Int<0>{}) = float_to_out<T>(y0);
            r_h0(c1, Int<0>{}, Int<0>{}) = float_to_out<T>(y1);
        }

        auto gOut_g = make_tensor(
            make_gmem_ptr(gOutTok + static_cast<int64_t>(batch_idx) * (num_q_heads * D_V) + h_q * D_V),
            out_layout
        );
        cute::copy_if(tiled_copy_h, tPred1, r_h0, thr_copy_h.partition_D(gOut_g));
        __syncthreads();
    }
}

namespace deltanet_infer_detail {

template <bool ALLOW_NEG_EIGVAL, bool BETA_GUARD>
inline void launch_deltanet_infer(
    float* p_s,
    cutlass::half_t* p_cs,
    const cutlass::half_t* p_proj,
    const cutlass::half_t* p_cw,
    const float* p_A_log,
    const float* p_dt,
    const cutlass::half_t* p_ow,
    cutlass::half_t* p_out_tok,
    int64_t B,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t num_groups,
    int64_t D_qkv,
    int64_t total_proj_dim,
    int64_t offset_b,
    int64_t offset_w,
    int64_t offset_f,
    int64_t offset_ogate,
    float scale,
    float eps,
    cudaStream_t stream
) {
    constexpr int NUM_THREADS = 128;
    constexpr int DK_DIM      = 64;
    constexpr int DV_DIM      = 64;
    using Element = cutlass::half_t;

    using TiledCopyHalf = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint4>, Element>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>, Stride<_8, _1>>{}
    ));
    using TiledCopyFloat = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint4>, float>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>, Stride<_8, _1>>{}
    ));

    dim3 grid(static_cast<unsigned int>(num_kv_heads), static_cast<unsigned int>(B));
    dim3 block(NUM_THREADS);

    kernel_deltanet_infer<
        Element, NUM_THREADS, DK_DIM, DV_DIM, ALLOW_NEG_EIGVAL, BETA_GUARD, TiledCopyHalf, TiledCopyFloat
    ><<<grid, block, 0, stream>>>(
        p_s, p_cs, p_proj, p_cw, p_A_log, p_dt, p_ow, p_out_tok,
        num_q_heads, num_kv_heads, num_groups, D_qkv, total_proj_dim,
        offset_b, offset_w, offset_f, offset_ogate, scale, eps
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace deltanet_infer_detail
using namespace deltanet_infer_detail;

// ============================================================================
// Part 2: Host Dispatch Function (deltablock_infer)
// ============================================================================

// Outputs: (y: (B, d_model) [FP16], past_states: (B, H_kv, 64, 64) [FP32 in-place], conv_state: (B, 3, D_qkv) [FP16 in-place])
std::tuple<at::Tensor, at::Tensor, at::Tensor>
deltablock_infer(
    const at::Tensor& x,
    const at::Tensor& past_states,
    const at::Tensor& conv_state,
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
    double scale,
    double eps,
    bool allow_neg_eigval,
    bool beta_guard
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
    check_t(past_states,   "past_states",   4, at::ScalarType::Float);
    check_t(conv_state,    "conv_state",    3, at::ScalarType::Half);
    check_t(W_proj,        "W_proj",        2, at::ScalarType::Half);
    check_t(conv_weight,   "conv_weight",   2, at::ScalarType::Half);
    check_t(A_log,         "A_log",         1, at::ScalarType::Float);
    check_t(dt_bias,       "dt_bias",       2, at::ScalarType::Float);
    check_t(o_norm_weight, "o_norm_weight", 1, at::ScalarType::Half);
    check_t(W_out,         "W_out",         2, at::ScalarType::Half);

    TORCH_CHECK(d_k == 64 && d_v == 64 && conv_size == 4, "d_k, d_v must be 64 and conv_size must be 4");
    TORCH_CHECK(num_kv_heads > 0 && num_q_heads > 0 && (num_q_heads % num_kv_heads == 0),
                "num_q_heads must be positive and divisible by num_kv_heads");

    int64_t B              = x.size(0);
    int64_t d_model        = x.size(1);
    int64_t num_groups     = num_q_heads / num_kv_heads;
    int64_t D_q            = num_q_heads * d_k;
    int64_t D_k            = num_kv_heads * d_k;
    int64_t D_v            = num_kv_heads * d_v;
    int64_t D_qkv          = D_q + D_k + D_v;
    int64_t D_out          = num_q_heads * d_v;
    int64_t offset_b       = D_qkv;
    int64_t offset_w       = offset_b + D_k;
    int64_t offset_f       = offset_w + D_v;
    int64_t offset_ogate   = offset_f + D_k;
    int64_t total_proj_dim = offset_ogate + D_out;

    TORCH_CHECK(d_model > 0 && d_model % 8 == 0, "d_model must be positive and divisible by 8");
    TORCH_CHECK(past_states.size(0) == B && past_states.size(1) == num_kv_heads &&
                past_states.size(2) == d_v && past_states.size(3) == d_k,
                "past_states shape mismatch");
    TORCH_CHECK(conv_state.size(0) == B && conv_state.size(1) == conv_size - 1 && conv_state.size(2) == D_qkv,
                "conv_state shape mismatch");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(past_states.data_ptr()) & 15) == 0,
                "past_states must be 16-byte aligned for in-place updates");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(conv_state.data_ptr()) & 15) == 0,
                "conv_state must be 16-byte aligned for in-place updates");

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

    if (has_mlp) {
        check_t(*norm1_weight,  "norm1_weight",  1, at::ScalarType::Half);
        check_t(*norm2_weight,  "norm2_weight",  1, at::ScalarType::Half);
        check_t(*mlp_W_gate_up, "mlp_W_gate_up", 2, at::ScalarType::Half);
        check_t(*mlp_W_down,    "mlp_W_down",    2, at::ScalarType::Half);
        TORCH_CHECK(norm1_weight->size(0) == d_model && norm2_weight->size(0) == d_model, "norm weight shape mismatch");
        TORCH_CHECK(mlp_W_gate_up->size(1) == d_model && mlp_W_gate_up->size(0) > 0 && (mlp_W_gate_up->size(0) % 16 == 0),
                    "mlp_W_gate_up shape invalid");
        int64_t hidden_dim = mlp_W_gate_up->size(0) / 2;
        TORCH_CHECK(mlp_W_down->size(0) == d_model && mlp_W_down->size(1) == hidden_dim, "mlp_W_down shape mismatch");
    }

    if (B == 0) {
        return std::make_tuple(torch::empty({0, d_model}, x.options()), past_states, conv_state);
    }

    // 2. PreNorm 1 + Unified Input Projection
    at::Tensor x_in = has_mlp
        ? std::get<0>(fused_rmsnorm_fwd(x, *norm1_weight, eps, false))
        : x;

    at::Tensor proj = at::mm(x_in, W_proj.t());
    if (has_mlp) {
        x_in.reset();
    }

    // 3. Launch Super-Fused Decode Kernel (L = 1)
    at::Tensor out_tok = torch::empty({B, D_out}, x.options());

    at::Tensor proj_a = ensure_aligned16(proj);
    at::Tensor cw_a   = ensure_aligned16(conv_weight);
    at::Tensor A_a    = ensure_aligned16(A_log);
    at::Tensor dt_a   = ensure_aligned16(dt_bias);
    at::Tensor ow_a   = ensure_aligned16(o_norm_weight);

    {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

        float*                 p_s      = past_states.data_ptr<float>();
        cutlass::half_t*       p_cs     = reinterpret_cast<cutlass::half_t*>(conv_state.data_ptr<at::Half>());
        const cutlass::half_t* p_proj   = reinterpret_cast<const cutlass::half_t*>(proj_a.data_ptr<at::Half>());
        const cutlass::half_t* p_cw     = reinterpret_cast<const cutlass::half_t*>(cw_a.data_ptr<at::Half>());
        const float*           p_A      = A_a.data_ptr<float>();
        const float*           p_dt     = dt_a.data_ptr<float>();
        const cutlass::half_t* p_ow     = reinterpret_cast<const cutlass::half_t*>(ow_a.data_ptr<at::Half>());
        cutlass::half_t*       p_out    = reinterpret_cast<cutlass::half_t*>(out_tok.data_ptr<at::Half>());

        if (allow_neg_eigval) {
            if (beta_guard) {
                launch_deltanet_infer<true, true>(
                    p_s, p_cs, p_proj, p_cw, p_A, p_dt, p_ow, p_out,
                    B, num_q_heads, num_kv_heads, num_groups, D_qkv, total_proj_dim,
                    offset_b, offset_w, offset_f, offset_ogate,
                    static_cast<float>(scale), static_cast<float>(eps), stream
                );
            } else {
                launch_deltanet_infer<true, false>(
                    p_s, p_cs, p_proj, p_cw, p_A, p_dt, p_ow, p_out,
                    B, num_q_heads, num_kv_heads, num_groups, D_qkv, total_proj_dim,
                    offset_b, offset_w, offset_f, offset_ogate,
                    static_cast<float>(scale), static_cast<float>(eps), stream
                );
            }
        } else {
            if (beta_guard) {
                launch_deltanet_infer<false, true>(
                    p_s, p_cs, p_proj, p_cw, p_A, p_dt, p_ow, p_out,
                    B, num_q_heads, num_kv_heads, num_groups, D_qkv, total_proj_dim,
                    offset_b, offset_w, offset_f, offset_ogate,
                    static_cast<float>(scale), static_cast<float>(eps), stream
                );
            } else {
                launch_deltanet_infer<false, false>(
                    p_s, p_cs, p_proj, p_cw, p_A, p_dt, p_ow, p_out,
                    B, num_q_heads, num_kv_heads, num_groups, D_qkv, total_proj_dim,
                    offset_b, offset_w, offset_f, offset_ogate,
                    static_cast<float>(scale), static_cast<float>(eps), stream
                );
            }
        }
    }

    proj_a.reset();
    proj.reset();

    // 4. Output Projection & Optional Post-Mixer SwiGLU MLP
    at::Tensor y_mix = at::mm(out_tok, W_out.t());
    out_tok.reset();

    if (!has_mlp) {
        return std::make_tuple(y_mix, past_states, conv_state);
    }

    auto [h, x_n2, rstd2_unused] = fused_residual_rmsnorm_fwd(x, y_mix, *norm2_weight, eps, false);
    y_mix.reset();
    rstd2_unused.reset();

    at::Tensor gate_up = at::mm(x_n2, mlp_W_gate_up->t());
    x_n2.reset();

    at::Tensor act = fused_swiglu_fwd(gate_up);
    gate_up.reset();

    at::Tensor ffn = at::mm(act, mlp_W_down->t());
    act.reset();

    at::Tensor y = fused_residual_add_fwd(h, ffn);
    h.reset();
    ffn.reset();

    return std::make_tuple(y, past_states, conv_state);
}
