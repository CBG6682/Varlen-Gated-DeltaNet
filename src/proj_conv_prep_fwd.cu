#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <optional>
#include <tuple>

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/swizzle.hpp>
#include <cute/swizzle_layout.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include "deltanet_common.cuh"
#include "deltanet_ops.cuh"

using namespace cute;

// ============================================================================
// Part 1: Device Helpers and Global Kernels
// ============================================================================

namespace proj_conv_prep_fwd_detail {

__device__ __forceinline__ float stable_sigmoid_f(float x) {
    float xc = fmaxf(fminf(x, 80.0f), -80.0f);
    return 1.0f / (1.0f + expf(-xc));
}

} // namespace proj_conv_prep_fwd_detail
using namespace proj_conv_prep_fwd_detail;

// Kernel 1 (Fwd): causal ShortConv1D (K=4) + SiLU + L2-norm + scale + head-first transpose
template <
    typename T,
    int NUM_THREADS,
    int CHUNK_SIZE,
    int D_HEAD,
    bool IS_TRAINING,
    bool HAS_PAST_CONV,
    bool OUT_CONV_STATE,
    typename TiledCopyHalf
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_fused_conv_act_l2_transpose_fwd(
    const T*       __restrict__ proj_ptr,          // (total_L, total_proj_dim) [FP16] - Unified projection
    const T*       __restrict__ conv_w_ptr,        // (4, D_qkv) [FP16] - Depthwise conv weights
    const T*       __restrict__ past_conv_ptr,     // (B, 3, D_qkv) [FP16] - Optional past conv cache
    const int32_t* __restrict__ chunk_offsets,     // (num_chunks + 1,) [Int32] - Chunk token starts
    const int32_t* __restrict__ chunk_valid_lens,  // (num_chunks,) [Int32] - Valid tokens per chunk
    const int32_t* __restrict__ chunk_seq_ids,     // (num_chunks,) [Int32] - Sequence id per chunk
    const int32_t* __restrict__ chunk_local_starts,// (num_chunks,) [Int32] - Local token start in sequence
    const int32_t* __restrict__ chunk_is_last,     // (num_chunks,) [Int32] - Last chunk flag per sequence
    T*             __restrict__ q_h_ptr,           // (H_q, total_L, 64) [FP16] - Output Query
    T*             __restrict__ k_h_ptr,           // (H_kv, total_L, 64) [FP16] - Output Key
    T*             __restrict__ v_h_ptr,           // (H_kv, total_L, 64) [FP16] - Output Value
    T*             __restrict__ final_conv_ptr,    // (B, 3, D_qkv) [FP16] - Updated conv state
    float*         __restrict__ q_inv_norm_ptr,    // (H_q, total_L) [FP32] - Saved Q inv norm
    float*         __restrict__ k_inv_norm_ptr,    // (H_kv, total_L) [FP32] - Saved K inv norm
    int64_t total_proj_dim,                        // scalar [int64] - Stride of proj
    int64_t D_qkv,                                 // scalar [int64] - Total Q+K+V width
    int64_t H_q,                                   // scalar [int64] - Number of query heads
    int64_t H_kv,                                  // scalar [int64] - Number of KV heads
    int64_t total_L,                               // scalar [int64] - Total packed tokens
    int64_t num_chunks,                            // scalar [int64] - Total varlen chunks
    float   scale                                  // scalar [float] - Query scale
) {
    // 1. Setup
    int chunk_idx   = blockIdx.x;
    int head_global = blockIdx.y;
    int tid         = threadIdx.x;
    if (chunk_idx >= num_chunks) return;

    int row_id   = tid / 8;
    int col_id   = tid % 8;
    int col_base = col_id * 8;

    int mode = 2;
    int head_local = head_global - static_cast<int>(H_q + H_kv);
    if (head_global < H_q) {
        mode = 0;
        head_local = head_global;
    } else if (head_global < H_q + H_kv) {
        mode = 1;
        head_local = head_global - static_cast<int>(H_q);
    }

    int32_t chunk_start = chunk_offsets[chunk_idx];
    int32_t valid_len   = chunk_valid_lens[chunk_idx];
    int32_t seq_id      = chunk_seq_ids[chunk_idx];
    int32_t local_start = chunk_local_starts[chunk_idx];
    int32_t is_last     = chunk_is_last[chunk_idx];

    int64_t col_offset = static_cast<int64_t>(head_global) * D_HEAD;

    // Halo lives in rows 5..7 so main chunk rows 8..71 stay 8-row aligned for Swizzle<3, 3, 3>
    __shared__ alignas(16) T sIn_raw[72 * D_HEAD];
    __shared__ alignas(16) T sW_raw[16 * D_HEAD];

    using SwizzleOp = Swizzle<3, 3, 3>;
    auto smem_layout_in   = composition(SwizzleOp{}, make_layout(make_shape(Int<72>{},         Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{})));
    auto smem_layout_main = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{})));
    auto smem_layout_w    = composition(SwizzleOp{}, make_layout(make_shape(Int<16>{},         Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{})));

    auto sIn      = make_tensor(make_smem_ptr(sIn_raw),              smem_layout_in);
    auto sIn_main = make_tensor(make_smem_ptr(sIn_raw + 8 * D_HEAD), smem_layout_main);
    auto sW       = make_tensor(make_smem_ptr(sW_raw),               smem_layout_w);

    TiledCopyHalf tiled_copy;
    auto thr_copy = tiled_copy.get_slice(tid);

    auto gW = make_tensor(
        make_gmem_ptr(conv_w_ptr + col_offset),
        make_layout(make_shape(Int<16>{}, Int<D_HEAD>{}), make_stride(D_qkv, Int<1>{}))
    );
    auto gProjMain = make_tensor(
        make_gmem_ptr(proj_ptr + static_cast<int64_t>(chunk_start) * total_proj_dim + col_offset),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(total_proj_dim, Int<1>{}))
    );

    T* out_base_ptr = (mode == 0) ? q_h_ptr : ((mode == 1) ? k_h_ptr : v_h_ptr);
    auto gOutMain = make_tensor(
        make_gmem_ptr(out_base_ptr + (static_cast<int64_t>(head_local) * total_L + chunk_start) * D_HEAD),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{}))
    );

    auto tXgW      = thr_copy.partition_S(gW);
    auto tXsW      = thr_copy.partition_D(sW);
    auto tXgProj   = thr_copy.partition_S(gProjMain);
    auto tXsInMain = thr_copy.partition_D(sIn_main);
    auto tXgOut    = thr_copy.partition_D(gOutMain);

    auto tCrW    = make_fragment_like(tXgW);
    auto tCrHalo = make_fragment_like(tXgW);
    auto tCrMain = make_fragment_like(tXgProj);

    auto tPred4     = make_fragment_like<bool>(tXgW);
    auto tPred3     = make_fragment_like<bool>(tXgW);
    auto tPredFinal = make_fragment_like<bool>(tXgW);
    auto tPred64    = make_fragment_like<bool>(tXgProj);

    auto cTile16   = make_identity_tensor(make_shape(Int<16>{}, Int<D_HEAD>{}));
    auto tXcTile16 = thr_copy.partition_S(cTile16);
    #pragma unroll
    for (int m = 0; m < size<1>(tPred4); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(tPred4); ++n) {
            int r = get<0>(tXcTile16(Int<0>{}, m, n));
            #pragma unroll
            for (int v = 0; v < size<0>(tPred4); ++v) {
                tPred4(v, m, n)     = (r < 4);
                tPred3(v, m, n)     = (r < 3);
                tPredFinal(v, m, n) = (r < 3) && (is_last != 0);
            }
        }
    }

    auto cTile64   = make_identity_tensor(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}));
    auto tXcTile64 = thr_copy.partition_S(cTile64);
    #pragma unroll
    for (int m = 0; m < size<1>(tPred64); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(tPred64); ++n) {
            int r = get<0>(tXcTile64(Int<0>{}, m, n));
            #pragma unroll
            for (int v = 0; v < size<0>(tPred64); ++v) {
                tPred64(v, m, n) = (r < valid_len);
            }
        }
    }

    // 2. Load conv weights, halo tokens, and chunk tokens into swizzled SMEM
    clear(tCrW);
    cute::copy_if(tiled_copy, tPred4, tXgW, tCrW);
    cute::copy(tiled_copy, tCrW, tXsW);

    clear(tCrHalo);
    if (local_start > 0) {
        auto gHaloPrev = make_tensor(
            make_gmem_ptr(proj_ptr + static_cast<int64_t>(chunk_start - 3) * total_proj_dim + col_offset),
            make_layout(make_shape(Int<16>{}, Int<D_HEAD>{}), make_stride(total_proj_dim, Int<1>{}))
        );
        cute::copy_if(tiled_copy, tPred3, thr_copy.partition_S(gHaloPrev), tCrHalo);
    } else if constexpr (HAS_PAST_CONV) {
        auto gHaloState = make_tensor(
            make_gmem_ptr(past_conv_ptr + static_cast<int64_t>(seq_id) * 3 * D_qkv + col_offset),
            make_layout(make_shape(Int<16>{}, Int<D_HEAD>{}), make_stride(D_qkv, Int<1>{}))
        );
        cute::copy_if(tiled_copy, tPred3, thr_copy.partition_S(gHaloState), tCrHalo);
    }

    if (row_id < 3) {
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            sIn(5 + row_id, col_base + v) = tCrHalo(v, 0, 0);
        }
    }

    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy, tXgProj, tCrMain);
    } else {
        clear(tCrMain);
        cute::copy_if(tiled_copy, tPred64, tXgProj, tCrMain);
    }
    cute::copy(tiled_copy, tCrMain, tXsInMain);
    __syncthreads();

    // 3. Emit updated conv_state from final 3 tokens of the sequence
    if constexpr (OUT_CONV_STATE) {
        if (is_last != 0 && row_id < 3) {
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                tCrHalo(v, 0, 0) = sIn(5 + valid_len + row_id, col_base + v);
            }
        }
        auto gFinalState = make_tensor(
            make_gmem_ptr(final_conv_ptr + static_cast<int64_t>(seq_id) * 3 * D_qkv + col_offset),
            make_layout(make_shape(Int<16>{}, Int<D_HEAD>{}), make_stride(D_qkv, Int<1>{}))
        );
        cute::copy_if(tiled_copy, tPredFinal, tCrHalo, thr_copy.partition_D(gFinalState));
    }

    // 4. Causal Conv1D + SiLU + L2-norm + Scale + Head-first store
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        int local_row = m * 16 + row_id;
        float silu_f[8];
        float local_sq = 0.0f;

        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            float conv_f = 0.0f;
            #pragma unroll
            for (int tap = 0; tap < 4; ++tap) {
                conv_f += half_to_float(sW(tap, col_base + v)) *
                          half_to_float(sIn(5 + local_row + tap, col_base + v));
            }
            float sf = conv_f * stable_sigmoid_f(conv_f);
            silu_f[v] = sf;
            local_sq += sf * sf;
        }

        if (mode < 2) {
            // Execute reduction across all warp lanes before storing valid tokens
            float sum_sq   = octet_reduce_sum(local_sq);
            float inv_norm = rsqrtf(sum_sq + 1e-12f);
            float mult     = (mode == 0) ? (inv_norm * scale) : inv_norm;

            if constexpr (IS_TRAINING) {
                if (col_id == 0 && local_row < valid_len) {
                    int64_t tok_idx = static_cast<int64_t>(head_local) * total_L + chunk_start + local_row;
                    if (mode == 0) {
                        q_inv_norm_ptr[tok_idx] = inv_norm;
                    } else {
                        k_inv_norm_ptr[tok_idx] = inv_norm;
                    }
                }
            }

            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                tCrMain(v, m, 0) = float_to_out<T>(silu_f[v] * mult);
            }
        } else {
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                tCrMain(v, m, 0) = float_to_out<T>(silu_f[v]);
            }
        }
    }

    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy, tCrMain, tXgOut);
    } else {
        cute::copy_if(tiled_copy, tPred64, tCrMain, tXgOut);
    }
}

// Kernel 2 (Fwd): gates sigmoid + channel decay + beta_guard + chunk prefix scan G_log2
template <
    typename T,
    int NUM_THREADS,
    int CHUNK_SIZE,
    int D_HEAD,
    bool ALLOW_NEG_EIGVAL,
    bool BETA_GUARD,
    typename TiledCopyHalf,
    typename TiledCopyFloat
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_fused_gates_decay_betaguard_transpose_fwd(
    const T*       __restrict__ proj_ptr,          // (total_L, total_proj_dim) [FP16] - Unified projection
    const T*       __restrict__ k_h_ptr,           // (H_kv, total_L, 64) [FP16] - Normalized Key
    const float*   __restrict__ A_log_ptr,         // (H_kv,) [FP32] - Log-decay rate
    const float*   __restrict__ dt_bias_ptr,       // (H_kv, 64) [FP32] - Decay bias
    const int32_t* __restrict__ chunk_offsets,     // (num_chunks + 1,) [Int32] - Chunk token starts
    const int32_t* __restrict__ chunk_valid_lens,  // (num_chunks,) [Int32] - Valid tokens per chunk
    T*             __restrict__ b_eff_h_ptr,       // (H_kv, total_L, 64) [FP16] - Effective erase gate
    T*             __restrict__ w_h_ptr,           // (H_kv, total_L, 64) [FP16] - Write gate
    float*         __restrict__ G_log2_h_ptr,      // (H_kv, total_L, 64) [FP32] - Chunk cumsum log2 decay
    int64_t total_proj_dim,                        // scalar [int64] - Stride of proj
    int64_t offset_b,                              // scalar [int64] - Column start of b
    int64_t offset_w,                              // scalar [int64] - Column start of w
    int64_t offset_f,                              // scalar [int64] - Column start of f
    int64_t total_L,                               // scalar [int64] - Total packed tokens
    int64_t num_chunks                             // scalar [int64] - Total varlen chunks
) {
    // 1. Setup
    int chunk_idx = blockIdx.x;
    int h         = blockIdx.y;
    int tid       = threadIdx.x;
    if (chunk_idx >= num_chunks) return;

    int row_id   = tid / 8;
    int col_id   = tid % 8;
    int col_base = col_id * 8;

    int32_t chunk_start = chunk_offsets[chunk_idx];
    int32_t valid_len   = chunk_valid_lens[chunk_idx];

    float neg_exp_A = -expf(A_log_ptr[h]);
    const float* dt_row_ptr = dt_bias_ptr + static_cast<int64_t>(h) * D_HEAD + col_base;
    float4 dt0 = *reinterpret_cast<const float4*>(dt_row_ptr);
    float4 dt1 = *reinterpret_cast<const float4*>(dt_row_ptr + 4);
    float r_dt[8] = {dt0.x, dt0.y, dt0.z, dt0.w, dt1.x, dt1.y, dt1.z, dt1.w};

    __shared__ alignas(16) float sG_raw[CHUNK_SIZE * D_HEAD];
    using SwizzleF32 = Swizzle<3, 2, 3>;
    auto smem_layout_f32 = composition(
        SwizzleF32{},
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{}))
    );
    auto sG = make_tensor(make_smem_ptr(sG_raw), smem_layout_f32);

    TiledCopyHalf tiled_copy_h;
    auto thr_copy_h = tiled_copy_h.get_slice(tid);

    int64_t head_tok_off = (static_cast<int64_t>(h) * total_L + chunk_start) * D_HEAD;
    int64_t col_off_h    = static_cast<int64_t>(h) * D_HEAD;

    auto gB_in = make_tensor(
        make_gmem_ptr(proj_ptr + static_cast<int64_t>(chunk_start) * total_proj_dim + offset_b + col_off_h),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(total_proj_dim, Int<1>{}))
    );
    auto gW_in = make_tensor(
        make_gmem_ptr(proj_ptr + static_cast<int64_t>(chunk_start) * total_proj_dim + offset_w + col_off_h),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(total_proj_dim, Int<1>{}))
    );
    auto gF_in = make_tensor(
        make_gmem_ptr(proj_ptr + static_cast<int64_t>(chunk_start) * total_proj_dim + offset_f + col_off_h),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(total_proj_dim, Int<1>{}))
    );
    auto gK_in = make_tensor(
        make_gmem_ptr(k_h_ptr + head_tok_off),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{}))
    );
    auto gB_out = make_tensor(
        make_gmem_ptr(b_eff_h_ptr + head_tok_off),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{}))
    );
    auto gW_out = make_tensor(
        make_gmem_ptr(w_h_ptr + head_tok_off),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{}))
    );

    auto tXgB = thr_copy_h.partition_S(gB_in);
    auto tXgW = thr_copy_h.partition_S(gW_in);
    auto tXgF = thr_copy_h.partition_S(gF_in);
    auto tXgK = thr_copy_h.partition_S(gK_in);
    auto tXgB_out = thr_copy_h.partition_D(gB_out);
    auto tXgW_out = thr_copy_h.partition_D(gW_out);

    auto tCrB = make_fragment_like(tXgB);
    auto tCrW = make_fragment_like(tXgW);
    auto tCrF = make_fragment_like(tXgF);
    auto tCrK = make_fragment_like(tXgK);

    auto tPred64   = make_fragment_like<bool>(tXgB);
    auto cTile64   = make_identity_tensor(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}));
    auto tXcTile64 = thr_copy_h.partition_S(cTile64);
    #pragma unroll
    for (int m = 0; m < size<1>(tPred64); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(tPred64); ++n) {
            bool in_bounds = (get<0>(tXcTile64(Int<0>{}, m, n)) < valid_len);
            #pragma unroll
            for (int v = 0; v < size<0>(tPred64); ++v) {
                tPred64(v, m, n) = in_bounds;
            }
        }
    }

    TiledCopyFloat tiled_copy_f32;
    auto thr_copy_f32 = tiled_copy_f32.get_slice(tid);
    auto gG_out = make_tensor(
        make_gmem_ptr(G_log2_h_ptr + head_tok_off),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_HEAD>{}), make_stride(Int<D_HEAD>{}, Int<1>{}))
    );
    auto tXsG      = thr_copy_f32.partition_S(sG);
    auto tXgG_out  = thr_copy_f32.partition_D(gG_out);
    auto tCrG      = make_fragment_like(tXsG);
    auto tPredF32  = make_fragment_like<bool>(tXsG);
    auto tXcF32    = thr_copy_f32.partition_S(cTile64);
    #pragma unroll
    for (int m = 0; m < size<1>(tPredF32); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(tPredF32); ++n) {
            bool in_bounds = (get<0>(tXcF32(Int<0>{}, m, n)) < valid_len);
            #pragma unroll
            for (int v = 0; v < size<0>(tPredF32); ++v) {
                tPredF32(v, m, n) = in_bounds;
            }
        }
    }

    // 2. Gates, channel-wise decay, beta_guard, and head-first gate stores
    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy_h, tXgB, tCrB);
        cute::copy(tiled_copy_h, tXgW, tCrW);
        cute::copy(tiled_copy_h, tXgF, tCrF);
        if constexpr (BETA_GUARD) {
            cute::copy(tiled_copy_h, tXgK, tCrK);
        }
    } else {
        clear(tCrB);
        clear(tCrW);
        clear(tCrF);
        cute::copy_if(tiled_copy_h, tPred64, tXgB, tCrB);
        cute::copy_if(tiled_copy_h, tPred64, tXgW, tCrW);
        cute::copy_if(tiled_copy_h, tPred64, tXgF, tCrF);
        if constexpr (BETA_GUARD) {
            clear(tCrK);
            cute::copy_if(tiled_copy_h, tPred64, tXgK, tCrK);
        }
    }

    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        #pragma unroll
        for (int v = 0; v < 8; v += 2) {
            __half2 w_h2 = __halves2half2(to_half_val(tCrW(v, m, 0)), to_half_val(tCrW(v + 1, m, 0)));
            __half2 sw_h2 = fast_sigmoid_h2(w_h2);
            tCrW(v, m, 0)     = half_to_out<T>(__low2half(sw_h2));
            tCrW(v + 1, m, 0) = half_to_out<T>(__high2half(sw_h2));
        }
    }

    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy_h, tCrW, tXgW_out);
    } else {
        cute::copy_if(tiled_copy_h, tPred64, tCrW, tXgW_out);
    }

    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        int local_row = m * 16 + row_id;
        float b_f[8];
        float g_f[8];
        float k_f[8];

        #pragma unroll
        for (int v = 0; v < 8; v += 2) {
            __half2 b_h2  = __halves2half2(to_half_val(tCrB(v, m, 0)), to_half_val(tCrB(v + 1, m, 0)));
            __half2 sb_h2 = fast_sigmoid_h2(b_h2);
            b_f[v]     = __half2float(__low2half(sb_h2));
            b_f[v + 1] = __half2float(__high2half(sb_h2));
        }

        constexpr float b_scale = ALLOW_NEG_EIGVAL ? 2.0f : 1.0f;
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            b_f[v] *= b_scale;
            g_f[v]  = neg_exp_A * stable_softplus_f(half_to_float(tCrF(v, m, 0)) + r_dt[v]);
            if constexpr (BETA_GUARD) {
                k_f[v] = half_to_float(tCrK(v, m, 0));
            }
        }

        if constexpr (BETA_GUARD) {
            // Keep g_f in natural log units for beta_guard before log2 scaling
            beta_guard_project_octet(k_f, b_f, g_f);
        }

        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            tCrB(v, m, 0) = float_to_out<T>(b_f[v]);
            sG(local_row, col_base + v) = (local_row < valid_len) ? (g_f[v] * DELTANET_RCP_LN2) : 0.0f;
        }
    }

    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy_h, tCrB, tXgB_out);
    } else {
        cute::copy_if(tiled_copy_h, tPred64, tCrB, tXgB_out);
    }
    __syncthreads();

    // 3. Chunk-local inclusive prefix scan along time and 128-bit FP32 store
    if (tid < D_HEAD) {
        float acc = 0.0f;
        #pragma unroll 8
        for (int r = 0; r < CHUNK_SIZE; ++r) {
            acc += sG(r, tid);
            sG(r, tid) = acc;
        }
    }
    __syncthreads();

    cute::copy(tiled_copy_f32, tXsG, tCrG);
    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy_f32, tCrG, tXgG_out);
    } else {
        cute::copy_if(tiled_copy_f32, tPredF32, tCrG, tXgG_out);
    }
}

// ============================================================================
// Part 2: Host Dispatch Helpers and PyTorch Entry Points
// ============================================================================

namespace proj_conv_prep_fwd_detail {

using TiledCopyChunkHalf = decltype(make_tiled_copy(
    Copy_Atom<UniversalCopy<uint4>, cutlass::half_t>{},
    Layout<Shape<_16, _8>, Stride<_8, _1>>{},
    Layout<Shape<_1, _8>, Stride<_8, _1>>{}
));

using TiledCopyChunkFloat = decltype(make_tiled_copy(
    Copy_Atom<UniversalCopy<uint4>, float>{},
    Layout<Shape<_8, _16>, Stride<_16, _1>>{},
    Layout<Shape<_1, _4>, Stride<_4, _1>>{}
));

struct ProjOffsets {
    int64_t D_q;
    int64_t D_k;
    int64_t D_v;
    int64_t D_out;
    int64_t D_qkv;
    int64_t offset_b;
    int64_t offset_w;
    int64_t offset_f;
    int64_t offset_ogate;
    int64_t total_proj_dim;
};

inline ProjOffsets compute_proj_offsets(
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t d_k,
    int64_t d_v
) {
    ProjOffsets po;
    po.D_q            = num_q_heads * d_k;
    po.D_k            = num_kv_heads * d_k;
    po.D_v            = num_kv_heads * d_v;
    po.D_out          = num_q_heads * d_v;
    po.D_qkv          = po.D_q + po.D_k + po.D_v;
    po.offset_b       = po.D_qkv;
    po.offset_w       = po.offset_b + po.D_k;
    po.offset_f       = po.offset_w + po.D_v;
    po.offset_ogate   = po.offset_f + po.D_k;
    po.total_proj_dim = po.offset_ogate + po.D_out;
    return po;
}

inline void validate_common_proj_args(
    const at::Tensor& proj,
    const at::Tensor& conv_weight,
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    const std::optional<at::Tensor>& past_conv_state,
    const at::Tensor& cu_seq_lens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t d_k,
    int64_t d_v,
    int64_t conv_size,
    int64_t chunk_size,
    const ProjOffsets& po,
    int64_t batch_size
) {
    TORCH_CHECK(d_k == 64 && d_v == 64, "d_k and d_v must be 64");
    TORCH_CHECK(conv_size == 4, "conv_size must be 4");
    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64");
    TORCH_CHECK(num_kv_heads > 0 && num_q_heads > 0 && (num_q_heads % num_kv_heads == 0),
                "num_q_heads must be positive and divisible by num_kv_heads");

    TORCH_CHECK(proj.is_cuda() && proj.scalar_type() == at::ScalarType::Half, "proj must be CUDA float16");
    TORCH_CHECK(proj.dim() == 2 && proj.is_contiguous(), "proj must be contiguous 2D");
    TORCH_CHECK(proj.size(1) == po.total_proj_dim, "proj column dimension mismatch");

    TORCH_CHECK(conv_weight.is_cuda() && conv_weight.scalar_type() == at::ScalarType::Half, "conv_weight must be CUDA float16");
    TORCH_CHECK(conv_weight.dim() == 2 && conv_weight.is_contiguous(), "conv_weight must be contiguous 2D");
    TORCH_CHECK(conv_weight.size(0) == conv_size && conv_weight.size(1) == po.D_qkv, "conv_weight shape mismatch");

    TORCH_CHECK(A_log.is_cuda() && A_log.scalar_type() == at::ScalarType::Float, "A_log must be CUDA float32");
    TORCH_CHECK(A_log.dim() == 1 && A_log.is_contiguous() && A_log.size(0) == num_kv_heads, "A_log shape mismatch");

    TORCH_CHECK(dt_bias.is_cuda() && dt_bias.scalar_type() == at::ScalarType::Float, "dt_bias must be CUDA float32");
    TORCH_CHECK(dt_bias.dim() == 2 && dt_bias.is_contiguous(), "dt_bias must be contiguous 2D");
    TORCH_CHECK(dt_bias.size(0) == num_kv_heads && dt_bias.size(1) == d_k, "dt_bias shape mismatch");

    TORCH_CHECK(cu_seq_lens.is_cuda() && cu_seq_lens.dim() == 1 && cu_seq_lens.size(0) >= 1,
                "cu_seq_lens must be a 1D CUDA tensor with size >= 1");
    TORCH_CHECK(cu_seq_lens.scalar_type() == at::ScalarType::Int || cu_seq_lens.scalar_type() == at::ScalarType::Long,
                "cu_seq_lens must be int32 or int64");

    if (past_conv_state.has_value()) {
        const auto& pcs = *past_conv_state;
        TORCH_CHECK(pcs.is_cuda() && pcs.scalar_type() == at::ScalarType::Half, "past_conv_state must be CUDA float16");
        TORCH_CHECK(pcs.dim() == 3 && pcs.is_contiguous(), "past_conv_state must be contiguous 3D");
        TORCH_CHECK(pcs.size(0) == batch_size && pcs.size(1) == conv_size - 1 && pcs.size(2) == po.D_qkv,
                    "past_conv_state shape mismatch");
    }
}

template <bool IS_TRAINING, bool HAS_PAST_CONV, bool OUT_CONV_STATE>
inline void launch_conv_act_l2_fwd(
    const cutlass::half_t* proj_ptr,
    const cutlass::half_t* conv_w_ptr,
    const cutlass::half_t* past_conv_ptr,
    const int32_t* chunk_offsets,
    const int32_t* chunk_valid_lens,
    const int32_t* chunk_seq_ids,
    const int32_t* chunk_local_starts,
    const int32_t* chunk_is_last,
    cutlass::half_t* q_h_ptr,
    cutlass::half_t* k_h_ptr,
    cutlass::half_t* v_h_ptr,
    cutlass::half_t* final_conv_ptr,
    float* q_inv_norm_ptr,
    float* k_inv_norm_ptr,
    int64_t total_proj_dim,
    int64_t D_qkv,
    int64_t H_q,
    int64_t H_kv,
    int64_t total_L,
    int64_t num_chunks,
    float scale,
    cudaStream_t stream
) {
    constexpr int NUM_THREADS = 128;
    constexpr int CHUNK_SIZE  = 64;
    constexpr int D_HEAD      = 64;
    dim3 grid(static_cast<unsigned int>(num_chunks), static_cast<unsigned int>(H_q + 2 * H_kv));
    dim3 block(NUM_THREADS);
    kernel_fused_conv_act_l2_transpose_fwd<
        cutlass::half_t, NUM_THREADS, CHUNK_SIZE, D_HEAD,
        IS_TRAINING, HAS_PAST_CONV, OUT_CONV_STATE, TiledCopyChunkHalf
    ><<<grid, block, 0, stream>>>(
        proj_ptr, conv_w_ptr, past_conv_ptr,
        chunk_offsets, chunk_valid_lens, chunk_seq_ids, chunk_local_starts, chunk_is_last,
        q_h_ptr, k_h_ptr, v_h_ptr, final_conv_ptr, q_inv_norm_ptr, k_inv_norm_ptr,
        total_proj_dim, D_qkv, H_q, H_kv, total_L, num_chunks, scale
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool ALLOW_NEG_EIGVAL, bool BETA_GUARD>
inline void launch_gates_decay_fwd(
    const cutlass::half_t* proj_ptr,
    const cutlass::half_t* k_h_ptr,
    const float* A_log_ptr,
    const float* dt_bias_ptr,
    const int32_t* chunk_offsets,
    const int32_t* chunk_valid_lens,
    cutlass::half_t* b_eff_h_ptr,
    cutlass::half_t* w_h_ptr,
    float* G_log2_h_ptr,
    const ProjOffsets& po,
    int64_t H_kv,
    int64_t total_L,
    int64_t num_chunks,
    cudaStream_t stream
) {
    constexpr int NUM_THREADS = 128;
    constexpr int CHUNK_SIZE  = 64;
    constexpr int D_HEAD      = 64;
    dim3 grid(static_cast<unsigned int>(num_chunks), static_cast<unsigned int>(H_kv));
    dim3 block(NUM_THREADS);
    kernel_fused_gates_decay_betaguard_transpose_fwd<
        cutlass::half_t, NUM_THREADS, CHUNK_SIZE, D_HEAD,
        ALLOW_NEG_EIGVAL, BETA_GUARD, TiledCopyChunkHalf, TiledCopyChunkFloat
    ><<<grid, block, 0, stream>>>(
        proj_ptr, k_h_ptr, A_log_ptr, dt_bias_ptr, chunk_offsets, chunk_valid_lens,
        b_eff_h_ptr, w_h_ptr, G_log2_h_ptr,
        po.total_proj_dim, po.offset_b, po.offset_w, po.offset_f, total_L, num_chunks
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace proj_conv_prep_fwd_detail
using namespace proj_conv_prep_fwd_detail;

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
) {
    int64_t batch_size = cu_seq_lens.size(0) - 1;
    ProjOffsets po = compute_proj_offsets(num_q_heads, num_kv_heads, d_k, d_v);
    validate_common_proj_args(
        proj, conv_weight, A_log, dt_bias, past_conv_state, cu_seq_lens,
        num_q_heads, num_kv_heads, d_k, d_v, conv_size, chunk_size, po, batch_size
    );

    c10::cuda::CUDAGuard device_guard(proj.device());

    int64_t total_L = proj.size(0);
    auto h_opts = proj.options();
    auto f_opts = proj.options().dtype(at::kFloat);

    at::Tensor q_h      = torch::empty({num_q_heads,  total_L, d_k}, h_opts);
    at::Tensor k_h      = torch::empty({num_kv_heads, total_L, d_k}, h_opts);
    at::Tensor v_h      = torch::empty({num_kv_heads, total_L, d_v}, h_opts);
    at::Tensor b_eff_h  = torch::empty({num_kv_heads, total_L, d_k}, h_opts);
    at::Tensor w_h      = torch::empty({num_kv_heads, total_L, d_v}, h_opts);
    at::Tensor G_log2_h = torch::empty({num_kv_heads, total_L, d_k}, f_opts);

    // Pre-seed final_conv_state so zero-length batch entries keep past state or zeros
    at::Tensor final_conv_state = output_conv_state
        ? (past_conv_state.has_value()
               ? past_conv_state->clone()
               : torch::zeros({batch_size, conv_size - 1, po.D_qkv}, h_opts))
        : at::empty({0}, h_opts);

    at::Tensor q_inv_norm = is_training ? torch::empty({num_q_heads,  total_L}, f_opts) : at::empty({0}, f_opts);
    at::Tensor k_inv_norm = is_training ? torch::empty({num_kv_heads, total_L}, f_opts) : at::empty({0}, f_opts);

    if (total_L == 0 || meta.num_chunks == 0) {
        return std::make_tuple(q_h, k_h, v_h, b_eff_h, w_h, G_log2_h, final_conv_state, q_inv_norm, k_inv_norm);
    }

    at::Tensor proj_a    = ensure_aligned16(proj);
    at::Tensor conv_w_a  = ensure_aligned16(conv_weight);
    at::Tensor A_log_a   = ensure_aligned16(A_log);
    at::Tensor dt_bias_a = ensure_aligned16(dt_bias);

    bool has_past_conv = past_conv_state.has_value();
    at::Tensor past_conv_a = has_past_conv ? ensure_aligned16(*past_conv_state) : at::Tensor();

    {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        const auto* proj_ptr      = reinterpret_cast<const cutlass::half_t*>(proj_a.data_ptr<at::Half>());
        const auto* conv_w_ptr    = reinterpret_cast<const cutlass::half_t*>(conv_w_a.data_ptr<at::Half>());
        const auto* past_conv_ptr = has_past_conv ? reinterpret_cast<const cutlass::half_t*>(past_conv_a.data_ptr<at::Half>()) : nullptr;

        auto* q_h_ptr        = reinterpret_cast<cutlass::half_t*>(q_h.data_ptr<at::Half>());
        auto* k_h_ptr        = reinterpret_cast<cutlass::half_t*>(k_h.data_ptr<at::Half>());
        auto* v_h_ptr        = reinterpret_cast<cutlass::half_t*>(v_h.data_ptr<at::Half>());
        auto* final_conv_ptr = output_conv_state ? reinterpret_cast<cutlass::half_t*>(final_conv_state.data_ptr<at::Half>()) : nullptr;
        float* q_inorm_ptr   = is_training ? q_inv_norm.data_ptr<float>() : nullptr;
        float* k_inorm_ptr   = is_training ? k_inv_norm.data_ptr<float>() : nullptr;

        const int32_t* c_off  = meta.chunk_offsets.data_ptr<int32_t>();
        const int32_t* c_vlen = meta.chunk_valid_lens.data_ptr<int32_t>();
        const int32_t* c_seq  = meta.chunk_seq_ids.data_ptr<int32_t>();
        const int32_t* c_loc  = meta.chunk_local_starts.data_ptr<int32_t>();
        const int32_t* c_last = meta.chunk_is_last.data_ptr<int32_t>();

        int k1_flags = (is_training ? 4 : 0) | (has_past_conv ? 2 : 0) | (output_conv_state ? 1 : 0);
        switch (k1_flags) {
            case 0: launch_conv_act_l2_fwd<false, false, false>(proj_ptr, conv_w_ptr, past_conv_ptr, c_off, c_vlen, c_seq, c_loc, c_last, q_h_ptr, k_h_ptr, v_h_ptr, final_conv_ptr, q_inorm_ptr, k_inorm_ptr, po.total_proj_dim, po.D_qkv, num_q_heads, num_kv_heads, total_L, meta.num_chunks, static_cast<float>(scale), stream); break;
            case 1: launch_conv_act_l2_fwd<false, false, true>(proj_ptr, conv_w_ptr, past_conv_ptr, c_off, c_vlen, c_seq, c_loc, c_last, q_h_ptr, k_h_ptr, v_h_ptr, final_conv_ptr, q_inorm_ptr, k_inorm_ptr, po.total_proj_dim, po.D_qkv, num_q_heads, num_kv_heads, total_L, meta.num_chunks, static_cast<float>(scale), stream); break;
            case 2: launch_conv_act_l2_fwd<false, true, false>(proj_ptr, conv_w_ptr, past_conv_ptr, c_off, c_vlen, c_seq, c_loc, c_last, q_h_ptr, k_h_ptr, v_h_ptr, final_conv_ptr, q_inorm_ptr, k_inorm_ptr, po.total_proj_dim, po.D_qkv, num_q_heads, num_kv_heads, total_L, meta.num_chunks, static_cast<float>(scale), stream); break;
            case 3: launch_conv_act_l2_fwd<false, true, true>(proj_ptr, conv_w_ptr, past_conv_ptr, c_off, c_vlen, c_seq, c_loc, c_last, q_h_ptr, k_h_ptr, v_h_ptr, final_conv_ptr, q_inorm_ptr, k_inorm_ptr, po.total_proj_dim, po.D_qkv, num_q_heads, num_kv_heads, total_L, meta.num_chunks, static_cast<float>(scale), stream); break;
            case 4: launch_conv_act_l2_fwd<true, false, false>(proj_ptr, conv_w_ptr, past_conv_ptr, c_off, c_vlen, c_seq, c_loc, c_last, q_h_ptr, k_h_ptr, v_h_ptr, final_conv_ptr, q_inorm_ptr, k_inorm_ptr, po.total_proj_dim, po.D_qkv, num_q_heads, num_kv_heads, total_L, meta.num_chunks, static_cast<float>(scale), stream); break;
            case 5: launch_conv_act_l2_fwd<true, false, true>(proj_ptr, conv_w_ptr, past_conv_ptr, c_off, c_vlen, c_seq, c_loc, c_last, q_h_ptr, k_h_ptr, v_h_ptr, final_conv_ptr, q_inorm_ptr, k_inorm_ptr, po.total_proj_dim, po.D_qkv, num_q_heads, num_kv_heads, total_L, meta.num_chunks, static_cast<float>(scale), stream); break;
            case 6: launch_conv_act_l2_fwd<true, true, false>(proj_ptr, conv_w_ptr, past_conv_ptr, c_off, c_vlen, c_seq, c_loc, c_last, q_h_ptr, k_h_ptr, v_h_ptr, final_conv_ptr, q_inorm_ptr, k_inorm_ptr, po.total_proj_dim, po.D_qkv, num_q_heads, num_kv_heads, total_L, meta.num_chunks, static_cast<float>(scale), stream); break;
            case 7: launch_conv_act_l2_fwd<true, true, true>(proj_ptr, conv_w_ptr, past_conv_ptr, c_off, c_vlen, c_seq, c_loc, c_last, q_h_ptr, k_h_ptr, v_h_ptr, final_conv_ptr, q_inorm_ptr, k_inorm_ptr, po.total_proj_dim, po.D_qkv, num_q_heads, num_kv_heads, total_L, meta.num_chunks, static_cast<float>(scale), stream); break;
        }
    }

    {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        const auto* proj_ptr    = reinterpret_cast<const cutlass::half_t*>(proj_a.data_ptr<at::Half>());
        const auto* k_h_ptr     = reinterpret_cast<const cutlass::half_t*>(k_h.data_ptr<at::Half>());
        const float* A_log_ptr  = A_log_a.data_ptr<float>();
        const float* dt_ptr     = dt_bias_a.data_ptr<float>();
        auto* b_eff_h_ptr       = reinterpret_cast<cutlass::half_t*>(b_eff_h.data_ptr<at::Half>());
        auto* w_h_ptr           = reinterpret_cast<cutlass::half_t*>(w_h.data_ptr<at::Half>());
        float* G_log2_ptr       = G_log2_h.data_ptr<float>();

        const int32_t* c_off  = meta.chunk_offsets.data_ptr<int32_t>();
        const int32_t* c_vlen = meta.chunk_valid_lens.data_ptr<int32_t>();

        int k2_flags = (allow_neg_eigval ? 2 : 0) | (beta_guard ? 1 : 0);
        switch (k2_flags) {
            case 0: launch_gates_decay_fwd<false, false>(proj_ptr, k_h_ptr, A_log_ptr, dt_ptr, c_off, c_vlen, b_eff_h_ptr, w_h_ptr, G_log2_ptr, po, num_kv_heads, total_L, meta.num_chunks, stream); break;
            case 1: launch_gates_decay_fwd<false, true>(proj_ptr, k_h_ptr, A_log_ptr, dt_ptr, c_off, c_vlen, b_eff_h_ptr, w_h_ptr, G_log2_ptr, po, num_kv_heads, total_L, meta.num_chunks, stream); break;
            case 2: launch_gates_decay_fwd<true, false>(proj_ptr, k_h_ptr, A_log_ptr, dt_ptr, c_off, c_vlen, b_eff_h_ptr, w_h_ptr, G_log2_ptr, po, num_kv_heads, total_L, meta.num_chunks, stream); break;
            case 3: launch_gates_decay_fwd<true, true>(proj_ptr, k_h_ptr, A_log_ptr, dt_ptr, c_off, c_vlen, b_eff_h_ptr, w_h_ptr, G_log2_ptr, po, num_kv_heads, total_L, meta.num_chunks, stream); break;
        }
    }

    return std::make_tuple(q_h, k_h, v_h, b_eff_h, w_h, G_log2_h, final_conv_state, q_inv_norm, k_inv_norm);
}

// Outputs: (q_h, k_h, v_h, b_eff_h, w_h, G_log2_h [FP32], final_conv_state, q_inv_norm, k_inv_norm)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
proj_conv_prep_fwd(
    const at::Tensor& proj,
    const at::Tensor& conv_weight,
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    const std::optional<at::Tensor>& past_conv_state,
    const at::Tensor& cu_seq_lens,
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
) {
    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64");
    TORCH_CHECK(cu_seq_lens.is_cuda() && cu_seq_lens.dim() == 1 && cu_seq_lens.size(0) >= 1,
                "cu_seq_lens must be a 1D CUDA tensor with size >= 1");
    VarlenChunkMeta meta = compute_varlen_chunk_metadata(cu_seq_lens, chunk_size);
    return proj_conv_prep_fwd_with_meta(
        proj, conv_weight, A_log, dt_bias, past_conv_state, cu_seq_lens, meta,
        num_q_heads, num_kv_heads, d_k, d_v, conv_size, chunk_size,
        scale, allow_neg_eigval, beta_guard, is_training, output_conv_state
    );
}
