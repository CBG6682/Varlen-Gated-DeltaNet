#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <tuple>

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm75.hpp>
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
// Part 1: Global Kernel (kernel_output_combine_fwd)
// ============================================================================

template <
    typename T,
    int NUM_THREADS,
    int CHUNK_SIZE,
    int SUB_CHUNK,
    int D_K,
    int D_V,
    bool IS_TRAINING,
    typename MMAAtom,
    typename TiledMMA16,
    typename TiledMMA221,
    typename S2RCopyAtomA_KM,
    typename S2RCopyAtomB_KM,
    typename S2RCopyAtomB_MNM,
    typename TiledCopyHalf,
    typename TiledCopyFloat
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_output_combine_fwd(
    const T*       __restrict__ gQ_h,             // (H_q, total_L, 64) [FP16] - Scaled normalized Query
    const T*       __restrict__ gK_h,             // (H_kv, total_L, 64) [FP16] - Normalized Key
    const float*   __restrict__ gG_log2_h,        // (H_kv, total_L, 64) [FP32] - Chunk cumsum log2 decay
    const float*   __restrict__ gS_list,          // (H_kv, num_chunks, 64, 64) [FP32] - Chunk initial states
    const T*       __restrict__ gR_list,          // (H_kv, num_chunks, 64, 64) [FP16] - Residual matrix R
    const T*       __restrict__ gProj,            // (total_L, total_proj_dim) [FP16] - Unified projection
    const T*       __restrict__ gONormW,          // (64,) [FP16] - Per-head RMSNorm weight
    const int32_t* __restrict__ chunk_offsets,    // (num_chunks + 1,) [Int32] - Chunk token starts
    const int32_t* __restrict__ chunk_valid_lens, // (num_chunks,) [Int32] - Valid tokens per chunk
    T*             __restrict__ gOutTok,          // (total_L, H_q * 64) [FP16] - Output (head-last)
    T*             __restrict__ gO_raw,           // (H_q, total_L, 64) [FP16] - Saved pre-norm output (nullable)
    float*         __restrict__ gO_rstd,          // (H_q, total_L) [FP32] - Saved per-head rstd (nullable)
    int64_t num_q_heads,                          // scalar [int64] - Total Query heads
    int64_t num_kv_heads,                         // scalar [int64] - Total KV heads
    int64_t num_groups,                           // scalar [int64] - Query heads per KV group
    int64_t num_chunks,                           // scalar [int64] - Total varlen chunks
    int64_t total_L,                              // scalar [int64] - Total packed tokens
    int64_t total_proj_dim,                       // scalar [int64] - Stride of gProj
    int64_t offset_ogate,                         // scalar [int64] - Column start of o_gate in gProj
    float   eps                                   // scalar [float] - RMSNorm epsilon
) {
    // 1. Setup & SMEM Allocation (exact 32.0 KB)
    int h_kv       = blockIdx.x;
    int curr_chunk = blockIdx.y;
    int tid        = threadIdx.x;

    if (h_kv >= num_kv_heads || curr_chunk >= num_chunks) return;

    int32_t valid_len   = chunk_valid_lens[curr_chunk];
    int32_t chunk_start = chunk_offsets[curr_chunk];
    if (valid_len <= 0) return;

    int row_id   = tid / 8;
    int col_id   = tid % 8;
    int col_base = col_id * 8;

    int last_sub       = (valid_len - 1) / SUB_CHUNK;
    int last_sub_valid = valid_len - last_sub * SUB_CHUNK;

    struct alignas(16) SmemStorage {
        T smem_1[CHUNK_SIZE * CHUNK_SIZE];
        union {
            T q_gamma[CHUNK_SIZE * D_K];
            struct {
                T     q_sub[SUB_CHUNK * D_K];
                T     k_sub[SUB_CHUNK * D_K];
                float G_sub[SUB_CHUNK * D_K];
            } sub;
        } smem_2;
        T smem_3[D_V * D_K];
        T smem_4[CHUNK_SIZE * D_V];
    };
    static_assert(sizeof(SmemStorage) == 32768, "SmemStorage must be exactly 32 KB (32768 bytes)");
    __shared__ SmemStorage smem;

    using SwizzleHalf = Swizzle<3, 3, 3>;
    using SwizzleF32  = Swizzle<3, 2, 3>;

    auto layout_64x64_km  = composition(SwizzleHalf{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto layout_64x64_f32 = composition(SwizzleF32{},  make_layout(make_shape(Int<D_V>{}, Int<D_K>{}),        make_stride(Int<D_K>{}, Int<1>{})));
    auto layout_64x64_mnm = composition(SwizzleHalf{}, make_layout(make_shape(Int<D_V>{}, Int<CHUNK_SIZE>{}), make_stride(Int<1>{}, Int<D_V>{})));
    auto layout_sub_half  = composition(SwizzleHalf{}, make_layout(make_shape(Int<SUB_CHUNK>{}, Int<D_K>{}),  make_stride(Int<D_K>{}, Int<1>{})));
    auto layout_sub_f32   = composition(SwizzleF32{},  make_layout(make_shape(Int<SUB_CHUNK>{}, Int<D_K>{}),  make_stride(Int<D_K>{}, Int<1>{})));

    auto s1_flat   = make_tensor(make_smem_ptr(smem.smem_1),           layout_64x64_km);
    auto s2_qgamma = make_tensor(make_smem_ptr(smem.smem_2.q_gamma),   layout_64x64_km);
    auto sSubQ     = make_tensor(make_smem_ptr(smem.smem_2.sub.q_sub), layout_sub_half);
    auto sSubK     = make_tensor(make_smem_ptr(smem.smem_2.sub.k_sub), layout_sub_half);
    auto sSubG     = make_tensor(make_smem_ptr(smem.smem_2.sub.G_sub), layout_sub_f32);
    auto s3_S      = make_tensor(make_smem_ptr(smem.smem_3),           layout_64x64_km);
    auto s4_R_flat = make_tensor(make_smem_ptr(smem.smem_4),           layout_64x64_km);
    auto s4_R_mnm  = make_tensor(make_smem_ptr(smem.smem_4),           layout_64x64_mnm);

    TiledCopyHalf  tiled_copy_h;
    TiledCopyFloat tiled_copy_f32;
    auto thr_copy_h   = tiled_copy_h.get_slice(tid);
    auto thr_copy_f32 = tiled_copy_f32.get_slice(tid);

    auto tXs1_s    = thr_copy_h.partition_S(s1_flat);
    auto tXs1_d    = thr_copy_h.partition_D(s1_flat);
    auto tXs2_d    = thr_copy_h.partition_D(s2_qgamma);
    auto tXsSubQ_d = thr_copy_h.partition_D(sSubQ);
    auto tXsSubK_d = thr_copy_h.partition_D(sSubK);
    auto tXsSubG_d = thr_copy_f32.partition_D(sSubG);
    auto tXs3_d    = thr_copy_h.partition_D(s3_S);
    auto tXs4_d    = thr_copy_h.partition_D(s4_R_flat);

    auto tCrBuf0 = make_fragment_like(tXs1_d);
    auto tCrBuf1 = make_fragment_like(tXs1_d);
    auto tCrSubA = make_fragment_like(tXsSubQ_d);
    auto tCrSubB = make_fragment_like(tXsSubK_d);

    auto cTile64     = make_identity_tensor(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}));
    auto tXcHalf64   = thr_copy_h.partition_S(cTile64);
    auto tXcF32_64   = thr_copy_f32.partition_S(cTile64);
    auto tCrG64      = make_fragment_like<float>(tXcF32_64);
    auto tPredHalf64 = make_fragment_like<bool>(tCrBuf0);
    auto tPredF32_64 = make_fragment_like<bool>(tCrG64);

    if (valid_len < CHUNK_SIZE) {
        #pragma unroll
        for (int m = 0; m < size<1>(tPredHalf64); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tPredHalf64); ++n) {
                bool in_bounds = (get<0>(tXcHalf64(Int<0>{}, m, n)) < valid_len);
                #pragma unroll
                for (int v = 0; v < size<0>(tPredHalf64); ++v) {
                    tPredHalf64(v, m, n) = in_bounds;
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < size<1>(tPredF32_64); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tPredF32_64); ++n) {
                bool in_bounds = (get<0>(tXcF32_64(Int<0>{}, m, n)) < valid_len);
                #pragma unroll
                for (int v = 0; v < size<0>(tPredF32_64); ++v) {
                    tPredF32_64(v, m, n) = in_bounds;
                }
            }
        }
    }

    auto cSub16      = make_identity_tensor(make_shape(Int<SUB_CHUNK>{}, Int<D_K>{}));
    auto tXcHalfSub  = thr_copy_h.partition_S(cSub16);
    auto tXcF32Sub   = thr_copy_f32.partition_S(cSub16);
    auto tCrGSub     = make_fragment_like<float>(tXcF32Sub);
    auto tPredHalfSub = make_fragment_like<bool>(tCrSubA);
    auto tPredF32Sub  = make_fragment_like<bool>(tCrGSub);

    if (last_sub_valid < SUB_CHUNK) {
        #pragma unroll
        for (int m = 0; m < size<1>(tPredHalfSub); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tPredHalfSub); ++n) {
                bool in_bounds = (get<0>(tXcHalfSub(Int<0>{}, m, n)) < last_sub_valid);
                #pragma unroll
                for (int v = 0; v < size<0>(tPredHalfSub); ++v) {
                    tPredHalfSub(v, m, n) = in_bounds;
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < size<1>(tPredF32Sub); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tPredF32Sub); ++n) {
                bool in_bounds = (get<0>(tXcF32Sub(Int<0>{}, m, n)) < last_sub_valid);
                #pragma unroll
                for (int v = 0; v < size<0>(tPredF32Sub); ++v) {
                    tPredF32Sub(v, m, n) = in_bounds;
                }
            }
        }
    }

    // Broadcast o_norm_weight 1D vector via L1
    const uint4 ow_u4 = *reinterpret_cast<const uint4*>(gONormW + col_base);
    const __half* ow_h = reinterpret_cast<const __half*>(&ow_u4);
    float r_ow[8];
    #pragma unroll
    for (int v = 0; v < 8; ++v) {
        r_ow[v] = __half2float(ow_h[v]);
    }

    auto layout_64x64_gmem = make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{}));
    auto layout_sub_gmem   = make_layout(make_shape(Int<SUB_CHUNK>{},  Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{}));

    int64_t kv_tok_base = (static_cast<int64_t>(h_kv) * total_L + chunk_start) * D_K;
    int64_t kv_mat_off  = (static_cast<int64_t>(h_kv) * num_chunks + curr_chunk) * (CHUNK_SIZE * D_K);

    // 2. Prologue: load stationary S_fp16 into smem_3 and R_c into smem_4
    {
        auto gS_in = make_tensor(make_gmem_ptr(gS_list + kv_mat_off), layout_64x64_gmem);
        auto gR_in = make_tensor(make_gmem_ptr(gR_list + kv_mat_off), layout_64x64_gmem);

        cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gS_in), tCrG64);
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                tCrBuf0(v, m, Int<0>{}) = float_to_out<T>(tCrG64(v, m, Int<0>{}));
            }
        }
        #pragma unroll
        for (int m = 0; m < size<1>(tXs3_d); ++m) {
            *reinterpret_cast<uint4*>(&tXs3_d(Int<0>{}, m, Int<0>{})) =
                *reinterpret_cast<const uint4*>(&tCrBuf0(Int<0>{}, m, Int<0>{}));
        }

        cute::copy(tiled_copy_h, thr_copy_h.partition_S(gR_in), tCrBuf0);
        #pragma unroll
        for (int m = 0; m < size<1>(tXs4_d); ++m) {
            *reinterpret_cast<uint4*>(&tXs4_d(Int<0>{}, m, Int<0>{})) =
                *reinterpret_cast<const uint4*>(&tCrBuf0(Int<0>{}, m, Int<0>{}));
        }
    }
    __syncthreads();

    // 3. Sequential loop across Query heads in GQA group
    for (int64_t g = 0; g < num_groups; ++g) {
        int64_t h_q        = static_cast<int64_t>(h_kv) * num_groups + g;
        int64_t q_tok_base = (h_q * total_L + chunk_start) * D_K;

        // 3.1 Zero smem_1 before assembling Aqk
        clear(tCrBuf0);
        #pragma unroll
        for (int m = 0; m < size<1>(tXs1_d); ++m) {
            *reinterpret_cast<uint4*>(&tXs1_d(Int<0>{}, m, Int<0>{})) =
                *reinterpret_cast<const uint4*>(&tCrBuf0(Int<0>{}, m, Int<0>{}));
        }

        // 3.2 Diagonal 16x16 blocks of Aqk (inclusive causal j <= row_id)
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            int sub_start = s * SUB_CHUNK;
            if (sub_start >= valid_len) break;

            int     sub_valid   = (s == last_sub) ? last_sub_valid : SUB_CHUNK;
            bool    is_full_sub = (sub_valid == SUB_CHUNK);
            int64_t q_sub_off   = q_tok_base  + static_cast<int64_t>(sub_start) * D_K;
            int64_t kv_sub_off  = kv_tok_base + static_cast<int64_t>(sub_start) * D_K;

            auto gQs = make_tensor(make_gmem_ptr(gQ_h      + q_sub_off),  layout_sub_gmem);
            auto gKs = make_tensor(make_gmem_ptr(gK_h      + kv_sub_off), layout_sub_gmem);
            auto gGs = make_tensor(make_gmem_ptr(gG_log2_h + kv_sub_off), layout_sub_gmem);

            if (is_full_sub) {
                cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gQs),   tCrSubA);
                cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gKs),   tCrSubB);
                cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gGs), tCrGSub);
            } else {
                clear(tCrSubA);
                clear(tCrSubB);
                clear(tCrGSub);
                cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gQs),   tCrSubA);
                cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gKs),   tCrSubB);
                cute::copy_if(tiled_copy_f32, tPredF32Sub,  thr_copy_f32.partition_S(gGs), tCrGSub);
            }

            *reinterpret_cast<uint4*>(&tXsSubK_d(Int<0>{}, Int<0>{}, Int<0>{})) =
                *reinterpret_cast<const uint4*>(&tCrSubB(Int<0>{}, Int<0>{}, Int<0>{}));
            #pragma unroll
            for (int x = 0; x < 2; ++x) {
                *reinterpret_cast<uint4*>(&tXsSubG_d(make_coord(Int<0>{}, x), Int<0>{}, Int<0>{})) =
                    *reinterpret_cast<const uint4*>(&tCrGSub(make_coord(Int<0>{}, x), Int<0>{}, Int<0>{}));
            }
            __syncthreads();

            float q_r[8];
            float g_r[8];
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                q_r[v] = (row_id < sub_valid) ? half_to_float(tCrSubA(v, Int<0>{}, Int<0>{})) : 0.0f;
                g_r[v] = (row_id < sub_valid) ? tCrGSub(v, Int<0>{}, Int<0>{}) : 0.0f;
            }

            for (int j = 0; j < SUB_CHUNK; ++j) {
                float partial = 0.0f;
                if (j <= row_id && row_id < sub_valid) {
                    #pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        partial += q_r[v] *
                                   half_to_float(sSubK(j, col_base + v)) *
                                   exp2f(g_r[v] - sSubG(j, col_base + v));
                    }
                }
                float a_rj = octet_reduce_sum(partial);
                if (col_id == 0 && j <= row_id && row_id < sub_valid) {
                    s1_flat(sub_start + row_id, sub_start + j) = float_to_out<T>(a_rj);
                }
            }
            __syncthreads();
        }

        // 3.3 Off-diagonal 16x16 blocks of Aqk via overflow-safe g_ref scaling
        {
            TiledMMA16 tiled_mma_16;
            auto thr_mma_16   = tiled_mma_16.get_slice(tid & 63);
            auto s2r_copy_a16 = make_tiled_copy_A(S2RCopyAtomA_KM{}, tiled_mma_16);
            auto s2r_copy_b16 = make_tiled_copy_B(S2RCopyAtomB_KM{}, tiled_mma_16);
            auto s2r_thr_a16  = s2r_copy_a16.get_slice(tid & 63);
            auto s2r_thr_b16  = s2r_copy_b16.get_slice(tid & 63);

            auto tCsA_16    = s2r_thr_a16.partition_S(sSubQ);
            auto tCsB_16    = s2r_thr_b16.partition_S(sSubK);
            auto tCrA_16    = thr_mma_16.partition_fragment_A(sSubQ);
            auto tCrB_16    = thr_mma_16.partition_fragment_B(sSubK);
            auto tCrA_16_vw = s2r_thr_a16.retile_D(tCrA_16);
            auto tCrB_16_vw = s2r_thr_b16.retile_D(tCrB_16);
            auto tCsC_16_00 = thr_mma_16.partition_C(
                local_tile(s1_flat, make_shape(Int<SUB_CHUNK>{}, Int<SUB_CHUNK>{}), make_coord(Int<0>{}, Int<0>{}))
            );
            auto tCrC_16    = thr_mma_16.make_fragment_C(tCsC_16_00);
            auto rC_half_16 = make_tensor_like<T>(tCrC_16);

            for (int i = 1; i < 4; ++i) {
                int i_start = i * SUB_CHUNK;
                if (i_start >= valid_len) break;

                int     i_valid   = (i == last_sub) ? last_sub_valid : SUB_CHUNK;
                bool    is_full_i = (i_valid == SUB_CHUNK);
                int64_t q_i_off   = q_tok_base  + static_cast<int64_t>(i_start) * D_K;
                int64_t kv_i_off  = kv_tok_base + static_cast<int64_t>(i_start) * D_K;

                // Anchor log2 decay at first token of target sub-chunk i so exp2 args remain <= 0
                const float* p_g_ref = gG_log2_h + kv_i_off + col_base;
                float4 gr0 = *reinterpret_cast<const float4*>(p_g_ref);
                float4 gr1 = *reinterpret_cast<const float4*>(p_g_ref + 4);
                float r_gref[8] = {gr0.x, gr0.y, gr0.z, gr0.w, gr1.x, gr1.y, gr1.z, gr1.w};

                auto gQi = make_tensor(make_gmem_ptr(gQ_h      + q_i_off),  layout_sub_gmem);
                auto gGi = make_tensor(make_gmem_ptr(gG_log2_h + kv_i_off), layout_sub_gmem);

                if (is_full_i) {
                    cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gQi),   tCrSubA);
                    cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gGi), tCrGSub);
                } else {
                    clear(tCrSubA);
                    clear(tCrGSub);
                    cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gQi),   tCrSubA);
                    cute::copy_if(tiled_copy_f32, tPredF32Sub,  thr_copy_f32.partition_S(gGi), tCrGSub);
                }

                #pragma unroll
                for (int v = 0; v < 8; ++v) {
                    float val = (row_id < i_valid)
                        ? (half_to_float(tCrSubA(v, Int<0>{}, Int<0>{})) *
                           exp2f(tCrGSub(v, Int<0>{}, Int<0>{}) - r_gref[v]))
                        : 0.0f;
                    tCrSubA(v, Int<0>{}, Int<0>{}) = float_to_out<T>(val);
                }
                *reinterpret_cast<uint4*>(&tXsSubQ_d(Int<0>{}, Int<0>{}, Int<0>{})) =
                    *reinterpret_cast<const uint4*>(&tCrSubA(Int<0>{}, Int<0>{}, Int<0>{}));
                __syncthreads();

                if (tid < 64) {
                    cute::copy(s2r_copy_a16, tCsA_16, tCrA_16_vw);
                }

                for (int j = 0; j < i; ++j) {
                    int64_t kv_j_off = kv_tok_base + static_cast<int64_t>(j * SUB_CHUNK) * D_K;
                    auto gKj = make_tensor(make_gmem_ptr(gK_h      + kv_j_off), layout_sub_gmem);
                    auto gGj = make_tensor(make_gmem_ptr(gG_log2_h + kv_j_off), layout_sub_gmem);

                    cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gKj),   tCrSubB);
                    cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gGj), tCrGSub);

                    #pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        float val = half_to_float(tCrSubB(v, Int<0>{}, Int<0>{})) *
                                    exp2f(r_gref[v] - tCrGSub(v, Int<0>{}, Int<0>{}));
                        tCrSubB(v, Int<0>{}, Int<0>{}) = float_to_out<T>(val);
                    }
                    *reinterpret_cast<uint4*>(&tXsSubK_d(Int<0>{}, Int<0>{}, Int<0>{})) =
                        *reinterpret_cast<const uint4*>(&tCrSubB(Int<0>{}, Int<0>{}, Int<0>{}));
                    __syncthreads();

                    if (tid < 64) {
                        cute::copy(s2r_copy_b16, tCsB_16, tCrB_16_vw);
                        clear(tCrC_16);
                        cute::gemm(MMAAtom{}, tCrA_16, tCrB_16, tCrC_16);

                        #pragma unroll
                        for (int idx = 0; idx < size(tCrC_16); ++idx) {
                            rC_half_16(idx) = float_to_out<T>(tCrC_16(idx));
                        }
                        auto s1_ij = local_tile(s1_flat, make_shape(Int<SUB_CHUNK>{}, Int<SUB_CHUNK>{}), make_coord(i, j));
                        cute::copy(rC_half_16, thr_mma_16.partition_C(s1_ij));
                    }
                    __syncthreads();
                }
            }
        }

        // 3.4 Build Q_gamma = q_g * exp2(G_log2) into smem_2.q_gamma
        {
            auto gQg = make_tensor(make_gmem_ptr(gQ_h      + q_tok_base),  layout_64x64_gmem);
            auto gGg = make_tensor(make_gmem_ptr(gG_log2_h + kv_tok_base), layout_64x64_gmem);

            if (valid_len == CHUNK_SIZE) {
                cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gQg),   tCrBuf0);
                cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gGg), tCrG64);
            } else {
                clear(tCrBuf0);
                clear(tCrG64);
                cute::copy_if(tiled_copy_h,   tPredHalf64, thr_copy_h.partition_S(gQg),   tCrBuf0);
                cute::copy_if(tiled_copy_f32, tPredF32_64, thr_copy_f32.partition_S(gGg), tCrG64);
            }

            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                int  local_row = m * 16 + row_id;
                bool row_valid = (local_row < valid_len);
                #pragma unroll
                for (int v = 0; v < 8; ++v) {
                    float val = row_valid
                        ? (half_to_float(tCrBuf0(v, m, Int<0>{})) * exp2f(tCrG64(v, m, Int<0>{})))
                        : 0.0f;
                    tCrBuf0(v, m, Int<0>{}) = float_to_out<T>(val);
                }
            }
            #pragma unroll
            for (int m = 0; m < size<1>(tXs2_d); ++m) {
                *reinterpret_cast<uint4*>(&tXs2_d(Int<0>{}, m, Int<0>{})) =
                    *reinterpret_cast<const uint4*>(&tCrBuf0(Int<0>{}, m, Int<0>{}));
            }
        }
        __syncthreads();

        // 3.5 GEMM 1 (Q_gamma @ S^T) + GEMM 2 (+= Aqk @ R_c) chained in registers
        {
            TiledMMA221 tiled_mma_221;
            auto thr_mma_221     = tiled_mma_221.get_slice(tid);
            auto s2r_copy_a221   = make_tiled_copy_A(S2RCopyAtomA_KM{},  tiled_mma_221);
            auto s2r_copy_b221_k = make_tiled_copy_B(S2RCopyAtomB_KM{},  tiled_mma_221);
            auto s2r_copy_b221_m = make_tiled_copy_B(S2RCopyAtomB_MNM{}, tiled_mma_221);

            auto s2r_thr_a221   = s2r_copy_a221.get_slice(tid);
            auto s2r_thr_b221_k = s2r_copy_b221_k.get_slice(tid);
            auto s2r_thr_b221_m = s2r_copy_b221_m.get_slice(tid);

            auto tCsA_Q   = s2r_thr_a221.partition_S(s2_qgamma);
            auto tCsA_Aqk = s2r_thr_a221.partition_S(s1_flat);
            auto tCsB_S   = s2r_thr_b221_k.partition_S(s3_S);
            auto tCsB_R   = s2r_thr_b221_m.partition_S(s4_R_mnm);
            auto tCsC_1   = thr_mma_221.partition_C(s1_flat);

            auto tCrA         = thr_mma_221.partition_fragment_A(s2_qgamma);
            auto tCrB         = thr_mma_221.partition_fragment_B(s3_S);
            auto tCrA_view    = s2r_thr_a221.retile_D(tCrA);
            auto tCrB_km_view = s2r_thr_b221_k.retile_D(tCrB);
            auto tCrB_mn_view = s2r_thr_b221_m.retile_D(tCrB);
            auto tCrC_O       = thr_mma_221.make_fragment_C(tCsC_1);
            auto rC_half_O    = make_tensor_like<T>(tCrC_O);

            clear(tCrC_O);
            cute::copy(s2r_copy_a221,   tCsA_Q, tCrA_view);
            cute::copy(s2r_copy_b221_k, tCsB_S, tCrB_km_view);
            cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC_O);

            cute::copy(s2r_copy_a221,   tCsA_Aqk, tCrA_view);
            cute::copy(s2r_copy_b221_m, tCsB_R,   tCrB_mn_view);
            cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC_O);
            __syncthreads();

            #pragma unroll
            for (int i = 0; i < size(tCrC_O); ++i) {
                rC_half_O(i) = float_to_out<T>(tCrC_O(i));
            }
            cute::copy(rC_half_O, tCsC_1);
        }
        __syncthreads();

        // 3.6 Fused epilogue: Per-Head RMSNorm * SiLU(o_gate) + head-last transpose store
        cute::copy(tiled_copy_h, tXs1_s, tCrBuf0);

        if constexpr (IS_TRAINING) {
            auto gORaw_tile = make_tensor(
                make_gmem_ptr(gO_raw + (h_q * total_L + chunk_start) * D_V),
                make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{}))
            );
            if (valid_len == CHUNK_SIZE) {
                cute::copy(tiled_copy_h, tCrBuf0, thr_copy_h.partition_D(gORaw_tile));
            } else {
                cute::copy_if(tiled_copy_h, tPredHalf64, tCrBuf0, thr_copy_h.partition_D(gORaw_tile));
            }
        }

        auto gOGate_tile = make_tensor(
            make_gmem_ptr(gProj + static_cast<int64_t>(chunk_start) * total_proj_dim + offset_ogate + h_q * D_V),
            make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(total_proj_dim, Int<1>{}))
        );
        if (valid_len == CHUNK_SIZE) {
            cute::copy(tiled_copy_h, thr_copy_h.partition_S(gOGate_tile), tCrBuf1);
        } else {
            clear(tCrBuf1);
            cute::copy_if(tiled_copy_h, tPredHalf64, thr_copy_h.partition_S(gOGate_tile), tCrBuf1);
        }

        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            int   local_row = m * 16 + row_id;
            float o_f[8];
            float local_sq = 0.0f;

            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                o_f[v] = half_to_float(tCrBuf0(v, m, Int<0>{}));
                local_sq += o_f[v] * o_f[v];
            }

            float sum_sq = octet_reduce_sum(local_sq);
            float rstd   = rsqrtf(sum_sq * (1.0f / static_cast<float>(D_V)) + eps);

            if constexpr (IS_TRAINING) {
                if (col_id == 0 && local_row < valid_len) {
                    gO_rstd[h_q * total_L + chunk_start + local_row] = rstd;
                }
            }

            #pragma unroll
            for (int v = 0; v < 8; v += 2) {
                __half2 on_h2  = __floats2half2_rn(o_f[v] * rstd * r_ow[v], o_f[v + 1] * rstd * r_ow[v + 1]);
                __half2 og_h2  = __halves2half2(to_half_val(tCrBuf1(v, m, Int<0>{})), to_half_val(tCrBuf1(v + 1, m, Int<0>{})));
                __half2 res_h2 = __hmul2(on_h2, fast_silu_h2(og_h2));
                tCrBuf0(v, m, Int<0>{})     = half_to_out<T>(__low2half(res_h2));
                tCrBuf0(v + 1, m, Int<0>{}) = half_to_out<T>(__high2half(res_h2));
            }
        }

        int64_t tok_stride = num_q_heads * D_V;
        auto gOutTok_tile = make_tensor(
            make_gmem_ptr(gOutTok + static_cast<int64_t>(chunk_start) * tok_stride + h_q * D_V),
            make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(tok_stride, Int<1>{}))
        );
        if (valid_len == CHUNK_SIZE) {
            cute::copy(tiled_copy_h, tCrBuf0, thr_copy_h.partition_D(gOutTok_tile));
        } else {
            cute::copy_if(tiled_copy_h, tPredHalf64, tCrBuf0, thr_copy_h.partition_D(gOutTok_tile));
        }

        if (g + 1 < num_groups) {
            __syncthreads();
        }
    }
}

// ============================================================================
// Part 2: Host Dispatch Functions
// ============================================================================

template <bool IS_TRAINING>
inline void launch_output_combine_fwd(
    const cutlass::half_t* p_q,
    const cutlass::half_t* p_k,
    const float*           p_G,
    const float*           p_S,
    const cutlass::half_t* p_R,
    const cutlass::half_t* p_proj,
    const cutlass::half_t* p_ow,
    const int32_t*         p_c_off,
    const int32_t*         p_c_vlen,
    cutlass::half_t*       p_out_tok,
    cutlass::half_t*       p_O_raw,
    float*                 p_o_rstd,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t num_groups,
    int64_t num_chunks,
    int64_t total_L,
    int64_t total_proj_dim,
    int64_t offset_ogate,
    float   eps,
    cudaStream_t stream
) {
    constexpr int NUM_THREADS = 128;
    constexpr int CHUNK_CONST = 64;
    constexpr int SUB_CONST   = 16;
    constexpr int DK_CONST    = 64;
    constexpr int DV_CONST    = 64;

    using Element     = cutlass::half_t;
    using MMAAtom     = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
    using TiledMMA16  = TiledMMA<MMAAtom, Layout<Shape<_1, _2, _1>, Stride<_1, _1, _2>>>;
    using TiledMMA221 = TiledMMA<MMAAtom, Layout<Shape<_2, _2, _1>, Stride<_1, _2, _4>>>;

    using S2RCopyAtomA_KM  = Copy_Atom<SM75_U32x2_LDSM_N, Element>;
    using S2RCopyAtomB_KM  = Copy_Atom<SM75_U32x1_LDSM_N, Element>;
    using S2RCopyAtomB_MNM = Copy_Atom<SM75_U16x2_LDSM_T, Element>;

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

    dim3 grid(static_cast<unsigned int>(num_kv_heads), static_cast<unsigned int>(num_chunks));
    dim3 block(NUM_THREADS);

    kernel_output_combine_fwd<
        Element, NUM_THREADS, CHUNK_CONST, SUB_CONST, DK_CONST, DV_CONST, IS_TRAINING,
        MMAAtom, TiledMMA16, TiledMMA221,
        S2RCopyAtomA_KM, S2RCopyAtomB_KM, S2RCopyAtomB_MNM,
        TiledCopyHalf, TiledCopyFloat
    ><<<grid, block, 0, stream>>>(
        p_q, p_k, p_G, p_S, p_R, p_proj, p_ow, p_c_off, p_c_vlen,
        p_out_tok, p_O_raw, p_o_rstd,
        num_q_heads, num_kv_heads, num_groups, num_chunks, total_L,
        total_proj_dim, offset_ogate, eps
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
) {
    TORCH_CHECK(d_k == 64 && d_v == 64, "d_k and d_v must be 64");
    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64");
    TORCH_CHECK(num_kv_heads > 0 && num_q_heads > 0 && (num_q_heads % num_kv_heads == 0),
                "num_q_heads must be positive and divisible by num_kv_heads");

    TORCH_CHECK(q_h.is_cuda(), "q_h must be a CUDA tensor");
    const c10::cuda::CUDAGuard device_guard(q_h.device());

    auto check_t = [&](const at::Tensor& t, const char* name, int64_t expected_dim, at::ScalarType dtype) {
        TORCH_CHECK(t.is_cuda() && t.device() == q_h.device(), name, " must be on the same CUDA device as q_h");
        TORCH_CHECK(t.scalar_type() == dtype, name, " dtype mismatch");
        TORCH_CHECK(t.dim() == expected_dim && t.is_contiguous(), name, " must be contiguous with expected rank");
    };

    check_t(q_h,           "q_h",           3, at::ScalarType::Half);
    check_t(k_h,           "k_h",           3, at::ScalarType::Half);
    check_t(G_log2_h,      "G_log2_h",      3, at::ScalarType::Float);
    check_t(S_list,        "S_list",        4, at::ScalarType::Float);
    check_t(R_list,        "R_list",        4, at::ScalarType::Half);
    check_t(proj,          "proj",          2, at::ScalarType::Half);
    check_t(o_norm_weight, "o_norm_weight", 1, at::ScalarType::Half);

    int64_t total_L        = q_h.size(1);
    int64_t num_chunks     = meta.num_chunks;
    int64_t num_groups     = num_q_heads / num_kv_heads;
    int64_t offset_ogate   = num_q_heads * d_k + 3 * num_kv_heads * d_k + 2 * num_kv_heads * d_v;
    int64_t total_proj_dim = offset_ogate + num_q_heads * d_v;

    TORCH_CHECK(q_h.size(0) == num_q_heads && q_h.size(2) == d_k, "q_h shape mismatch");
    TORCH_CHECK(k_h.size(0) == num_kv_heads && k_h.size(1) == total_L && k_h.size(2) == d_k, "k_h shape mismatch");
    TORCH_CHECK(G_log2_h.size(0) == num_kv_heads && G_log2_h.size(1) == total_L && G_log2_h.size(2) == d_k,
                "G_log2_h shape mismatch");
    TORCH_CHECK(S_list.size(0) == num_kv_heads && S_list.size(1) == num_chunks &&
                S_list.size(2) == d_v && S_list.size(3) == d_k, "S_list shape mismatch");
    TORCH_CHECK(R_list.size(0) == num_kv_heads && R_list.size(1) == num_chunks &&
                R_list.size(2) == chunk_size && R_list.size(3) == d_v, "R_list shape mismatch");
    TORCH_CHECK(proj.size(0) == total_L && proj.size(1) == total_proj_dim, "proj shape mismatch");
    TORCH_CHECK(o_norm_weight.size(0) == d_v, "o_norm_weight shape mismatch");

    auto h_opts = q_h.options();
    auto f_opts = G_log2_h.options();

    if (num_chunks <= 0 || total_L == 0) {
        at::Tensor out_tok = at::empty({total_L, num_q_heads * d_v}, h_opts);
        at::Tensor O_raw   = is_training ? at::empty({num_q_heads, total_L, d_v}, h_opts) : at::empty({0}, h_opts);
        at::Tensor o_rstd  = is_training ? at::empty({num_q_heads, total_L}, f_opts)      : at::empty({0}, f_opts);
        return std::make_tuple(out_tok, O_raw, o_rstd);
    }

    TORCH_CHECK(meta.chunk_offsets.is_cuda() && meta.chunk_offsets.device() == q_h.device() &&
                meta.chunk_offsets.scalar_type() == at::ScalarType::Int &&
                meta.chunk_offsets.dim() == 1 && meta.chunk_offsets.is_contiguous() &&
                meta.chunk_offsets.size(0) >= num_chunks + 1, "meta.chunk_offsets mismatch");
    TORCH_CHECK(meta.chunk_valid_lens.is_cuda() && meta.chunk_valid_lens.device() == q_h.device() &&
                meta.chunk_valid_lens.scalar_type() == at::ScalarType::Int &&
                meta.chunk_valid_lens.dim() == 1 && meta.chunk_valid_lens.is_contiguous() &&
                meta.chunk_valid_lens.size(0) >= num_chunks, "meta.chunk_valid_lens mismatch");

    at::Tensor out_tok = torch::empty({total_L, num_q_heads * d_v}, h_opts);
    at::Tensor O_raw   = is_training ? torch::empty({num_q_heads, total_L, d_v}, h_opts) : at::empty({0}, h_opts);
    at::Tensor o_rstd  = is_training ? torch::empty({num_q_heads, total_L}, f_opts)      : at::empty({0}, f_opts);

    at::Tensor q_a    = ensure_aligned16(q_h);
    at::Tensor k_a    = ensure_aligned16(k_h);
    at::Tensor G_a    = ensure_aligned16(G_log2_h);
    at::Tensor S_a    = ensure_aligned16(S_list);
    at::Tensor R_a    = ensure_aligned16(R_list);
    at::Tensor proj_a = ensure_aligned16(proj);
    at::Tensor ow_a   = ensure_aligned16(o_norm_weight);

    const auto* p_q       = reinterpret_cast<const cutlass::half_t*>(q_a.data_ptr<at::Half>());
    const auto* p_k       = reinterpret_cast<const cutlass::half_t*>(k_a.data_ptr<at::Half>());
    const float* p_G      = G_a.data_ptr<float>();
    const float* p_S      = S_a.data_ptr<float>();
    const auto* p_R       = reinterpret_cast<const cutlass::half_t*>(R_a.data_ptr<at::Half>());
    const auto* p_proj    = reinterpret_cast<const cutlass::half_t*>(proj_a.data_ptr<at::Half>());
    const auto* p_ow      = reinterpret_cast<const cutlass::half_t*>(ow_a.data_ptr<at::Half>());
    const int32_t* p_off  = meta.chunk_offsets.data_ptr<int32_t>();
    const int32_t* p_vlen = meta.chunk_valid_lens.data_ptr<int32_t>();
    auto* p_out_tok       = reinterpret_cast<cutlass::half_t*>(out_tok.data_ptr<at::Half>());
    auto* p_O_raw         = is_training ? reinterpret_cast<cutlass::half_t*>(O_raw.data_ptr<at::Half>()) : nullptr;
    float* p_o_rstd       = is_training ? o_rstd.data_ptr<float>() : nullptr;

    {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        float eps_f = static_cast<float>(eps);
        if (is_training) {
            launch_output_combine_fwd<true>(
                p_q, p_k, p_G, p_S, p_R, p_proj, p_ow, p_off, p_vlen,
                p_out_tok, p_O_raw, p_o_rstd,
                num_q_heads, num_kv_heads, num_groups, num_chunks, total_L,
                total_proj_dim, offset_ogate, eps_f, stream
            );
        } else {
            launch_output_combine_fwd<false>(
                p_q, p_k, p_G, p_S, p_R, p_proj, p_ow, p_off, p_vlen,
                p_out_tok, p_O_raw, p_o_rstd,
                num_q_heads, num_kv_heads, num_groups, num_chunks, total_L,
                total_proj_dim, offset_ogate, eps_f, stream
            );
        }
    }

    return std::make_tuple(out_tok, O_raw, o_rstd);
}

// Outputs: (out_tok: (total_L, H_q * 64) [FP16], O_raw: (H_q, total_L, 64) [FP16] or empty, o_rstd: (H_q, total_L) [FP32] or empty)
std::tuple<at::Tensor, at::Tensor, at::Tensor>
output_combine_fwd(
    const at::Tensor& q_h,
    const at::Tensor& k_h,
    const at::Tensor& G_log2_h,
    const at::Tensor& S_list,
    const at::Tensor& R_list,
    const at::Tensor& proj,
    const at::Tensor& o_norm_weight,
    const at::Tensor& cu_seq_lens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t d_k,
    int64_t d_v,
    int64_t chunk_size,
    double eps,
    bool is_training
) {
    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64");
    TORCH_CHECK(cu_seq_lens.is_cuda() && cu_seq_lens.dim() == 1 && cu_seq_lens.size(0) >= 1,
                "cu_seq_lens must be a 1D CUDA tensor with size >= 1");
    VarlenChunkMeta meta = compute_varlen_chunk_metadata(cu_seq_lens, chunk_size);
    return output_combine_fwd_with_meta(
        q_h, k_h, G_log2_h, S_list, R_list, proj, o_norm_weight, meta,
        num_q_heads, num_kv_heads, d_k, d_v, chunk_size, eps, is_training
    );
}
