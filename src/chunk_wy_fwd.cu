#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <tuple>

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm75.hpp>
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
// Part 1: Global Kernels
// ============================================================================

// Kernel 1: compute sub-chunk T (BC=16) with overflow-safe decay scaling and solve A = (I + T)^{-1} in FP32
template <
    typename T,
    int NUM_THREADS,
    int CHUNK_SIZE,
    int SUB_CHUNK,
    int D_K,
    typename MMAAtom,
    typename TiledMMA16,
    typename S2RCopyAtomA,
    typename S2RCopyAtomB,
    typename TiledCopyHalf,
    typename TiledCopyFloat
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_chunk_wy_T_solve_fwd(
    const T*       __restrict__ gK_h,             // (H_kv, total_L, 64) [FP16] - Normalized Key
    const T*       __restrict__ gB_eff_h,         // (H_kv, total_L, 64) [FP16] - Effective erase gate
    const float*   __restrict__ gG_log2_h,        // (H_kv, total_L, 64) [FP32] - Chunk cumsum log2 decay
    const int32_t* __restrict__ chunk_offsets,    // (num_chunks + 1,) [Int32] - Chunk token starts
    const int32_t* __restrict__ chunk_valid_lens, // (num_chunks,) [Int32] - Valid tokens per chunk
    T*             __restrict__ gA,               // (H_kv, num_chunks, 64, 64) [FP16] - Output inverse A
    int64_t total_L,                              // scalar [int64] - Total packed tokens
    int64_t num_chunks                            // scalar [int64] - Total varlen chunks
) {
    // 1. Setup (exact 32 KB SMEM budget)
    int chunk_idx = blockIdx.x;
    int head_idx  = blockIdx.y;
    int tid       = threadIdx.x;
    if (chunk_idx >= num_chunks) return;

    int32_t chunk_start = chunk_offsets[chunk_idx];
    int32_t valid_len   = chunk_valid_lens[chunk_idx];

    int row_id   = tid / 8;
    int col_id   = tid % 8;
    int col_base = col_id * 8;

    int last_sub       = (valid_len > 0) ? ((valid_len - 1) / SUB_CHUNK) : 0;
    int last_sub_valid = (valid_len > 0) ? (valid_len - last_sub * SUB_CHUNK) : 0;

    // 32 KB SMEM layout: 16 KB FP32 sA + 8 KB FP16 sT_off + 4 KB FP16 sSubA/B + 4 KB FP32 sG_sub
    struct alignas(16) SmemStorageT {
        float sA_raw[CHUNK_SIZE * CHUNK_SIZE];
        T     sT_off_raw[CHUNK_SIZE * CHUNK_SIZE];
        T     sSubA_raw[SUB_CHUNK * D_K];
        T     sSubB_raw[SUB_CHUNK * D_K];
        float sG_sub_raw[SUB_CHUNK * D_K];
    };
    __shared__ SmemStorageT smem;

    using SwizzleHalf = Swizzle<3, 3, 3>;
    using SwizzleF32  = Swizzle<3, 2, 3>;

    auto layout_A_f32    = composition(SwizzleF32{},  make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<CHUNK_SIZE>{}), make_stride(Int<CHUNK_SIZE>{}, Int<1>{})));
    auto layout_T_flat   = composition(SwizzleHalf{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<CHUNK_SIZE>{}), make_stride(Int<CHUNK_SIZE>{}, Int<1>{})));
    auto layout_sub_half = composition(SwizzleHalf{}, make_layout(make_shape(Int<SUB_CHUNK>{},  Int<D_K>{}),        make_stride(Int<D_K>{},        Int<1>{})));
    auto layout_sub_f32  = composition(SwizzleF32{},  make_layout(make_shape(Int<SUB_CHUNK>{},  Int<D_K>{}),        make_stride(Int<D_K>{},        Int<1>{})));

    auto sA     = make_tensor(make_smem_ptr(smem.sA_raw),     layout_A_f32);
    auto sT_off = make_tensor(make_smem_ptr(smem.sT_off_raw), layout_T_flat);
    auto sSubA  = make_tensor(make_smem_ptr(smem.sSubA_raw),  layout_sub_half);
    auto sSubB  = make_tensor(make_smem_ptr(smem.sSubB_raw),  layout_sub_half);
    auto sG_sub = make_tensor(make_smem_ptr(smem.sG_sub_raw), layout_sub_f32);

    TiledCopyHalf  tiled_copy_h;
    TiledCopyFloat tiled_copy_f32;
    auto thr_copy_h   = tiled_copy_h.get_slice(tid);
    auto thr_copy_f32 = tiled_copy_f32.get_slice(tid);

    int64_t head_tok_base = static_cast<int64_t>(head_idx) * total_L + chunk_start;
    int64_t mat_offset_A  = (static_cast<int64_t>(head_idx) * num_chunks + chunk_idx) * (CHUNK_SIZE * CHUNK_SIZE);

    auto gA_out = make_tensor(
        make_gmem_ptr(gA + mat_offset_A),
        make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<CHUNK_SIZE>{}), make_stride(Int<CHUNK_SIZE>{}, Int<1>{}))
    );

    auto tXsSubA_d = thr_copy_h.partition_D(sSubA);
    auto tXsSubB_d = thr_copy_h.partition_D(sSubB);
    auto tXsG_d    = thr_copy_f32.partition_D(sG_sub);
    auto tXsT_d    = thr_copy_h.partition_D(sT_off);
    auto tXgA_d    = thr_copy_h.partition_D(gA_out);

    auto tCrSubA   = make_fragment_like(tXsSubA_d);
    auto tCrSubB   = make_fragment_like(tXsSubB_d);
    auto tCrGSub   = make_fragment_like(tXsG_d);
    auto tCrTZero  = make_fragment_like(tXsT_d);
    auto tCrAOut   = make_fragment_like(tXgA_d);

    auto tPredHalfSub = make_fragment_like<bool>(tXsSubA_d);
    auto tPredF32Sub  = make_fragment_like<bool>(tXsG_d);

    auto cSub16     = make_identity_tensor(make_shape(Int<SUB_CHUNK>{}, Int<D_K>{}));
    auto tXcSubHalf = thr_copy_h.partition_S(cSub16);
    #pragma unroll
    for (int m = 0; m < size<1>(tPredHalfSub); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(tPredHalfSub); ++n) {
            bool in_bounds = (get<0>(tXcSubHalf(Int<0>{}, m, n)) < last_sub_valid);
            #pragma unroll
            for (int v = 0; v < size<0>(tPredHalfSub); ++v) {
                tPredHalfSub(v, m, n) = in_bounds;
            }
        }
    }

    auto tXcSubF32 = thr_copy_f32.partition_S(cSub16);
    #pragma unroll
    for (int m = 0; m < size<1>(tPredF32Sub); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(tPredF32Sub); ++n) {
            bool in_bounds = (get<0>(tXcSubF32(Int<0>{}, m, n)) < last_sub_valid);
            #pragma unroll
            for (int v = 0; v < size<0>(tPredF32Sub); ++v) {
                tPredF32Sub(v, m, n) = in_bounds;
            }
        }
    }

    // Sub-chunk 16x16x64 MMA partitioning for the first 2 warps (tid < 64)
    TiledMMA16 tiled_mma_16;
    auto thr_mma_16    = tiled_mma_16.get_slice(tid & 63);
    auto s2r_copy_a    = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma_16);
    auto s2r_copy_b    = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma_16);
    auto s2r_thr_a     = s2r_copy_a.get_slice(tid & 63);
    auto s2r_thr_b     = s2r_copy_b.get_slice(tid & 63);
    auto tCsA          = s2r_thr_a.partition_S(sSubA);
    auto tCsB          = s2r_thr_b.partition_S(sSubB);
    auto tCrA          = thr_mma_16.partition_fragment_A(sSubA);
    auto tCrB          = thr_mma_16.partition_fragment_B(sSubB);
    auto tCrA_view     = s2r_thr_a.retile_D(tCrA);
    auto tCrB_view     = s2r_thr_b.retile_D(tCrB);
    auto tCsT_block00  = thr_mma_16.partition_C(local_tile(sT_off, make_shape(Int<SUB_CHUNK>{}, Int<SUB_CHUNK>{}), make_coord(Int<0>{}, Int<0>{})));
    auto tCrC_sub      = thr_mma_16.make_fragment_C(tCsT_block00);
    auto rC_half_sub   = make_tensor_like<T>(tCrC_sub);

    // Zero-initialize sA and sT_off
    clear(tCrTZero);
    cute::copy(tiled_copy_h, tCrTZero, tXsT_d);
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        int local_row = m * 16 + row_id;
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            sA(local_row, col_base + v) = 0.0f;
        }
    }
    __syncthreads();

    // 2. Compute 4 diagonal 16x16 blocks A_ss = (I + T_ss)^{-1} in FP32
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        int sub_start = s * SUB_CHUNK;
        if (sub_start >= valid_len) {
            if (tid < SUB_CHUNK) {
                sA(sub_start + tid, sub_start + tid) = 1.0f;
            }
            __syncthreads();
            continue;
        }

        int sub_valid    = (s == last_sub) ? last_sub_valid : SUB_CHUNK;
        bool is_full_sub = (sub_valid == SUB_CHUNK);
        int64_t sub_off  = (head_tok_base + sub_start) * D_K;

        auto gKs = make_tensor(make_gmem_ptr(gK_h      + sub_off), make_layout(make_shape(Int<SUB_CHUNK>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
        auto gBs = make_tensor(make_gmem_ptr(gB_eff_h  + sub_off), make_layout(make_shape(Int<SUB_CHUNK>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
        auto gGs = make_tensor(make_gmem_ptr(gG_log2_h + sub_off), make_layout(make_shape(Int<SUB_CHUNK>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

        if (is_full_sub) {
            cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gKs),   tCrSubA);
            cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gBs),   tCrSubB);
            cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gGs), tCrGSub);
        } else {
            clear(tCrSubA);
            clear(tCrSubB);
            clear(tCrGSub);
            cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gKs),   tCrSubA);
            cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gBs),   tCrSubB);
            cute::copy_if(tiled_copy_f32, tPredF32Sub,  thr_copy_f32.partition_S(gGs), tCrGSub);
        }
        cute::copy(tiled_copy_h,   tCrSubA, tXsSubA_d);
        cute::copy(tiled_copy_h,   tCrSubB, tXsSubB_d);
        cute::copy(tiled_copy_f32, tCrGSub, tXsG_d);
        __syncthreads();

        float e_r[8];
        float g_r[8];
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            e_r[v] = (row_id < sub_valid)
                ? (half_to_float(sSubB(row_id, col_base + v)) * half_to_float(sSubA(row_id, col_base + v)))
                : 0.0f;
            g_r[v] = (row_id < sub_valid) ? sG_sub(row_id, col_base + v) : 0.0f;
        }

        // Compute T_ss[row_id, j] in FP32 and stage into upper scratch block (0, 3) of sA
        for (int j = 0; j < SUB_CHUNK; ++j) {
            float partial = 0.0f;
            if (j < row_id && row_id < sub_valid) {
                #pragma unroll
                for (int v = 0; v < 8; ++v) {
                    partial += e_r[v] *
                               half_to_float(sSubA(j, col_base + v)) *
                               exp2f(g_r[v] - sG_sub(j, col_base + v));
                }
            }
            float t_rj = octet_reduce_sum(partial);
            if (col_id == 0) {
                sA(row_id, 48 + j) = t_rj;
            }
        }
        __syncthreads();

        if (tid < SUB_CHUNK) {
            int c = tid;
            float a_col[SUB_CHUNK];
            #pragma unroll
            for (int r = 0; r < SUB_CHUNK; ++r) {
                a_col[r] = (r == c) ? 1.0f : 0.0f;
            }
            for (int r = c + 1; r < SUB_CHUNK; ++r) {
                float acc = 0.0f;
                for (int m = c; m < r; ++m) {
                    acc += sA(r, 48 + m) * a_col[m];
                }
                a_col[r] = -acc;
            }
            #pragma unroll
            for (int r = 0; r < SUB_CHUNK; ++r) {
                sA(sub_start + r, sub_start + c) = a_col[r];
            }
        }
        __syncthreads();
    }

    // 3. Compute 6 off-diagonal 16x16 blocks T_{ij} (0 <= j < i < 4) via Tensor Core GEMM
    for (int i = 1; i < 4; ++i) {
        int i_start = i * SUB_CHUNK;
        if (i_start >= valid_len) break;

        int i_valid      = (i == last_sub) ? last_sub_valid : SUB_CHUNK;
        bool is_full_i   = (i_valid == SUB_CHUNK);
        int64_t i_off    = (head_tok_base + i_start) * D_K;

        // Anchor log2 decay at first token of target sub-chunk i so all exp2 args stay <= 0
        const float* g_ref_ptr = gG_log2_h + i_off + col_base;
        float4 gr0 = *reinterpret_cast<const float4*>(g_ref_ptr);
        float4 gr1 = *reinterpret_cast<const float4*>(g_ref_ptr + 4);
        float r_gref[8] = {gr0.x, gr0.y, gr0.z, gr0.w, gr1.x, gr1.y, gr1.z, gr1.w};

        auto gKi = make_tensor(make_gmem_ptr(gK_h      + i_off), make_layout(make_shape(Int<SUB_CHUNK>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
        auto gBi = make_tensor(make_gmem_ptr(gB_eff_h  + i_off), make_layout(make_shape(Int<SUB_CHUNK>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
        auto gGi = make_tensor(make_gmem_ptr(gG_log2_h + i_off), make_layout(make_shape(Int<SUB_CHUNK>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

        if (is_full_i) {
            cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gKi),   tCrSubA);
            cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gBi),   tCrSubB);
            cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gGi), tCrGSub);
        } else {
            clear(tCrSubA);
            clear(tCrSubB);
            clear(tCrGSub);
            cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gKi),   tCrSubA);
            cute::copy_if(tiled_copy_h,   tPredHalfSub, thr_copy_h.partition_S(gBi),   tCrSubB);
            cute::copy_if(tiled_copy_f32, tPredF32Sub,  thr_copy_f32.partition_S(gGi), tCrGSub);
        }
        cute::copy(tiled_copy_f32, tCrGSub, tXsG_d);
        __syncthreads();

        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            float val = (row_id < i_valid)
                ? (half_to_float(tCrSubB(v, 0, 0)) * half_to_float(tCrSubA(v, 0, 0)) *
                   exp2f(sG_sub(row_id, col_base + v) - r_gref[v]))
                : 0.0f;
            tCrSubA(v, 0, 0) = float_to_out<T>(val);
        }
        cute::copy(tiled_copy_h, tCrSubA, tXsSubA_d);
        __syncthreads();

        if (tid < 64) {
            cute::copy(s2r_copy_a, tCsA, tCrA_view);
        }

        for (int j = 0; j < i; ++j) {
            int64_t j_off = (head_tok_base + j * SUB_CHUNK) * D_K;
            auto gKj = make_tensor(make_gmem_ptr(gK_h      + j_off), make_layout(make_shape(Int<SUB_CHUNK>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
            auto gGj = make_tensor(make_gmem_ptr(gG_log2_h + j_off), make_layout(make_shape(Int<SUB_CHUNK>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

            cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gKj),   tCrSubB);
            cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gGj), tCrGSub);
            cute::copy(tiled_copy_f32, tCrGSub, tXsG_d);
            __syncthreads();

            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                float val = half_to_float(tCrSubB(v, 0, 0)) *
                            exp2f(r_gref[v] - sG_sub(row_id, col_base + v));
                tCrSubB(v, 0, 0) = float_to_out<T>(val);
            }
            cute::copy(tiled_copy_h, tCrSubB, tXsSubB_d);
            __syncthreads();

            if (tid < 64) {
                cute::copy(s2r_copy_b, tCsB, tCrB_view);
                clear(tCrC_sub);
                cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC_sub);

                #pragma unroll
                for (int idx = 0; idx < size(tCrC_sub); ++idx) {
                    rC_half_sub(idx) = float_to_out<T>(tCrC_sub(idx));
                }
                auto sT_ij = local_tile(sT_off, make_shape(Int<SUB_CHUNK>{}, Int<SUB_CHUNK>{}), make_coord(i, j));
                cute::copy(rC_half_sub, thr_mma_16.partition_C(sT_ij));
            }
            __syncthreads();
        }
    }

    // 4. Schur-complement 4x4 block assembly in FP32 in sA and 128-bit store to gA
    int r0 = tid / 16;
    int r1 = r0 + 8;
    int c  = tid % 16;

    auto mul_TA_store = [&](int t_bi, int t_bj, int a_bi, int a_bj, int dst_bi, int dst_bj, bool add_prev) {
        int tr0 = t_bi * SUB_CHUNK + r0;
        int tr1 = t_bi * SUB_CHUNK + r1;
        int tc0 = t_bj * SUB_CHUNK;
        int ar0 = a_bi * SUB_CHUNK;
        int ac  = a_bj * SUB_CHUNK + c;
        float acc0 = add_prev ? sA(dst_bi * SUB_CHUNK + r0, dst_bj * SUB_CHUNK + c) : 0.0f;
        float acc1 = add_prev ? sA(dst_bi * SUB_CHUNK + r1, dst_bj * SUB_CHUNK + c) : 0.0f;
        #pragma unroll
        for (int k = 0; k < SUB_CHUNK; ++k) {
            float a_kc = sA(ar0 + k, ac);
            acc0 += half_to_float(sT_off(tr0, tc0 + k)) * a_kc;
            acc1 += half_to_float(sT_off(tr1, tc0 + k)) * a_kc;
        }
        sA(dst_bi * SUB_CHUNK + r0, dst_bj * SUB_CHUNK + c) = acc0;
        sA(dst_bi * SUB_CHUNK + r1, dst_bj * SUB_CHUNK + c) = acc1;
    };

    auto mul_negAA_store = [&](int l_bi, int l_bj, int r_bi, int r_bj, int dst_bi, int dst_bj) {
        int lr0 = l_bi * SUB_CHUNK + r0;
        int lr1 = l_bi * SUB_CHUNK + r1;
        int lc0 = l_bj * SUB_CHUNK;
        int rr0 = r_bi * SUB_CHUNK;
        int rc  = r_bj * SUB_CHUNK + c;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        #pragma unroll
        for (int k = 0; k < SUB_CHUNK; ++k) {
            float r_kc = sA(rr0 + k, rc);
            acc0 += sA(lr0, lc0 + k) * r_kc;
            acc1 += sA(lr1, lc0 + k) * r_kc;
        }
        sA(dst_bi * SUB_CHUNK + r0, dst_bj * SUB_CHUNK + c) = -acc0;
        sA(dst_bi * SUB_CHUNK + r1, dst_bj * SUB_CHUNK + c) = -acc1;
    };

    if (valid_len > 16) {
        mul_TA_store(1, 0, 0, 0, 0, 1, false);
        if (valid_len > 32) {
            mul_TA_store(2, 1, 1, 1, 0, 2, false);
        }
        if (valid_len > 48) {
            mul_TA_store(3, 2, 2, 2, 0, 3, false);
        }
        __syncthreads();

        mul_negAA_store(1, 1, 0, 1, 1, 0);
        if (valid_len > 32) {
            mul_negAA_store(2, 2, 0, 2, 2, 1);
        }
        if (valid_len > 48) {
            mul_negAA_store(3, 3, 0, 3, 3, 2);
        }
        __syncthreads();
    }

    if (valid_len > 32) {
        mul_TA_store(2, 0, 0, 0, 0, 1, false);
        mul_TA_store(2, 1, 1, 0, 0, 1, true);
        if (valid_len > 48) {
            mul_TA_store(3, 1, 1, 1, 0, 2, false);
            mul_TA_store(3, 2, 2, 1, 0, 2, true);
        }
        __syncthreads();

        mul_negAA_store(2, 2, 0, 1, 2, 0);
        if (valid_len > 48) {
            mul_negAA_store(3, 3, 0, 2, 3, 1);
        }
        __syncthreads();
    }

    if (valid_len > 48) {
        mul_TA_store(3, 0, 0, 0, 0, 1, false);
        mul_TA_store(3, 1, 1, 0, 0, 1, true);
        mul_TA_store(3, 2, 2, 0, 0, 1, true);
        __syncthreads();

        mul_negAA_store(3, 3, 0, 1, 3, 0);
        __syncthreads();
    }

    // Clear upper-triangular scratch blocks (0,1), (0,2), (0,3) unconditionally before store
    #pragma unroll
    for (int col_blk = 1; col_blk < 4; ++col_blk) {
        sA(r0, col_blk * SUB_CHUNK + c) = 0.0f;
        sA(r1, col_blk * SUB_CHUNK + c) = 0.0f;
    }
    __syncthreads();

    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        int local_row = m * 16 + row_id;
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            tCrAOut(v, m, 0) = float_to_out<T>(sA(local_row, col_base + v));
        }
    }
    cute::copy(tiled_copy_h, tCrAOut, tXgA_d);
}

// Kernel 2: compute exact WY factors Y = A @ E_bar and U = A @ (W * V)
template <
    typename T,
    int NUM_THREADS,
    int CHUNK_SIZE,
    int D_K,
    int D_V,
    typename MMAAtom,
    typename TiledMMA221,
    typename S2RCopyAtomA,
    typename S2RCopyAtomB,
    typename TiledCopyHalf,
    typename TiledCopyFloat
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_chunk_wy_YU_fwd(
    const T*       __restrict__ gA,               // (H_kv, num_chunks, 64, 64) [FP16] - WY inverse A
    const T*       __restrict__ gK_h,             // (H_kv, total_L, 64) [FP16] - Normalized Key
    const T*       __restrict__ gV_h,             // (H_kv, total_L, 64) [FP16] - Activated Value
    const T*       __restrict__ gB_eff_h,         // (H_kv, total_L, 64) [FP16] - Effective erase gate
    const T*       __restrict__ gW_h,             // (H_kv, total_L, 64) [FP16] - Write gate
    const float*   __restrict__ gG_log2_h,        // (H_kv, total_L, 64) [FP32] - Chunk cumsum log2 decay
    const int32_t* __restrict__ chunk_offsets,    // (num_chunks + 1,) [Int32] - Chunk token starts
    const int32_t* __restrict__ chunk_valid_lens, // (num_chunks,) [Int32] - Valid tokens per chunk
    T*             __restrict__ gY,               // (H_kv, num_chunks, 64, 64) [FP16] - Output WY factor Y
    T*             __restrict__ gU,               // (H_kv, num_chunks, 64, 64) [FP16] - Output WY factor U
    int64_t total_L,                              // scalar [int64] - Total packed tokens
    int64_t num_chunks                            // scalar [int64] - Total varlen chunks
) {
    // 1. Setup (24 KB SMEM)
    int chunk_idx = blockIdx.x;
    int head_idx  = blockIdx.y;
    int tid       = threadIdx.x;
    if (chunk_idx >= num_chunks) return;

    int32_t chunk_start = chunk_offsets[chunk_idx];
    int32_t valid_len   = chunk_valid_lens[chunk_idx];

    int row_id   = tid / 8;
    int col_id   = tid % 8;
    int col_base = col_id * 8;

    union SmemAZ {
        struct {
            T sA_raw[CHUNK_SIZE * CHUNK_SIZE];
            T sZ_raw[CHUNK_SIZE * D_V];
        } mat;
        float sG_raw[CHUNK_SIZE * D_K];
    };
    __shared__ alignas(16) SmemAZ smem_az;
    __shared__ alignas(16) T sE_raw[CHUNK_SIZE * D_K];

    using SwizzleHalf = Swizzle<3, 3, 3>;
    using SwizzleF32  = Swizzle<3, 2, 3>;

    auto layout_f32      = composition(SwizzleF32{},  make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}),        make_stride(Int<D_K>{},        Int<1>{})));
    auto layout_flat     = composition(SwizzleHalf{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<CHUNK_SIZE>{}), make_stride(Int<CHUNK_SIZE>{}, Int<1>{})));
    auto layout_mma_A    = composition(SwizzleHalf{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<CHUNK_SIZE>{}), make_stride(Int<CHUNK_SIZE>{}, Int<1>{})));
    auto layout_mma_B    = composition(SwizzleHalf{}, make_layout(make_shape(Int<D_K>{},        Int<CHUNK_SIZE>{}), make_stride(Int<1>{},        Int<D_K>{})));
    auto layout_mma_C    = composition(SwizzleHalf{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}),        make_stride(Int<D_K>{},        Int<1>{})));

    auto sG      = make_tensor(make_smem_ptr(smem_az.sG_raw),     layout_f32);
    auto sA_flat = make_tensor(make_smem_ptr(smem_az.mat.sA_raw), layout_flat);
    auto sZ_flat = make_tensor(make_smem_ptr(smem_az.mat.sZ_raw), layout_flat);
    auto sE_flat = make_tensor(make_smem_ptr(sE_raw),             layout_flat);

    auto sA_mma  = make_tensor(make_smem_ptr(smem_az.mat.sA_raw), layout_mma_A);
    auto sE_mma  = make_tensor(make_smem_ptr(sE_raw),             layout_mma_B);
    auto sZ_mma  = make_tensor(make_smem_ptr(smem_az.mat.sZ_raw), layout_mma_B);
    auto sC_E    = make_tensor(make_smem_ptr(sE_raw),             layout_mma_C);
    auto sC_Z    = make_tensor(make_smem_ptr(smem_az.mat.sZ_raw), layout_mma_C);

    TiledCopyHalf  tiled_copy_h;
    TiledCopyFloat tiled_copy_f32;
    auto thr_copy_h   = tiled_copy_h.get_slice(tid);
    auto thr_copy_f32 = tiled_copy_f32.get_slice(tid);

    int64_t head_tok_off = (static_cast<int64_t>(head_idx) * total_L + chunk_start) * D_K;
    int64_t head_mat_off = (static_cast<int64_t>(head_idx) * num_chunks + chunk_idx) * (CHUNK_SIZE * CHUNK_SIZE);

    auto gA_in = make_tensor(make_gmem_ptr(gA + head_mat_off), make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<CHUNK_SIZE>{}), make_stride(Int<CHUNK_SIZE>{}, Int<1>{})));
    auto gY_out = make_tensor(make_gmem_ptr(gY + head_mat_off), make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto gU_out = make_tensor(make_gmem_ptr(gU + head_mat_off), make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{})));

    auto gK_in = make_tensor(make_gmem_ptr(gK_h      + head_tok_off), make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto gB_in = make_tensor(make_gmem_ptr(gB_eff_h  + head_tok_off), make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto gV_in = make_tensor(make_gmem_ptr(gV_h      + head_tok_off), make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{})));
    auto gW_in = make_tensor(make_gmem_ptr(gW_h      + head_tok_off), make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{})));
    auto gG_in = make_tensor(make_gmem_ptr(gG_log2_h + head_tok_off), make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

    auto tXgA   = thr_copy_h.partition_S(gA_in);
    auto tXsA_d = thr_copy_h.partition_D(sA_flat);
    auto tXgK   = thr_copy_h.partition_S(gK_in);
    auto tXgB   = thr_copy_h.partition_S(gB_in);
    auto tXgV   = thr_copy_h.partition_S(gV_in);
    auto tXgW   = thr_copy_h.partition_S(gW_in);
    auto tXsE_d = thr_copy_h.partition_D(sE_flat);
    auto tXsZ_d = thr_copy_h.partition_D(sZ_flat);
    auto tXsE_s = thr_copy_h.partition_S(sE_flat);
    auto tXsZ_s = thr_copy_h.partition_S(sZ_flat);
    auto tXgY   = thr_copy_h.partition_D(gY_out);
    auto tXgU   = thr_copy_h.partition_D(gU_out);

    auto tCrBuf0 = make_fragment_like(tXgK);
    auto tCrBuf1 = make_fragment_like(tXgB);

    auto tXgG   = thr_copy_f32.partition_S(gG_in);
    auto tXsG_d = thr_copy_f32.partition_D(sG);
    auto tCrG   = make_fragment_like(tXgG);

    auto tPred64   = make_fragment_like<bool>(tXgK);
    auto tPredF32  = make_fragment_like<bool>(tXgG);
    auto cTile64   = make_identity_tensor(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}));
    auto tXcHalf   = thr_copy_h.partition_S(cTile64);
    auto tXcF32    = thr_copy_f32.partition_S(cTile64);

    #pragma unroll
    for (int m = 0; m < size<1>(tPred64); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(tPred64); ++n) {
            bool in_bounds = (get<0>(tXcHalf(Int<0>{}, m, n)) < valid_len);
            #pragma unroll
            for (int v = 0; v < size<0>(tPred64); ++v) {
                tPred64(v, m, n) = in_bounds;
            }
        }
    }

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

    TiledMMA221 tiled_mma;
    auto thr_mma    = tiled_mma.get_slice(tid);
    auto s2r_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_a  = s2r_copy_a.get_slice(tid);
    auto s2r_thr_b  = s2r_copy_b.get_slice(tid);

    auto tCsA       = s2r_thr_a.partition_S(sA_mma);
    auto tCsB_E     = s2r_thr_b.partition_S(sE_mma);
    auto tCsB_Z     = s2r_thr_b.partition_S(sZ_mma);
    auto tCsC_E     = thr_mma.partition_C(sC_E);
    auto tCsC_Z     = thr_mma.partition_C(sC_Z);

    auto tCrA       = thr_mma.partition_fragment_A(sA_mma);
    auto tCrB       = thr_mma.partition_fragment_B(sE_mma);
    auto tCrA_view  = s2r_thr_a.retile_D(tCrA);
    auto tCrB_view  = s2r_thr_b.retile_D(tCrB);
    auto tCrC       = thr_mma.make_fragment_C(tCsC_E);
    auto rC_half    = make_tensor_like<T>(tCrC);

    // 2. Phase 1: build E_bar = exp2(G_log2) * b_eff * k into sE
    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy_f32, tXgG, tCrG);
        cute::copy(tiled_copy_h,   tXgK, tCrBuf0);
        cute::copy(tiled_copy_h,   tXgB, tCrBuf1);
    } else {
        clear(tCrG);
        clear(tCrBuf0);
        clear(tCrBuf1);
        cute::copy_if(tiled_copy_f32, tPredF32, tXgG, tCrG);
        cute::copy_if(tiled_copy_h,   tPred64,  tXgK, tCrBuf0);
        cute::copy_if(tiled_copy_h,   tPred64,  tXgB, tCrBuf1);
    }
    cute::copy(tiled_copy_f32, tCrG, tXsG_d);
    __syncthreads();

    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        int local_row = m * 16 + row_id;
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            float e_val = (local_row < valid_len)
                ? (exp2f(sG(local_row, col_base + v)) *
                   half_to_float(tCrBuf1(v, m, 0)) *
                   half_to_float(tCrBuf0(v, m, 0)))
                : 0.0f;
            tCrBuf0(v, m, 0) = float_to_out<T>(e_val);
        }
    }
    cute::copy(tiled_copy_h, tCrBuf0, tXsE_d);
    __syncthreads();

    // 3. Phase 2: load A into sA and build Z = W * V into sZ (overwriting sG union)
    cute::copy(tiled_copy_h, tXgA, tCrBuf0);
    cute::copy(tiled_copy_h, tCrBuf0, tXsA_d);

    if (valid_len == CHUNK_SIZE) {
        cute::copy(tiled_copy_h, tXgV, tCrBuf0);
        cute::copy(tiled_copy_h, tXgW, tCrBuf1);
    } else {
        clear(tCrBuf0);
        clear(tCrBuf1);
        cute::copy_if(tiled_copy_h, tPred64, tXgV, tCrBuf0);
        cute::copy_if(tiled_copy_h, tPred64, tXgW, tCrBuf1);
    }

    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        int local_row = m * 16 + row_id;
        #pragma unroll
        for (int v = 0; v < 8; v += 2) {
            __half2 v_h2 = __halves2half2(to_half_val(tCrBuf0(v, m, 0)), to_half_val(tCrBuf0(v + 1, m, 0)));
            __half2 w_h2 = __halves2half2(to_half_val(tCrBuf1(v, m, 0)), to_half_val(tCrBuf1(v + 1, m, 0)));
            __half2 z_h2 = (local_row < valid_len) ? __hmul2(w_h2, v_h2) : __float2half2_rn(0.0f);
            tCrBuf0(v, m, 0)     = half_to_out<T>(__low2half(z_h2));
            tCrBuf0(v + 1, m, 0) = half_to_out<T>(__high2half(z_h2));
        }
    }
    cute::copy(tiled_copy_h, tCrBuf0, tXsZ_d);
    __syncthreads();

    // 4. GEMM 1: Y = A @ E_bar (keep tCrA resident for GEMM 2)
    cute::copy(s2r_copy_a, tCsA,   tCrA_view);
    cute::copy(s2r_copy_b, tCsB_E, tCrB_view);
    clear(tCrC);
    cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < size(tCrC); ++i) {
        rC_half(i) = float_to_out<T>(tCrC(i));
    }
    cute::copy(rC_half, tCsC_E);
    __syncthreads();

    cute::copy(tiled_copy_h, tXsE_s, tCrBuf0);
    cute::copy(tiled_copy_h, tCrBuf0, tXgY);

    // 5. GEMM 2: U = A @ Z
    cute::copy(s2r_copy_b, tCsB_Z, tCrB_view);
    clear(tCrC);
    cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < size(tCrC); ++i) {
        rC_half(i) = float_to_out<T>(tCrC(i));
    }
    cute::copy(rC_half, tCsC_Z);
    __syncthreads();

    cute::copy(tiled_copy_h, tXsZ_s, tCrBuf0);
    cute::copy(tiled_copy_h, tCrBuf0, tXgU);
}

// ============================================================================
// Part 2: Host Dispatch Functions
// ============================================================================

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
) {
    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64");
    auto check_h3d = [&](const at::Tensor& t, const char* name, at::ScalarType dtype) {
        TORCH_CHECK(t.is_cuda() && t.scalar_type() == dtype, name, " dtype or device mismatch");
        TORCH_CHECK(t.dim() == 3 && t.is_contiguous(), name, " must be contiguous 3D");
    };
    check_h3d(k_h,      "k_h",      at::ScalarType::Half);
    check_h3d(v_h,      "v_h",      at::ScalarType::Half);
    check_h3d(b_eff_h,  "b_eff_h",  at::ScalarType::Half);
    check_h3d(w_h,      "w_h",      at::ScalarType::Half);
    check_h3d(G_log2_h, "G_log2_h", at::ScalarType::Float);

    int64_t num_kv_heads = k_h.size(0);
    int64_t total_L      = k_h.size(1);
    int64_t d_k          = k_h.size(2);
    int64_t d_v          = v_h.size(2);

    TORCH_CHECK(d_k == 64 && d_v == 64, "d_k and d_v must be 64");
    TORCH_CHECK(v_h.size(0) == num_kv_heads && v_h.size(1) == total_L, "v_h shape mismatch");
    TORCH_CHECK(b_eff_h.size(0) == num_kv_heads && b_eff_h.size(1) == total_L && b_eff_h.size(2) == d_k, "b_eff_h shape mismatch");
    TORCH_CHECK(w_h.size(0) == num_kv_heads && w_h.size(1) == total_L && w_h.size(2) == d_v, "w_h shape mismatch");
    TORCH_CHECK(G_log2_h.size(0) == num_kv_heads && G_log2_h.size(1) == total_L && G_log2_h.size(2) == d_k, "G_log2_h shape mismatch");

    c10::cuda::CUDAGuard device_guard(k_h.device());
    auto h_opts = k_h.options();

    if (meta.num_chunks <= 0) {
        at::Tensor Y = at::empty({num_kv_heads, 0, chunk_size, d_k}, h_opts);
        at::Tensor U = at::empty({num_kv_heads, 0, chunk_size, d_v}, h_opts);
        at::Tensor A = is_training ? at::empty({num_kv_heads, 0, chunk_size, chunk_size}, h_opts) : at::empty({0}, h_opts);
        return std::make_tuple(Y, U, A);
    }

    at::Tensor A_buf = torch::empty({num_kv_heads, meta.num_chunks, chunk_size, chunk_size}, h_opts);
    at::Tensor Y     = torch::empty({num_kv_heads, meta.num_chunks, chunk_size, d_k}, h_opts);
    at::Tensor U     = torch::empty({num_kv_heads, meta.num_chunks, chunk_size, d_v}, h_opts);

    at::Tensor k_a = ensure_aligned16(k_h);
    at::Tensor v_a = ensure_aligned16(v_h);
    at::Tensor b_a = ensure_aligned16(b_eff_h);
    at::Tensor w_a = ensure_aligned16(w_h);
    at::Tensor G_a = ensure_aligned16(G_log2_h);

    constexpr int NUM_THREADS = 128;
    constexpr int CHUNK_CONST = 64;
    constexpr int SUB_CHUNK   = 16;
    constexpr int DK_CONST    = 64;
    constexpr int DV_CONST    = 64;

    using Element   = cutlass::half_t;
    using MMAAtom   = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
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
        Layout<Shape<_8, _16>, Stride<_16, _1>>{},
        Layout<Shape<_1, _4>, Stride<_4, _1>>{}
    ));

    const Element* k_ptr     = reinterpret_cast<const Element*>(k_a.data_ptr<at::Half>());
    const Element* v_ptr     = reinterpret_cast<const Element*>(v_a.data_ptr<at::Half>());
    const Element* b_ptr     = reinterpret_cast<const Element*>(b_a.data_ptr<at::Half>());
    const Element* w_ptr     = reinterpret_cast<const Element*>(w_a.data_ptr<at::Half>());
    const float*   G_ptr     = G_a.data_ptr<float>();
    const int32_t* c_off     = meta.chunk_offsets.data_ptr<int32_t>();
    const int32_t* c_vlen    = meta.chunk_valid_lens.data_ptr<int32_t>();
    Element*       A_ptr     = reinterpret_cast<Element*>(A_buf.data_ptr<at::Half>());
    Element*       Y_ptr     = reinterpret_cast<Element*>(Y.data_ptr<at::Half>());
    Element*       U_ptr     = reinterpret_cast<Element*>(U.data_ptr<at::Half>());

    // 1. Kernel 1: solve A = (I + T)^{-1} per chunk
    {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        dim3 grid(static_cast<unsigned int>(meta.num_chunks), static_cast<unsigned int>(num_kv_heads));
        dim3 block(NUM_THREADS);
        kernel_chunk_wy_T_solve_fwd<
            Element, NUM_THREADS, CHUNK_CONST, SUB_CHUNK, DK_CONST,
            MMAAtom, TiledMMA16, S2RCopyAtomA_KM, S2RCopyAtomB_KM, TiledCopyHalf, TiledCopyFloat
        ><<<grid, block, 0, stream>>>(
            k_ptr, b_ptr, G_ptr, c_off, c_vlen, A_ptr, total_L, meta.num_chunks
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // 2. Kernel 2: compute Y = A @ E_bar and U = A @ (W * V)
    {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        dim3 grid(static_cast<unsigned int>(meta.num_chunks), static_cast<unsigned int>(num_kv_heads));
        dim3 block(NUM_THREADS);
        kernel_chunk_wy_YU_fwd<
            Element, NUM_THREADS, CHUNK_CONST, DK_CONST, DV_CONST,
            MMAAtom, TiledMMA221, S2RCopyAtomA_KM, S2RCopyAtomB_MNM, TiledCopyHalf, TiledCopyFloat
        ><<<grid, block, 0, stream>>>(
            A_ptr, k_ptr, v_ptr, b_ptr, w_ptr, G_ptr, c_off, c_vlen, Y_ptr, U_ptr, total_L, meta.num_chunks
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (!is_training) {
        A_buf.reset();
        return std::make_tuple(Y, U, at::empty({0}, h_opts));
    }
    return std::make_tuple(Y, U, A_buf);
}

// Outputs: (Y: (H_kv, num_chunks, 64, 64) [FP16], U: (H_kv, num_chunks, 64, 64) [FP16], A: (H_kv, num_chunks, 64, 64) [FP16] or empty)
std::tuple<at::Tensor, at::Tensor, at::Tensor>
chunk_wy_fwd(
    const at::Tensor& k_h,
    const at::Tensor& v_h,
    const at::Tensor& b_eff_h,
    const at::Tensor& w_h,
    const at::Tensor& G_log2_h,
    const at::Tensor& cu_seq_lens,
    int64_t chunk_size,
    bool is_training
) {
    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64");
    TORCH_CHECK(cu_seq_lens.is_cuda() && cu_seq_lens.dim() == 1 && cu_seq_lens.size(0) >= 1,
                "cu_seq_lens must be a 1D CUDA tensor with size >= 1");
    VarlenChunkMeta meta = compute_varlen_chunk_metadata(cu_seq_lens, chunk_size);
    return chunk_wy_fwd_with_meta(
        k_h, v_h, b_eff_h, w_h, G_log2_h, meta, chunk_size, is_training
    );
}
