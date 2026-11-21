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
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include "deltanet_common.cuh"
#include "deltanet_ops.cuh"

using namespace cute;

// Chunk WY backward kernel computing raw input gradients (dk, dv, db, dw, dgamma, da).
template <
    typename T,
    int NUM_THREADS,
    int CHUNK_SIZE,
    int D_K,
    int D_V,
    typename MMAAtom,
    typename TiledMMAType,
    typename S2RCopyAtomA,
    typename S2RCopyAtomB_KMajor,
    typename S2RCopyAtomB_MNMajor,
    typename TiledCopyG2R,
    typename TiledCopyStore
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_chunk_wy_bwd(
    const T* __restrict__ gdW,                 // (num_kv_heads, num_chunks, d_k, d_k) [FP16] - Gradient dW
    const T* __restrict__ gdY,                 // (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Gradient dY
    const T* __restrict__ gK,                  // (num_kv_heads, total_L, d_k) [FP16] - Raw Key
    const T* __restrict__ gV,                  // (num_kv_heads, total_L, d_v) [FP16] - Raw Value
    const T* __restrict__ gB,                  // (num_kv_heads, total_L, d_k) [FP16] - Raw Beta
    const T* __restrict__ gW_gate,             // (num_kv_heads, total_L, d_v) [FP16] - Raw W gate
    const T* __restrict__ gGamma,              // (num_kv_heads, total_L, 1) [FP16] - Raw Gamma
    const T* __restrict__ gAlpha,              // (num_kv_heads, total_L, 1) [FP16] - Raw Alpha
    const T* __restrict__ gDK_ext,             // (num_kv_heads, total_L, d_k) [FP16] - External Key gradient
    const T* __restrict__ gDV_ext,             // (num_kv_heads, total_L, d_v) [FP16] - External Value gradient
    const float* __restrict__ gDA_ext,         // (num_kv_heads, total_L, 1) [FP32] - External log alpha gradient
    const float* __restrict__ gD_trace,        // (num_kv_heads, num_chunks) [FP32] - Trace dW
    T* __restrict__ gDK,                       // (num_kv_heads, total_L, d_k) [FP16] - Output dk_raw
    T* __restrict__ gDV,                       // (num_kv_heads, total_L, d_v) [FP16] - Output dv_raw
    T* __restrict__ gDB,                       // (num_kv_heads, total_L, d_k) [FP16] - Output db_raw
    T* __restrict__ gDW,                       // (num_kv_heads, total_L, d_v) [FP16] - Output dw_raw
    T* __restrict__ gDGamma,                   // (num_kv_heads, total_L, 1) [FP16] - Output dgamma_raw
    T* __restrict__ gDAlpha,                   // (num_kv_heads, total_L, 1) [FP16] - Output dalpha_raw
    const int32_t* __restrict__ chunk_offsets, // (num_chunks + 1,) [Int32] - Token offsets per chunk
    int64_t total_L,                           // scalar [int64] - Total tokens
    int64_t num_chunks                         // scalar [int64] - Total chunks
) {
    // Stage 1 - Setup & Grid Initialization
    int chunk_idx = blockIdx.x;
    int head_idx  = blockIdx.y;
    int tid       = threadIdx.x;

    if (chunk_idx >= num_chunks) return;

    int32_t chunk_start_token = chunk_offsets[chunk_idx];
    int32_t chunk_end_token   = chunk_offsets[chunk_idx + 1];
    int valid_tokens          = chunk_end_token - chunk_start_token;

    int64_t head_offset = (int64_t)head_idx * total_L;

    // Shared Memory Allocation: exactly 32 KB (4 x 8 KB buffers)
    struct SmemStorageChunkWYBwd {
        alignas(16) T sU[CHUNK_SIZE * D_K]; // 8 KB
        alignas(16) T sV[CHUNK_SIZE * D_V]; // 8 KB
        alignas(16) T sK[CHUNK_SIZE * D_K]; // 8 KB
        alignas(16) T sD[CHUNK_SIZE * D_K]; // 8 KB: loads dY then dW; reused as s_da
    };
    static_assert(sizeof(SmemStorageChunkWYBwd) == 32768, "SmemStorageChunkWYBwd must be exactly 32 KB (32768 bytes)");
    __shared__ SmemStorageChunkWYBwd smem;

    using SwizzleOp = Swizzle<3, 3, 3>;

    auto smem_layout_flat_k = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto smem_layout_flat_v = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{})));

    // SMEM MMA layouts:
    auto smem_layout_mma_A = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto smem_layout_mma_B_Row = composition(SwizzleOp{}, make_layout(make_shape(Int<D_K>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto smem_layout_mma_B_Col = composition(SwizzleOp{}, make_layout(make_shape(Int<D_K>{}, Int<D_K>{}), make_stride(Int<1>{}, Int<D_K>{})));
    auto smem_layout_mma_C = composition(SwizzleOp{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

    auto sU_flat = make_tensor(make_smem_ptr(smem.sU), smem_layout_flat_k);
    auto sV_flat = make_tensor(make_smem_ptr(smem.sV), smem_layout_flat_v);
    auto sK_flat = make_tensor(make_smem_ptr(smem.sK), smem_layout_flat_k);
    auto sD_flat = make_tensor(make_smem_ptr(smem.sD), smem_layout_flat_k);

    auto sU_mma = make_tensor(make_smem_ptr(smem.sU), smem_layout_mma_A);
    auto sV_mma = make_tensor(make_smem_ptr(smem.sV), smem_layout_mma_A);
    auto sK_mma = make_tensor(make_smem_ptr(smem.sK), smem_layout_mma_A);

    auto sD_mma_Row = make_tensor(make_smem_ptr(smem.sD), smem_layout_mma_B_Row);
    auto sD_mma_Col = make_tensor(make_smem_ptr(smem.sD), smem_layout_mma_B_Col);

    auto sC_mma = make_tensor(make_smem_ptr(smem.sU), smem_layout_mma_C);

    TiledMMAType tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    auto s2r_copy_a     = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_copy_b_row = make_tiled_copy_B(S2RCopyAtomB_KMajor{}, tiled_mma);
    auto s2r_copy_b_col = make_tiled_copy_B(S2RCopyAtomB_MNMajor{}, tiled_mma);

    auto s2r_thr_copy_a     = s2r_copy_a.get_slice(tid);
    auto s2r_thr_copy_b_row = s2r_copy_b_row.get_slice(tid);
    auto s2r_thr_copy_b_col = s2r_copy_b_col.get_slice(tid);

    auto tCsA_K = s2r_thr_copy_a.partition_S(sK_mma);
    auto tCsA_V = s2r_thr_copy_a.partition_S(sV_mma);
    auto tCsA_U = s2r_thr_copy_a.partition_S(sU_mma);

    auto tCsB_Row = s2r_thr_copy_b_row.partition_S(sD_mma_Row);
    auto tCsB_Col = s2r_thr_copy_b_col.partition_S(sD_mma_Col);
    auto tCsC     = thr_mma.partition_C(sC_mma);

    auto tCrA = thr_mma.partition_fragment_A(sK_mma);
    auto tCrB = thr_mma.partition_fragment_B(sD_mma_Row);

    auto tCrA_view     = s2r_thr_copy_a.retile_D(tCrA);
    auto tCrB_row_view = s2r_thr_copy_b_row.retile_D(tCrB);
    auto tCrB_col_view = s2r_thr_copy_b_col.retile_D(tCrB);

    TiledCopyG2R tiled_copy_g2r;
    TiledCopyStore tiled_copy_store;
    auto thr_copy_g2r   = tiled_copy_g2r.get_thread_slice(tid);
    auto thr_copy_store = tiled_copy_store.get_thread_slice(tid);

    constexpr int ROWS_PER_STP = 16;
    constexpr int NUM_STPS     = CHUNK_SIZE / ROWS_PER_STP; // 4 steps

    int row_in_step = tid / (D_K / 8);
    int col_vec_idx = tid % (D_K / 8);

    if (valid_tokens <= 0) {
        return;
    }

    // Stage 1.1 - 2-Warp Prefix Scan for log_alpha_cumsum
    // Temporarily use smem.sD as exchange buffer
    float* s_temp_log_alpha = reinterpret_cast<float*>(smem.sD);
    float* s_temp_warp0_sum = reinterpret_cast<float*>(smem.sD) + CHUNK_SIZE;

    if (tid < CHUNK_SIZE) {
        float alpha_val = 0.0f;
        if (tid < valid_tokens) {
            int64_t token_idx = (int64_t)chunk_start_token + tid;
            alpha_val = stable_log_sigmoid(__half2float(to_half_val(gAlpha[head_offset + token_idx])));
        }
        #pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            float up = __shfl_up_sync(0xffffffff, alpha_val, offset);
            if ((tid & 31) >= offset) {
                alpha_val += up;
            }
        }
        if (tid == 31) {
            *s_temp_warp0_sum = alpha_val;
        }
        s_temp_log_alpha[tid] = alpha_val;
    }
    __syncthreads();

    if (tid >= 32 && tid < CHUNK_SIZE) {
        s_temp_log_alpha[tid] += *s_temp_warp0_sum;
    }
    __syncthreads();

    float chunk_last_val = (valid_tokens > 0) ? expf(s_temp_log_alpha[valid_tokens - 1]) : 1.0f;

    // Cache decay_t per thread step to free smem.sD
    float r_decay_t[NUM_STPS];
    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        int local_row = st * ROWS_PER_STP + row_in_step;
        r_decay_t[st] = (local_row < valid_tokens) ? expf(s_temp_log_alpha[valid_tokens - 1] - s_temp_log_alpha[local_row]) : 0.0f;
    }
    __syncthreads(); // Release smem.sD

    // Stage 2 - Forward Recompute: U, V_new, K_decay into sU, sV, sK
    const T* p_k_chunk      = gK      + head_offset * D_K + (int64_t)chunk_start_token * D_K;
    const T* p_v_chunk      = gV      + head_offset * D_V + (int64_t)chunk_start_token * D_V;
    const T* p_b_chunk      = gB      + head_offset * D_K + (int64_t)chunk_start_token * D_K;
    const T* p_w_gate_chunk = gW_gate + head_offset * D_V + (int64_t)chunk_start_token * D_V;
    const T* p_gamma_chunk  = gGamma  + head_offset;

    auto gK_chunk   = make_tensor(make_gmem_ptr(p_k_chunk), make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto gV_chunk   = make_tensor(make_gmem_ptr(p_v_chunk), make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{})));
    auto gB_chunk   = make_tensor(make_gmem_ptr(p_b_chunk), make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto gW_g_chk   = make_tensor(make_gmem_ptr(p_w_gate_chunk), make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{})));

    auto dummy_tile = local_tile(sU_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(0, 0));
    auto dummy_part = thr_copy_g2r.partition_S(dummy_tile);

    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        int local_row = st * ROWS_PER_STP + row_in_step;
        bool row_valid = (local_row < valid_tokens);
        int64_t token_idx = (int64_t)chunk_start_token + local_row;

        auto gK_step = local_tile(gK_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto gV_step = local_tile(gV_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_V>{}), make_coord(st, 0));
        auto gB_step = local_tile(gB_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto gW_step = local_tile(gW_g_chk, make_shape(Int<ROWS_PER_STP>{}, Int<D_V>{}), make_coord(st, 0));

        decltype(make_fragment_like(dummy_part)) rK, rB, rV, rW_g;
        clear(rK); clear(rB); clear(rV); clear(rW_g);

        decltype(make_fragment_like<bool>(dummy_part)) tPred;
        #pragma unroll
        for (int i = 0; i < size(tPred); ++i) tPred(i) = row_valid;

        cute::copy_if(tiled_copy_g2r, tPred, thr_copy_g2r.partition_S(gK_step), rK);
        cute::copy_if(tiled_copy_g2r, tPred, thr_copy_g2r.partition_S(gB_step), rB);
        cute::copy_if(tiled_copy_g2r, tPred, thr_copy_g2r.partition_S(gV_step), rV);
        cute::copy_if(tiled_copy_g2r, tPred, thr_copy_g2r.partition_S(gW_step), rW_g);

        float sqrt_gamma_f = 0.0f;
        if (row_valid) {
            __half g_val = to_half_val(p_gamma_chunk[token_idx]);
            __half sp    = fast_softplus_h(g_val);
            sqrt_gamma_f = sqrtf(fmaxf(__half2float(sp), 0.0f) + 1e-4f);
        }
        __half2 sqrt_gamma_h2 = __half2half2(__float2half(sqrt_gamma_f));

        float thread_sq_sum = 0.0f;
        if (row_valid) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float ki = half_to_float(rK(i));
                thread_sq_sum += ki * ki;
            }
        }
        #pragma unroll
        for (int mask = 4; mask > 0; mask /= 2) {
            thread_sq_sum += __shfl_xor_sync(0xffffffff, thread_sq_sum, mask);
        }
        float k_norm = fmaxf(sqrtf(fmaxf(thread_sq_sum, 0.0f)), 1e-4f);
        float inv_norm_scale = row_valid ? (sqrt_gamma_f / k_norm) : 0.0f;
        __half2 inv_norm_scale_h2 = __half2half2(__float2half(inv_norm_scale));

        decltype(make_fragment_like(dummy_part)) rK_scaled, rU, rV_new, rK_decay;
        half2* rK_h2        = reinterpret_cast<half2*>(&rK(0));
        half2* rK_scaled_h2 = reinterpret_cast<half2*>(&rK_scaled(0));
        half2* rB_h2        = reinterpret_cast<half2*>(&rB(0));
        half2* rU_h2        = reinterpret_cast<half2*>(&rU(0));
        half2* rV_h2        = reinterpret_cast<half2*>(&rV(0));
        half2* rW_h2        = reinterpret_cast<half2*>(&rW_g(0));
        half2* rV_new_h2    = reinterpret_cast<half2*>(&rV_new(0));

        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            rK_scaled_h2[p] = __hmul2(rK_h2[p], inv_norm_scale_h2);
            half2 b_sig     = fast_sigmoid_h2(rB_h2[p]);
            rU_h2[p]        = row_valid ? __hmul2(b_sig, rK_scaled_h2[p]) : __float2half2_rn(0.0f);

            half2 v_scaled  = __hmul2(rV_h2[p], sqrt_gamma_h2);
            half2 w_sig     = fast_sigmoid_h2(rW_h2[p]);
            rV_new_h2[p]    = row_valid ? __hmul2(w_sig, v_scaled) : __float2half2_rn(0.0f);
        }

        float decay_t = r_decay_t[st];
        __half2 decay_h2 = __half2half2(__float2half(decay_t));
        half2* rK_decay_h2 = reinterpret_cast<half2*>(&rK_decay(0));
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            rK_decay_h2[p] = row_valid ? __hmul2(rK_scaled_h2[p], decay_h2) : __float2half2_rn(0.0f);
        }

        // Store U, V_new, K_decay into SMEM
        auto sU_step = local_tile(sU_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto sV_step = local_tile(sV_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_V>{}), make_coord(st, 0));
        auto sK_step = local_tile(sK_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto tUsU    = thr_copy_g2r.partition_D(sU_step);
        auto tVsV    = thr_copy_g2r.partition_D(sV_step);
        auto tKsK    = thr_copy_g2r.partition_D(sK_step);
        *reinterpret_cast<uint4*>(&tUsU(0)) = *reinterpret_cast<const uint4*>(&rU(0));
        *reinterpret_cast<uint4*>(&tVsV(0)) = *reinterpret_cast<const uint4*>(&rV_new(0));
        *reinterpret_cast<uint4*>(&tKsK(0)) = *reinterpret_cast<const uint4*>(&rK_decay(0));
    }
    __syncthreads();

    // Stage 3 - GEMM with dY: dv_accum = K_decay @ dY^T and dk_Y = V_new @ dY
    int64_t mat_offset_y = ((int64_t)head_idx * num_chunks + chunk_idx) * ((int64_t)D_V * D_K);
    const T* p_dy_chunk = gdY + mat_offset_y;
    auto gDY_chunk = make_tensor(make_gmem_ptr(p_dy_chunk), make_layout(make_shape(Int<D_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        auto gDY_step = local_tile(gDY_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto sD_step  = local_tile(sD_flat,   make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        decltype(make_fragment_like(dummy_part)) rDY;
        cute::copy(tiled_copy_g2r, thr_copy_g2r.partition_S(gDY_step), rDY);
        auto tDsD = thr_copy_g2r.partition_D(sD_step);
        *reinterpret_cast<uint4*>(&tDsD(0)) = *reinterpret_cast<const uint4*>(&rDY(0));
    }
    __syncthreads();

    // 1. GEMM A: dv_accum = K_decay @ dY^T
    auto tCrC_DV = thr_mma.make_fragment_C(tCsC);
    clear(tCrC_DV);
    cute::copy(s2r_copy_a,     tCsA_K,   tCrA_view);
    cute::copy(s2r_copy_b_row, tCsB_Row, tCrB_row_view);
    cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC_DV);

    auto rC_half_DV = make_tensor_like<T>(tCrC_DV);
    #pragma unroll
    for (int i = 0; i < size(rC_half_DV); ++i) {
        rC_half_DV(i) = float_to_out<T>(tCrC_DV(i));
    }

    // 2. GEMM B: dk_Y = V_new @ dY
    auto tCrC_DK = thr_mma.make_fragment_C(tCsC);
    clear(tCrC_DK);
    cute::copy(s2r_copy_a,     tCsA_V,   tCrA_view);
    cute::copy(s2r_copy_b_col, tCsB_Col, tCrB_col_view);
    cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC_DK);
    __syncthreads();

    // Store dv_accum to sV (V_new is no longer needed)
    cute::copy(rC_half_DV, thr_mma.partition_C(make_tensor(make_smem_ptr(smem.sV), smem_layout_mma_C)));
    __syncthreads();

    // Stage 4 - GEMM with dW: du_accum = K_decay @ dW^T and tCrC_DK += (-U) @ dW
    int64_t mat_offset_w = ((int64_t)head_idx * num_chunks + chunk_idx) * ((int64_t)D_K * D_K);
    const T* p_dw_chunk = gdW + mat_offset_w;
    auto gDW_chunk = make_tensor(make_gmem_ptr(p_dw_chunk), make_layout(make_shape(Int<D_K>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        auto gDW_step = local_tile(gDW_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto sD_step  = local_tile(sD_flat,   make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        decltype(make_fragment_like(dummy_part)) rDW;
        cute::copy(tiled_copy_g2r, thr_copy_g2r.partition_S(gDW_step), rDW);
        auto tDsD = thr_copy_g2r.partition_D(sD_step);
        *reinterpret_cast<uint4*>(&tDsD(0)) = *reinterpret_cast<const uint4*>(&rDW(0));
    }
    __syncthreads();

    // 3. GEMM C: du_accum = K_decay @ dW^T -> du = -du_accum
    auto tCrC_DU = thr_mma.make_fragment_C(tCsC);
    clear(tCrC_DU);
    cute::copy(s2r_copy_a,     tCsA_K,   tCrA_view);
    cute::copy(s2r_copy_b_row, tCsB_Row, tCrB_row_view);
    cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC_DU);

    auto rC_half_DU = make_tensor_like<T>(tCrC_DU);
    #pragma unroll
    for (int i = 0; i < size(rC_half_DU); ++i) {
        rC_half_DU(i) = float_to_out<T>(-tCrC_DU(i)); // du = -du_accum
    }

    // 4. GEMM D: tCrC_DK += (-U) @ dW -> dk_decay = V_new @ dY - U @ dW
    cute::copy(s2r_copy_a,     tCsA_U,   tCrA_view);
    cute::copy(s2r_copy_b_col, tCsB_Col, tCrB_col_view);
    #pragma unroll
    for (int i = 0; i < size(tCrA); ++i) {
        tCrA(i) = -tCrA(i); // Negate -U registers
    }
    cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC_DK);
    __syncthreads();

    auto rC_half_DK = make_tensor_like<T>(tCrC_DK);
    #pragma unroll
    for (int i = 0; i < size(rC_half_DK); ++i) {
        rC_half_DK(i) = float_to_out<T>(tCrC_DK(i));
    }

    // Store du into sU and dk_decay into sK
    cute::copy(rC_half_DU, thr_mma.partition_C(make_tensor(make_smem_ptr(smem.sU), smem_layout_mma_C)));
    cute::copy(rC_half_DK, thr_mma.partition_C(make_tensor(make_smem_ptr(smem.sK), smem_layout_mma_C)));
    __syncthreads();

    // Stage 5 - Recurrence Epilogue & Backward Activations
    // Reuse smem.sD as buffer for s_da and s_warp_shared
    float* s_da          = reinterpret_cast<float*>(smem.sD);
    float* s_warp_shared = reinterpret_cast<float*>(smem.sD) + CHUNK_SIZE;

    const T* p_dk_ext = gDK_ext + head_offset * D_K + (int64_t)chunk_start_token * D_K;
    const T* p_dv_ext = gDV_ext + head_offset * D_V + (int64_t)chunk_start_token * D_V;

    auto gDK_ext_chk = make_tensor(make_gmem_ptr(p_dk_ext), make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto gDV_ext_chk = make_tensor(make_gmem_ptr(p_dv_ext), make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{})));

    T* p_dk_out = gDK + head_offset * D_K + (int64_t)chunk_start_token * D_K;
    T* p_dv_out = gDV + head_offset * D_V + (int64_t)chunk_start_token * D_V;
    T* p_db_out = gDB + head_offset * D_K + (int64_t)chunk_start_token * D_K;
    T* p_dw_out = gDW + head_offset * D_V + (int64_t)chunk_start_token * D_V;

    auto gDK_out_chk = make_tensor(make_gmem_ptr(p_dk_out), make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto gDV_out_chk = make_tensor(make_gmem_ptr(p_dv_out), make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{})));
    auto gDB_out_chk = make_tensor(make_gmem_ptr(p_db_out), make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto gDW_out_chk = make_tensor(make_gmem_ptr(p_dw_out), make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_V>{}), make_stride(Int<D_V>{}, Int<1>{})));

    #pragma unroll
    for (int st = 0; st < NUM_STPS; ++st) {
        int local_row = st * ROWS_PER_STP + row_in_step;
        bool row_valid = (local_row < valid_tokens);
        int64_t token_idx = (int64_t)chunk_start_token + local_row;

        // Load du, dv_accum, dk_decay from SMEM
        auto sU_step = local_tile(sU_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto sV_step = local_tile(sV_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_V>{}), make_coord(st, 0));
        auto sK_step = local_tile(sK_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));

        decltype(make_fragment_like(dummy_part)) rDU, rDV_accum, rDK_decay;
        cute::copy(thr_copy_store.partition_S(sU_step), rDU);
        cute::copy(thr_copy_store.partition_S(sV_step), rDV_accum);
        cute::copy(thr_copy_store.partition_S(sK_step), rDK_decay);

        // Load external gradients dk_ext, dv_ext from GMEM
        auto gDK_ext_step = local_tile(gDK_ext_chk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto gDV_ext_step = local_tile(gDV_ext_chk, make_shape(Int<ROWS_PER_STP>{}, Int<D_V>{}), make_coord(st, 0));

        decltype(make_fragment_like(dummy_part)) rDK_ext, rDV_ext;
        clear(rDK_ext); clear(rDV_ext);
        decltype(make_fragment_like<bool>(dummy_part)) tPred;
        #pragma unroll
        for (int i = 0; i < size(tPred); ++i) tPred(i) = row_valid;

        cute::copy_if(tiled_copy_g2r, tPred, thr_copy_g2r.partition_S(gDK_ext_step), rDK_ext);
        cute::copy_if(tiled_copy_g2r, tPred, thr_copy_g2r.partition_S(gDV_ext_step), rDV_ext);

        // Reload raw inputs K, V, B, W
        auto gK_step = local_tile(gK_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto gV_step = local_tile(gV_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_V>{}), make_coord(st, 0));
        auto gB_step = local_tile(gB_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto gW_step = local_tile(gW_g_chk, make_shape(Int<ROWS_PER_STP>{}, Int<D_V>{}), make_coord(st, 0));

        decltype(make_fragment_like(dummy_part)) rK, rB, rV, rW_g;
        clear(rK); clear(rB); clear(rV); clear(rW_g);

        cute::copy_if(tiled_copy_g2r, tPred, thr_copy_g2r.partition_S(gK_step), rK);
        cute::copy_if(tiled_copy_g2r, tPred, thr_copy_g2r.partition_S(gB_step), rB);
        cute::copy_if(tiled_copy_g2r, tPred, thr_copy_g2r.partition_S(gV_step), rV);
        cute::copy_if(tiled_copy_g2r, tPred, thr_copy_g2r.partition_S(gW_step), rW_g);

        // 1. Recompute forward scalars: sqrt_gamma, k_norm, inv_norm_scale, decay_t
        float sqrt_gamma_f = 0.0f;
        float gamma_sp_f   = 0.0f;
        __half gamma_raw   = __float2half(0.0f);
        if (row_valid) {
            gamma_raw    = to_half_val(p_gamma_chunk[token_idx]);
            __half sp    = fast_softplus_h(gamma_raw);
            gamma_sp_f   = __half2float(sp);
            sqrt_gamma_f = sqrtf(fmaxf(gamma_sp_f, 0.0f) + 1e-4f);
        }
        __half2 sqrt_gamma_h2 = __half2half2(__float2half(sqrt_gamma_f));

        float thread_sq_sum = 0.0f;
        if (row_valid) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float ki = half_to_float(rK(i));
                thread_sq_sum += ki * ki;
            }
        }
        #pragma unroll
        for (int mask = 4; mask > 0; mask /= 2) {
            thread_sq_sum += __shfl_xor_sync(0xffffffff, thread_sq_sum, mask);
        }
        float k_norm = fmaxf(sqrtf(fmaxf(thread_sq_sum, 0.0f)), 1e-4f);
        float inv_norm_scale = row_valid ? (sqrt_gamma_f / k_norm) : 0.0f;
        __half2 inv_norm_scale_h2 = __half2half2(__float2half(inv_norm_scale));

        float decay_t = r_decay_t[st];
        __half2 decay_h2 = __half2half2(__float2half(decay_t));

        // 2. dv_total = dv_accum + dv_ext
        decltype(make_fragment_like(dummy_part)) rDV_total;
        half2* rDV_acc_h2 = reinterpret_cast<half2*>(&rDV_accum(0));
        half2* rDV_ext_h2 = reinterpret_cast<half2*>(&rDV_ext(0));
        half2* rDV_tot_h2 = reinterpret_cast<half2*>(&rDV_total(0));
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            rDV_tot_h2[p] = row_valid ? __hadd2(rDV_acc_h2[p], rDV_ext_h2[p]) : __float2half2_rn(0.0f);
        }

        // 3. Forward K_scaled, U, V_new, K_decay
        decltype(make_fragment_like(dummy_part)) rK_scaled, rK_decay, rU, rV_new;
        decltype(make_fragment_like(dummy_part)) rSig_B, rSig_W;
        half2* rK_h2        = reinterpret_cast<half2*>(&rK(0));
        half2* rK_sc_h2     = reinterpret_cast<half2*>(&rK_scaled(0));
        half2* rB_h2        = reinterpret_cast<half2*>(&rB(0));
        half2* rSig_B_h2    = reinterpret_cast<half2*>(&rSig_B(0));
        half2* rU_h2        = reinterpret_cast<half2*>(&rU(0));
        half2* rV_h2        = reinterpret_cast<half2*>(&rV(0));
        half2* rW_h2        = reinterpret_cast<half2*>(&rW_g(0));
        half2* rSig_W_h2    = reinterpret_cast<half2*>(&rSig_W(0));
        half2* rV_new_h2    = reinterpret_cast<half2*>(&rV_new(0));
        half2* rK_dec_h2    = reinterpret_cast<half2*>(&rK_decay(0));

        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            rK_sc_h2[p]  = __hmul2(rK_h2[p], inv_norm_scale_h2);
            rSig_B_h2[p] = fast_sigmoid_h2(rB_h2[p]);
            rU_h2[p]     = __hmul2(rSig_B_h2[p], rK_sc_h2[p]);

            half2 v_sc   = __hmul2(rV_h2[p], sqrt_gamma_h2);
            rSig_W_h2[p] = fast_sigmoid_h2(rW_h2[p]);
            rV_new_h2[p] = __hmul2(rSig_W_h2[p], v_sc);

            rK_dec_h2[p] = __hmul2(rK_sc_h2[p], decay_h2);
        }

        // 4. dk_total = dk_decay * decay_t + du * sigmoid(B) + dk_ext
        decltype(make_fragment_like(dummy_part)) rDK_total;
        half2* rDK_dec_h2 = reinterpret_cast<half2*>(&rDK_decay(0));
        half2* rDU_h2     = reinterpret_cast<half2*>(&rDU(0));
        half2* rDK_ext_h2 = reinterpret_cast<half2*>(&rDK_ext(0));
        half2* rDK_tot_h2 = reinterpret_cast<half2*>(&rDK_total(0));
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            half2 dk_d = __hmul2(rDK_dec_h2[p], decay_h2);
            half2 dk_u = __hmul2(rDU_h2[p], rSig_B_h2[p]);
            rDK_tot_h2[p] = row_valid ? __hadd2(__hadd2(dk_d, dk_u), rDK_ext_h2[p]) : __float2half2_rn(0.0f);
        }

        // 5. Term_t = dk_decay * k_decay for log_alpha gradient
        float thread_term_sum = 0.0f;
        if (row_valid) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float dkd = half_to_float(rDK_decay(i));
                float kd  = half_to_float(rK_decay(i));
                thread_term_sum += dkd * kd;
            }
        }
        #pragma unroll
        for (int mask = 4; mask > 0; mask /= 2) {
            thread_term_sum += __shfl_xor_sync(0xffffffff, thread_term_sum, mask);
        }
        if (col_vec_idx == 0 && row_valid) {
            s_da[local_row] = -thread_term_sum;
        }

        // 6. db_raw = du * u * (1 - sigmoid(b))
        decltype(make_fragment_like(dummy_part)) rDB;
        half2* rDB_h2 = reinterpret_cast<half2*>(&rDB(0));
        half2 one_h2  = __float2half2_rn(1.0f);
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            half2 sig_grad = __hsub2(one_h2, rSig_B_h2[p]);
            rDB_h2[p] = row_valid ? __hmul2(__hmul2(rDU_h2[p], rU_h2[p]), sig_grad) : __float2half2_rn(0.0f);
        }

        // 7. dw_raw = dv_total * v_new * (1 - sigmoid(w))
        decltype(make_fragment_like(dummy_part)) rDW;
        half2* rDW_out_h2 = reinterpret_cast<half2*>(&rDW(0));
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            half2 sig_grad = __hsub2(one_h2, rSig_W_h2[p]);
            rDW_out_h2[p] = row_valid ? __hmul2(__hmul2(rDV_tot_h2[p], rV_new_h2[p]), sig_grad) : __float2half2_rn(0.0f);
        }

        // 8. dv_raw = (dv_total * sigmoid(w)) * sqrt_gamma
        decltype(make_fragment_like(dummy_part)) rDV;
        half2* rDV_out_h2 = reinterpret_cast<half2*>(&rDV(0));
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            half2 dv_sc = __hmul2(rDV_tot_h2[p], rSig_W_h2[p]);
            rDV_out_h2[p] = row_valid ? __hmul2(dv_sc, sqrt_gamma_h2) : __float2half2_rn(0.0f);
        }

        // 9. D_K and D_V for L2-norm and softplus backward
        float thread_dot_k = 0.0f;
        float thread_dot_v = 0.0f;
        if (row_valid) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                thread_dot_k += half_to_float(rDK_total(i)) * half_to_float(rK_scaled(i));
                thread_dot_v += half_to_float(rDV_total(i)) * half_to_float(rV_new(i));
            }
        }
        #pragma unroll
        for (int mask = 4; mask > 0; mask /= 2) {
            thread_dot_k += __shfl_xor_sync(0xffffffff, thread_dot_k, mask);
            thread_dot_v += __shfl_xor_sync(0xffffffff, thread_dot_v, mask);
        }

        // 10. dk_raw = (1/k_norm) * (sqrt_gamma * dk_total - k_scaled * (D_K / sqrt_gamma))
        decltype(make_fragment_like(dummy_part)) rDK;
        float scale_dk_term = row_valid ? (thread_dot_k / fmaxf(sqrt_gamma_f, 1e-4f)) : 0.0f;
        __half2 scale_dk_h2 = __half2half2(__float2half(scale_dk_term));
        float inv_norm_k    = row_valid ? (1.0f / k_norm) : 0.0f;
        __half2 inv_norm_k2 = __half2half2(__float2half(inv_norm_k));

        half2* rDK_out_h2 = reinterpret_cast<half2*>(&rDK(0));
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            half2 term1 = __hmul2(sqrt_gamma_h2, rDK_tot_h2[p]);
            half2 term2 = __hmul2(rK_sc_h2[p], scale_dk_h2);
            rDK_out_h2[p] = row_valid ? __hmul2(__hsub2(term1, term2), inv_norm_k2) : __float2half2_rn(0.0f);
        }

        // 11. dgamma_raw: leader thread computes and writes directly
        if (col_vec_idx == 0 && row_valid) {
            float sig_gamma = __half2float(fast_sigmoid_h(gamma_raw));
            float denom     = fmaxf(2.0f * gamma_sp_f, 1e-4f);
            float dgamma    = ((thread_dot_k + thread_dot_v) / denom) * sig_gamma;
            gDGamma[head_offset + token_idx] = float_to_out<T>(dgamma);
        }

        // Store db, dw, dv, dk to GMEM
        auto gDB_step = local_tile(gDB_out_chk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
        auto gDW_step = local_tile(gDW_out_chk, make_shape(Int<ROWS_PER_STP>{}, Int<D_V>{}), make_coord(st, 0));
        auto gDV_step = local_tile(gDV_out_chk, make_shape(Int<ROWS_PER_STP>{}, Int<D_V>{}), make_coord(st, 0));
        auto gDK_step = local_tile(gDK_out_chk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));

        cute::copy_if(tiled_copy_store, tPred, rDB, thr_copy_store.partition_D(gDB_step));
        cute::copy_if(tiled_copy_store, tPred, rDW, thr_copy_store.partition_D(gDW_step));
        cute::copy_if(tiled_copy_store, tPred, rDV, thr_copy_store.partition_D(gDV_step));
        cute::copy_if(tiled_copy_store, tPred, rDK, thr_copy_store.partition_D(gDK_step));
    }
    __syncthreads();

    // Stage 6 - Gradient Alpha (dalpha_raw) via 2-Warp Reverse Suffix Scan
    if (tid == 0) {
        float total_term_sum = 0.0f;
        for (int t = 0; t < valid_tokens; ++t) {
            total_term_sum += (-s_da[t]);
        }
        float d_trace_val = gD_trace[head_idx * num_chunks + chunk_idx];
        s_da[valid_tokens - 1] += total_term_sum + d_trace_val * chunk_last_val;
    }
    __syncthreads();

    if (tid < valid_tokens) {
        float da_ext_val = gDA_ext[head_offset + chunk_start_token + tid];
        s_da[tid] += da_ext_val;
    }
    __syncthreads();

    // 2-Warp Reverse Suffix Scan
    float da_val = 0.0f;
    if (tid < CHUNK_SIZE) {
        da_val = (tid < valid_tokens) ? s_da[tid] : 0.0f;
        #pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            float down = __shfl_down_sync(0xffffffff, da_val, offset);
            if ((tid & 31) + offset < 32) {
                da_val += down;
            }
        }
        if (tid == 32) {
            *s_warp_shared = da_val; // Suffix sum of Warp 1 (tokens 32..63)
        }
    }
    __syncthreads();

    if (tid < 32) {
        da_val += *s_warp_shared; // Warp 0 adds suffix sum of Warp 1
    }
    __syncthreads();

    if (tid < valid_tokens) {
        float a_raw = __half2float(to_half_val(gAlpha[head_offset + chunk_start_token + tid]));
        float sig_a = 1.0f / (1.0f + expf(-fmaxf(fminf(a_raw, 10.0f), -10.0f)));
        float d_alpha = da_val * (1.0f - sig_a);
        gDAlpha[head_offset + chunk_start_token + tid] = float_to_out<T>(d_alpha);
    }
}

// HOST FUNCTION (chunk_wy_bwd)

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
    const at::Tensor& da_ext,          // (num_kv_heads, total_L, 1) [FP32] - External log alpha gradient
    int64_t chunk_size                 // scalar [int64] - Chunk size (default 64)
) {
    // Stage 1 - Input Validation
    at::cuda::CUDAGuard device_guard(k.device());

    TORCH_CHECK(dW.is_cuda() && dY.is_cuda(), "dW and dY must be CUDA tensors");
    TORCH_CHECK(k.is_cuda() && v.is_cuda() && b.is_cuda() && w.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(gamma.is_cuda() && a.is_cuda() && cu_seq_lens.is_cuda(), "Gamma, a, cu_seq_lens must be CUDA tensors");
    TORCH_CHECK(dk_ext.is_cuda() && dv_ext.is_cuda() && da_ext.is_cuda(), "Ext gradients must be CUDA tensors");

    TORCH_CHECK(dW.is_contiguous(), "dW must be contiguous");
    TORCH_CHECK(dY.is_contiguous(), "dY must be contiguous");
    TORCH_CHECK(k.is_contiguous() && v.is_contiguous() && b.is_contiguous() && w.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(gamma.is_contiguous() && a.is_contiguous(), "Gamma and a must be contiguous");
    TORCH_CHECK(dk_ext.is_contiguous() && dv_ext.is_contiguous() && da_ext.is_contiguous(), "Ext gradients must be contiguous");

    TORCH_CHECK(dW.scalar_type() == at::ScalarType::Half, "dW must be float16");
    TORCH_CHECK(dY.scalar_type() == at::ScalarType::Half, "dY must be float16");
    TORCH_CHECK(k.scalar_type() == at::ScalarType::Half, "k must be float16");
    TORCH_CHECK(v.scalar_type() == at::ScalarType::Half, "v must be float16");
    TORCH_CHECK(b.scalar_type() == at::ScalarType::Half, "b must be float16");
    TORCH_CHECK(w.scalar_type() == at::ScalarType::Half, "w must be float16");
    TORCH_CHECK(gamma.scalar_type() == at::ScalarType::Half, "gamma must be float16");
    TORCH_CHECK(a.scalar_type() == at::ScalarType::Half, "a must be float16");
    TORCH_CHECK(dk_ext.scalar_type() == at::ScalarType::Half, "dk_ext must be float16");
    TORCH_CHECK(dv_ext.scalar_type() == at::ScalarType::Half, "dv_ext must be float16");
    TORCH_CHECK(da_ext.scalar_type() == at::ScalarType::Float, "da_ext must be float32");

    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64 for SM75 Tensor Core alignment");

    int64_t num_kv_heads = k.size(0);
    int64_t total_L      = k.size(1);
    int64_t d_k          = k.size(2);
    int64_t d_v          = v.size(2);

    TORCH_CHECK(d_k == 64, "d_k must be 64");
    TORCH_CHECK(d_v == 64, "d_v must be 64");

    // Stage 2 - Kernel Launch
    auto [chunk_offsets_gpu, chunk_valid_lens_gpu, num_chunks] = compute_chunk_offsets_and_valid_lens(cu_seq_lens, chunk_size);

    auto options = k.options();
    if (num_chunks <= 0) {
        at::Tensor dk     = at::empty_like(k);
        at::Tensor dv     = at::empty_like(v);
        at::Tensor db     = at::empty_like(b);
        at::Tensor dw     = at::empty_like(w);
        at::Tensor dgamma = at::empty_like(gamma);
        at::Tensor da     = at::empty_like(a);
        return std::make_tuple(dk, dv, db, dw, dgamma, da);
    }

    at::Tensor d_trace = dW.diagonal(0, -2, -1).to(at::kFloat).sum(-1).contiguous();

    at::Tensor dk     = torch::empty_like(k);
    at::Tensor dv     = torch::empty_like(v);
    at::Tensor db     = torch::empty_like(b);
    at::Tensor dw     = torch::empty_like(w);
    at::Tensor dgamma = torch::empty_like(gamma);
    at::Tensor da     = torch::empty_like(a);

    constexpr int NUM_THREADS = 128;
    constexpr int CHUNK_SIZE_CONST = 64;
    constexpr int D_K_CONST = 64;
    constexpr int D_V_CONST = 64;

    dim3 grid(num_chunks, num_kv_heads);
    dim3 block(NUM_THREADS);

    using Element = cutlass::half_t;
    using MMAAtom = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
    using TiledMMAType = TiledMMA<MMAAtom, Layout<Shape<_2, _2, _1>, Stride<_1, _2, _4>>>;

    using S2RCopyAtomA       = Copy_Atom<SM75_U32x2_LDSM_N, Element>;
    using S2RCopyAtomB_KMaj  = Copy_Atom<SM75_U32x1_LDSM_N, Element>;
    using S2RCopyAtomB_MNMaj = Copy_Atom<SM75_U16x2_LDSM_T, Element>;

    using CopyAtomG2R = Copy_Atom<UniversalCopy<uint4>, Element>;
    using TiledCopyG2R = decltype(make_tiled_copy(
        CopyAtomG2R{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>, Stride<_8, _1>>{}
    ));

    using CopyAtomStore = Copy_Atom<UniversalCopy<uint4>, Element>;
    using TiledCopyStore = decltype(make_tiled_copy(
        CopyAtomStore{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>, Stride<_8, _1>>{}
    ));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    {
        at::Tensor dW_a     = ensure_aligned16(dW);
        at::Tensor dY_a     = ensure_aligned16(dY);
        at::Tensor k_a      = ensure_aligned16(k);
        at::Tensor v_a      = ensure_aligned16(v);
        at::Tensor b_a      = ensure_aligned16(b);
        at::Tensor w_a      = ensure_aligned16(w);
        at::Tensor dk_ext_a = ensure_aligned16(dk_ext);
        at::Tensor dv_ext_a = ensure_aligned16(dv_ext);

        kernel_chunk_wy_bwd<
            Element,
            NUM_THREADS,
            CHUNK_SIZE_CONST,
            D_K_CONST,
            D_V_CONST,
            MMAAtom,
            TiledMMAType,
            S2RCopyAtomA,
            S2RCopyAtomB_KMaj,
            S2RCopyAtomB_MNMaj,
            TiledCopyG2R,
            TiledCopyStore
        ><<<grid, block, 0, stream>>>(
            reinterpret_cast<const Element*>(dW_a.data_ptr<at::Half>()),
            reinterpret_cast<const Element*>(dY_a.data_ptr<at::Half>()),
            reinterpret_cast<const Element*>(k_a.data_ptr<at::Half>()),
            reinterpret_cast<const Element*>(v_a.data_ptr<at::Half>()),
            reinterpret_cast<const Element*>(b_a.data_ptr<at::Half>()),
            reinterpret_cast<const Element*>(w_a.data_ptr<at::Half>()),
            reinterpret_cast<const Element*>(gamma.data_ptr<at::Half>()),
            reinterpret_cast<const Element*>(a.data_ptr<at::Half>()),
            reinterpret_cast<const Element*>(dk_ext_a.data_ptr<at::Half>()),
            reinterpret_cast<const Element*>(dv_ext_a.data_ptr<at::Half>()),
            da_ext.data_ptr<float>(),
            d_trace.data_ptr<float>(),
            reinterpret_cast<Element*>(dk.data_ptr<at::Half>()),
            reinterpret_cast<Element*>(dv.data_ptr<at::Half>()),
            reinterpret_cast<Element*>(db.data_ptr<at::Half>()),
            reinterpret_cast<Element*>(dw.data_ptr<at::Half>()),
            reinterpret_cast<Element*>(dgamma.data_ptr<at::Half>()),
            reinterpret_cast<Element*>(da.data_ptr<at::Half>()),
            chunk_offsets_gpu.data_ptr<int32_t>(),
            total_L,
            num_chunks
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Stage 3 - Return Results
    return std::make_tuple(dk, dv, db, dw, dgamma, da);
}
