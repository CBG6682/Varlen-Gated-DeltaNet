#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <optional>
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
// Part 1: Global Kernel (kernel_chunk_scan_fwd)
// ============================================================================

template <
    typename T,
    int NUM_THREADS,
    int CHUNK_SIZE,
    int D_K,
    int D_V,
    int B_V,
    typename MMAAtom,
    typename TiledMMA411,
    typename TiledMMA141,
    typename S2RCopyAtomA_KM,
    typename S2RCopyAtomB_KM,
    typename S2RCopyAtomA_MNM,
    typename S2RCopyAtomB_MNM,
    typename TiledCopyHalf,
    typename TiledCopyR,
    typename TiledCopyFloat
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_chunk_scan_fwd(
    const T*       __restrict__ gY,               // (H_kv, num_chunks, 64, 64) [FP16] - Erase WY auxiliary
    T*                          gU_R,             // (H_kv, num_chunks, 64, 64) [FP16] - In: U, Out in-place: R
    const T*       __restrict__ gK_h,             // (H_kv, total_L, 64) [FP16] - Normalized Key
    const float*   __restrict__ gG_log2_h,        // (H_kv, total_L, 64) [FP32] - Chunk cumsum log2 decay
    const float*   __restrict__ gStateOld,        // (B, H_kv, 64, 64) [FP32] - Initial state (nullable)
    const int32_t* __restrict__ cu_chunk_lens,    // (B + 1,) [Int32] - Cumulative chunk offsets
    const int32_t* __restrict__ chunk_offsets,    // (num_chunks + 1,) [Int32] - Chunk token starts
    const int32_t* __restrict__ chunk_valid_lens, // (num_chunks,) [Int32] - Valid tokens per chunk
    float*         __restrict__ gS_list,          // (H_kv, num_chunks, 64, 64) [FP32] - Per-chunk initial states
    float*         __restrict__ gFinalStates,     // (B, H_kv, 64, 64) [FP32] - Final recurrent states
    bool    use_state_old,                        // scalar [bool] - Initialize from gStateOld when true
    int64_t total_L,                              // scalar [int64] - Total packed tokens
    int64_t num_chunks                            // scalar [int64] - Total varlen chunks
) {
    // 1. Setup & SMEM Layouts (16.0 KB <= 32 KB)
    int batch_idx = blockIdx.x;
    int head_idx  = blockIdx.y;
    int v_idx     = blockIdx.z;
    int tid       = threadIdx.x;

    int row_id   = tid / 8;
    int col_base = (tid % 8) * 8;

    int32_t start_c = cu_chunk_lens[batch_idx];
    int32_t end_c   = cu_chunk_lens[batch_idx + 1];

    int64_t mat_elements_vk = static_cast<int64_t>(D_V) * D_K;
    int64_t mat_elements_ck = static_cast<int64_t>(CHUNK_SIZE) * D_K;
    int64_t mat_elements_cv = static_cast<int64_t>(CHUNK_SIZE) * D_V;
    int64_t slice_offset_v  = static_cast<int64_t>(v_idx) * B_V * D_K;
    int64_t state_offset    = (static_cast<int64_t>(batch_idx) * gridDim.y + head_idx) * mat_elements_vk + slice_offset_v;
    int64_t head_tok_base   = static_cast<int64_t>(head_idx) * total_L;

    __shared__ alignas(16) float sS_f32[B_V * D_K];      // 4 KB
    __shared__ alignas(16) T     sS_f16[B_V * D_K];      // 2 KB
    __shared__ alignas(16) T     sMat[CHUNK_SIZE * D_K]; // 8 KB
    __shared__ alignas(16) T     sR[CHUNK_SIZE * B_V];   // 2 KB

    using SwizzleF32  = Swizzle<3, 2, 3>;
    using SwizzleHalf = Swizzle<3, 3, 3>;
    using SwizzleR    = Swizzle<1, 3, 3>;

    auto layout_s_f32   = composition(SwizzleF32{},  make_layout(make_shape(Int<B_V>{}, Int<D_K>{}),        make_stride(Int<D_K>{}, Int<1>{})));
    auto layout_s_f16   = composition(SwizzleHalf{}, make_layout(make_shape(Int<B_V>{}, Int<D_K>{}),        make_stride(Int<D_K>{}, Int<1>{})));
    auto layout_mat_km  = composition(SwizzleHalf{}, make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto layout_mat_mnm = composition(SwizzleHalf{}, make_layout(make_shape(Int<D_K>{}, Int<CHUNK_SIZE>{}), make_stride(Int<1>{}, Int<D_K>{})));
    // Swizzle<1, 3, 3> flips bit 3 every 4 rows to avoid bank conflicts on SM75_U16x4_LDSM_T
    auto layout_r_flat  = composition(SwizzleR{},    make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<B_V>{}), make_stride(Int<B_V>{}, Int<1>{})));
    auto layout_r_mnm   = composition(SwizzleR{},    make_layout(make_shape(Int<B_V>{}, Int<CHUNK_SIZE>{}), make_stride(Int<1>{}, Int<B_V>{})));

    auto sS_f32_flat = make_tensor(make_smem_ptr(sS_f32), layout_s_f32);
    auto sS_f16_flat = make_tensor(make_smem_ptr(sS_f16), layout_s_f16);
    auto sMat_flat   = make_tensor(make_smem_ptr(sMat),   layout_mat_km);
    auto sMat_mnm    = make_tensor(make_smem_ptr(sMat),   layout_mat_mnm);
    auto sR_flat     = make_tensor(make_smem_ptr(sR),     layout_r_flat);
    auto sR_mnm      = make_tensor(make_smem_ptr(sR),     layout_r_mnm);

    TiledCopyHalf  tiled_copy_h;
    TiledCopyR     tiled_copy_r;
    TiledCopyFloat tiled_copy_f32;
    auto thr_copy_h   = tiled_copy_h.get_slice(tid);
    auto thr_copy_r   = tiled_copy_r.get_slice(tid);
    auto thr_copy_f32 = tiled_copy_f32.get_slice(tid);

    auto tXsS_f32_s = thr_copy_f32.partition_S(sS_f32_flat);
    auto tXsS_f32_d = thr_copy_f32.partition_D(sS_f32_flat);
    auto tXsS_f16_d = thr_copy_h.partition_D(sS_f16_flat);
    auto tXsMat_d   = thr_copy_h.partition_D(sMat_flat);
    auto tXsR_s     = thr_copy_r.partition_S(sR_flat);
    auto tXsR_d     = thr_copy_r.partition_D(sR_flat);

    auto rS_f32   = make_fragment_like(tXsS_f32_s);
    auto rS_f16   = make_fragment_like(tXsS_f16_d);
    auto rR_slice = make_fragment_like(tXsR_s);
    auto tCrMat   = make_fragment_like(tXsMat_d);

    auto layout_vk_slice  = make_layout(make_shape(Int<B_V>{}, Int<D_K>{}),        make_stride(Int<D_K>{}, Int<1>{}));
    auto layout_ck_full   = make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{}));
    auto layout_cv_slice  = make_layout(make_shape(Int<CHUNK_SIZE>{}, Int<B_V>{}), make_stride(Int<D_V>{}, Int<1>{}));

    auto gFinal_slice = make_tensor(make_gmem_ptr(gFinalStates + state_offset), layout_vk_slice);
    auto tXgFinal     = thr_copy_f32.partition_D(gFinal_slice);

    // 2. Empty sequence handling
    if (start_c >= end_c) {
        if (use_state_old && gStateOld != nullptr) {
            auto gOld_slice = make_tensor(make_gmem_ptr(gStateOld + state_offset), layout_vk_slice);
            cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gOld_slice), rS_f32);
        } else {
            clear(rS_f32);
        }
        cute::copy(tiled_copy_f32, rS_f32, tXgFinal);
        return;
    }

    // Declare G-tile fragments and boundary predicates once outside the chunk loop
    auto cTile64    = make_identity_tensor(make_shape(Int<CHUNK_SIZE>{}, Int<D_K>{}));
    auto tXcK       = thr_copy_h.partition_S(cTile64);
    auto tXcG       = thr_copy_f32.partition_S(cTile64);
    auto tCrG       = make_tensor<float>(shape(tXcG));
    auto tPredK     = make_fragment_like<bool>(tXsMat_d);
    auto tPredG     = make_tensor<bool>(shape(tXcG));

    int32_t last_valid_len = chunk_valid_lens[end_c - 1];
    if (last_valid_len < CHUNK_SIZE) {
        #pragma unroll
        for (int m = 0; m < size<1>(tPredK); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tPredK); ++n) {
                bool in_bounds = (get<0>(tXcK(Int<0>{}, m, n)) < last_valid_len);
                #pragma unroll
                for (int v = 0; v < size<0>(tPredK); ++v) {
                    tPredK(v, m, n) = in_bounds;
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < size<1>(tPredG); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tPredG); ++n) {
                bool in_bounds = (get<0>(tXcG(Int<0>{}, m, n)) < last_valid_len);
                #pragma unroll
                for (int v = 0; v < size<0>(tPredG); ++v) {
                    tPredG(v, m, n) = in_bounds;
                }
            }
        }
    }

    // Shared GEMM operand & accumulator fragments (identical shapes on TiledMMA411 and TiledMMA141)
    TiledMMA411 tiled_mma_1;
    TiledMMA141 tiled_mma_2;
    auto thr_mma_1 = tiled_mma_1.get_slice(tid);
    auto thr_mma_2 = tiled_mma_2.get_slice(tid);

    auto s2r_copy_a1 = make_tiled_copy_A(S2RCopyAtomA_KM{},  tiled_mma_1);
    auto s2r_copy_b1 = make_tiled_copy_B(S2RCopyAtomB_KM{},  tiled_mma_1);
    auto s2r_copy_a2 = make_tiled_copy_A(S2RCopyAtomA_MNM{}, tiled_mma_2);
    auto s2r_copy_b2 = make_tiled_copy_B(S2RCopyAtomB_MNM{}, tiled_mma_2);

    auto s2r_thr_a1 = s2r_copy_a1.get_slice(tid);
    auto s2r_thr_b1 = s2r_copy_b1.get_slice(tid);
    auto s2r_thr_a2 = s2r_copy_a2.get_slice(tid);
    auto s2r_thr_b2 = s2r_copy_b2.get_slice(tid);

    auto tCsA_1 = s2r_thr_a1.partition_S(sMat_flat);
    auto tCsB_1 = s2r_thr_b1.partition_S(sS_f16_flat);
    auto tCsC_R = thr_mma_1.partition_C(sR_flat);

    auto tCsA_2 = s2r_thr_a2.partition_S(sR_mnm);
    auto tCsB_2 = s2r_thr_b2.partition_S(sMat_mnm);
    auto tCsC_S = thr_mma_2.partition_C(sS_f32_flat);
    auto tCcC_S = thr_mma_2.partition_C(make_identity_tensor(make_shape(Int<B_V>{}, Int<D_K>{})));

    auto tCrA        = thr_mma_1.partition_fragment_A(sMat_flat);
    auto tCrB        = thr_mma_1.partition_fragment_B(sS_f16_flat);
    auto tCrC        = thr_mma_1.make_fragment_C(tCsC_R);
    auto rC_half     = make_tensor_like<T>(tCrC);
    auto tCrA_view_1 = s2r_thr_a1.retile_D(tCrA);
    auto tCrB_view_1 = s2r_thr_b1.retile_D(tCrB);
    auto tCrA_view_2 = s2r_thr_a2.retile_D(tCrA);
    auto tCrB_view_2 = s2r_thr_b2.retile_D(tCrB);

    // 3. Prologue: initialize sS_f32
    if (use_state_old && gStateOld != nullptr) {
        auto gOld_slice = make_tensor(make_gmem_ptr(gStateOld + state_offset), layout_vk_slice);
        cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gOld_slice), rS_f32);
    } else {
        clear(rS_f32);
    }
    #pragma unroll
    for (int x = 0; x < 2; ++x) {
        *reinterpret_cast<uint4*>(&tXsS_f32_d(make_coord(Int<0>{}, x), Int<0>{}, Int<0>{})) =
            *reinterpret_cast<const uint4*>(&rS_f32(make_coord(Int<0>{}, x), Int<0>{}, Int<0>{}));
    }
    __syncthreads();

    // 4. Sequential chunk scan loop
    for (int32_t c = start_c; c < end_c; ++c) {
        int32_t chunk_start   = chunk_offsets[c];
        int32_t valid_len     = chunk_valid_lens[c];
        bool    is_first_zero = (c == start_c && !use_state_old);

        int64_t s_list_off = (static_cast<int64_t>(head_idx) * num_chunks + c) * mat_elements_vk + slice_offset_v;
        int64_t y_off      = (static_cast<int64_t>(head_idx) * num_chunks + c) * mat_elements_ck;
        int64_t ur_off     = (static_cast<int64_t>(head_idx) * num_chunks + c) * mat_elements_cv + static_cast<int64_t>(v_idx) * B_V;
        int64_t tok_off    = (head_tok_base + chunk_start) * D_K;

        auto gS_slice  = make_tensor(make_gmem_ptr(gS_list + s_list_off), layout_vk_slice);
        auto gY_chunk  = make_tensor(make_gmem_ptr(gY      + y_off),      layout_ck_full);
        auto gUR_slice = make_tensor(make_gmem_ptr(gU_R    + ur_off),     layout_cv_slice);
        auto gK_chunk  = make_tensor(make_gmem_ptr(gK_h    + tok_off),    layout_ck_full);
        auto gG_chunk  = make_tensor(make_gmem_ptr(gG_log2_h + tok_off),  layout_ck_full);

        // Emit S_c and stage U_slice into sR
        cute::copy(tiled_copy_f32, tXsS_f32_s, rS_f32);
        cute::copy(tiled_copy_f32, rS_f32, thr_copy_f32.partition_D(gS_slice));

        cute::copy(tiled_copy_r, thr_copy_r.partition_S(gUR_slice), rR_slice);
        *reinterpret_cast<uint4*>(&tXsR_d(Int<0>{}, Int<0>{}, Int<0>{})) =
            *reinterpret_cast<const uint4*>(&rR_slice(Int<0>{}, Int<0>{}, Int<0>{}));

        // GEMM 1: R_c = U_c - Y_c @ S_c^T
        if (!is_first_zero) {
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                rS_f16(v, Int<0>{}, Int<0>{}) = float_to_out<T>(rS_f32(v, Int<0>{}, Int<0>{}));
            }
            *reinterpret_cast<uint4*>(&tXsS_f16_d(Int<0>{}, Int<0>{}, Int<0>{})) =
                *reinterpret_cast<const uint4*>(&rS_f16(Int<0>{}, Int<0>{}, Int<0>{}));

            cute::copy(tiled_copy_h, thr_copy_h.partition_S(gY_chunk), tCrMat);
            #pragma unroll
            for (int m = 0; m < size<1>(tXsMat_d); ++m) {
                *reinterpret_cast<uint4*>(&tXsMat_d(Int<0>{}, m, Int<0>{})) =
                    *reinterpret_cast<const uint4*>(&tCrMat(Int<0>{}, m, Int<0>{}));
            }
            __syncthreads();

            clear(tCrC);
            cute::copy(s2r_copy_a1, tCsA_1, tCrA_view_1);
            cute::copy(s2r_copy_b1, tCsB_1, tCrB_view_1);
            cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC);

            #pragma unroll
            for (int i = 0; i < size(tCrC); ++i) {
                rC_half(i) = float_to_out<T>(half_to_float(tCsC_R(i)) - tCrC(i));
            }
            cute::copy(rC_half, tCsC_R);
            __syncthreads();

            cute::copy(tiled_copy_r, tXsR_s, rR_slice);
            cute::copy(tiled_copy_r, rR_slice, thr_copy_r.partition_D(gUR_slice));
        }

        // Recompute K_tail[r, d] = k[r, d] * exp2f(G_last[d] - G_log2[r, d]) into sMat
        const float* p_g_last = gG_log2_h + (head_tok_base + chunk_start + valid_len - 1) * D_K;
        float4 g_last0 = *reinterpret_cast<const float4*>(p_g_last + col_base);
        float4 g_last1 = *reinterpret_cast<const float4*>(p_g_last + col_base + 4);
        float g_last_reg[8] = {g_last0.x, g_last0.y, g_last0.z, g_last0.w,
                               g_last1.x, g_last1.y, g_last1.z, g_last1.w};

        if (valid_len == CHUNK_SIZE) {
            cute::copy(tiled_copy_h,   thr_copy_h.partition_S(gK_chunk),   tCrMat);
            cute::copy(tiled_copy_f32, thr_copy_f32.partition_S(gG_chunk), tCrG);
        } else {
            clear(tCrMat);
            clear(tCrG);
            cute::copy_if(tiled_copy_h,   tPredK, thr_copy_h.partition_S(gK_chunk),   tCrMat);
            cute::copy_if(tiled_copy_f32, tPredG, thr_copy_f32.partition_S(gG_chunk), tCrG);
        }

        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            int  local_row = m * 16 + row_id;
            bool row_valid = (local_row < valid_len);
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                float k_val = row_valid
                    ? (half_to_float(tCrMat(v, m, Int<0>{})) * exp2f(g_last_reg[v] - tCrG(v, m, Int<0>{})))
                    : 0.0f;
                tCrMat(v, m, Int<0>{}) = float_to_out<T>(k_val);
            }
        }
        #pragma unroll
        for (int m = 0; m < size<1>(tXsMat_d); ++m) {
            *reinterpret_cast<uint4*>(&tXsMat_d(Int<0>{}, m, Int<0>{})) =
                *reinterpret_cast<const uint4*>(&tCrMat(Int<0>{}, m, Int<0>{}));
        }
        __syncthreads();

        // GEMM 2: S_{c+1} = S_c * exp2(G_last) + R_slice^T @ K_tail in FP32
        if (is_first_zero) {
            clear(tCrC);
        } else {
            #pragma unroll
            for (int n = 0; n < size<2>(tCrC); ++n) {
                int    d0     = get<1>(tCcC_S(Int<0>{}, Int<0>{}, n));
                float2 g_pair = *reinterpret_cast<const float2*>(p_g_last + d0);
                float  e0     = exp2f(g_pair.x);
                float  e1     = exp2f(g_pair.y);
                tCrC(Int<0>{}, Int<0>{}, n) = tCsC_S(Int<0>{}, Int<0>{}, n) * e0;
                tCrC(Int<1>{}, Int<0>{}, n) = tCsC_S(Int<1>{}, Int<0>{}, n) * e1;
                tCrC(Int<2>{}, Int<0>{}, n) = tCsC_S(Int<2>{}, Int<0>{}, n) * e0;
                tCrC(Int<3>{}, Int<0>{}, n) = tCsC_S(Int<3>{}, Int<0>{}, n) * e1;
            }
        }
        cute::copy(s2r_copy_a2, tCsA_2, tCrA_view_2);
        cute::copy(s2r_copy_b2, tCsB_2, tCrB_view_2);
        cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC);

        cute::copy(tCrC, tCsC_S);
        __syncthreads();
    }

    // 5. Epilogue: write final state slice [FP32]
    cute::copy(tiled_copy_f32, tXsS_f32_s, rS_f32);
    cute::copy(tiled_copy_f32, rS_f32, tXgFinal);
}

// ============================================================================
// Part 2: Host Dispatch Functions
// ============================================================================

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
) {
    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64");
    TORCH_CHECK(batch_size >= 0, "batch_size must be >= 0");
    TORCH_CHECK(Y.is_cuda(), "Y must be a CUDA tensor");
    const c10::cuda::CUDAGuard device_guard(Y.device());

    auto check_t = [&](const at::Tensor& t, const char* name, int64_t expected_dim, at::ScalarType dtype) {
        TORCH_CHECK(t.is_cuda() && t.device() == Y.device(), name, " must be on the same CUDA device as Y");
        TORCH_CHECK(t.scalar_type() == dtype, name, " dtype mismatch");
        TORCH_CHECK(t.dim() == expected_dim && t.is_contiguous(), name, " must be contiguous with expected rank");
    };

    check_t(Y,        "Y",        4, at::ScalarType::Half);
    check_t(U_R,      "U_R",      4, at::ScalarType::Half);
    check_t(k_h,      "k_h",      3, at::ScalarType::Half);
    check_t(G_log2_h, "G_log2_h", 3, at::ScalarType::Float);

    int64_t num_kv_heads = Y.size(0);
    int64_t num_chunks   = Y.size(1);
    int64_t total_L      = k_h.size(1);
    int64_t d_k          = Y.size(3);
    int64_t d_v          = U_R.size(3);

    TORCH_CHECK(Y.size(2) == chunk_size && d_k == 64, "Y trailing dimensions must be (64, 64)");
    TORCH_CHECK(U_R.size(0) == num_kv_heads && U_R.size(1) == num_chunks &&
                U_R.size(2) == chunk_size && d_v == 64, "U_R shape mismatch");
    TORCH_CHECK(k_h.size(0) == num_kv_heads && k_h.size(2) == d_k, "k_h shape mismatch");
    TORCH_CHECK(G_log2_h.size(0) == num_kv_heads && G_log2_h.size(1) == total_L &&
                G_log2_h.size(2) == d_k, "G_log2_h shape mismatch");
    TORCH_CHECK(meta.num_chunks == num_chunks, "meta.num_chunks mismatch");

    TORCH_CHECK(meta.cu_chunk_lens.is_cuda() && meta.cu_chunk_lens.device() == Y.device() &&
                meta.cu_chunk_lens.scalar_type() == at::ScalarType::Int &&
                meta.cu_chunk_lens.dim() == 1 && meta.cu_chunk_lens.is_contiguous() &&
                meta.cu_chunk_lens.size(0) == batch_size + 1, "meta.cu_chunk_lens mismatch");

    if (num_chunks > 0) {
        TORCH_CHECK(meta.chunk_offsets.is_cuda() && meta.chunk_offsets.device() == Y.device() &&
                    meta.chunk_offsets.scalar_type() == at::ScalarType::Int &&
                    meta.chunk_offsets.dim() == 1 && meta.chunk_offsets.is_contiguous() &&
                    meta.chunk_offsets.size(0) >= num_chunks + 1, "meta.chunk_offsets mismatch");
        TORCH_CHECK(meta.chunk_valid_lens.is_cuda() && meta.chunk_valid_lens.device() == Y.device() &&
                    meta.chunk_valid_lens.scalar_type() == at::ScalarType::Int &&
                    meta.chunk_valid_lens.dim() == 1 && meta.chunk_valid_lens.is_contiguous() &&
                    meta.chunk_valid_lens.size(0) >= num_chunks, "meta.chunk_valid_lens mismatch");
    }

    bool use_state_old = false;
    at::Tensor state_old_c;
    if (state_old.has_value() && state_old.value().defined()) {
        const at::Tensor& s_in = state_old.value();
        TORCH_CHECK(s_in.is_cuda() && s_in.device() == Y.device(), "state_old must be on the same CUDA device as Y");
        TORCH_CHECK(s_in.scalar_type() == at::ScalarType::Float, "state_old must be FP32 (torch.float32)");
        TORCH_CHECK(s_in.dim() == 4 && s_in.size(0) == batch_size && s_in.size(1) == num_kv_heads &&
                    s_in.size(2) == d_v && s_in.size(3) == d_k, "state_old must have shape (B, H_kv, 64, 64)");
        state_old_c = ensure_aligned16(s_in.contiguous());
        use_state_old = true;
    }

    auto f_opts = G_log2_h.options();
    if (num_chunks <= 0 || batch_size == 0) {
        at::Tensor S_list = at::empty({num_kv_heads, num_chunks, d_v, d_k}, f_opts);
        at::Tensor final_states;
        if (batch_size == 0) {
            final_states = at::empty({0, num_kv_heads, d_v, d_k}, f_opts);
        } else if (use_state_old) {
            final_states = state_old_c.clone();
        } else {
            final_states = at::zeros({batch_size, num_kv_heads, d_v, d_k}, f_opts);
        }
        return std::make_tuple(S_list, U_R, final_states);
    }

    at::Tensor S_list       = torch::empty({num_kv_heads, num_chunks, d_v, d_k}, f_opts);
    at::Tensor final_states = torch::empty({batch_size, num_kv_heads, d_v, d_k}, f_opts);

    at::Tensor Y_a = ensure_aligned16(Y);
    at::Tensor k_a = ensure_aligned16(k_h);
    at::Tensor G_a = ensure_aligned16(G_log2_h);

    bool ur_aligned = (U_R.numel() == 0) || ((reinterpret_cast<uintptr_t>(U_R.data_ptr()) & 15) == 0);
    at::Tensor R_work = ur_aligned ? U_R : U_R.clone();

    constexpr int NUM_THREADS = 128;
    constexpr int CHUNK_CONST = 64;
    constexpr int DK_CONST    = 64;
    constexpr int DV_CONST    = 64;
    constexpr int BV_CONST    = 16;

    using Element    = cutlass::half_t;
    using MMAAtom    = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
    using TiledMMA411 = TiledMMA<MMAAtom, Layout<Shape<_4, _1, _1>, Stride<_1, _4, _4>>>;
    using TiledMMA141 = TiledMMA<MMAAtom, Layout<Shape<_1, _4, _1>, Stride<_1, _1, _4>>>;

    using S2RCopyAtomA_KM  = Copy_Atom<SM75_U32x2_LDSM_N, Element>;
    using S2RCopyAtomB_KM  = Copy_Atom<SM75_U32x1_LDSM_N, Element>;
    using S2RCopyAtomA_MNM = Copy_Atom<SM75_U16x4_LDSM_T, Element>;
    using S2RCopyAtomB_MNM = Copy_Atom<SM75_U16x2_LDSM_T, Element>;

    using TiledCopyHalf = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint4>, Element>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>, Stride<_8, _1>>{}
    ));
    using TiledCopyR = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint4>, Element>{},
        Layout<Shape<_64, _2>, Stride<_2, _1>>{},
        Layout<Shape<_1, _8>, Stride<_8, _1>>{}
    ));
    using TiledCopyFloat = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint4>, float>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>, Stride<_8, _1>>{}
    ));

    const Element* p_Y         = reinterpret_cast<const Element*>(Y_a.data_ptr<at::Half>());
    Element*       p_UR        = reinterpret_cast<Element*>(R_work.data_ptr<at::Half>());
    const Element* p_k         = reinterpret_cast<const Element*>(k_a.data_ptr<at::Half>());
    const float*   p_G         = G_a.data_ptr<float>();
    const float*   p_state_old = use_state_old ? state_old_c.data_ptr<float>() : nullptr;
    const int32_t* p_cu_c      = meta.cu_chunk_lens.data_ptr<int32_t>();
    const int32_t* p_c_off     = meta.chunk_offsets.data_ptr<int32_t>();
    const int32_t* p_c_vlen    = meta.chunk_valid_lens.data_ptr<int32_t>();
    float*         p_S_list    = S_list.data_ptr<float>();
    float*         p_final     = final_states.data_ptr<float>();

    {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        dim3 grid(
            static_cast<unsigned int>(batch_size),
            static_cast<unsigned int>(num_kv_heads),
            static_cast<unsigned int>(DV_CONST / BV_CONST)
        );
        dim3 block(NUM_THREADS);

        kernel_chunk_scan_fwd<
            Element, NUM_THREADS, CHUNK_CONST, DK_CONST, DV_CONST, BV_CONST,
            MMAAtom, TiledMMA411, TiledMMA141,
            S2RCopyAtomA_KM, S2RCopyAtomB_KM, S2RCopyAtomA_MNM, S2RCopyAtomB_MNM,
            TiledCopyHalf, TiledCopyR, TiledCopyFloat
        ><<<grid, block, 0, stream>>>(
            p_Y, p_UR, p_k, p_G, p_state_old,
            p_cu_c, p_c_off, p_c_vlen,
            p_S_list, p_final,
            use_state_old, total_L, num_chunks
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (!ur_aligned) {
        U_R.copy_(R_work);
    }

    return std::make_tuple(S_list, U_R, final_states);
}

// Outputs: (S_list: (H_kv, num_chunks, 64, 64) [FP32], R_list: (H_kv, num_chunks, 64, 64) [FP16 in-place U_R], final_states: (B, H_kv, 64, 64) [FP32])
std::tuple<at::Tensor, at::Tensor, at::Tensor>
chunk_scan_fwd(
    const at::Tensor& Y,
    at::Tensor& U_R,
    const at::Tensor& k_h,
    const at::Tensor& G_log2_h,
    const std::optional<at::Tensor>& state_old,
    const at::Tensor& cu_seq_lens,
    int64_t chunk_size
) {
    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64");
    TORCH_CHECK(cu_seq_lens.is_cuda() && cu_seq_lens.dim() == 1 && cu_seq_lens.size(0) >= 1,
                "cu_seq_lens must be a 1D CUDA tensor with size >= 1");
    int64_t batch_size = cu_seq_lens.size(0) - 1;
    VarlenChunkMeta meta = compute_varlen_chunk_metadata(cu_seq_lens, chunk_size);
    return chunk_scan_fwd_with_meta(
        Y, U_R, k_h, G_log2_h, state_old, meta, batch_size, chunk_size
    );
}
