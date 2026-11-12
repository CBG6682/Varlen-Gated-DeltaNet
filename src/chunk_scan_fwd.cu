#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <tuple>
#include <optional>
#include <vector>
#include "deltanet_common.cuh"
#include "deltanet_ops.cuh"

using namespace cute;

// =========================================================================================
// SECTION 1: GLOBAL KERNEL (kernel_chunk_scan_fwd)
// =========================================================================================

// -----------------------------------------------------------------------------------------
// Global Kernel: kernel_chunk_scan_fwd
// -----------------------------------------------------------------------------------------
// Execution Flow:
// - Stage 1 - Setup & Grid Initialization:
//   + batch_idx = blockIdx.x, head_idx = blockIdx.y, v_idx = blockIdx.z, tid = threadIdx.x.
//   + Sequence chunk boundaries: start_c = cu_chunk_lens[batch_idx], end_c = cu_chunk_lens[batch_idx + 1].
//   + V-dimension slice offset: slice_offset_v = v_idx * B_V * D_K.
//   + State layout offset (B, H, d_v, d_k): (batch_idx * gridDim.y + head_idx) * (d_v * d_k) + slice_offset_v.
// - Stage 2 - CuTe Tensor Partitioning & Shared Memory Allocation:
//   + Compact 12 KB SMEM: sS (2 KB for B_V x D_K), sW (8 KB for D_K x D_K), sY (2 KB for B_V x D_K).
//   + Swizzle<3, 3, 3> layouts to eliminate bank conflicts.
//   + Partition sS_mma (Operand A), sW_mma (Operand B), sY_mma (Operand C) with TiledMMA Layout<Shape<_1, _4, _1>>.
//   + S2R loads via explicit LDSM copy atoms: SM75_U32x2_LDSM_N for A (K-major) and SM75_U16x2_LDSM_T for B (MN-major).
//   + 128-bit vectorized uint4 GMEM loads and stores.
// - Stage 3 - Buffer Initialization & Register Fragments:
//   + NUM_STPS_V = B_V / ROWS_PER_STP (1 step), NUM_STPS_K = D_K / ROWS_PER_STP (4 steps).
//   + Register fragments allocated once at kernel entry.
//   + Empty sequence handling (start_c >= end_c): write state_old or zero to final_states, then exit.
// - Stage 4 - Chunk Recurrence (Epilogue directly into Accumulator):
//   + Initialize SMEM sS (S_curr) with state_old (or zero).
//   + Sequential loop over chunks c = start_c .. end_c - 1:
//     - Load W_c from gW to sW, Y_c from gY_S to sY.
//     - Write S_curr from sS into GMEM gY_S[:, c] along V slice.
//     - Compute S_next = S_curr @ W_c + Y_c:
//       * If first chunk and zero state: S_next = Y_c (bypass GEMM).
//       * Otherwise: load Y_c into tCrC accumulator (FP32), execute cute::gemm accumulation,
//         and write tCrC (converted to FP16) to sS.
//     - On final chunk (c + 1 == end_c), write S_next to GMEM final_states[b, h].
// -----------------------------------------------------------------------------------------
template <
    typename T,
    int NUM_THREADS,
    int D_K,
    int D_V,
    int B_V,
    typename MMAAtom,
    typename TiledMMAType,
    typename S2RCopyAtomA,
    typename S2RCopyAtomB,
    typename TiledCopyG2R,
    typename TiledCopyStore
>
__global__ void __launch_bounds__(NUM_THREADS) kernel_chunk_scan_fwd(
    const T* __restrict__ gW,                  // (num_kv_heads, num_chunks, d_k, d_k) [FP16]
    T* gY_S,                                   // (num_kv_heads, num_chunks, d_v, d_k) [FP16] - In: Y, Out: S in-place
    const T* __restrict__ gStateOld,           // (batch_size, num_kv_heads, d_v, d_k) [FP16] (nullable)
    const int32_t* __restrict__ cu_chunk_lens, // (batch_size + 1,) [Int32]
    T* __restrict__ gFinalStates,              // (batch_size, num_kv_heads, d_v, d_k) [FP16]
    bool use_state_old,                        // scalar [bool]
    int64_t num_chunks,                        // scalar [int64]
    int64_t batch_size                         // scalar [int64]
) {
    // -------------------------------------------------------------------------------------
    // Stage 1 - Setup & Grid Initialization
    // -------------------------------------------------------------------------------------
    int batch_idx = blockIdx.x;
    int head_idx  = blockIdx.y;
    int v_idx     = blockIdx.z; // V slice index: 0 .. (D_V / B_V - 1)
    int tid       = threadIdx.x;

    int32_t start_c = cu_chunk_lens[batch_idx];
    int32_t end_c   = cu_chunk_lens[batch_idx + 1];

    int64_t mat_elements_v = (int64_t)D_V * D_K;
    int64_t mat_elements_k = (int64_t)D_K * D_K;

    int64_t slice_offset_v = (int64_t)v_idx * B_V * D_K;
    int64_t state_offset   = ((int64_t)batch_idx * gridDim.y + head_idx) * mat_elements_v + slice_offset_v;

    // -------------------------------------------------------------------------------------
    // Stage 2 - Shared Memory Allocation & CuTe Layouts
    // Compact 12 KB SMEM: sS (2 KB), sW (8 KB), sY (2 KB)
    // -------------------------------------------------------------------------------------
    __shared__ alignas(16) T sS[B_V * D_K]; // 2 KB: current state S_curr for slice (16 x 64)
    __shared__ alignas(16) T sW[D_K * D_K]; // 8 KB: full chunk transition matrix W_c (64 x 64)
    __shared__ alignas(16) T sY[B_V * D_K]; // 2 KB: input update matrix Y_c for slice (16 x 64)

    using SwizzleOp = Swizzle<3, 3, 3>;

    auto smem_layout_flat_v = composition(SwizzleOp{}, make_layout(make_shape(Int<B_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto smem_layout_flat_k = composition(SwizzleOp{}, make_layout(make_shape(Int<D_K>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

    // SMEM MMA Layouts:
    // Matrix A (S_curr): Shape (M=16, K=64), Stride (64, 1) -> Row-Major (K-major)
    auto smem_layout_mma_A = composition(SwizzleOp{}, make_layout(make_shape(Int<B_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    // Matrix B (W_c): Shape (N=64, K=64), Stride (1, 64) -> Transposed in SMEM (MN-major)
    auto smem_layout_mma_B = composition(SwizzleOp{}, make_layout(make_shape(Int<D_K>{}, Int<D_K>{}), make_stride(Int<1>{}, Int<D_K>{})));
    // Matrix C (S_next / Y_c): Shape (M=16, N=64), Stride (64, 1) -> Row-Major
    auto smem_layout_mma_C = composition(SwizzleOp{}, make_layout(make_shape(Int<B_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

    auto sS_flat = make_tensor(make_smem_ptr(sS), smem_layout_flat_v);
    auto sW_flat = make_tensor(make_smem_ptr(sW), smem_layout_flat_k);
    auto sY_flat = make_tensor(make_smem_ptr(sY), smem_layout_flat_v);

    auto sS_mma = make_tensor(make_smem_ptr(sS), smem_layout_mma_A);
    auto sW_mma = make_tensor(make_smem_ptr(sW), smem_layout_mma_B);
    auto sY_mma = make_tensor(make_smem_ptr(sY), smem_layout_mma_C);

    // TiledMMA & Partitioning (Layout<Shape<_1, _4, _1>>: 1 warp vertical M=16, 4 warps horizontal N=64)
    TiledMMAType tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    // S2R TiledCopy via LDSM atoms
    auto s2r_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_copy_a.get_slice(tid);
    auto s2r_thr_copy_b = s2r_copy_b.get_slice(tid);

    auto tCsA = s2r_thr_copy_a.partition_S(sS_mma);
    auto tCsB = s2r_thr_copy_b.partition_S(sW_mma);

    auto tCrA = thr_mma.partition_fragment_A(sS_mma);
    auto tCrB = thr_mma.partition_fragment_B(sW_mma);
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

    TiledCopyG2R tiled_copy_g2r;
    TiledCopyStore tiled_copy_store;
    auto thr_copy_g2r   = tiled_copy_g2r.get_thread_slice(tid);
    auto thr_copy_store = tiled_copy_store.get_thread_slice(tid);

    constexpr int ROWS_PER_STP = 16;
    constexpr int NUM_STPS_V   = B_V / ROWS_PER_STP; // 16 / 16 = 1 step
    constexpr int NUM_STPS_K   = D_K / ROWS_PER_STP; // 64 / 16 = 4 steps

    auto dummy_tile = local_tile(sS_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(0, 0));
    auto dummy_part = thr_copy_g2r.partition_S(dummy_tile);

    decltype(make_fragment_like(dummy_part)) rLoad[NUM_STPS_V];
    decltype(make_fragment_like(dummy_part)) rStore[NUM_STPS_V];

    #pragma unroll
    for (int st = 0; st < NUM_STPS_V; ++st) {
        clear(rLoad[st]);
        clear(rStore[st]);
    }

    // -------------------------------------------------------------------------------------
    // Zero-length sequence handling: start_c >= end_c
    // Write state_old (or zero) to final_states then exit
    // -------------------------------------------------------------------------------------
    if (start_c >= end_c) {
        T* p_final = gFinalStates + state_offset;
        auto gFinal_chunk = make_tensor(make_gmem_ptr(p_final), make_shape(Int<B_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{}));
        if (use_state_old && gStateOld != nullptr) {
            const T* p_state_old = gStateOld + state_offset;
            auto gStateOld_chunk = make_tensor(make_gmem_ptr(p_state_old), make_shape(Int<B_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{}));
            #pragma unroll
            for (int st = 0; st < NUM_STPS_V; ++st) {
                auto gState_step = local_tile(gStateOld_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
                auto gFinal_step = local_tile(gFinal_chunk,    make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
                cute::copy(tiled_copy_g2r, thr_copy_g2r.partition_S(gState_step), rLoad[st]);
                auto tFinal = thr_copy_store.partition_D(gFinal_step);
                *reinterpret_cast<uint4*>(&tFinal(0)) = *reinterpret_cast<const uint4*>(&rLoad[st](0));
            }
        } else {
            #pragma unroll
            for (int st = 0; st < NUM_STPS_V; ++st) {
                auto gFinal_step = local_tile(gFinal_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
                auto tFinal = thr_copy_store.partition_D(gFinal_step);
                *reinterpret_cast<uint4*>(&tFinal(0)) = *reinterpret_cast<const uint4*>(&rLoad[st](0));
            }
        }
        return;
    }

    // -------------------------------------------------------------------------------------
    // Stage 3: Initialize SMEM sS (S_curr) prior to the chunk loop
    // -------------------------------------------------------------------------------------
    if (use_state_old && gStateOld != nullptr) {
        const T* p_state_old = gStateOld + state_offset;
        auto gStateOld_chunk = make_tensor(make_gmem_ptr(p_state_old), make_shape(Int<B_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{}));

        #pragma unroll
        for (int st = 0; st < NUM_STPS_V; ++st) {
            auto gState_step = local_tile(gStateOld_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
            auto sS_step     = local_tile(sS_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));

            cute::copy(tiled_copy_g2r, thr_copy_g2r.partition_S(gState_step), rLoad[st]);
            auto tSsS = thr_copy_g2r.partition_D(sS_step);
            *reinterpret_cast<uint4*>(&tSsS(0)) = *reinterpret_cast<const uint4*>(&rLoad[st](0));
        }
    } else {
        #pragma unroll
        for (int st = 0; st < NUM_STPS_V; ++st) {
            auto sS_step = local_tile(sS_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
            clear(rLoad[st]);
            auto tSsS = thr_copy_g2r.partition_D(sS_step);
            *reinterpret_cast<uint4*>(&tSsS(0)) = *reinterpret_cast<const uint4*>(&rLoad[st](0));
        }
    }
    __syncthreads();

    // -------------------------------------------------------------------------------------
    // Stage 4: Sequential Chunk Loop c = start_c .. end_c - 1
    // -------------------------------------------------------------------------------------
    for (int32_t c = start_c; c < end_c; ++c) {
        int64_t w_offset = ((int64_t)head_idx * num_chunks + c) * mat_elements_k;
        int64_t y_offset = ((int64_t)head_idx * num_chunks + c) * mat_elements_v + slice_offset_v;

        const T* p_w = gW + w_offset;
        T* p_y = gY_S + y_offset;

        auto gW_chunk = make_tensor(make_gmem_ptr(p_w), make_shape(Int<D_K>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{}));
        auto gY_chunk = make_tensor(make_gmem_ptr(p_y), make_shape(Int<B_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{}));

        // 1.1 Load W_c into SMEM sW (NUM_STPS_K steps)
        #pragma unroll
        for (int st = 0; st < NUM_STPS_K; ++st) {
            auto gW_step = local_tile(gW_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
            auto sW_step = local_tile(sW_flat,  make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));

            decltype(make_fragment_like(dummy_part)) rW_tmp;
            cute::copy(tiled_copy_g2r, thr_copy_g2r.partition_S(gW_step), rW_tmp);
            auto tWsW = thr_copy_g2r.partition_D(sW_step);
            *reinterpret_cast<uint4*>(&tWsW(0)) = *reinterpret_cast<const uint4*>(&rW_tmp(0));
        }

        // 1.2 Load Y_c into SMEM sY (NUM_STPS_V steps)
        #pragma unroll
        for (int st = 0; st < NUM_STPS_V; ++st) {
            auto gY_step = local_tile(gY_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
            auto sY_step = local_tile(sY_flat,  make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));

            decltype(make_fragment_like(dummy_part)) rY_tmp;
            cute::copy(tiled_copy_g2r, thr_copy_g2r.partition_S(gY_step), rY_tmp);
            auto tYsY = thr_copy_g2r.partition_D(sY_step);
            *reinterpret_cast<uint4*>(&tYsY(0)) = *reinterpret_cast<const uint4*>(&rY_tmp(0));
        }
        __syncthreads();

        // 2. Overwrite GMEM location of Y_c with S_curr (from sS)
        // Since Y_c is buffered safely in SMEM sY, this GMEM location now stores S_c
        #pragma unroll
        for (int st = 0; st < NUM_STPS_V; ++st) {
            auto sS_step = local_tile(sS_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
            auto tSsS    = thr_copy_store.partition_S(sS_step);
            cute::copy(tSsS, rLoad[st]);

            auto gS_step = local_tile(gY_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
            auto tSgS    = thr_copy_store.partition_D(gS_step);
            *reinterpret_cast<uint4*>(&tSgS(0)) = *reinterpret_cast<const uint4*>(&rLoad[st](0));
        }
        __syncthreads();

        bool is_first_chunk_zero = (c == start_c && !use_state_old);

        if (is_first_chunk_zero) {
            // S_curr = 0, thus 0 @ W_c + Y_c = Y_c. Bypass GEMM and directly assign S_next = Y_c
            #pragma unroll
            for (int st = 0; st < NUM_STPS_V; ++st) {
                auto sY_step = local_tile(sY_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
                auto tYsY    = thr_copy_store.partition_S(sY_step);
                cute::copy(tYsY, rStore[st]);

                // Update SMEM sS for next chunk
                auto sS_step = local_tile(sS_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
                auto tSsS    = thr_copy_g2r.partition_D(sS_step);
                *reinterpret_cast<uint4*>(&tSsS(0)) = *reinterpret_cast<const uint4*>(&rStore[st](0));

                // Write to final_states on last chunk
                if (c + 1 == end_c) {
                    T* p_final = gFinalStates + state_offset;
                    auto gFinal_chunk = make_tensor(make_gmem_ptr(p_final), make_shape(Int<B_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{}));
                    auto gFinal_step  = local_tile(gFinal_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
                    auto tFinal       = thr_copy_store.partition_D(gFinal_step);
                    *reinterpret_cast<uint4*>(&tFinal(0)) = *reinterpret_cast<const uint4*>(&rStore[st](0));
                }
            }
            __syncthreads();
        } else {
            // Compute S_next = S_curr @ W_c + Y_c with direct accumulator epilogue
            auto tCsC_Y = thr_mma.partition_C(sY_mma);
            auto tCrC   = thr_mma.make_fragment_C(tCsC_Y);
            #pragma unroll
            for (int i = 0; i < size(tCrC); ++i) {
                tCrC(i) = half_to_float(tCsC_Y(i));
            }

            // S2R load S_curr and W_c via LDSM into registers
            cute::copy(s2r_copy_a, tCsA, tCrA_view);
            cute::copy(s2r_copy_b, tCsB, tCrB_view);

            // Tensor Core MMA: tCrC = S_curr @ W_c + Y_c
            cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC);

            __syncthreads(); // Ensure warps finish reading sS and sW before updating sS

            // Write tCrC (converted to FP16) to SMEM sS
            auto sS_mma_C = make_tensor(make_smem_ptr(sS), smem_layout_mma_C);
            auto tCsC_S   = thr_mma.partition_C(sS_mma_C);
            #pragma unroll
            for (int i = 0; i < size(tCrC); ++i) {
                tCsC_S(i) = float_to_out<T>(tCrC(i));
            }
            __syncthreads();

            // Write to final_states on last chunk
            if (c + 1 == end_c) {
                #pragma unroll
                for (int st = 0; st < NUM_STPS_V; ++st) {
                    auto sS_step = local_tile(sS_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
                    auto tSsS    = thr_copy_store.partition_S(sS_step);
                    cute::copy(tSsS, rStore[st]);

                    T* p_final = gFinalStates + state_offset;
                    auto gFinal_chunk = make_tensor(make_gmem_ptr(p_final), make_shape(Int<B_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{}));
                    auto gFinal_step  = local_tile(gFinal_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
                    auto tFinal       = thr_copy_store.partition_D(gFinal_step);
                    *reinterpret_cast<uint4*>(&tFinal(0)) = *reinterpret_cast<const uint4*>(&rStore[st](0));
                }
            }
        }
    }
}

// =========================================================================================
// SECTION 2: HOST COORDINATION FUNCTION (chunk_scan_fwd)
// =========================================================================================

std::tuple<at::Tensor, at::Tensor>
chunk_scan_fwd(
    const at::Tensor& W,                           // (num_kv_heads, num_chunks, d_k, d_k) [FP16]
    const at::Tensor& Y,                           // (num_kv_heads, num_chunks, d_v, d_k) [FP16]
    const std::optional<at::Tensor>& state_old,    // (batch_size, num_kv_heads, d_v, d_k) [FP16]
    const at::Tensor& cu_seq_lens,                 // (batch_size + 1,) [Int32 / Int64]
    int64_t chunk_size                             // scalar [int64] (default 64)
) {
    // -------------------------------------------------------------------------------------
    // Stage 1 - Input Validation
    // -------------------------------------------------------------------------------------
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
    const c10::cuda::CUDAGuard device_guard(W.device());

    TORCH_CHECK(Y.is_cuda(), "Y must be a CUDA tensor");
    TORCH_CHECK(cu_seq_lens.is_cuda(), "cu_seq_lens must be a CUDA tensor");

    TORCH_CHECK(Y.device() == W.device(), "Y must be on the same device as W");
    TORCH_CHECK(cu_seq_lens.device() == W.device(), "cu_seq_lens must be on the same device as W");

    TORCH_CHECK(W.scalar_type() == at::ScalarType::Half, "W must be FP16 (torch.float16)");
    TORCH_CHECK(Y.scalar_type() == at::ScalarType::Half, "Y must be FP16 (torch.float16)");
    TORCH_CHECK(cu_seq_lens.scalar_type() == at::ScalarType::Int || cu_seq_lens.scalar_type() == at::ScalarType::Long,
                "cu_seq_lens must be Int32 or Int64");

    TORCH_CHECK(W.dim() == 4, "W must be 4D tensor with shape (num_kv_heads, num_chunks, d_k, d_k)");
    TORCH_CHECK(Y.dim() == 4, "Y must be 4D tensor with shape (num_kv_heads, num_chunks, d_v, d_k)");
    TORCH_CHECK(cu_seq_lens.dim() == 1, "cu_seq_lens must be 1D tensor with shape (batch_size + 1,)");

    TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
    TORCH_CHECK(Y.is_contiguous(), "Y must be contiguous");

    int64_t num_kv_heads = W.size(0);
    int64_t num_chunks   = W.size(1);
    int64_t d_k          = W.size(2);
    int64_t d_k_col      = W.size(3);

    int64_t num_y_heads  = Y.size(0);
    int64_t num_y_chunks = Y.size(1);
    int64_t d_v          = Y.size(2);
    int64_t d_v_k        = Y.size(3);

    TORCH_CHECK(num_y_heads == num_kv_heads, "Y num_kv_heads must match W num_kv_heads");
    TORCH_CHECK(num_y_chunks == num_chunks, "Y num_chunks must match W num_chunks");
    TORCH_CHECK(d_k == 64 && d_k_col == 64, "W trailing dimensions must be 64x64 (d_k == 64 on SM75+)");
    TORCH_CHECK(d_v == 64 && d_v_k == 64, "Y trailing dimensions must be 64x64 (d_v == 64 on SM75+)");
    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64 for optimal SM alignment");

    int64_t batch_size = cu_seq_lens.size(0) - 1;
    TORCH_CHECK(batch_size >= 0, "batch_size must be >= 0");

    at::Tensor cu_chunk_lens = compute_cu_chunk_lens(cu_seq_lens, chunk_size);
    at::Tensor cu_chunk_lens_i32 = cu_chunk_lens.to(at::kInt).contiguous();

    bool use_state_old = false;
    at::Tensor state_old_c;

    if (state_old.has_value() && state_old.value().defined()) {
        const at::Tensor& s_in = state_old.value();
        TORCH_CHECK(s_in.is_cuda(), "state_old must be a CUDA tensor");
        TORCH_CHECK(s_in.device() == W.device(), "state_old must be on the same device as W");
        TORCH_CHECK(s_in.scalar_type() == at::ScalarType::Half, "state_old must be FP16 (torch.float16)");
        TORCH_CHECK(s_in.dim() == 4 && s_in.size(0) == batch_size && s_in.size(1) == num_kv_heads
                    && s_in.size(2) == d_v && s_in.size(3) == d_k,
                    "state_old must be (B, H, d_v, d_k)");
        state_old_c = s_in.contiguous();
        use_state_old = true;
    }

    auto options = Y.options();
    // Reuse Y memory buffer to store S in-place to minimize peak VRAM
    at::Tensor S = Y;
    at::Tensor final_states;

    if (num_chunks == 0 || batch_size == 0) {
        if (batch_size == 0) {
            final_states = at::empty({0, num_kv_heads, d_v, d_k}, options);
        } else if (use_state_old) {
            final_states = state_old_c.clone();
        } else {
            final_states = at::zeros({batch_size, num_kv_heads, d_v, d_k}, options);
        }
        return std::make_tuple(S, final_states);
    }

    final_states = at::empty({batch_size, num_kv_heads, d_v, d_k}, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // -------------------------------------------------------------------------------------
    // Stage 2 - Kernel Launch
    // -------------------------------------------------------------------------------------
    {
        using T = cutlass::half_t;

        const T* p_w = reinterpret_cast<const T*>(W.data_ptr<at::Half>());
        T* p_ys = reinterpret_cast<T*>(S.data_ptr<at::Half>());
        const T* p_state_old = use_state_old ? reinterpret_cast<const T*>(state_old_c.data_ptr<at::Half>()) : nullptr;
        const int32_t* p_cu_chunk_lens = cu_chunk_lens_i32.data_ptr<int32_t>();
        T* p_final_states = reinterpret_cast<T*>(final_states.data_ptr<at::Half>());

        constexpr int NUM_THRDS    = 128;
        constexpr int DK_DIM       = 64;
        constexpr int DV_DIM       = 64;
        constexpr int BV_DIM       = 16;
        constexpr int ROWS_PER_STP = 16;

        using MMAAtomOp = SM75_16x8x8_F32F16F16F32_TN;
        using MMAAtom   = MMA_Atom<MMAAtomOp>;
        using TiledMMA  = TiledMMA<MMAAtom, Layout<Shape<_1, _4, _1>>>;

        using S2RCopyAtomA = Copy_Atom<SM75_U32x2_LDSM_N, T>;
        using S2RCopyAtomB = Copy_Atom<SM75_U16x2_LDSM_T, T>;

        using CopyAtom = Copy_Atom<UniversalCopy<uint4>, T>;
        using TiledCopyG2R = decltype(make_tiled_copy(
            CopyAtom{},
            make_layout(make_shape(Int<ROWS_PER_STP>{}, Int<DK_DIM / 8>{}), LayoutRight{}),
            make_layout(make_shape(Int<1>{}, Int<8>{}), LayoutRight{})
        ));
        using TiledCopyStore = TiledCopyG2R;

        dim3 grid(batch_size, num_kv_heads, DV_DIM / BV_DIM);
        dim3 block(NUM_THRDS);

        kernel_chunk_scan_fwd<
            T, NUM_THRDS, DK_DIM, DV_DIM, BV_DIM,
            MMAAtom, TiledMMA, S2RCopyAtomA, S2RCopyAtomB, TiledCopyG2R, TiledCopyStore
        ><<<grid, block, 0, stream>>>(
            p_w,
            p_ys,
            p_state_old,
            p_cu_chunk_lens,
            p_final_states,
            use_state_old,
            num_chunks,
            batch_size
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // -------------------------------------------------------------------------------------
    // Stage 3 - Return Results
    // -------------------------------------------------------------------------------------
    return std::make_tuple(S, final_states);
}
