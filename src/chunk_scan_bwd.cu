#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

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

// =========================================================================================
// SECTION 1: GLOBAL KERNEL (kernel_chunk_scan_bwd)
// =========================================================================================

// -----------------------------------------------------------------------------------------
// Global Kernel: kernel_chunk_scan_bwd
// -----------------------------------------------------------------------------------------
// Execution Flow:
// - Stage 1 - Setup & Grid Initialization:
//   + batch_idx = blockIdx.x, head_idx = blockIdx.y, v_idx = blockIdx.z, tid = threadIdx.x.
//   + Chunk boundaries: start_c = cu_chunk_lens[batch_idx], end_c = cu_chunk_lens[batch_idx + 1].
//   + V slice offset: slice_offset_v = v_idx * B_V * D_K.
//   + State layout offset (B, H, d_v, d_k): (batch_idx * gridDim.y + head_idx) * (d_v * d_k) + slice_offset_v.
// - Stage 2 - CuTe Tensor Partitioning & Shared Memory Allocation:
//   + Compact 10 KB SMEM: sDeltaS (2 KB for B_V x D_K), sW (8 KB for D_K x D_K, aliased as sDeltaS_accum).
//   + Swizzle<3, 3, 3> layouts to eliminate bank conflicts.
//   + Batched GEMM 1 (dW_c = S_c^T @ Delta_S) is deferred to cuBLAS at::bmm on Host.
//   + Single Tensor Core GEMM in kernel: Delta_S_accum = Delta_S @ W_c^T (K-major LDSM_N).
//   + TiledMMA with Layout<Shape<_1, _4, _1>> (1 warp vertical M=16, 4 warps horizontal N=64).
// - Stage 3 - Buffer Initialization & Register Fragments:
//   + NUM_STPS_V = B_V / ROWS_PER_STP (1 step), NUM_STPS_K = D_K / ROWS_PER_STP (4 steps).
//   + Allocate register fragments once at kernel start.
//   + Zero-length sequence handling (start_c >= end_c): write grad_final_states to grad_state_old, then return.
//   + Load grad_final_states into sDeltaS (or initialize to zero).
// - Stage 4 - Reverse Scan Loop c = end_c - 1 .. start_c:
//   + 1. Prefetch W_c from GMEM into registers rW_tmp to overlap memory latency.
//   + 2. Store gradient dY[:, c] = Delta_S to GMEM via 128-bit vectorized stores while prefetching W_c.
//   + 3. Copy rW_tmp to SMEM sW, synchronize via __syncthreads().
//   + 4. Perform GEMM: Delta_S_accum = Delta_S @ W_c^T, epilogue writes to sDeltaS_accum (aliased on sW).
//   + 5. Update Delta_S_new = Delta_S_accum + grad_S_list[:, c] in sDeltaS for chunk c - 1.
//   + At loop termination (c = start_c): if has_state_old, write Delta_S to GMEM grad_state_old[b, h].
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
__global__ void __launch_bounds__(NUM_THREADS) kernel_chunk_scan_bwd(
    const T* __restrict__ gGradSList,          // (num_kv_heads, num_chunks, d_v, d_k) [FP16]
    const T* __restrict__ gW,                  // (num_kv_heads, num_chunks, d_k, d_k) [FP16]
    const T* __restrict__ gGradFinalStates,    // (batch_size, num_kv_heads, d_v, d_k) [FP16] (nullable)
    const int32_t* __restrict__ cu_chunk_lens, // (batch_size + 1,) [Int32]
    T* __restrict__ gGradY,                    // (num_kv_heads, num_chunks, d_v, d_k) [FP16]
    T* __restrict__ gGradStateOld,             // (batch_size, num_kv_heads, d_v, d_k) [FP16] (nullable)
    bool has_state_old,                        // scalar [bool]
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
    // Compact 10 KB SMEM: sDeltaS (2 KB), sW (8 KB, aliased as sDeltaS_accum)
    // -------------------------------------------------------------------------------------
    __shared__ alignas(16) T sDeltaS[B_V * D_K]; // 2 KB: Delta_S accumulator for slice (16 x 64)
    __shared__ alignas(16) T sW[D_K * D_K];      // 8 KB: full W_c matrix (64 x 64), aliased as sDeltaS_accum

    T* sDeltaS_accum = sW; // Epilogue alias for GEMM

    using SwizzleOp = Swizzle<3, 3, 3>;

    auto smem_layout_flat_v = composition(SwizzleOp{}, make_layout(make_shape(Int<B_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    auto smem_layout_flat_k = composition(SwizzleOp{}, make_layout(make_shape(Int<D_K>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

    // GEMM Layouts: Delta_S_new = Delta_S @ W_c^T
    // Op A (Delta_S): Shape (M=16, K=64), Stride (64, 1) -> Row-Major (K-major)
    auto smem_layout_mma_A = composition(SwizzleOp{}, make_layout(make_shape(Int<B_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    // Op B (W_c): Shape (N=64, K=64), Stride (64, 1) -> Row-Major (A @ B^T = Delta_S @ W^T)
    auto smem_layout_mma_B = composition(SwizzleOp{}, make_layout(make_shape(Int<D_K>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
    // Op C (Delta_S_accum): Shape (M=16, N=64), Stride (64, 1) -> Row-Major
    auto smem_layout_mma_C = composition(SwizzleOp{}, make_layout(make_shape(Int<B_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

    auto sDeltaS_flat = make_tensor(make_smem_ptr(sDeltaS), smem_layout_flat_v);
    auto sW_flat      = make_tensor(make_smem_ptr(sW),      smem_layout_flat_k);
    auto sAccum_flat  = make_tensor(make_smem_ptr(sDeltaS_accum), smem_layout_flat_v);

    auto sDeltaS_mma = make_tensor(make_smem_ptr(sDeltaS), smem_layout_mma_A);
    auto sW_mma      = make_tensor(make_smem_ptr(sW),      smem_layout_mma_B);
    auto sAccum_mma  = make_tensor(make_smem_ptr(sDeltaS_accum), smem_layout_mma_C);

    // TiledMMA & Partitioning (Layout<Shape<_1, _4, _1>>)
    TiledMMAType tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    // S2R Copy for GEMM (LDSM_N)
    auto s2r_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_copy_a.get_slice(tid);
    auto s2r_thr_copy_b = s2r_copy_b.get_slice(tid);

    auto tCsA = s2r_thr_copy_a.partition_S(sDeltaS_mma);
    auto tCsB = s2r_thr_copy_b.partition_S(sW_mma);
    auto tCsC = thr_mma.partition_C(sAccum_mma);

    auto tCrA = thr_mma.partition_fragment_A(sDeltaS_mma);
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

    auto dummy_tile = local_tile(sDeltaS_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(0, 0));
    auto dummy_part = thr_copy_g2r.partition_S(dummy_tile);

    decltype(make_fragment_like(dummy_part)) rTmp[NUM_STPS_V];
    decltype(make_fragment_like(dummy_part)) rW_tmp[NUM_STPS_K];

    #pragma unroll
    for (int st = 0; st < NUM_STPS_V; ++st) {
        clear(rTmp[st]);
    }
    #pragma unroll
    for (int st = 0; st < NUM_STPS_K; ++st) {
        clear(rW_tmp[st]);
    }

    // -------------------------------------------------------------------------------------
    // Zero-length sequence handling: start_c >= end_c
    // If has_state_old: write grad_final_states (or 0) to grad_state_old
    // -------------------------------------------------------------------------------------
    if (start_c >= end_c) {
        if (has_state_old && gGradStateOld != nullptr) {
            T* p_grad_state_old = gGradStateOld + state_offset;
            auto gGradStateOld_chunk = make_tensor(make_gmem_ptr(p_grad_state_old), make_layout(make_shape(Int<B_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
            if (gGradFinalStates != nullptr) {
                const T* p_grad_final = gGradFinalStates + state_offset;
                auto gGradFinal_chunk = make_tensor(make_gmem_ptr(p_grad_final), make_layout(make_shape(Int<B_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));
                #pragma unroll
                for (int st = 0; st < NUM_STPS_V; ++st) {
                    auto gGradFinal_step    = local_tile(gGradFinal_chunk,    make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
                    auto gGradStateOld_step = local_tile(gGradStateOld_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
                    cute::copy(tiled_copy_g2r, thr_copy_g2r.partition_S(gGradFinal_step), rTmp[st]);
                    auto tD_gstate = thr_copy_store.partition_D(gGradStateOld_step);
                    *reinterpret_cast<uint4*>(&tD_gstate(0)) = *reinterpret_cast<const uint4*>(&rTmp[st](0));
                }
            } else {
                #pragma unroll
                for (int st = 0; st < NUM_STPS_V; ++st) {
                    auto gGradStateOld_step = local_tile(gGradStateOld_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
                    auto tD_gstate = thr_copy_store.partition_D(gGradStateOld_step);
                    *reinterpret_cast<uint4*>(&tD_gstate(0)) = *reinterpret_cast<const uint4*>(&rTmp[st](0));
                }
            }
        }
        return;
    }

    // -------------------------------------------------------------------------------------
    // Stage 3: Initialize Delta_S Accumulator from grad_final_states
    // -------------------------------------------------------------------------------------
    if (gGradFinalStates != nullptr) {
        const T* p_grad_final = gGradFinalStates + state_offset;
        auto gGradFinal_chunk = make_tensor(make_gmem_ptr(p_grad_final), make_layout(make_shape(Int<B_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

        #pragma unroll
        for (int st = 0; st < NUM_STPS_V; ++st) {
            auto gGradFinal_step = local_tile(gGradFinal_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
            auto sDeltaS_step    = local_tile(sDeltaS_flat,     make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));

            cute::copy(tiled_copy_g2r, thr_copy_g2r.partition_S(gGradFinal_step), rTmp[st]);
            auto tDeltaS = thr_copy_g2r.partition_D(sDeltaS_step);
            *reinterpret_cast<uint4*>(&tDeltaS(0)) = *reinterpret_cast<const uint4*>(&rTmp[st](0));
        }
    } else {
        #pragma unroll
        for (int st = 0; st < NUM_STPS_V; ++st) {
            auto sDeltaS_step = local_tile(sDeltaS_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
            clear(rTmp[st]);
            auto tDeltaS = thr_copy_g2r.partition_D(sDeltaS_step);
            *reinterpret_cast<uint4*>(&tDeltaS(0)) = *reinterpret_cast<const uint4*>(&rTmp[st](0));
        }
    }
    __syncthreads();

    // -------------------------------------------------------------------------------------
    // Stage 4: Reverse Scan Loop c = end_c - 1 .. start_c
    // -------------------------------------------------------------------------------------
    for (int32_t c = end_c - 1; c >= start_c; --c) {
        int64_t chunk_mat_offset_v = ((int64_t)head_idx * num_chunks + c) * mat_elements_v + slice_offset_v;
        int64_t chunk_mat_offset_k = ((int64_t)head_idx * num_chunks + c) * mat_elements_k;

        // 1. Prefetch W_c from GMEM into registers rW_tmp (NUM_STPS_K steps)
        const T* p_w_curr = gW + chunk_mat_offset_k;
        auto gW_chunk = make_tensor(make_gmem_ptr(p_w_curr), make_layout(make_shape(Int<D_K>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

        #pragma unroll
        for (int st = 0; st < NUM_STPS_K; ++st) {
            auto gW_step = local_tile(gW_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
            cute::copy(tiled_copy_g2r, thr_copy_g2r.partition_S(gW_step), rW_tmp[st]);
        }

        // 2. Store gradient dY[:, c] = Delta_S to GMEM while prefetching W_c
        T* p_gy = gGradY + chunk_mat_offset_v;
        auto gGradY_chunk = make_tensor(make_gmem_ptr(p_gy), make_layout(make_shape(Int<B_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

        #pragma unroll
        for (int st = 0; st < NUM_STPS_V; ++st) {
            auto sDeltaS_step = local_tile(sDeltaS_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
            auto gGradY_step  = local_tile(gGradY_chunk,  make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));

            auto tS_delta = thr_copy_store.partition_S(sDeltaS_step);
            decltype(make_fragment_like(tS_delta)) rDelta_slice;
            cute::copy(tS_delta, rDelta_slice);

            auto tD_gy = thr_copy_store.partition_D(gGradY_step);
            *reinterpret_cast<uint4*>(&tD_gy(0)) = *reinterpret_cast<const uint4*>(&rDelta_slice(0));
        }

        // 3. Store W_c from registers rW_tmp to SMEM sW
        #pragma unroll
        for (int st = 0; st < NUM_STPS_K; ++st) {
            auto sW_step = local_tile(sW_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
            auto tWsW    = thr_copy_g2r.partition_D(sW_step);
            *reinterpret_cast<uint4*>(&tWsW(0)) = *reinterpret_cast<const uint4*>(&rW_tmp[st](0));
        }
        __syncthreads(); // Synchronize to ensure sW and sDeltaS are ready for GEMM

        // 4. GEMM: Delta_S_accum = Delta_S @ W_c^T
        auto tCrC_DeltaS = thr_mma.make_fragment_C(tCsC);
        clear(tCrC_DeltaS);

        cute::copy(s2r_copy_a, tCsA, tCrA_view);
        cute::copy(s2r_copy_b, tCsB, tCrB_view);
        __syncthreads();

        cute::gemm(MMAAtom{}, tCrA, tCrB, tCrC_DeltaS);

        // GEMM Epilogue: Convert to FP16 and write to sDeltaS_accum (aliased on sW)
        auto rC_half_DeltaS = make_tensor_like<T>(tCrC_DeltaS);
        #pragma unroll
        for (int i = 0; i < size(rC_half_DeltaS); ++i) {
            rC_half_DeltaS(i) = float_to_out<T>(tCrC_DeltaS(i));
        }
        cute::copy(rC_half_DeltaS, tCsC);
        __syncthreads();

        // 5. Accumulate Delta_S_new = Delta_S_accum + grad_S_list[:, c] into sDeltaS
        const T* p_grad_s = gGradSList + chunk_mat_offset_v;
        auto gGradS_chunk = make_tensor(make_gmem_ptr(p_grad_s), make_layout(make_shape(Int<B_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

        #pragma unroll
        for (int st = 0; st < NUM_STPS_V; ++st) {
            auto sAccum_step  = local_tile(sAccum_flat,  make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
            auto gGradS_step  = local_tile(gGradS_chunk,  make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
            auto sDeltaS_step = local_tile(sDeltaS_flat, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));

            auto tS_accum = thr_copy_store.partition_S(sAccum_step);
            decltype(make_fragment_like(tS_accum)) rAccum_slice;
            cute::copy(tS_accum, rAccum_slice);

            decltype(make_fragment_like(tS_accum)) rGradS_slice;
            cute::copy(tiled_copy_g2r, thr_copy_g2r.partition_S(gGradS_step), rGradS_slice);

            decltype(make_fragment_like(tS_accum)) rNew_delta;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float acc_f = half_to_float(rAccum_slice(i));
                float gs_f  = half_to_float(rGradS_slice(i));
                rNew_delta(i) = float_to_out<T>(acc_f + gs_f);
            }

            auto tD_new_delta = thr_copy_g2r.partition_D(sDeltaS_step);
            *reinterpret_cast<uint4*>(&tD_new_delta(0)) = *reinterpret_cast<const uint4*>(&rNew_delta(0));
        }
        __syncthreads();
    }

    // -------------------------------------------------------------------------------------
    // Stage 5: Write initial state gradient (grad_state_old) if requested
    // -------------------------------------------------------------------------------------
    if (has_state_old && gGradStateOld != nullptr) {
        T* p_grad_state_old = gGradStateOld + state_offset;
        auto gGradStateOld_chunk = make_tensor(make_gmem_ptr(p_grad_state_old), make_layout(make_shape(Int<B_V>{}, Int<D_K>{}), make_stride(Int<D_K>{}, Int<1>{})));

        #pragma unroll
        for (int st = 0; st < NUM_STPS_V; ++st) {
            auto sDeltaS_step       = local_tile(sDeltaS_flat,        make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));
            auto gGradStateOld_step = local_tile(gGradStateOld_chunk, make_shape(Int<ROWS_PER_STP>{}, Int<D_K>{}), make_coord(st, 0));

            auto tS_delta = thr_copy_store.partition_S(sDeltaS_step);
            decltype(make_fragment_like(tS_delta)) rDelta_slice;
            cute::copy(tS_delta, rDelta_slice);

            auto tD_gstate = thr_copy_store.partition_D(gGradStateOld_step);
            *reinterpret_cast<uint4*>(&tD_gstate(0)) = *reinterpret_cast<const uint4*>(&rDelta_slice(0));
        }
    }
}

// =========================================================================================
// SECTION 2: HOST COORDINATION FUNCTION (chunk_scan_bwd)
// =========================================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor>
chunk_scan_bwd(
    const at::Tensor& grad_S_list,                     // (num_kv_heads, num_chunks, d_v, d_k) [FP16]
    const at::Tensor& W,                               // (num_kv_heads, num_chunks, d_k, d_k) [FP16]
    const at::Tensor& S_list,                          // (num_kv_heads, num_chunks, d_v, d_k) [FP16]
    const at::Tensor& cu_seq_lens,                     // (batch_size + 1,) [Int32 / Int64]
    const std::optional<at::Tensor>& grad_final_states, // (batch_size, num_kv_heads, d_v, d_k) [FP16]
    bool has_state_old,                                // scalar [bool]
    int64_t chunk_size                                 // scalar [int64] (default 64)
) {
    // -------------------------------------------------------------------------------------
    // Stage 1 - Input Validation
    // -------------------------------------------------------------------------------------
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
    const c10::cuda::CUDAGuard device_guard(W.device());

    TORCH_CHECK(grad_S_list.is_cuda(), "grad_S_list must be a CUDA tensor");
    TORCH_CHECK(S_list.is_cuda(), "S_list must be a CUDA tensor");
    TORCH_CHECK(cu_seq_lens.is_cuda(), "cu_seq_lens must be a CUDA tensor");

    TORCH_CHECK(grad_S_list.device() == W.device(), "grad_S_list must be on the same device as W");
    TORCH_CHECK(S_list.device() == W.device(), "S_list must be on the same device as W");
    TORCH_CHECK(cu_seq_lens.device() == W.device(), "cu_seq_lens must be on the same device as W");

    TORCH_CHECK(grad_S_list.scalar_type() == at::ScalarType::Half, "grad_S_list must be FP16 (torch.float16)");
    TORCH_CHECK(W.scalar_type() == at::ScalarType::Half, "W must be FP16 (torch.float16)");
    TORCH_CHECK(S_list.scalar_type() == at::ScalarType::Half, "S_list must be FP16 (torch.float16)");
    TORCH_CHECK(cu_seq_lens.scalar_type() == at::ScalarType::Int || cu_seq_lens.scalar_type() == at::ScalarType::Long,
                "cu_seq_lens must be Int32 or Int64");

    TORCH_CHECK(grad_S_list.dim() == 4, "grad_S_list must be 4D tensor with shape (num_kv_heads, num_chunks, d_v, d_k)");
    TORCH_CHECK(W.dim() == 4, "W must be 4D tensor with shape (num_kv_heads, num_chunks, d_k, d_k)");
    TORCH_CHECK(S_list.dim() == 4, "S_list must be 4D tensor with shape (num_kv_heads, num_chunks, d_v, d_k)");
    TORCH_CHECK(cu_seq_lens.dim() == 1, "cu_seq_lens must be 1D tensor with shape (batch_size + 1,)");

    const at::Tensor gS = grad_S_list.contiguous();
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
    TORCH_CHECK(S_list.is_contiguous(), "S_list must be contiguous");

    int64_t num_kv_heads = W.size(0);
    int64_t num_chunks   = W.size(1);
    int64_t d_k          = W.size(2);
    int64_t d_k_col      = W.size(3);

    int64_t num_s_heads  = S_list.size(0);
    int64_t num_s_chunks = S_list.size(1);
    int64_t d_v          = S_list.size(2);
    int64_t d_v_k        = S_list.size(3);

    TORCH_CHECK(num_s_heads == num_kv_heads && num_s_chunks == num_chunks,
                "S_list dimensions (num_kv_heads, num_chunks) must match W");
    TORCH_CHECK(gS.size(0) == num_kv_heads && gS.size(1) == num_chunks,
                "grad_S_list dimensions (num_kv_heads, num_chunks) must match W");
    TORCH_CHECK(gS.size(2) == d_v && gS.size(3) == d_k,
                "grad_S_list trailing dimensions must match S_list (d_v, d_k)");

    TORCH_CHECK(d_k == 64 && d_k_col == 64, "W trailing dimensions must be 64x64 (d_k == 64 on SM75+)");
    TORCH_CHECK(d_v == 64 && d_v_k == 64, "S_list trailing dimensions must be 64x64 (d_v == 64 on SM75+)");
    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64 for optimal SM alignment");

    int64_t batch_size = cu_seq_lens.size(0) - 1;
    TORCH_CHECK(batch_size >= 0, "batch_size must be >= 0");

    at::Tensor cu_chunk_lens = compute_cu_chunk_lens(cu_seq_lens, chunk_size);
    at::Tensor cu_chunk_lens_i32 = cu_chunk_lens.to(at::kInt).contiguous();

    bool has_grad_final = false;
    at::Tensor grad_final_states_c;

    if (grad_final_states.has_value() && grad_final_states.value().defined()) {
        const at::Tensor& gf_in = grad_final_states.value();
        TORCH_CHECK(gf_in.is_cuda(), "grad_final_states must be a CUDA tensor");
        TORCH_CHECK(gf_in.device() == W.device(), "grad_final_states must be on the same device as W");
        TORCH_CHECK(gf_in.scalar_type() == at::ScalarType::Half, "grad_final_states must be FP16 (torch.float16)");
        TORCH_CHECK(gf_in.dim() == 4 && gf_in.size(0) == batch_size && gf_in.size(1) == num_kv_heads
                    && gf_in.size(2) == d_v && gf_in.size(3) == d_k,
                    "grad_final_states must be (B, H, d_v, d_k)");
        grad_final_states_c = gf_in.contiguous();
        has_grad_final = true;
    }

    auto options = W.options();
    at::Tensor grad_Y = at::zeros_like(gS);
    at::Tensor grad_state_old;

    if (num_chunks == 0 || batch_size == 0) {
        at::Tensor grad_W = at::zeros_like(W);
        if (has_state_old) {
            if (batch_size == 0) {
                grad_state_old = at::empty({0, num_kv_heads, d_v, d_k}, options);
            } else if (has_grad_final) {
                grad_state_old = grad_final_states_c.clone();
            } else {
                grad_state_old = at::zeros({batch_size, num_kv_heads, d_v, d_k}, options);
            }
        } else {
            grad_state_old = at::Tensor();
        }
        return std::make_tuple(grad_W, grad_Y, grad_state_old);
    }

    if (has_state_old) {
        grad_state_old = at::empty({batch_size, num_kv_heads, d_v, d_k}, options);
    } else {
        grad_state_old = at::Tensor();
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // -------------------------------------------------------------------------------------
    // Stage 2 - Reverse Time Scan Kernel Launch
    // -------------------------------------------------------------------------------------
    {
        using T = cutlass::half_t;

        const T* p_grad_s_list   = reinterpret_cast<const T*>(gS.data_ptr<at::Half>());
        const T* p_w             = reinterpret_cast<const T*>(W.data_ptr<at::Half>());
        const T* p_grad_final    = has_grad_final ? reinterpret_cast<const T*>(grad_final_states_c.data_ptr<at::Half>()) : nullptr;
        const int32_t* p_cu_chunk_lens = cu_chunk_lens_i32.data_ptr<int32_t>();

        T* p_grad_y         = reinterpret_cast<T*>(grad_Y.data_ptr<at::Half>());
        T* p_grad_state_old = has_state_old ? reinterpret_cast<T*>(grad_state_old.data_ptr<at::Half>()) : nullptr;

        constexpr int NUM_THRDS    = 128;
        constexpr int DK_DIM       = 64;
        constexpr int DV_DIM       = 64;
        constexpr int BV_DIM       = 16;
        constexpr int ROWS_PER_STP = 16;

        using MMAAtomOp = SM75_16x8x8_F32F16F16F32_TN;
        using MMAAtom   = MMA_Atom<MMAAtomOp>;
        using TiledMMA  = TiledMMA<MMAAtom, Layout<Shape<_1, _4, _1>, Stride<_1, _1, _4>>>;

        using S2RCopyAtomA = Copy_Atom<SM75_U32x2_LDSM_N, T>;
        using S2RCopyAtomB = Copy_Atom<SM75_U32x1_LDSM_N, T>;

        using CopyAtom = Copy_Atom<UniversalCopy<uint4>, T>;
        using TiledCopyG2R = decltype(make_tiled_copy(
            CopyAtom{},
            make_layout(make_shape(Int<ROWS_PER_STP>{}, Int<DK_DIM / 8>{}), LayoutRight{}),
            make_layout(make_shape(Int<1>{}, Int<8>{}), LayoutRight{})
        ));
        using TiledCopyStore = TiledCopyG2R;

        dim3 grid(batch_size, num_kv_heads, DV_DIM / BV_DIM);
        dim3 block(NUM_THRDS);

        kernel_chunk_scan_bwd<
            T, NUM_THRDS, DK_DIM, DV_DIM, BV_DIM,
            MMAAtom, TiledMMA, S2RCopyAtomA, S2RCopyAtomB,
            TiledCopyG2R, TiledCopyStore
        ><<<grid, block, 0, stream>>>(
            p_grad_s_list,
            p_w,
            p_grad_final,
            p_cu_chunk_lens,
            p_grad_y,
            p_grad_state_old,
            has_state_old,
            num_chunks,
            batch_size
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // -------------------------------------------------------------------------------------
    // Stage 3 - Compute grad_W via Batched GEMM (at::bmm) on Host
    // -------------------------------------------------------------------------------------
    // Parallel GEMM (dW = S^T @ dY) across all chunks via cuBLAS bmm:
    // S_list: (num_kv_heads, num_chunks, d_v, d_k) -> flatten(0, 1) -> transpose(-1, -2) -> (H*C, d_k, d_v)
    // grad_Y: (num_kv_heads, num_chunks, d_v, d_k) -> flatten(0, 1) -> (H*C, d_v, d_k)
    // bmm: (H*C, d_k, d_v) @ (H*C, d_v, d_k) -> (H*C, d_k, d_k) -> view_as(W)
    at::Tensor grad_W = at::bmm(
        S_list.flatten(0, 1).transpose(-1, -2),
        grad_Y.flatten(0, 1)
    ).view_as(W);

    // -------------------------------------------------------------------------------------
    // Stage 4 - Return Results
    // -------------------------------------------------------------------------------------
    return std::make_tuple(grad_W, grad_Y, grad_state_old);
}
