#include <torch/extension.h>
#include <torch/library.h>
#include "deltanet_ops.cuh"

// 1. PyTorch Dispatcher Registration (TORCH_LIBRARY)

TORCH_LIBRARY(deltanet, m) {
    // chunk_wy_fwd: (W, Y)
    m.def("chunk_wy_fwd(Tensor k, Tensor v, Tensor b, Tensor w, Tensor gamma, Tensor a, Tensor cu_seq_lens, int chunk_size=64) -> (Tensor, Tensor)");

    // chunk_wy_bwd: (dk, dv, db, dw, dgamma, da)
    m.def("chunk_wy_bwd(Tensor dW, Tensor dY, Tensor k, Tensor v, Tensor b, Tensor w, Tensor gamma, Tensor a, Tensor cu_seq_lens, Tensor dk_ext, Tensor dv_ext, Tensor da_ext, int chunk_size=64) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

    // chunk_scan_fwd: (S, final_states)
    m.def("chunk_scan_fwd(Tensor W, Tensor Y, Tensor? state_old, Tensor cu_seq_lens, int chunk_size=64) -> (Tensor, Tensor)");

    // chunk_scan_bwd: (grad_W, grad_Y, grad_state_old)
    m.def("chunk_scan_bwd(Tensor grad_S_list, Tensor W, Tensor S_list, Tensor cu_seq_lens, Tensor? grad_final_states=None, bool has_state_old=False, int chunk_size=64) -> (Tensor, Tensor, Tensor)");

    // output_combine_fwd: out
    m.def("output_combine_fwd(Tensor q, Tensor k, Tensor v, Tensor b, Tensor w, Tensor gamma, Tensor a, Tensor S, Tensor cu_seq_lens, int num_q_heads, int num_kv_heads, int d_k=64, int d_v=64, int chunk_size=64) -> Tensor");

    // output_combine_bwd: (dq, dk_ext, dv_ext, da_ext, dS)
    m.def("output_combine_bwd(Tensor grad_output, Tensor q, Tensor k, Tensor v, Tensor b, Tensor w, Tensor gamma, Tensor a, Tensor S, Tensor cu_seq_lens, int num_q_heads, int num_kv_heads, int d_k=64, int d_v=64, int chunk_size=64) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

    // deltanet_fwd: (y, final_states, W, S)
    m.def("deltanet_fwd(Tensor? past_states, Tensor q, Tensor k, Tensor v, Tensor w, Tensor b, Tensor a, Tensor gamma, Tensor cu_seq_lens, int num_q_heads, int num_kv_heads, int d_k=64, int d_v=64, int chunk_size=64, bool forward_only=False) -> (Tensor, Tensor, Tensor, Tensor)");

    // deltanet_bwd: (dq, dk, dv, dw, db, da, dgamma, grad_past_states)
    m.def("deltanet_bwd(Tensor grad_output, Tensor? grad_final_states, Tensor? past_states, Tensor q, Tensor k, Tensor v, Tensor w, Tensor b, Tensor a, Tensor gamma, Tensor cu_seq_lens, Tensor? W=None, Tensor? S=None, int num_q_heads=1, int num_kv_heads=1, int d_k=64, int d_v=64, int chunk_size=64, bool has_past_states=False) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

    // deltanet_infer: (out, past_state)
    m.def("deltanet_infer(Tensor past_state, Tensor q, Tensor k, Tensor v, Tensor b, Tensor w, Tensor a, Tensor gamma, int num_q_heads, int num_kv_heads, int d_k=64, int d_v=64) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(deltanet, CUDA, m) {
    m.impl("chunk_wy_fwd", &chunk_wy_fwd);
    m.impl("chunk_wy_bwd", &chunk_wy_bwd);
    m.impl("chunk_scan_fwd", &chunk_scan_fwd);
    m.impl("chunk_scan_bwd", &chunk_scan_bwd);
    m.impl("output_combine_fwd", &output_combine_fwd);
    m.impl("output_combine_bwd", &output_combine_bwd);
    m.impl("deltanet_fwd", &deltanet_fwd);
    m.impl("deltanet_bwd", &deltanet_bwd);
    m.impl("deltanet_infer", &deltanet_infer);
}

// 2. Python C++ Extension Module (PYBIND11_MODULE)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeltaNet v2 CUDA Extension (Turing SM75+ / CuTe)";

    // 1. Chunk WY Forward
    // Return values:
    // 1. W: (num_kv_heads, num_chunks, d_k, d_k) [FP16] - Chunk transition matrix W
    // 2. Y: (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Chunk update matrix Y
    m.def("chunk_wy_fwd", &chunk_wy_fwd,
          "Chunk WY Forward Coordination\n\n"
          "Outputs:\n"
          "  1. W: (num_kv_heads, num_chunks, d_k, d_k) [FP16] - Chunk transition matrix\n"
          "  2. Y: (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Chunk update matrix",
          py::arg("k"), py::arg("v"), py::arg("b"), py::arg("w"), py::arg("gamma"), py::arg("a"),
          py::arg("cu_seq_lens"), py::arg("chunk_size") = 64);

    // 2. Chunk WY Backward
    // Return values:
    // 1. dk:     (num_kv_heads, total_L, d_k) [FP16] - Gradient w.r.t raw Key
    // 2. dv:     (num_kv_heads, total_L, d_v) [FP16] - Gradient w.r.t raw Value
    // 3. db:     (num_kv_heads, total_L, d_k) [FP16] - Gradient w.r.t raw Beta gate
    // 4. dw:     (num_kv_heads, total_L, d_v) [FP16] - Gradient w.r.t raw W gate
    // 5. dgamma: (num_kv_heads, total_L, 1)   [FP16] - Gradient w.r.t raw Gamma
    // 6. da:     (num_kv_heads, total_L, 1)   [FP16] - Gradient w.r.t raw Alpha
    m.def("chunk_wy_bwd", &chunk_wy_bwd,
          "Chunk WY Backward Coordination\n\n"
          "Outputs:\n"
          "  1. dk:     (num_kv_heads, total_L, d_k) [FP16] - Gradient w.r.t raw Key\n"
          "  2. dv:     (num_kv_heads, total_L, d_v) [FP16] - Gradient w.r.t raw Value\n"
          "  3. db:     (num_kv_heads, total_L, d_k) [FP16] - Gradient w.r.t raw Beta gate\n"
          "  4. dw:     (num_kv_heads, total_L, d_v) [FP16] - Gradient w.r.t raw W gate\n"
          "  5. dgamma: (num_kv_heads, total_L, 1)   [FP16] - Gradient w.r.t raw Gamma\n"
          "  6. da:     (num_kv_heads, total_L, 1)   [FP16] - Gradient w.r.t raw Alpha",
          py::arg("dW"), py::arg("dY"), py::arg("k"), py::arg("v"), py::arg("b"), py::arg("w"),
          py::arg("gamma"), py::arg("a"), py::arg("cu_seq_lens"),
          py::arg("dk_ext"), py::arg("dv_ext"), py::arg("da_ext"),
          py::arg("chunk_size") = 64);

    // 3. Chunk Scan Forward
    // Return values:
    // 1. S:            (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Recurrent states per chunk
    // 2. final_states: (batch_size, num_kv_heads, d_v, d_k) [FP16] - Final hidden states
    m.def("chunk_scan_fwd", &chunk_scan_fwd,
          "Chunk Scan Forward Recurrence\n\n"
          "Outputs:\n"
          "  1. S:            (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Recurrent states per chunk\n"
          "  2. final_states: (batch_size, num_kv_heads, d_v, d_k) [FP16] - Final hidden states",
          py::arg("W"), py::arg("Y"), py::arg("state_old") = py::none(),
          py::arg("cu_seq_lens"), py::arg("chunk_size") = 64);

    // 4. Chunk Scan Backward
    // Return values:
    // 1. grad_W:         (num_kv_heads, num_chunks, d_k, d_k) [FP16] - Gradient w.r.t matrix W
    // 2. grad_Y:         (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Gradient w.r.t matrix Y
    // 3. grad_state_old: (batch_size, num_kv_heads, d_v, d_k) [FP16] - Gradient w.r.t state_old (if provided)",
    m.def("chunk_scan_bwd", &chunk_scan_bwd,
          "Chunk Scan Backward Recurrence\n\n"
          "Outputs:\n"
          "  1. grad_W:         (num_kv_heads, num_chunks, d_k, d_k) [FP16] - Gradient w.r.t matrix W\n"
          "  2. grad_Y:         (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Gradient w.r.t matrix Y\n"
          "  3. grad_state_old: (batch_size, num_kv_heads, d_v, d_k) [FP16] - Gradient w.r.t state_old (if provided)",
          py::arg("grad_S_list"), py::arg("W"), py::arg("S_list"), py::arg("cu_seq_lens"),
          py::arg("grad_final_states") = py::none(), py::arg("has_state_old") = false,
          py::arg("chunk_size") = 64);

    // 5. Output Combine Forward
    // Return values:
    // 1. out: (num_q_heads, total_L, d_v) [FP16] - Combined attention output
    m.def("output_combine_fwd", &output_combine_fwd,
          "Output Combine Forward\n\n"
          "Outputs:\n"
          "  1. out: (num_q_heads, total_L, d_v) [FP16] - Combined attention output",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("b"), py::arg("w"),
          py::arg("gamma"), py::arg("a"), py::arg("S"), py::arg("cu_seq_lens"),
          py::arg("num_q_heads"), py::arg("num_kv_heads"),
          py::arg("d_k") = 64, py::arg("d_v") = 64, py::arg("chunk_size") = 64);

    // 6. Output Combine Backward
    // Return values:
    // 1. dq:     (num_q_heads, total_L, d_k)          [FP16] - Gradient w.r.t raw Query
    // 2. dk_ext: (num_kv_heads, total_L, d_k)         [FP16] - External gradient w.r.t k_scaled
    // 3. dv_ext: (num_kv_heads, total_L, d_v)         [FP16] - External gradient w.r.t v_new
    // 4. da_ext: (num_kv_heads, total_L, 1)           [FP32] - External gradient w.r.t log_alpha_cumsum
    // 5. dS:     (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Gradient w.r.t recurrent state S
    m.def("output_combine_bwd", &output_combine_bwd,
          "Output Combine Backward\n\n"
          "Outputs:\n"
          "  1. dq:     (num_q_heads, total_L, d_k)          [FP16] - Gradient w.r.t raw Query\n"
          "  2. dk_ext: (num_kv_heads, total_L, d_k)         [FP16] - External gradient w.r.t k_scaled\n"
          "  3. dv_ext: (num_kv_heads, total_L, d_v)         [FP16] - External gradient w.r.t v_new\n"
          "  4. da_ext: (num_kv_heads, total_L, 1)           [FP32] - External gradient w.r.t log_alpha_cumsum\n"
          "  5. dS:     (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Gradient w.r.t recurrent state S",
          py::arg("grad_output"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("b"),
          py::arg("w"), py::arg("gamma"), py::arg("a"), py::arg("S"), py::arg("cu_seq_lens"),
          py::arg("num_q_heads"), py::arg("num_kv_heads"),
          py::arg("d_k") = 64, py::arg("d_v") = 64, py::arg("chunk_size") = 64);

    // 7. DeltaNet Forward Coordination
    // Return values:
    // 1. y:            (num_q_heads, total_L, d_v)          [FP16] - Combined attention output
    // 2. final_states: (batch_size, num_kv_heads, d_v, d_k) [FP16] - Final hidden states
    // 3. W:            (num_kv_heads, num_chunks, d_k, d_k) [FP16] - Transition matrix W (empty tensor if forward_only=True)
    // 4. S:            (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Recurrent states S (empty tensor if forward_only=True)
    m.def("deltanet_fwd", &deltanet_fwd,
          "DeltaNet Forward Coordination\n\n"
          "Outputs:\n"
          "  1. y:            (num_q_heads, total_L, d_v)          [FP16] - Combined attention output\n"
          "  2. final_states: (batch_size, num_kv_heads, d_v, d_k) [FP16] - Final hidden states\n"
          "  3. W:            (num_kv_heads, num_chunks, d_k, d_k) [FP16] - Transition matrix W (empty tensor if forward_only=True)\n"
          "  4. S:            (num_kv_heads, num_chunks, d_v, d_k) [FP16] - Recurrent states S (empty tensor if forward_only=True)",
          py::arg("past_states") = py::none(), py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("w"), py::arg("b"), py::arg("a"), py::arg("gamma"), py::arg("cu_seq_lens"),
          py::arg("num_q_heads"), py::arg("num_kv_heads"),
          py::arg("d_k") = 64, py::arg("d_v") = 64, py::arg("chunk_size") = 64,
          py::arg("forward_only") = false);

    // 8. DeltaNet Backward Coordination
    // Return values:
    // 1. dq:               (num_q_heads, total_L, d_k)          [FP16] - Gradient w.r.t raw Query
    // 2. dk:               (num_kv_heads, total_L, d_k)         [FP16] - Gradient w.r.t raw Key
    // 3. dv:               (num_kv_heads, total_L, d_v)         [FP16] - Gradient w.r.t raw Value
    // 4. dw:               (num_kv_heads, total_L, d_v)         [FP16] - Gradient w.r.t raw W gate
    // 5. db:               (num_kv_heads, total_L, d_k)         [FP16] - Gradient w.r.t raw Beta gate
    // 6. da:               (num_kv_heads, total_L, 1)           [FP16] - Gradient w.r.t raw Alpha
    // 7. dgamma:           (num_kv_heads, total_L, 1)           [FP16] - Gradient w.r.t raw Gamma
    // 8. grad_past_states: (batch_size, num_kv_heads, d_v, d_k) [FP16] - Gradient w.r.t past_states (if provided)
    m.def("deltanet_bwd", &deltanet_bwd,
          "DeltaNet Backward Coordination\n\n"
          "Outputs:\n"
          "  1. dq:               (num_q_heads, total_L, d_k)          [FP16] - Gradient w.r.t raw Query\n"
          "  2. dk:               (num_kv_heads, total_L, d_k)         [FP16] - Gradient w.r.t raw Key\n"
          "  3. dv:               (num_kv_heads, total_L, d_v)         [FP16] - Gradient w.r.t raw Value\n"
          "  4. dw:               (num_kv_heads, total_L, d_v)         [FP16] - Gradient w.r.t raw W gate\n"
          "  5. db:               (num_kv_heads, total_L, d_k)         [FP16] - Gradient w.r.t raw Beta gate\n"
          "  6. da:               (num_kv_heads, total_L, 1)           [FP16] - Gradient w.r.t raw Alpha\n"
          "  7. dgamma:           (num_kv_heads, total_L, 1)           [FP16] - Gradient w.r.t raw Gamma\n"
          "  8. grad_past_states: (batch_size, num_kv_heads, d_v, d_k) [FP16] - Gradient w.r.t past_states (if provided)",
          py::arg("grad_output"), py::arg("grad_final_states") = py::none(),
          py::arg("past_states") = py::none(), py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("w"), py::arg("b"), py::arg("a"), py::arg("gamma"), py::arg("cu_seq_lens"),
          py::arg("W") = py::none(), py::arg("S") = py::none(), py::arg("num_q_heads"), py::arg("num_kv_heads"),
          py::arg("d_k") = 64, py::arg("d_v") = 64, py::arg("chunk_size") = 64,
          py::arg("has_past_states") = false);

    // Alias delta_net_bwd
    m.def("delta_net_bwd", &deltanet_bwd,
          "DeltaNet Backward Coordination (Alias)\n\n"
          "Outputs:\n"
          "  1. dq:               (num_q_heads, total_L, d_k)          [FP16] - Gradient w.r.t raw Query\n"
          "  2. dk:               (num_kv_heads, total_L, d_k)         [FP16] - Gradient w.r.t raw Key\n"
          "  3. dv:               (num_kv_heads, total_L, d_v)         [FP16] - Gradient w.r.t raw Value\n"
          "  4. dw:               (num_kv_heads, total_L, d_v)         [FP16] - Gradient w.r.t raw W gate\n"
          "  5. db:               (num_kv_heads, total_L, d_k)         [FP16] - Gradient w.r.t raw Beta gate\n"
          "  6. da:               (num_kv_heads, total_L, 1)           [FP16] - Gradient w.r.t raw Alpha\n"
          "  7. dgamma:           (num_kv_heads, total_L, 1)           [FP16] - Gradient w.r.t raw Gamma\n"
          "  8. grad_past_states: (batch_size, num_kv_heads, d_v, d_k) [FP16] - Gradient w.r.t past_states (if provided)",
          py::arg("grad_output"), py::arg("grad_final_states") = py::none(),
          py::arg("past_states") = py::none(), py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("w"), py::arg("b"), py::arg("a"), py::arg("gamma"), py::arg("cu_seq_lens"),
          py::arg("W") = py::none(), py::arg("S") = py::none(), py::arg("num_q_heads"), py::arg("num_kv_heads"),
          py::arg("d_k") = 64, py::arg("d_v") = 64, py::arg("chunk_size") = 64,
          py::arg("has_past_states") = false);

    // 9. DeltaNet Inference (Step L = 1)
    m.def("deltanet_infer", &deltanet_infer,
          "DeltaNet Inference Coordination (Step L = 1)\n\n"
          "Outputs:\n"
          "  1. out:        (batch_size, num_q_heads, d_v)       [FP16] - Attention output\n"
          "  2. past_state: (batch_size, num_kv_heads, d_v, d_k) [FP16] - In-place updated hidden states",
          py::arg("past_state"), py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("b"), py::arg("w"), py::arg("a"), py::arg("gamma"),
          py::arg("num_q_heads"), py::arg("num_kv_heads"),
          py::arg("d_k") = 64, py::arg("d_v") = 64);
}
