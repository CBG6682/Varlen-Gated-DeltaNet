"""
Master Benchmark Suite: Runs all benchmarks (Inference and Training) and produces
a consolidated side-by-side performance comparison.
"""

import argparse
import os
import sys

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from tests.bench_inference import run_inference_benchmark
from tests.bench_training import run_training_benchmark
from tests.utils import print_baseline_status, print_hardware_info, print_section


def main():
    parser = argparse.ArgumentParser(description="DeltaNet v2 Comprehensive Benchmark Suite")
    parser.add_argument("--dim", type=int, default=768, help="Hidden dimension (d_model)")
    parser.add_argument("--q-heads", type=int, default=8, help="Number of Query heads")
    parser.add_argument("--kv-heads", type=int, default=8, help="Number of Key/Value heads")
    parser.add_argument("--d-k", type=int, default=64, help="Head dimension for Keys")
    parser.add_argument("--d-v", type=int, default=64, help="Head dimension for Values")
    parser.add_argument("--chunk-size", type=int, default=64, help="Chunk size C")
    parser.add_argument("--infer-batches", type=str, default="1,16,64,256,1024,2048", help="Inference batch sizes")
    parser.add_argument("--train-batch", type=int, default=4, help="Training batch size B")
    parser.add_argument("--train-seq-lens", type=str, default="512,1024,2048,4096", help="Training sequence lengths")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=30, help="Number of timed iterations")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 instead of float16")
    args = parser.parse_args()

    print_hardware_info()
    print_baseline_status()

    infer_batches = [int(b.strip()) for b in args.infer_batches.split(",") if b.strip()]
    train_seq_lens = [int(l.strip()) for l in args.train_seq_lens.split(",") if l.strip()]
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    print_section("STARTING COMPLETE DELTANET V2 BENCHMARK SUITE")

    # 1. Autoregressive Inference Benchmark
    run_inference_benchmark(
        d_model=args.dim,
        num_q_heads=args.q_heads,
        num_kv_heads=args.kv_heads,
        d_k=args.d_k,
        d_v=args.d_v,
        batch_sizes=infer_batches,
        num_warmup=args.warmup,
        num_iters=args.iters,
        dtype=dtype,
    )

    # 2. Training Pass Benchmark
    run_training_benchmark(
        d_model=args.dim,
        num_q_heads=args.q_heads,
        num_kv_heads=args.kv_heads,
        d_k=args.d_k,
        d_v=args.d_v,
        chunk_size=args.chunk_size,
        batch_size=args.train_batch,
        seq_lens=train_seq_lens,
        num_warmup=args.warmup,
        num_iters=args.iters,
        dtype=dtype,
    )

    print_section("BENCHMARK SUITE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
