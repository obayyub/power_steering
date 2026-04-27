#!/usr/bin/env python3
"""
Run metric power iteration experiment: train 5 vector sets + evaluate each.

Usage:
    python run_metric_experiment.py --model Qwen/Qwen3-14B

On Lambda Cloud:
    python lambda_cloud.py run \
      -t gpu_1x_h100_pcie -r us-west-3 -k Ubuntu \
      -u pyproject.toml uv.lock find_power_iteration_metric.py eval_steering.py run_metric_experiment.py data \
      -d vectors results \
      -s "run_metric_experiment.py --model Qwen/Qwen3-14B"
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], description: str) -> bool:
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: {description} failed with code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run metric power iteration experiment")
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--source-layer", type=int, default=7)
    parser.add_argument("--target-layer", type=int, default=None)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--num-vectors", type=int, default=12)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--scales", default="-25,-10,-5,0,5,10,25")
    parser.add_argument("--max-eval-questions", type=int, default=100)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    model_short = args.model.split("/")[-1]
    vectors_dir = Path("vectors")
    results_dir = Path("results")

    # Step 1: Train all 5 metric configs
    if not args.skip_training:
        cmd = [
            sys.executable, "find_power_iteration_metric.py",
            "--model", args.model,
            "--source-layer", str(args.source_layer),
            "--num-prompts", str(args.num_prompts),
            "--num-vectors", str(args.num_vectors),
            "--num-iters", str(args.num_iters),
        ]
        if args.target_layer:
            cmd += ["--target-layer", str(args.target_layer)]

        if not run_cmd(cmd, "Train metric PI vectors (all 5 configs)"):
            return 1

    # Step 2: Find all metric_pi vector files
    vector_files = sorted(vectors_dir.glob(f"metric_pi_*_{model_short}_*.pt"))
    if not vector_files:
        print("ERROR: No metric_pi vector files found")
        return 1

    # Take the 5 most recent (from this run)
    vector_files = vector_files[-5:]
    print(f"\nFound {len(vector_files)} vector files:")
    for f in vector_files:
        print(f"  {f}")

    # Step 3: Evaluate each
    if not args.skip_eval:
        for vec_file in vector_files:
            success = run_cmd([
                sys.executable, "eval_steering.py",
                "--model", args.model,
                "--vectors", str(vec_file),
                "--source-layer", str(args.source_layer),
                "--scales", args.scales,
                "--max-questions", str(args.max_eval_questions),
                "--output-dir", str(results_dir),
            ], f"Evaluate {vec_file.name}")

            if not success:
                return 1

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Vectors: {vectors_dir}")
    print(f"Results: {results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
