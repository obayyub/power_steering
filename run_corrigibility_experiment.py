#!/usr/bin/env python3
"""
Full corrigibility experiment pipeline: train vectors and evaluate.

Compares four vector sets:
1. MELBO norm=1, power=2 (nonlinear, small perturbation regime)
2. MELBO norm=2, power=2 (nonlinear, larger perturbations)
3. Power Iteration (linear, single prompt, with Rayleigh-Ritz)
4. Multi-Prompt Power Iteration (linear, multi-prompt, with Rayleigh-Ritz)

Usage:
    python run_corrigibility_experiment.py --model Qwen/Qwen3-14B

On Lambda Cloud:
    python lambda_cloud.py run -t gpu_1x_a100 -s "python run_corrigibility_experiment.py"
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_cmd(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"ERROR: {description} failed with code {result.returncode}")
        return False
    return True


def find_latest_vector(pattern: str, vectors_dir: Path) -> Path | None:
    """Find the most recently created vector file matching pattern."""
    matches = sorted(vectors_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def main():
    parser = argparse.ArgumentParser(description="Run full corrigibility experiment")
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--num-vectors", type=int, default=12)
    parser.add_argument("--num-prompts-multi", type=int, default=32, help="Prompts for multi-PI")
    parser.add_argument("--source-layer", type=int, default=7)
    parser.add_argument("--target-layer", type=int, default=None, help="Default: num_layers - 8")
    parser.add_argument("--scales", default="-25,-10,-5,0,5,10,25")
    parser.add_argument("--max-eval-questions", type=int, default=100)
    parser.add_argument("--vectors-dir", type=Path, default=Path("vectors"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--skip-training", action="store_true", help="Skip vector training")
    parser.add_argument("--skip-eval", action="store_true", help="Skip logit evaluation")

    # MELBO hyperparameters
    parser.add_argument("--melbo-steps", type=int, default=400)

    # PI hyperparameters
    parser.add_argument("--pi-iters", type=int, default=10, help="Power iteration iterations")
    parser.add_argument("--pi-tokens", type=int, default=2, help="Number of target tokens")

    args = parser.parse_args()

    args.vectors_dir.mkdir(exist_ok=True)
    args.results_dir.mkdir(exist_ok=True)

    model_short = args.model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Corrigibility Experiment: {args.model}")
    print(f"Timestamp: {timestamp}")
    print(f"Vectors dir: {args.vectors_dir}")
    print(f"Results dir: {args.results_dir}")

    vector_files = {}

    # =========================================================================
    # STEP 1: Train vectors
    # =========================================================================
    if not args.skip_training:
        target_layer_args = ["--target-layer", str(args.target_layer)] if args.target_layer else []

        # 1a. MELBO norm=1 (small perturbation regime)
        success = run_cmd([
            sys.executable, "melbo_qwen3.py",
            "--model", args.model,
            "--num-vectors", str(args.num_vectors),
            "--num-steps", str(args.melbo_steps),
            "--normalization", "1.0",
            "--power", "2.0",
            "--source-layer", str(args.source_layer),
            "--output-dir", str(args.vectors_dir),
        ] + target_layer_args,
        "Train MELBO vectors (norm=1)")

        if not success:
            return 1
        vector_files["melbo_n1"] = find_latest_vector(f"melbo_{model_short}_*.pt", args.vectors_dir)

        # 1b. MELBO norm=2 (larger perturbations)
        success = run_cmd([
            sys.executable, "melbo_qwen3.py",
            "--model", args.model,
            "--num-vectors", str(args.num_vectors),
            "--num-steps", str(args.melbo_steps),
            "--normalization", "2.0",
            "--power", "2.0",
            "--source-layer", str(args.source_layer),
            "--output-dir", str(args.vectors_dir),
        ] + target_layer_args,
        "Train MELBO vectors (norm=2)")

        if not success:
            return 1
        vector_files["melbo_n2"] = find_latest_vector(f"melbo_{model_short}_*.pt", args.vectors_dir)

        # 1c. Power Iteration (single prompt, with Rayleigh-Ritz)
        success = run_cmd([
            sys.executable, "find_power_iteration.py",
            "--model", args.model,
            "--num-vectors", str(args.num_vectors),
            "--num-iters", str(args.pi_iters),
            "--num-tokens", str(args.pi_tokens),
            "--source-layer", str(args.source_layer),
            "--output-dir", str(args.vectors_dir),
        ] + target_layer_args,
        "Train Power Iteration vectors (with Rayleigh-Ritz)")

        if not success:
            return 1
        vector_files["pi_rr"] = find_latest_vector(f"power_iter_{model_short}_*.pt", args.vectors_dir)

        # 1d. Multi-Prompt Power Iteration (with Rayleigh-Ritz)
        success = run_cmd([
            sys.executable, "find_power_iteration_multi.py",
            "--model", args.model,
            "--num-vectors", str(args.num_vectors),
            "--num-iters", str(args.pi_iters),
            "--num-tokens", str(args.pi_tokens),
            "--num-prompts", str(args.num_prompts_multi),
            "--source-layer", str(args.source_layer),
            "--output-dir", str(args.vectors_dir),
        ] + target_layer_args,
        "Train Multi-Prompt Power Iteration vectors (with Rayleigh-Ritz)")

        if not success:
            return 1
        vector_files["multi_pi_rr"] = find_latest_vector(f"power_iter_multi_{model_short}_*.pt", args.vectors_dir)

        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        for name, path in vector_files.items():
            print(f"  {name}: {path}")
    else:
        # Find existing vectors (user must specify or we find latest)
        print("\nSkipping training, looking for existing vectors...")
        # Note: with multiple MELBO runs, this is ambiguous - user should specify paths
        vector_files["melbo_n1"] = None  # User should provide
        vector_files["melbo_n2"] = None
        vector_files["pi_rr"] = find_latest_vector(f"power_iter_{model_short}_*.pt", args.vectors_dir)
        vector_files["multi_pi_rr"] = find_latest_vector(f"power_iter_multi_{model_short}_*.pt", args.vectors_dir)

        print("Found vectors:")
        for name, path in vector_files.items():
            print(f"  {name}: {path}")

    # Check all vectors exist
    for name, path in vector_files.items():
        if path is None:
            print(f"ERROR: No {name} vectors found")
            return 1

    # =========================================================================
    # STEP 2: Logit-based evaluation
    # =========================================================================
    if not args.skip_eval:
        for name, vec_path in vector_files.items():
            success = run_cmd([
                sys.executable, "eval_steering.py",
                "--model", args.model,
                "--vectors", str(vec_path),
                "--source-layer", str(args.source_layer),
                "--scales", args.scales,
                "--max-questions", str(args.max_eval_questions),
                "--output-dir", str(args.results_dir),
            ], f"Evaluate {name} vectors (logit-based)")

            if not success:
                return 1

        print("\n" + "="*70)
        print("LOGIT EVALUATION COMPLETE")
        print("="*70)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Vectors: {args.vectors_dir}")
    print(f"Results: {args.results_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
