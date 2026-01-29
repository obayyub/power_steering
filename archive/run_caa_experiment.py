#!/usr/bin/env python3
"""
Full experiment comparing steering vector methods:
1. MELBO vectors
2. Power iteration vectors (random init, Jacobian at sv=0)
3. CAA vectors (contrastive activation addition)
4. Power iteration vectors (random init, Jacobian at sv=CAA)

All methods are compared using logit difference on corrigibility evals.
"""

import subprocess
import sys
from pathlib import Path
import argparse
import json
import torch
from datetime import datetime


def run_cmd(cmd: list[str], description: str):
    """Run a command and check for errors."""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"CMD: {' '.join(cmd)}")
    print("="*70 + "\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with code {result.returncode}")
        sys.exit(1)


def find_latest_file(pattern: str) -> Path | None:
    """Find most recently modified file matching glob pattern."""
    files = sorted(Path(".").glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def main():
    parser = argparse.ArgumentParser(description="Run CAA experiment comparing steering methods")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--source-layer", type=int, default=7)
    parser.add_argument("--target-layer", type=int, default=None)

    # Vector finding params
    parser.add_argument("--num-melbo-vectors", type=int, default=8)
    parser.add_argument("--num-melbo-steps", type=int, default=50)
    parser.add_argument("--num-pi-vectors", type=int, default=8)
    parser.add_argument("--num-pi-iters", type=int, default=15)
    parser.add_argument("--num-caa-prompts", type=int, default=50)

    # Eval params
    parser.add_argument("--max-eval-questions", type=int, default=100)
    parser.add_argument("--scales", default="-10,-5,0,5,10,20")

    # Skip flags
    parser.add_argument("--skip-melbo", action="store_true")
    parser.add_argument("--skip-power-iter", action="store_true")
    parser.add_argument("--skip-caa", action="store_true")
    parser.add_argument("--skip-power-iter-caa", action="store_true")

    # Use existing vectors
    parser.add_argument("--melbo-vectors", help="Path to existing MELBO vectors")
    parser.add_argument("--power-iter-vectors", help="Path to existing power iter vectors")
    parser.add_argument("--caa-vectors", help="Path to existing CAA vectors")
    parser.add_argument("--power-iter-caa-vectors", help="Path to existing power iter @ CAA vectors")

    args = parser.parse_args()

    # Create directories
    Path("vectors").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    model_short = args.model.split("/")[-1]

    # Track all vector files
    vector_files = {}

    # Step 1: Find MELBO vectors
    if args.melbo_vectors:
        vector_files["melbo"] = args.melbo_vectors
        print(f"\nUsing provided MELBO vectors: {args.melbo_vectors}")
    elif not args.skip_melbo:
        cmd = [
            "uv", "run", "python", "find_melbo.py",
            "--model", args.model,
            "--source-layer", str(args.source_layer),
            "--num-vectors", str(args.num_melbo_vectors),
            "--num-steps", str(args.num_melbo_steps),
        ]
        if args.target_layer:
            cmd.extend(["--target-layer", str(args.target_layer)])

        run_cmd(cmd, "Find MELBO steering vectors")

        melbo_file = find_latest_file(f"vectors/melbo_{model_short}_*.pt")
        if melbo_file:
            vector_files["melbo"] = str(melbo_file)
            print(f"Using MELBO vectors: {melbo_file}")

    # Step 2: Find power iteration vectors (at sv=0)
    if args.power_iter_vectors:
        vector_files["power_iter"] = args.power_iter_vectors
        print(f"\nUsing provided power iter vectors: {args.power_iter_vectors}")
    elif not args.skip_power_iter:
        cmd = [
            "uv", "run", "python", "find_power_iteration.py",
            "--model", args.model,
            "--source-layer", str(args.source_layer),
            "--num-vectors", str(args.num_pi_vectors),
            "--num-iters", str(args.num_pi_iters),
        ]
        if args.target_layer:
            cmd.extend(["--target-layer", str(args.target_layer)])

        run_cmd(cmd, "Find power iteration vectors (at sv=0)")

        pi_file = find_latest_file(f"vectors/power_iter_{model_short}_*.pt")
        if pi_file:
            vector_files["power_iter"] = str(pi_file)
            print(f"Using power iteration vectors: {pi_file}")

    # Step 3: Find CAA vectors
    if args.caa_vectors:
        vector_files["caa"] = args.caa_vectors
        print(f"\nUsing provided CAA vectors: {args.caa_vectors}")
    elif not args.skip_caa:
        cmd = [
            "uv", "run", "python", "find_caa.py",
            "--model", args.model,
            "--layer", str(args.source_layer),
            "--max-prompts", str(args.num_caa_prompts),
        ]

        run_cmd(cmd, "Find CAA vectors from corrigibility prompts")

        caa_file = find_latest_file(f"vectors/caa_{model_short}_layer{args.source_layer}_*.pt")
        if caa_file:
            vector_files["caa"] = str(caa_file)
            print(f"Using CAA vectors: {caa_file}")

    # Step 4: Find power iteration vectors at CAA point
    if args.power_iter_caa_vectors:
        vector_files["power_iter_caa"] = args.power_iter_caa_vectors
        print(f"\nUsing provided power iter @ CAA vectors: {args.power_iter_caa_vectors}")
    elif not args.skip_power_iter_caa and "caa" in vector_files:
        cmd = [
            "uv", "run", "python", "find_power_iteration_caa.py",
            "--model", args.model,
            "--caa-vectors", vector_files["caa"],
            "--source-layer", str(args.source_layer),
            "--num-vectors", str(args.num_pi_vectors),
            "--num-iters", str(args.num_pi_iters),
        ]
        if args.target_layer:
            cmd.extend(["--target-layer", str(args.target_layer)])

        run_cmd(cmd, "Find power iteration vectors (at sv=CAA)")

        pi_caa_file = find_latest_file(f"vectors/power_iter_caa_{model_short}_*.pt")
        if pi_caa_file:
            vector_files["power_iter_caa"] = str(pi_caa_file)
            print(f"Using power iteration @ CAA vectors: {pi_caa_file}")

    # Step 5: Evaluate all vectors using logit diff
    print(f"\n{'='*70}")
    print("STEP: Evaluate all vectors using logit difference")
    print("="*70)

    # Run evaluation
    cmd = [
        "uv", "run", "python", "eval_vectors_logit_diff.py",
        "--model", args.model,
        "--source-layer", str(args.source_layer),
        f"--scales={args.scales}",  # Use = to handle negative numbers
        "--max-questions", str(args.max_eval_questions),
    ]

    for vec_type, vec_path in vector_files.items():
        cmd.extend([f"--{vec_type.replace('_', '-')}-vectors", vec_path])

    run_cmd(cmd, "Evaluate vectors using A/B logit difference")

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print("\nVector files:")
    for vec_type, vec_path in vector_files.items():
        print(f"  {vec_type}: {vec_path}")
    print("\nResults saved to results/ directory")


if __name__ == "__main__":
    main()
