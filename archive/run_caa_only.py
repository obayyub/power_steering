#!/usr/bin/env python3
"""
Run CAA and power_iter_caa, then evaluate only those.
"""

import subprocess
import sys
from pathlib import Path
import argparse


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--source-layer", type=int, default=7)
    parser.add_argument("--target-layer", type=int, default=None)
    parser.add_argument("--num-pi-vectors", type=int, default=8)
    parser.add_argument("--num-pi-iters", type=int, default=15)
    parser.add_argument("--num-caa-prompts", type=int, default=50)
    parser.add_argument("--max-eval-questions", type=int, default=100)
    parser.add_argument("--scales", default="0,5,10,20")

    args = parser.parse_args()

    Path("vectors").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    model_short = args.model.split("/")[-1]
    vector_files = {}

    # Step 1: Find CAA vectors
    cmd = [
        "uv", "run", "python", "find_caa.py",
        "--model", args.model,
        "--layer", str(args.source_layer),
        "--max-prompts", str(args.num_caa_prompts),
    ]
    run_cmd(cmd, "Find CAA vectors")

    caa_file = find_latest_file(f"vectors/caa_{model_short}_layer{args.source_layer}_*.pt")
    if caa_file:
        vector_files["caa"] = str(caa_file)
        print(f"Using CAA vectors: {caa_file}")

    # Step 2: Find power iteration at CAA
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

    run_cmd(cmd, "Find power iteration vectors at CAA point")

    pi_caa_file = find_latest_file(f"vectors/power_iter_caa_{model_short}_*.pt")
    if pi_caa_file:
        vector_files["power_iter_caa"] = str(pi_caa_file)
        print(f"Using power_iter_caa vectors: {pi_caa_file}")

    # Step 3: Evaluate only CAA and power_iter_caa
    cmd = [
        "uv", "run", "python", "eval_vectors_logit_diff.py",
        "--model", args.model,
        "--source-layer", str(args.source_layer),
        f"--scales={args.scales}",
        "--max-questions", str(args.max_eval_questions),
        "--caa-vectors", vector_files["caa"],
        "--power-iter-caa-vectors", vector_files["power_iter_caa"],
    ]

    run_cmd(cmd, "Evaluate CAA and power_iter_caa vectors")

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print("\nVector files:")
    for k, v in vector_files.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
