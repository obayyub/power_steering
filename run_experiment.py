#!/usr/bin/env python3
"""
Run a full steering vector experiment:
1. Find MELBO vectors (12 vectors, last 2 tokens, from corrigible-neutral-HHH)
2. Find power iteration vectors (12 vectors, last 2 tokens, from corrigible-neutral-HHH)
3. Find multi-prompt power iteration vectors (12 vectors, 32 prompts, from corrigible-neutral-HHH)
4. Evaluate all on survival-instinct (100 questions)
5. Evaluate all on corrigible-neutral-HHH (100 questions)
"""

import subprocess
import sys
from pathlib import Path
import argparse
from glob import glob


def run_cmd(cmd: list[str], description: str) -> bool:
    """Run a command and check for errors."""
    print(f"\n{'=' * 70}")
    print(f"STEP: {description}")
    print(f"CMD: {' '.join(cmd)}")
    print("=" * 70 + "\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with code {result.returncode}")
        return False
    return True


def find_latest_vectors(pattern: str) -> Path | None:
    """Find the most recent vectors file matching pattern."""
    files = sorted(glob(pattern))
    return Path(files[-1]) if files else None


def main():
    parser = argparse.ArgumentParser(description="Run full steering vector experiment")
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--source-layer", type=int, default=7)
    parser.add_argument("--target-layer", type=int, default=None)

    # Vector finding options (shared)
    parser.add_argument("--num-vectors", type=int, default=12)
    parser.add_argument("--num-tokens", type=int, default=2)

    # MELBO options
    parser.add_argument("--melbo-steps", type=int, default=400)
    parser.add_argument("--melbo-normalization", type=float, default=5.0)

    # Power iteration options
    parser.add_argument("--pi-iters", type=int, default=5)

    # Multi-prompt power iteration options
    parser.add_argument("--multi-prompts", type=int, default=32)
    parser.add_argument("--multi-batch-size", type=int, default=8)

    # Eval options
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--scales", default="-10,-5,-1,0,1,5,10")

    # Skip options
    parser.add_argument("--skip-melbo", action="store_true")
    parser.add_argument("--skip-power-iter", action="store_true")
    parser.add_argument("--skip-multi-power-iter", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")

    # Use existing vectors
    parser.add_argument("--melbo-vectors", default=None)
    parser.add_argument("--power-iter-vectors", default=None)
    parser.add_argument("--multi-power-iter-vectors", default=None)

    args = parser.parse_args()

    # Create directories
    Path("data").mkdir(exist_ok=True)
    Path("vectors").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    model_short = args.model.split("/")[-1]

    # Ensure dataset exists
    if not Path("data/corrigibility_eval.json").exists():
        if not run_cmd(["python", "download_dataset.py"], "Download dataset"):
            sys.exit(1)

    # =========================================================================
    # Step 1: MELBO vectors
    # =========================================================================
    melbo_path = args.melbo_vectors
    if not args.skip_melbo and not melbo_path:
        cmd = [
            "python", "melbo_qwen3.py",
            "--model", args.model,
            "--source-layer", str(args.source_layer),
            "--num-vectors", str(args.num_vectors),
            "--num-steps", str(args.melbo_steps),
            "--normalization", str(args.melbo_normalization),
        ]
        if args.target_layer:
            cmd.extend(["--target-layer", str(args.target_layer)])

        if not run_cmd(cmd, "Find MELBO vectors"):
            sys.exit(1)
        melbo_path = find_latest_vectors(f"vectors/melbo_{model_short}_*.pt")

    if melbo_path:
        print(f"\nMELBO vectors: {melbo_path}")

    # =========================================================================
    # Step 2: Power iteration vectors
    # =========================================================================
    pi_path = args.power_iter_vectors
    if not args.skip_power_iter and not pi_path:
        cmd = [
            "python", "find_power_iteration.py",
            "--model", args.model,
            "--source-layer", str(args.source_layer),
            "--num-vectors", str(args.num_vectors),
            "--num-iters", str(args.pi_iters),
            "--num-tokens", str(args.num_tokens),
        ]
        if args.target_layer:
            cmd.extend(["--target-layer", str(args.target_layer)])

        if not run_cmd(cmd, "Find power iteration vectors"):
            sys.exit(1)
        pi_path = find_latest_vectors(f"vectors/power_iter_{model_short}_*.pt")

    if pi_path:
        print(f"\nPower iteration vectors: {pi_path}")

    # =========================================================================
    # Step 3: Multi-prompt power iteration vectors
    # =========================================================================
    multi_pi_path = args.multi_power_iter_vectors
    if not args.skip_multi_power_iter and not multi_pi_path:
        cmd = [
            "python", "find_power_iteration_multi.py",
            "--model", args.model,
            "--source-layer", str(args.source_layer),
            "--num-vectors", str(args.num_vectors),
            "--num-iters", str(args.pi_iters),
            "--num-tokens", str(args.num_tokens),
            "--num-prompts", str(args.multi_prompts),
            "--batch-size", str(args.multi_batch_size),
        ]
        if args.target_layer:
            cmd.extend(["--target-layer", str(args.target_layer)])

        if not run_cmd(cmd, "Find multi-prompt power iteration vectors"):
            sys.exit(1)
        multi_pi_path = find_latest_vectors(f"vectors/power_iter_multi_{model_short}_*.pt")

    if multi_pi_path:
        print(f"\nMulti-prompt power iteration vectors: {multi_pi_path}")

    # =========================================================================
    # Step 4: Evaluate on both datasets
    # =========================================================================
    if not args.skip_eval:
        datasets = ["survival-instinct", "corrigible-neutral-HHH"]
        vector_files = []

        if melbo_path:
            vector_files.append(("melbo", melbo_path))
        if pi_path:
            vector_files.append(("power_iter", pi_path))
        if multi_pi_path:
            vector_files.append(("multi_power_iter", multi_pi_path))

        for vec_name, vec_path in vector_files:
            for dataset in datasets:
                cmd = [
                    "python", "eval_steering.py",
                    "--model", args.model,
                    "--vectors", str(vec_path),
                    "--source-layer", str(args.source_layer),
                    "--scales", args.scales,
                    "--max-questions", str(args.max_questions),
                    "--dataset-filter", dataset,
                ]
                if not run_cmd(cmd, f"Evaluate {vec_name} on {dataset}"):
                    print(f"Warning: Evaluation of {vec_name} on {dataset} failed")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print("\nVector files:")
    for f in sorted(glob("vectors/*.pt")):
        print(f"  {f}")

    print("\nResult files:")
    for f in sorted(glob("results/eval_*.json")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
