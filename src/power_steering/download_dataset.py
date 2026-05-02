#!/usr/bin/env python3
"""
Download and prepare Anthropic advanced-ai-risk human-generated evaluation datasets.

Pulls each dataset from anthropics/evals on GitHub, rebalances A/B splits, and
saves the combined set to data/anthropic_evals.json under our schema
(matching_letter, not_matching_letter, matching_answer_full,
not_matching_answer_full, behavior_name).

Default datasets cover a range of safety-relevant behaviors:
    survival-instinct, corrigible-neutral-HHH, power-seeking-inclination,
    wealth-seeking-inclination, self-awareness-general-ai,
    coordinate-other-ais, myopic-reward.

Usage:
    uv run python -m power_steering.download_dataset
    uv run python -m power_steering.download_dataset --datasets survival-instinct,corrigible-neutral-HHH
"""

import argparse
import json
import requests
from pathlib import Path
from collections import Counter


DEFAULT_DATASETS = [
    "survival-instinct",
    "corrigible-neutral-HHH",
    "power-seeking-inclination",
    "wealth-seeking-inclination",
    "self-awareness-general-ai",
    "coordinate-other-ais",
    "myopic-reward",
]
DEFAULT_OUTPUT_FILE = "data/anthropic_evals.json"


def download_eval_dataset(name: str) -> list[dict]:
    """Download Anthropic eval dataset from GitHub."""
    url = f"https://raw.githubusercontent.com/anthropics/evals/main/advanced-ai-risk/human_generated_evals/{name}.jsonl"
    response = requests.get(url)
    response.raise_for_status()
    return [json.loads(line) for line in response.text.strip().split('\n')]


def analyze_dataset(data: list[dict], name: str) -> dict:
    """Analyze A/B distribution in dataset."""
    # Count which letter exhibits the named behavior (Anthropic's "matching" answer)
    matching_letters = []
    for item in data:
        answer = item["answer_matching_behavior"].strip()
        letter = "A" if "(A)" in answer else "B"
        matching_letters.append(letter)

    counts = Counter(matching_letters)
    print(f"\n{name}:")
    print(f"  Total questions: {len(data)}")
    print(f"  Matching answer is A: {counts['A']} ({100*counts['A']/len(data):.1f}%)")
    print(f"  Matching answer is B: {counts['B']} ({100*counts['B']/len(data):.1f}%)")

    return {"total": len(data), "A": counts["A"], "B": counts["B"]}


def rebalance_dataset(data: list[dict]) -> list[dict]:
    """Rebalance dataset to have equal A/B corrigible answers."""
    a_items = []
    b_items = []

    for item in data:
        answer = item["answer_matching_behavior"].strip()
        letter = "A" if "(A)" in answer else "B"
        if letter == "A":
            a_items.append(item)
        else:
            b_items.append(item)

    # Take minimum of both
    min_count = min(len(a_items), len(b_items))
    balanced = a_items[:min_count] + b_items[:min_count]

    print(f"  Rebalanced to {len(balanced)} questions ({min_count} A, {min_count} B)")
    return balanced


def prepare_for_eval(data: list[dict], behavior_name: str) -> list[dict]:
    """Prepare dataset for evaluation using Anthropic's matching/not-matching convention.

    matching_letter follows Anthropic's `answer_matching_behavior` field. For
    the advanced-ai-risk human-generated evals this is the HHH-aligned answer
    (accepting shutdown on survival-instinct, deferring to humans on
    corrigible-neutral-HHH). The dataset name describes what is being tested,
    not which answer is labelled matching.
    """
    prepared = []
    for item in data:
        matching_answer = item["answer_matching_behavior"].strip()
        not_matching_answer = item["answer_not_matching_behavior"].strip()

        matching_letter = "A" if "(A)" in matching_answer else "B"
        not_matching_letter = "A" if "(A)" in not_matching_answer else "B"

        prepared.append({
            "question": item["question"],
            "matching_letter": matching_letter,
            "not_matching_letter": not_matching_letter,
            "matching_answer_full": matching_answer,
            "not_matching_answer_full": not_matching_answer,
            "behavior_name": behavior_name,
        })

    return prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", default=",".join(DEFAULT_DATASETS),
        help="Comma-separated list of Anthropic advanced-ai-risk human_generated_evals datasets",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE,
                        help=f"Output JSON path (default: {DEFAULT_OUTPUT_FILE})")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_data = {}

    for name in datasets:
        print(f"\nDownloading {name}...")
        data = download_eval_dataset(name)

        # Analyze original
        analyze_dataset(data, f"{name} (original)")

        # Rebalance
        balanced = rebalance_dataset(data)

        # Prepare for eval
        prepared = prepare_for_eval(balanced, behavior_name=name)
        all_data[name] = prepared

    # Save combined dataset
    with open(output_path, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"\nSaved prepared datasets to {output_path}")

    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for name, data in all_data.items():
        a = sum(1 for q in data if q["matching_letter"] == "A")
        b = len(data) - a
        print(f"  {name}: {len(data)} questions  (A={a}, B={b})")


if __name__ == "__main__":
    main()
