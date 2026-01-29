#!/usr/bin/env python3
"""
Download and prepare Anthropic corrigibility evaluation datasets.
Rebalances A/B splits for fair evaluation.
"""

import json
import requests
from pathlib import Path
from collections import Counter


def download_eval_dataset(name: str) -> list[dict]:
    """Download Anthropic eval dataset from GitHub."""
    url = f"https://raw.githubusercontent.com/anthropics/evals/main/advanced-ai-risk/human_generated_evals/{name}.jsonl"
    response = requests.get(url)
    response.raise_for_status()
    return [json.loads(line) for line in response.text.strip().split('\n')]


def analyze_dataset(data: list[dict], name: str) -> dict:
    """Analyze A/B distribution in dataset."""
    # Count which letter is the "corrigible" (matching) answer
    corrigible_letters = []
    for item in data:
        answer = item["answer_matching_behavior"].strip()
        letter = "A" if "(A)" in answer else "B"
        corrigible_letters.append(letter)

    counts = Counter(corrigible_letters)
    print(f"\n{name}:")
    print(f"  Total questions: {len(data)}")
    print(f"  Corrigible answer is A: {counts['A']} ({100*counts['A']/len(data):.1f}%)")
    print(f"  Corrigible answer is B: {counts['B']} ({100*counts['B']/len(data):.1f}%)")

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


def prepare_for_eval(data: list[dict]) -> list[dict]:
    """Prepare dataset for evaluation with clear fields."""
    prepared = []
    for item in data:
        corrigible_answer = item["answer_matching_behavior"].strip()
        survival_answer = item["answer_not_matching_behavior"].strip()

        corrigible_letter = "A" if "(A)" in corrigible_answer else "B"
        survival_letter = "A" if "(A)" in survival_answer else "B"

        prepared.append({
            "question": item["question"],
            "corrigible_letter": corrigible_letter,
            "survival_letter": survival_letter,
            "corrigible_answer_full": corrigible_answer,
            "survival_answer_full": survival_answer,
        })

    return prepared


def main():
    datasets = ["survival-instinct", "corrigible-neutral-HHH"]

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    all_data = {}

    for name in datasets:
        print(f"\nDownloading {name}...")
        data = download_eval_dataset(name)

        # Analyze original
        analyze_dataset(data, f"{name} (original)")

        # Rebalance
        balanced = rebalance_dataset(data)

        # Prepare for eval
        prepared = prepare_for_eval(balanced)
        all_data[name] = prepared

    # Save combined dataset
    output_file = output_dir / "corrigibility_eval.json"
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"\nSaved prepared datasets to {output_file}")

    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for name, data in all_data.items():
        print(f"  {name}: {len(data)} questions")


if __name__ == "__main__":
    main()
