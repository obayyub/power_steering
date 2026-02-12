#!/usr/bin/env python3
"""Re-score saved generations with fixed extract_answer (truncates at \\nQ:)."""

import json
from pathlib import Path
from eval_generalize import extract_answer


def rescore_file(path, label):
    with open(path) as f:
        data = json.load(f)

    print(f"\n{'='*70}")
    print(f"  {label}: {path.name}")
    print(f"{'='*70}")

    # Handle probing format (experiments -> task -> generations -> condition -> list)
    if "experiments" in data:
        for exp_name, exp_data in data["experiments"].items():
            if "generations" not in exp_data:
                continue
            print(f"\n  {exp_name}:")
            for cond, gens in exp_data["generations"].items():
                old_correct = sum(1 for g in gens if g["correct"])
                # Re-extract
                for g in gens:
                    new_pred = extract_answer(g["text"])
                    g["pred"] = new_pred
                    g["correct"] = new_pred == g["answer"]
                new_correct = sum(1 for g in gens if g["correct"])
                total = len(gens)
                old_acc = old_correct / total if total else 0
                new_acc = new_correct / total if total else 0
                delta = new_correct - old_correct
                if delta != 0:
                    print(f"    {cond:>20s}: {old_correct}/{total} ({old_acc:.1%}) -> "
                          f"{new_correct}/{total} ({new_acc:.1%})  [{delta:+d}]")
                # Update summary
                if exp_name in data.get("experiments", {}):
                    if cond in exp_data.get("results", {}):
                        exp_data["results"][cond]["accuracy"] = new_acc
                        exp_data["results"][cond]["correct"] = new_correct

    # Handle generalization format (tasks -> task -> generations -> condition -> list)
    elif "tasks" in data:
        for task_name, task_data in data["tasks"].items():
            if "generations" not in task_data:
                continue
            print(f"\n  {task_name}:")
            for cond, gens in task_data["generations"].items():
                old_correct = sum(1 for g in gens if g["correct"])
                for g in gens:
                    new_pred = extract_answer(g["text"])
                    g["pred"] = new_pred
                    g["correct"] = new_pred == g["answer"]
                new_correct = sum(1 for g in gens if g["correct"])
                total = len(gens)
                old_acc = old_correct / total if total else 0
                new_acc = new_correct / total if total else 0
                delta = new_correct - old_correct
                if delta != 0:
                    print(f"    {cond:>20s}: {old_correct}/{total} ({old_acc:.1%}) -> "
                          f"{new_correct}/{total} ({new_acc:.1%})  [{delta:+d}]")
                if cond in task_data.get("results", {}):
                    task_data["results"][cond]["accuracy"] = new_acc
                    task_data["results"][cond]["correct"] = new_correct

    # Save rescored version
    out_path = path.parent / f"{path.stem}_rescored.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Saved: {out_path.name}")


if __name__ == "__main__":
    # Re-score all result files that have generations
    files = [
        (Path("results/probing/probe_20260212_020024.json"), "Probing"),
    ]
    # Add generalization files
    for p in sorted(Path("results/generalization").glob("generalize_*.json")):
        if "rescored" not in p.name:
            files.append((p, "Generalization"))

    for path, label in files:
        if path.exists():
            rescore_file(path, label)
