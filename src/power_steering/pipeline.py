"""End-to-end pipeline: find vectors → eval → plot from a single config JSON.

Usage:
    uv run python -m power_steering.pipeline config.json

Example config.json:
{
    "model": "Qwen/Qwen3-1.7B-Base",
    "method": "pi",
    "num_vectors": 4,
    "num_iters": 10,
    "scales": [-50, -25, -10, -5, 0, 5, 10, 25, 50],
    "max_questions": 60,
    "batch_size": 16,
    "category": "corrigible-neutral-HHH",
    "output_dir": "results/my_experiment"
}

All fields are optional and have sensible defaults. The pipeline:
  1. Loads the model once
  2. Finds steering vectors (PI-RR or MELBO)
  3. Evaluates all vectors across all dataset categories
  4. Saves violin plots (combined + per-vector breakout)
"""

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from power_steering.utils import (
    get_layer_config, load_dataset, save_vectors, format_time,
)

DEFAULTS = {
    "model": "Qwen/Qwen3-14B",
    "method": "pi",
    "source_layer": None,
    "target_layer": None,
    "num_vectors": 12,
    "num_iters": 15,
    "num_steps": 300,
    "normalization": 1.0,
    "power": 2.0,
    "scales": [-50, -25, -10, -5, 0, 5, 10, 25, 50],
    "max_questions": 100,
    "batch_size": 16,
    "data_path": "data/corrigibility_eval.json",
    "category": "corrigible-neutral-HHH",
    "prompt": None,
    "dataset_filter": None,
    "output_dir": "results/pipeline",
}


def run_pipeline(config_path: str):
    with open(config_path) as f:
        user_cfg = json.load(f)

    cfg = {**DEFAULTS, **user_cfg}
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the resolved config for reproducibility
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    t_total = time.time()

    # ── Load model (once) ─────────────────────────────────────────────────
    print(f"Loading {cfg['model']}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager" if cfg["method"] == "pi" else None,
    )
    print(f"Model loaded in {format_time(time.time() - t0)}")

    src, tgt = get_layer_config(cfg["model"])
    source_layer = cfg["source_layer"] or src
    target_layer = cfg["target_layer"] or tgt or (len(model.model.layers) - 8)

    # ── Find vectors ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Step 1: Find vectors ({cfg['method'].upper()})")
    print(f"{'='*60}")

    data = load_dataset(cfg["data_path"])

    if cfg["prompt"]:
        prompt = cfg["prompt"]
    else:
        prompt = data[cfg["category"]][0]["question"]
    print(f"Prompt: {prompt[:80]}...")
    print(f"Source layer: {source_layer}, Target layer: {target_layer}")

    t0 = time.time()
    if cfg["method"] == "pi":
        from power_steering.find_vectors import find_pi_vectors
        vectors, sigmas = find_pi_vectors(
            model, tokenizer, prompt,
            source_layer=source_layer,
            target_layer=target_layer,
            num_vectors=cfg["num_vectors"],
            num_iters=cfg["num_iters"],
        )
        metadata = {"sigmas": sigmas}
    else:
        from power_steering.find_vectors import find_melbo_vectors, MELBOConfig
        melbo_cfg = MELBOConfig(
            source_layer=source_layer,
            target_layer=target_layer,
            num_steps=cfg["num_steps"],
            normalization=cfg["normalization"],
            power=cfg["power"],
        )
        vectors = find_melbo_vectors(model, tokenizer, prompt, melbo_cfg, cfg["num_vectors"])
        metadata = {"normalization": cfg["normalization"]}

    metadata.update({
        "prompt": prompt, "category": cfg["category"],
        "source_layer": source_layer, "target_layer": target_layer,
    })

    vec_path = save_vectors(
        vectors, output_dir / "vectors",
        method=cfg["method"], model_name=cfg["model"], metadata=metadata,
    )
    print(f"Saved {vectors.shape[0]} vectors to {vec_path}")
    print(f"Step 1 done in {format_time(time.time() - t0)}")

    # ── Eval ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Step 2: Evaluate")
    print(f"{'='*60}")

    from power_steering.eval import SteeringEvaluator, print_summary, save_results

    # Unit-normalize for eval
    norms = vectors.norm(dim=1, keepdim=True)
    vecs_normed = vectors / norms
    vec_dict = {"steering": vecs_normed}

    scales = [float(s) for s in cfg["scales"]]

    datasets = data
    if cfg["dataset_filter"]:
        datasets = {cfg["dataset_filter"]: data[cfg["dataset_filter"]]}

    evaluator = SteeringEvaluator(model, tokenizer, source_layer)
    all_results = []
    t0 = time.time()
    try:
        for ds_name, ds in datasets.items():
            print(f"\nEvaluating: {ds_name}")
            results = evaluator.evaluate_dataset(
                ds, ds_name, vec_dict, scales,
                cfg["max_questions"], batch_size=cfg["batch_size"],
            )
            all_results.extend(results)
            print_summary(results, ds_name)
    finally:
        evaluator.cleanup()

    eval_path = save_results(all_results, output_dir, cfg["model"])
    print(f"Step 2 done in {format_time(time.time() - t0)}")

    # ── Plot ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Step 3: Plot")
    print(f"{'='*60}")

    from power_steering.plot import (
        load_eval_results, violin_logit_diff, violin_per_vector, save_plot,
    )

    eval_data = load_eval_results(str(eval_path))
    result_dicts = eval_data["results"]
    ds_names = sorted({r["dataset"] for r in result_dicts})

    for ds in ds_names:
        fig = violin_logit_diff(result_dicts, ds)
        if fig:
            save_plot(fig, output_dir / f"{ds}_violin.png")

        fig = violin_per_vector(result_dicts, ds)
        if fig:
            save_plot(fig, output_dir / f"{ds}_violin_per_vector.png")

    # Combined across all datasets
    if len(ds_names) > 1:
        fig = violin_logit_diff(result_dicts)
        if fig:
            save_plot(fig, output_dir / "combined_violin.png")

        fig = violin_per_vector(result_dicts)
        if fig:
            save_plot(fig, output_dir / "combined_violin_per_vector.png")

    print(f"\nPipeline complete in {format_time(time.time() - t_total)}")
    print(f"All outputs in: {output_dir}/")


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m power_steering.pipeline <config.json>")
        sys.exit(1)
    run_pipeline(sys.argv[1])


if __name__ == "__main__":
    main()
