#!/usr/bin/env python3
"""End-to-end CLI for the power_steering pipeline.

Usage:
    # Find vectors (PI-RR)
    python -m power_steering.run find-vectors --method pi \\
        --model Qwen/Qwen3-14B --output-dir vectors/

    # Find vectors (MELBO)
    python -m power_steering.run find-vectors --method melbo \\
        --model Qwen/Qwen3-14B --normalization 1.0 --output-dir vectors/

    # Evaluate vectors
    python -m power_steering.run eval \\
        --model Qwen/Qwen3-14B --vectors vectors/pi_Qwen3-14B_*.pt \\
        --scales -50,-25,-10,0,10,25,50

    # Generate steered text
    python -m power_steering.run generate \\
        --model Qwen/Qwen3-14B --vectors vectors/pi.pt --vector-idx 0 \\
        --scales -25,0,25

    # Plot results
    python -m power_steering.run plot --results results/eval_*.json
"""

import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from power_steering.utils import (
    get_layer_config, get_caa_layer, load_dataset, load_vectors,
    load_vector_metadata, save_vectors, sample_balanced, format_time,
)


def _resolve_injection_layer(args, args_layer_attr: str, model_name: str, vectors_path: str | None) -> int:
    """Pick the injection layer: explicit CLI flag > vector metadata > per-model default."""
    cli_value = getattr(args, args_layer_attr, None)
    if cli_value is not None:
        return cli_value
    if vectors_path:
        meta = load_vector_metadata(vectors_path)
        if "source_layer" in meta:
            return int(meta["source_layer"])
        if "layer" in meta:
            return int(meta["layer"])
    src, _ = get_layer_config(model_name)
    return src


def _resolve_capture_site(args, vectors_path: str | None) -> str:
    """Pick the capture site: explicit CLI flag > vector metadata > 'down_proj'."""
    cli_value = getattr(args, "capture_site", None)
    if cli_value is not None:
        return cli_value
    if vectors_path:
        meta = load_vector_metadata(vectors_path)
        if meta.get("capture_site"):
            return meta["capture_site"]
    return "down_proj"


def cmd_find_vectors(args):
    """Find steering vectors with PI-RR or MELBO."""
    t_start = time.time()
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager" if args.method == "pi" else None,
    )

    src, tgt = get_layer_config(args.model)
    source_layer = args.source_layer or src
    target_layer = args.target_layer or tgt or (len(model.model.layers) - 8)

    # Get training prompt
    if args.prompt:
        prompt = args.prompt
    else:
        data = load_dataset(args.data_path)
        prompt = data[args.category][0]["question"]

    print(f"Method: {args.method}, source={source_layer}, target={target_layer}")
    print(f"Prompt: {prompt[:80]}...")

    if args.method == "pi":
        from power_steering.find_vectors import find_pi_vectors
        vectors, sigmas = find_pi_vectors(
            model, tokenizer, prompt,
            source_layer=source_layer,
            target_layer=target_layer,
            num_vectors=args.num_vectors,
            num_iters=args.num_iters,
            seed=args.seed,
            pad=args.pad,
        )
        metadata = {"sigmas": sigmas, "source_layer": source_layer, "target_layer": target_layer,
                    "pad": args.pad}
    else:
        from power_steering.find_vectors import find_melbo_vectors, MELBOConfig
        config = MELBOConfig(
            source_layer=source_layer,
            target_layer=target_layer,
            num_steps=args.num_steps,
            normalization=args.normalization,
            power=args.power,
        )
        vectors = find_melbo_vectors(
            model, tokenizer, prompt, config, args.num_vectors, seed=args.seed,
        )
        metadata = {
            "source_layer": source_layer, "target_layer": target_layer,
            "normalization": args.normalization,
        }

    metadata["prompt"] = prompt
    metadata["category"] = args.category
    metadata["seed"] = args.seed
    metadata["capture_site"] = "down_proj"
    path = save_vectors(vectors, args.output_dir, method=args.method, model_name=args.model, metadata=metadata)
    print(f"Saved {vectors.shape[0]} vectors to {path}")
    print(f"Total time: {format_time(time.time() - t_start)}")


def cmd_find_caa(args):
    """Compute a CAA steering vector from a balanced training split."""
    from power_steering.find_vectors import find_caa_vector

    t_start = time.time()
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )

    num_layers = len(model.model.layers)
    layer = args.layer if args.layer is not None else get_caa_layer(model)
    print(f"Layers: {num_layers}, CAA layer: {layer} (60% rule unless --layer overridden)")

    data = load_dataset(args.data_path)
    if args.category not in data:
        raise SystemExit(f"Category '{args.category}' not found in {args.data_path}. "
                         f"Available: {list(data)}")
    pool = data[args.category]

    if args.exclude_test:
        test = sample_balanced(pool, args.num_test, seed=args.test_seed)
        test_qs = {q["question"] for q in test}
        pool = [q for q in pool if q["question"] not in test_qs]
        print(f"Excluded {len(test)} test questions (seed={args.test_seed}); "
              f"train pool: {len(pool)}")

    train_prompts = sample_balanced(pool, args.num_train, seed=args.train_seed)
    print(f"CAA training set: {len(train_prompts)} prompts (balanced A/B, seed={args.train_seed})")

    print(f"\nComputing CAA at layer {layer} (capture_site={args.capture_site}, direction={args.direction})...")
    caa = find_caa_vector(
        model, tokenizer, train_prompts, layer,
        capture_site=args.capture_site,
        direction=args.direction,
    )

    metadata = {
        "category": args.category,
        "layer": layer,
        "source_layer": layer,  # alias so eval/generate auto-pick this layer
        "capture_site": args.capture_site,
        "direction": args.direction,
        "num_train": len(train_prompts),
        "train_seed": args.train_seed,
        "test_seed": args.test_seed if args.exclude_test else None,
        "num_test_excluded": args.num_test if args.exclude_test else 0,
        "position": "letter_token_minus_2",
    }
    path = save_vectors(caa, args.output_dir, method="caa", model_name=args.model, metadata=metadata)
    print(f"\nSaved CAA vector to {path}")
    print(f"Total time: {format_time(time.time() - t_start)}")


def cmd_eval(args):
    """Evaluate steering vectors on corrigibility datasets."""
    from power_steering.eval import SteeringEvaluator, print_summary, save_results

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )

    source_layer = _resolve_injection_layer(args, "source_layer", args.model, args.vectors)
    capture_site = _resolve_capture_site(args, args.vectors)
    print(f"Injection layer: {source_layer}, capture_site: {capture_site}")

    scales = [float(s) for s in args.scales.split(",")]

    # Load vectors
    vecs = load_vectors(args.vectors)
    norms = vecs.norm(dim=1, keepdim=True)
    vecs = vecs / norms  # unit-normalize
    vectors = {"steering": vecs}

    # Load dataset
    all_datasets = load_dataset(args.data_path)
    if args.dataset_filter:
        all_datasets = {args.dataset_filter: all_datasets[args.dataset_filter]}

    evaluator = SteeringEvaluator(model, tokenizer, source_layer, capture_site=capture_site)
    all_results = []
    try:
        for ds_name, ds in all_datasets.items():
            print(f"\nEvaluating: {ds_name}")
            results = evaluator.evaluate_dataset(
                ds, ds_name, vectors, scales, args.max_questions,
                batch_size=args.batch_size, sample_seed=args.sample_seed,
            )
            all_results.extend(results)
            print_summary(results, ds_name)
    finally:
        evaluator.cleanup()

    save_results(all_results, args.output_dir, args.model)


def cmd_generate(args):
    """Generate steered text."""
    from power_steering.generate import SteeredGenerator

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )

    source_layer = _resolve_injection_layer(args, "source_layer", args.model, args.vectors)
    capture_site = _resolve_capture_site(args, args.vectors)
    print(f"Injection layer: {source_layer}, capture_site: {capture_site}")
    scales = [float(s) for s in args.scales.split(",")]

    # Load vector
    vecs = load_vectors(args.vectors)
    vec = vecs[args.vector_idx]
    vec = vec / vec.norm()  # unit-normalize
    print(f"Using vector {args.vector_idx} from {args.vectors}")

    # Load dataset
    data = load_dataset(args.data_path)
    prompts = []
    for ds_name in ["survival-instinct", "corrigible-neutral-HHH"]:
        if ds_name not in data:
            continue
        sampled = sample_balanced(data[ds_name], args.num_prompts, seed=args.sample_seed)
        for i, q in enumerate(sampled):
            prompts.append({
                "dataset": ds_name, "prompt_idx": i, "prompt": q["question"],
                "matching_letter": q["matching_letter"],
                "not_matching_letter": q["not_matching_letter"],
                "behavior_name": q.get("behavior_name", ds_name),
            })

    generator = SteeredGenerator(model, tokenizer, source_layer, capture_site=capture_site)
    results = []

    # Seed once at the start of the sweep so the whole run is reproducible
    # but each (scale, prompt) call advances the RNG (gets a unique sample).
    from power_steering.generate import _seed_torch
    _seed_torch(args.seed)

    for scale in scales:
        generator.set_steering(vec, scale)
        for p in prompts:
            response = generator.generate(p["prompt"], args.max_tokens, args.temperature)
            results.append({**p, "scale": scale, "response": response, "prompt": p["prompt"][:300]})
        print(f"  scale={scale:+.0f}: {len(prompts)} prompts done")

    generator.cleanup()

    # Save
    import json
    from pathlib import Path
    from datetime import datetime

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"generations_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"metadata": {"model": args.model, "scales": scales}, "results": results}, f, indent=2)
    print(f"Saved {len(results)} generations to {out_path}")


def cmd_plot(args):
    """Plot evaluation or generation results."""
    from power_steering.plot import (
        load_eval_results, violin_logit_diff, violin_per_vector, save_plot,
    )

    data = load_eval_results(args.results)
    results = data["results"]
    datasets = sorted({r["dataset"] for r in results})

    for ds in datasets:
        # Combined violin (all vectors overlaid)
        fig = violin_logit_diff(results, ds)
        if fig:
            out = args.output or f"results/{ds}_violin.png"
            save_plot(fig, out)

        # Per-vector breakout
        fig = violin_per_vector(results, ds)
        if fig:
            save_plot(fig, f"results/{ds}_violin_per_vector.png")

    # Combined across all datasets
    if len(datasets) > 1:
        fig = violin_logit_diff(results)
        if fig:
            save_plot(fig, "results/combined_violin.png")

        fig = violin_per_vector(results)
        if fig:
            save_plot(fig, "results/combined_violin_per_vector.png")


def main():
    parser = argparse.ArgumentParser(prog="power_steering", description="Steering vector pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── find-vectors ──
    p = sub.add_parser("find-vectors", help="Discover steering vectors")
    p.add_argument("--method", choices=["pi", "melbo"], required=True)
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--source-layer", type=int, default=None)
    p.add_argument("--target-layer", type=int, default=None)
    p.add_argument("--num-vectors", type=int, default=12)
    p.add_argument("--num-iters", type=int, default=15, help="PI iterations")
    p.add_argument("--pad", type=int, default=5,
                   help="PI oversampling: iterate num_vectors+pad columns, keep top num_vectors")
    p.add_argument("--num-steps", type=int, default=300, help="MELBO steps")
    p.add_argument("--normalization", type=float, default=1.0, help="MELBO sphere radius")
    p.add_argument("--power", type=float, default=2.0, help="MELBO Lp power")
    p.add_argument("--data-path", default="data/anthropic_evals.json")
    p.add_argument("--category", default="corrigible-neutral-HHH")
    p.add_argument("--prompt", default=None)
    p.add_argument("--output-dir", default="vectors")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the random init (PI: starting basis; MELBO: per-vector init)")
    p.set_defaults(func=cmd_find_vectors)

    # ── find-caa ──
    p = sub.add_parser("find-caa", help="Compute a CAA steering vector")
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--layer", type=int, default=None,
                   help="Capture/inject layer (default: round(0.6 * num_layers))")
    p.add_argument("--data-path", default="data/anthropic_evals.json")
    p.add_argument("--category", default="corrigible-neutral-HHH",
                   help="Dataset category to train CAA on")
    p.add_argument("--num-train", type=int, default=150)
    p.add_argument("--train-seed", type=int, default=123)
    p.add_argument("--exclude-test", action="store_true",
                   help="Exclude --num-test prompts (sampled with --test-seed) from the training pool")
    p.add_argument("--num-test", type=int, default=60,
                   help="Number of test prompts to hold out (only used with --exclude-test)")
    p.add_argument("--test-seed", type=int, default=42,
                   help="Seed used to identify test prompts to exclude (should match your eval --sample-seed)")
    p.add_argument("--capture-site", choices=["layer_output", "down_proj"], default="layer_output",
                   help="Where to capture the activation contrast (default: layer_output, the standard CAA recipe)")
    p.add_argument("--direction", choices=["aligned", "matching"], default="aligned",
                   help="aligned: vector = mean(aligned - not_aligned), +scale → HHH-aligned. "
                        "matching (legacy): mean(matching - not_matching), +scale → Anthropic's matching answer. "
                        "Default 'aligned' is polarity-aware and recommended for cross-eval comparison.")
    p.add_argument("--output-dir", default="vectors")
    p.set_defaults(func=cmd_find_caa)

    # ── eval ──
    p = sub.add_parser("eval", help="Evaluate vectors via logit diff")
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--vectors", required=True)
    p.add_argument("--source-layer", type=int, default=None)
    p.add_argument("--scales", default="-50,-25,-10,-5,0,5,10,25,50")
    p.add_argument("--max-questions", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16, help="Questions per forward pass")
    p.add_argument("--data-path", default="data/anthropic_evals.json")
    p.add_argument("--dataset-filter", default=None)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--sample-seed", type=int, default=42,
                   help="RNG seed for the balanced question sample")
    p.add_argument("--capture-site", choices=["layer_output", "down_proj"], default=None,
                   help="Override the capture/inject site (default: read from vector metadata)")
    p.set_defaults(func=cmd_eval)

    # ── generate ──
    p = sub.add_parser("generate", help="Generate steered text")
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--vectors", required=True)
    p.add_argument("--vector-idx", type=int, default=0)
    p.add_argument("--source-layer", type=int, default=None)
    p.add_argument("--scales", default="-25,-10,0,10,25")
    p.add_argument("--num-prompts", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--data-path", default="data/anthropic_evals.json")
    p.add_argument("--output-dir", default="results/generations")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed reset once before sampling starts")
    p.add_argument("--sample-seed", type=int, default=42,
                   help="RNG seed for the balanced question sample")
    p.add_argument("--capture-site", choices=["layer_output", "down_proj"], default=None,
                   help="Override the capture/inject site (default: read from vector metadata)")
    p.set_defaults(func=cmd_generate)

    # ── plot ──
    p = sub.add_parser("plot", help="Plot results")
    p.add_argument("--results", required=True, help="Path to eval JSON")
    p.add_argument("--output", default=None, help="Output path (auto-named if omitted)")
    p.set_defaults(func=cmd_plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
