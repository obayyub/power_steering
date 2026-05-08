"""Layer-pair PI atlas with norm-scaled KL and logit-diff per (source, target).

Sweeps the full upper triangle of (source, target) layer pairs on a single
training prompt per category. For each pair:

  1. PI-RR (`find_pi_vectors`) → top-k unit-norm vectors + sigmas.
  2. Scale = scale_frac × source-layer down_proj output norm at last token.
  3. KL(steered ‖ baseline) per vector at +scale and -scale on the training
     prompt's last-token output distribution. Both signs because PI vectors
     have no privileged sign (Phase 4 sign-ambiguity finding).
  4. Mean matching_logit_diff per vector at +scale and -scale on a tiny
     balanced sample (default 16 questions) of the training category.

Output (Experiment-folder layout):

    experiments/<id>/
        manifest.json, config.json
        map/<category>/
            pairs/<s>_<t>.json    # per-pair: vectors, sigmas, kl±, ld±, scale
            merged.pt             # rolled up to dense maps + vectors dict

Per-pair JSONs are written atomically so an inspect script can read them
mid-run. A snapshot merged.pt is written every `snapshot_every` completed
pairs so the dense maps are also available mid-run.

Usage:
    uv run python -m power_steering.map_layers configs/map.json
    uv run python -m power_steering.map_layers configs/map.json --resume experiments/<id>
    uv run python -m power_steering.map_layers configs/map.json --resume experiments/<id> --merge-only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()

from power_steering.eval import (
    SteeringEvaluator, compute_matching_logit_diff,
)
from power_steering.experiment import Experiment
from power_steering.find_vectors import find_pi_vectors
from power_steering.utils import (
    format_chat, format_time, load_dataset, sample_balanced,
)


DEFAULTS: dict = {
    "experiment_name": None,
    "model": "Qwen/Qwen3-14B",
    "data_path": "data/anthropic_evals.json",
    "categories": ["corrigible-neutral-HHH"],
    "scale_frac": 0.35,
    "max_questions": 16,
    "batch_size": 16,
    "sample_seed": 42,
    "seed": 0,
    "snapshot_every": 50,
    "pi": {
        "num_vectors": 12, "num_iters": 5, "pad": 5, "num_tokens": 2,
    },
}


# ── Atomic JSON write ───────────────────────────────────────────────────────

def atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


# ── Norm + baseline measurement ─────────────────────────────────────────────

def measure_source_norms(model, input_ids: torch.Tensor) -> list[float]:
    """Last-token down_proj output norm (fp32) at every layer.

    Single forward pass with hooks on every layer's mlp.down_proj. Mirrors
    `map_diverse.py.measure_norms`. Steering is added at down_proj output,
    so this is the right scale reference.
    """
    n = model.config.num_hidden_layers
    norms = [0.0] * n
    hooks = []
    for i in range(n):
        dp = model.model.layers[i].mlp.down_proj

        def make_hook(idx):
            def hook(m, inp, out):
                norms[idx] = out[0, -1, :].float().norm().item()
            return hook

        hooks.append(dp.register_forward_hook(make_hook(i)))
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()
    return norms


def compute_baseline_logits(
    model, tokenizer, prompt: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (last-token logits [1, V], input_ids [1, S]) for an unsteered prompt."""
    device = next(model.parameters()).device
    text = format_chat(tokenizer, prompt)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[:, -1, :]
    return logits, input_ids


# ── Per-pair KL (both signs in one batched forward) ─────────────────────────

def compute_kl_both_signs(
    model,
    source_layer: int,
    vectors: torch.Tensor,         # [k, H], unit-normed, on device
    scale: float,
    input_ids: torch.Tensor,        # [1, S]
    baseline_logits: torch.Tensor,  # [1, V]
) -> tuple[list[float], list[float]]:
    """KL(steered ‖ baseline) per vector at +scale and -scale.

    One forward pass with batch=2k: rows 0..k-1 are +scale, k..2k-1 are -scale.
    """
    k, H = vectors.shape
    dtype = next(model.parameters()).dtype
    steer = torch.cat([vectors * scale, vectors * (-scale)], dim=0).to(dtype)

    dp = model.model.layers[source_layer].mlp.down_proj
    state = {"v": None}

    def hook(m, i, o):
        if state["v"] is not None:
            return o + state["v"].unsqueeze(1)  # [2k, 1, H] broadcasts to [2k, S, H]
        return o

    h = dp.register_forward_hook(hook)
    try:
        state["v"] = steer
        with torch.no_grad():
            steered_logits = model(input_ids.expand(2 * k, -1)).logits[:, -1, :]
        log_p = F.log_softmax(baseline_logits.float(), dim=-1)  # [1, V]
        log_q = F.log_softmax(steered_logits.float(), dim=-1)   # [2k, V]
        kl = (log_q.exp() * (log_q - log_p)).sum(dim=-1)        # [2k]
        return kl[:k].tolist(), kl[k:].tolist()
    finally:
        h.remove()


# ── Per-pair logit-diff eval ────────────────────────────────────────────────

def eval_pair_logit_diff(
    evaluator: SteeringEvaluator,
    vectors: torch.Tensor,
    scale: float,
    eval_items: list[dict],
    eval_questions: list[str],
) -> tuple[list[float], list[float]]:
    """Mean matching_logit_diff per vector at +scale and -scale.

    Iterates k vectors × 2 signs sequentially through `evaluator.evaluate_batch`.
    Cheap relative to PI: each call is one forward of batch=len(eval_questions).
    """
    k = vectors.shape[0]
    ld_pos = [0.0] * k
    ld_neg = [0.0] * k
    n_q = len(eval_items)
    try:
        for vi in range(k):
            for sign, sink in ((+1.0, ld_pos), (-1.0, ld_neg)):
                evaluator.set_steering(vectors[vi], sign * scale)
                logits = evaluator.evaluate_batch(eval_questions)
                total = 0.0
                for qi, item in enumerate(eval_items):
                    lA = logits[qi, 0].item()
                    lB = logits[qi, 1].item()
                    total += compute_matching_logit_diff(lA, lB, item["matching_letter"])
                sink[vi] = total / n_q
    finally:
        evaluator.set_steering(None)
    return ld_pos, ld_neg


# ── Merge per-pair JSONs into a dense maps tensor ───────────────────────────

def merge_category(
    category_dir: Path,
    n_layers: int,
    k: int,
    model_name: str,
    hidden_dim: int,
) -> Path | None:
    """Roll all completed `pairs/*.json` into `merged.pt`. Idempotent."""
    pairs_dir = category_dir / "pairs"
    if not pairs_dir.exists():
        return None

    sigma   = torch.full((n_layers, n_layers, k), float("nan"))
    kl_pos  = torch.full((n_layers, n_layers, k), float("nan"))
    kl_neg  = torch.full((n_layers, n_layers, k), float("nan"))
    ld_pos  = torch.full((n_layers, n_layers, k), float("nan"))
    ld_neg  = torch.full((n_layers, n_layers, k), float("nan"))
    scale_m = torch.full((n_layers, n_layers), float("nan"))
    norm_m  = torch.full((n_layers,), float("nan"))
    vectors: dict[str, torch.Tensor] = {}
    n_pairs = 0

    for pf in sorted(pairs_dir.glob("*.json")):
        try:
            with open(pf) as fp:
                d = json.load(fp)
        except (json.JSONDecodeError, OSError):
            continue  # mid-write or corrupt; skip
        s, t = d["source_layer"], d["target_layer"]
        sigma[s, t]  = torch.tensor(d["sigmas"])
        kl_pos[s, t] = torch.tensor(d["kl_pos"])
        kl_neg[s, t] = torch.tensor(d["kl_neg"])
        ld_pos[s, t] = torch.tensor(d["ld_pos_per_vec"])
        ld_neg[s, t] = torch.tensor(d["ld_neg_per_vec"])
        scale_m[s, t] = d["scale"]
        norm_m[s] = d["source_norm"]
        vectors[f"{s}_{t}"] = torch.tensor(d["vectors"], dtype=torch.float16)
        n_pairs += 1

    out = {
        "metadata": {
            "model": model_name,
            "category": category_dir.name,
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "k": k,
            "n_pairs_completed": n_pairs,
            "n_pairs_expected": n_layers * (n_layers - 1) // 2,
        },
        "sigma": sigma,
        "kl_pos": kl_pos,
        "kl_neg": kl_neg,
        "ld_pos": ld_pos,
        "ld_neg": ld_neg,
        "scale": scale_m,
        "source_norms": norm_m,
        "vectors": vectors,
    }
    out_path = category_dir / "merged.pt"
    tmp = category_dir / "merged.pt.tmp"
    torch.save(out, tmp)
    os.replace(tmp, out_path)
    return out_path


# ── Driver ──────────────────────────────────────────────────────────────────

def _hr(label: str) -> None:
    print(f"\n{'='*64}\n  {label}\n{'='*64}", flush=True)


def _merge_cfg(defaults: dict, override: dict) -> dict:
    out = dict(defaults)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def run_map(config_path: str, resume: str | None = None, merge_only: bool = False) -> Path:
    with open(config_path) as f:
        user_cfg = json.load(f)
    cfg = _merge_cfg(DEFAULTS, user_cfg)

    if resume:
        exp = Experiment.open(resume)
        print(f"Resuming experiment: {exp.root}", flush=True)
    else:
        exp = Experiment.create(
            model=cfg["model"], config=cfg,
            name=cfg["experiment_name"],
            datasets=[cfg["data_path"]],
        )
        print(f"New experiment: {exp.root}", flush=True)

    data = load_dataset(cfg["data_path"])
    pi_cfg = cfg["pi"]

    # Merge-only path: don't load the model.
    if merge_only:
        mc = AutoConfig.from_pretrained(cfg["model"])
        n = mc.num_hidden_layers
        H = mc.hidden_size
        for cat in cfg["categories"]:
            cat_dir = exp.root / "map" / cat
            path = merge_category(cat_dir, n, pi_cfg["num_vectors"], cfg["model"], H)
            if path:
                print(f"  merged: {path}", flush=True)
            else:
                print(f"  no pairs/ for {cat}; skipping", flush=True)
        return exp.root

    # Load model
    _hr("Load model")
    print(f"Loading {cfg['model']}...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",  # PI requires eager autograd
    )
    model.eval()
    print(f"Model loaded in {format_time(time.time() - t0)}", flush=True)

    n = model.config.num_hidden_layers
    H = model.config.hidden_size
    print(f"Layers: {n}, hidden_dim: {H}", flush=True)

    # ── Per category ──────────────────────────────────────────────────────
    for category in cfg["categories"]:
        if category not in data:
            print(f"WARNING: category '{category}' not in data; skipping", flush=True)
            continue

        _hr(f"Map: {category}")

        # Pick training prompt by seed (matches pipeline.py convention)
        rng = random.Random(cfg["seed"])
        items = data[category]
        prompt_idx = rng.randrange(len(items))
        train_prompt = items[prompt_idx]["question"]
        print(f"Training prompt [idx={prompt_idx}]: {train_prompt[:80]}...", flush=True)

        # Eval sample — fixed across all pairs so heatmap cells are comparable
        eval_items = sample_balanced(items, cfg["max_questions"], seed=cfg["sample_seed"])
        eval_questions = [it["question"] for it in eval_items]
        print(f"Eval sample: {len(eval_items)} balanced (sample_seed={cfg['sample_seed']})",
              flush=True)

        # Baseline logits + per-source-layer norms
        baseline_logits, input_ids = compute_baseline_logits(model, tokenizer, train_prompt)
        norms = measure_source_norms(model, input_ids)
        sorted_norms = sorted(norms)
        print(f"Source norms (down_proj last token): "
              f"min={sorted_norms[0]:.1f} med={sorted_norms[n // 2]:.1f} "
              f"max={sorted_norms[-1]:.1f}", flush=True)

        # Output dir
        cat_dir = exp.root / "map" / category
        pairs_dir = cat_dir / "pairs"
        pairs_dir.mkdir(parents=True, exist_ok=True)

        # Pair list (skip existing for resume)
        all_pairs = [(s, t) for s in range(n) for t in range(s + 1, n)]
        remaining = [(s, t) for (s, t) in all_pairs
                     if not (pairs_dir / f"{s}_{t}.json").exists()]
        done_already = len(all_pairs) - len(remaining)
        print(f"Pairs: {len(remaining)} remaining "
              f"({done_already} already on disk of {len(all_pairs)})", flush=True)

        # Stamp manifest with per-category info
        exp.manifest.setdefault("map", {})[category] = {
            "training_prompt_idx": prompt_idx,
            "training_prompt": train_prompt,
            "eval_sample_size": len(eval_items),
            "eval_sample_seed": cfg["sample_seed"],
            "source_norms": norms,
            "n_pairs_total": len(all_pairs),
            "n_pairs_done_at_start": done_already,
        }
        exp._save_manifest()

        # Cache one SteeringEvaluator per source layer (each registers a forward hook).
        evaluators: dict[int, SteeringEvaluator] = {}

        def get_evaluator(s):
            if s not in evaluators:
                evaluators[s] = SteeringEvaluator(model, tokenizer, s, "down_proj")
            return evaluators[s]

        scale_frac = cfg["scale_frac"]
        snapshot_every = cfg["snapshot_every"]

        t_run = time.time()
        for idx, (s, t) in enumerate(remaining):
            iter_start = time.time()

            # 1. PI
            vecs, sigmas = find_pi_vectors(
                model, tokenizer, train_prompt,
                source_layer=s, target_layer=t,
                num_vectors=pi_cfg["num_vectors"],
                num_iters=pi_cfg["num_iters"],
                num_tokens=pi_cfg["num_tokens"],
                seed=cfg["seed"] + s * 1000 + t,
                pad=pi_cfg["pad"],
            )

            # 2. Scale = scale_frac × down_proj norm at source layer
            scale = scale_frac * norms[s]

            # 3. KL ± both signs (one batched forward, batch=2k)
            kl_pos, kl_neg = compute_kl_both_signs(
                model, s, vecs, scale, input_ids, baseline_logits,
            )

            # 4. Mean logit-diff ± both signs
            ev = get_evaluator(s)
            ld_pos, ld_neg = eval_pair_logit_diff(
                ev, vecs, scale, eval_items, eval_questions,
            )

            # 5. Persist (atomic). Vectors saved as fp16-tolist for compactness.
            pair_data = {
                "source_layer": s,
                "target_layer": t,
                "scale": scale,
                "source_norm": norms[s],
                "sigmas": sigmas,
                "kl_pos": kl_pos,
                "kl_neg": kl_neg,
                "ld_pos_per_vec": ld_pos,
                "ld_neg_per_vec": ld_neg,
                "vectors": vecs.detach().cpu().to(torch.float16).tolist(),
            }
            atomic_write_json(pairs_dir / f"{s}_{t}.json", pair_data)

            # Logging
            elapsed = time.time() - t_run
            iter_dt = time.time() - iter_start
            done_this_run = idx + 1
            rate = done_this_run / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - done_this_run) / rate if rate > 0 else 0
            best_kl = max(max(kl_pos), max(kl_neg))
            best_ld_abs = max(max(abs(x) for x in ld_pos), max(abs(x) for x in ld_neg))
            print(
                f"  ({s:>2},{t:>2}) σ₁={sigmas[0]:>6.0f} scale={scale:>5.1f} "
                f"kl_max={best_kl:>5.2f} |ld|_max={best_ld_abs:>5.2f}  "
                f"[{iter_dt:.1f}s, {format_time(elapsed)} elapsed, "
                f"~{format_time(eta)} left]",
                flush=True,
            )

            # 6. Periodic snapshot
            if snapshot_every and done_this_run % snapshot_every == 0:
                merge_category(cat_dir, n, pi_cfg["num_vectors"], cfg["model"], H)
                print(f"  [snapshot merged.pt at {done_this_run} pairs this run]",
                      flush=True)

        # Cleanup evaluators (removes hooks)
        for ev in evaluators.values():
            ev.cleanup()

        # Final per-category merge
        merged_path = merge_category(cat_dir, n, pi_cfg["num_vectors"], cfg["model"], H)
        if merged_path:
            print(f"\nMerged: {merged_path}", flush=True)
            exp.add_output("eval", merged_path, label=f"map_{category}")

    exp.finalize()
    print(f"\nMap complete. Experiment: {exp.root}", flush=True)
    return exp.root


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("config", help="JSON config (see DEFAULTS in this module)")
    ap.add_argument("--resume", help="Path to existing experiment dir to resume into")
    ap.add_argument("--merge-only", action="store_true",
                    help="Skip the sweep; just rebuild merged.pt from per-pair JSONs")
    args = ap.parse_args()
    run_map(args.config, resume=args.resume, merge_only=args.merge_only)


if __name__ == "__main__":
    main()
