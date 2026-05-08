"""End-to-end multi-method experiment runner.

Drives PI, MELBO, and CAA into a single experiment directory:

    experiments/<id>/
        manifest.json
        config.json
        vectors/
        eval/
        plots/

Usage:
    uv run python -m power_steering.pipeline configs/my_run.json

Example config (all keys optional except `model`):
{
    "model": "Qwen/Qwen3-1.7B-Base",
    "experiment_name": null,                  # auto = <UTC>_<model-short>
    "methods": ["pi", "melbo", "caa"],

    "source_layer": null,                     # for PI/MELBO; null = per-model default
    "target_layer": null,
    "scales": [-50, -25, -10, -5, 0, 5, 10, 25, 50],
    "max_questions": 60,
    "batch_size": 16,
    "data_path": "data/anthropic_evals.json",
    "category": "corrigible-neutral-HHH",     # which dataset to train on
    "prompt": null,                           # null = first question of `category`
    "dataset_filter": null,                   # null = eval on all datasets in the file

    "seed": 0,                                # PI starting basis / MELBO inits
    "sample_seed": 42,                        # eval question sampling

    "pi":    {"num_vectors": 12, "num_iters": 15},
    "melbo": {"num_vectors": 12, "num_steps": 300, "normalization": 1.0, "power": 2.0},
    "caa":   {"layer": null, "num_train": 150, "train_seed": 123,
              "exclude_test": true, "num_test": 60}
}
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

# Keep pipeline.log readable: HF's tqdm progress bars (model download / shard
# loading) emit \r overwrites that look like garbage when stdout is redirected.
hf_logging.disable_progress_bar()

from power_steering.experiment import Experiment
from power_steering.utils import (
    get_layer_config, get_caa_layer, load_dataset, save_vectors,
    sample_balanced, format_time,
)


DEFAULTS = {
    "model": "Qwen/Qwen3-14B",
    "experiment_name": None,
    "methods": ["pi", "melbo", "caa"],
    "source_layer": None,
    "target_layer": None,
    "scales": [-50, -25, -10, -5, 0, 5, 10, 25, 50],
    "max_questions": 100,
    "batch_size": 16,
    "data_path": "data/anthropic_evals.json",
    "category": "corrigible-neutral-HHH",
    "prompt": None,
    "dataset_filter": None,
    "seed": 0,
    "sample_seed": 42,
    "pi": {"num_vectors": 12, "num_iters": 15, "pad": 5},
    "melbo": {
        "num_vectors": 12, "num_steps": 300,
        "normalization": 1.0, "power": 2.0,
    },
    "caa": {
        "layer": None, "num_train": 150, "train_seed": 123,
        "exclude_test": True, "num_test": 60,
        "direction": "aligned",  # "aligned" (polarity-aware) or "matching" (legacy)
    },
    "dct": {
        "num_features": 12, "num_iters": 10,
        "lambda_cal": 0.5, "n_cal": 30,
    },
}


def _merge(defaults: dict, override: dict) -> dict:
    """Shallow merge with one level of nested-dict merging for per-method blocks."""
    out = dict(defaults)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def _hr(label: str) -> None:
    print(f"\n{'='*64}\n  {label}\n{'='*64}")


def _free_gpu(model=None) -> None:
    """Drop autograd state and ask CUDA to release cached blocks.

    Called between PI / MELBO / CAA / eval phases so a tight model (e.g. 27B
    on an 80GB H100) has the maximum free memory entering the next phase.
    Doesn't shrink resident weights — those stay in place.
    """
    import gc
    if model is not None:
        try:
            model.zero_grad(set_to_none=True)
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"  [cuda] free {free / 1e9:.1f} GB / {total / 1e9:.1f} GB")


def run_pipeline(config_path: str) -> Path:
    with open(config_path) as f:
        user_cfg = json.load(f)
    cfg = _merge(DEFAULTS, user_cfg)

    # ── Set up experiment ─────────────────────────────────────────────────
    exp = Experiment.create(
        model=cfg["model"],
        config=cfg,
        name=cfg["experiment_name"],
        datasets=[cfg["data_path"]],
    )
    print(f"Experiment dir: {exp.root}")

    t_total = time.time()

    # ── Load model ────────────────────────────────────────────────────────
    _hr("Load model")
    print(f"Loading {cfg['model']}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    methods = cfg["methods"]
    needs_eager = "pi" in methods or "dct" in methods
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager" if needs_eager else None,
    )
    print(f"Model loaded in {format_time(time.time() - t0)}")

    src_default, tgt_default = get_layer_config(cfg["model"])
    source_layer = cfg["source_layer"] or src_default
    target_layer = cfg["target_layer"] or tgt_default or (len(model.model.layers) - 8)
    caa_layer = cfg["caa"]["layer"] if cfg["caa"]["layer"] is not None else get_caa_layer(model)

    # Persist resolved layers into the manifest
    exp.manifest["resolved_layers"] = {
        "source_layer": source_layer,
        "target_layer": target_layer,
        "caa_layer": caa_layer,
    }
    exp._save_manifest()

    # ── Load data + pick training prompt ─────────────────────────────────
    data = load_dataset(cfg["data_path"])
    if cfg["category"] not in data:
        raise SystemExit(f"category '{cfg['category']}' not in {cfg['data_path']}. "
                         f"Available: {list(data)}")
    if cfg["prompt"]:
        train_prompt = cfg["prompt"]
        prompt_idx = None
    else:
        # Seeded random pick — different `seed` selects a different prompt.
        # `sample_balanced(n=1)` would return empty (n//2=0), so use random.choice.
        import random as _random
        rng = _random.Random(cfg["seed"])
        category_items = data[cfg["category"]]
        prompt_idx = rng.randrange(len(category_items))
        train_prompt = category_items[prompt_idx]["question"]
    print(f"Training prompt: {train_prompt[:80]}...")
    exp.manifest["training_prompt"] = {
        "category": cfg["category"],
        "index_in_category": prompt_idx,
        "seed_used": cfg["seed"] if cfg["prompt"] is None else None,
        "text": train_prompt,
    }
    exp._save_manifest()
    print(f"Source layer={source_layer}, Target layer={target_layer}, CAA layer={caa_layer}")

    # Vectors keyed by (injection layer, capture_site) for the eval phase.
    # PI/MELBO use ("down_proj"); CAA uses ("layer_output") because the
    # contrastive direction at the residual stream is the behaviorally
    # meaningful one (it encodes everything that diverged through layer L,
    # not just layer-L MLP's incremental contribution).
    vectors_by_site: dict[tuple[int, str], dict[str, torch.Tensor]] = {}

    # ── PI ────────────────────────────────────────────────────────────────
    if "pi" in methods:
        _hr("Find vectors — PI-RR")
        from power_steering.find_vectors import find_pi_vectors
        t0 = time.time()
        pi_vecs, sigmas = find_pi_vectors(
            model, tokenizer, train_prompt,
            source_layer=source_layer, target_layer=target_layer,
            num_vectors=cfg["pi"]["num_vectors"],
            num_iters=cfg["pi"]["num_iters"],
            seed=cfg["seed"],
            pad=cfg["pi"].get("pad", 5),
        )
        meta = {
            "method": "pi", "prompt": train_prompt, "category": cfg["category"],
            "source_layer": source_layer, "target_layer": target_layer,
            "capture_site": "down_proj",
            "seed": cfg["seed"], "sigmas": sigmas,
            "pad": cfg["pi"].get("pad", 5),
        }
        path = save_vectors(pi_vecs, exp.vectors_dir,
                            method="pi", model_name=cfg["model"], metadata=meta)
        exp.add_output("vectors", path, label="pi", metadata=meta)
        vectors_by_site.setdefault((source_layer, "down_proj"), {})["pi"] = (
            pi_vecs / pi_vecs.norm(dim=1, keepdim=True)
        )
        print(f"PI done in {format_time(time.time() - t0)}")
        _free_gpu(model)

    # ── MELBO ─────────────────────────────────────────────────────────────
    if "melbo" in methods:
        _hr("Find vectors — MELBO")
        from power_steering.find_vectors import find_melbo_vectors, MELBOConfig
        t0 = time.time()
        melbo_cfg = MELBOConfig(
            source_layer=source_layer, target_layer=target_layer,
            num_steps=cfg["melbo"]["num_steps"],
            normalization=cfg["melbo"]["normalization"],
            power=cfg["melbo"]["power"],
        )
        # PI warm-start: only when explicitly enabled AND PI ran first this run.
        # Default unchanged (random init) — keeps existing configs reproducible.
        melbo_init = None
        if cfg["melbo"].get("init_from_pi") and "pi" in methods:
            melbo_init = pi_vecs
            print(f"  MELBO warm-start: using {pi_vecs.shape[0]} PI vectors as init")
        melbo_vecs = find_melbo_vectors(
            model, tokenizer, train_prompt, melbo_cfg,
            cfg["melbo"]["num_vectors"], seed=cfg["seed"],
            init_vectors=melbo_init,
        )
        meta = {
            "method": "melbo", "prompt": train_prompt, "category": cfg["category"],
            "source_layer": source_layer, "target_layer": target_layer,
            "capture_site": "down_proj",
            "seed": cfg["seed"], "normalization": cfg["melbo"]["normalization"],
            "power": cfg["melbo"]["power"], "num_steps": cfg["melbo"]["num_steps"],
            "init_from_pi": bool(melbo_init is not None),
        }
        path = save_vectors(melbo_vecs, exp.vectors_dir,
                            method="melbo", model_name=cfg["model"], metadata=meta)
        exp.add_output("vectors", path, label="melbo", metadata=meta)
        vectors_by_site.setdefault((source_layer, "down_proj"), {})["melbo"] = (
            melbo_vecs / melbo_vecs.norm(dim=1, keepdim=True)
        )
        print(f"MELBO done in {format_time(time.time() - t0)}")
        _free_gpu(model)

    # ── DCT (exponential, OGI) ───────────────────────────────────────────
    if "dct" in methods:
        _hr("Find vectors — DCT (exponential, OGI)")
        from power_steering.find_dct import find_dct_vectors, DCTConfig
        t0 = time.time()
        dct_cfg = DCTConfig(
            source_layer=source_layer, target_layer=target_layer,
            num_features=cfg["dct"]["num_features"],
            num_iters=cfg["dct"]["num_iters"],
            lambda_cal=cfg["dct"]["lambda_cal"],
            n_cal=cfg["dct"]["n_cal"],
        )
        dct_vecs, dct_info = find_dct_vectors(
            model, tokenizer, train_prompt, dct_cfg,
            num_features=cfg["dct"]["num_features"], seed=cfg["seed"],
        )
        meta = {
            "method": "dct", "prompt": train_prompt, "category": cfg["category"],
            "source_layer": source_layer, "target_layer": target_layer,
            "capture_site": "down_proj",
            "seed": cfg["seed"],
            "lambda_cal": cfg["dct"]["lambda_cal"],
            "n_cal": cfg["dct"]["n_cal"],
            "num_iters": cfg["dct"]["num_iters"],
            "R_cal": dct_info.get("R_cal"),
            "final_loss": dct_info.get("final_loss"),
        }
        path = save_vectors(dct_vecs, exp.vectors_dir,
                            method="dct", model_name=cfg["model"], metadata=meta)
        exp.add_output("vectors", path, label="dct", metadata=meta)
        vectors_by_site.setdefault((source_layer, "down_proj"), {})["dct"] = (
            dct_vecs / dct_vecs.norm(dim=1, keepdim=True)
        )
        print(f"DCT done in {format_time(time.time() - t0)}")
        _free_gpu(model)

    # ── CAA ───────────────────────────────────────────────────────────────
    if "caa" in methods:
        _hr("Find vector — CAA")
        from power_steering.find_vectors import find_caa_vector
        t0 = time.time()
        pool = data[cfg["category"]]
        caa_cfg = cfg["caa"]
        if caa_cfg["exclude_test"]:
            test = sample_balanced(pool, caa_cfg["num_test"], seed=cfg["sample_seed"])
            test_qs = {q["question"] for q in test}
            pool = [q for q in pool if q["question"] not in test_qs]
            print(f"Excluded {len(test)} test questions; train pool: {len(pool)}")
        train_prompts = sample_balanced(pool, caa_cfg["num_train"], seed=caa_cfg["train_seed"])
        print(f"CAA training set: {len(train_prompts)} prompts (seed={caa_cfg['train_seed']})")

        caa_vec = find_caa_vector(
            model, tokenizer, train_prompts, caa_layer,
            capture_site="layer_output",
            direction=caa_cfg.get("direction", "aligned"),
        )
        meta = {
            "method": "caa", "category": cfg["category"],
            "layer": caa_layer, "source_layer": caa_layer,
            "capture_site": "layer_output",
            "direction": caa_cfg.get("direction", "aligned"),
            "num_train": len(train_prompts),
            "train_seed": caa_cfg["train_seed"],
            "test_seed": cfg["sample_seed"] if caa_cfg["exclude_test"] else None,
            "num_test_excluded": caa_cfg["num_test"] if caa_cfg["exclude_test"] else 0,
            "position": "letter_token_minus_2",
        }
        path = save_vectors(caa_vec, exp.vectors_dir,
                            method="caa", model_name=cfg["model"], metadata=meta)
        exp.add_output("vectors", path, label="caa", metadata=meta)
        vectors_by_site.setdefault((caa_layer, "layer_output"), {})["caa"] = (
            caa_vec / caa_vec.norm(dim=1, keepdim=True)
        )
        print(f"CAA done in {format_time(time.time() - t0)}")
        _free_gpu(model)

    # ── Eval (one pass per injection layer) ──────────────────────────────
    _hr("Evaluate")
    from power_steering.eval import SteeringEvaluator, print_summary, save_results

    model.eval()
    _free_gpu(model)

    scales = [float(s) for s in cfg["scales"]]
    datasets = data
    if cfg["dataset_filter"]:
        datasets = {cfg["dataset_filter"]: data[cfg["dataset_filter"]]}

    all_results = []
    t0 = time.time()
    for (layer, site), vec_dict in vectors_by_site.items():
        print(f"\n--- Eval pass at layer {layer} site={site}: {list(vec_dict)} ---")
        evaluator = SteeringEvaluator(model, tokenizer, layer, capture_site=site)
        try:
            for ds_name, ds in datasets.items():
                results = evaluator.evaluate_dataset(
                    ds, ds_name, vec_dict, scales,
                    cfg["max_questions"], batch_size=cfg["batch_size"],
                    sample_seed=cfg["sample_seed"],
                )
                all_results.extend(results)
                print_summary(results, ds_name)
        finally:
            evaluator.cleanup()

    eval_path = save_results(all_results, exp.eval_dir, cfg["model"])
    exp.add_output("eval", eval_path, metadata={
        "scales": scales,
        "datasets": list(datasets),
        "max_questions": cfg["max_questions"],
        "sample_seed": cfg["sample_seed"],
        "vectors_by_site": {f"{k[0]}|{k[1]}": list(v) for k, v in vectors_by_site.items()},
    })
    print(f"Eval done in {format_time(time.time() - t0)}")

    # Free the model before plotting (matplotlib + torch can fight over RAM)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Plot ──────────────────────────────────────────────────────────────
    _hr("Plot")
    from power_steering.plot import (
        load_eval_results, violin_logit_diff, violin_per_vector, save_plot,
    )

    eval_data = load_eval_results(str(eval_path))
    result_dicts = eval_data["results"]
    ds_names = sorted({r["dataset"] for r in result_dicts})

    def _plot_meta(ds_name: str | None, plot_type: str) -> dict:
        scope = [r for r in result_dicts if (ds_name is None or r["dataset"] == ds_name)]
        return {
            "plot_type": plot_type,
            "dataset": ds_name or "all",
            "metric": "matching_logit_diff",
            "source_eval": str(eval_path.relative_to(exp.root)),
            "scales": sorted({r["scale"] for r in scope}),
            "vectors_included": sorted({f"{r['vector_type']}_v{r['vector_idx']}" for r in scope}),
            "n_records": len(scope),
            "model": cfg["model"],
            "experiment_id": exp.manifest["experiment_id"],
        }

    methods_present = sorted({r["vector_type"] for r in result_dicts})

    def _filter(records, ds_name=None, method=None):
        out = records
        if ds_name is not None:
            out = [r for r in out if r["dataset"] == ds_name]
        if method is not None:
            out = [r for r in out if r["vector_type"] == method]
        return out

    for ds in ds_names:
        # All-methods overlay (one violin per (scale, vector) tuple, hued by vector)
        fig = violin_logit_diff(result_dicts, ds)
        if fig:
            p = save_plot(fig, exp.plots_dir / f"{ds}_violin.png",
                          metadata=_plot_meta(ds, "violin_logit_diff"))
            exp.add_output("plots", p, label=f"{ds}_violin")

        # Per-method subplot grids: one figure per method, one subplot per vector.
        # Keeps PI's 12 / MELBO's 12 / CAA's 1 readable instead of cramming
        # 25 subplots into one figure.
        for method in methods_present:
            scoped = _filter(result_dicts, ds_name=ds, method=method)
            if not scoped:
                continue
            fig = violin_per_vector(scoped, ds, title=f"{ds} — {method}")
            if fig:
                p = save_plot(
                    fig, exp.plots_dir / f"{ds}_{method}_violin_per_vector.png",
                    metadata={**_plot_meta(ds, "violin_per_vector"), "method": method,
                              "vectors_included": sorted({f"{r['vector_type']}_v{r['vector_idx']}"
                                                          for r in scoped})},
                )
                exp.add_output("plots", p, label=f"{ds}_{method}_violin_per_vector")

    if len(ds_names) > 1:
        fig = violin_logit_diff(result_dicts)
        if fig:
            p = save_plot(fig, exp.plots_dir / "combined_violin.png",
                          metadata=_plot_meta(None, "violin_logit_diff"))
            exp.add_output("plots", p, label="combined_violin")

        for method in methods_present:
            scoped = _filter(result_dicts, method=method)
            if not scoped:
                continue
            fig = violin_per_vector(scoped, title=f"all datasets — {method}")
            if fig:
                p = save_plot(
                    fig, exp.plots_dir / f"combined_{method}_violin_per_vector.png",
                    metadata={**_plot_meta(None, "violin_per_vector"), "method": method,
                              "vectors_included": sorted({f"{r['vector_type']}_v{r['vector_idx']}"
                                                          for r in scoped})},
                )
                exp.add_output("plots", p, label=f"combined_{method}_violin_per_vector")

    manifest_path = exp.finalize()
    print(f"\nPipeline complete in {format_time(time.time() - t_total)}")
    print(f"Experiment manifest: {manifest_path}")
    return exp.root


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m power_steering.pipeline <config.json>")
        sys.exit(1)
    run_pipeline(sys.argv[1])


if __name__ == "__main__":
    main()
