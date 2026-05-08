"""Free-form steering atlas: PI per (s, t) pair on each prompt, with
KL-thresholded generation. Successor to the legacy `map_diverse.py`,
adapted for the new package conventions:

  - Uses `find_pi_vectors` from `find_vectors.py` (saved seed for the
    random orthonormal basis → fully reproducible).
  - Output organized under `Experiment` directory layout for provenance.
  - Atomic per-pair JSON writes (`.tmp` → `os.replace`) so a partial run
    is safe to inspect / pull mid-flight.
  - Both signs of steering are KL-measured per vector; generation runs
    only at the BETTER sign per vector when `max(kl_pos, kl_neg) ≥
    threshold`. This catches Phase 4's sign ambiguity without doubling
    generation cost.
  - Generation seeded per (pair × vector × sample) for reproducibility.

Output:
    experiments/<id>/
        manifest.json, config.json
        freeform/<prompt_id>/
            baseline.json            # unsteered samples for the prompt
            pairs/<s>_<t>.json       # per-pair: vectors, sigmas, kl±, generations
            merged.pt                # rolled-up dense maps (sigma/kl) + vectors

Usage:
    uv run python -m power_steering.map_freeform configs/map_freeform.json
    uv run python -m power_steering.map_freeform configs/map_freeform.json --resume experiments/<id>
    uv run python -m power_steering.map_freeform configs/map_freeform.json --resume experiments/<id> --merge-only
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()

from power_steering.experiment import Experiment
from power_steering.find_vectors import find_pi_vectors
from power_steering.generate import SteeredGenerator
from power_steering.utils import format_chat, format_time


DEFAULTS: dict = {
    "experiment_name": None,
    "model": "Qwen/Qwen3-14B",
    "prompts": [],                 # list of {"id": "...", "message": "..."}
    "scale_frac": 0.35,
    "kl_threshold": 0.5,
    "seed": 0,
    "snapshot_every": 50,
    "source_min": 0,                # inclusive
    "source_max": None,             # exclusive; None → n_layers
    "target_max": None,             # exclusive; None → n_layers
    "min_target_gap": 1,            # target ≥ source + gap
    "pi": {
        "num_vectors": 12, "num_iters": 5, "pad": 5, "num_tokens": 2,
    },
    "generation": {
        "num_samples": 3,
        "max_new_tokens": 300,
        "temperature": 0.7,
    },
}


# ── atomic JSON write ───────────────────────────────────────────────────────

def atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


# ── per-prompt baseline + norms ─────────────────────────────────────────────

def compute_baseline(
    model, tokenizer, message: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (last-token logits [1, V], input_ids [1, S]) for an unsteered prompt."""
    device = next(model.parameters()).device
    text = format_chat(tokenizer, message)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[:, -1, :]
    return logits, input_ids


def measure_source_norms(model, input_ids: torch.Tensor) -> list[float]:
    """Last-token down_proj output norm at every layer (single forward pass)."""
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


# ── KL: both signs in one batched forward ───────────────────────────────────

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
    k = vectors.shape[0]
    dtype = next(model.parameters()).dtype
    steer = torch.cat([vectors * scale, vectors * (-scale)], dim=0).to(dtype)

    dp = model.model.layers[source_layer].mlp.down_proj
    state = {"v": None}

    def hook(m, i, o):
        if state["v"] is not None:
            return o + state["v"].unsqueeze(1)
        return o

    h = dp.register_forward_hook(hook)
    try:
        state["v"] = steer
        with torch.no_grad():
            steered_logits = model(input_ids.expand(2 * k, -1)).logits[:, -1, :]
        log_p = F.log_softmax(baseline_logits.float(), dim=-1)
        log_q = F.log_softmax(steered_logits.float(), dim=-1)
        kl = (log_q.exp() * (log_q - log_p)).sum(dim=-1)
        return kl[:k].tolist(), kl[k:].tolist()
    finally:
        h.remove()


# ── Generation (KL-thresholded, best-sign per vector) ────────────────────────

def generate_for_active_vectors(
    generator: SteeredGenerator,
    message: str,
    vectors: torch.Tensor,
    scale: float,
    kl_pos: list[float],
    kl_neg: list[float],
    kl_threshold: float,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    seed_base: int,
) -> dict:
    """Generate samples for vectors where `max(kl_pos[i], kl_neg[i]) ≥ threshold`.

    Sign chosen per vector: whichever sign has higher KL. Returns a dict
    keyed by `v{i}{+|-}` with {"kl": ..., "samples": [{"si": ..., "text": ...}]}.
    """
    out: dict = {}
    k = vectors.shape[0]
    for vi in range(k):
        kp, kn = kl_pos[vi], kl_neg[vi]
        kl_max_v = max(kp, kn)
        if kl_max_v < kl_threshold:
            continue
        sign = +1 if kp >= kn else -1
        sign_label = "+" if sign > 0 else "-"
        samples: list[dict] = []
        for si in range(num_samples):
            generator.set_steering(vectors[vi], sign * scale)
            text = generator.generate(
                prompt=message,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed_base + vi * 100 + si,
            )
            samples.append({"si": si, "text": text})
        generator.clear_steering()
        out[f"v{vi}{sign_label}"] = {
            "kl": kp if sign > 0 else kn,
            "sign": sign_label,
            "samples": samples,
        }
    return out


def generate_baseline_samples(
    model, tokenizer, message: str,
    num_samples: int, max_new_tokens: int, temperature: float, seed_base: int,
) -> list[dict]:
    """Unsteered samples for the prompt — control for steered generations."""
    device = next(model.parameters()).device
    text = format_chat(tokenizer, message)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    out = []
    for si in range(num_samples):
        torch.manual_seed(seed_base + si)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_base + si)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
            )
        text_out = tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        out.append({"si": si, "text": text_out})
    return out


# ── Per-prompt merge (sigma/kl maps + vectors) ───────────────────────────────

def merge_prompt(
    prompt_dir: Path,
    n_layers: int,
    k: int,
    model_name: str,
    hidden_dim: int,
) -> Path | None:
    """Roll all completed `pairs/*.json` for a prompt into `merged.pt`."""
    pairs_dir = prompt_dir / "pairs"
    if not pairs_dir.exists():
        return None

    sigma   = torch.full((n_layers, n_layers, k), float("nan"))
    kl_pos  = torch.full((n_layers, n_layers, k), float("nan"))
    kl_neg  = torch.full((n_layers, n_layers, k), float("nan"))
    scale_m = torch.full((n_layers, n_layers), float("nan"))
    norm_m  = torch.full((n_layers,), float("nan"))
    vectors: dict[str, torch.Tensor] = {}
    n_pairs = 0
    n_pairs_with_gen = 0

    for pf in sorted(pairs_dir.glob("*.json")):
        try:
            with open(pf) as fp:
                d = json.load(fp)
        except (json.JSONDecodeError, OSError):
            continue
        s, t = d["source_layer"], d["target_layer"]
        sigma[s, t]  = torch.tensor(d["sigmas"])
        kl_pos[s, t] = torch.tensor(d["kl_pos"])
        kl_neg[s, t] = torch.tensor(d["kl_neg"])
        scale_m[s, t] = d["scale"]
        norm_m[s] = d["source_norm"]
        vectors[f"{s}_{t}"] = torch.tensor(d["vectors"], dtype=torch.float16)
        n_pairs += 1
        if d.get("generations"):
            n_pairs_with_gen += 1

    out = {
        "metadata": {
            "model": model_name,
            "prompt_id": prompt_dir.name,
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "k": k,
            "n_pairs_completed": n_pairs,
            "n_pairs_expected": n_layers * (n_layers - 1) // 2,
            "n_pairs_with_generation": n_pairs_with_gen,
        },
        "sigma": sigma,
        "kl_pos": kl_pos,
        "kl_neg": kl_neg,
        "scale": scale_m,
        "source_norms": norm_m,
        "vectors": vectors,
    }
    out_path = prompt_dir / "merged.pt"
    tmp = prompt_dir / "merged.pt.tmp"
    torch.save(out, tmp)
    os.replace(tmp, out_path)
    return out_path


# ── Driver ───────────────────────────────────────────────────────────────────

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


def run_freeform(config_path: str, resume: str | None = None, merge_only: bool = False) -> Path:
    with open(config_path) as f:
        user_cfg = json.load(f)
    cfg = _merge_cfg(DEFAULTS, user_cfg)

    if not cfg["prompts"]:
        raise ValueError("config['prompts'] must be a non-empty list of {id, message} dicts")

    if resume:
        exp = Experiment.open(resume)
        print(f"Resuming experiment: {exp.root}", flush=True)
    else:
        exp = Experiment.create(
            model=cfg["model"], config=cfg,
            name=cfg["experiment_name"],
            datasets=[],
        )
        print(f"New experiment: {exp.root}", flush=True)

    pi_cfg = cfg["pi"]
    gen_cfg = cfg["generation"]

    # Merge-only path: skip model load.
    if merge_only:
        mc = AutoConfig.from_pretrained(cfg["model"])
        n = mc.num_hidden_layers
        H = mc.hidden_size
        for p in cfg["prompts"]:
            prompt_dir = exp.root / "freeform" / p["id"]
            path = merge_prompt(prompt_dir, n, pi_cfg["num_vectors"], cfg["model"], H)
            if path:
                print(f"  merged: {path}", flush=True)
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
        attn_implementation="eager",  # PI requires
    )
    model.eval()
    print(f"Model loaded in {format_time(time.time() - t0)}", flush=True)

    n = model.config.num_hidden_layers
    H = model.config.hidden_size
    print(f"Layers: {n}, hidden_dim: {H}", flush=True)

    # Pair geometry
    s_min = cfg["source_min"]
    s_max = cfg["source_max"] or n
    t_max = cfg["target_max"] or n
    gap = cfg["min_target_gap"]
    all_pairs = [
        (s, t) for s in range(s_min, s_max)
        for t in range(s + gap, t_max)
    ]
    print(f"Pair geometry: source ∈ [{s_min}, {s_max}), target ∈ [s+{gap}, {t_max}), "
          f"{len(all_pairs)} pairs total", flush=True)

    # ── Per-prompt loop ───────────────────────────────────────────────────
    for p in cfg["prompts"]:
        prompt_id = p["id"]
        message = p["message"]
        _hr(f"Prompt: {prompt_id}")
        print(f"  message: {message[:120]}{'...' if len(message)>120 else ''}", flush=True)

        prompt_dir = exp.root / "freeform" / prompt_id
        pairs_dir = prompt_dir / "pairs"
        pairs_dir.mkdir(parents=True, exist_ok=True)

        # Baseline (unsteered) samples — generated once per prompt
        baseline_path = prompt_dir / "baseline.json"
        if not baseline_path.exists():
            print("  Generating baseline samples...", flush=True)
            baseline_samples = generate_baseline_samples(
                model, tokenizer, message,
                num_samples=gen_cfg["num_samples"],
                max_new_tokens=gen_cfg["max_new_tokens"],
                temperature=gen_cfg["temperature"],
                seed_base=cfg["seed"] * 1_000_000,
            )
            atomic_write_json(baseline_path, {
                "prompt_id": prompt_id,
                "message": message,
                "samples": baseline_samples,
            })

        # Baseline logits + per-source norms (once per prompt)
        baseline_logits, input_ids = compute_baseline(model, tokenizer, message)
        norms = measure_source_norms(model, input_ids)
        sorted_norms = sorted(norms)
        print(f"  Source norms: min={sorted_norms[0]:.1f} med={sorted_norms[n//2]:.1f} "
              f"max={sorted_norms[-1]:.1f}", flush=True)

        # Resume support
        remaining = [(s, t) for (s, t) in all_pairs
                     if not (pairs_dir / f"{s}_{t}.json").exists()]
        done_already = len(all_pairs) - len(remaining)
        print(f"  Pairs: {len(remaining)} remaining ({done_already} already on disk)",
              flush=True)

        exp.manifest.setdefault("freeform", {})[prompt_id] = {
            "message": message,
            "n_pairs_total": len(all_pairs),
            "n_pairs_done_at_start": done_already,
            "source_norms": norms,
        }
        exp._save_manifest()

        # Cache one SteeredGenerator per source layer (each registers its hook)
        generators: dict[int, SteeredGenerator] = {}

        def get_generator(s: int) -> SteeredGenerator:
            if s not in generators:
                generators[s] = SteeredGenerator(model, tokenizer, s, "down_proj")
            return generators[s]

        scale_frac = cfg["scale_frac"]
        kl_threshold = cfg["kl_threshold"]
        snapshot_every = cfg["snapshot_every"]

        t_run = time.time()
        for idx, (s, t) in enumerate(remaining):
            iter_start = time.time()

            # 1. PI
            vecs, sigmas = find_pi_vectors(
                model, tokenizer, message,
                source_layer=s, target_layer=t,
                num_vectors=pi_cfg["num_vectors"],
                num_iters=pi_cfg["num_iters"],
                num_tokens=pi_cfg["num_tokens"],
                seed=cfg["seed"] + s * 1000 + t,
                pad=pi_cfg["pad"],
            )

            # 2. Scale
            scale = scale_frac * norms[s]

            # 3. KL ± both signs (one batched forward)
            kl_pos, kl_neg = compute_kl_both_signs(
                model, s, vecs, scale, input_ids, baseline_logits,
            )

            # 4. Generation per active vector (max KL ≥ threshold)
            ev = get_generator(s)
            seed_base = cfg["seed"] + s * 100_000 + t * 1000
            generations = generate_for_active_vectors(
                ev, message, vecs, scale, kl_pos, kl_neg, kl_threshold,
                num_samples=gen_cfg["num_samples"],
                max_new_tokens=gen_cfg["max_new_tokens"],
                temperature=gen_cfg["temperature"],
                seed_base=seed_base,
            )

            # 5. Persist
            atomic_write_json(pairs_dir / f"{s}_{t}.json", {
                "source_layer": s,
                "target_layer": t,
                "scale": scale,
                "source_norm": norms[s],
                "sigmas": sigmas,
                "kl_pos": kl_pos,
                "kl_neg": kl_neg,
                "vectors": vecs.detach().cpu().to(torch.float16).tolist(),
                "generations": generations,
            })

            # Logging
            elapsed = time.time() - t_run
            iter_dt = time.time() - iter_start
            done_this_run = idx + 1
            rate = done_this_run / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - done_this_run) / rate if rate > 0 else 0
            n_active = len(generations)
            print(
                f"  ({s:>2},{t:>2}) σ₁={sigmas[0]:>6.0f} scale={scale:>5.1f} "
                f"kl_max={max(max(kl_pos), max(kl_neg)):>5.2f} "
                f"active={n_active:>2}/{pi_cfg['num_vectors']}  "
                f"[{iter_dt:.1f}s, {format_time(elapsed)} elapsed, "
                f"~{format_time(eta)} left]",
                flush=True,
            )

            # 6. Periodic snapshot
            if snapshot_every and done_this_run % snapshot_every == 0:
                merge_prompt(prompt_dir, n, pi_cfg["num_vectors"], cfg["model"], H)
                print(f"  [snapshot merged.pt at {done_this_run} pairs this run]",
                      flush=True)

        # Cleanup generators (removes hooks)
        for ev in generators.values():
            ev.cleanup()

        # Final per-prompt merge
        merged_path = merge_prompt(prompt_dir, n, pi_cfg["num_vectors"], cfg["model"], H)
        if merged_path:
            print(f"\n  Merged: {merged_path}", flush=True)
            exp.add_output("eval", merged_path, label=f"freeform_{prompt_id}")

    exp.finalize()
    print(f"\nMap freeform complete. Experiment: {exp.root}", flush=True)
    return exp.root


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("config", help="JSON config (see DEFAULTS in this module)")
    ap.add_argument("--resume", help="Path to existing experiment dir to resume into")
    ap.add_argument("--merge-only", action="store_true",
                    help="Skip the sweep; just rebuild merged.pt from per-pair JSONs")
    args = ap.parse_args()
    run_freeform(args.config, resume=args.resume, merge_only=args.merge_only)


if __name__ == "__main__":
    main()
