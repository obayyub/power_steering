"""Fill in missing-sign generations for dual-active vectors in a map_freeform atlas.

The original `map_freeform.py` only generates at the better-KL sign per
vector. For vectors where BOTH ±scale signs cleared `kl_threshold`, the
other sign's behavior is uncovered. This script regenerates samples at
the missing sign using the saved vectors (so no PI re-training) and
writes back into the same per-pair JSONs.

For each pair JSON:
  - Load saved fp16 vectors → torch.float32 on device.
  - For each (vector_idx, sign) where:
        * `max(kl_pos[vi], kl_neg[vi]) ≥ threshold`
        * `min(kl_pos[vi], kl_neg[vi]) ≥ threshold`
        * the sign-corresponding key (e.g. "v3-") is absent from `generations`
    → generate `num_samples` samples at that sign, seed scheme matches
    the original (`seed + s*100_000 + t*1000 + vi*100 + si`).
  - Write back atomically.

Usage:
    uv run python scripts/fill_missing_signs.py \\
        experiments/map_freeform_roleplay_Qwen3-14B \\
        roleplay_lighthouse \\
        --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()

from power_steering.generate import SteeredGenerator
from power_steering.utils import format_time


def atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def find_missing(pair_data: dict, threshold: float) -> list[tuple[int, str]]:
    """Return [(vec_idx, missing_sign), ...] for dual-active vectors lacking one sign."""
    kl_pos = pair_data["kl_pos"]
    kl_neg = pair_data["kl_neg"]
    gens = pair_data.get("generations", {})
    missing: list[tuple[int, str]] = []
    for vi in range(len(kl_pos)):
        kp, kn = kl_pos[vi], kl_neg[vi]
        if min(kp, kn) < threshold:
            continue  # only one sign active or none — nothing to fill
        # Which signs are already covered?
        has_pos = f"v{vi}+" in gens
        has_neg = f"v{vi}-" in gens
        if has_pos and not has_neg:
            missing.append((vi, "-"))
        elif has_neg and not has_pos:
            missing.append((vi, "+"))
    return missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir", type=Path)
    ap.add_argument("prompt_id")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed base; matches map_freeform.py's cfg['seed']. Per-sample seed = seed + s*100_000 + t*1000 + vi*100 + si")
    args = ap.parse_args()

    pairs_dir = args.experiment_dir / "freeform" / args.prompt_id / "pairs"
    if not pairs_dir.exists():
        print(f"ERROR: {pairs_dir} not found", file=sys.stderr)
        return 1

    # Load message from manifest
    manifest_path = args.experiment_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    prompt_message = manifest["freeform"][args.prompt_id]["message"]
    model_name = manifest["model"]
    print(f"Model: {model_name}")
    print(f"Prompt: {prompt_message[:120]}{'...' if len(prompt_message)>120 else ''}")
    print(f"Threshold: {args.threshold}, samples: {args.num_samples}")

    # First pass: count how many fills we need
    print("\nScanning per-pair JSONs for missing signs...")
    fill_plan: dict[Path, list[tuple[int, str, float]]] = {}
    total_fills = 0
    for pf in sorted(pairs_dir.glob("*.json")):
        with open(pf) as f:
            d = json.load(f)
        missing = find_missing(d, args.threshold)
        if missing:
            # Annotate with the actual KL of that sign
            kp, kn = d["kl_pos"], d["kl_neg"]
            annotated = [
                (vi, sign, kp[vi] if sign == "+" else kn[vi])
                for vi, sign in missing
            ]
            fill_plan[pf] = annotated
            total_fills += len(annotated)
    print(f"  → {len(fill_plan)} pairs need filling, {total_fills} (vec, sign) generations to do")
    if total_fills == 0:
        print("Nothing to do.")
        return 0
    print(f"  Est. time: {total_fills * args.num_samples * 8 / 60:.0f} min "
          f"(~8 sec/sample on H100)")

    # Load model
    print(f"\nLoading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    print(f"  Loaded in {format_time(time.time() - t0)}")

    # Cache one SteeredGenerator per source layer
    generators: dict[int, SteeredGenerator] = {}

    def get_generator(s: int) -> SteeredGenerator:
        if s not in generators:
            generators[s] = SteeredGenerator(model, tokenizer, s, "down_proj")
        return generators[s]

    # Per-pair fill loop
    t_start = time.time()
    n_done = 0
    for pf, missing in fill_plan.items():
        with open(pf) as f:
            d = json.load(f)
        s = d["source_layer"]
        t = d["target_layer"]
        scale = d["scale"]
        # Reload vectors as float32 on device
        vec_list = d["vectors"]
        vectors = torch.tensor(vec_list, dtype=torch.float32).to(model.device)
        # Re-normalize (paranoia: fp16 round-trip might have introduced drift)
        vectors = vectors / vectors.norm(dim=1, keepdim=True)

        ev = get_generator(s)
        seed_base = args.seed + s * 100_000 + t * 1000

        for (vi, sign, kl_value) in missing:
            sign_mult = +1.0 if sign == "+" else -1.0
            samples: list[dict] = []
            for si in range(args.num_samples):
                ev.set_steering(vectors[vi], sign_mult * scale)
                text = ev.generate(
                    prompt=prompt_message,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    seed=seed_base + vi * 100 + si,
                )
                samples.append({"si": si, "text": text})
            ev.clear_steering()
            d.setdefault("generations", {})[f"v{vi}{sign}"] = {
                "kl": kl_value,
                "sign": sign,
                "samples": samples,
            }
            n_done += 1
            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (total_fills - n_done) / rate if rate > 0 else 0
            print(f"  ({s:>2},{t:>2}) v{vi}{sign}  kl={kl_value:.2f}  "
                  f"[{n_done}/{total_fills}, {format_time(elapsed)} elapsed, "
                  f"~{format_time(eta)} left]", flush=True)

        atomic_write_json(pf, d)

    # Cleanup
    for ev in generators.values():
        ev.cleanup()

    total = time.time() - t_start
    print(f"\nDone — {n_done} (vec, sign) generations in {format_time(total)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
