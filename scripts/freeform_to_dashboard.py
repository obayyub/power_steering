"""Convert `experiments/map_freeform_*/` outputs into the dashboard's format.

Reads each `experiments/map_freeform_*/freeform/<prompt_id>/`:
  - merged.pt        (sigma, kl_pos, kl_neg, scale, source_norms, vectors)
  - baseline.json    (samples: [{si, text}])
  - pairs/<s>_<t>.json (vectors, sigmas, kl_pos, kl_neg, generations dict)

Writes dashboard-ingestible files:
  - dashboard/dashboard_data.json   — summary: per_prompt with sigmas + kls
  - dashboard/diverse_pairs/<prompt_id>/<s>_<t>.json — per-pair generations
                                       (flat list of {v, s, text} entries)

The dashboard's index.html will then show the new prompts under the Diverse
tab without modification. KL is reported as max(kl_pos, kl_neg) per vector
since the dashboard expects a single KL per vector.

Usage:
    uv run python scripts/freeform_to_dashboard.py
        # default: scans experiments/ for map_freeform_*/

    uv run python scripts/freeform_to_dashboard.py path/to/exp1 path/to/exp2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parent.parent
DASHBOARD = REPO / "dashboard"
PAIRS_OUT = DASHBOARD / "diverse_pairs"


def find_experiment_dirs() -> list[Path]:
    return sorted((REPO / "experiments").glob("map_freeform_*"))


def convert_one_prompt(prompt_dir: Path) -> tuple[dict, int, int, str, str]:
    """Return (per_prompt_entry, n_layers, n_vectors, model, prompt_id)."""
    prompt_id = prompt_dir.name
    merged = torch.load(prompt_dir / "merged.pt", map_location="cpu", weights_only=True)
    sigma = merged["sigma"]
    kl_pos = merged["kl_pos"]
    kl_neg = merged["kl_neg"]
    n = int(sigma.shape[0])
    k = int(sigma.shape[2])
    model = merged["metadata"]["model"]

    # pairs entry: dashboard expects sigmas + kls per vector
    pairs: dict[str, dict] = {}
    for s in range(n):
        for t in range(s + 1, n):
            if torch.isnan(sigma[s, t, 0]):
                continue
            kls = torch.maximum(kl_pos[s, t], kl_neg[s, t]).tolist()
            pairs[f"{s}_{t}"] = {
                "s": s, "t": t,
                "sigmas": [round(v, 2) for v in sigma[s, t].tolist()],
                "kls": [round(v, 4) for v in kls],
            }

    # baseline samples
    baseline: list[dict] = []
    bp = prompt_dir / "baseline.json"
    if bp.exists():
        with open(bp) as f:
            bl = json.load(f)
        for sample in bl.get("samples", []):
            baseline.append({"s": sample["si"], "text": sample["text"]})

    # Per-pair generations — flatten dict keyed by "v0+"/"v3-" into [{v, s, text}]
    pairs_out_dir = PAIRS_OUT / prompt_id
    pairs_out_dir.mkdir(parents=True, exist_ok=True)
    for pf in sorted((prompt_dir / "pairs").glob("*.json")):
        with open(pf) as f:
            d = json.load(f)
        flat: list[dict] = []
        for vec_label, gen in d.get("generations", {}).items():
            # "v3+" → vec_idx=3, sign='+'
            vi = int(vec_label[1:-1])
            sign = vec_label[-1]
            for sample in gen.get("samples", []):
                flat.append({
                    "v": vi,
                    "s": sample["si"],
                    "text": sample["text"],
                    "sign": sign,
                    "kl": gen.get("kl"),
                })
        with open(pairs_out_dir / pf.name, "w") as fout:
            json.dump({"generations": flat}, fout)

    entry = {"pairs": pairs, "baseline": baseline}
    return entry, n, k, model, prompt_id


def get_prompt_text(prompt_id: str) -> str:
    """Look up the prompt text from a matching config in scripts/configs/."""
    cfg_dir = REPO / "scripts" / "configs"
    for cfg_path in cfg_dir.glob("map_freeform_*.json"):
        with open(cfg_path) as f:
            cfg = json.load(f)
        for p in cfg.get("prompts", []):
            if p["id"] == prompt_id:
                return p["message"]
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="*", type=Path,
                    help="Specific experiment dirs (default: all map_freeform_* under experiments/)")
    ap.add_argument("--keep-existing", action="store_true",
                    help="Merge into existing dashboard_data.json instead of overwriting prompts.")
    args = ap.parse_args()

    exp_dirs = list(args.experiments) if args.experiments else find_experiment_dirs()
    if not exp_dirs:
        print("No map_freeform_* experiment dirs found.", file=sys.stderr)
        return 1

    PAIRS_OUT.mkdir(parents=True, exist_ok=True)
    DASHBOARD.mkdir(parents=True, exist_ok=True)

    per_prompt: dict[str, dict] = {}
    prompts_meta: list[dict] = []
    n_layers = n_vectors = None
    model = None

    for exp_dir in exp_dirs:
        ff = exp_dir / "freeform"
        if not ff.exists():
            print(f"  skipping {exp_dir} — no freeform/ subdir")
            continue
        for prompt_dir in sorted(ff.iterdir()):
            if not prompt_dir.is_dir():
                continue
            entry, n, k, m, pid = convert_one_prompt(prompt_dir)
            per_prompt[pid] = entry
            n_layers = n_layers or n
            n_vectors = n_vectors or k
            model = model or m
            prompts_meta.append({"id": pid, "text": get_prompt_text(pid)})
            print(f"  [{pid}] {len(entry['pairs'])} pairs, "
                  f"{len(entry['baseline'])} baseline samples")

    out: dict = {
        "model": model,
        "num_layers": n_layers,
        "num_vectors": n_vectors,
        "prompt_ids": [pm["id"] for pm in prompts_meta],
        "prompts": prompts_meta,
        "per_prompt": per_prompt,
    }

    existing_path = DASHBOARD / "dashboard_data.json"
    if args.keep_existing and existing_path.exists() and existing_path.stat().st_size > 0:
        with open(existing_path) as f:
            existing = json.load(f)
        # Merge our prompts into existing per_prompt, keep other top-level fields
        merged = dict(existing)
        merged.setdefault("per_prompt", {})
        merged["per_prompt"].update(per_prompt)
        # Update prompts list (dedupe by id)
        existing_pids = {p["id"] for p in merged.get("prompts", [])}
        for pm in prompts_meta:
            if pm["id"] not in existing_pids:
                merged.setdefault("prompts", []).append(pm)
        # Keep prompt_ids in sync with the merged prompts list
        merged["prompt_ids"] = [p["id"] for p in merged.get("prompts", [])]
        # Prefer our model/num_layers if existing was empty
        merged["model"] = merged.get("model") or model
        merged["num_layers"] = merged.get("num_layers") or n_layers
        merged["num_vectors"] = merged.get("num_vectors") or n_vectors
        out = merged

    with open(existing_path, "w") as f:
        json.dump(out, f)

    print(f"\nWrote {existing_path} ({existing_path.stat().st_size:,} bytes)")
    print(f"Per-pair files under {PAIRS_OUT}/")
    print()
    print("To view, serve the dashboard dir over HTTP (browsers block local fetch):")
    print(f"  cd {DASHBOARD}")
    print(f"  python -m http.server 8000")
    print("  → http://localhost:8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
