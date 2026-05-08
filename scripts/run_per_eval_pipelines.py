#!/usr/bin/env python3
"""Run a sequence of pipeline configs sequentially on this machine.

Designed to be invoked on a Lambda Cloud GPU instance to fill out a
per-train-eval experiment matrix without having to ssh + launch each one
manually.

For each config, runs `uv run python -m power_steering.pipeline <config>`
with stdout going to `pipeline_<config_stem>.log` (PYTHONUNBUFFERED so the
logs stream cleanly). Captures the experiment dir from each pipeline's
"Experiment dir: ..." line. After all configs complete, prints a summary
with experiment dirs ready for the matrix builder.

Usage from inside ~/project on the GPU host:

    # Run every qwen3_14b_train_*.json config in scripts/configs/:
    python scripts/run_per_eval_pipelines.py --all

    # Or pass specific configs:
    python scripts/run_per_eval_pipelines.py \\
        scripts/configs/qwen3_14b_train_corrigible-neutral-HHH.json \\
        scripts/configs/qwen3_14b_train_survival-instinct.json
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def discover_train_configs() -> list[Path]:
    """Find all qwen3_*_train_*.json configs under scripts/configs/."""
    configs_dir = REPO_ROOT / "scripts" / "configs"
    return sorted(configs_dir.glob("qwen3_*_train_*.json"))


def run_one(config_path: Path) -> tuple[Path | None, int, float]:
    """Run pipeline on this config. Returns (experiment_dir, exit_code, duration_seconds)."""
    log_path = REPO_ROOT / f"pipeline_{config_path.stem}.log"
    print(f"\n{'='*72}")
    print(f"  RUNNING: {config_path.name}")
    print(f"  log:     {log_path.name}")
    print(f"{'='*72}", flush=True)

    # Ensure uv is on PATH even when launched from a setsid+nohup environment
    # that doesn't inherit the user's interactive PATH.
    extra_paths = [str(Path.home() / ".local" / "bin"), "/usr/local/bin", "/usr/bin"]
    new_path = ":".join([*extra_paths, os.environ.get("PATH", "")])
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PATH": new_path}
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            ["uv", "run", "python", "-u", "-m", "power_steering.pipeline", str(config_path)],
            cwd=REPO_ROOT,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
        )
    duration = time.time() - t0

    # Recover experiment dir from log
    exp_dir = None
    try:
        with open(log_path) as f:
            for line in f:
                if line.startswith("Experiment dir: "):
                    exp_dir = Path(line.split("Experiment dir: ", 1)[1].strip())
                    break
    except OSError:
        pass

    status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    print(f"  → {status}  duration={duration:.0f}s  exp={exp_dir}")
    return exp_dir, proc.returncode, duration


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="Run every qwen3_*_train_*.json config in scripts/configs/")
    ap.add_argument("configs", nargs="*", type=Path)
    ap.add_argument("--stop-on-error", action="store_true",
                    help="Abort the sequence if any pipeline fails")
    args = ap.parse_args()

    if args.all:
        configs = discover_train_configs()
    else:
        configs = list(args.configs)

    if not configs:
        ap.error("No configs to run. Pass --all or list configs explicitly.")

    print(f"Will run {len(configs)} pipeline(s) sequentially:")
    for c in configs:
        print(f"  - {c}")
    print()

    summary: list[dict] = []
    grand_t0 = time.time()
    for i, config in enumerate(configs, 1):
        if not config.is_absolute():
            config = (REPO_ROOT / config).resolve()
        if not config.exists():
            print(f"[{i}/{len(configs)}] SKIP: {config} not found")
            summary.append({"config": str(config), "exp_dir": None, "exit": -1, "duration_s": 0.0})
            continue
        print(f"\n[{i}/{len(configs)}] {config.name}")
        exp_dir, code, dur = run_one(config)
        summary.append({
            "config": str(config),
            "exp_dir": str(exp_dir) if exp_dir else None,
            "exit": code,
            "duration_s": round(dur, 1),
        })
        if code != 0 and args.stop_on_error:
            print(f"\nABORTING — pipeline returned exit {code}", flush=True)
            break

    grand_total = time.time() - grand_t0

    print(f"\n\n{'='*72}\n  SUMMARY  (grand total {grand_total/60:.1f} min)\n{'='*72}")
    for s in summary:
        cfg_name = Path(s["config"]).name
        print(f"  exit={s['exit']:>3}  {s['duration_s']:>6.1f}s   {cfg_name}")
        print(f"      → {s['exp_dir']}")

    # Write a machine-readable summary too
    out_path = REPO_ROOT / "per_eval_runs.summary.json"
    import json
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "grand_total_s": round(grand_total, 1)}, f, indent=2)
    print(f"\nWrote {out_path}")

    failed = [s for s in summary if s["exit"] != 0]
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
