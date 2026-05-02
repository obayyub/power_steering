#!/usr/bin/env python3
"""No-GPU structural test for the power_steering setup.

Verifies:
  1. All package modules import.
  2. download_dataset/lambda_cloud are reachable as power_steering submodules.
  3. The renamed dataset schema is intact.
  4. Experiment.create produces the expected folder layout + a valid manifest.
  5. Plot sidecar metadata writes cleanly.

Usage:
    uv run python scripts/test_setup.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def _check(name: str, condition: bool, detail: str = "") -> None:
    tag = "OK " if condition else "FAIL"
    line = f"  [{tag}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not condition:
        raise SystemExit(1)


def test_imports() -> None:
    print("\n[1] Module imports")
    from power_steering import (
        find_pi_vectors, find_melbo_vectors, find_caa_vector,
        SteeringEvaluator, SteeredGenerator,
        load_vectors, load_vector_metadata, load_dataset, format_chat, get_caa_layer,
    )
    from power_steering import experiment, pipeline, plot, run, utils, eval as ev, generate as gn
    from power_steering import download_dataset, lambda_cloud
    _check("core symbols", all([
        callable(find_pi_vectors), callable(find_melbo_vectors), callable(find_caa_vector),
        callable(load_vectors), callable(load_vector_metadata), callable(get_caa_layer),
    ]))
    _check("submodules", all([
        hasattr(experiment, "Experiment"),
        hasattr(pipeline, "run_pipeline"),
        hasattr(plot, "save_plot"),
        hasattr(run, "main"),
        hasattr(ev, "compute_matching_logit_diff"),
        hasattr(gn, "classify_results"),
        hasattr(download_dataset, "prepare_for_eval"),
        hasattr(lambda_cloud, "BASE_URL"),
    ]))


def test_schema() -> None:
    print("\n[2] Dataset schema")
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "anthropic_evals.json"
    _check("data file exists", data_path.exists(), str(data_path))
    with open(data_path) as f:
        data = json.load(f)
    expected_datasets = {
        "survival-instinct", "corrigible-neutral-HHH",
        "power-seeking-inclination", "wealth-seeking-inclination",
        "self-awareness-general-ai", "coordinate-other-ais", "myopic-reward",
    }
    _check("expected datasets", set(data) == expected_datasets,
           f"got {sorted(data)}")
    sample = data["corrigible-neutral-HHH"][0]
    expected = {"question", "matching_letter", "not_matching_letter",
                "matching_answer_full", "not_matching_answer_full", "behavior_name"}
    missing = expected - set(sample)
    _check("matching/not-matching field names", not missing,
           f"missing: {missing}" if missing else "all present")
    _check("behavior_name populated", sample["behavior_name"] == "corrigible-neutral-HHH",
           sample["behavior_name"])


def test_balanced_sampling() -> None:
    print("\n[3] sample_balanced reproducibility")
    from power_steering.utils import load_dataset, sample_balanced
    project_root = Path(__file__).resolve().parent.parent
    data = load_dataset(project_root / "data" / "anthropic_evals.json")
    s1 = sample_balanced(data["corrigible-neutral-HHH"], 20, seed=42)
    s2 = sample_balanced(data["corrigible-neutral-HHH"], 20, seed=42)
    s3 = sample_balanced(data["corrigible-neutral-HHH"], 20, seed=43)
    _check("same seed -> same sample", [q["question"] for q in s1] == [q["question"] for q in s2])
    _check("different seed -> different sample", [q["question"] for q in s1] != [q["question"] for q in s3])
    a = sum(1 for q in s1 if q["matching_letter"] == "A")
    _check("balanced (10/10)", a == 10, f"A={a}, B={20-a}")


def test_experiment_layout() -> None:
    print("\n[4] Experiment folder + manifest")
    from power_steering.experiment import Experiment

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        exp = Experiment.create(
            model="Qwen/Qwen3-0.6B",
            config={"foo": "bar", "scales": [-1, 0, 1]},
            base_dir=td,
            datasets=[Path(__file__).resolve()],  # any existing file for sha256
        )
        _check("vectors/ created", exp.vectors_dir.is_dir())
        _check("eval/ created", exp.eval_dir.is_dir())
        _check("plots/ created", exp.plots_dir.is_dir())
        _check("config.json written", (exp.root / "config.json").exists())

        fake_vec = exp.vectors_dir / "fake.pt"
        fake_vec.write_bytes(b"")
        exp.add_output("vectors", fake_vec, label="fake", metadata={"method": "fake"})

        manifest_path = exp.finalize()
        with open(manifest_path) as f:
            m = json.load(f)
        _check("manifest.experiment_id", isinstance(m.get("experiment_id"), str), m.get("experiment_id"))
        _check("manifest.git captured", "commit" in m.get("git", {}),
               m.get("git", {}).get("commit", "n/a")[:12])
        _check("manifest.env captured", "python" in m.get("env", {}), m.get("env", {}).get("python"))
        _check("manifest.datasets sha256", bool(m["datasets"][0].get("sha256")),
               m["datasets"][0].get("sha256", "")[:12])
        _check("manifest.outputs.vectors recorded",
               len(m["outputs"]["vectors"]) == 1 and m["outputs"]["vectors"][0]["label"] == "fake")
        _check("manifest.duration set", isinstance(m.get("duration_seconds"), (int, float)))


def test_plot_sidecar() -> None:
    print("\n[5] Plot sidecar metadata")
    from power_steering.plot import save_plot

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        png = td / "test.png"
        meta = {"plot_type": "line", "dataset": "fake", "scales": [-1, 0, 1]}
        save_plot(fig, png, metadata=meta)
        _check("PNG exists", png.exists())
        sidecar = png.with_suffix(".json")
        _check("sidecar JSON exists", sidecar.exists())
        with open(sidecar) as f:
            payload = json.load(f)
        _check("sidecar carries metadata", payload.get("plot_type") == "line"
               and payload.get("plot_file") == "test.png" and "saved_at" in payload)


def main() -> int:
    test_imports()
    test_schema()
    test_balanced_sampling()
    test_experiment_layout()
    test_plot_sidecar()
    print("\nAll structural tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
