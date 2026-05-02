"""Experiment-as-folder layout, manifest builder, and reproducibility capture.

A single experiment occupies one directory:

    experiments/<experiment_id>/
        manifest.json       # full reproduction record
        config.json         # the input config that produced this run
        vectors/            # *.pt with embedded metadata (model, method, seeds, layer, ...)
        eval/               # *.json eval results
        generations/        # *.json generation results
        plots/              # *.png + same-name *.json sidecar metadata

`experiment_id` defaults to `<UTC-timestamp>_<model-short>` and can be overridden.
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_sha256(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_git_state(repo_root: str | Path) -> dict:
    """Record commit SHA and dirty status. Returns {} if not a git repo."""
    repo_root = str(repo_root)
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    try:
        status = subprocess.check_output(
            ["git", "-C", repo_root, "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        diff_stat = None
        if status:
            diff_stat = subprocess.check_output(
                ["git", "-C", repo_root, "diff", "--stat"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
    except subprocess.CalledProcessError:
        status, diff_stat = "", None
    return {
        "commit": sha,
        "dirty": bool(status),
        "dirty_files": status.splitlines() if status else [],
        "diff_stat": diff_stat,
    }


def capture_env() -> dict:
    """Record python/torch/transformers versions, platform, hostname, CUDA state."""
    info: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }
    try:
        import torch  # local import so this module is import-cheap
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    return info


class Experiment:
    """One experiment folder. Owns the manifest and exposes save helpers.

    Typical use:

        exp = Experiment.create(model="Qwen/Qwen3-14B", config=cfg)
        path = exp.vectors_dir / "pi.pt"          # write your vectors there
        exp.add_output("vectors", path, kind="pi", metadata={...})
        exp.add_output("eval", eval_path)
        exp.add_output("plots", plot_path, metadata={"dataset": "..."})
        exp.finalize()  # writes manifest.json
    """

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    def __init__(self, root: Path, manifest: dict):
        self.root = root
        self.manifest = manifest
        self._t_start = time.time()
        for sub in ("vectors", "eval", "generations", "plots"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        *,
        model: str,
        config: dict,
        name: str | None = None,
        base_dir: str | Path | None = None,
        datasets: list[str | Path] | None = None,
    ) -> "Experiment":
        """Make a new experiment directory and seed the manifest."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_short = model.split("/")[-1]
        experiment_id = name or f"{ts}_{model_short}"

        base = Path(base_dir) if base_dir else cls.PROJECT_ROOT / "experiments"
        root = base / experiment_id
        root.mkdir(parents=True, exist_ok=False)

        ds_records = []
        for ds_path in datasets or []:
            ds_path = Path(ds_path)
            ds_records.append({
                "path": str(ds_path),
                "exists": ds_path.exists(),
                "sha256": _file_sha256(ds_path),
                "size_bytes": ds_path.stat().st_size if ds_path.exists() else None,
            })

        manifest = {
            "experiment_id": experiment_id,
            "created_at": _utc_now_iso(),
            "model": model,
            "config": config,
            "datasets": ds_records,
            "git": capture_git_state(cls.PROJECT_ROOT),
            "env": capture_env(),
            "outputs": {
                "vectors": [],
                "eval": [],
                "generations": [],
                "plots": [],
            },
            "duration_seconds": None,
        }

        # Save the input config alongside for easy diffing later
        with open(root / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        exp = cls(root, manifest)
        exp._save_manifest()  # write an in-progress copy immediately
        return exp

    @classmethod
    def open(cls, root: str | Path) -> "Experiment":
        """Re-open an existing experiment to append outputs."""
        root = Path(root)
        with open(root / "manifest.json") as f:
            manifest = json.load(f)
        return cls(root, manifest)

    # ── output recording ────────────────────────────────────────────────────

    @property
    def vectors_dir(self) -> Path:
        return self.root / "vectors"

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"

    @property
    def generations_dir(self) -> Path:
        return self.root / "generations"

    @property
    def plots_dir(self) -> Path:
        return self.root / "plots"

    def add_output(
        self,
        kind: str,
        path: str | Path,
        *,
        label: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Register a produced artifact. Path is stored relative to the experiment root."""
        if kind not in self.manifest["outputs"]:
            self.manifest["outputs"][kind] = []
        path = Path(path)
        try:
            rel = path.resolve().relative_to(self.root.resolve())
            rel_str = str(rel)
        except ValueError:
            rel_str = str(path)
        entry = {"path": rel_str, "abs_path": str(path.resolve())}
        if label:
            entry["label"] = label
        if metadata:
            entry["metadata"] = metadata
        self.manifest["outputs"][kind].append(entry)
        self._save_manifest()

    def write_plot_sidecar(self, plot_path: str | Path, metadata: dict) -> Path:
        """Write `<plot>.json` next to a saved PNG with full plot context."""
        plot_path = Path(plot_path)
        sidecar = plot_path.with_suffix(".json")
        with open(sidecar, "w") as f:
            json.dump({**metadata, "plot_file": plot_path.name, "saved_at": _utc_now_iso()}, f, indent=2)
        return sidecar

    # ── lifecycle ───────────────────────────────────────────────────────────

    def _save_manifest(self) -> None:
        with open(self.root / "manifest.json", "w") as f:
            json.dump(self.manifest, f, indent=2)

    def finalize(self) -> Path:
        """Stamp duration + last-updated and persist the manifest."""
        self.manifest["duration_seconds"] = round(time.time() - self._t_start, 1)
        self.manifest["finalized_at"] = _utc_now_iso()
        self._save_manifest()
        return self.root / "manifest.json"
