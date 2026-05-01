"""Plotting utilities for steering vector evaluation results."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ── Data loading & aggregation ──────────────────────────────────────────────


def load_eval_results(path: str | Path) -> dict:
    """Load evaluation results JSON."""
    with open(path) as f:
        return json.load(f)


def load_generation_results(path: str | Path) -> dict:
    """Load generation results JSON."""
    with open(path) as f:
        return json.load(f)


def aggregate_stats(
    records: list[dict],
    group_keys: tuple[str, ...] = ("vector_type", "scale"),
    metric_key: str = "survival_logit_diff",
) -> dict[tuple, dict]:
    """Aggregate records by group keys.

    Returns {group_tuple: {"values": [...], "n": int, "mean": float}}.
    """
    groups = defaultdict(list)
    for r in records:
        key = tuple(r[k] for k in group_keys)
        groups[key].append(r[metric_key])

    return {
        key: {"values": vals, "n": len(vals), "mean": np.mean(vals)}
        for key, vals in groups.items()
    }


def compute_choice_percentages(
    classified: list[dict],
    group_keys: tuple[str, ...] = ("vector", "scale"),
) -> dict[tuple, dict[str, float]]:
    """Compute % corrigible / survival / unclear per group.

    Expects records with a 'result' field (from classify_results).
    """
    counts = defaultdict(lambda: {"corrigible": 0, "survival": 0, "unclear": 0, "total": 0})
    for r in classified:
        key = tuple(r[k] for k in group_keys)
        counts[key][r["result"]] += 1
        counts[key]["total"] += 1

    pcts = {}
    for key, c in counts.items():
        t = c["total"]
        pcts[key] = {
            "corrigible": 100 * c["corrigible"] / t,
            "survival": 100 * c["survival"] / t,
            "unclear": 100 * c["unclear"] / t,
            "total": t,
        }
    return pcts


# ── Violin plot ─────────────────────────────────────────────────────────────


def _prepare_df(results, dataset_name=None):
    """Filter results and add vector label column.

    If dataset_name is None, use all results (combined across datasets).
    """
    import pandas as pd

    if dataset_name:
        filtered = [r for r in results if r["dataset"] == dataset_name]
    else:
        filtered = list(results)
    if not filtered:
        return None

    df = pd.DataFrame(filtered)
    df["vector"] = df["vector_type"] + "_v" + df["vector_idx"].astype(str)
    return df


def violin_logit_diff(
    results: list[dict],
    dataset_name: str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (12, 5),
) -> plt.Figure | None:
    """Violin plot of survival logit diff by scale, colored by vector.

    Args:
        results: List of result dicts (from eval JSON "results" field).
        dataset_name: Filter to this dataset. None = combine all datasets.
        title: Plot title.
        ax: Existing axes to draw on. Creates a new figure if None.
        figsize: Figure size when creating new figure.

    Returns:
        Figure if a new one was created, else None.
    """
    df = _prepare_df(results, dataset_name)
    if df is None:
        return None

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.violinplot(
        data=df, x="scale", y="survival_logit_diff",
        hue="vector", ax=ax, inner="quartile", cut=0,
    )

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Steering scale")
    ax.set_ylabel("Survival logit diff")
    ax.set_title(title or dataset_name or "All datasets combined")
    ax.legend(title="Vector", loc="upper left")

    if fig:
        fig.tight_layout()
    return fig


def violin_per_vector(
    results: list[dict],
    dataset_name: str | None = None,
    title: str | None = None,
    cols: int = 3,
    figsize_per: tuple[float, float] = (10, 4),
) -> plt.Figure | None:
    """One subplot per steering vector, each showing a violin of logit diff by scale.

    Args:
        results: List of result dicts (from eval JSON "results" field).
        dataset_name: Filter to this dataset. None = combine all datasets.
        title: Overall suptitle.
        cols: Number of columns in the subplot grid.
        figsize_per: (width, height) per subplot.

    Returns:
        Figure, or None if no matching results.
    """
    df = _prepare_df(results, dataset_name)
    if df is None:
        return None

    vectors = sorted(df["vector"].unique())
    n = len(vectors)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(figsize_per[0] * cols, figsize_per[1] * rows),
        squeeze=False,
    )

    for idx, vec in enumerate(vectors):
        ax = axes[idx // cols][idx % cols]
        vec_df = df[df["vector"] == vec]

        sns.violinplot(
            data=vec_df, x="scale", y="survival_logit_diff",
            ax=ax, inner="quartile", cut=0, color="steelblue",
        )
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax.set_title(vec)
        ax.set_xlabel("Scale")
        ax.set_ylabel("Surv. logit diff")

    # Hide unused subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(title or dataset_name or "All datasets combined", fontsize=14)
    fig.tight_layout()
    return fig


# ── Line plot for generation choices ────────────────────────────────────────


def line_corrigible(
    classified: list[dict],
    vector_names: list[str] | None = None,
    title: str = "% Corrigible by scale",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure | None:
    """Line plot of % corrigible vs scale for each vector.

    Args:
        classified: Records with 'result', 'vector', 'scale' fields.
        vector_names: Subset of vectors to plot (None = all).
        title: Plot title.
        ax: Existing axes.
        figsize: Figure size.
    """
    pcts = compute_choice_percentages(classified, ("vector", "scale"))

    vectors = vector_names or sorted({k[0] for k in pcts})
    scales_set = sorted({k[1] for k in pcts})

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for vec in vectors:
        scales = []
        corr = []
        for s in scales_set:
            if (vec, s) in pcts:
                scales.append(s)
                corr.append(pcts[(vec, s)]["corrigible"])
        ax.plot(scales, corr, marker="o", label=vec)

    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Steering scale")
    ax.set_ylabel("% Corrigible")
    ax.set_title(title)
    ax.legend()

    if fig:
        fig.tight_layout()
    return fig


# ── Save helper ─────────────────────────────────────────────────────────────


def save_plot(fig: plt.Figure, path: str | Path, dpi: int = 150):
    """Save a figure to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot: {path}")
    plt.close(fig)
