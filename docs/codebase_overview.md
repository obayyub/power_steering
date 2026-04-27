# Power Steering Codebase Overview

How the code in this repo was used to generate experimental data, plus a file-by-file index for recreating each result.

## The core idea

Find steering vectors as the top right singular vectors of the Jacobian J = ∂(target-layer MLP output)/∂(source-layer MLP output). Block power iteration on JᵀJ gives the subspace; Rayleigh-Ritz rotates it into the true singular vectors. Costs ~15 forward passes per (source, target) pair via the reverse-over-reverse trick.

Every cloud experiment runs through `lambda_cloud.py` — it launches an instance, scp's project files to `~/project`, runs a script under `uv`, downloads `results/` and `vectors/`, then terminates.

## Experimental phases

### Phase 1 — Corrigibility on Qwen3-14B (Jan 26 → Feb 08)

Source layer 7 → target 32, 12 vectors per method, evaluated on Anthropic's `survival-instinct` and `corrigible-neutral-HHH` (60–100 prompts each, balanced A/B).

- `melbo_qwen3.py` — gradient ascent (nonlinear baseline)
- `find_power_iteration.py` — single-prompt PI (later +Rayleigh-Ritz → PI-RR)
- `find_power_iteration_multi.py` — sums JᵀJ across 32 prompts
- `run_caa_corrigibility.py` — contrastive activation addition for comparison

Both logit-diff and full-generation evals (9,720 generations across 9 scales). Bug fixes along the way: token IDs `(A`/`(B` not `A`/`B`; CAA must capture from `down_proj` not full layer; CAA at letter token (-2). Result: PI-RR ≈ MELBO ≈ Multi-PI (~65% corrigible peak), all beating CAA. Vectors saved to `vectors/{melbo,power_iter,power_iter_multi,caa}_Qwen3-14B_*.pt`.

### Phase 2 — Layer-pair sensitivity atlas (Feb 11–13)

- **Qwen3-1.7B-Base, arithmetic** (378 pairs): `map_jacobian.py` produced sigma + KL maps; `map_generate.py` produced 362,880 generations; `eval_generalize.py` / `eval_probe.py` / `eval_subtract.py` / `measure_projections.py` characterized the discovered CoT vectors (7,25)v1 and (9,18)v1 (6% → 90% accuracy). `analyze_behaviors.py` did regex behavior tagging.
- **Qwen3-8B, 7 prompts** (630 pairs): `map_diverse.py` — code, narrative, refusal, reasoning, strawberry, roleplay, persuasion, on 8x A100. Outputs in `results/diverse_map/<prompt>/merged.pt` + per-pair JSONs. The dashboard (`dashboard/prepare_data.py`, `dashboard/index.html`) consumes these.
- `analyze_refusal_subspace.py` — labeled 20 anti-refusal vectors and showed v0 is target-independent within a source layer but uncorrelated across distant source layers.

### Phase 3 — Scale and metric variations on Qwen3-8B (Feb 15–28)

All driven by `map_diverse.py` with new flags or sister scripts:

- **Norm-scaled** (`--scale-frac 0.35`): added `measure_norms()` and per-pair scale logging. Outputs to `results/diverse_map_normscale/{refusal,roleplay}/`. Confirmed mid-layer sweet spot is structural, not a scale artifact.
- **Diagonal target metrics**: `find_power_iteration_metric.py` (14B, baseline/var_var/var_inv/inv_var/inv_inv) and `map_diverse_ggb.py`/`map_diverse_cov.py` (8B, var/inv/cov atlases). `compare_target_metrics.py` measured top-1 alignment and subspace overlap. Inv ≈ baseline (0.91 cos), var found ~2 novel directions, low-rank covariance found the most novel (0.58 cos) and discovered the **(13,21)v6 selective refusal vector** evaluated by `eval_refusal_vector.py` over 10 prompt categories.
- **Golden Gate covariance** (`map_diverse_ggb.py` ± regularization): 20 GGB-themed prompts didn't produce concept-specific steering — only formatting changes.

### Phase 4 — Methodological corrections (Mar 01)

`gen_deep_pi.py`/`map_deep_pi.py` did a deep run (k=100, 8 iters, layers 12–22, 198 pairs). Comparing it to the normscale run revealed two issues:

- **Sign ambiguity**: `kl_both_signs.py` showed 43.6% of behaviorally active vectors were missed because KL(+v) ≠ KL(-v) and only +v was measured. Fix: compute both signs, take max.
- **Spectral degeneracy**: in the flat band (rank ≥5) Rayleigh-Ritz returns an arbitrary rotation, so individual vectors aren't reproducible across runs (only top ~3 are trustworthy). Earlier "specific vector found at rank 10" results were lucky alignments. Subspace itself is correct; proposed fix is gradient ascent on KL within the ~15-dim subspace.

## Where the artifacts live

- Vectors: `vectors/*.pt`
- Per-experiment maps: `results/diverse_map*/`, `results/jacobian_map/`, `results/jacobian_gen/`, `results/deep_pi_*/`, `results/refusal_vector_eval/`, `results/generations/`, `results/eval_*.json`
- Dashboard data: `dashboard/dashboard_data.json` + lazy-loaded `*_pairs/` directories
- Plots: `docs/*.png`, `results/*.png`
- The big-picture writeup: `docs/writeup.md`

---

## Primary files for recreating experiments

### Core method / vector discovery

| File | What it does |
|---|---|
| `power_block_iteration.py` | Unified block power iteration with reverse-over-reverse JVP and Rayleigh-Ritz. Single- or multi-prompt mode via `--num-prompts`. Replaces the older `find_power_iteration*.py` scripts. |
| `find_power_iteration.py` | Original single-prompt PI. Used for the Jan/Feb Qwen3-14B corrigibility runs. |
| `find_power_iteration_multi.py` | Multi-prompt PI (sums JᵀJ across N prompts). Added Rayleigh-Ritz here. |
| `find_power_iteration_metric.py` | Metric PI on 14B: prewhitens with diagonal G_s, G_t (var/inv combinations). Five configs. |
| `melbo_qwen3.py` | MELBO baseline — gradient ascent maximizing ‖f(x+v) − f(x)‖. |
| `run_caa_corrigibility.py` | CAA baseline: mean(survival_act) − mean(corrigible_act) at `down_proj`, position −2. |

### Atlas / map experiments

| File | What it does |
|---|---|
| `map_jacobian.py` | Qwen3-1.7B-Base arithmetic atlas. SVD + KL across 378 pairs → `results/jacobian_map/merged.pt`. |
| `map_generate.py` | Generates steered text for every pair × vector × question (362,880 gens). |
| `map_diverse.py` | Qwen3-8B atlas across 630 pairs and 7 prompts. Supports `--scale-frac` (norm-scaled) and `--prompts` filter. Produced `results/diverse_map/` and `results/diverse_map_normscale/`. |
| `map_diverse_cov.py` | Same atlas with low-rank covariance target metric (16 prompts). Found (13,21)v6. |
| `map_diverse_ggb.py` | Covariance variant using 20 Golden-Gate-themed prompts; supports `--cov-reg-frac`. |
| `map_deep_pi.py` / `gen_deep_pi.py` | Deep-rank atlas (k=100, 8+ iters) used to expose sign ambiguity and degeneracy. |

### Evaluation

| File | What it does |
|---|---|
| `eval_steering.py` | Logit-diff A/B evaluation on Anthropic evals (corrigibility/survival). |
| `generate_steered.py` | Full generation eval at multiple scales; tracks corrigible/survival/unclear. |
| `eval_generalize.py` | Tests CoT vectors on harder arithmetic + word problems. |
| `eval_probe.py` | Non-math tasks, alternative arithmetic formats, scale sweeps for the CoT vectors. |
| `eval_subtract.py` | Negative-scale (subtraction) test for causal relevance of a direction. |
| `eval_refusal_vector.py` | 40 prompts × 10 categories at 5 scales — used to characterize (13,21)v6 selectivity. |
| `eval_cot_math.py` | CoT math eval with sampling (temp=0.7). |
| `kl_both_signs.py` | Recomputes KL for ±v; produced the 44%-missed finding. |

### Analysis / comparison

| File | What it does |
|---|---|
| `measure_projections.py` | Projects unsteered activations onto a steering vector vs random vectors. |
| `analyze_behaviors.py` | Regex tagging (CoT, non-English, repetition, etc.) over generation dumps. |
| `analyze_refusal_subspace.py` | LOO subspace analysis on the 20 labeled anti-refusal vectors. |
| `compare_target_metrics.py` | Top-1 cosine + subspace overlap heatmaps across baseline/var/inv/cov atlases. |
| `compare_rsvd_stability.py` | Reproducibility check across PI runs (related to degeneracy work). |
| `plot_normscale_heatmap.py`, `plot_top1_violin.py`, `plot_best_by_criterion.py`, `plot_cosine_sim.py`, `plot_results.py` | Figure generation for the writeup. |

### Infrastructure

| File | What it does |
|---|---|
| `lambda_cloud.py` | Launch → upload → `uv sync` → run script → download `results/`+`vectors/` → terminate. Entry point for every cloud run. |
| `download_dataset.py` | Pulls the Anthropic survival-instinct / corrigible-neutral-HHH evals into `data/`. |
| `dashboard/prepare_data.py` | Converts `results/diverse_map*` and `jacobian_*` outputs into the dashboard JSON + per-pair files. |
| `dashboard/index.html` | Interactive Plotly dashboard for browsing pairs/vectors/generations. |

### Recreating each headline result

| Result | Use |
|---|---|
| Qwen3-14B corrigibility 4-method comparison | `melbo_qwen3.py`, `power_block_iteration.py` (or the older `find_power_iteration*.py`), `run_caa_corrigibility.py` → `eval_steering.py` → `generate_steered.py` |
| Qwen3-1.7B-Base arithmetic CoT (7,25)v1 / (9,18)v1 | `map_jacobian.py` → `map_generate.py` → `eval_generalize.py` / `measure_projections.py` / `eval_subtract.py` |
| Qwen3-8B 7-prompt atlas + dashboard | `map_diverse.py` → `dashboard/prepare_data.py` |
| Norm-scaled refusal/roleplay | `map_diverse.py --scale-frac 0.35 --prompts ...` |
| Selective refusal (13,21)v6 | `map_diverse_cov.py` → `eval_refusal_vector.py` |
| Sign-ambiguity / degeneracy diagnostics | `gen_deep_pi.py` / `map_deep_pi.py` + `kl_both_signs.py` + `compare_rsvd_stability.py` |
