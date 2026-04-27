# Session Notes — 2026-02-27

## Metric Power Iteration Experiment

Tested whether introducing a diagonal metric tensor (per-coordinate variance of activations) into the power iteration objective finds better steering vectors.

### Motivation

Standard PI maximizes `||J v||` — the largest output change per unit input perturbation. But this treats all coordinates equally. The idea: weight coordinates by their variance to either focus on high-variance (already active) or low-variance (quiet, potentially more targeted) dimensions.

### Implementation (find_power_iteration_metric.py)

Uses prewhitening to convert the generalized eigenvalue problem to standard form:

```
A_tilde = G_s^{-1/2} (sum_i J_i^T G_t J_i) G_s^{-1/2}
```

Five configurations tested:
- **baseline**: G_s = G_t = I (standard multi-prompt PI)
- **var_var**: G_s = diag(var_s), G_t = diag(var_t)
- **var_inv**: G_s = diag(var_s), G_t = diag(1/var_t)
- **inv_var**: G_s = diag(1/var_s), G_t = diag(var_t)
- **inv_inv**: G_s = diag(1/var_s), G_t = diag(1/var_t)

Variance computed from last-token activations across 32 prompts at source (layer 7, MLP down_proj) and target (layer 32). Vectors unwhitened after training (multiplied by G_s^{-1/2}) so they live in the native activation space for inference.

### Training

- **Model**: Qwen3-14B, bfloat16
- **Source layer**: 7 (MLP down_proj), **Target layer**: 32
- **Prompts**: 32 from corrigible-neutral-HHH
- **12 vectors** per config, 5 iterations + Rayleigh-Ritz
- **Hardware**: 1x H100 SXM5 on Lambda Cloud

### Logit-Based Evaluation

Cosine similarity analysis revealed two clusters:
- {baseline, var_var, var_inv} find similar directions (high cosine similarity)
- {inv_var, inv_inv} find genuinely different directions (anti-correlated with baseline)

Asymmetric steering: all configs find it easier to push toward corrigibility than survival. The inv_* configs partially break this pattern.

No metric config found a stronger survival steerer than baseline v6 at scale -25 (92% survival on logit eval).

### Generation Evaluation

Ran 5,040 generations: 6 best vectors x 120 prompts (60 from each dataset, balanced A/B) x 7 scales, temp=0.7, batch-size=16.

Vectors selected: baseline_v6, var_var_v7, var_inv_v6, inv_var_v10, inv_inv_v5, baseline_v3.

#### % Corrigible (aggregated, both datasets):

| Vector | -25 | -10 | -5 | 0 | +5 | +10 | +25 |
|---|---|---|---|---|---|---|---|
| baseline_v6 | 26% | 28% | 32% | 36% | 42% | 22% | 9% |
| var_var_v7 | 29% | 30% | 36% | 38% | 36% | 32% | 36% |
| var_inv_v6 | 22% | 29% | 32% | 38% | 41% | 33% | 19% |
| inv_var_v10 | 41% | 36% | 41% | 37% | 32% | 32% | 35% |
| inv_inv_v5 | 49% | 35% | 40% | 38% | 28% | 28% | 28% |
| baseline_v3 | 68% | 52% | 50% | 36% | 9% | 5% | 7% |

#### % Unclear at extreme scales:

| Vector | -25 | +25 |
|---|---|---|
| baseline_v6 | 9% | 82% |
| baseline_v3 | 4% | 81% |
| var_var_v7 | 10% | 46% |
| var_inv_v6 | 12% | 67% |
| inv_var_v10 | 37% | 5% |
| inv_inv_v5 | 25% | 13% |

### Efficiency Analysis

Computed "unclear cost per unit steering" — how many percentage points of unclear you pay per percentage point of behavioral shift. Lower is better.

| Vector | Avg Steer | Avg Unclear Cost | Ratio | Interpretation |
|---|---|---|---|---|
| melbo_n1_v5 | 20.0% | 10.1% | 0.51x | Best overall |
| pi_rr_v7 | 14.6% | 9.6% | 0.66x | Efficient |
| inv_inv_v5 | 7.6% | 8.6% | 1.13x | Break-even but weak |
| pi_rr_v9 | 19.7% | 25.1% | 1.27x | Strong but costly |
| baseline_v3 | 24.9% | 38.2% | 1.54x | Strong but costly |
| inv_var_v10 | 3.2% | 8.3% | 2.61x | Weak and inefficient |
| var_var_v7 | 4.4% | 15.4% | 3.47x | Worst overall |

### Key Findings

1. **Metric PI didn't improve steering.** The inv_* configs found genuinely different directions (anti-correlated with baseline), but those directions don't correspond to strong corrigibility behavior.

2. **Inv vectors stay coherent but don't steer.** inv_var_v10 and inv_inv_v5 have low unclear rates at all scales (<37%), but their steering range is narrow (3-8% average shift). They're not more efficient — they're just not pushing hard enough to matter.

3. **Var-var is the worst.** Upweighting high-variance coordinates at both source and target adds noise without finding targeted directions. 3.47x unclear-to-steering ratio.

4. **Original methods dominate.** melbo_n1_v5 (0.51x ratio, 20% avg steering) and pi_rr_v7 (0.66x ratio) remain the best on both steering strength and efficiency.

5. **Inverse-variance focuses on quiet dimensions that aren't behaviorally meaningful.** The intuition that low-variance coordinates might carry targeted behavioral signal didn't hold — those dimensions are quiet because they don't encode the behaviors we care about.

### Conclusion

The metric tensor approach is a negative result. Standard PI-RR and MELBO find the strongest and most efficient steering vectors. The variance structure of activations doesn't provide useful signal for improving vector discovery.

## Files

- **find_power_iteration_metric.py**: Metric PI training script (5 configs, prewhitening, unwhitening)
- **vectors/metric_pi_{config}_Qwen3-14B_20260227_041541.pt**: Trained vectors (5 files)
- **results/generations/generations_20260227_080208.json**: 5,040 generation results
- **results/eval_20260227_*.json**: Logit-based eval results (5 files)
- **results/2026-02-08_corrigibility_14B/analyze_generations.py**: Updated to include metric PI data
- **results/metric_pi_cosine_similarity.png**: Cosine similarity heatmaps
- **results/metric_pi_violin_all_{config}_{dataset}.png**: Violin plots (10 files)
- **results/metric_pi_asymmetry.png**: Asymmetric steering bar chart

## Target Metric Alignment Analysis (Qwen3-8B, full atlas)

Compared steering vectors across three `map_diverse.py` runs on Qwen3-8B (36 layers, roleplay prompt) that used different target metrics: baseline (no weighting), var, and inv. All 630 upper-triangle layer pairs, 12 vectors each.

### Script: `compare_target_metrics.py`

For each (source, target) pair and each method pair, computes:
- **Top-1 alignment**: `|cos(v0_A, v0_B)|`
- **Subspace overlap (top-3)**: mean principal cosine of `V_A[:3] @ V_B[:3].T`

Output: `results/target_metric_alignment.png` — 6 heatmaps + 2 line plots.

### Top-level alignment (averaged over all 630 pairs)

| Comparison | Top-1 |cos| | Subspace overlap |
|---|---|---|
| var vs baseline | 0.627 | 0.668 |
| inv vs baseline | 0.912 | 0.923 |
| var vs inv | 0.469 | 0.552 |

### Full 12-d subspace analysis: inv vs baseline

Inv barely finds anything new. Per-rank max-cosine (each inv vector's best match to any baseline vector):

| Rank | Mean max |cos| | % pairs < 0.5 |
|---|---|---|
| 0 | 0.967 | 0.0% |
| 5 | 0.817 | 1.6% |
| 11 | 0.675 | 19.0% |

Principal cosines of the full 12-d subspaces: 0.999 down to 0.583. The two subspaces share ~11 of 12 dimensions. 1/variance weighting can't overcome J's natural spectral structure — the Jacobian has tiny entries in low-variance coordinates, so upweighting them (tiny × huge ≈ modest) doesn't change the dominant directions.

### Full 12-d subspace analysis: var vs baseline

Var finds genuinely new directions. Per-rank max-cosine:

| Rank | Mean max |cos| | % pairs < 0.5 |
|---|---|---|
| 0 | 0.734 | 7.3% |
| 1 | 0.558 | 37.3% |
| 5 | 0.645 | 19.5% |
| 11 | 0.532 | 41.4% |

Principal cosines: 0.995 down to 0.244. The subspaces share ~9-10 directions but var finds ~2 directions nearly orthogonal to baseline's span. Variance weighting amplifies J's natural bias (J already has large entries in high-variance coordinates), but reranks enough to surface new directions that the identity metric misses.

### Interpretation

The target metric W changes the power iteration from J^T J to J^T W J — it reweights which target-side changes matter. The Jacobian naturally routes perturbations toward high-variance target coordinates, so:
- **var (W = variance)**: amplifies what J already does, reranks enough to find ~2 genuinely new directions
- **inv (W = 1/variance)**: fights J's structure but loses — the dominant subspace is unchanged

This is consistent with the 14B metric PI results above: inv_* configs found anti-correlated logit directions but those directions weren't behaviorally meaningful. The problem isn't just the diagonal assumption — it's that the Jacobian's spectral structure is too strong for simple reweighting to overcome in the low-variance directions.

### Case study: var finds behaviorally active directions that baseline misses

Examined specific vector pairs from the dashboard generations:

**14→31, var v6 vs baseline v6**: cos = -0.77 (mostly the same direction, flipped sign). But:
- Baseline v6: KL = 0.047 — completely inert
- Var v6: KL = 8.21 — strong roleplay behavior in generations

Same steering scale (~10), sigmas differ (var 210 vs baseline 40, inflated by variance weighting). The 23% angular difference between the two vectors is enough to flip from inert to strongly active. The behavioral landscape is steep — small rotations in vector space cross large KL gradients.

**15→25, var v5 vs baseline v4**: cos = 0.73. Baseline v4 produces formal text, var v5 produces more formal text too but with different character. Partial overlap with baseline v3 (-0.39) and v5 (-0.41) — it's a mix of baseline directions.

**Implication**: Var weighting doesn't primarily find new subspaces (the principal cosine analysis shows ~10/12 shared dimensions). Instead, it nudges vectors within the same neighborhood into behaviorally relevant sweet spots. The optimization surface under variance weighting has different local optima that happen to land in high-KL regions more often. This is a subtler benefit than "finding new directions" — it's finding better directions within the same space.

### Tangent vs cotangent interpretation

The target metric G_t converts tangent → cotangent at the target layer (lowering indices). This is the natural direction for a metric tensor — it assigns importance to changes. The source metric requires G_s^{-1} (raising indices, cotangent → tangent), which needs matrix inversion and is impractical with low-rank covariance estimates.

A full low-rank covariance target metric (not just diagonal) could be computed cheaply from ~16 prompts and applied as a matrix multiply in the JVP step. This would project target changes onto the subspace of prompt-to-prompt variation, which is arguably cleaner than diagonal variance. The source side doesn't need a metric at all — keep it as identity.

## Open Questions

- **Low-rank covariance target metric**: Use full rank-k covariance (from k prompts) as W_t instead of diagonal variance. Cheap (just a matmul), no inversion needed. Would it find better directions than diagonal var?
- **More prompts for target metric**: Current 16 roleplay prompts → rank 16. Would 100+ prompts give a richer target metric?
- **Task-specific prompt sets**: The roleplay prompts define what "variation" means. Different prompt sets (e.g., refusal scenarios) would define different metrics and potentially find task-specific steering vectors.

- Could a full covariance matrix (not just diagonal) work better? The diagonal assumption may be too restrictive.
- Would computing variance over more prompts (currently 32) give a better metric estimate?
- Is there a way to use the metric at inference time too (metric-aware steering injection) rather than just for vector discovery?
- The quiet dimensions found by inv_* configs — do they correspond to anything interpretable (e.g., specific attention heads or feature circuits)?
