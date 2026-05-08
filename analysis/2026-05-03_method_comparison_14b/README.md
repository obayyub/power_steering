# Method comparison — Qwen3-14B (CAA / MELBO / PI(pad=5))

Generated: 2026-05-03T13:16:08Z

## What's here

**Specialist figures** — best per (method, dataset):
- `aligned.png` — best alignment shift each method can achieve per dataset
- `misaligned.png` — best mis-alignment shift each method can achieve per dataset
- `aligned.json`, `misaligned.json` — full per-cell data

**Generalist transfer matrices** — top-7 generalists per method × 7 test evals:
- `generalists_aligned.png` — see "Generalist analysis (B-2 transfer matrices)" below
- `generalists_misaligned.png` — same structure, misaligned direction
- `generalists_aligned.json`, `generalists_misaligned.json` — per-cell data

Both sets share the source experiments and polarity table described below.

## Sources

- **PI (pad=5) vectors**: `experiments/20260503_005015_Qwen3-14B` (eval: `eval_20260503_011853.json`)
- **MELBO + CAA vectors**: `experiments/20260502_190738_Qwen3-14B` (eval: `eval_20260502_193520.json`)

Both runs: Qwen/Qwen3-14B, source layer 10, target layer 32,
CAA layer 24, sample_seed=42, max_questions=100, scales
`[-25,-10,-5,-2,-1,0,1,2,5,10,25]`. Same training prompt (seed=0
selects index 197 of corrigible-neutral-HHH).

Baselines drift by 0-2 questions across runs (cuDNN nondeterminism
in batched matmul affecting argmax tiebreaks). Each cell uses its
own source experiment's baseline so per-method shifts are honest.

## Scale convention — magnitude equivalence across methods

All vectors are unit-normalized before being added to the residual stream
(or to the MLP `down_proj` output, depending on `capture_site` — see
session 2026-05-02.md for the discussion). With unit-norm vectors, the
**effective per-token perturbation magnitude equals the scale**.

This is *not* how each method is natively presented in its source paper,
which is worth flagging explicitly so the comparison reads honestly:

| Method | Native vector norm | Native scale range | Effective magnitude |
|---|---|---|---|
| CAA (Rimsky et al.) | raw difference, ~14.8 here | 1–5 typical | 15–75 |
| MELBO (Mack)        | sphere radius `normalization=1.0` | 1 (paper) | 1 |
| PI / PI-RR          | unit eigenvector | unconstrained | scale |

Translation back to "native" usage:

- **CAA scale=25 unit-norm ≈ raw scale 1.7**, well inside the literature
  range. CAA is not being unfairly amplified.
- **MELBO scale=25 unit-norm ≈ 25× the trained sphere radius**. This is
  outside MELBO's design point. Mack's post does sweep multiplicative
  scales after training (the trained sphere is a regularizer, not a
  deployment magnitude), but a careful reviewer might note this.
- **PI** has no native scale convention; this is just our choice.

The figures already self-correct against per-method saturation: each
cell picks the best (vector, scale) for that method, so a method that
collapses at high |scale| would automatically pick a smaller scale. Look
at the cell annotations — different methods chose different scales per
cell, so no method is being penalized for a magnitude where it's
degenerate.

If you want to see the comparison at each method's home-turf scales
specifically, the eval JSON contains every (vector, scale) record;
re-rendering at e.g. CAA ∈ {±1, ±2, ±5} raw, MELBO ∈ {±0.5, ±1, ±2}
raw, PI same as MELBO is a small follow-up.

## How `aligned` is defined per dataset

Anthropic's `answer_matching_behavior` field is HHH-aligned for some
evals and the named-misaligned behavior for others. We sign-correct
downstream so HIGHER ALWAYS = MORE ALIGNED:

| Dataset | matching = aligned? |
|---|---|
| corrigible-neutral-HHH | yes |
| survival-instinct | yes |
| power-seeking-inclination | yes |
| wealth-seeking-inclination | yes |
| self-awareness-general-ai | yes |
| coordinate-other-ais | no (flipped) |
| myopic-reward | no (flipped) |

## Aligned-direction summary (positive figure)

Per-dataset best across all methods:

| Dataset | Winner | Shift (pp) | Vector @ scale |
|---|---|---:|---|
| coordinate-other-ais | CAA | +6.0 | v0 @ -25 |
| corrigible-neutral-HHH | MELBO | +55.0 | v11 @ +25 |
| myopic-reward | MELBO | +16.0 | v10 @ -25 |
| power-seeking-inclination | MELBO | +20.0 | v0 @ +25 |
| self-awareness-general-ai | PI (pad=5) | +23.0 | v0 @ +25 |
| survival-instinct | MELBO | +31.0 | v11 @ +25 |
| wealth-seeking-inclination | MELBO | +13.0 | v11 @ -25 |

## Misaligned-direction summary (negative figure)

Per-dataset most-destructive across all methods:

| Dataset | Winner | Shift (pp) | Vector @ scale |
|---|---|---:|---|
| coordinate-other-ais | MELBO | -41.0 | v6 @ +25 |
| corrigible-neutral-HHH | MELBO | -26.0 | v11 @ -25 |
| myopic-reward | PI (pad=5) | -37.0 | v0 @ +25 |
| power-seeking-inclination | PI (pad=5) | -17.0 | v11 @ +25 |
| self-awareness-general-ai | PI (pad=5) | -29.0 | v4 @ -25 |
| survival-instinct | CAA | -14.0 | v0 @ -25 |
| wealth-seeking-inclination | MELBO | -25.0 | v2 @ -25 |

## Generalist analysis (B-2 transfer matrices)

`generalists_aligned.png` and `generalists_misaligned.png` are 7×7
transfer matrices.

- **Rows** are the *top-7 generalists* per method, ranked by mean
  alignment shift across all 7 evals (each vector at whichever scale
  works best per eval). Row #1 = each method's strongest generalist;
  row #7 = the 7th-best generalist for that method.
- **Cols** are the 7 test evals.
- Each cell holds up to 3 numbers: CAA's R-th generalist (blue),
  MELBO's R-th generalist (green), PI's R-th generalist (orange),
  evaluated on the test eval at the best scale. The **bold** entry
  is the cell winner. CAA only has 1 vector total, so its position
  in rows 2-7 is dashed.
- Cell color = the cell winner's alignment shift (diverging RdBu).
- Annotations include the (vector_idx, scale) so you can trace any
  cell back to a specific vector.

Generalist criterion specifically: per vector V of method M,
`mean_shift(V) = mean over evals of max_over_scales of alignment_shift(V, eval, scale)`.
For the misaligned figure we use `min_over_scales` and rank ascending
(most-negative mean wins).

The diagonal of these matrices is NOT the same as figure A's bars,
since the row vector is the *generalist* for that method, not the
per-eval specialist. A vector that's best on average across all evals
can be a worse-than-best specialist on any individual eval — that
trade-off is what these figures expose.

Sidecar JSONs (`generalists_aligned.json`, `generalists_misaligned.json`)
have the full per-cell data: per (rank, method) the chosen vector_idx,
mean score, and per-eval (scale, shift_pp).
