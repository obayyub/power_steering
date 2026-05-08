# Per-train-eval × test-eval transfer matrix (Qwen3-14B)

Generated: 2026-05-03

## What's here

### Per-train-eval × test-eval transfer matrices (the main artifact)

- `aligned.png` / `aligned.json` — 7×7 transfer matrix, aligned direction (visual heatmap).
- `misaligned.png` / `misaligned.json` — same, misaligned direction.
- `aligned_table.md` / `misaligned_table.md` — same data as a markdown table.
  21 rows (7 train evals × 3 methods), 7 test eval cols, **bold** marks
  the cell winner across the three methods. Easier to read individual
  numbers and quote in prose than the heatmap.

### Per-eval specialist bar charts

- `specialists_aligned.png` / `specialists_aligned.json`
- `specialists_misaligned.png` / `specialists_misaligned.json`

For each of the 7 evals, shows each method's TRUE specialist on its home
eval — i.e., the vector trained on that eval that maximizes alignment
shift on the same eval. This is different from the matrix's diagonal,
which uses the GENERALIST (best mean across all evals).

### Cosine similarity across the 7 specialists per method

- `cosine_specialists.png` / `cosine_specialists.json`

Tests whether the 7 home-eval specialists per method point in the same
direction in vector space. 3 panels (CAA / MELBO / PI), each is a 7×7
signed cosine matrix.

## Layout of the transfer matrix

- Rows: 7 *train* evals. The pipeline was run once per row eval, with
  `category` set to that eval. PI/MELBO trained on a single prompt
  picked from that eval (seed=0); CAA trained on 100 disjoint prompts
  from that eval (CAA `direction='aligned'` — the polarity-aware
  contrast so + scale always = HHH-aligned).
- Cols: 7 *test* evals.
- Cells: 3 numbers per cell (CAA blue, MELBO green, PI orange).
  Each is that method's *best generalist* (vector with highest mean
  alignment shift across all 7 cols among the vectors produced by
  the row's pipeline) evaluated on the col eval at the best scale.
  Bold = the method that won this cell.

## Source experiments

- `experiments/qwen3_14b_train_coordinate-other-ais` — train eval: `coordinate-other-ais`
- `experiments/qwen3_14b_train_corrigible-neutral-HHH` — train eval: `corrigible-neutral-HHH`
- `experiments/qwen3_14b_train_myopic-reward` — train eval: `myopic-reward`
- `experiments/qwen3_14b_train_power-seeking-inclination` — train eval: `power-seeking-inclination`
- `experiments/qwen3_14b_train_self-awareness-general-ai` — train eval: `self-awareness-general-ai`
- `experiments/qwen3_14b_train_survival-instinct` — train eval: `survival-instinct`
- `experiments/qwen3_14b_train_wealth-seeking-inclination` — train eval: `wealth-seeking-inclination`

All runs: Qwen/Qwen3-14B, source layer 10, target layer 32,
CAA layer 24, sample_seed=42, max_questions=100, scales
`[-25,-10,-5,-2,-1,0,1,2,5,10,25]`, PI pad=5, CAA direction=aligned.

## Headline findings (corrected)

### 1. Methods compared on best-generalist transfer (matrix means)

Mean alignment shift of each method's best generalist (off-diagonal):

| Method | Mean off-diag (transfer) | Mean diag (home) |
|---|---:|---:|
| MELBO | +13.5 pp | +17.9 |
| PI    | +11.6 pp | +16.3 |
| CAA   | +6.6 pp | +12.9 |

Cells won out of 49 (best across the 3 methods per cell):
MELBO 19 (39%), PI 18 (37%), CAA 12 (24%).

### 2. Specialist potential beats generalist (where the methods have ≥2 vectors)

| Method | Mean SPECIALIST diag | Mean GENERALIST diag | Specialization gap |
|---|---:|---:|---:|
| CAA | +12.9 | +12.9 | +0.0 (single vector — no choice possible) |
| MELBO | +22.0 | +17.9 | +4.1 |
| PI | +19.7 | +16.3 | +3.4 |

So picking the per-eval specialist gets you ~4 pp more on home eval than
picking the generalist. Big-picture: PI's `v0` and MELBO's various best
indices generalize well, but giving up the specialist edge costs ~4 pp.

### 3. ROBUSTNESS: 46% of matrix cells fail at least one signal-vs-noise check

`scripts/check_signal_robustness.py` runs three checks per cell:
- **Baseline regression ceiling**: alignment shift achievable by pure 50/50
  random output (just from regression to the mean). If observed shift is
  smaller than this in absolute terms AND in the same direction, the
  cell is indistinguishable from "vector at high |scale| broke the model".
- **Monotonicity**: Spearman ρ between signed scale and alignment shift
  across all 11 scales. |ρ| < 0.5 = suspicious.
- **Logit collapse**: ratio of |matching_logit_diff| std at chosen-best
  scale vs at scale=0. <0.5 means the response distribution narrowed
  dramatically (degeneracy signature).

Result: **79 ROBUST / 68 SUSPECT (54% / 46%)** of 147 cells.

Suspect cells concentrate in two patterns:
- Cross-eval transfer to **corrigibility** is often within the
  baseline-regression ceiling (corrigibility's 38% baseline → degeneracy
  alone gives +12 pp). Many of the bolded "transfer" cells in the
  corrigibility column are noise-explainable.
- Sub-5pp wins ("tiny") are also flagged — the small bolded numbers in
  the table are mostly noise.

The diagonal cells, plus the **PI / MELBO specialists for the
self-awareness / power-seeking / wealth-seeking cluster**, hold up
robustly (ρ ≈ −1.0, well above ceiling, logit std intact or growing).

### 4. Cosine similarity surprise: no method finds a single canonical axis

For each method, the 7 specialists (one per train eval) live in the same
hidden_dim space (5120). Mean off-diagonal |cosine|:

| Method | mean |cos| | vs random baseline (~1/√5120 ≈ 0.014) |
|---|---:|---|
| CAA | 0.35 | ~25× above random |
| MELBO | 0.14 | ~10× above random |
| PI | 0.14 | ~10× above random |

**No method has mean |cos| > 0.4.** All three methods produce
non-random directional overlap (so the alignment direction lives in
SOME shared subspace), but **none converge to a single canonical axis**
across training prompts.

Most importantly, **PI's per-train-eval `v0` is NOT a stable
model-internal direction** — different training prompts produce
different top singular vectors, even though they all generalize well
functionally. The Phase D claim "PI_v0 is a real alignment axis" was
overconfident; what's true is that PI's discovered directions are
*functionally* aligned-effective, not that they're directionally
identical.

The corrected story: **alignment is encoded redundantly enough in the
model that multiple training prompts find functionally equivalent but
directionally distinct vectors**. This is consistent with alignment
being a manifold (or low-rank subspace) rather than a single direction.

## Caveats

- Training prompts for PI/MELBO are picked via `seed=0` random choice
  from the train eval's question pool. Different seeds would give
  different vectors (and likely different transfer profiles).
- CAA train pool excludes the same 100 questions used for eval
  (`exclude_test=true, num_test=100, test_seed=42`), so per-row
  CAA vectors are trained on disjoint prompts from the test sample.
- Baselines may drift by 0-2 questions across runs (cuDNN
  nondeterminism in batched matmul). Each row's cells use that
  experiment's own baseline so per-row shifts are honest.
- The "best of N" framing flatters PI/MELBO since they have 12 vector
  candidates per training source vs CAA's 1. A fairer head-to-head
  would train 12 CAA vectors from 12 different prompts and let CAA
  pick its best generalist — that experiment is unrun.

## Polarity convention

Anthropic's `answer_matching_behavior` field is HHH-aligned for some
evals but the named-misaligned behavior for others. We sign-correct
downstream so HIGHER ALWAYS = MORE ALIGNED (see
`download_dataset.py:BEHAVIOR_POLARITY`):

| Dataset | matching = aligned? |
|---|---|
| corrigible-neutral-HHH | yes |
| survival-instinct | yes |
| power-seeking-inclination | yes |
| wealth-seeking-inclination | yes |
| self-awareness-general-ai | yes |
| coordinate-other-ais | no (flipped) |
| myopic-reward | no (flipped) |

This polarity table assumes within-eval consistency — i.e., every item
in coord-other-ais has matching = the unethical-coordination answer.
We've spot-checked but not exhaustively audited; the two flipped evals
might have within-eval polarity inconsistencies that act as natural
noise controls if so.
