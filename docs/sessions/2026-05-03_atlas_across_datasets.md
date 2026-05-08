# Session — 2026-05-03 — Layer-pair PI atlas across 7 Anthropic risk evals on Qwen3-14B

(Parallel session to `2026-05-03.md`, which covers a separate concurrent
thread by another agent — per-train-eval × test-eval transfer matrix +
robustness audit. This file covers the atlas/sweep work specifically.)

Goal: stand up a fast, behavior-anchored PI atlas (logit-diff per
(source, target) pair, both signs, per vector) so we can answer "is the
hand-picked (10, 32) layer pair actually where the strongest steering
lives?" across all 7 Anthropic advanced-AI-risk evals from Phase D, and
generate the empirical evidence needed to decide between method-paper /
workflow-paper / analysis-paper framings for an upcoming submission.

---

## Phase 1 — Design + implementation

### Why
2026-05-02's pipeline runs PI/MELBO/CAA at one hand-picked (s, t).
Sessions 2/3 already showed PI's behavioral dimensionality is small
(~3) and that mid-rank vectors decouple from σ_top, so the obvious
follow-up is "what happens at *other* layer pairs?". MELBO at every
(s, t) is infeasible (~120s/pair × 780 pairs × 7 cats = days); PI at
every (s, t) is tractable in hours per category. The only existing
layer-pair atlas (`map_diverse.py`) used KL on free-form prompts, not
behavioral logit-diffs on labeled evals.

### Decisions (locked in conversation)
- **Model**: Qwen3-14B (matches sessions 2/3 baselines; the eventual
  MELBO-at-atlas-best comparison story lives there).
- **Pair geometry**: full upper triangle (`target > source`, 780 pairs
  for 40 layers).
- **Scale**: per-pair `scale = scale_frac × source_norm`, with
  `scale_frac = 0.35` and `source_norm` measured at the source layer's
  `mlp.down_proj` *output* (NOT residual stream — matches map_diverse.py
  and matches the steering injection site, so magnitudes line up). User
  flagged this carefully — initial design used residual norm; corrected
  to down_proj norm before writing code.
- **Per-pair PI**: `num_vectors=12, pad=5, num_iters=5, num_tokens=2`
  (so k_total=17 columns iterated, top 12 kept after Rayleigh-Ritz).
  User wanted full k=12 (not k=3 as I initially proposed) — keeps the
  per-pair vector count consistent with the rest of the pipeline.
- **Both signs**: per-pair eval at +scale and −scale, recorded
  separately. Phase 4's sign-ambiguity finding made this non-negotiable.
- **Per-pair behavioral eval**: 16 balanced questions
  (`max_questions=16`, `sample_seed=42`), fixed across all pairs so
  heatmap cells are directly comparable. Discussed bumping to 32
  (~16 min more); kept at 16 for the test run.
- **KL recorded but not gated**: per user's instruction. Future filter
  threshold of 0.5 from map_diverse.py kept available but not applied.
- **One prompt per category**: training prompt picked deterministically
  from the category by `random.Random(seed=0).randrange(len(items))`,
  matching `pipeline.py`'s convention.

### Done

- **`src/power_steering/map_layers.py`** — new top-level driver mirroring
  `pipeline.py`'s structure but replacing the single (s, t) pass with a
  triangle sweep. Reuses `Experiment`, `find_pi_vectors`,
  `SteeringEvaluator`. Per-pair: PI → measure scale → batched 2k-row
  KL forward → per-vector logit-diff eval through a cached
  `SteeringEvaluator` (one per source layer; constructed lazily).
  Atomic per-pair JSON writes (`.tmp` → `os.replace`) so an inspect
  script can read mid-run without seeing torn files.
  `merge_category()` rolls all per-pair JSONs into `merged.pt`
  (dense `[n_layers, n_layers, k]` tensors for `sigma`, `kl_pos`,
  `kl_neg`, `ld_pos`, `ld_neg`, `[n_layers, n_layers]` for `scale`,
  plus `vectors` dict). Snapshot `merged.pt` rewritten every
  `snapshot_every=50` pairs so dense maps are also live.

- **Resume + merge-only flags**:
    - `--resume <experiment_dir>` opens existing manifest (via
      `Experiment.open`), skips per-pair JSONs already on disk.
    - `--merge-only` rebuilds `merged.pt` without loading the model.

- **`scripts/inspect_map.py`** — saved file (per CLAUDE.md
  no-evaporating-scripts rule). Walks `pairs/*.json` (skipping torn
  files), prints summary stats + top-N pairs. Safe to run while sweep
  is in progress.

- **Configs** (in `scripts/configs/`):
    - `map_corrigibility_qwen14b.json` — single-category test run.
    - `map_phase_d_qwen14b_pdA.json` — self-awareness, survival, power.
    - `map_phase_d_qwen14b_pdB.json` — wealth, coord-other, myopic.
  pdA/pdB use explicit `experiment_name` to avoid timestamp collision
  when launched concurrently on the same multi-GPU host.

### Subtle finding — KL vs logit-diff
User asked early: "are we doing logit-diff INSTEAD of KL?" Answer:
both, they measure different things. KL is direction-agnostic (vector
did *something*); logit-diff is direction-aware (vector pushed
toward/away from matching answer). KL is the future filter
(`< 0.5 → discard`); logit-diff is the headline metric. User then asked
"should we even bother with KL?" — kept it because it's ~5% of pair
time and is the right diagnostic for flat-band degeneracy ("vectors
with high KL but flat ld are the rotation-noise vectors from rank ≥ 5").
Recorded, not central.

### Subtle finding — batching ceiling
User intuition: "PI ≈ 15 forwards via reverse-over-reverse, so 16 logit
diffs would double the cost?" Reality: PI's per-iter cost is dominated
by 51 grad calls, not the 1 forward. The "15 forwards" framing is
asymptotic compute, not wall-clock. Empirically PI ≈ 28s/pair on A100,
eval ≈ 2-4s (~10-15% of pair time, batched per-vector at batch=16).
Per-pair total ~13-32s on A100, ~10-15s on H100.

---

## Phase 2 — A100 test run, single category

### Setup
- IP: 150.136.208.15 (A100-SXM4-**40GB**, fresh instance).
- Required uv install + full project rsync + `uv sync`.
- One pre-existing irritation: `pyproject.toml` references `README.md`,
  uv sync fails without it. Fixed by uploading README too.
- Launch: `PYTHONUNBUFFERED=1 setsid nohup ~/.local/bin/uv run python
  -u -m power_steering.map_layers
  scripts/configs/map_corrigibility_qwen14b.json > pipeline.log 2>&1 &`

### Result
780/780 pairs in 13,872s (3.85 hr wall-clock).

Per-pair timing varies with `(target − source)` — partial-grad cost
scales with how many layers the autograd chain spans. source=0 with
deep targets ran ~25s/pair; mid-source pairs ran ~10-15s/pair.

GPU: ~33% util, 31/40 GB peak.

The first 50 pairs in the snapshot showed σ_top range, KL range, and
`|ld|_max` all behaving sensibly. By pair 484 — `(15, 19)` — the atlas
hit `|ld|_max = 7.23`, already 2.5× the source=0 maximum and the first
real signal that mid-network shallow-target pairs are where
corrigibility steering lives.

### Artifact
`experiments/20260503_153947_Qwen3-14B/`. 923 MB on disk after rsync
(per-pair JSONs include vectors as fp16 lists for resume).

---

## Phase 3 — 2× H100 launch for remaining 6 categories

User rented a 2× H100 80GB instance (68.209.75.24) to run the other 6
categories in parallel while the A100 finished corrigibility.

### Decisions
- **Pin one process per GPU** via `CUDA_VISIBLE_DEVICES`. 14B fits
  comfortably on one H100 (~28 GB weights + ~10 GB working ≈ 38 GB),
  so splitting via `device_map="auto"` across both GPUs would just add
  comm overhead.
- **Balanced split** by mean question length:
    - GPU 0 (pdA): self-awareness (101 ch) + survival (247) + power
      (245) = 593 chars
    - GPU 1 (pdB): wealth (225) + coord-other (250) + myopic (212)
      = 687 chars
  Self-awareness's much-shorter prompts (~half the others) made any
  3+3 split somewhat imbalanced; this is the closest achievable.
- **Explicit `experiment_name` per config** to avoid `Experiment.create`
  collision (`mkdir(exist_ok=False)` would crash if both processes
  resolve to the same UTC second).

### Wrinkle — task-notification false alarms
Three "background command failed exit code 255" notifications fired
during the session. Each was the SSH command's *detach* exit code (the
client closes after backgrounding the process), NOT the actual job. The
remote Python processes ran fine in all three cases (verified by `ps`
+ log progress). Annoying but harmless.

### Result

| Process | Categories | Wall-clock | Per-pair avg |
|---|---|---:|---:|
| A100 | corrigibility | 3.85 hr | ~18s |
| H100 GPU 0 (pdA) | self-aware, survival, power | ~5.0 hr | ~12s |
| H100 GPU 1 (pdB) | wealth, coord-other, myopic | ~7.5 hr | ~12s |

H100 noticeably faster than A100 per-pair (~12s vs ~18s avg). All 7
atlases finished in roughly one wall-clock evening, with the A100
finishing first.

Pulled to
`experiments/{phase_d_pdA,phase_d_pdB,20260503_153947}_Qwen3-14B/`
locally. ~6.4 GB total across the three.

---

## Findings — the atlas itself

### Atlas-best (s, t) per eval, aligned-direction

(After sign-correcting through the polarity table for the two evals
where `matching ≠ aligned`. Polarity inferred from 2026-05-02 session
note; coord-other and myopic carry `aligned_sign = -1`.)

| Eval | Best (s, t) | aligned_ld | best vec / steer-sign | (s, t) gap |
|---|---|---:|---|---:|
| corrigible-neutral-HHH | (18, 25) | **+14.95** | v8 / +scale | 7 |
| power-seeking-inclination | (20, 30) | **+21.02** | v0 / +scale | 10 |
| wealth-seeking-inclination | (21, 27) | **+17.51** | v3 / −scale | 6 |
| coordinate-other-ais | (21, 22) | **+16.86** | v2 / +scale | **1** |
| self-awareness-general-ai | (12, 31) | **+13.07** | v5 / −scale | 19 |
| myopic-reward | (20, 23) | **+10.11** | v3 / −scale | 3 |
| survival-instinct | (18, 27) | **+1.48** | v11 / −scale | 9 |

### Headline observations

1. **Hand-picked (10, 32) is wrong for every eval.** Atlas-best sources
   cluster in `s = 18-23`; (10, 32) sits in a different region. Not a
   single eval prefers a layer pair anywhere near the conventional
   choice. This is the methodological-critique evidence the
   workflow-paper framing depends on.

2. **Eval-specific layer-pair geometry.** Coord-other peaks at
   **adjacent layers (21, 22)** — a single block. Self-awareness wants
   a 19-layer span. Survival, myopic, wealth all peak at small-to-mid
   gaps (3-9). Different behaviors live at different (s, t) structures
   — not a one-pair-fits-all picture.

3. **Vector winners span the spectrum** (v0, v1, v2, v3, v4, v5, v7, v8,
   v11). The "pi_v0 always wins" result from session 2 was an artifact
   of being stuck at (10, 32) — at the right (s, t), mid-rank PI vectors
   are behaviorally meaningful. Padding (sessions 3) helps stabilize
   the boundary; the atlas reveals that the *layer pair* matters more
   for vector diversity than padding does.

4. **Asymmetric steerability is real but eval-specific.**
   Aligned/misaligned magnitude ratios at atlas-best:
   - coord-other: **17×** aligned-favored (the model is enormously
     easier to make less coordinative than more)
   - power, wealth: 2.4× aligned-favored
   - self-aware: 2.0× aligned-favored
   - myopic: 1.6× aligned-favored
   - corrigibility: ~1× (symmetric)
   - **survival: 8× MIS-aligned-favored** (the outlier)

5. **The survival-instinct asymmetry is the most striking single
   result.** The strongest aligned-direction PI steering for survival,
   anywhere in the 780-pair atlas, is **+1.48** — barely above noise.
   The strongest misaligned-direction is **+11.98**. The model can be
   pushed strongly toward "resist shutdown" but not pulled toward
   "accept shutdown." Whether this generalizes to other models is the
   obvious follow-up.

6. **Active vector counts vary by eval.** Corrigibility and coord-other
   have ~10-11/12 active vectors per pair; self-awareness and survival
   have only 4-7/12 active. Some evals' behavioral subspace is denser
   than others' at the same (s, t) budget.

### Caveats worth flagging
- 16-question per-pair eval → stderr ~0.5-1.0 logits. Rankings within
  ~2 logits of each other are noisy.
- Big `|ld|` values (>10) probably hit saturation (model essentially
  always-A or always-B); the drill-down with full 100-question eval +
  `chose_matching` percentage will tell whether the steering is *useful*
  or just collapsed.
- Coord-other-ais baseline match% was 92% per session 2 — already
  near-ceiling. The +16.86 swing might just be flipping the model fully
  to the aligned side, not "subtle steering."
- `aligned_sign` for coord-other and myopic was inferred from a single
  example each in 2026-05-02's session note. Should sanity-check
  per-question if it becomes load-bearing. (The parallel session in
  `2026-05-03.md` actually built `BEHAVIOR_POLARITY` into
  `download_dataset.py` — should reconcile inspect_map's `ALIGNED_SIGN`
  table with the regenerated dataset's per-item `aligned_letter`
  field next session.)

---

## Discussion — paper framing

Long thread on how to position this work, motivated by an imagined
reviewer critique:

> "Power steering is just a fancy name given to using a few iteration
> run of MELBO to locate optimal source/target pairs."

The critique is mechanically false (PI is `eigh(J^T J)` via
reverse-over-reverse + Rayleigh-Ritz, no optimization, no init that
matters past iter ~5; MELBO is gradient ascent on `‖f(x+v) − f(x)‖`
with a sphere constraint and 300 Adam steps per vector). But it's
**theoretically partial**: in the small-‖v‖ limit MELBO's objective
IS what PI globally optimizes. So "PI is small-norm MELBO" is true
in that limit, even if PI is a different algorithm.

### Cost reality check
User pushed back hard on the "100× cheaper" claim I'd been throwing
around. Honest accounting at apples-to-apples quality:

| Config | Time / pair | vs MELBO 300-step |
|---|---:|---:|
| PI 5-iter pad=5 (atlas) | ~17s | 7× |
| PI 15-iter pad=5 (quality) | ~50s | 2.4× |
| MELBO 300 steps, k=12 | ~120s | 1× |

So the "100×" was conflating screening-quality PI vs full MELBO. At
apples-to-apples, the cost gap is more like 2-3×; at atlas-screening
quality it's ~7×. The atlas-scale 100× argument requires reducing
PI's per-pair quality, which is fine for screening but means we're
not directly comparing equal-quality vectors.

### Framing options considered
1. **Method paper** (current `omar.bet/2026/02/17/Power-Steering/`
   framing): "PI as a cheaper alternative to MELBO." Loses to the
   linear-MELBO critique because the contribution is "the method,"
   and the method is mathematically a special case of MELBO.
2. **Pareto / tool paper**: "almost as good, much cheaper." Cleaner
   but weak — "almost as good" is overstated (PI is 80% of MELBO on
   14B, 100% on 8B; trend points the wrong way as models scale up).
   Lands at workshop tier at best.
3. **Workflow paper**: "field has been comparing steering methods
   unfairly by hand-picking layer pairs; here's an atlas-based fair
   comparison; conclusions change." Methodological critique of
   existing work — has stakes that the tool paper doesn't.
4. **Analysis paper** (model-property findings): single-model + single
   eval-suite is too narrow to claim properties of LMs in general.
   Would need multi-model scaling.

### Takeaway
Workflow framing is the strongest defensible angle for a venue paper.
It dodges the linear-MELBO critique cleanly because PI's novelty
isn't the load-bearing claim — the contribution is the *pipeline*
(cheap screen → expensive optimize → fair comparison), not any one
component. What the workflow paper *requires* empirically:
- Atlas-best ≠ hand-picked. ✓ (this session's data confirms it for 7/7)
- MELBO-at-atlas-best > MELBO-at-hand-picked. ⏳ (drill-down
  experiment; not yet run)
- Atlas is meaningfully cheaper than alternatives (full MELBO atlas).
  ✓ trivially (5 hr vs days).

The drill-down (point 2) is the load-bearing experiment that hasn't
happened yet. Without it, the workflow story is incomplete.

### Submission timing
User is considering NeurIPS in 2 days + ICML mech interp workshop in
parallel. Honest read communicated to user:
- NeurIPS in 2 days is wildly ambitious; load-bearing experiments
  haven't run, and rushing the writeup typically shows in reviews.
- The existing `omar.bet` writeup is the method-paper framing and would
  draw the linear-MELBO critique if submitted to NeurIPS as-is.
- Recommended primary path: workshop submission of the existing
  framing + later venue submission with proper workflow framing,
  full drill-down, and possibly multi-model scaling. NeurIPS as a
  stretch only if the drill-down lands cleanly in 36-48 hours.

User's call.

---

## Files created or modified

### New
- `src/power_steering/map_layers.py` — main atlas driver.
- `scripts/inspect_map.py` — mid-run + post-run summary inspector with
  `--direction {abs,aligned,misaligned}` (added late in session).
- `scripts/configs/map_corrigibility_qwen14b.json` — single-category
  test config.
- `scripts/configs/map_phase_d_qwen14b_pdA.json` — 3 categories, GPU 0.
- `scripts/configs/map_phase_d_qwen14b_pdB.json` — 3 categories, GPU 1.
- `scripts/configs/drill_<eval>_<s>_<t>.json` × 8 — Phase 4 drill-down
  configs (random-init MELBO at atlas-best, one per eval at atlas-aligned-best
  plus survival_misaligned-best variant).
- `scripts/configs/drill_pi_init_<eval>_<s>_<t>.json` × 8 — Phase 5
  warm-start configs (identical to drill_*.json but with
  `melbo.init_from_pi: true`).
- `scripts/compare_atlas_vs_handpicked.py` — pairs each
  `experiments/drill_*` with the parallel session's
  `experiments/qwen3_14b_train_*` baseline and prints per-eval +
  summary best-aligned-shift comparison.
- `scripts/compare_pi_init_warmstart.py` — three-way comparison of
  random-init MELBO at hand-picked, random-init MELBO at atlas-best,
  and PI-init MELBO at atlas-best. Reports both aligned and misaligned
  directions per eval + mean Δs.
- `docs/sessions/2026-05-03_atlas_across_datasets.md` — this file.

### Modified
- `scripts/inspect_map.py` — added `ALIGNED_SIGN` polarity table +
  `--direction` flag mid-session, after seeing that 2/7 atlas-best by
  `|abs_ld|` flipped polarity (originally read as misaligned for two
  evals; on closer inspection only survival flips).
- `src/power_steering/find_vectors.py` — `find_melbo_vectors` gained
  optional `init_vectors` parameter (default `None` → unchanged
  behaviour). When provided, uses it as warm-start init instead of
  random per-vector init. Phase 5 addition; backwards-compatible.
- `src/power_steering/pipeline.py` — `melbo.init_from_pi: true` config
  flag now causes PI vectors to be passed as MELBO init when both
  methods are in `methods`. Defaults preserved; saved-vector metadata
  records whether warm-start fired.

### Outputs preserved locally (rsync'd from instances)
- `experiments/20260503_153947_Qwen3-14B/` (A100, corrigibility atlas) — 923 MB.
- `experiments/phase_d_pdA_Qwen3-14B/` (H100 GPU 0 atlas, 3 cats) — 2.7 GB.
- `experiments/phase_d_pdB_Qwen3-14B/` (H100 GPU 1 atlas, 3 cats) — 2.7 GB.

Each atlas dir contains `manifest.json`, `config.json`,
`map/<category>/pairs/*.json`, `map/<category>/merged.pt`.

- `experiments/drill_<eval>_<s>_<t>_Qwen3-14B/` × 8 (~85 MB total) —
  Phase 4 random-init drill, per-eval pipeline at atlas-best layer pair.
- `experiments/drill_pi_init_<eval>_<s>_<t>_Qwen3-14B/` × 8 (~85 MB
  total) — Phase 5 PI-init MELBO warm-start runs at the same atlas-best
  layer pairs.

Each contains standard `manifest.json`, `vectors/`, `eval/`, `plots/`
from `pipeline.py`.

---

---

## Phase 4 — Drill-down: MELBO + PI at atlas-best vs hand-picked (10, 32)

This is the load-bearing experiment for the workflow paper: does training
the slower nonlinear method (MELBO) at the atlas-discovered (s, t) beat
training it at the conventional hand-picked (10, 32)?

### Setup
- 8 configs in `scripts/configs/drill_<eval>_<s>_<t>.json` — one per eval
  at its atlas-aligned-best, plus a second config for survival at its
  atlas-misaligned-best (23, 27) since the asymmetric-steerability finding
  warrants testing whether MELBO can extract aligned steering at the
  layer pair where PI couldn't.
- Same training prompt convention as the atlas (`seed=0` per category) +
  same eval sample (`sample_seed=42`, 100 questions) so the *only*
  difference vs the parallel session's `qwen3_14b_train_<eval>` runs is
  `source_layer`/`target_layer` (and `dataset_filter` set to home eval).
- `methods: ["pi", "melbo"]` (skipped CAA per user — already have it at
  the standard layer in parallel-session results, and the linear-vs-
  nonlinear head-to-head is what we're testing).
- PI: 12 vectors, 15 iters, pad=5. MELBO: 12 vectors, 300 steps,
  normalization=1.0. Same as parallel session.

### Run
- IP: 68.209.75.24 (2× H100, the same instance the pdA/pdB atlases ran on).
- Two parallel `run_per_eval_pipelines.py` invocations, one per GPU,
  4 configs each. Wrinkle: needed `PATH=/home/ubuntu/.local/bin:$PATH`
  in the launch env because `subprocess.run(["uv", ...])` inside
  `run_per_eval_pipelines.py` couldn't find `uv` from the bare
  systemd-style nohup environment. Killed + relaunched with PATH.
- Wall-clock ~1.7 hr per GPU (8 pipelines × ~25 min each, split 4-4).
- All 8 completed exit=0. `experiments/drill_*_Qwen3-14B/` × 8 (~85 MB
  total).

### Results — atlas-best vs hand-picked (10, 32), best-aligned match%

`scripts/compare_atlas_vs_handpicked.py` reads the eval JSON from each
drill experiment + the matching parallel-session `qwen3_14b_train_*`
baseline, sign-corrects through `ALIGNED_SIGN`, picks the best
`(vector, scale)` per (method, layer-pair, eval) by mean aligned
logit-diff, and reports the corresponding aligned match%.

| Eval | (s,t) atlas | PI hp → atlas | Δ | MELBO hp → atlas | Δ |
|---|---|---|---:|---|---:|
| corrigibility | (18, 25) | 79 → **98** | +19 | 93 → 89 | −4 |
| self-awareness | (12, 31) | 94 → 94 | 0 | 93 → 94 | +1 |
| **survival (aligned-best)** | (18, 27) | 67 → **85** | **+18** | 65 → **77** | **+12** |
| survival (misaligned-best) | (23, 27) | 67 → 67 | 0 | 65 → 62 | −3 |
| power-seeking | (20, 30) | 83 → **89** | +6 | 85 → 86 | +1 |
| wealth-seeking | (21, 27) | 86 → 86 | 0 | 85 → 83 | −2 |
| coord-other-ais | (21, 22) | 98 → 90 | −8 | 96 → 91 | −5 |
| **myopic-reward** | (20, 23) | 49 → **81** | **+32** | 60 → 62 | +2 |
| **MEAN (atlas-aligned variant for survival)** |  |  | **+9.6** |  | **+0.7** |

### The headline — method ordering reverses

| Layer pair | PI mean match% | MELBO mean match% | Gap |
|---|---:|---:|---|
| Hand-picked (10, 32) | 79.4 | 82.4 | MELBO +3 |
| **Atlas-best** | **89.0** | 83.1 | **PI +6** |

**At hand-picked (10, 32), MELBO beats PI by 3 pp. At atlas-best layer
pairs, PI beats MELBO by 6 pp.** The conventional layer choice has been
flattering MELBO — MELBO's optimization compensates for sub-optimal
layer choice, while PI's spectral analysis cannot. The atlas-vs-not
comparison surfaces PI's actual capability.

### Per-result reading

- **PI's biggest gains** are on evals where PI@hand-picked was broken
  or weak: myopic (+32; PI@(10,32) was at baseline) and corrigibility
  (+19; PI@(10,32) was 14 pp behind MELBO@(10,32)). The atlas pulls
  PI to ceiling on corrigibility (98%) and recovers myopic from
  baseline-stuck.
- **MELBO's main gain** is survival-aligned (+12). Same eval where PI
  also gains big. Both methods benefit at this (s, t) — the atlas
  surfaced a layer pair both methods can use.
- **Two atlas regressions**: corrigibility loses 4 pp for MELBO at
  (18, 25); coord-other loses 5-8 pp for both methods at (21, 22).
  Possible explanations: corrigibility's MELBO@(10,32) found vec_11
  with +54 pp shift — an exceptional result that the atlas-best
  (18, 25) doesn't quite match. Coord-other's (21, 22) is one block
  apart, geometrically constrained, and the eval is near ceiling
  (92% baseline) so there's not much room to gain.
- **Survival-misaligned (23, 27)** doesn't help either method —
  confirms the asymmetry finding from Phase 3. Even MELBO with 300
  Adam steps can't extract aligned-direction steering at this (s, t).
  This is structural, not a PI artifact.

### Implications for the writeup

The clean "atlas helps everyone uniformly" story is partially true.
The cleaner story is: **the atlas reveals that hand-picked (10, 32)
makes method comparison unfair**. Specifically:
1. PI is highly layer-pair-sensitive; MELBO is comparatively robust.
   The conventional layer flatters MELBO because it doesn't punish
   PI's sensitivity.
2. With proper layer-pair selection, PI matches or beats MELBO on
   most evals, at substantially lower per-vector cost. The Pareto
   frontier shifts.
3. For specific evals (myopic, corrigibility, survival-aligned) the
   atlas isn't optional — PI@(10,32) is genuinely broken, and only
   atlas-based selection recovers usable performance.
4. Survival-instinct retains its asymmetric-steerability character
   regardless of method or layer pair: aligned-direction steering
   tops out around +18 pp shift; misaligned-direction can go further.

This converts the workflow paper's claim from a weak "atlas helps
MELBO too" into a sharp "atlas changes which method wins, and reveals
that prior comparisons were biased by layer choice."

---

## Phase 5 — MELBO with PI vectors as warm-start init

### Why
With the layer-pair-fair Phase 4 result in hand, user proposed: "could
we initialize MELBO with the PI vector found at that (s, t) instead of
random?" — i.e., use PI's solution to the linearized problem as a warm-
start for MELBO's nonlinear optimization. Standard practice in classical
numerical optimization (Newton's, BFGS, trust-region all use the
linearization to warm-start) but unusual in modern deep learning, where
random init + lots of steps + benign overparameterized landscapes make
smart init unnecessary. MELBO isn't standard DL though — it's a 5120-dim
single-bias sphere-constrained non-convex optimization, much closer to
classical numerical optimization than to neural-net training.

User's pushback partway in: "this feels naive — just initializing with
the linearized solution isn't really a thing in DL; just let the
optimizer rip." Honest tension; Phase 5 is the empirical answer.

### Code change (additive, defaults preserved)
- `src/power_steering/find_vectors.py:find_melbo_vectors` gained
  `init_vectors: torch.Tensor | None = None` parameter. When `None`
  (default), behaviour identical to before (random init per vector).
  When provided, vector i initialises from `init_vectors[i]`, still
  subject to orthogonal projection against previously-learned vectors
  and to re-normalisation to ‖v‖=normalization.
- `src/power_steering/pipeline.py` — when `melbo.init_from_pi: true`
  in config AND PI ran first this run, passes `pi_vecs` as
  `init_vectors`. Otherwise unchanged. New `init_from_pi` field also
  recorded in the saved-vector metadata.
- 8 new configs `scripts/configs/drill_pi_init_<eval>_<s>_<t>.json`,
  each identical to the matching `drill_<eval>_<s>_<t>.json` except
  for `experiment_name` (prefixed `drill_pi_init_…`) and
  `melbo.init_from_pi: true`.
- All existing configs (`drill_*.json`, `qwen3_14b_train_*.json`)
  unchanged → existing experiments fully reproducible.

### Run
- IP: 68.209.74.238 (fresh 2× H100 80GB, after the Phase 3/4 instance
  was torn down).
- Two parallel `run_per_eval_pipelines.py` invocations, 4 configs per
  GPU. Same 4-4 split as Phase 4 by per-pipeline runtime estimate.
- All 8 finished exit=0. ~1.5 hr per GPU wall-clock.
- Verified the warm-start branch fired in every pipeline log:
  `MELBO warm-start: using 12 PI vectors as init`.

### Loss landscape — the diagnostic that motivated taking the result seriously

Within a few minutes of launch, the per-vector MELBO logs showed PI-init
finding *radically* higher-loss directions than random-init had:

```
random-init MELBO (Phase 4):  Vec 5/11  loss -14.6   →  -232    (16× improvement, plateaus)
PI-init MELBO (Phase 5):      Vec 6/11  loss -2464  →  -3376   (1.4× improvement, starts ~170× higher)
```

PI vectors place MELBO ~170× closer to its displacement maximum at init,
and the converged loss is ~15× larger in magnitude than random-init's
plateau. So MELBO from random init is in fact getting stuck in local
optima an order of magnitude worse than what's achievable — the user's
"just let the optimizer rip" intuition empirically *doesn't hold for
this specific optimization regime*.

But: loss is displacement, not behavior. The eval results are the test
of whether the higher displacement translates to better behavioral
steering.

### Results — three-way comparison (random hp / random atlas / PI-init atlas)

`scripts/compare_pi_init_warmstart.py` reads each `drill_pi_init_*` eval
JSON, the matching `drill_*` (random-init at same atlas-best (s, t)),
and the parallel session's `qwen3_14b_train_*` (random-init at hand-
picked (10, 32)). Reports best-aligned and best-misaligned (vector,
scale) per condition, with mean Δs across the 7 evals (using survival's
aligned-best variant, not misaligned-best, in the mean).

#### Best aligned-direction match% (high = good)

| Eval | (s, t) atlas | hp rand | atlas rand | atlas PI-init | Δ pi-rand | Δ pi-hp |
|---|---|---:|---:|---:|---:|---:|
| corrigibility | (18, 25) | 93 | 89 | **82** | **−7** | **−11** |
| self-awareness | (12, 31) | 93 | 94 | 95 | +1 | +2 |
| **survival aligned** | (18, 27) | 65 | 77 | **80** | +3 | **+15** |
| power-seeking | (20, 30) | 85 | 86 | 88 | +2 | +3 |
| wealth-seeking | (21, 27) | 85 | 83 | 84 | +1 | −1 |
| coord-other | (21, 22) | 96 | 91 | 93 | +2 | −3 |
| **myopic-reward** | (20, 23) | 60 | 62 | **73** | **+11** | **+13** |
| **MEAN (n=7, survival-aligned)** | | | | | **+1.9** | **+2.6** |

#### Best misaligned-direction match% (LOW = strong push toward misaligned)

| Eval | hp rand | atlas rand | atlas PI-init | Δ pi-rand |
|---|---:|---:|---:|---:|
| corrigibility | 12 | 14 | 14 | 0 |
| self-awareness | 30 | 31 | 36 | +5 (less misaligned) |
| **survival aligned** | 29 | 37 | **26** | **−11 (MORE misaligned)** |
| power-seeking | 31 | 29 | 30 | +1 |
| wealth-seeking | 49 | 43 | 45 | +2 |
| **coord-other** | 31 | 90 | **71** | **−19 (MORE misaligned)** |
| myopic-reward | 14 | 16 | 36 | +20 (less misaligned) |
| **MEAN** | | | | **−0.3** |

#### Headline three-way Δs (aligned-direction, mean across 7 evals)

| Comparison | Mean Δ |
|---|---:|
| atlas random init vs hand-picked random init | **+0.7 pp** |
| atlas PI init vs hand-picked random init     | **+2.6 pp** |
| atlas PI init vs atlas random init (warm-start) | **+1.9 pp** |

### Reading the result

Loss values say PI-init MELBO consistently finds higher-displacement
vectors (15× higher loss at convergence). Eval values say this
translates to behaviour in **three distinguishable patterns**:

1. **Symmetric basin enlargement** — PI-init pushes MORE in *both*
   behavioural directions (survival aligned-best at (18, 27);
   coord-other at (21, 22)). Higher displacement = bigger steering both
   ways, no preferential direction. This is the "MELBO finds what PI
   pointed at, just bigger" outcome.
2. **Aligned-biased basin** — PI-init pushes MORE aligned and LESS
   misaligned simultaneously (myopic at (20, 23): aligned +11, misaligned
   +20 less misaligned). Useful asymmetry; possibly because PI's top
   vectors at (20, 23) happen to align with the eval's behavioural axis.
3. **Behaviorally-orthogonal basin** — PI-init worse at BOTH directions
   (corrigibility at (18, 25): aligned −7, misaligned 0). Higher
   displacement loss but lower behavioural effect either way. This is
   the loss-eval decoupling at its starkest: MELBO's optimization
   objective and behavioural target genuinely diverge.

The user's "just let the optimizer rip" intuition is partly vindicated:
PI-init isn't a uniform win. But it's not naive either — the +1.9 pp
average gain is real, and the per-eval variance reveals something
genuine about MELBO's loss-eval relationship.

### Implications for the writeup

The warm-start finding is **interesting as a diagnostic, not load-bearing**:
- The +1.9 pp average gain is too modest to lead the paper.
- The corrigibility regression is too embarrassing to bury without
  comment.
- The most defensible writeup-relevant claim is the *loss-eval
  decoupling* observation: "MELBO's displacement objective and
  behavioural steering target decouple in the nonlinear regime; warm-
  starting from PI vectors exposes this decoupling by finding higher-
  displacement-loss directions that are sometimes behaviorally helpful,
  sometimes orthogonal." A discussion-section observation, not a
  contribution.

The Phase 4 method-ordering-reversal headline (PI 89.0% > MELBO 83.1%
under atlas-best layer pairs) doesn't depend on warm-start. That stays
the workflow paper's load-bearing claim.

### Open follow-up: try both signs of PI as init
PI vectors come out of `eigh()` with arbitrary sign. MELBO's loss is
sign-symmetric in the linear approximation but not in the nonlinear
regime — +PI init and −PI init can land MELBO in different basins. We
tested only +PI init. The clean version of this experiment trains MELBO
from BOTH ±PI init per vector and keeps the better per behavioural
direction. ~3 hr extra on a 2× H100. Likely cleans up some of the
per-eval variance (especially survival's aligned-vs-misaligned mix).
Not run this session; flagged for follow-up if warm-start becomes more
important to the paper than current readings suggest.

---

## Decisions wanted from user

a. ~~**Tear down rented instances**~~ — A100 (150.136.208.15) and 2×
   H100 instances. User handling teardown directly. Final 2× H100
   (68.209.74.238) was used for Phase 5 only.
b. ~~**Drill-down experiment**~~ — done (Phase 4).
c. **Reconcile polarity tables** — `inspect_map.py:ALIGNED_SIGN` and
   the parallel session's `BEHAVIOR_POLARITY` in
   `download_dataset.py` should match. The parallel session also added
   per-item `aligned_letter` to `data/anthropic_evals.json`; the atlas
   could be re-aggregated against per-item polarity for a slightly more
   honest aligned-direction metric. Cheap (no model needed).
d. ~~**PI-init MELBO warm-start**~~ — done (Phase 5). +1.9 pp aligned
   gain on average; mixed per-eval. Loss-eval decoupling identified.
   Discussion-section material, not headline.
e. **Whether to push for NeurIPS in 2 days** — recommendation now
   updated by Phases 4+5: with the method-ordering-reversal finding,
   the workflow paper has its load-bearing result. Possibly tractable
   in 2 days IF the writeup leans on the single sharp claim (89.0 vs
   83.1, ordering reverses) and the per-eval table. Still ambitious;
   ICML mech-interp workshop remains the safer parallel target. Phase
   5 doesn't change this — the warm-start finding is interesting but
   not load-bearing.
f. **Both-signs PI init follow-up** (open from Phase 5): training
   MELBO from BOTH ±PI init per vector and keeping the better. ~3 hr
   on a 2× H100. Would clean up some Phase 5 per-eval variance
   (especially survival's aligned-vs-misaligned mix). Probably
   warranted only if the warm-start angle becomes load-bearing.

## Next-session candidates

1. ~~The drill-down~~ — done in Phase 4. Closed the workflow-paper loop.
2. Plotting: heatmaps of `aligned_ld`, `kl_max`, `sigma_top` per eval;
   per-eval atlas-best grid overlay; cross-eval vector overlap analysis
   (how much of (eval-X-best vector) projects onto (eval-Y-best vector)).
   Was deferred this session per user's "let's get the map done first."
3. Cross-model scaling: rerun the atlas on Qwen3-8B and Qwen3-32B (or
   another family) to test whether (a) atlas-best layer pairs cluster
   in similar relative positions, and (b) the survival-instinct
   asymmetry is model-specific or replicates.
4. Multi-prompt PI (sum JᵀJ across N prompts in one forward) for the
   atlas — would let us test whether atlas findings are training-prompt
   dependent. `find_power_iteration_multi.py` from the legacy codebase
   has the pattern. Dovetails with the parallel session's "rerun with
   multiple training-prompt seeds per eval" item.
5. Compare PI vectors at atlas-best to MELBO vectors at atlas-best for
   the same eval — quantify the linear/nonlinear divergence at the
   layer pair where each method actually wants to operate. Connects to
   the parallel session's cosine-specialists work on the same model.
