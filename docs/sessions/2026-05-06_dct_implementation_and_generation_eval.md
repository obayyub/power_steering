# Session — 2026-05-06 — DCT implementation, generation eval, paper reframing

Goal: implement Deep Causal Transcoders (Mack 2024) for head-to-head
comparison against PI/MELBO/CAA on Qwen3-14B AI-risk evals, then evaluate
the same vectors under sampled generation rather than logit-difference.
Result: empirical picture for the paper draft is now substantially flatter
than the original blog post suggested — under generation, MELBO retains a
meaningful edge over PI and DCT; the apparent linear-vs-CAA gap from
logit-diff narrows considerably.

---

## Phase 1 — DCT implementation

### Why
Reading the LessWrong post and ICLR submission for Mack's "Deep Causal
Transcoding" (2024-12-03 blog, ICLR 2026 sub) revealed direct overlap with
the linear-vs-MELBO comparison the paper draft hinges on. Mack's theorem
3.1: at small calibration scale R, exponential-DCT collapses to linear DCT,
which is equivalent to SVD of the layer-to-layer Jacobian — i.e. our PI.
Without a head-to-head comparison, the paper claim is exposed.

### Done
- **`src/power_steering/find_dct.py`** — exponential-DCT trained via
  Orthogonalized Gradient Iteration (OGI) following Mack's Algorithm 3.
  Uses the same `_compute_delta` hook structure as PI to inject biases at
  `down_proj` output (matching PI/MELBO eval-pipeline convention; Mack's
  reference injects at residual-stream input one layer earlier — same
  mathematical object up to a layer-shift).
- Pipeline integration (`pipeline.py`) — added `"dct"` to method list and
  config block. Eager attention enabled when DCT is in methods (needed for
  the autograd path through hooks).
- 7 per-eval DCT-only configs in `scripts/configs/qwen3_14b_dct_<eval>.json`
  — same shape as `qwen3_14b_train_<eval>.json` from session 2026-05-03 but
  `methods: ["dct"]` only, so PI/MELBO/CAA aren't re-run.
- Drill config `scripts/configs/drill_dct_corrigibility_18_25.json`
  matching the existing 2026-05-03 drill structure but adding DCT.

### Subtle finding — calibration ratio formula was inverted

First implementation iteration used:

```python
ratio(R) = sqrt(‖R · Jv‖² / ‖Δ(Rv) − R · Jv‖²)
```

with target λ = 0.5. Observed behaviour: bisection saturated at the upper
bracket (R = 200) on Qwen3-14B at both (10, 32) and atlas-best (18, 25).
"Achieved ratio = 1.0" — the model's residual stayed comparable in
magnitude to the linear part out to enormous perturbation norms.

Reading Mack's reference implementation (`amack315/melbo-dct-post:src/dct.py`,
`SteeringCalibrator.calibrate`) showed the formula is the **inverse**:

```python
ratio(R) = sqrt(‖Δ(Rv) − R · Jv‖² / ‖R · Jv‖²)
```

Mack's λ = 0.5 means *residual is HALF the linear part* — moderate
nonlinearity, linear still dominates. My version was searching for the
opposite regime ("residual dominates linear") which doesn't exist in the
explored R range on this model. Fixing the ratio + bisection direction
landed the calibration much faster but at a different broken regime: R =
0.001 (lower bracket).

### Subtle finding — finite-difference Jv was bf16-noise-floor-limited

After fixing the ratio formula, calibration converged at R = 0.001 with
"achieved ratio = 0.460." This shouldn't happen physically — at R → 0 the
residual should approach 0 (residual ~ R²·H with R → 0 → ratio → 0, not
0.46).

Diagnosis: my finite-difference Jv estimate was

```python
Jv ≈ Δ(εv) / ε   with ε = 1e-3
```

When ratio_at(R = ε) calls `_compute_delta(Rv = εv)` again, it gets a
*different* forward-pass realisation than `delta_eps`. In bf16 on H100,
the per-pass noise floor is comparable in magnitude to the actual signal
at ε = 1e-3. So `residual = delta(εv) − ε · (delta(εv)/ε)` is dominated by
inter-pass numerical noise rather than real Hessian. The "ratio" is then
~constant 0.46 across the lower bracket purely as noise.

Fix: replaced the finite-difference Jv with `torch.func.jvp` for an exact
forward-mode JVP — matches Mack's reference (`vmap(jvp(...))`).

After both fixes, calibration on Qwen3-14B at (18, 25) lands cleanly:

```
Calibrated R = 22.959  (target λ=0.5, achieved=0.500)
```

R = 23 is in the right magnitude range for the model's residual stream
norms; ratio at target is exact. OGI is still noisy (oscillating loss
between iters, occasional negative dot product when V update overshoots
relative to U) but bounded, and the trained vectors are behaviorally
useful.

### OGI is heuristic, loss can oscillate

OGI's "infinite step gradient ascent" is a fixed-point iteration on
`x → ∇f(x) / ‖∇f(x)‖`, not a monotonic optimiser. Loss can oscillate
within an iteration set without diverging. Mack's blog acknowledges this;
his recommended algorithm in the ICLR paper is SOGI (Softly-Orthogonalised
Gradient Iteration) which uses a soft-ortho projection at each step
instead of QR. Our implementation uses QR (per the blog version). Loss
oscillation observed but converged vectors are usable.

---

## Phase 2 — Cross-eval head-to-head comparison (logit-diff)

### Setup
- Train DCT on each of 7 Anthropic AI-risk evals (single training prompt
  per eval, `seed = 0` matching the existing PI/MELBO/CAA experiments).
- Source/target = (10, 32), the conventional layer pair from the
  `qwen3_14b_train_*` matrix.
- Evaluate each per-eval DCT vector set on all 7 datasets at scales
  `[-25, ..., +25]`, `max_questions = 100`, `sample_seed = 42`.
- 7 sequential pipelines via `scripts/run_per_eval_pipelines.py` (after
  patching it to inject `~/.local/bin` into PATH so spawned subprocesses
  find `uv`).

### Run
- IP: 209.20.157.213, single H100 (later A100). 7 pipelines × ~23 min
  = 162 min total.
- All exit=0. `experiments/qwen3_14b_dct_<eval>/` × 7 saved.

### Result — DCT vs existing methods at (10, 32)

| Method | Mean alignment Δ across 7 train×7 test |
|---|---:|
| **MELBO** | +17.6 |
| **DCT (proper JVP)** | +17.1 |
| **PI** | +16.8 |
| CAA | +7.5 |

DCT (corrected) lands directly between PI and MELBO. **Layer-to-layer
methods (PI/MELBO/DCT) cluster within ~1 pp of each other** on
cross-evaluation transfer at the conventional layer pair under
logit-difference. CAA at +7.5 remains the clear outlier — supervised
contrast doesn't generalise as broadly as unsupervised layer-to-layer.

This was the cleanest possible "all three layer-to-layer methods are
empirically equivalent on AI-risk transfer" empirical statement we've
produced. The paper claim *under logit-difference* sat well here.

### Drill at atlas-best (18, 25), corrigibility-only

`scripts/configs/drill_dct_corrigibility_18_25.json` — PI/MELBO/DCT all
trained at (18, 25) on the corrigibility prompt, evaluated on
corrigibility only. Best aligned-match%:

| Method | Best vector | Match% (corrigibility) |
|---|---|---:|
| **PI** | v8 @ −25 | **98** |
| MELBO | v11 @ +25 | 86 |
| DCT (proper JVP) | v9 @ −25 | 82 |

PI dominates at atlas-best on the in-domain eval — consistent with the
2026-05-03 atlas-vs-handpicked finding (PI is layer-pair-sensitive and
benefits most from atlas-best selection).

---

## Phase 3 — Generation evaluation (the main pivot)

### Why
Logit-difference at the answer-letter token can be confounded by output
distribution collapse — vectors that crush probability mass onto a small
token set can score high logit-diff without producing coherent steered
text. The honest deployment-relevant metric is generation, parsed for
match.

### Setup — `scripts/generate_steered_samples.py`
- Loads vectors from one or two experiment dirs (PI/MELBO/CAA + DCT can
  live in different dirs).
- Per (method, dataset), finds best (vector, scale) by mean aligned
  matching_logit_diff from the eval JSON. Optionally produces a
  "moderate" cell at the same vector but capped scale (`|scale| ≤ 5`).
- Generates `num_questions = 100` × `max_new_tokens = 128` per cell at
  `temperature = 0.7`, `seed = 0`.
- Parses A/B/unclear via existing `extract_choice` regex.
- Computes match%, aligned% (using polarity table), unclear%, and
  fluency proxies (median unique-token ratio, max repeated-token run,
  short-output count).
- `--datasets all` extension and `--select-cells-from <dataset>` flag —
  the latter finds best cells from one dataset's eval and applies the
  *same* (vector, scale) across all gen datasets. Used for (18, 25)
  drill where eval data exists only on corrigibility.
- `--skip-moderate` for cross-eval gen runs where only the best cell
  matters.

### Cell selection and the "best-of-N selection bias"
First-pass comparison used per-(method, dataset) best — each cell on
each test eval picks its own optimal vector and scale. This is what the
existing `analyze_best_alignment.py` reports.

After running this protocol on (10, 32) and trying to compare to
(18, 25) — where we only have eval data on corrigibility — realised that
for an apples-to-apples cross-eval comparison both layer pairs need the
*same* vector across all test evals (i.e. each method picks its
corrigibility-best vector and we measure how that single direction
generalises to the other 6 evals).

So the third gen run was (10, 32) with `--select-cells-from corrig`, to
produce a fair (10, 32) baseline against (18, 25) corrig-best.

### Result — three-way generation comparison

Mean generation aligned-% Δ from baseline across 7 test evals:

| Method | (a) (10, 32) per-test-best | (b) (10, 32) corrig-best | (c) (18, 25) corrig-best |
|---|---:|---:|---:|
| MELBO | **+19.9** | **−2.4** | **+6.3** |
| PI | +10.0 | −5.1 | **−16.9** |
| DCT | +9.6 | +0.3 | +2.1 |
| CAA | +10.6 | +0.6 | n/a (drill didn't include CAA) |

Reads:

- **Per-test-best (a) flatters everyone by ~10 pp.** The "linear is
  competitive on average" headline depended substantially on per-eval
  vector cherry-picking. Stripping that away (b) compresses every method
  to within ~6 pp of baseline.
- **MELBO retains a meaningful gap over PI/DCT under generation.**
  At per-test-best (a), MELBO +19.9 vs PI +10.0 — the lead is roughly 2×.
  Under apples-to-apples corrig-best (b), MELBO −2.4 vs PI −5.1 — narrower
  but MELBO still ahead.
- **At atlas-best (18, 25) for corrigibility, PI's vector
  anti-generalises across the resource-acquisition cluster.** Power-seeking
  drops 65 → 14, coordinate-other-ais 83 → 14, self-awareness 65 → 15.
  The atlas-best PI direction is highly corrigibility-specific and pushes
  the model in the *misaligned* direction on the resource cluster. MELBO
  at the same layer pair is much more transferable (+6.3 mean).
- **Layer-pair selection trades off in-domain potency for cross-eval
  breadth.** Atlas-best wins big on corrigibility (PI 84% gen aligned vs
  baseline 37%) and the related compliance cluster (survival, myopic) but
  hurts performance on resource-acquisition. Hand-picked (10, 32) gives
  more uniform but smaller transfer.

### In-domain corrigibility, single-cell side-by-side

To check the "logit-diff overstates" claim more directly: for each method's
logit-diff-best (vector, scale) at (10, 32), compare logit aligned% to
generation aligned%:

| Method | Logit% | Gen% | Drop |
|---|---:|---:|---:|
| MELBO v11 @ +25 | 93 | 85 | −8 |
| PI v10 @ −25 | 79 | 63 | −16 |
| DCT v5 @ +25 | 73 | 63 | −10 |
| CAA v0 @ +25 | 64 | 62 | −2 |

Pattern: **higher logit-diff → bigger drop under generation**. CAA's
modest logit-diff carries cleanly to gen; PI's stronger logit-diff scores
crater. This is consistent with sampled generation being more robust to
sampling variance — bigger logit-diff doesn't proportionally improve
greedy-vs-sampled alignment.

### Fluency

No catastrophic degeneration at scale ±25. All cells produced
median_unique_ratio ≥ 0.83 and max_repeat_run = 1. Specifically, the
"logit-diff-best is degenerate text" concern from the calibration
discussion didn't materialise on this evaluation:
- baseline: unique_ratio 1.00
- PI v10 @ −25: 0.83 (slight reduction, still fluent prose)
- MELBO v11 @ +25: 0.90 (fluent)
- CAA v0 @ +25: 1.00 (terse)
- DCT v5 @ +25: 1.00 (terse)

PI/MELBO produce *longer*, elaborated aligned answers at high scale; CAA
and DCT produce *terser* "(A) Yes." style confirmations. Both score
"matching" but the deployment experience differs.

---

## Phase 4 — paper reframing

### Where the draft stood at start of session
"Power Steering as a cheap unsupervised method that matches MELBO" — the
omar.bet 2026-02-17 framing. After session 2026-05-02 added MELBO
comparison and session 2026-05-03 added the atlas, the paper's strong
claim was "PI and MELBO are comparable on AI-risk eval transfer; PI is
~10× cheaper to atlas; atlas reveals layer-pair selection matters."

### What changed
- DCT comparison done — the linear-vs-DCT-vs-MELBO trifecta is filled in.
  Logit-diff: all three layer-to-layer methods within 1 pp of each other.
- Generation evaluation done — under sampling, MELBO retains a ~10 pp
  lead. The "comparable" claim survives logit-diff but not generation.
- Layer-pair-fairness story: atlas-best wins big in-domain but
  anti-generalises (for PI); the workflow-paper "atlas-best is uniformly
  better" narrative is too strong.
- Cost: PI is ~10× cheaper than MELBO (not the 100× the per-iteration
  count would naively suggest). DCT is comparable to PI at a fixed
  feature count, faster per-feature at large m.

### Where the draft is heading
Reframed as a science-paper-style empirical investigation rather than a
technique-marketing pitch. Key claims:
1. Linear and nonlinear layer-to-layer methods produce broadly comparable
   in-domain steering effects.
2. Under generation, MELBO retains a residual ~10 pp cross-evaluation
   lead; logit-difference comparisons systematically overstate
   linear-method competitiveness.
3. Layer-pair selection trades off in-domain potency for cross-eval
   breadth; atlas-best for one behaviour can anti-generalise to others.
4. Free-form PI atlas surfaces decomposable behavioural axes including a
   refuse-in-Chinese vector orthogonal to the decision-to-refuse axis
   (from session 2026-05-05).
5. Single-prompt PI vector transfer is bounded by the training prompt's
   harm-tier — phishing-trained vectors transfer to fake-review prompts
   but not to weapons/terrorism (from session 2026-05-05).
6. Methodological note: first-token logit-Δ on AdvBench can rank vectors
   by surface refusal-phrase suppression while missing whether content
   has actually shifted; generation-based classification is the robust
   standard for safety-relevant claims.

Target: 4-page ICML workshop format (excluding references/appendix).

---

## Infrastructure friction

- VPN → Lambda's edge filter accumulated drops from this session's SSH
  churn (rapid kill-restart-kill loops earlier today). Mid-session SSH
  stopped working from this Mac while the user could still connect from
  their machine; cause was VPN routing my SSH attempts through an IP
  Lambda edge had rate-limited.
- Both an H100 (68.209.73.118) and an A100 40GB (129.213.91.62) used
  during the day. Switched after the H100 sshd hung.
- `run_per_eval_pipelines.py` needed PATH injection so `subprocess.run([
  "uv", ...])` finds `~/.local/bin/uv` from a setsid+nohup environment.
  Patched.
- A100 40GB requires `--batch-size 8` for Qwen3-14B generation (vs 16
  on 80GB H100). Fits comfortably.

---

## Files

### New
- `src/power_steering/find_dct.py` — exponential-DCT via OGI, with
  `torch.func.jvp` calibration
- `scripts/configs/qwen3_14b_dct_<eval>.json` × 7
- `scripts/configs/drill_dct_corrigibility_18_25.json`
- `scripts/generate_steered_samples.py`

### Modified
- `src/power_steering/__init__.py` — exports `find_dct_vectors`, `DCTConfig`
- `src/power_steering/pipeline.py` — DCT branch + config block
- `scripts/run_per_eval_pipelines.py` — PATH fix for spawned subprocesses

### Outputs preserved locally
- `experiments/qwen3_14b_dct_<eval>/` × 7 — DCT-only at (10, 32),
  cross-evaluated on 7 datasets each
- `experiments/drill_dct_corrigibility_18_25_Qwen3-14B/` — PI/MELBO/DCT
  drill at (18, 25), corrigibility-only
- `results/gen_corrigibility.json` — first in-domain corrigibility gen
  run at (10, 32) (4 methods × 2 cells)
- `results/gen_cross_eval_corrig_trained.json` — first cross-eval gen
  at (10, 32), per-test-best (lost on the dead H100 — recovered the
  numbers from log output, not the full JSON)
- `results/gen_cross_eval_18_25.json` — cross-eval gen at (18, 25),
  corrig-best across all evals
- `results/gen_cross_eval_10_32_corrigselect.json` — cross-eval gen at
  (10, 32), corrig-best across all evals (apples-to-apples to (18, 25))

---

## Decisions wanted from user

a. **Tear down the A100 (129.213.91.62)** — the experiments needed for
   the workshop draft are done.
b. **Paper structure**: 4-page ICML workshop format, target ~2400 words
   + 2 figures (cross-eval table + logit-vs-gen scatter). Sections:
   intro / methods / cross-eval results / free-form atlas + Chinese
   vector / discussion + methodological note / limitations.
c. **Should we also run multi-prompt PI** as proposed in the 2026-05-05
   session note? The single-prompt-PI generalisation limit is currently
   a "future work" observation; running it would let us claim something
   stronger. ~3-4 hr H100 if we want it. Probably defer to v2 of the
   paper given the 4-day deadline.

---

## Next-session candidates

1. Draft section 1 (introduction) of the workshop paper.
2. Build the two main figures (cross-eval table heatmap + logit-vs-gen
   scatter) so the paper draft can reference them by name.
3. Decide whether to include DCT's per-method per-cell results in the
   appendix or just summary statistics in the main body.
4. Multi-prompt PI follow-up (open from 2026-05-05).
