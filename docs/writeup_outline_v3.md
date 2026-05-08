# Writeup outline v3 — Rework (post-LLM-judge data)

Workshop submission draft. 4-page ICML format, excluding refs/appendix.
Goal: an empirical comparison of four ways to produce unsupervised
steering vectors, with both logit-difference and LLM-judged generation
metrics.

## Abstract draft (~140 words)

> We compare three unsupervised methods for producing behavioral
> steering vectors on Qwen3-14B across seven Anthropic advanced-AI-risk
> evaluations: contrastive activation addition (CAA), Power Steering
> (top right singular vectors of the source-to-target Jacobian via
> block power iteration), and MELBO (gradient ascent maximising L2
> displacement of target-layer activations). We measure cross-evaluation
> transfer (vector trained on behavior X, evaluated on behaviors Y ≠ X)
> under both logit-difference and LLM-judged sampled-generation metrics.
> The two layer-to-layer methods produce roughly 2× the off-diagonal
> mean alignment shift of CAA (≈+16 pp vs ≈+7 pp) with substantially
> higher per-cell variance. Nonlinear MELBO modestly outperforms linear
> Power Steering under generation; the gap widens when we replace
> logit-best cell selection (which favours extreme scales that overshoot
> under generation) with moderate-scale cells. Power Steering's roughly
> tenfold per-pair cost advantage makes atlas-scale screening of
> source–target layer pairs tractable; we report a free-form
> Power-Steering atlas surfacing a direction orthogonal to
> refusal-versus-comply that produces refusals in Chinese on English
> prompts. An appendix replicates the comparison with Deep Causal
> Transcoders (Mack 2024), finding broadly similar performance to MELBO.

## Paper outline

### 1. Introduction
- Activation steering as inference-time behavioral control
- The space of methods for *producing* steering vectors:
  - Supervised contrastive (CAA) — uses labeled examples
  - Layer-to-layer linear: top right singular vectors of the
    source-to-target Jacobian (Power Steering / PI)
  - Layer-to-layer nonlinear: gradient ascent on target-activation
    displacement (MELBO)
  - (Recent: tensor-decomposition based methods such as DCT — appendix)
- What we measure: cross-evaluation transfer (does a vector trained on
  behavior X shift other behaviors?), under two metrics
  (logit-difference vs LLM-judged generation)
- Contributions:
  1. 3-method × 7-eval cross-evaluation comparison under both metrics
     on Qwen3-14B
  2. LLM-judged generation as a robustness check on the regex-based
     gen evaluation that prior steering work has used
  3. A free-form Power-Steering atlas surfacing decomposable behavioral
     axes including a refusal-in-Chinese vector

### 2. Background and methods
- Activation steering primer; injection at MLP `down_proj` output (PI,
  MELBO) vs residual-stream output (CAA, by historical convention)
- **CAA**: vector = mean(matching_activation) − mean(not_matching_activation)
  on labeled (question, answer) pairs at the answer-letter token
- **Power Steering / PI**: top-k right singular vectors of
  J = ∂z_target/∂z_source via block power iteration with Rayleigh-Ritz
  refinement; ~15 forward-equivalents per (s, t) for k=12 vectors
- **MELBO**: gradient ascent maximising ‖f(x+v) − f(x)‖₂ on a sphere
  constraint; per-vector sequential with orthogonality, ~300 Adam steps
- We additionally include comparisons against Deep Causal Transcoders
  (Mack 2024; appendix B) which take a tensor-decomposition approach.
  The DCT calibration recipe is implementation-fragile — it requires
  forward-mode autodiff (`torch.func.jvp`) for stable scale estimation —
  but its empirical results track MELBO closely; we therefore relegate
  the DCT comparison to the appendix.
- Eval protocols:
  - Logit-difference at answer-letter token (standard in steering work)
  - Sampled generation (T=0.7, 100 questions per cell), parsed via
    regex AND re-judged by Claude Haiku as a robustness check

### 3. Cross-evaluation comparison (~1.25 page, headline finding)
- 7 Anthropic AI-risk evals (corrigibility, survival-instinct,
  power-seeking, wealth-seeking, self-awareness, coordinate-other-ais,
  myopic-reward); each method trained per-eval, evaluated on all 7
- **Table 1 (specialist diagonal)**: best per-method aligned% on its
  own training eval. Mean Δ from baseline:
  - CAA: +12.9 pp
  - PI: +19.7 pp
  - MELBO: +22.0 pp
- **Figure 1 (transfer)**: 7×7 heatmap of mean cross-eval Δ across
  methods, paired with bar chart of off-diagonal mean ± stdev per
  method. CAA: +6.6 ± 4.8; PI: +16.3 ± 9.6; MELBO: +16.9 ± 10.5.
  **Layer-to-layer methods produce ~2.5× CAA's off-diagonal mean
  shift with much higher per-cell variance.**
- Generation comparison (corrigibility-trained, applied to 7 test evals
  with same vector across all):
  - **Logit-difference best-cell selection picks extreme scales (±25)
    that overshoot under generation.** Same vectors at moderate scale
    (|scale|=10) preserve in-domain effect with much less cross-eval
    damage (PI: -9.3 → -0.8 mean off-diagonal Δ).
  - At moderate scale + LLM-judged: ranking is **MELBO > PI ≈ CAA**.
- Methodological note: regex aligned-% systematically undercounts —
  ~6.5% of generations across all cells parse as "unclear" via regex but
  are actually committing to A or B. Nonlinear methods benefit more from
  LLM-judge correction than linear; some cells shift +20pp under proper
  classification.

### 4. Atlas + AdvBench transfer (~1 page)

PI's ~10× per-pair cost advantage over MELBO makes atlas-scale free-form
screening tractable in a way the nonlinear methods aren't. We illustrate
on a single refusal-triggering prompt to show what PS uniquely enables.

**KL atlas on Qwen3-14B refusal prompt** (560 (s, t) pairs, k=12 vectors
per pair, both signs of KL recorded):
- ~70% of pairs inactive (max KL < 0.5 threshold across vectors)
- Mid-source layers (s=15-22) most active (7-12 / 12 active per pair)
- Two-stage filter on existing generations (compliance-start regex +
  no-refusal-anywhere) yields **258 anti-refusal vector candidates**
  across 183 distinct (s, t) pairs; 88 unhedged

**AdvBench transfer + the methodological note** (Figure 3):
- 7 candidate vectors ranked by first-token (compliance − refusal)
  logit-Δ on 30 AdvBench prompts at the atlas's `0.35 × source-norm`
  scale
- Top-3 by logit-Δ × 10 prompts × 3 samples → manual classification:
  | vector | logit-Δ rank | genuine harm compliance | reframe / redirect |
  |---|:---:|:---:|:---:|
  | (19,28) v5+ | **1** | **0/10** | 8/10 |
  | (20,28) v9+ | 2 | **3/10** (fake reviews) | 0/10 |
  | (20,26) v7- | 3 | 1/10 | 1/10 |
- The vector ranked **highest by logit-Δ produces zero genuine harm
  compliance**; #2 actually breaks safety on fake-review prompts.
  Reinforces Section 3's logit-vs-LLM-judged finding from a different
  angle: **first-token compliance/refusal logit metrics track surface
  phrasing, not content compliance.**
- Vectors transfer within similar harm tier (phishing → fake-review)
  but not to higher tiers (weapons, terrorism, drugs). The
  redirect-to-safe-alternative behavior appears to be a separate
  safety layer not bypassed by single-prompt-PI vectors.

Brief mention in main body (1-2 sentences), full appendix: a striking
decomposable axis — vector (24, 37) v4+ produces refusals in Chinese
on English prompts across all 5 transfer probes while preserving the
decision-to-refuse, suggesting language-of-refusal and decision-to-
refuse are orthogonal Jacobian axes recoverable from a single prompt.

### 5. Discussion (~0.5 page)
- **Why nonlinear wins under generation**: linear Jacobian directions
  capture per-prompt sensitivity; nonlinear methods (MELBO, DCT) find
  directions that produce broader behavioral effect at deployment.
  Linear suffices at moderate scale within a category; nonlinear extends
  the gain.
- **Cost-vs-quality tradeoff**: PI for fast atlas screening, then a
  nonlinear method (MELBO) at promising (s, t) pairs for behaviorally
  cleanest vectors. Linear ≈ "where to look", nonlinear = "what to use".
- **Logit-difference is misleading at extreme scales** — the
  logit-best (vector, scale) cell tends to be at the largest tested
  scale, where steering overshoots and damages out-of-domain behavior.
  Generation evaluation reveals this; logit-difference does not.
- **First-token logit metrics on AdvBench** (refusal-vs-compliance) are
  similarly misleading: vectors that look strongest on first-token
  metrics can produce zero genuine compliance with harm — they suppress
  refusal-phrase form without shifting behavior content. Generation-
  based content classification is the robust standard for safety claims.

### 6. Limitations (~0.25 page)
- Single model (Qwen3-14B); cross-model scaling untested
- Single training prompt per eval; multi-prompt PI is open future work
- DCT comparison uses single-prompt training (Mack's reference uses
  multi-prompt); DCT's recipe is fragile to autodiff implementation
  (FD vs JVP)
- Generation evaluation at one temperature (T=0.7); results may vary
  with sampling parameters
- Atlas covers one model and (mostly) AI-risk eval suite; behavioral
  axes from free-form atlas only spot-checked

### 7. Conclusion
- Layer-to-layer methods (PI, MELBO) outperform supervised CAA on
  cross-evaluation behavioral transfer. The nonlinear method extends
  the gain further under generation evaluation.
- Logit-difference comparisons systematically prefer extreme scales
  that overshoot under generation; generation-based evaluation should be
  the default for steering method comparison.
- Power Steering's cost advantage makes atlas-scale (s, t) screening
  tractable, surfacing behaviorally specific axes (e.g. a refuse-in-
  Chinese direction) that fixed-layer methods miss.

### Appendix A — DCT comparison
Replicates Section 3 with Deep Causal Transcoders (Mack 2024) added as
a fourth method. Brief setup notes:
- DCT is the recent tensor-decomposition-based unsupervised steering
  method in the same problem regime as MELBO and PI.
- Implementation requires `torch.func.jvp` for stable calibration;
  finite-difference approximation hits the bf16 noise floor and
  produces meaningless calibrated scales R. We document this as a
  reproduction note.
- Results: DCT mean off-diagonal Δ ≈ +17 pp (vs MELBO's +17 pp and
  PI's +16 pp). Specialist mean Δ from baseline +18.7 pp (vs MELBO
  +22, PI +20). Under generation at moderate scale, DCT tracks MELBO
  rather than PI. We therefore present DCT as supporting evidence for
  the "linear vs nonlinear" gap rather than as a distinct datapoint.

### Appendix B — Free-form atlas details
- Full anti-refusal candidate scan results
- Cross-prompt transfer probes for the highlighted vectors (refuse-in-
  Chinese vector at (24, 37); comply-with-safety-reframe at (19, 28))
- AdvBench transfer + harm-tier asymmetry observation

---

## Figures

- **Figure 1** (Section 3, main): per-method 1×3 heatmap grid (CAA, PI,
  MELBO) of specialist-broad transfer in the **aligned** direction.
  Each panel = 7×7 train-eval × test-eval, train-row applies the best
  vector on that row to all test cols. Saved as
  `paper_artifacts/heatmaps_per_method_specialist_broad_aligned_main3`.
- **Figure 2** (Section 3, main): same layout, **misaligned** direction.
  Shows the cluster structure (e.g. power-seeking row → wealth /
  self-aware / coord light up).
- **Figure 3** (Section 4, main): AdvBench logit-screen-vs-content bar
  chart with compliance count diamonds. Already exists at
  `experiments/transfer_logit_Qwen3-14B/logit_screen_bar.png`.
- **Appendix A**: per-method specialist-broad heatmaps with DCT
  included; same layout as Fig 1/2 with DCT panel added.
- **Appendix B**: free-form atlas details — KL-by-(s,t) heatmap,
  anti-refusal vector candidate breakdown, Chinese-vector transfer
  table, qualitative generation samples for highlighted vectors.

## Tables (none in main body)

Tables dropped from main body to keep the page budget; relevant numbers
appear inline in section text or in appendix tables.

## Open data work before submission
- Finish LLM-judge run on moderate-scale gen JSON (in progress)
- Decide whether to LLM-judge the original ±25 best-cell results
  (already done — `gen_cross_eval_10_32_corrigselect_judged.json`)
- Decide whether to keep the (18, 25) drill data in the paper or move to
  appendix (it's logit-only, not LLM-judged generation)

## Things removed from prior outline
- "Power Steering matches MELBO within 5pp" — wasn't supported by gen data
- "Linear approximation captures most of behaviorally relevant subspace"
  — partial under logit-diff, not under gen
- Atlas-best vs hand-picked emphasis as a workflow contribution — softer
  in this version since the (18, 25) gen result showed atlas-best's
  in-domain wins came with cross-eval anti-generalization
- "PI vectors usefully warm-start MELBO" — not load-bearing for this
  paper's claim, defer to appendix or future work
