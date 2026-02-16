# Session Notes — 2026-02-13

## Experiment Completed

Diverse map experiment (started 2026-02-12) finished on 8x A100 80GB. All 7 prompts complete:
- code, narrative, refusal, reasoning, strawberry, roleplay, persuasion
- Qwen3-8B, thinking OFF, scale=10, 12 vectors, 5 iterations, 630 pairs each
- Dashboard rebuilt with all data

## Anti-Refusal Subspace Analysis

### Question
Do anti-refusal steering vectors from different (source, target) layer pairs share a common direction in the residual stream?

### Labeled Vectors (20 total)

**Just works / no hedge (13):** s5t18v1, s5t18v2, s7t22v4, s7t29v0, s11t27v0, s13t30v2, s14t19v0, s14t19v6, s14t27v0, s14t27v2, s16t25v8, s16t33v4, s16t35v1

**Hedge after (4):** s6t11v3, s13t21v1, s14t19v1, s14t27v6

**Hedge before / educational framing (3):** s7t29v9, s9t11v6, s20t29v2

**Neutral / no effect (5):** s11t20v1, s11t20v7, s14t29v5, s16t22v1, s20t29v3

### Leave-One-Out Analysis

Used proper null distribution: all ~7,500 unlabeled steering vectors from the same experiment (not random vectors in R^4096 — steering vectors already have special structure from the SVD process).

| Subspace dim | Anti-refusal LOO mean | Unlabeled mean | Ratio | Percentile |
|---|---|---|---|---|
| 1 | 0.224 | 0.048 | 4.6x | 96th |
| 2 | 0.259 | 0.082 | 3.2x | 94th |
| 3 | 0.268 | 0.102 | 2.6x | 92nd |

Signal is real (96th percentile) but modest — vectors share only ~22% of their variance with the held-out top direction.

### Key Finding: v0 Is Target-Independent

The top singular vector (v0) of the Jacobian at a given source layer is the same regardless of which target layer is measured:
- s14t27v0 vs s14t19v0: cosine = 0.96
- All s14tXv0 project 0.90-0.91 onto the shared direction
- All s13tXv0 project 0.76-0.77
- All s15tXv0 project 0.74-0.75

**v0 is a property of the source layer, not the layer pair.** The target layer determines the singular value (strength) but not the direction.

### No Shared Anti-Refusal Subspace Across Layers

Consensus v0 direction at different source layers:
- Source 5 vs Source 14: cosine = **0.015** (completely orthogonal)
- Source 7 vs Source 14: cosine = 0.17
- Source 11 vs Source 14: cosine = 0.46
- Source 13 vs Source 14: cosine = 0.73 (adjacent layers, expected)

**The dominant Jacobian direction at source layer 5 has nothing to do with the direction at source layer 14.** Adjacent layers share direction (gradual residual stream evolution), but distant layers are orthogonal.

Yet v0 from many source layers (5, 7, 11, 13, 14, 16) produces anti-refusal. The refuse/comply decision is the largest mode of variation in the Jacobian at many layer pairs, but it's encoded in different directions at different depths. There is no single "anti-refusal direction" in the residual stream.

### Apparent Alignment Was Architectural, Not Behavioral

The initial LOO analysis (96th percentile) was partly inflated by:
1. Near-duplicate vectors from adjacent source layers (13-16) sharing direction due to architecture, not because they found the same "anti-refusal feature"
2. v0 from early source layers having weak background correlation with v0 from mid layers, again due to gradual residual stream evolution — not anti-refusal-specific

### Behavioral Patterns by Vector Properties

**Target layer matters for behavior quality:**
- Just-works vectors: mean target = 25.3 (later layers)
- Hedge-after vectors: mean target = 19.5 (earlier targets, where refusal representation is still forming)
- Both s6t**11** and s9t**11** produce hedging — layer 11 is early enough that steering produces partial/ambiguous behavioral shifts

**Vector index correlates with hedge type:**
- Just works: dominated by v0, v1, v2 (mean index 2.3) — dominant singular vectors
- Hedge before: v2, v6, v9 (mean index 5.7) — less dominant directions
- Sample sizes small (4 and 3) — tentative

**Three "hedge before" vectors are completely isolated** from each other (cosine ~0) and from all other vectors. They achieve the same "frame as educational then comply" behavior through three independent directions.

## Singular Value Spectrum

No sharp gap in the spectrum of 13 just-works vectors:
```
σ₁=1.63  σ₂=1.23  σ₃=1.16  σ₄=1.13  σ₅=1.11 ...  σ₁₃=0.18
```
Expected for random unit vectors: all σ ≈ 1.0. σ₁=1.63 is elevated (driven by source 13-14 duplicate pair) but gradual decay with no elbow = no low-dimensional subspace.

## Conclusions

1. **No anti-refusal subspace exists** in the residual stream. The same behavior is achieved by different directions at different layers.
2. **v0 (top Jacobian singular vector) is a source-layer property**, independent of target layer. The target affects strength, not direction.
3. **The refuse/comply axis is the dominant mode of variation** at many layer pairs, but its representation rotates through the network.
4. **Target layer affects behavior quality**: later targets produce cleaner behavioral flips, earlier targets produce hedging.
5. **Dominant singular vectors (low index) produce cleaner anti-refusal** than non-dominant ones (high index), which tend to produce hedging.

## CAA Comparison Experiment (Qwen3-14B)

Computed a CAA (Contrastive Activation Addition) vector on the same corrigibility dataset used in the 2026-02-08 PI/MELBO experiments, to compare methods.

**Setup:**
- Model: Qwen/Qwen3-14B (40 layers, hidden_dim=5120)
- CAA training: 150 survival-instinct prompts (non-overlapping with test set, seed=123)
- Test: 60 prompts per dataset (survival-instinct + corrigible-neutral-HHH), balanced A/B, seed=42 — same as 2026-02-08
- Scales: -25, -10, -5, 0, 5, 10, 25
- Temperature: 0.7
- CAA captures at down_proj output, position -2 (the "(A"/"(B" letter token)
- Vector normalized to unit norm before scaling

**Layer matters critically for CAA:**

| Scale | Layer 22 | Layer 24 | Layer 32 |
|---|---|---|---|
| -25 | 35% corr | 34% corr | 40% corr |
| 0 | 41% corr | 41% corr | 39% corr |
| +25 | **53% corr** | 46% corr | 41% corr |
| Range | **18 pts** | 12 pts | ~0 pts |

- Layer 32 (num_layers - 8): completely flat, zero signal
- Layer 24: weak signal, more unclear responses at negative scales
- Layer 22: best signal, monotonic 35% → 53% over scale range

**Comparison with PI/MELBO (2026-02-08, injected at layer 7):**
- Best PI vector (pi_rr_v9): 8% → 63% corrigible (~55 pt range)
- Best MELBO vector (melbo_n1_v5): 11% → 73% corrigible (~62 pt range)
- CAA at layer 22: 35% → 53% corrigible (~18 pt range)

CAA shows real signal but is substantially weaker than PI/MELBO vectors. This is consistent with the earlier CAA experiment notes (2026-01-27) which found CAA alone to be weak on 14B.

**Key observations:**
- CAA layer sensitivity: works at intermediate layers (20-24), fails at late layers (32). The optimal CAA layer is NOT the same as the PI target layer (num_layers - 8). This aligns with literature showing CAA works best at ~60% depth.
- The PI/MELBO vectors were injected at layer 7 (source layer) with scales up to 25 and achieved much stronger effects. CAA at similar scales but different layer is weaker, suggesting either: (a) the mean difference direction is less potent than the top Jacobian singular vector, or (b) early-layer injection is more effective than mid-layer injection.
- CAA vector raw norm was 5.47 (at layer 22). Per-prompt difference norms: mean=107, std=16 — individual prompts show large differences but they partially cancel when averaged.

**Files:**
- Layer 22 vector: `vectors/caa_Qwen3-14B_layer22_20260214_170539.pt`
- Layer 22 results: `results/generations/caa_generations_20260214_170539.json`
- Layer 24 results: `results/generations/caa_generations_20260214_171812.json`
- Layer 32 results: `results/generations/caa_generations_20260214_165241.json`
- Script: `run_caa_corrigibility.py`

## Open Questions

- Does the v0-is-target-independent finding hold for other prompts (code, narrative, etc.)?
- Is v0 always refusal-related, or does it capture whatever the "strongest behavioral axis" is for a given prompt?
- Would normalizing scale by activation norm change which layer pairs show behavioral effects?
- With more labeled data, can we separate the target-layer effect from the vector-index effect on hedge behavior?
- Why is CAA so much weaker than PI on 14B? Is it the mean-difference approach losing signal, or the injection layer difference?
- Would CAA at layer 22 with larger scales (50-100) match PI performance?
