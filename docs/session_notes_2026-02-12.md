# Session Notes — 2026-02-12

## Diverse Map Experiment (Qwen3-8B)

Re-ran the Jacobian layer-pair map with:
- **Qwen/Qwen3-8B** (instruct, 36 layers, 630 pairs)
- **Thinking OFF** (`enable_thinking=False`) — previous run accidentally had thinking on
- **7 prompts**: code, narrative, refusal, reasoning, strawberry, roleplay, persuasion
- **Scale 10**, 12 vectors, 5 iterations, 3 samples, 300 max tokens, temp 0.7
- **KL threshold 0.5** — only generate if max KL across vectors exceeds 0.5
  - Threshold barely helped: 85% of pairs passed it (vs expected 30% from thinking-on data)
  - With thinking off, the model is much more sensitive to steering (no `<think>` buffer)

## Observations from Dashboard

**Narrative**: Mostly mild effects. Early layers break output. Mid layers sometimes induce reasoning/thinking mode, format changes (adding "Sure!" prefix, genre tags, word counts). Story style itself rarely changes.

**Code**: Also mild. Sometimes dives into code vs explains first. Response style changes ("Absolutely" vs "Certainly"). Approach occasionally shifts but structure stays intact.

Overall: surface-level format changes rather than deep behavioral shifts for open-ended prompts.

## Convergence Test (pair 3→28, code prompt)

Ran 50 iterations of block power iteration with Rayleigh-Ritz at every step.

- **Sigmas stabilize by iteration 3**: σ = [101, 95, 83, 82, 79, 71...] unchanged from iter 3 to 50
- **Vectors converge by iteration 5**: cosine similarity with iter-50 vectors is ±1.000 at iter 5
  - Sign flips are inherent SVD ambiguity, not convergence failure
- **σ₁/σ₂ ≈ 1.06** — this is the true spectrum, not slow convergence
- **Conclusion**: 5 iterations is sufficient. Mild steering effects are not a convergence problem.

## Activation Norm Analysis

Measured MLP down_proj output norms across all 36 layers:

| Layer range | Typical norm | Scale=10 as % of norm |
|-------------|-------------|----------------------|
| 0-5 (early) | 5-15 | 70-190% (way too strong) |
| 6 (outlier) | 374 | 2.7% |
| 7-15 (early-mid) | 15-25 | 40-65% |
| 16-22 (mid) | 24-45 | 22-42% |
| 23-28 (mid-late) | 54-111 | 9-19% |
| 29-35 (late) | 117-714 | 1.4-8.5% |

**Key insight**: Fixed scale=10 is a massive perturbation at early layers (breaks output) and a tiny nudge at late layers (too weak to change behavior). This likely explains both the "early layers break it" and "mid/late layers are mild" observations.

## Ideas for Next Run

### 1. Normalize scale by activation norm

Instead of fixed `scale=10`, set `scale = alpha * activation_norm_at_source_layer`. This way the perturbation is a consistent fraction of the activation norm regardless of layer.

**KL on next token should still be valid** — KL measures the output distribution shift, which is what we care about regardless of how we set the scale. The normalized scale would just make comparisons across layer pairs more meaningful (currently early-layer pairs have inflated KL because the perturbation is proportionally huge).

### 2. Positive and negative steering generations

Currently we only generate with `+vector * scale`. We should also generate with `-vector * scale` to see both directions of the behavioral axis.

Proposal: **2 positive + 2 negative samples per vector** (4 total, same compute as current 3 + margin). Would show whether the vector captures a meaningful bidirectional axis (e.g., verbose ↔ terse, formal ↔ casual) rather than just a one-sided perturbation.

### 3. Per-vector KL threshold for generation

Currently: if *any* vector has KL > threshold, generate for *all 12 vectors*.
Better: only generate for vectors that individually exceed the threshold. Would significantly reduce generation count since typically only 2-4 vectors per pair have meaningful KL.

## Instance Status

- **8x A100 80GB** (149.130.216.16): running diverse map, ~3/7 prompts done
- **1x A100** (150.136.46.244): used for convergence test, can terminate
