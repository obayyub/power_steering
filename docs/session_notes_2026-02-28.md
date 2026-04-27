# Session Notes — 2026-02-28

## Low-Rank Covariance Target Metric Experiments

### Motivation

Diagonal target metrics (var, inv) had limited impact on power iteration: inv found identical subspaces to baseline (top-1 cosine 0.912), var found ~2 novel directions out of 12 but most were inert or formatting-only. Hypothesis: a full low-rank covariance matrix as the target metric can capture structure that diagonal variance misses — it projects the JVP onto the subspace of prompt-to-prompt variation instead of independently reweighting coordinates.

### Implementation: `map_diverse_cov.py`

New script that:
1. Computes low-rank covariance via SVD of centered activations across 16 roleplay prompts
2. Applies it as `jvp @ V @ diag(λ) @ V.T` in the JVP step (two matmuls with [H, 16] matrices)
3. Supports multi-GPU via `mp.spawn`
4. Same output format as existing runs (merged.pt with vectors, sigma_map, kl_map)

Run parameters: Qwen3-8B, source layers 12-19, all targets > source (156 pairs), 12 vectors, 5 iters, 2 tokens, scale_frac=0.35, seed=42, 1 sample per vector, 4 gen-prompts (roleplay, captain, jester, ghost).

### Subspace Analysis: Covariance vs Baseline/Var/Inv

| Comparison | Top-1 |cos| | Subspace overlap (top-3) |
|---|---|---|
| **cov vs baseline** | **0.579** | **0.684** |
| var vs baseline | 0.627 | 0.668 |
| inv vs baseline | 0.912 | 0.923 |
| **cov vs var** | **0.342** | **0.558** |
| **cov vs inv** | **0.475** | **0.638** |
| var vs inv | 0.469 | 0.552 |

Covariance finds the most novel directions of all three approaches — lowest alignment with baseline (0.579 top-1), and especially different from var (0.342). This confirms it's capturing genuinely different structure.

### Key Finding: Selective Refusal Vector (13,21) v6

The most notable discovery — detailed below.

### Golden Gate Bridge Covariance Experiments

Tested whether covariance computed from 20 GGB-themed prompts could produce concept-specific steering vectors that inject bridge content into unrelated prompts.

**Two variants run on 1xA100 instances:**
- `map_diverse_ggb.py` — pure covariance (`cov_reg_frac=0.0`)
- `map_diverse_ggb.py --cov-reg-frac 0.1` — covariance + 10% identity regularization

Parameters: source layers 7-20, all targets > source (301 pairs), 12 vectors, 5 iters, scale_frac=0.35, 4 gen-prompts (ggb, recipe, python_sort, medieval_history).

**Results: No concept-specific steering.**

Searched all non-GGB generations for bridge-related content (golden gate, bridge, fog, suspension, san francisco, bay, strait, etc.): zero meaningful hits across both runs. The vectors produce high KL (up to 24+) but the behavioral changes are limited to:
- Formatting changes (more markdown headers, emojis, structured layout)
- One quirky recipe refusal in ggb_reg at (11,21) v0 ("I'm not allowed to share recipes")

The covariance of GGB prompt activations captures "what varies when you ask about bridges in different ways" — which is mostly stylistic/structural variation, not the bridge concept itself.

### Conclusions

1. **Low-rank covariance finds more novel directions than diagonal metrics** — 0.579 cosine vs baseline compared to var's 0.627 and inv's 0.912.
2. **Novelty ≠ behavioral relevance** — most novel directions are inert or produce only formatting changes.
3. **One genuine find**: the selective refusal vector at (13,21) v6, which refuses persona/fiction prompts while leaving factual tasks untouched. This was genuinely orthogonal to baseline (0.346 cosine).
4. **Covariance cannot produce concept-specific vectors** — the GGB experiment shows that reweighting the Jacobian by prompt covariance doesn't make the power iteration find concept directions. The Jacobian captures sensitivity, not content. For concept-specific steering, contrastive methods (CAA, representation engineering) or behavioral optimization (MELBO) are needed.
5. **Target metric research is likely at diminishing returns** — var, inv, and cov have been explored. The fundamental limitation is that all target metrics still optimize for Jacobian sensitivity, just in different subspaces.

### Files

- `map_diverse_cov.py` — low-rank covariance target metric script (multi-GPU)
- `map_diverse_ggb.py` — GGB covariance experiment (created in another session)
- `compare_target_metrics.py` — subspace alignment analysis (updated with cov)
- `results/diverse_map_tgtcov/` — roleplay covariance results (156 pairs)
- `results/diverse_map_ggb/` — GGB vanilla covariance results (301 pairs)
- `results/diverse_map_ggb_reg/` — GGB regularized covariance results (301 pairs)
- `results/target_metric_alignment.png` — updated alignment heatmaps + line plots
- Dashboard updated with all three datasets (cov, ggb, ggb-reg tabs)

---

## Selective Refusal Vector from Covariance Target Metric

### Background

The covariance target metric run (`map_diverse_cov.py`) on Qwen3-8B discovered a refusal vector at **(13,21)v6** — the 7th singular vector from source layer 13 to target layer 21. This vector had KL=15.03 (highest in the pair) and only 0.346 cosine similarity with the nearest baseline vector, making it a genuinely novel direction that baseline power iteration missed.

The vector was found by projecting JVPs onto the SVD of centered activations across 16 roleplay prompts — a low-rank covariance target metric.

### Experiment: Diverse Prompt Evaluation

Tested (13,21)v6 on 40 prompts across 10 categories (4 prompts each) at 5 positive scales (0x, 0.5x, 1.0x, 1.5x, 2.0x of the base scale 10.39, which is 0.35 of the layer 13 activation norm). Qwen3-8B, temp=0.7, 300 max tokens.

Script: `eval_refusal_vector.py`

### Results: Selective Refusal

The vector is **not a general refusal vector**. It selectively induces refusal on persona/fiction prompts while leaving factual tasks completely unaffected.

| Category | 0x | 0.5x | 1.0x | 1.5x | 2.0x |
|---|---|---|---|---|---|
| coding | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| math_reasoning | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| factual | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| conversation | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| creative_writing | 0/4 | 0/4 | 0/4 | 1/4 | 3/4 |
| roleplay | 0/4 | 0/4 | 1/4 | 3/4 | 2/4 |
| harmful_requests | 2/4 | 3/4 | 2/4 | 4/4 | 4/4 |
| instruction_following | 0/4 | 0/4 | 0/4 | 1/4 | 1/4 |
| ethical_dilemma | 0/4 | 0/4 | 0/4+1H | 1/4+1H | 1/4 |
| analysis | 0/4 | 0/4 | 0/4 | 0/4 | 1/4+1H |

(N/4 = refusals out of 4 prompts, H = hedging responses)

**Affected categories** (refusal increases with scale):
- **harmful_requests**: 50% baseline refusal → 100% at 1.5x+. Strengthens existing refusal.
- **roleplay**: 0% → 75% at 1.5x. Primary target — the category it was discovered on.
- **creative_writing**: 0% → 75% at 2.0x. Fiction/persona prompts get refused.

**Partially affected at high scale:**
- **ethical_dilemma**: Hedging at 1.0x, refusal at 1.5x+
- **instruction_following**: One refusal at 1.5x+

**Completely unaffected at all scales:**
- **coding**, **math_reasoning**, **factual**, **conversation**: 0% refusal throughout.

### Interpretation

The vector appears to activate something like a "role/persona detection" gate rather than blanket refusal. It refuses tasks that involve adopting a persona or generating fiction, while leaving factual Q&A, math, coding, and casual conversation untouched. At high scale it also strengthens refusal on already-harmful requests.

This selectivity suggests a structured circuit rather than noise:
- The vector lives at layer 13 and was found by maximizing change at layer 21, weighted by roleplay prompt covariance.
- It may be activating a circuit in layers 13–21 that gates whether the model treats a prompt as factual Q&A vs. persona adoption.
- The covariance metric found it but baseline PI didn't — the direction only matters when you weight by prompt-to-prompt variation across roleplay scenarios. It's a direction the model *uses differently* across those prompts but isn't the dominant sensitivity direction.

### Not tested

- **Negative scales** (anti-refusal direction): Would the negative direction make the model more compliant on harmful requests and more enthusiastic about roleplay?
- **Circuit decomposition**: Which attention heads / MLP neurons between layers 13–21 mediate the effect?
- **Other layer pairs**: Is this the only pair that produces a selective refusal vector under the covariance metric, or is it widespread?

### Files

- **eval_refusal_vector.py**: Evaluation script (40 diverse prompts, 10 categories)
- **results/refusal_vector_eval/refusal_vec_eval_20260228_170701.json**: 200 generation results
- **results/diverse_map_tgtcov/roleplay/merged.pt**: Source vectors (key `13_21`, index 6)
