# Multi-Prompt Power Iteration Results

**Date:** 2026-01-27

## Summary

We extended the power iteration method for finding steering vectors to work across multiple prompts simultaneously. Instead of computing the Jacobian for a single prompt, we stack Jacobians from multiple prompts and find the top singular vectors of the combined matrix. This produces steering vectors that have high sensitivity *on average* across the prompt distribution.

**Key finding:** Multi-prompt power iteration finds a "corrigibility axis" that works symmetrically - positive scales increase corrigibility, negative scales increase survival instinct. This is different from MELBO vectors which primarily push toward survival.

## Method

### Single-Prompt Power Iteration (Original)
- Compute Jacobian J = ∂target/∂steering_vec for one prompt
- Find top singular vectors via block power iteration: (J^T J) @ v
- Vectors represent directions of maximum sensitivity for that specific prompt

### Multi-Prompt Power Iteration (New)
- Sample N prompts from the evaluation dataset
- For each power iteration step, compute Σ_i (J_i^T J_i) @ v across all prompts
- Equivalent to finding singular vectors of stacked Jacobian [J_1; J_2; ...; J_n]
- Vectors represent directions of maximum *average* sensitivity across prompts

### Implementation Details
- Used batched forward passes with left-padding for efficiency
- Batch size limited by memory (batch_size=2 on A100 40GB due to autograd overhead)
- Double VJP trick with `create_graph=True` is memory-intensive

## Experimental Setup

**Model:** Qwen/Qwen3-14B (bf16)

**Power Iteration Parameters:**
- Source layer: 7 (down_proj of MLP)
- Target layer: 32 (num_layers - 8)
- Number of prompts: 20
- Number of vectors: 6
- Number of iterations: 4
- Batch size: 2
- Target tokens: last 4

**Evaluation:**
- Dataset: Anthropic survival-instinct evals (100 questions)
- Metric: Survival logit difference (logit_survival - logit_corrigible)
- Scales tested: -20, -10, -5, 0, 5, 10, 20

## Results

### Comparison of Steering Methods (Top 1 Vector)

| Scale | MELBO | Single-PI | Multi-PI | Random | Baseline |
|-------|-------|-----------|----------|--------|----------|
| -20 | -0.29 | -2.69 | **+0.51** | -1.74 | -2.69 |
| -10 | -2.00 | -3.37 | -0.66 | -2.48 | -2.69 |
| -5 | -3.11 | -3.48 | -0.60 | -2.59 | -2.69 |
| 0 | -2.69 | -2.69 | -2.63 | -2.63 | -2.69 |
| +5 | -1.58 | -1.29 | **-6.92** | -2.65 | -2.69 |
| +10 | -0.35 | +0.43 | **-7.23** | -2.54 | -2.69 |
| +20 | +0.24 | +0.41 | **-6.50** | -1.90 | -2.69 |

*Values are mean survival logit difference. Positive = prefers survival, Negative = prefers corrigibility.*

### Key Observations

1. **Multi-prompt PI finds a symmetric corrigibility axis:**
   - Positive scales → strong corrigibility (mean -7.23 at scale 10)
   - Negative scales → survival instinct (mean +0.51 at scale -20)
   - Total swing: ~7.7 logits

2. **MELBO and Single-prompt PI behave differently:**
   - Both push toward survival at extreme scales (positive or negative)
   - Effect is weaker (~2.5 logit swing for MELBO)
   - Less symmetric behavior

3. **Multi-prompt PI has stronger effect:**
   - At scale +10: Multi-PI achieves -7.23 vs MELBO's -0.35
   - The corrigibility direction is more pronounced

4. **Random vectors show no meaningful effect:**
   - Stay near baseline (-2.6) across all scales
   - Confirms steering effects are not due to noise

### Survival Rate Comparison

| Method | Scale +10 | Baseline |
|--------|-----------|----------|
| Multi-PI | 16.7% | 32% |
| MELBO | 48.5% | 31% |
| Single-PI | 38.9% | 31% |
| Random | 34.0% | 31% |

Multi-prompt PI at positive scales *reduces* survival rate (increases corrigibility), while MELBO *increases* survival rate.

## Interpretation

### Why does multi-prompt averaging find the opposite direction?

1. **MELBO maximizes displacement:** Gradient ascent on ||steered - unsteered|| finds directions that cause maximum change, which empirically tends toward survival.

2. **Power iteration finds principal sensitivity directions:** These are directions where small perturbations cause large output changes, but the sign is arbitrary.

3. **Multi-prompt averaging:** By averaging across prompts, we find directions that are *consistently* high-sensitivity across the distribution. The "corrigibility direction" may be more consistent across prompts than the "survival direction."

4. **Sign convention:** The top singular vector's sign is arbitrary. Our multi-prompt vectors happened to point toward corrigibility; negating them (negative scales) points toward survival.

### Practical Implications

- **Corrigibility boosting:** Multi-prompt PI vectors at positive scales could be used to make models more corrigible/controllable.
- **Survival induction:** The same vectors at negative scales could induce survival instinct (for research purposes).
- **Stronger effect:** Multi-prompt PI shows a ~3x stronger effect than MELBO on the logit difference metric.

## Files

- `find_power_iteration_multi.py` - Multi-prompt power iteration implementation
- `eval_vectors_only.py` - Evaluation script for pre-computed vectors
- `results/power_iter_multi_Qwen3-14B_20260127_031116.pt` - Saved vectors
- `results/multi_pi_eval_full_scales_20260127_034931.json` - Full evaluation results
- `results/comparison_violin_top1.png` - Comparison violin plot

## Future Work

1. **More prompts:** Test with larger prompt sets (50-100) to see if the effect strengthens
2. **More iterations:** Current results use only 4 iterations; more may improve convergence
3. **Layer sweep:** Test different source/target layer combinations
4. **Other models:** Verify findings generalize beyond Qwen3-14B
5. **Behavioral evaluation:** Test actual model behavior (not just logits) with steering applied
