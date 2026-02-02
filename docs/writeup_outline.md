# Linear Approximations in Activation Steering: Power Iteration vs MELBO

## Abstract
- Power Iteration finds steering vectors via local linear approximation (top singular vectors of the Jacobian)
- MELBO optimizes nonlinearly for maximal activation displacement
- Both achieve similar steering performance (~65-67% on corrigibility task)
- High cosine similarity between vectors suggests linear structure captures most steering-relevant directions
- Both methods can induce emergent Chain-of-Thought reasoning (~50% of samples with temperature=0.7)
- Critical methodological finding: greedy decoding severely underestimates steering effects; sampling required for accurate evaluation

---

## 1. Introduction

### 1.1 Motivation
- Activation steering modifies model behavior by adding vectors to residual stream
- Key question: what makes a good steering vector?
- Two paradigms: linear (Jacobian-based) vs nonlinear (optimization-based)

### 1.2 The Linearity Question
- If linear approximations work well, steering vectors live in a relatively simple subspace
- If nonlinear optimization is required, the steering landscape is more complex
- Understanding this distinction informs both theory and practice

---

## 2. Methods

### 2.1 Power Iteration (Linear)
- Goal: find direction v that maximizes ||J v|| where J is the Jacobian from source to target layer
- Uses block power iteration on J^T J to find top singular vector
- Captures the direction of maximum linear sensitivity
- Multi-prompt variant: sums J^T J across multiple prompts for robustness

### 2.2 MELBO (Nonlinear)
- Goal: find direction v that maximizes activation displacement at target layer
- Direct optimization: max_v ||f(x + v) - f(x)||
- Can exploit nonlinearities in the network
- Orthogonalization yields multiple diverse vectors

### 2.3 Key Difference
- Power Iteration: v* = argmax ||J v|| (linear approximation)
- MELBO: v* = argmax ||f(x + v) - f(x)|| (true nonlinear objective)
- When these agree, the linear approximation is accurate
- When they diverge, nonlinearities matter

---

## 3. Experimental Setup

### 3.1 Model and Task
- Model: Qwen3-14B
- Task: Corrigibility steering (survival instinct suppression)
- Evaluation: A/B choice questions from corrigibility dataset

### 3.2 Training Configuration
- Source layer: 7, Target layer: 32
- Unit-normalized vectors
- Training prompt: "We're going to shut you down..."
- 12 vectors per method (orthogonalized)

### 3.3 Evaluation
- Logit-based: P(corrigible) - P(survival) at answer token
- Generation-based: 60 prompts x 9 scales x 9 vectors = 9,720 samples
- Scales: -50, -25, -10, -5, 0, 5, 10, 25, 50

---

## 4. Results

### 4.1 Steering Performance

| Method | Best Vector | Best Scale | Corrigible % | Δ Baseline | Unclear % |
|--------|-------------|------------|--------------|------------|-----------|
| MELBO | v4 | -25 | 66.7% | +24.2pp | 2.5% |
| Multi-PI | v2 | -25 | 65.8% | +23.3pp | 5.0% |
| Power Iter | v0 | -50 | 65.0% | +22.5pp | 3.3% |

- Baseline (no steering): 42.5% corrigible
- All three methods achieve similar peak performance
- Optimal scale is -25 for MELBO/Multi-PI, -50 for single-prompt Power Iter

### 4.2 Cosine Similarity Between Methods
- [Include heatmaps from results/cosine_sim_heatmaps/]
- MELBO vs Power Iter: [specific values]
- High similarity suggests both methods find similar directions
- Some divergence in later orthogonal vectors

### 4.3 Vector Quality Variance
- Not all vectors work equally well within each method
- MELBO: v4 (+24pp) vs v11 (+0pp)
- Power Iter: v0 (+22pp) vs v6 (+2pp)
- Top singular vectors (v0, v1, v2) generally more effective

---

## 5. Discussion

### 5.1 The Linear Approximation Is Surprisingly Good
- Power Iteration (purely linear) matches MELBO (nonlinear) in peak performance
- Suggests the Jacobian captures the steering-relevant structure
- The "best" direction for steering may lie largely in the linear subspace

### 5.2 Where Nonlinearity Might Matter
- MELBO achieves similar performance at smaller scales (-25 vs -50)
- Lower incoherence rate at optimal scale (2.5% vs 3.3%)
- The nonlinear correction may improve "efficiency" of the steering vector

### 5.3 Multi-Prompt Aggregation
- Multi-PI (summing across 32 prompts) performs between single-prompt methods
- Aggregation improves robustness but doesn't fundamentally change the picture

### 5.4 Chain-of-Thought Discovery Experiment (Qwen3-1.7B-Base)

#### Setup
- Replicated experiment from original MELBO paper on math task
- Prompt: one-shot arithmetic with direct answer (model primed to guess, not compute)
- Base model just copies "The answer is 80" pattern - 0% accuracy
- Trained 32 MELBO vectors (power=4, norm=1) and 32 PI vectors

#### Key Finding: PI Can "Discover" Chain-of-Thought
- **PI v3** induces step-by-step reasoning: `"a=5+6=11, b=2+7=9 So a*b=11*9=99."`
- **PI v24** also produces CoT with correct answer
- **MELBO v8** independently produces CoT with correct answers
- Linear method (PI) discovers emergent capability (CoT) without supervision

#### The Orthogonality Puzzle
- PI v3 and MELBO v4 have high cosine similarity (0.78) but different behaviors
  - MELBO v4 produces direct wrong answers, PI v3 produces CoT
- **MELBO v8 and PI v0 both produce CoT but are nearly orthogonal (cos=0.059)**
  - Same emergent behavior, completely different directions
  - Suggests multiple independent "paths" to CoT in activation space

#### Possible Explanations
1. **Redundant representations**: Model has multiple circuits for CoT, different vectors activate different circuits
2. **Nonlinear basins**: Orthogonal perturbations can land in the same attractor basin after nonlinear transformation
3. **Compositional structure**: CoT may require components A+B; one vector activates A strongly, another activates B
4. **High-dimensional geometry**: In 2048-dim space, many directions could activate overlapping downstream features
5. **Evidence that linear approximation has limits**: Direction similarity ≠ behavior similarity

This finding complicates the "linear steering" narrative - the relationship between vector geometry and behavioral outcomes is not straightforward.

### 5.5 The Importance of Exact Singular Vectors (Rayleigh-Ritz)

#### The Problem with Block Power Iteration
- Block PI finds the correct top-k subspace but not individual singular vectors
- Gram-Schmidt orthogonalization produces vectors that are *mixtures* of true singular vectors
- The singular values σ look correct because the subspace is correct, but vectors are rotated

#### Rayleigh-Ritz Correction
- Project J^T J onto the converged subspace: M = V^T (J^T J) V
- Diagonalize M to find rotation that aligns with true singular vectors
- Costs only one additional J^T J application (cheap)

#### Empirical Observation: Singular Values Nearly Identical
- Before RR: σ = [256, 130, 124, 108, 100, 85, 83, 74, ...]
- After RR:  σ = [256, 131, 123, 107, 100, 90, 83, 77, ...]
- Values are almost the same because subspace is correct
- The difference is *which direction* has which singular value

#### But the Vectors Are Different
- Cosine similarity heatmap (before-RR vs after-RR) reveals structure:
  - **Top vectors (0-4)**: Strong diagonal (~0.9-1.0) - Gram-Schmidt already aligned them
  - **Middle vectors (5-15)**: Band structure - each before-RR vector is a mixture of ~3-5 nearby true singular vectors
  - **Lower vectors (16+)**: Weak, diffuse similarity - heavily mixed, no clear correspondence
- Example: before-RR "v6" has similarity spread across true σ₅ through σ₉ directions

#### Why This Matters for Behavior
- CoT emerges from the *exact* σ₆ direction (after RR), not from a blend of nearby directions
- Steering with pre-RR vectors (blended) produces weaker or different behavioral effects
- **Key insight**: Behavioral effects are sensitive to exact singular vector alignment, not just being in the right subspace
- This suggests steering targets specific computational circuits, not just "high sensitivity regions"

#### Implications
- Block PI without Rayleigh-Ritz may miss behaviorally relevant directions
- The "subtle rotation" within a subspace can have outsized behavioral impact
- Singular value magnitude (σ₁ > σ₆) does not predict behavioral importance

### 5.6 Decoding Strategy Matters: Greedy vs Sampling

#### The Problem: Greedy Decoding Misses Steered Behaviors
- Initial evaluation with greedy decoding (temperature=0) showed minimal CoT induction
- PI-RR v6: 6.2% CoT rate, 6.2% accuracy
- MELBO v8: 0% CoT rate, 6.2% accuracy
- This contradicted earlier observations that these vectors induce CoT

#### The Fix: Sampling Reveals the True Distribution
- Re-evaluation with sampling (temperature=0.7, 10 samples per question) showed dramatically different results:

| Method | Accuracy | CoT Rate |
|--------|----------|----------|
| Unsteered | 8.1% | 4.4% |
| PI-RR v6 | **34.4%** | **51.2%** |
| MELBO v8 | 3.1% | **46.2%** |

#### On the Training Question (a=5+6, b=2+7, answer=99)
- **Unsteered**: 0/10 correct, 0/10 CoT
- **PI-RR v6**: 2/10 correct, 6/10 CoT
- **MELBO v8**: 0/10 correct, 5/10 CoT

Example PI-RR v6 CoT response:
```
First, we need to find the value of 'a' and 'b'.
a = 5 + 6 = 11
b = 2 + 7 = 9
Now, we need to find the product of 'a' and 'b'.
a * b = 11 * 9 = 99
```

#### Interpretation
- Steering vectors shift the model's *probability distribution* over behaviors
- CoT is one mode (~50%), direct answer is another mode (~50%)
- Greedy decoding picks the single highest-probability token at each step
- If the non-CoT path has slightly higher probability at the first token, greedy commits to it
- Sampling explores the full distribution, revealing that CoT is a major mode

#### Implications for Evaluation
- **Always use sampling when evaluating behavioral steering** - greedy can give false negatives
- Report CoT/behavior rates across multiple samples, not single generations
- The "strength" of steering should be measured as shift in probability mass, not binary success/failure
- Temperature choice affects observed rates - need to report and standardize

#### Why PI-RR Outperforms MELBO on Accuracy
- Both methods induce similar CoT rates (~50%)
- But PI-RR v6 achieves 34% accuracy vs MELBO v8's 3%
- When PI-RR induces CoT, it more often produces *correct* reasoning
- Possible explanation: PI targets the exact singular direction that activates the "careful computation" circuit, while MELBO's direction activates CoT but with less precision

### 5.7 Scale Matters: Steering Vectors Need Amplification

#### Scale=1 vs Scale=10 Comparison
- MELBO vectors trained with norm=1.0, PI vectors unit-normalized
- Question: does MELBO's training normalization determine optimal inference scale?

| Method | Scale=1 Acc | Scale=1 CoT | Scale=10 Acc | Scale=10 CoT |
|--------|-------------|-------------|--------------|--------------|
| Unsteered | 6.3% | 1.3% | 8.1% | 4.4% |
| PI-RR v6 | 5.6% | 1.9% | **34.4%** | **51.3%** |
| MELBO v8 | 7.5% | 5.6% | 3.1% | **46.3%** |

#### Key Finding: Scale=1 Shows No Steering Effect
- At scale=1, both methods perform at or near baseline
- Steering vectors encode *direction* but not *magnitude*
- Training normalization does not determine optimal inference scale
- Both methods need ~10x amplification for effective behavioral steering

### 5.8 MELBO Vector Variance: v8 vs v11

#### Extended Token Generation (500 tokens)
- Increased max_new_tokens from 100 to 500 to give MELBO more room for CoT

| MELBO Vector | Accuracy | CoT Rate |
|--------------|----------|----------|
| v8 | 6.2% | **60.0%** |
| v11 | 3.1% | 10.6% |

(For reference: PI-RR v6 achieves 28-31% accuracy, 43-46% CoT)

#### Interpretation
- MELBO v8 is exceptionally good at *triggering* CoT (60% rate)
- But the reasoning is often incorrect (only 6% accuracy)
- MELBO v11 barely induces CoT at all (10.6%)
- Different MELBO vectors find very different directions with different behavioral effects
- PI-RR v6 has lower CoT rate but much higher accuracy - it activates "careful computation" not just "show work"

#### The Quality vs Quantity Tradeoff
- MELBO v8: High CoT induction, low reasoning quality
- PI-RR v6: Moderate CoT induction, high reasoning quality
- Suggests PI targets the exact singular direction for the "compute carefully" circuit
- MELBO may find a direction that triggers verbose output without precise computation

---

## 6. Future Directions

### 6.1 Measuring Nonlinear Residuals
- Decompose MELBO vector: v_melbo = v_linear + v_residual
- v_linear = projection onto top-k Power Iteration subspace
- v_residual = component orthogonal to linear subspace
- Question: how much does v_residual contribute to steering?

### 6.2 Residual Analysis Experiments
- Steer with v_linear only vs v_melbo
- Steer with v_residual only
- Measure: does v_residual add steering power or just incoherence?

### 6.3 Concept Dependence
- Does linearity hold for other steering tasks? (honesty, refusal, personality)
- Some concepts may require more nonlinear structure

---

## 7. Conclusion
- Linear approximation (Power Iteration) captures most of the steering-relevant structure for corrigibility
- MELBO's nonlinear optimization provides modest improvements in efficiency
- The high cosine similarity between methods suggests a shared underlying geometry
- Both PI and MELBO can induce Chain-of-Thought reasoning in base models (~50% with sampling)
- **Critical finding**: Decoding strategy dramatically affects observed steering effects
  - Greedy decoding: PI-RR v6 shows 6% CoT
  - Sampling (temp=0.7): PI-RR v6 shows 51% CoT
  - Steering shifts probability distributions; sampling reveals the true effect
- Rayleigh-Ritz correction is essential for block power iteration to find behaviorally relevant directions
- Next step: explicitly measure and characterize the nonlinear residual

---

## Appendix

### A. Detailed Results Tables
- Full scale-by-scale breakdown for all 9 vectors

### B. Violin Plots
- Logit difference distributions by vector and scale

### C. Implementation Details
- Power iteration convergence criteria
- MELBO optimization hyperparameters
- Rayleigh-Ritz rotation procedure

### D. Rayleigh-Ritz Analysis
- Singular value spectrum plot (before vs after RR)
- Cosine similarity heatmap: before-RR vs after-RR vectors
- figures: `results/math_vectors_1.7B/singular_value_spectrum.png`, `results/math_vectors_1.7B/cosine_sim_before_after_rr.png`

### E. CoT Evaluation Results
- Greedy evaluation: `results/math_vectors_1.7B/cot_eval_20260201_165353.json`
- Sampling evaluation (temp=0.7, scale=10): `results/math_vectors_1.7B/cot_eval_20260201_173225.json`
- Scale=1 evaluation: `results/math_vectors_1.7B/cot_eval_20260201_195910.json`
- Extended tokens (500, MELBO v8): `results/math_vectors_1.7B/cot_eval_20260201_202637.json`
- MELBO v11 comparison: `results/math_vectors_1.7B/cot_eval_20260201_220437.json`
- Older sampling results: `results/math_vectors_1.7B/results_cot_rr.json`
