# Linear Approximations in Activation Steering: Power Iteration vs MELBO

## Abstract
- Power Iteration finds steering vectors via local linear approximation (top singular vectors of the Jacobian)
- MELBO optimizes nonlinearly for maximal activation displacement
- Both achieve similar steering performance (~65-67% on corrigibility task)
- High cosine similarity between vectors suggests linear structure captures most steering-relevant directions
- Raises question: how much does the nonlinear residual matter?

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
