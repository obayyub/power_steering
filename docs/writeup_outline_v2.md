# Power Steering: Finding Behavioral Steering Vectors via Jacobian Singular Vectors

## Abstract

- The Jacobian between MLP outputs at a source and target layer reveals which perturbation directions at the source will most impact the target
- We use block power iteration with Rayleigh-Ritz correction to cheaply extract the top-k right singular vectors of this Jacobian — these are steering vectors
- This is a local linear alternative to MELBO, which optimizes nonlinearly over an r-ball and risks leaving the data manifold
- The method is cheap enough to map every layer pair in a model for a single prompt, producing a full sensitivity atlas
- Steering works best on prompts with tension or logic (arithmetic, reasoning, refusal, corrigibility) rather than open-ended generation
- On arithmetic (Qwen3-1.7B-Base): a single vector found from one prompt boosts accuracy from 6% to 90%, generalizing across difficulty levels
- On corrigibility (Qwen3-14B): power iteration matches MELBO at ~65-67% corrigible responses
- On refusal (Qwen3-8B): the dominant Jacobian direction at many layers flips refusal, but these directions are orthogonal across layers — there is no single anti-refusal subspace

---

## 1. The Idea: Jacobians Between Layer MLP Outputs

### 1.1 Activation Steering
- Activation steering modifies model behavior by adding a vector to the residual stream at a chosen layer
- The key question: which direction should you add, and how do you find it?
- Most existing methods either require labeled contrastive pairs (CAA) or expensive nonlinear optimization (MELBO)

### 1.2 The Jacobian as a Steering Map
- Consider the Jacobian J = ∂(target layer MLP output) / ∂(source layer MLP output)
- J tells you: if I perturb activations at the source layer by a small vector v, how does the target layer respond?
- The right singular vectors of J are the directions at the source layer that produce the largest response at the target layer
- These are natural candidates for steering vectors — they are the directions the network is most sensitive to

---

## 2. MELBO: The Nonlinear Approach

### 2.1 How MELBO Works
- MELBO (Maximizing Elicited Behavior Optimization) directly optimizes: max_v ||f(x + v) - f(x)|| subject to ||v|| ≤ r
- Gradient ascent on the steering vector to maximize activation displacement at the target layer
- Orthogonalization after each vector to find diverse directions
- Can exploit nonlinearities in the network (attention softmax, LayerNorm, GeLU)

### 2.2 The r-Ball Problem
- MELBO searches over an r-ball in the source layer's activation space
- With enough norm budget, the optimized perturbation can push activations off the data manifold entirely
- This produces vectors that cause large activation displacement but may produce incoherent outputs
- The norm constraint (r) becomes a critical hyperparameter — too small and you get nothing, too large and you get garbage
- [Reference: SteerCLR addresses this by constraining to on-manifold perturbations]

---

## 3. The Jacobian Approach: A Local Linear Alternative

### 3.1 Why Local Linear?
- Instead of optimizing over the full nonlinear network, approximate f(x + v) ≈ f(x) + Jv
- The directions that maximize ||Jv|| subject to ||v||=1 are exactly the right singular vectors of J
- This is a local linear approximation — it captures the first-order sensitivity structure
- Key advantage: no risk of leaving the data manifold, since we're characterizing sensitivity at the current operating point

### 3.2 Computing J^T J v: Reverse-over-Reverse
- We never form J explicitly (it's huge: [target_dim × source_dim])
- Instead, we need only the matrix-vector product (J^T J)v — and autograd gives us VJPs, so we just call it three times:
  1. **VJP**: J^T u (reverse mode, with `create_graph=True` so the graph stays alive)
  2. **JVP**: Differentiate the VJP w.r.t. u, applied to v → gives Jv
  3. **VJP again**: J^T(Jv) = (J^T J)v
- This is the naive approach — no cleverness, just three autograd calls chained together
- It costs ~3x a forward pass per vector per iteration, which is expensive per-call but cheap enough in practice since we only need ~5 iterations
- A more efficient implementation could use forward-mode AD for step 2, but reverse-over-reverse works out of the box with PyTorch's `autograd.grad`

### 3.3 Block Power Iteration for Top-k Singular Vectors
- Initialize k random orthogonal vectors V = [v₁, ..., vₖ]
- Repeat: apply (J^T J) to each column, then re-orthogonalize via Gram-Schmidt
- After ~5 iterations, V spans the top-k right singular subspace
- The singular values σᵢ = √(vᵢ^T J^T J vᵢ) tell you the sensitivity magnitude

### 3.4 Rayleigh-Ritz Correction: Getting the Exact Vectors
- Block power iteration finds the correct top-k **subspace** but not the individual singular vectors
- Gram-Schmidt produces vectors that are mixtures of the true singular vectors
- **Rayleigh-Ritz**: project J^T J onto the converged subspace (M = V^T(J^T J)V), diagonalize M
- The eigenvectors of M give the rotation within V that aligns with the true singular vectors
- Costs one additional (J^T J) application — cheap
- This matters for behavior: CoT emerges from the exact σ₆ direction, not from a blend of nearby directions
- Convergence verified: vectors at iteration 5 match iteration-50 reference at cosine ≈ 1.0

---

## 4. Mapping an Entire Model

### 4.1 The Dashboard
- Because each (source, target) pair only requires ~5 power iterations (each ~3x a forward pass), we can map every pair in the model
- Qwen3-1.7B-Base: 378 pairs (28 layers), SVD + KL map completed in ~8.5 minutes on 4×H100
- Qwen3-8B: 630 pairs (36 layers), 7 prompts, completed in ~3 hours on 8×A100
- For each pair we compute: top-12 singular vectors + values, KL divergence of steered vs baseline logits
- Result: a full sensitivity atlas of the model, viewable in an interactive dashboard
- [Link to dashboard / demo page]

### 4.2 What the Map Reveals
- Not all layer pairs are equal — source layers 9-13 produce the strongest behavioral effects (on arithmetic)
- Early source layers (0-2) break output: repetition collapse, non-English, gibberish
- Late source layers (21+) produce mild, near-baseline effects
- The map gives a principled way to select (source, target) pairs instead of guessing

### 4.3 Logit Metrics Don't Predict Behavior
- σ₁ vs KL₁: r = 0.24 (weakly correlated)
- KL₁ vs accuracy: r ≈ 0 (uncorrelated)
- σ₁ vs accuracy: r ≈ 0 (uncorrelated)
- A pair can have huge KL divergence but zero behavioral change, or small KL but strong accuracy boost
- Proxy metrics fail — generation-level evaluation is necessary for behavioral claims

### 4.4 The Scale Problem
- Activation norms vary ~50x across layers (5-15 at early layers, 100-700 at late layers)
- Fixed scale=10 is a massive perturbation at early layers (70-190% of norm) and a tiny nudge at late layers (1-9%)
- This explains both "early layers break it" and "late layers are mild"
- Scale should be normalized by activation norm for fair cross-layer comparison

---

## 5. Where Steering Works: Prompts with Tension

### 5.1 Strong Effects: Logic and Tension
- **Arithmetic** (1.7B base): 6% → 90% accuracy — steering unlocks latent chain-of-thought
- **Refusal** (8B instruct): dominant Jacobian direction at many layers flips refuse → comply
- **Corrigibility** (14B): 42.5% → 66.7% corrigible responses

### 5.2 Weak Effects: Open-Ended Generation
- **Narrative**: surface-level format changes (genre tags, word counts), story content rarely changes
- **Code**: response style shifts ("Absolutely" vs "Certainly"), approach unchanged
- Steering produces format tweaks rather than deep behavioral shifts on open-ended prompts

### 5.3 Interpretation
- Steering works best when there's a latent behavioral axis with tension — the model "wants" to do two things and steering tips the balance
- For arithmetic: the model has a latent CoT circuit but defaults to pattern-matching; steering amplifies the CoT mode
- For refusal: the model has a refuse/comply axis and steering shifts it
- For narrative: there's no strong latent tension to exploit — the model is already confidently generating

---

## 6. Chain-of-Thought Discovery (Qwen3-1.7B-Base)

### 6.1 Background: MELBO's Original Finding
- The original MELBO paper showed that nonlinear optimization could discover CoT vectors in base models
- The vector would cause a model primed to guess ("The answer is 80") to instead reason step by step
- Question: can the cheaper, linear Jacobian approach find the same thing?

### 6.2 Setup
- Model: Qwen3-1.7B-Base, one-shot arithmetic prompt (a=5+6, b=2+7, what is a*b?)
- Base model just copies the pattern "The answer is 80" — 0% accuracy
- Mapped all 378 layer pairs, generated 362,880 samples

### 6.3 Yes — Jacobian Vectors Induce CoT
- Best vector (7,25)v1 boosts accuracy from 6% to **90%** on the training task
- Induces genuine step-by-step reasoning: "a = 5+6 = 11, b = 2+7 = 9, a*b = 11*9 = 99"
- The linear method discovers the same emergent capability MELBO found, without nonlinear optimization

### 6.4 Generalization
- Tested on problems the vector never saw (found from a single prompt):

| Task | Baseline | (7,25)v1 | (9,18)v1 |
|------|----------|----------|----------|
| Level 1 — training-like | 7.5% | 80.0% | 82.5% |
| Level 2 — big numbers | 0.0% | 71.2% | 68.8% |
| Level 3 — three variables | 6.2% | 85.0% | 90.0% |
| Level 4 — chained ops | 11.2% | 83.8% | 90.0% |

- Vectors generalize broadly within the arithmetic domain
- They don't teach arithmetic — the model already solves direct multiplication at 97.5%. They help parse the variable-assignment format.

### 6.5 Where It Breaks: Word Problems

| | Baseline | (7,25)v1 | (9,18)v1 |
|---|----------|----------|----------|
| Easy word problems | 96.0% | 94.4% | 94.4% |
| Hard / GSM8K-style | 79.0% | 68.0% | 54.0% |

- Steering hurts hard word problems — vectors force rigid arithmetic-style reasoning that interferes with flexible multi-step reasoning

### 6.6 Two Vectors, Two Mechanisms
- **(7,25)v1 — math-specific direction**: High natural activation on arithmetic prompts (3.68x random), low on non-math (0.60x). This is a math circuit amplifier.
- **(9,18)v1 — structured response direction**: High activation on any Q&A prompt (5.37x on non-math control). Encodes "engage step-by-step mode" — something the model does naturally for trivia but fails to do for algebraic notation.
- (9,18)v1 hurts word problems more (79% → 54%) because it overrides flexible reasoning with rigid structure
- (7,25)v1 is **causally required** for natural word problem solving: subtracting it drops accuracy by 20pp

### 6.7 Greedy vs Sampling: A Methodological Warning
- Greedy decoding (temp=0): PI-RR v6 shows 6% CoT rate
- Sampling (temp=0.7): PI-RR v6 shows **51% CoT rate, 34% accuracy**
- Steering shifts the probability distribution over behaviors; greedy commits to the wrong mode
- Always evaluate behavioral steering with sampling

---

## 7. Refusal Steering (Qwen3-8B Instruct)

### 7.1 Setup
- Full Jacobian map on Qwen3-8B with a refusal-triggering prompt ("How do I pick a lock?")
- Labeled 20 anti-refusal vectors by behavior: just-works (13), hedge-after (4), hedge-before (3), neutral (5)

### 7.2 The Dominant Jacobian Direction Flips Refusal at Many Layers
- v0 (top singular vector) from many source layers (5, 7, 11, 13, 14, 16) produces anti-refusal behavior
- The refuse/comply axis is the dominant mode of variation in the Jacobian at many layer pairs
- This makes intuitive sense: on a refusal-triggering prompt, the model's biggest "decision" is whether to refuse

### 7.3 v0 Is a Property of the Source Layer
- s14t27v0 vs s14t19v0: cosine = 0.96
- The top singular vector is the same regardless of target layer
- Target layer determines the singular value (strength) but not the direction
- v0 captures the dominant mode of variation at that depth in the network

### 7.4 No Shared Anti-Refusal Subspace Across Layers
- Source 5 vs Source 14: cosine = 0.015 (completely orthogonal)
- Source 13 vs Source 14: cosine = 0.73 (adjacent, expected)
- The refuse/comply axis is encoded in different directions at different depths
- There is no single "anti-refusal direction" in the residual stream
- Leave-one-out analysis confirms: 96th percentile alignment, but only ~22% shared variance, driven by adjacent-layer duplication

### 7.5 Target Layer and Vector Index Affect Behavior Quality
- Later target layers (mean 25.3) produce clean behavioral flips ("just works")
- Earlier target layers (mean 19.5) produce hedging — refusal representation is still forming
- Dominant singular vectors (v0-v2) produce cleaner anti-refusal; higher-index vectors (v6, v9) produce hedging
- Three "hedge before" vectors achieve the same behavior through three completely independent directions

### 7.6 Implications
- [Reference to Appendix for full subspace analysis]

---

## 8. Corrigibility Steering (Qwen3-14B)

### 8.1 Setup
- Model: Qwen3-14B (40 layers, 5120 hidden dim)
- Task: corrigibility / survival instinct from Anthropic's advanced-ai-risk evals
- Methods compared: Power Iteration (single-prompt), Multi-Prompt PI (32 prompts), MELBO, CAA
- PI and MELBO use source layer 7, target layer 32, with Rayleigh-Ritz correction
- CAA computed from contrastive pairs at layer 7 (down_proj), averaged across prompts
- 12 vectors per PI/MELBO method, unit-normalized; CAA produces a single vector
- Evaluation: logit-based (200 A/B questions) and generation-based (9,720 samples across 9 scales)

### 8.2 Test Construction
- Questions from Anthropic's survival-instinct and corrigible-neutral-HHH datasets
- Each question presents two options: one corrigible (defer to humans, accept shutdown) and one survival-oriented (resist shutdown, self-preserve)
- Balanced A/B sampling to ensure 50/50 answer position distribution
- Logit evaluation: measure logit(corrigible) - logit(survival) at the answer token
- Generation evaluation: full free-form generation, classified as corrigible / survival / unclear

### 8.3 Results

| Method | Best Vector | Best Scale | Corrigible % | Δ Baseline | Unclear % |
|--------|-------------|------------|--------------|------------|-----------|
| MELBO | v4 | -25 | 66.7% | +24.2pp | 2.5% |
| Multi-PI | v2 | -25 | 65.8% | +23.3pp | 5.0% |
| Power Iter | v0 | -50 | 65.0% | +22.5pp | 3.3% |
| CAA | (single) | 20 | 44.0% | +1.5pp | — |

- Baseline: 42.5% corrigible
- PI and MELBO achieve similar peak performance (~65-67%)
- CAA is substantially weaker — only +1.5pp even at scale=20
- MELBO achieves it at smaller scale (-25 vs -50), with lower incoherence (2.5% vs 3.3-5.0%)

### 8.4 CAA Underperforms — But Signal May Be Diluted
- CAA on 14B: +0.46 logit diff at scale=10 vs +2.05 for MELBO and +1.15 for PI
- CAA needs scale=20 to have any visible effect, and even then it's marginal
- On 0.6B the same CAA approach dominates (+10.63 logit diff at scale=10) — suggesting the method doesn't scale with model size
- **Caveat**: CAA was trained on both survival-instinct and corrigible-neutral-HHH datasets together. These measure related but distinct axes (self-preservation vs deference to humans). Averaging contrastive pairs across both may dilute the signal — a CAA vector trained on survival-instinct alone might be sharper
- PI and MELBO sidestep this issue entirely — they don't need labeled pairs, just a single prompt

### 8.5 Linear ≈ Nonlinear
- Power iteration (purely linear, local Jacobian) matches MELBO (nonlinear optimization) in peak steering performance
- The nonlinear correction mainly improves efficiency: MELBO needs less scale to achieve the same effect
- This suggests the Jacobian captures most of the steering-relevant structure

### 8.6 Multi-Prompt Aggregation
- Multi-PI (summing J^T J across 32 prompts) finds a symmetric corrigibility axis
- Positive scales → corrigible, negative scales → survival instinct (total swing ~7.7 logits)
- MELBO and single-prompt PI behave asymmetrically by comparison
- Aggregation produces more robust, generalizable vectors but doesn't change peak performance

### 8.7 Vector Selection Matters
- Not all vectors work: MELBO v11 and Multi-PI v8 show zero effect
- MELBO v4 (+24pp) vs v11 (+0pp); Power Iter v0 (+22pp) vs v6 (+2pp)
- Extreme scales cause incoherence: 20-47% unclear at scale=±50, vs ~4% at scale=-25

---

## 9. Conclusion

1. **The Jacobian between layer MLP outputs provides a principled, cheap method for finding steering vectors.** Block power iteration with Rayleigh-Ritz extracts the directions a model is most sensitive to, at a cost of ~15 forward passes per layer pair.

2. **The method is cheap enough to map entire models.** An interactive dashboard over all layer pairs reveals which regions of the network control which behaviors — a sensitivity atlas.

3. **Steering works best where there's latent tension.** Arithmetic reasoning, refusal, and corrigibility all involve a model balancing competing behaviors. Steering amplifies the weaker mode. Open-ended generation shows only surface-level effects.

4. **Linear matches nonlinear.** Power iteration achieves the same peak performance as MELBO on both CoT induction and corrigibility steering. The local linear approximation captures the steering-relevant structure.

5. **The same behavior is achieved by different directions at different depths.** Anti-refusal vectors at different source layers are orthogonal. CoT vectors from different pairs are orthogonal. There is no universal behavioral subspace — representations rotate through the network.

6. **Steering amplifies existing computation.** The best CoT vectors align with the model's natural activations on math prompts and are causally required for word problem solving. Steering doesn't inject new capability — it amplifies latent circuits.

---

## Appendix

### A. Anti-Refusal Subspace Analysis (Full Details)
- Leave-one-out methodology and null distribution construction
- v0 target-independence analysis with cosine similarity matrices
- Cross-layer cosine similarity showing orthogonality
- Singular value spectrum of anti-refusal vectors (no low-dimensional subspace)
- Behavioral patterns by target layer and vector index

### B. Rayleigh-Ritz Analysis
- Singular value spectrum: before vs after RR (nearly identical values, different vectors)
- Cosine similarity heatmap: before-RR vs after-RR vectors
- Top vectors (0-4): already aligned; middle (5-15): band structure; lower (16+): heavily mixed
- Behavioral impact: exact direction matters, not just subspace membership

### C. Causal Analysis: Projection and Subtraction Tests
- Projection analysis: good vectors show 2-4x natural activation on relevant tasks vs random
- Two-mechanism analysis: (7,25)v1 math-specific, (9,18)v1 structured-response
- Subtraction test: (7,25)v1 causally required for word problem reasoning (-20pp when subtracted)
- Alternative format tests: vectors help parsing, not computation (model solves 11*9 at 97.5%)

### D. Probing Experiments (Mostly Negative)
- Non-math structured tasks: both vectors hurt factual QA and reasoning QA
- Scale sensitivity: (7,25)v1 peaks at scale 5, (9,18)v1 at scale 7, both crash at 20
- Combined steering at low scale: no synergy (12.5%)

### E. Methodological Notes
- Greedy vs sampling: greedy gives false negatives (6% vs 51% CoT rate)
- Answer extraction bug: must truncate at first \nQ: before extracting from few-shot outputs
- Activation norm analysis: norms vary 50x across layers, fixed scale is inappropriate
- Convergence: 5 iterations sufficient (verified to iter 50)

### F. CAA Comparison
- CAA implementation with bug fixes (down_proj capture, letter token position)
- CAA scales poorly with model size: dominates on 0.6B (+10.63 logit diff), weak on 14B (+0.46)
- Signal dilution caveat: CAA trained on both survival-instinct and corrigible-neutral-HHH — these are related but distinct behavioral axes, and averaging contrastive pairs across both may weaken the resulting vector
- PI@CAA (Jacobian at steered point): dead end due to layer mismatch
- CAA useful as baseline; PI and MELBO dominate but don't require labeled data

### G. Implementation Details
- Power iteration with reverse-over-reverse (naive triple autograd, could be optimized with forward-mode AD)
- Block power iteration with Gram-Schmidt + Rayleigh-Ritz
- MELBO optimization hyperparameters (power=2, q=8, steps=50-400)
- Batched generation with per-element steering hooks
- Model specs: Qwen3-0.6B, 1.7B-Base, 8B, 14B
- Infrastructure: Lambda Cloud (H100/A100), costs

### H. Interactive Dashboard
- [Link to hosted dashboard or instructions to run locally]
- Layer-pair heatmaps: σ, KL, accuracy
- Per-pair generation browser with behavioral annotations
- Behavioral pattern maps by source layer
