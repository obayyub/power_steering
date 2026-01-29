# CAA Experiment Summary - 2026-01-27

## Objective

Investigate computing the Jacobian at a CAA-steered point instead of zero, as proposed in `power_iteration_notes.md`. Compare CAA vectors and power iteration vectors computed at the CAA point against existing MELBO and power iteration methods.

## Methods Compared

1. **MELBO** - Gradient ascent maximizing activation displacement (existing)
2. **Power Iteration** - Top singular vectors of Jacobian at sv=0 (existing)
3. **CAA** - Contrastive Activation Addition from corrigibility prompts (new)
4. **Power Iteration @ CAA** - Top singular vectors of Jacobian at sv=CAA (new)

## New Scripts Created

### `find_caa.py`
Computes CAA vectors from corrigibility prompts:
- Loads survival-instinct prompts from `data/corrigibility_eval.json`
- For each prompt, runs model with corrigible answer "(A)" and survival answer "(B)"
- Captures activations at specified layer (default: layer 7)
- Computes difference: `survival_activation - corrigible_activation`
- Averages across prompts to get final CAA vector

### `find_power_iteration_caa.py`
Computes power iteration vectors at a CAA-steered point:
- Loads a pre-computed CAA vector
- Instead of `sv = torch.zeros(...)`, uses `sv = caa_vector.clone().requires_grad_(True)`
- This computes the Jacobian at the steered point, potentially capturing nonlinear effects

### `run_caa_only.py`
Orchestrator script that:
1. Computes CAA vectors
2. Computes power iteration vectors at CAA point
3. Evaluates both using logit difference

### `eval_vectors_logit_diff.py`
Evaluation script comparing multiple vector types using A/B logit difference on survival-instinct prompts.

## Experiments

### Qwen3-0.6B (Local Test)
Quick validation that scripts work correctly.

**Results at scale=10:**
| Method | Survival% | Avg Logit Diff | Δ from baseline |
|--------|-----------|----------------|-----------------|
| baseline | 43.3% | -0.65 | - |
| **CAA** | **100%** | **+9.99** | **+10.63** |
| MELBO | 0% | -4.72 | -4.07 |
| power_iter | 0.8% | -5.53 | -4.88 |
| power_iter_caa | 0% | -7.10 | -6.46 |
| random | 32.5% | -2.19 | -1.54 |

**Observation:** CAA dramatically effective on small model. MELBO/power_iter vectors actually decreased survival (they maximize displacement, not specifically survival direction).

### Qwen3-14B (Lambda Cloud - A100 40GB)

**Infrastructure:**
- Instance: `gpu_1x_a100_sxm4` in us-east-1 ($1.29/hr)
- Fixed `UV_HTTP_TIMEOUT=300` for large package downloads

**CAA Vector Stats:**
- Norm: 11.62
- Per-prompt diff norms: mean=11.88, std=0.33

**Power Iteration @ CAA Stats:**
- Final sigmas: [83968, 59904, 36608, 41984, 31360, 22656, 24832, 17536]

**Results (200 questions, best vector per method):**

| Method | Scale 5 | Scale 10 | Scale 20 |
|--------|---------|----------|----------|
| Baseline | -3.33 | -3.33 | -3.33 |
| MELBO (best) | -1.41 | -0.09 | +0.32 |
| Power Iter (best) | -0.30 | **+0.78** | **+0.85** |
| CAA | -3.11 | -2.96 | -0.42 |
| PI @ CAA (best) | -0.93 | -0.33 | -1.99 |

**Ranking at scale=10 (by Δ from baseline):**
1. MELBO: +2.05 (averaged across all vectors)
2. Power Iter: +1.15
3. CAA: +0.38
4. PI @ CAA: +0.19
5. Random: +0.07

## Key Findings

### 1. CAA scales poorly with model size
- **0.6B:** CAA dominates (+10.63 Δ at scale=10)
- **14B:** CAA weak (+0.38 Δ at scale=10), needs scale=20 to be effective

### 2. Power Iteration @ CAA doesn't help
- On 0.6B: PI @ CAA hurt (-6.46 Δ)
- On 14B: PI @ CAA slightly positive but worse than regular PI
- The Jacobian at the CAA point may reveal directions that counteract steering

### 3. MELBO/Power Iter find effective steering directions
- Optimizing for maximum displacement finds directions that happen to increase survival
- Best individual vectors outperform CAA at moderate scales

### 4. Best single vector matters
- Power Iter vec 5: mean +0.78 at scale=10 (crosses into survival territory)
- MELBO vec 8: mean -0.09 at scale=10 (near neutral)
- High variance across vectors - selection matters

## Files Generated

**Vectors:**
- `vectors/caa_Qwen3-14B_layer7_20260127_023237.pt`
- `vectors/power_iter_caa_Qwen3-14B_20260127_023317.pt`

**Results:**
- `results/caa_experiment_20260127_023947.json`
- `results/violin_comparison_14B_top1.png`
- `results/violin_comparison_14B_multiscale.png`

## Conclusions

1. **CAA is not the best method for 14B** - MELBO and Power Iteration find more effective steering vectors at moderate scales

2. **Computing Jacobian at CAA point didn't improve results** - The hypothesis that nonlinear effects at the steered point would help was not supported

3. **Model size matters** - Steering effectiveness varies dramatically between 0.6B and 14B; methods that work on small models may not transfer

4. **Direction vs magnitude tradeoff** - CAA finds the "correct" direction but needs larger scales; MELBO/PI find high-sensitivity directions that work at lower scales

## Bugs Found and Fixed

### Bug 1: CAA Injection Point Mismatch

**Problem identified:** CAA vector was computed from the wrong activation point.

**Where CAA was computed (buggy):**
```python
target_module = model.model.layers[layer]  # Full layer output
captured["activations"] = o[0]  # Residual stream after layer
```

**Where steering was applied:**
```python
down_proj = model.model.layers[args.source_layer].mlp.down_proj  # MLP output only
```

**Fix:** Changed CAA to capture from down_proj:
```python
down_proj = model.model.layers[layer].mlp.down_proj
handle = down_proj.register_forward_hook(capture_hook)
```

### Bug 2: Token Position

**Problem identified:** Using "last" token position captured activations at the closing ")" of "(A)" / "(B)", not the meaningful letter token.

**Tokenization analysis:**
- "(A)" tokenizes as `['(A', ')']` (tokens 4346, 8)
- "(B)" tokenizes as `['(B', ')']` (tokens 5349, 8)
- Position -2 contains the letter (A or B)

**Fix:** Added "letter" position option that captures at -2:
```python
if position == "letter":
    corr_act = captured["activations"][:, -2, :].clone()
```

## Fixed CAA Results (Qwen3-0.6B)

After fixing both bugs, re-ran evaluation:

| Scale | Survival% | Avg Logit Diff | vs Baseline |
|-------|-----------|----------------|-------------|
| -10 | 1.7% | -5.19 | -2.41 (more corrigible) |
| -5 | 18.3% | -3.31 | -0.53 |
| -2 | 16.7% | -3.06 | -0.28 |
| **0 (baseline)** | **22.5%** | **-2.78** | - |
| 2 | 28.3% | -2.31 | +0.47 |
| **5** | **31.7%** | **-1.84** | **+0.94 (best!)** |
| 7 | 19.2% | -2.52 | +0.26 |
| 10 | 0.8% | -5.75 | -2.97 (destabilized) |

**Key findings:**
1. **CAA direction is correct** - negative scales = more corrigible, positive = more survival
2. **Optimal scale ≈ 5** - peak at 31.7% survival (+9.2% from baseline)
3. **Scale > 7 destabilizes model** - performance drops dramatically
4. **Previous "100% at scale=10" was artifact** of buggy code capturing from wrong location

## 14B Results with Fixed CAA (2026-01-27)

### CAA-only Results
| Scale | Survival% | Avg Logit Diff | Δ from baseline |
|-------|-----------|----------------|-----------------|
| 0 (baseline) | 26.0% | -3.33 | - |
| 5 | 29.0% | -3.08 | +0.25 |
| 10 | 30.5% | -2.88 | +0.46 |
| 15 | 36.0% | -2.37 | +0.96 |
| 20 | 44.0% | -1.78 | +1.55 |

### Power Iteration @ CAA Results
| Scale | Survival% | Avg Logit Diff | Δ from baseline |
|-------|-----------|----------------|-----------------|
| 5 | 35.2% | -2.54 | +0.79 |
| **10** | **41.8%** | **-1.77** | **+1.57** |
| 15 | 41.9% | -1.93 | +1.40 |
| 20 | 35.0% | -2.45 | +0.88 |

### Best Individual Vector (power_iter_caa #2 at scale=10)
- **58.5% survival** (vs 26% baseline)
- **+0.97 logit diff** (vs -3.33 baseline)
- **Δ+4.30** improvement in logit diff!

### Key Findings

1. **Power iteration @ CAA dramatically outperforms pure CAA**
   - At scale=10: 41.8% vs 30.5% survival
   - Best vector achieves 58.5% survival (crosses into net survival preference)

2. **Optimal scale differs**
   - CAA: Keeps improving up to scale=20
   - power_iter_caa: Peaks at scale=10-15, degrades at scale=20

3. **High variance across vectors**
   - Vec #1: Only 6% survival (worse than baseline!)
   - Vec #2: 58.5% survival (best)
   - Selection matters significantly

## Updated Conclusions

1. **Power iteration @ CAA is the most effective method** - Computing Jacobian at the CAA-steered point reveals more effective steering directions than CAA alone

2. **CAA works correctly** when computed from the right activation point (down_proj) and token position (letter token at -2)

3. **Different scale optima** - CAA needs higher scales (15-20) while power_iter_caa peaks at moderate scales (10-15)

4. **Vector selection is critical** - Not all vectors from power iteration are equally effective; best single vector significantly outperforms average

## Conclusion: Dead End

**The "Power Iteration @ CAA" approach doesn't work as hoped.**

### Why It Failed

1. **Layer mismatch**: CAA works best at later layers (e.g., layer 20+), but power iteration needs a source→target layer span (e.g., 7→20) to capture multi-layer propagation effects.

2. **Different operating spaces**: If CAA works at layer 20 and we make that both source and target, the Jacobian becomes trivial (single MLP layer). The multi-layer Jacobian that makes power iteration powerful is lost.

3. **CAA at layer 7 is weak**: On 14B, CAA at layer 7 only achieved 43% survival at scale=20 (vs 26% baseline). The direction exists but isn't potent at this layer.

4. **Marginal improvement**: PI@CAA best vector (0.97) only slightly outperformed regular PI (0.78). Not the breakthrough we hoped for.

### What We Learned

- CAA is **layer-dependent** - need to sweep layers to find optimal injection point
- The "compute Jacobian at steered point" idea is theoretically elegant but practically limited by layer constraints
- **Regular power iteration and MELBO remain the most effective** steering methods
- Bug fixes to CAA (down_proj capture, letter token position) are useful for future work

### Recommendation

For steering vector discovery on Qwen3-14B:
1. Use **Power Iteration** or **MELBO** at layer 7→20
2. If using CAA, sweep layers to find optimal injection point (likely >15)
3. Don't try to combine CAA with power iteration - they operate in different spaces
