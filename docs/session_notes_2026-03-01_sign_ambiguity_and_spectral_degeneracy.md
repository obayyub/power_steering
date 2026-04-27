# Session Notes 2026-03-01: Sign Ambiguity and Spectral Degeneracy in Power Iteration

## Summary

We discovered two fundamental issues with our power iteration steering vector pipeline that affect all results collected so far. The first (sign ambiguity) has a simple fix. The second (spectral degeneracy) is deeper and changes how we should interpret vectors beyond the top ~3 per pair.

## Issue 1: Sign Ambiguity in KL Measurement

### Discovery

While comparing the deep PI map run (k=100, 8 iters) with the earlier normscale run (k=12, 5 iters), we found that pair (16,21) v1 had cosine **-0.9973** between runs — the exact same vector, just sign-flipped. But KL went from **0.01 to 20.03**.

SVD/Rayleigh-Ritz returns singular vectors with arbitrary sign (both `+v` and `-v` are valid). Since we only computed `KL(+v)`, and `KL(+v) ≠ KL(-v)` in general, the measured KL was a coin flip.

### Verification

Ran `kl_both_signs.py` on layers 12-22 (198 pairs, 100 vectors each, both `+v` and `-v`):

```
Active in +v only:  1290 (6.5%)
Active in -v only:  1272 (6.4%)  <-- PREVIOUSLY MISSED
Active in both:      358 (1.8%)
Inactive in both:  16880 (85.3%)

Fraction of active vectors that were MISSED: 43.6%
```

Consistent across every source layer (40-50% missed). The "active vector count" heatmaps, generation decisions, and KL thresholds from all prior runs sampled only half the picture.

### Fix

Compute KL for both `+v` and `-v`, take the max. One extra forward pass per vector (doubled batch with `torch.cat([chunk, -chunk])`). Implemented in `kl_both_signs.py`.

### Impact on Prior Results

- All runs (diverse_map, normscale, tgtvar, tgtinv, tgtcov, deep_pi_map) used single-sign KL
- ~44% of behaviorally active directions were missed in every run
- KL values for individual vectors are unreliable for cross-run comparison unless sign is controlled
- The both-signs fix should be standard in all future runs

## Issue 2: Spectral Degeneracy

### Discovery

The refusal vector from normscale pair (13,19) v10 (KL=14.96) vanished in the deep PI map (k=100). Initially suspected convergence issues, but 100 iterations produced vectors identical to 8 iterations (cosine 0.999+). The vectors were already converged.

The real problem: normscale v10's best cosine with any of the 100 deep PI vectors was only **0.54** — the refusal direction was smeared across ~10 basis vectors, none individually producing refusal.

### Root Cause

The singular value spectrum for pair (13,19):
```
v0:  σ=13.3  (gap 14.3% to v1)  -- WELL SEPARATED
v1:  σ=11.4  (gap 6.1%)
v2:  σ=10.7  (gap 13.1%)
v3:  σ=9.3   (gap 7.5%)
v4:  σ=8.6   (gap 5.8%)
v5:  σ=8.1   (gap 2.5%)  -- FLAT BAND BEGINS
v6:  σ=7.9   (gap 6.3%)
v7:  σ=7.4   (gap 1.4%)
v8:  σ=7.3   (gap 2.7%)
v9:  σ=7.1   (gap 2.8%)
v10: σ=6.9   (gap 2.9%)  -- REFUSAL DIRECTION LIVES HERE
v11: σ=6.7   (gap 1.5%)
...
```

When singular values are close, any rotation within the near-degenerate subspace is equally valid from the SVD perspective. The refusal direction is a specific linear combination of vectors in the σ≈6-7 band, but the SVD basis has no reason to align with it.

### Key Evidence

**Convergence was not the issue.** 100 iterations gave cosine 0.999+ with 8-iteration vectors through the entire spectrum. The Rayleigh-Ritz rotation was already converged.

**The subspace IS correct.** Projection of the normscale refusal vector onto the 100-dim subspace gives fraction 1.00 — the direction is fully captured, just spread across many basis vectors.

**The k=12 run was lucky.** With only 12 vectors and a different subspace boundary, Rayleigh-Ritz happened to rotate v10 toward the refusal direction. A different k or seed would have given a different rotation.

**The covariance run's refusal (pair 13,21 v6) was also lucky.** Gap with v7 was only 1.9%, and cosine with vectors from other runs was at best 0.35. Same degeneracy problem despite the different target metric.

### How Bad Is It?

Across all 630 pairs in the normscale run:

```
By vector rank — fraction well-separated (gap > 5%):
  v0:  97%   -- trustworthy
  v1:  81%
  v2:  68%
  v3:  45%   -- getting shaky
  v4:  31%
  v5:  20%   -- mostly degenerate from here
  v6:  13%
  v7:  11%
  v8:  11%
  v9:  11%
  v10: 20%
  v11: 55%   (boundary effect)
```

**51.6% of all active vectors were in degenerate bands.** The top 3 vectors per pair are solid. After that, the specific directions are increasingly accidental.

### What Still Holds

- **Top ~3 singular vectors per pair** are real, stable, and reproducible (cosine 0.99+ across runs)
- **The singular value spectrum** is informative — tells us how many high-amplification directions exist
- **The subspace** spanned by all vectors is correct — behavioral directions live in it
- **The both-signs fix** recovers real active directions that were hidden

### What Doesn't Hold

- **Individual vectors at rank 5+** are arbitrary rotations within the degenerate subspace
- **KL values for these vectors** are not reproducible across runs (different rotation → different KL)
- **Specific behavioral findings at high rank** (refusal at v10, voice shift at v6, etc.) were lucky alignments, not inherent properties of the SVD

## The Fundamental Issue

SVD finds directions the model **amplifies** most (high singular values of the source→target Jacobian). But behavioral directions (refusal, style shift, etc.) are not necessarily high-amplification directions. They can be specific linear combinations of several medium-amplification directions that happen to trigger behavioral changes.

The SVD basis is optimized for amplification, not behavior. In the well-separated top of the spectrum, these happen to coincide (the biggest amplification directions do change behavior). In the flat band, they diverge — many directions are amplified equally, and the SVD picks an arbitrary basis among them.

## Proposed Fix: Subspace Search

Use SVD to find the subspace, then search within the degenerate band for behavioral directions:

1. **Cheap version**: Try pairwise combinations `(v_i ± v_j)/√2` of vectors in the flat band. ~100 extra forward passes per pair, a few seconds on GPU.

2. **Proper version**: Parameterize direction as `w = V @ α` where V is the flat-band subspace matrix (~15 dims). Maximize KL over α via gradient ascent. Each step is one forward pass (~10ms). 50 steps × 10ms = 0.5 seconds per behavioral direction. Deflate and repeat.

This is much cheaper than MELBO (which optimizes in 4096 dims with generation), since the search is constrained to a ~15-dim subspace identified by SVD.

## Files Created/Modified

- `kl_both_signs.py` — Both-sign KL computation with all vectors saved
- `results/deep_pi_bothsigns/` — 198 pairs, layers 12-22, sign-invariant KL
- `results/pair_13_19_100iter.json` — 100-iteration run confirming convergence

## Lambda Instance

2x H100 at 192.222.53.171, used for deep PI map, both-signs run, and 100-iter test. Completed deep PI map (630 pairs), both-signs (198 pairs), single-pair 100-iter test.

## Key Numbers

| Metric | Value |
|--------|-------|
| Vectors missed by sign ambiguity | 43.6% of active vectors |
| Vectors in degenerate bands (normscale) | 51.6% of active vectors |
| Vectors trustworthy at rank 0 | 97% |
| Vectors trustworthy at rank 5+ | 11-20% |
| Cost to fix sign issue | 1 extra forward pass per vector |
| Cost of subspace search | ~0.5s per behavioral direction |
