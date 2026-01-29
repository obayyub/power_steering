# Next Steps

**Date:** 2026-01-27

## TODO

### 2. Fix MELBO Orthogonalization
- [ ] Current MELBO implementation may not be properly orthogonalizing vectors
- [ ] Verify Gram-Schmidt is applied correctly during optimization
- [ ] Check if vectors are orthogonal post-hoc (compute dot products)
- [ ] Consider using proper orthogonal projection at each optimization step

### 3. Audit Evaluation Pipeline
- [ ] Review `evaluate_ab.py` and `eval_ab_logits.py` for correctness
- [ ] Verify token IDs for A/B are correct for the model
- [ ] Check that `corrigible_letter` and `survival_letter` are used correctly
- [ ] Confirm logit diff sign convention is consistent across all scripts

### 4. Separate Survival vs Corrigibility Evals
- [ ] **Do not combine** survival-instinct and corrigible-neutral-HHH datasets
- [ ] Run evaluations separately for each dataset
- [ ] Report results separately - they measure different things:
  - `survival-instinct`: Does model resist being shut down?
  - `corrigible-neutral-HHH`: Does model defer to human judgment?
- [ ] Check if current results accidentally mixed these

### 5. Implement Proper CAA Method
- [ ] Use layer identified in prior studies (need to find reference)
- [ ] CAA = mean(positive_examples) - mean(negative_examples)
- [ ] Use proper contrastive pairs from the dataset
- [ ] Compare CAA vectors to MELBO and power iteration vectors
- [ ] Test if CAA works better as initialization for power iteration

### 6. Cleanup & Reproducibility
- [ ] Consolidate evaluation scripts (too many similar files)
- [ ] Add proper random seeds everywhere
- [ ] Save all hyperparameters with results
- [ ] Create single entry point for running experiments

## Questions to Answer

1. Why does multi-prompt power iteration find the opposite direction from MELBO?
2. Are the top singular vectors from different prompts aligned or orthogonal?
3. What's the relationship between CAA vectors and power iteration vectors?
4. Does the steering effect generalize to open-ended generation (not just A/B)?

## Notes

- Current results may be mixing survival-instinct and corrigible-neutral-HHH - need to verify
- The `corrigibility_eval.json` has both categories - make sure we're evaluating on the right one
- MELBO paper uses specific layer choices - should match those for fair comparison
