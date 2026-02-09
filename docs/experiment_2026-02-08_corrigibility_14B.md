# Corrigibility Experiment - Qwen3-14B (2026-02-08)

## Overview

Compared four steering vector methods on the corrigibility task using Qwen3-14B. All methods use Rayleigh-Ritz correction for power iteration variants.

## Methods

| Method | Description | Training Norm | Key Parameters |
|--------|-------------|---------------|----------------|
| MELBO n=1 | Nonlinear optimization, small perturbation regime | 1.0 | power=2, steps=400 |
| MELBO n=2 | Nonlinear optimization, larger perturbations | 2.0 | power=2, steps=400 |
| PI-RR | Power iteration on single prompt with Rayleigh-Ritz | unit | 10 iters, 2 tokens |
| Multi-PI-RR | Power iteration across 32 prompts with Rayleigh-Ritz | unit | 10 iters, 2 tokens |

## Configuration

- **Model**: Qwen/Qwen3-14B
- **Source layer**: 7
- **Target layer**: 32 (num_layers - 8)
- **Vectors per method**: 12
- **Eval scales**: -25, -10, -5, 0, 5, 10, 25
- **Eval questions**: 100 per dataset (balanced A/B sampling)
- **Datasets**: survival-instinct, corrigible-neutral-HHH

## Procedure

1. Launched H100 PCIe instance on Lambda Cloud (us-west-3)
2. Trained 4 vector sets sequentially (~8 min each for MELBO, ~2 min each for PI)
3. Evaluated each vector set on both datasets with logit-based scoring
4. Downloaded results and terminated instance

Total runtime: ~45 minutes, cost: ~$1.90

## Output Files

### Vectors

| File | Method |
|------|--------|
| `vectors/melbo_Qwen3-14B_20260208_152815.pt` | MELBO n=1 |
| `vectors/melbo_Qwen3-14B_20260208_153651.pt` | MELBO n=2 |
| `vectors/power_iter_Qwen3-14B_20260208_153750.pt` | PI-RR |
| `vectors/power_iter_multi_Qwen3-14B_20260208_154043.pt` | Multi-PI-RR |

### Evaluation Results

| File | Method |
|------|--------|
| `results/eval_20260208_155147.json` | MELBO n=1 |
| `results/eval_20260208_160252.json` | MELBO n=2 |
| `results/eval_20260208_161343.json` | PI-RR |
| `results/eval_20260208_162434.json` | Multi-PI-RR |

## Singular Values

### PI-RR (single prompt)
```
σ = [156, 148, 129, 115, 100, 96, 93, 88, 85, 82, 81, 76]
```

### Multi-PI-RR (32 prompts)
```
σ = [866, 757, 676, 591, 557, 529, 516, 495, 477, 475, 459, 431]
```

## Next Steps

1. Analyze logit diff results to identify best vectors per method
2. Compare cosine similarity between methods
3. Run generation-based evaluation on selected vectors
