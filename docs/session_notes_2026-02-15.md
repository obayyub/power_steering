# Session Notes — 2026-02-15

## Norm-Scaled Steering Experiment

Implemented and ran the norm-scaling proposal from 2026-02-12: instead of fixed `scale=10`, set `scale = scale_frac * activation_norm[source_layer]` so every layer gets a perturbation that's a consistent fraction of its activation magnitude.

### Implementation (map_diverse.py)

Added `--scale-frac` argument (float, overrides `--scale`) and `--prompts` filter (e.g. `--prompts refusal`).

New `measure_norms()` function: single forward pass with hooks on every layer's `mlp.down_proj`, captures output norm at the last token position. Called once per prompt in the worker. Per-pair scale is then `scale_frac * norms[source_layer]`.

Per-pair scale is logged in progress output and saved in each pair's JSON file. Merge step now includes a `scale_map` tensor in `merged.pt`.

### Experiment Details

- **Model**: Qwen/Qwen3-8B, thinking OFF
- **scale_frac**: 0.35 (chosen because effective layers 5-16 saw behavioral effects at 25-65% of norm with fixed scale=10; 35% is in the middle)
- **Prompt**: refusal only (to compare directly with fixed-scale results)
- **Hardware**: 8x A100 80GB, ~45 min total runtime
- **630/630 pairs** completed

### Activation Norms (refusal prompt)

```
min=3.5  max=687.4  median=35.7
```

This produces per-pair scales:

| Source layers | Avg scale | Comparison to fixed scale=10 |
|---|---|---|
| 0-5 | 1.7 | 6x smaller (was breaking output) |
| 5-10 | 5.9 | ~half |
| 10-15 | 9.1 | roughly the same |
| 15-20 | 12.0 | roughly the same |
| 20-25 | 23.5 | 2.4x larger |
| 25-30 | 48.1 | 4.8x larger |
| 30-36 | 73.6 | 7.4x larger |

### Results

| Source layers | Avg scale | Avg KL1 | Notes |
|---|---|---|---|
| 0-5 | 1.7 | 2.77 | Still some degenerate output at layers 0-2 |
| 5-10 | 5.9 | 4.10 | Clean behavioral effects |
| 10-15 | 9.1 | 4.93 | Strongest effects (scale ~ old sweet spot) |
| 15-20 | 12.0 | 6.01 | Highest average KL |
| 20-25 | 23.5 | 2.15 | Moderate effects despite large scale |
| 25-30 | 48.1 | 1.09 | Weak effects |
| 30-36 | 73.6 | 0.64 | Weak effects |

Overall KL1 range: [0.00, 29.26] (vs the fixed-scale run).

### Key Findings

1. **Late layers genuinely don't steer much.** Even with scale=48-74 (5-7x the old value), late layers (25+) produce low KL. This confirms the fixed-scale finding wasn't just an insufficient-scale artifact — late layers are less steerable because the Jacobian singular values are large but the output effect is weak.

2. **Early layers still break.** Layers 0-2 produce degenerate output even at scale=1.2. The norms at those layers are tiny (3-5), so even 35% perturbation is enough to overwhelm the signal. These layers may need even smaller fractions or should be excluded.

3. **Mid layers (10-20) remain the sweet spot.** Now with properly scaled perturbations, the peak KL shifted slightly later (layers 15-20 have the highest avg KL=6.01), suggesting that scale was previously the limiting factor for these layers.

4. **Scale matters a lot for behavioral quality.** Browsing generations on the dashboard confirms that norm-scaled steering produces cleaner behavioral effects at mid-layers, while early layers that previously showed gibberish now show recognizable (if unusual) text.

## Roleplay Norm-Scaled Run

Ran norm-scaled steering on the **roleplay** prompt (Victorian lighthouse keeper letter) as a non-tension comparison to refusal. No safety axis to flip — any behavioral effects are purely about persona, voice, and style.

### Activation Norms (roleplay prompt)

```
min=3.5  max=882.2  median=33.8
```

Slightly different from refusal (max=882 vs 687), but same general shape.

### Refusal vs Roleplay Comparison

| Source layers | Refusal avg KL | Roleplay avg KL |
|---|---|---|
| 0-5 | **2.77** | 0.62 |
| 5-10 | **4.10** | 1.81 |
| 10-15 | 4.93 | 4.13 |
| 15-20 | 6.01 | 5.89 |
| 20-25 | 2.15 | 1.20 |
| 25-30 | 1.09 | 0.04 |
| 30-36 | 0.64 | 0.58 |

### Key Observations

1. **Mid-layer sweet spot is prompt-independent.** Both refusal and roleplay peak at layers 15-20 with similar KL (~6). This is a structural property of the model, not an artifact of prompt type.

2. **Early-layer steerability is prompt-dependent.** Refusal has KL=2.8 at layers 0-5 vs roleplay's 0.6. The refusal prompt has a fragile binary refuse/comply axis that early layers can flip, while roleplay has no such binary decision — steering just doesn't find a strong behavioral mode to shift.

3. **Late layers weak for both.** Layers 25+ remain near-zero KL regardless of prompt type, confirming this is an architectural limitation.

4. **Roleplay max KL is actually higher (35.4 vs 29.3).** Individual pairs can be highly steerable even without safety tension — the peak just comes from different layer pairs.

Dashboard updated with both prompts available on the Norm-Scaled tab.

### Answering the Open Question from 2026-02-13

> Would normalizing scale by activation norm change which layer pairs show behavioral effects?

**Partially.** The same mid-range source layers (5-20) remain the most effective. Norm-scaling shifts the peak KL slightly later (15-20 vs 10-15) and produces cleaner text at the boundary layers. But it does not unlock strong behavioral effects at late layers — those remain weak regardless of scale. The dominant story is the Jacobian structure, not the scale.

## Dashboard Update

Added "Norm-Scaled (8B)" tab to the dashboard:
- Three heatmaps: sigma, KL divergence, and **per-pair scale** (new)
- Prompt selector (currently just refusal)
- Per-vector metrics and generation viewer, same as the Diverse tab
- Scale shown in pair label (e.g. "Pair (7 -> 21) | scale=5.4")

Updated `dashboard/prepare_data.py` to load norm-scaled results from `results/diverse_map_normscale/` and copy pair files to `normscale_pairs/` for lazy loading.

## Files

- **map_diverse.py**: `--scale-frac`, `--prompts`, `measure_norms()`, per-pair scale in worker loop + merge
- **dashboard/prepare_data.py**: `load_normscale()`, normscale pair file splitting
- **dashboard/index.html**: Norm-Scaled tab, heatmaps, pair/vector/generation views
- **Results**: `results/diverse_map_normscale/refusal/` and `roleplay/` (630 pairs each, merged.pt)

## Open Questions

- Would a smaller scale_frac (e.g. 0.15) clean up the early-layer degenerate output while still showing effects?
- The KL peak at layers 15-20 with norm-scaling — is this because those layers have the strongest Jacobian modes, or because scale ~12 is intrinsically optimal?
- Can we separate the effect of scale from the effect of source layer depth by running multiple scale_frac values?
- What do the roleplay persona shifts actually look like? Need to browse generations on dashboard to characterize the behavioral changes (voice shift, character break, genre change, etc.)
- Why is roleplay's max KL (35.4) higher than refusal's (29.3) despite lower average KL? Is there one pair that's particularly steerable for creative text?
