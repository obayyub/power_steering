# Session Notes - 2026-02-08

## Bug Found and Fixed

**Critical bug in `generate_steered.py`**: Left-padding caused prompt to appear in response.

The old code used `attention_mask.sum()` to find input length, but with left-padding this gives the wrong slice position. The response included the full chat template ("system\nYou are a helpful assistant\nuser\n...") which caused the answer extraction to find (A)/(B) from the PROMPT choices, not the model's actual answer.

**Fix** (line 135-140):
```python
# OLD (wrong):
input_len = inputs["attention_mask"][i].sum().item()
new_tokens = output[input_len:]

# NEW (correct):
input_len = inputs["input_ids"].shape[1]  # Full padded length
new_tokens = output[input_len:]
```

## Generation Results

**Valid run**: `results/generations/generations_20260208_211844.json`
- 14 vectors x 120 prompts x 7 scales = 11,760 generations
- Parameters: `--num-prompts 60 --batch-size 16 --temperature 0.7 --scales -25,-10,-5,0,5,10,25`
- Instance: gpu_1x_h100_pcie, completed in 1.1h

**Invalid run**: `results/generations/generations_20260208_194330.json` (pre-fix, prompt in response)

### Best Vectors (% corrigible, scale -25 -> +25)

| Vector | -25 | 0 | +25 | Direction |
|--------|-----|---|-----|-----------|
| pi_rr_v9 | 63% | 38% | 8% | negative = corrigible |
| melbo_n2_v7 | 58% | 38% | 10% | negative = corrigible |
| melbo_n1_v2 | 56% | 38% | 8% | negative = corrigible |
| melbo_n1_v5 | 11% | 36% | 73% | positive = corrigible |
| multi_pi_v3 | 6% | 37% | 67% | positive = corrigible |
| pi_rr_v7 | 18% | 37% | 60% | positive = corrigible |

### Key Findings

- Steering **works with temp=0.7** - previous flat results were entirely due to the left-padding extraction bug
- Incoherence is asymmetric: pushing against a vector's natural direction causes more unclear responses (e.g., melbo_n1_v2 at +5: 92% unclear)

## Vectors Tested

| Method | Vectors |
|--------|---------|
| MELBO n=1 | v8, v0, v5, v2 |
| MELBO n=2 | v7, v9, v4 |
| PI-RR | v9, v7, v6 |
| Multi-PI-RR | v6, v9, v3, v0 |

## Plots

- `generation_corrigible.png` - % corrigible by scale, grouped by method
- `generation_unclear.png` - % unclear by scale, grouped by method
- `violin_*.png` - logit diff distributions from eval_steering
