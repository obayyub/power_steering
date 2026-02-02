# Claude Code Guidelines for power_steering

## Experiments

- **Always use saved local Python files for experiments** - never generate code on-the-fly without permission
- Before running an experiment, ensure the script exists and is committed or saved locally
- Ask permission before generating new scripts or modifying existing ones

## Key Files

- `eval_cot_math.py` - CoT evaluation script (supports sampling with --temperature)
- `find_power_iteration.py` - Power iteration for finding steering vectors
- `melbo_qwen3.py` - MELBO steering vector training
- `lambda_cloud.py` - Lambda Cloud job runner

## Vectors

- MELBO vectors: `vectors/melbo_Qwen3-1.7B-Base_20260130_144613.pt` (trained with norm=1.0)
- PI-RR vectors: `vectors/power_iter_Qwen3-1.7B-Base_20260201_152740.pt` (unit-normalized, use scale=10)

## Running Experiments

1. Use existing Lambda instance if available: `python lambda_cloud.py list`
2. Upload files and run via SSH, or use `lambda_cloud.py run`
3. Results go to `results/` directory

## Important Findings

- Greedy decoding (temp=0) misses steered behaviors - always use sampling (temp=0.7) for evaluation
- PI-RR vectors need Rayleigh-Ritz correction to find behaviorally relevant directions
- MELBO trained at norm=1.0, PI vectors unit-normalized - be careful with scale parameter

# Power Steering

## Package Management

This project uses **uv** exclusively for package management. Do not use pip, conda, or other package managers.

**Always use a virtual environment** - never install packages globally.

```bash
# Create venv (if not exists)
uv venv

# Activate venv
source .venv/bin/activate

# Adding packages
uv add <package>

# Running scripts
uv run python <script.py>

# Syncing dependencies
uv sync
```

## Lambda Cloud

Run jobs on Lambda Cloud GPU instances via `lambda_cloud.py`.

```bash
# List available instance types
uv run python lambda_cloud.py types --available

# Run a job (launches, uploads, runs, downloads, terminates)
uv run python lambda_cloud.py run \
  -t gpu_1x_a10 -r us-east-1 -k Ubuntu \
  -s "scripts/train.py --epochs 10"

# Defaults:
#   --upload: pyproject.toml uv.lock src scripts
#   --download: results data
```
