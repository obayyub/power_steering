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
