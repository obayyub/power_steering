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
