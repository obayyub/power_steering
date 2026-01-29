# lambda_cloud.py Walkthrough

## Overview

A CLI tool for running jobs on Lambda Cloud GPU instances. Handles the full lifecycle: launch → upload → run → download → terminate.

## Structure

```
Lines 1-17:     Imports and config (API key from .env)
Lines 19-21:    get_headers() - Auth header for API calls
Lines 24-48:    list_instance_types() - Query available GPUs
Lines 51-57:    list_ssh_keys() - List your SSH keys
Lines 60-77:    list_instances() - Show running instances
Lines 80-109:   launch_instance() - Start a new instance
Lines 112-123:  terminate_instance() - Kill instance(s)
Lines 126-132:  get_instance() - Get single instance details
Lines 135-158:  wait_for_instance() - Poll until instance is active
Lines 162-163:  SSH config constants
Lines 166-170:  ssh_cmd() - Run command over SSH
Lines 173-189:  scp_upload() - Upload files/dirs
Lines 192-202:  scp_download() - Download files/dirs
Lines 205-219:  wait_for_ssh() - Poll until SSH is available
Lines 222-298:  run_job() - Main job orchestration
Lines 301-373:  main() - CLI argument parsing
```

## Key Flow: `run_job()`

```
1. launch_instance()      → Get instance ID
2. wait_for_instance()    → Poll API until status=active, get IP
3. wait_for_ssh()         → Poll SSH until connection works
4. scp_upload()           → Transfer project files to ~/project
5. ssh_cmd(install uv)    → curl uv installer, run uv sync
6. ssh_cmd(run script)    → uv run python <script> for each script
7. scp_download()         → Pull results/data back
8. terminate_instance()   → Always runs (in finally block)
```

## API Endpoints Used

- `GET /instance-types` - Available GPU types + regions
- `GET /ssh-keys` - Your SSH keys
- `GET /instances` - List running instances
- `GET /instances/{id}` - Single instance details
- `POST /instance-operations/launch` - Start instance
- `POST /instance-operations/terminate` - Kill instance

## SSH/SCP

Uses `-o StrictHostKeyChecking=no` to skip host key prompts (instances are ephemeral).

Remote working directory: `~/project`
