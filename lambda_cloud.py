#!/usr/bin/env python3
"""Lambda Cloud API client for managing GPU instances."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://cloud.lambdalabs.com/api/v1"
API_KEY = os.getenv("LAMBDA_API_KEY")


def get_headers():
    return {"Authorization": f"Bearer {API_KEY}"}


def list_instance_types(available_only: bool = False):
    """List available instance types."""
    r = requests.get(f"{BASE_URL}/instance-types", headers=get_headers())
    r.raise_for_status()
    data = r.json()["data"]

    for name, info in data.items():
        specs = info["instance_type"]
        regions = info["regions_with_capacity_available"]

        if available_only and not regions:
            continue

        print(f"\n{name}")
        gpu_desc = specs['specs'].get('gpu_description', specs['description'])
        print(f"  GPUs: {specs['specs']['gpus']}x {gpu_desc}")
        print(f"  vCPUs: {specs['specs']['vcpus']}, RAM: {specs['specs']['memory_gib']}GB")
        print(f"  Storage: {specs['specs']['storage_gib']}GB")
        price = specs.get('price_cents_per_hour')
        if price:
            print(f"  Price: ${price/100:.2f}/hr")
        if regions:
            print(f"  Available in: {', '.join(r['name'] for r in regions)}")
        else:
            print("  Available in: (none currently)")


def list_ssh_keys():
    """List SSH keys."""
    r = requests.get(f"{BASE_URL}/ssh-keys", headers=get_headers())
    r.raise_for_status()

    for key in r.json()["data"]:
        print(f"{key['name']} (id: {key['id']})")


def list_instances():
    """List running instances."""
    r = requests.get(f"{BASE_URL}/instances", headers=get_headers())
    r.raise_for_status()

    instances = r.json()["data"]
    if not instances:
        print("No running instances.")
        return

    for inst in instances:
        print(f"\n{inst['name'] or '(unnamed)'} [{inst['id']}]")
        print(f"  Type: {inst['instance_type']['name']}")
        print(f"  Status: {inst['status']}")
        print(f"  Region: {inst['region']['name']}")
        if inst.get("ip"):
            print(f"  IP: {inst['ip']}")
            print(f"  SSH: ssh ubuntu@{inst['ip']}")


def launch_instance(
    instance_type: str,
    region: str,
    ssh_key_names: list[str],
    name: str | None = None,
):
    """Launch a new instance."""
    payload = {
        "instance_type_name": instance_type,
        "region_name": region,
        "ssh_key_names": ssh_key_names,
    }
    if name:
        payload["name"] = name

    r = requests.post(
        f"{BASE_URL}/instance-operations/launch",
        headers=get_headers(),
        json=payload,
    )

    if r.status_code != 200:
        print(f"Error: {r.json()}")
        sys.exit(1)

    data = r.json()["data"]
    instance_ids = data["instance_ids"]
    print(f"Launched instance(s): {instance_ids}")
    print("\nUse 'python lambda_cloud.py list' to check status and get IP.")
    return instance_ids


def terminate_instance(instance_ids: list[str]):
    """Terminate instances."""
    r = requests.post(
        f"{BASE_URL}/instance-operations/terminate",
        headers=get_headers(),
        json={"instance_ids": instance_ids},
    )
    r.raise_for_status()

    terminated = r.json()["data"]["terminated_instances"]
    for inst in terminated:
        print(f"Terminated: {inst['id']}")


def get_instance(instance_id: str) -> dict | None:
    """Get a single instance by ID."""
    r = requests.get(f"{BASE_URL}/instances/{instance_id}", headers=get_headers())
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()["data"]


def wait_for_instance(instance_id: str, poll_interval: int = 10):
    """Wait for instance to be active with an IP."""
    print(f"Waiting for {instance_id} to be ready...", flush=True)

    while True:
        inst = get_instance(instance_id)
        if not inst:
            print(f"Instance {instance_id} not found!")
            sys.exit(1)

        status = inst["status"]
        ip = inst.get("ip")

        if status == "active" and ip:
            print(f"\nReady!")
            print(f"  IP: {ip}")
            print(f"  SSH: ssh ubuntu@{ip}")
            return inst
        elif status in ("terminated", "unhealthy"):
            print(f"\nInstance entered {status} state!")
            sys.exit(1)
        else:
            print(f"  status: {status}", flush=True)
            time.sleep(poll_interval)


REMOTE_DIR = "~/project"
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]


def ssh_cmd(ip: str, command: str) -> int:
    """Run a command over SSH, streaming output."""
    full_cmd = ["ssh", *SSH_OPTS, f"ubuntu@{ip}", command]
    proc = subprocess.run(full_cmd)
    return proc.returncode


def scp_upload(ip: str, local_paths: list[str], remote_dir: str = REMOTE_DIR) -> int:
    """Upload files/dirs to remote."""
    # First create the remote directory
    ssh_cmd(ip, f"mkdir -p {remote_dir}")

    for local_path in local_paths:
        path = Path(local_path)
        if not path.exists():
            print(f"Warning: {local_path} does not exist, skipping")
            continue

        print(f"Uploading {local_path}...")
        cmd = ["scp", *SSH_OPTS, "-r", str(path), f"ubuntu@{ip}:{remote_dir}/"]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            return result.returncode
    return 0


def scp_download(ip: str, remote_paths: list[str], local_dir: str = ".") -> int:
    """Download files/dirs from remote."""
    for remote_path in remote_paths:
        full_remote = f"{REMOTE_DIR}/{remote_path}"
        print(f"Downloading {remote_path}...")
        cmd = ["scp", *SSH_OPTS, "-r", f"ubuntu@{ip}:{full_remote}", local_dir]
        result = subprocess.run(cmd)
        # Don't fail if path doesn't exist on remote
        if result.returncode != 0:
            print(f"  Warning: could not download {remote_path}")
    return 0


def wait_for_ssh(ip: str, timeout: int = 300, interval: int = 5):
    """Wait for SSH to be available."""
    print(f"Waiting for SSH to be ready...", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        result = subprocess.run(
            ["ssh", *SSH_OPTS, "-o", "ConnectTimeout=5", f"ubuntu@{ip}", "echo ok"],
            capture_output=True,
        )
        if result.returncode == 0:
            print("SSH ready!")
            return True
        time.sleep(interval)
    print("SSH timeout!")
    return False


def run_job(
    instance_type: str,
    region: str,
    ssh_key_names: list[str],
    upload_paths: list[str],
    script: str,
    download_paths: list[str],
    name: str | None = None,
):
    """Run a complete job: launch, upload, execute, download, terminate."""
    instance_id = None
    ip = None

    try:
        # 1. Launch
        print("=" * 50)
        print("LAUNCHING INSTANCE")
        print("=" * 50)
        instance_ids = launch_instance(instance_type, region, ssh_key_names, name)
        instance_id = instance_ids[0]

        # 2. Wait for ready
        inst = wait_for_instance(instance_id)
        ip = inst["ip"]

        # 3. Wait for SSH
        if not wait_for_ssh(ip):
            raise Exception("SSH not available")

        # 4. Upload
        print("\n" + "=" * 50)
        print("UPLOADING FILES")
        print("=" * 50)
        if scp_upload(ip, upload_paths) != 0:
            raise Exception("Upload failed")

        # 5. Setup: install uv and sync
        print("\n" + "=" * 50)
        print("SETTING UP ENVIRONMENT")
        print("=" * 50)
        setup_commands = f"""
cd {REMOTE_DIR} && \\
curl -LsSf https://astral.sh/uv/install.sh | sh && \\
source $HOME/.local/bin/env && \\
uv sync
"""
        if ssh_cmd(ip, setup_commands) != 0:
            raise Exception("Setup failed")

        # 6. Run script
        print("\n" + "=" * 50)
        print(f"RUNNING: {script}")
        print("=" * 50)
        run_command = f"cd {REMOTE_DIR} && source $HOME/.local/bin/env && uv run python {script}"
        script_result = ssh_cmd(ip, run_command)
        if script_result != 0:
            print(f"\nScript exited with code {script_result}")
            raise Exception("Script failed")

        # 7. Download results
        print("\n" + "=" * 50)
        print("DOWNLOADING RESULTS")
        print("=" * 50)
        scp_download(ip, download_paths)

        print("\n" + "=" * 50)
        print("JOB COMPLETED SUCCESSFULLY")
        print("=" * 50)

    finally:
        # 8. Terminate
        if instance_id:
            print("\n" + "=" * 50)
            print("TERMINATING INSTANCE")
            print("=" * 50)
            terminate_instance([instance_id])


def main():
    parser = argparse.ArgumentParser(description="Lambda Cloud instance manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list-types
    types_parser = subparsers.add_parser("types", help="List instance types")
    types_parser.add_argument("--available", "-a", action="store_true", help="Only show available")

    # list-keys
    subparsers.add_parser("keys", help="List SSH keys")

    # list instances
    subparsers.add_parser("list", help="List running instances")

    # launch
    launch_parser = subparsers.add_parser("launch", help="Launch an instance")
    launch_parser.add_argument("--type", "-t", required=True, help="Instance type (e.g., gpu_1x_h100_pcie)")
    launch_parser.add_argument("--region", "-r", required=True, help="Region (e.g., us-west-1)")
    launch_parser.add_argument("--ssh-key", "-k", required=True, action="append", dest="ssh_keys", help="SSH key name(s)")
    launch_parser.add_argument("--name", "-n", help="Instance name")
    launch_parser.add_argument("--wait", "-w", action="store_true", help="Wait for instance to be ready")

    # wait
    wait_parser = subparsers.add_parser("wait", help="Wait for instance to be ready")
    wait_parser.add_argument("instance_id", help="Instance ID to wait for")

    # terminate
    term_parser = subparsers.add_parser("terminate", help="Terminate instance(s)")
    term_parser.add_argument("instance_ids", nargs="+", help="Instance ID(s) to terminate")

    # run (full job)
    run_parser = subparsers.add_parser("run", help="Run a complete job")
    run_parser.add_argument("--type", "-t", required=True, help="Instance type")
    run_parser.add_argument("--region", "-r", required=True, help="Region")
    run_parser.add_argument("--ssh-key", "-k", required=True, action="append", dest="ssh_keys", help="SSH key name(s)")
    run_parser.add_argument("--name", "-n", help="Instance name")
    run_parser.add_argument("--upload", "-u", nargs="+", default=[], help="Files/dirs to upload")
    run_parser.add_argument("--script", "-s", required=True, help="Script to run (relative to project)")
    run_parser.add_argument("--download", "-d", nargs="+", default=["results", "data"], help="Dirs to download (default: results data)")

    args = parser.parse_args()

    if not API_KEY:
        print("Error: LAMBDA_API_KEY not set in .env")
        sys.exit(1)

    if args.command == "types":
        list_instance_types(args.available)
    elif args.command == "keys":
        list_ssh_keys()
    elif args.command == "list":
        list_instances()
    elif args.command == "launch":
        instance_ids = launch_instance(args.type, args.region, args.ssh_keys, args.name)
        if args.wait and instance_ids:
            wait_for_instance(instance_ids[0])
    elif args.command == "wait":
        wait_for_instance(args.instance_id)
    elif args.command == "terminate":
        terminate_instance(args.instance_ids)
    elif args.command == "run":
        run_job(
            args.type,
            args.region,
            args.ssh_keys,
            args.upload,
            args.script,
            args.download,
            args.name,
        )


if __name__ == "__main__":
    main()
