#!/usr/bin/env python3
"""
Multi-System Experiment Runner & Sync Tool
Syncs code and runs commands across configured machines.
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

# Constants
CONFIG_PATH = Path("config/machines.yml")
PROJECT_ROOT = Path(__file__).parent.parent

def load_config() -> Dict[str, Any]:
    """Load machine configuration."""
    if not CONFIG_PATH.exists():
        print(f"❌ Config not found: {CONFIG_PATH}")
        sys.exit(1)
        
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config.get("machines", {})

def get_machine_config(config: Dict[str, Any], machine_name: str) -> Dict[str, Any]:
    """Get config for a specific machine."""
    if machine_name not in config:
        print(f"❌ Machine '{machine_name}' not found in config.")
        print(f"   Available: {', '.join(config.keys())}")
        sys.exit(1)
    return config[machine_name]

def run_local(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a local shell command."""
    print(f"💻 Local: {cmd}")
    return subprocess.run(cmd, shell=True, check=check, cwd=PROJECT_ROOT)

def build_ssh_cmd(machine_conf: Dict[str, Any], remote_cmd: Optional[str] = None) -> list:
    """Build SSH command prefix."""
    host = machine_conf.get("ssh_host")
    user = machine_conf.get("ssh_user")
    
    if not host or "192.168.1.X" in host:
        print(f"❌ Invalid SSH host for {machine_conf['name']}. Please update config/machines.yml")
        sys.exit(1)
        
    target = f"{user}@{host}" if user else host
    base = ["ssh", "-o", "StrictHostKeyChecking=no", target]
    
    if remote_cmd:
        base.append(remote_cmd)
        
    return base

def sync_code(machine: str, conf: Dict[str, Any], dry_run: bool = False):
    """Sync code to remote machine using rsync."""
    print(f"\n🔄 Syncing to {machine} ({conf['name']})...")
    
    remote_root = conf.get("remote_root", "~/ai-ml-pipeline")
    host = conf.get("ssh_host")
    user = conf.get("ssh_user", "")
    target = f"{user}@{host}:{remote_root}" if user else f"{host}:{remote_root}"
    
    exclude_flags = "--exclude '.git' --exclude '.pixi' --exclude '__pycache__' --exclude '*.pyc' --exclude 'logs/'"
    
    # Sync core directories
    dirs_to_sync = ["projects/", "config/", "scripts/", "pixi.toml", "pixi.lock"]
    
    for item in dirs_to_sync:
        cmd = f"rsync -avz {exclude_flags} {item} {target}/{item}"
        if dry_run:
            print(f"   [DRY] {cmd}")
        else:
            try:
                subprocess.run(cmd, shell=True, check=True, cwd=PROJECT_ROOT)
            except subprocess.CalledProcessError:
                print(f"❌ Sync failed for {item}")
                return

    print("✅ Sync complete!")

def run_remote(machine: str, conf: Dict[str, Any], command: str):
    """Run command on remote machine."""
    print(f"\n🚀 Running on {machine}: {command}")
    
    remote_root = conf.get("remote_root", "~/ai-ml-pipeline")
    ssh_cmd_list = build_ssh_cmd(conf)
    
    # Wrap command to run in remote root
    full_remote_cmd = f"cd {remote_root} && {command}"
    ssh_cmd_list.append(full_remote_cmd)
    
    try:
        subprocess.run(ssh_cmd_list, check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Remote execution failed")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Multi-System Runner")
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")
    
    # List
    subparsers.add_parser("list", help="List configured machines")
    
    # Sync
    sync_parser = subparsers.add_parser("sync", help="Sync code to remote")
    sync_parser.add_argument("machine", help="Machine key (e.g., pc-3070ti)")
    sync_parser.add_argument("--dry-run", action="store_true", help="Show commands only")
    
    # Run
    run_parser = subparsers.add_parser("run", help="Run command on remote")
    run_parser.add_argument("machine", help="Machine key (e.g., pc-3070ti)")
    run_parser.add_argument("command", help="Command to run (quoted)")
    
    args = parser.parse_args()
    config = load_config()
    
    if args.action == "list":
        print("\n🌍 Configured Machines:")
        for k, v in config.items():
            print(f"  - {k:15} : {v['name']} ({v['gpu']})")
            print(f"    Host: {v.get('ssh_host', 'Not Set')} | Primary: {v.get('primary', False)}")
            
    elif args.action == "sync":
        conf = get_machine_config(config, args.machine)
        sync_code(args.machine, conf, args.dry_run)
        
    elif args.action == "run":
        conf = get_machine_config(config, args.machine)
        run_remote(args.machine, conf, args.command)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
