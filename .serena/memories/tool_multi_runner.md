Implemented `scripts/multi_runner.py` for multi-machine synchronization and execution.
- Usage: `pixi run python scripts/multi_runner.py [sync|run|list] <machine_id>`
- Dependencies: `rsync`, `openssh`, `pyyaml` added to pixi.toml.
- Configuration: `config/machines.yml` extended with `ssh_host`, `ssh_user`, `remote_root`.
- Verified on `pc-3070ti`.