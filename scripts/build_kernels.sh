#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python flashloop_engine/csrc/setup.py build_ext --build-lib "$repo_dir/build"
echo 'Kernel built. Run: export PYTHONPATH="$PWD/build:${PYTHONPATH:-}"'
