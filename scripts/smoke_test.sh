#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/smoke_test.sh /path/to/Ouro-1.4B [output-dir]" >&2
  exit 2
fi
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_path="$1"
result_dir="${2:-$repo_dir/outputs/smoke}"
mkdir -p "$result_dir"
export PYTHONPATH="$repo_dir/build:$repo_dir:${PYTHONPATH:-}"
for backend in engine torch; do
  python "$repo_dir/examples/generate.py" --backend "$backend" \
    --model "$model_path" --max-new-tokens 64 --output "$result_dir/$backend.json"
done
