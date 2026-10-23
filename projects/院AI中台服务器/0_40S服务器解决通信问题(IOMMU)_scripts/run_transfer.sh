#!/usr/bin/env bash
# Usage: bash scripts/run_transfer.sh [container] [log_file]
set -euo pipefail
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
container=${1:-llamafactory-webui}
log_file=${2:-l40s_transfer_check.txt}
sudo docker exec -i \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -e CUDA_LAUNCH_BLOCKING=0 \
    "$container" python -u - \
    < "$script_dir/transfer_check.py" 2>&1 | tee "$log_file"
