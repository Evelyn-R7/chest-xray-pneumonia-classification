#!/usr/bin/env bash
set -euo pipefail
source "$HOME/.venvs/chest-xray/bin/activate"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/transfer_run_common.sh
run_transfer_experiment configs/experiments/vgg16_transfer_v1.yaml full
