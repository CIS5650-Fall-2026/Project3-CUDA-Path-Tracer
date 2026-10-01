#!/usr/bin/env bash
set -euo pipefail
scene="${1:-results/starter-sphere.json}"
docker exec -e PROJECT3_HEADLESS=1 boyuann-project0-profile bash -c '
set -e
cd /workspace/Project3-CUDA-Path-Tracer-main
exec ./build/bin/cis565_path_tracer "$1"
' bash "$scene"
