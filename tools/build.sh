#!/usr/bin/env bash
set -euo pipefail
docker exec boyuann-project0-profile bash -c '
set -e
cd /workspace/Project3-CUDA-Path-Tracer-main
/workspace/diagnostics/project3-tools/bin/cmake --build build -j2
'
