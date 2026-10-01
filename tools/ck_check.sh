docker exec -i \
    -u "$(id -u):$(id -g)" \
    -e CUDA_VISIBLE_DEVICES="${GPU:-0}" \
    "${CONTAINER:-boyuann-project0-profile}" bash <<'BASH'
set -e
set -o pipefail
cd /workspace/Project3-CUDA-Path-Tracer-main

/workspace/diagnostics/project3-tools/bin/cmake --build build -j2

mkdir -p results
test_dir=$(mktemp -d results/ck-check-XXXXXX)

python3 - "$test_dir" <<'PY'
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
scene = json.loads(Path("scenes/cornell.json").read_text())
scene["Camera"]["RES"] = [256, 256]
scene["Camera"]["ITERATIONS"] = 300
scene["Camera"]["DEPTH"] = 8
for name in ["straight", "resumed", "interrupted"]:
    scene["Camera"]["FILE"] = str(folder / name)
    (folder / f"{name}.json").write_text(json.dumps(scene, indent=4) + "\n")
print("Test folder:", folder)
PY

tracer=./build/bin/cis565_path_tracer

echo "=== one run, 300 iterations ==="
PROJECT3_HEADLESS=1 $tracer "$test_dir/straight.json" | grep -v "^ *$"

echo "=== stop after 100, then resume ==="
PROJECT3_HEADLESS=1 PROJECT3_CHECKPOINT="$test_dir/resumed.ckpt" PROJECT3_STOP_AFTER=100 \
    $tracer "$test_dir/resumed.json" | grep -v "^ *$"
PROJECT3_HEADLESS=1 PROJECT3_CHECKPOINT="$test_dir/resumed.ckpt" \
    $tracer "$test_dir/resumed.json" | grep -v "^ *$"

echo "=== Ctrl-C after 3 seconds, then resume ==="
PROJECT3_HEADLESS=1 PROJECT3_CHECKPOINT="$test_dir/interrupted.ckpt" \
    timeout -s INT 3 $tracer "$test_dir/interrupted.json" | grep -v "^ *$" || true
PROJECT3_HEADLESS=1 PROJECT3_CHECKPOINT="$test_dir/interrupted.ckpt" \
    $tracer "$test_dir/interrupted.json" | grep -v "^ *$"

echo "=== image checksums ==="
md5sum "$test_dir"/*.png
BASH
