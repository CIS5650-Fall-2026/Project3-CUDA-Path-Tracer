docker exec -i \
    -u "$(id -u):$(id -g)" \
    -e CUDA_VISIBLE_DEVICES="${GPU:-0}" \
    "${CONTAINER:-boyuann-project0-profile}" bash <<'BASH'
set -e
set -o pipefail
cd /workspace/Project3-CUDA-Path-Tracer-main

/workspace/diagnostics/project3-tools/bin/cmake --build build -j2

mkdir -p results
test_dir=$(mktemp -d results/fog-check-XXXXXX)

python3 - "$test_dir" <<'PY'
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
scene = json.loads(Path("scenes/cornell.json").read_text())
scene["Camera"]["RES"] = [256, 256]
scene["Camera"]["ITERATIONS"] = 1024
scene["Camera"]["DEPTH"] = 12
scene["Camera"]["FILE"] = str(folder / "fog")
scene["Fog"] = {"DENSITY": 0.08, "ALBEDO": [0.9, 0.9, 0.9]}
(folder / "fog.json").write_text(json.dumps(scene, indent=4) + "\n")
print("Test folder:", folder)
PY

run() {
    local mode=$1 fog=$2 dl=$3
    PROJECT3_HEADLESS=1 PROJECT3_FOG=$fog PROJECT3_DIRECT_LIGHTING=$dl \
        ./build/bin/cis565_path_tracer "$test_dir/fog.json" \
        > "$test_dir/fog-$mode-run.txt" 2>&1
    for f in "$test_dir"/fog.*.png; do mv "$f" "$test_dir/fog-$mode.png"; done
    grep -h "Average\|Mean" "$test_dir/fog-$mode-run.txt" | sed "s/^/$mode: /"
}

run off 0 1
run on-nodl 1 0
run on-dl 1 1
BASH
