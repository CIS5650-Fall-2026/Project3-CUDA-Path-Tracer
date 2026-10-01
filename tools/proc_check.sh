docker exec -i \
    -u "$(id -u):$(id -g)" \
    -e CUDA_VISIBLE_DEVICES="${GPU:-0}" \
    "${CONTAINER:-boyuann-project0-profile}" bash <<'BASH'
set -e
set -o pipefail
cd /workspace/Project3-CUDA-Path-Tracer-main

/workspace/diagnostics/project3-tools/bin/cmake --build build -j2

mkdir -p results
test_dir=$(mktemp -d results/proc-check-XXXXXX)
samples=${SAMPLES:-64}

python3 - "$test_dir" "$samples" <<'PY'
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
scene = json.loads(Path("scenes/cornell.json").read_text())
scene["Camera"]["RES"] = [384, 384]
scene["Camera"]["ITERATIONS"] = int(sys.argv[2])
scene["Camera"]["DEPTH"] = 8
scene["Camera"]["FILE"] = str(folder / "proc")

scene["Materials"]["marble"] = {
    "TYPE": "Diffuse", "RGB": [0.92, 0.9, 0.85],
    "TEXTURE": "marble", "RGB2": [0.25, 0.3, 0.4], "TEXTURE_SCALE": 1.5,
}
scene["Materials"]["checker"] = {
    "TYPE": "Diffuse", "RGB": [0.9, 0.75, 0.2],
    "TEXTURE": "checker", "RGB2": [0.15, 0.15, 0.2], "TEXTURE_SCALE": 2.0,
}
scene["Objects"] = [o for o in scene["Objects"] if o["TYPE"] != "sphere"]
scene["Objects"] += [
    {"TYPE": "mandelbulb", "MATERIAL": "marble", "TRANS": [-2.0, 3.2, -0.5],
     "ROTAT": [-90.0, 0.0, 0.0], "SCALE": [2.2, 2.2, 2.2]},
    {"TYPE": "menger", "MATERIAL": "checker", "TRANS": [2.3, 1.6, -1.0],
     "ROTAT": [0.0, 30.0, 0.0], "SCALE": [1.6, 1.6, 1.6]},
]
(folder / "proc.json").write_text(json.dumps(scene, indent=4) + "\n")
print("Test folder:", folder)
PY

PROJECT3_HEADLESS=1 PROJECT3_DIRECT_LIGHTING=1 \
    ./build/bin/cis565_path_tracer "$test_dir/proc.json" \
    > "$test_dir/proc-run.txt" 2>&1
for f in "$test_dir"/proc.*.png; do mv "$f" "$test_dir/proc.png"; done
grep -h "Average\|Mean" "$test_dir/proc-run.txt"
BASH
