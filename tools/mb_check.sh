docker exec -i \
    -u "$(id -u):$(id -g)" \
    -e CUDA_VISIBLE_DEVICES="${GPU:-0}" \
    "${CONTAINER:-boyuann-project0-profile}" bash <<'BASH'
set -e
set -o pipefail
cd /workspace/Project3-CUDA-Path-Tracer-main

/workspace/diagnostics/project3-tools/bin/cmake --build build -j2

mkdir -p results
test_dir=$(mktemp -d results/mb-check-XXXXXX)

python3 - "$test_dir" <<'PY'
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
scene = json.loads(Path("scenes/cornell.json").read_text())
scene["Camera"]["RES"] = [256, 256]
scene["Camera"]["ITERATIONS"] = 256
scene["Camera"]["DEPTH"] = 8
scene["Camera"]["FILE"] = str(folder / "mb")

scene["Materials"]["mb_red"] = {"TYPE": "Diffuse", "RGB": [0.9, 0.3, 0.2]}
scene["Materials"]["mb_blue"] = {"TYPE": "Diffuse", "RGB": [0.2, 0.4, 0.9]}

scene["Objects"] = [o for o in scene["Objects"] if o["TYPE"] != "sphere"]
scene["Objects"] += [
    {"TYPE": "sphere", "MATERIAL": "mb_red", "TRANS": [-3.0, 3.0, 0.0],
     "ROTAT": [0.0, 0.0, 0.0], "SCALE": [2.0, 2.0, 2.0],
     "VELOCITY": [3.0, 0.0, 0.0]},
    {"TYPE": "cube", "MATERIAL": "mb_blue", "TRANS": [2.5, 1.5, -1.0],
     "ROTAT": [0.0, 30.0, 0.0], "SCALE": [2.0, 3.0, 2.0],
     "VELOCITY": [0.0, 2.0, 0.0]},
    {"TYPE": "sphere", "MATERIAL": "diffuse_white", "TRANS": [0.5, 6.5, -2.0],
     "ROTAT": [0.0, 0.0, 0.0], "SCALE": [1.5, 1.5, 1.5]},
]
(folder / "mb.json").write_text(json.dumps(scene, indent=4) + "\n")
print("Test folder:", folder)
PY

for mode in off on; do
    mb=0
    if [ "$mode" = on ]; then mb=1; fi
    PROJECT3_HEADLESS=1 PROJECT3_MOTION_BLUR=$mb \
        ./build/bin/cis565_path_tracer "$test_dir/mb.json" \
        > "$test_dir/mb-$mode-run.txt" 2>&1
    for f in "$test_dir"/mb.*.png; do mv "$f" "$test_dir/mb-$mode.png"; done
    grep -h "Average\|Mean" "$test_dir/mb-$mode-run.txt" | sed "s/^/$mode: /"
done
BASH
