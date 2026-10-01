docker exec -i \
    -u "$(id -u):$(id -g)" \
    -e CUDA_VISIBLE_DEVICES="${GPU:-0}" \
    "${CONTAINER:-boyuann-project0-profile}" bash <<'BASH'
set -e
set -o pipefail
cd /workspace/Project3-CUDA-Path-Tracer-main

/workspace/diagnostics/project3-tools/bin/cmake --build build -j2

mkdir -p results
test_dir=$(mktemp -d results/obj-check-XXXXXX)
samples=${SAMPLES:-32}
mesh=${MESH:-bunny.obj}

python3 - "$test_dir" "$samples" "$mesh" <<'PY'
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
scene = json.loads(Path("scenes/cornell.json").read_text())
scene["Camera"]["RES"] = [256, 256]
scene["Camera"]["ITERATIONS"] = int(sys.argv[2])
scene["Camera"]["DEPTH"] = 8
scene["Camera"]["FILE"] = str(folder / "obj")

scene["Materials"]["bunny_white"] = {"TYPE": "Diffuse", "RGB": [0.85, 0.85, 0.8]}
scene["Objects"] = [o for o in scene["Objects"] if o["TYPE"] != "sphere"]
scene["Objects"].append({
    "TYPE": "mesh",
    "FILE": "scenes/models/" + sys.argv[3],
    "MATERIAL": "bunny_white",
    "TRANS": [0.6, -1.16, 0.0],
    "ROTAT": [0.0, 0.0, 0.0],
    "SCALE": [35.0, 35.0, 35.0],
})
(folder / "obj.json").write_text(json.dumps(scene, indent=4) + "\n")
print("Test folder:", folder)
PY

for mode in naive culling bvh; do
    culling=0
    bvh=0
    if [ "$mode" = culling ]; then culling=1; fi
    if [ "$mode" = bvh ]; then bvh=1; fi
    PROJECT3_HEADLESS=1 PROJECT3_DIRECT_LIGHTING=1 \
    PROJECT3_MESH_CULLING=$culling PROJECT3_BVH=$bvh \
        ./build/bin/cis565_path_tracer "$test_dir/obj.json" \
        > "$test_dir/obj-$mode-run.txt" 2>&1
    for f in "$test_dir"/obj.*.png; do mv "$f" "$test_dir/obj-$mode.png"; done
    grep -h "Loaded\|Average\|Mean" "$test_dir/obj-$mode-run.txt" | sed "s/^/$mode: /"
done
BASH
