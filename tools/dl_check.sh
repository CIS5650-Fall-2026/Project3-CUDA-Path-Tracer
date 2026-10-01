docker exec -i \
    -u "$(id -u):$(id -g)" \
    -e CUDA_VISIBLE_DEVICES="${GPU:-0}" \
    "${CONTAINER:-boyuann-project0-profile}" bash <<'BASH'
set -e
set -o pipefail
cd /workspace/Project3-CUDA-Path-Tracer-main

/workspace/diagnostics/project3-tools/bin/cmake --build build -j2

mkdir -p results
test_dir=$(mktemp -d results/dl-check-XXXXXX)

python3 - "$test_dir" <<'PY'
import copy
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
base = json.loads(Path("scenes/cornell.json").read_text())
base["Camera"]["RES"] = [256, 256]
base["Camera"]["DEPTH"] = 8

# Same total power from a light with 1/9 of the area.
small = copy.deepcopy(base)
small["Materials"]["light"]["EMITTANCE"] *= 9.0
for obj in small["Objects"]:
    if obj["MATERIAL"] == "light":
        obj["SCALE"] = [1.0, 0.3, 1.0]

for scene_name, scene in [("big", base), ("small", small)]:
    for name, samples in [("ref", 2048), ("64", 64), ("512", 512)]:
        variant = copy.deepcopy(scene)
        variant["Camera"]["ITERATIONS"] = samples
        variant["Camera"]["FILE"] = str(folder / f"dl-{scene_name}-{name}")
        (folder / f"dl-{scene_name}-{name}.json").write_text(
            json.dumps(variant, indent=4) + "\n"
        )

print("Test folder:", folder)
PY

run() {
    local scene=$1 samples=$2 mode=$3 dl=$4
    PROJECT3_HEADLESS=1 PROJECT3_DIRECT_LIGHTING=$dl \
        ./build/bin/cis565_path_tracer "$test_dir/dl-$scene-$samples.json" \
        > "$test_dir/dl-$scene-$samples-$mode-run.txt" 2>&1
    for f in "$test_dir"/dl-$scene-$samples.*.png; do
        mv "$f" "$test_dir/dl-$scene-$samples-$mode.png"
    done
    grep -h "Average\|Mean" "$test_dir/dl-$scene-$samples-$mode-run.txt" \
        | sed "s/^/$scene $samples $mode: /"
}

for scene in big small; do
    run $scene ref off 0
    for samples in 64 512; do
        run $scene $samples off 0
        run $scene $samples on 1
    done
done
BASH
