docker exec -i \
    -u "$(id -u):$(id -g)" \
    boyuann-project0-profile bash <<'BASH'
set -e
set -o pipefail
cd /workspace/Project3-CUDA-Path-Tracer-main

/workspace/diagnostics/project3-tools/bin/cmake --build build -j2

mkdir -p results
test_dir=$(mktemp -d results/dof-check-XXXXXX)

python3 - "$test_dir" <<'PY'
import copy
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
scene = json.loads(Path("scenes/cornell.json").read_text())

camera = scene["Camera"]
camera["RES"] = [384, 384]
camera["ITERATIONS"] = 1024
camera["DEPTH"] = 12
camera["EYE"] = [0.0, 4.0, 9.0]
camera["LOOKAT"] = [0.0, 4.0, 0.0]
camera["UP"] = [0.0, 1.0, 0.0]
camera["FOCAL_DISTANCE"] = 9.0

scene["Materials"]["dof_test"] = {
    "TYPE": "Diffuse",
    "RGB": [0.8, 0.8, 0.8],
}

scene["Objects"] = [
    obj for obj in scene["Objects"]
    if obj["TYPE"] != "sphere"
]

for position in [
    [-2.0, 4.0, 3.0],
    [0.0, 4.0, 0.0],
    [2.0, 4.0, -3.0],
]:
    scene["Objects"].append({
        "TYPE": "sphere",
        "MATERIAL": "dof_test",
        "TRANS": position,
        "ROTAT": [0.0, 0.0, 0.0],
        "SCALE": [1.6, 1.6, 1.6],
    })

for name, aperture in [("off", 0.0), ("on", 0.6)]:
    variant = copy.deepcopy(scene)
    variant["Camera"]["APERTURE_RADIUS"] = aperture
    variant["Camera"]["FILE"] = str(folder / f"dof-{name}")

    (folder / f"dof-{name}.json").write_text(
        json.dumps(variant, indent=4) + "\n"
    )

print("Test folder:", folder)
PY

for mode in off on; do
    PROJECT3_HEADLESS=1 PROJECT3_SORT_MATERIALS=1 \
        ./build/bin/cis565_path_tracer "$test_dir/dof-$mode.json" \
        2>&1 | tee "$test_dir/dof-$mode-run.txt"
done

find "$test_dir" -maxdepth 1 -name '*.png' -print
BASH
