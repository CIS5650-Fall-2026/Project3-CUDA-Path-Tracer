docker exec -i \
    -u "$(id -u):$(id -g)" \
    -e CUDA_VISIBLE_DEVICES="${GPU:-0}" \
    "${CONTAINER:-boyuann-project0-profile}" bash <<'BASH'
set -e
set -o pipefail
cd /workspace/Project3-CUDA-Path-Tracer-main

/workspace/diagnostics/project3-tools/bin/cmake --build build -j2

mkdir -p results
test_dir=$(mktemp -d results/rr-check-XXXXXX)

python3 - "$test_dir" <<'PY'
import copy
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
base = json.loads(Path("scenes/cornell.json").read_text())

camera = base["Camera"]
camera["RES"] = [384, 384]
camera["ITERATIONS"] = 512
camera["DEPTH"] = 16

# Closed scene: move the camera inside and seal the open front of the box.
closed = copy.deepcopy(base)
closed["Camera"]["EYE"] = [0.0, 5.0, 4.5]
closed["Objects"].append({
    "TYPE": "cube",
    "MATERIAL": "diffuse_white",
    "TRANS": [0.0, 5.0, 5.0],
    "ROTAT": [0.0, 0.0, 0.0],
    "SCALE": [10.0, 10.0, 0.01],
})

for scene_name, scene in [("open", base), ("closed", closed)]:
    for mode in ["off", "on"]:
        variant = copy.deepcopy(scene)
        variant["Camera"]["FILE"] = str(folder / f"rr-{scene_name}-{mode}")
        (folder / f"rr-{scene_name}-{mode}.json").write_text(
            json.dumps(variant, indent=4) + "\n"
        )

print("Test folder:", folder)
PY

for scene in open closed; do
    for mode in off on; do
        rr=0
        if [ "$mode" = on ]; then rr=1; fi

        echo "=== $scene scene, Russian roulette $mode ==="
        PROJECT3_HEADLESS=1 \
        PROJECT3_RUSSIAN_ROULETTE=$rr \
        PROJECT3_PRINT_PATH_COUNTS=1 \
            ./build/bin/cis565_path_tracer "$test_dir/rr-$scene-$mode.json" \
            2>&1 | tee "$test_dir/rr-$scene-$mode-run.txt"
    done
done

echo
echo "=== Summary ==="
python3 - "$test_dir" <<'PY'
import re
import sys
from pathlib import Path

folder = Path(sys.argv[1])

for scene in ["open", "closed"]:
    stats = {}
    for mode in ["off", "on"]:
        text = (folder / f"rr-{scene}-{mode}-run.txt").read_text()
        ms = float(re.search(r"Average time per iteration: ([\d.]+)", text).group(1))
        rgb = [float(v) for v in
               re.search(r"Mean radiance: ([\d.]+) ([\d.]+) ([\d.]+)", text).groups()]
        stats[mode] = (ms, sum(rgb) / 3.0)

    off_ms, off_mean = stats["off"]
    on_ms, on_mean = stats["on"]
    print(f"{scene:6s}: time {off_ms:.3f} -> {on_ms:.3f} ms/iter "
          f"({100.0 * (off_ms - on_ms) / off_ms:+.1f}% faster), "
          f"mean radiance {off_mean:.5f} vs {on_mean:.5f} "
          f"({100.0 * (on_mean - off_mean) / off_mean:+.2f}%)")
PY
BASH
