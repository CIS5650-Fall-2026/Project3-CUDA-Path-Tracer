docker exec -i \
    -u "$(id -u):$(id -g)" \
    -e CUDA_VISIBLE_DEVICES="${GPU:-0}" \
    "${CONTAINER:-boyuann-project0-profile}" bash <<'BASH'
set -e
set -o pipefail
cd /workspace/Project3-CUDA-Path-Tracer-main

/workspace/diagnostics/project3-tools/bin/cmake --build build -j2

mkdir -p results
test_dir=$(mktemp -d results/perf-XXXXXX)
echo "Test folder: $test_dir"

python3 - "$test_dir" <<'PY'
import copy
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])

def save(name, scene, res, samples, depth=8):
    s = copy.deepcopy(scene)
    s["Camera"]["RES"] = res
    s["Camera"]["ITERATIONS"] = samples
    s["Camera"]["DEPTH"] = depth
    s["Camera"]["FILE"] = str(folder / name)
    (folder / f"{name}.json").write_text(json.dumps(s, indent=4) + "\n")

cornell = json.loads(Path("scenes/cornell.json").read_text())
save("cornell", cornell, [384, 384], 256)

closed = copy.deepcopy(cornell)
closed["Camera"]["EYE"] = [0.0, 5.0, 4.5]
closed["Objects"].append({"TYPE": "cube", "MATERIAL": "diffuse_white",
    "TRANS": [0.0, 5.0, 5.0], "ROTAT": [0.0, 0.0, 0.0], "SCALE": [10.0, 10.0, 0.01]})
save("closed", closed, [384, 384], 256, depth=16)
save("open16", cornell, [384, 384], 256, depth=16)

cover = json.loads(Path("scenes/cover.json").read_text())
save("cover", cover, [400, 256], 32)

fog = copy.deepcopy(cornell)
fog["Fog"] = {"DENSITY": 0.08, "ALBEDO": [0.9, 0.9, 0.9]}
save("fog", fog, [384, 384], 256, depth=12)

moving = copy.deepcopy(cornell)
for o in moving["Objects"]:
    if o["TYPE"] == "sphere":
        o["VELOCITY"] = [3.0, 0.0, 0.0]
save("moving", moving, [384, 384], 256)

for mesh in ["bunny.obj", "bunny-hires.obj"]:
    s = copy.deepcopy(cornell)
    s["Materials"]["bunny_white"] = {"TYPE": "Diffuse", "RGB": [0.85, 0.85, 0.8]}
    s["Objects"] = [o for o in s["Objects"] if o["TYPE"] != "sphere"]
    s["Objects"].append({"TYPE": "mesh", "FILE": "scenes/models/" + mesh,
        "MATERIAL": "bunny_white", "TRANS": [0.6, -1.16, 0.0],
        "ROTAT": [0.0, 0.0, 0.0], "SCALE": [35.0, 35.0, 35.0]})
    save(mesh.replace(".obj", ""), s, [256, 256], 16)
PY

csv="$test_dir/timings.csv"
echo "test,scene,variant,run,ms_per_iter" > "$csv"

# measure TEST SCENE VARIANT ENV...
measure() {
    local test=$1 scene=$2 variant=$3
    shift 3
    for run in 1 2 3; do
        ms=$(env PROJECT3_HEADLESS=1 "$@" ./build/bin/cis565_path_tracer "$test_dir/$scene.json" 2>&1 \
            | sed -n 's/Average time per iteration: \([0-9.]*\) ms/\1/p')
        echo "$test,$scene,$variant,$run,$ms" | tee -a "$csv"
    done
    rm -f "$test_dir"/$scene.*.png
}

measure sort cornell off
measure sort cornell on PROJECT3_SORT_MATERIALS=1
measure sort cover off PROJECT3_DIRECT_LIGHTING=1 PROJECT3_BVH=1
measure sort cover on PROJECT3_DIRECT_LIGHTING=1 PROJECT3_BVH=1 PROJECT3_SORT_MATERIALS=1

measure rr open16 off
measure rr open16 on PROJECT3_RUSSIAN_ROULETTE=1
measure rr closed off
measure rr closed on PROJECT3_RUSSIAN_ROULETTE=1

measure dl cornell off
measure dl cornell on PROJECT3_DIRECT_LIGHTING=1

measure mb moving off
measure mb moving on PROJECT3_MOTION_BLUR=1

measure halton cornell off
measure halton cornell on PROJECT3_HALTON=1

measure fog fog off
measure fog fog on PROJECT3_FOG=1

for mesh in bunny bunny-hires; do
    measure mesh $mesh naive
    measure mesh $mesh culling PROJECT3_MESH_CULLING=1
    measure mesh $mesh bvh PROJECT3_BVH=1
done

for depth in 0 4 8 12 16 20 24 32; do
    measure bvhdepth bunny-hires "$depth" PROJECT3_BVH=1 PROJECT3_BVH_MAX_DEPTH=$depth
done

# Paths alive after each bounce, one iteration, for the stream compaction plots.
for scene in open16 closed; do
    for rr in 0 1; do
        PROJECT3_HEADLESS=1 PROJECT3_PRINT_PATH_COUNTS=1 PROJECT3_RUSSIAN_ROULETTE=$rr \
            ./build/bin/cis565_path_tracer "$test_dir/$scene.json" \
            | grep "Paths after" > "$test_dir/paths-$scene-rr$rr.txt"
    done
done
rm -f "$test_dir"/*.png

# Per-kernel GPU time for the stacked bar chart (sorting off / on, Cornell box).
for sort in 0 1; do
    PROJECT3_HEADLESS=1 PROJECT3_SORT_MATERIALS=$sort \
        nsys profile -o "$test_dir/nsys-sort$sort" --force-overwrite true \
        ./build/bin/cis565_path_tracer "$test_dir/cornell.json" > /dev/null 2>&1
    nsys stats -r cuda_gpu_kern_sum -f csv "$test_dir/nsys-sort$sort.nsys-rep" \
        > "$test_dir/kernels-sort$sort.csv" 2>/dev/null || true
done
rm -f "$test_dir"/*.png

echo "done"
BASH
