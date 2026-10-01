docker exec -i \
    -u "$(id -u):$(id -g)" \
    -e CUDA_VISIBLE_DEVICES="${GPU:-0}" \
    "${CONTAINER:-boyuann-project0-profile}" bash <<'BASH'
set -e
set -o pipefail
cd /workspace/Project3-CUDA-Path-Tracer-main

/workspace/diagnostics/project3-tools/bin/cmake --build build -j2

mkdir -p results
test_dir=$(mktemp -d results/qmc-check-XXXXXX)

python3 - "$test_dir" <<'PY'
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
scene = json.loads(Path("scenes/cornell.json").read_text())
scene["Camera"]["RES"] = [256, 256]
scene["Camera"]["DEPTH"] = 8
for samples in [4096, 16, 64, 256]:
    scene["Camera"]["ITERATIONS"] = samples
    scene["Camera"]["FILE"] = str(folder / f"qmc-{samples}")
    (folder / f"qmc-{samples}.json").write_text(json.dumps(scene, indent=4) + "\n")
print("Test folder:", folder)
PY

run() {
    local samples=$1 mode=$2 halton=$3
    PROJECT3_HEADLESS=1 PROJECT3_HALTON=$halton \
        ./build/bin/cis565_path_tracer "$test_dir/qmc-$samples.json" \
        > "$test_dir/qmc-$samples-$mode-run.txt" 2>&1
    for f in "$test_dir"/qmc-$samples.*.png; do
        mv "$f" "$test_dir/qmc-$samples-$mode.png"
    done
    grep -h "Average\|Mean" "$test_dir/qmc-$samples-$mode-run.txt" \
        | sed "s/^/$samples $mode: /"
}

run 4096 ref 0
for samples in 16 64 256; do
    run $samples random 0
    run $samples halton 1
done
BASH
