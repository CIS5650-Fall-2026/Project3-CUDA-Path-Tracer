#!/usr/bin/env bash
# Profiles the pipeline with Nsight Systems across a few scenes and toggle settings,
# then sums the GPU kernel times into stages with nsys-stages.py. The app's own console
# output does not come through nsys on Windows, so the wall-clock time per iteration is
# taken from the kernel timeline instead
# Writes analysis/data/nsys-<scene>-<config>_cuda_gpu_kern_sum.csv (per kernel) and stage-timing.csv (per stage)
# Restores the files it edits on exit

set -euo pipefail
cd "$(dirname "$0")/../.."

NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2026.1.3/target-windows-x64/nsys.exe"
SRC=src/pathtrace.cu
EXE=build/bin/Release/cis565_path_tracer.exe
ITERS=300
# usage: bash analysis/scripts/run-nsys.sh [scene-config substring]

# scene label, config label, scene file, toggles to turn off (the rest keep their default)
RUNS=(
    "cornell|all-on|scenes/core/cornell.json|"
    "cornell|no-sort|scenes/core/cornell.json|SORT_BY_MATERIAL=0"
    "cornell|no-compaction|scenes/core/cornell.json|STREAM_COMPACTION=0"
    "cornell|no-sort-no-compaction|scenes/core/cornell.json|SORT_BY_MATERIAL=0 STREAM_COMPACTION=0"
    "cornell_closed|all-on|scenes/core/cornell_closed.json|"
    "cornell_closed|no-sort|scenes/core/cornell_closed.json|SORT_BY_MATERIAL=0"
    "cornell_closed|no-compaction|scenes/core/cornell_closed.json|STREAM_COMPACTION=0"
    "cornell_closed|no-sort-no-compaction|scenes/core/cornell_closed.json|SORT_BY_MATERIAL=0 STREAM_COMPACTION=0"
    "kitchen_v0|all-on|scenes/kitchen/kitchen_v0.json|"
    "kitchen_v0|no-bvh|scenes/kitchen/kitchen_v0.json|BVH=0"
    "kitchen_v1|all-on|scenes/kitchen/kitchen_v1.json|"
)
SCENES=(scenes/core/cornell.json scenes/core/cornell_closed.json scenes/kitchen/kitchen_v0.json scenes/kitchen/kitchen_v1.json)

BAK=$(mktemp -d)
cp "$SRC" "$BAK/"
for s in "${SCENES[@]}"; do cp "$s" "$BAK/$(basename "$s")"; done
restore() {
    { cat "$BAK/pathtrace.cu" > "$SRC"
      for s in "${SCENES[@]}"; do cat "$BAK/$(basename "$s")" > "$s"; done
      rm -rf "$BAK"
    } || echo "WARNING: restore incomplete, originals kept in $BAK" >&2
}
trap restore EXIT

mkdir -p analysis/logs
for s in "${SCENES[@]}"; do
    sed -i -E "s/\"ITERATIONS\": *[0-9]+/\"ITERATIONS\":$ITERS/" "$s"
done

# an optional argument limits the runs to those whose "scene-config" contains it,
# so one new row does not mean profiling everything again
ONLY=${1:-}

for run in "${RUNS[@]}"; do
    IFS='|' read -r scene config file toggles <<< "$run"
    [[ "$scene-$config" == *"$ONLY"* ]] || continue

    # every toggle back to its default, then the ones this run turns off
    cat "$BAK/pathtrace.cu" > "$SRC"
    for t in $toggles; do
        sed -i -E "s/^(#define ${t%=*}) +[0-9]+/\1 ${t#*=}/" "$SRC"
    done
    cmake --build build --config Release > /dev/null

    rep="analysis/logs/nsys-$scene-$config"
    echo "=== $scene, $config ==="
    # cuda trace only; no CPU sampling, so the profiler adds as little as possible
    "$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none \
        --output "$rep" --force-overwrite=true "$EXE" "$file"
    rm -f "$(basename "${file%.json}")".*samp.png

    # one row per kernel: total time, instance count, name; and one row per launch with
    # its start time, from which the iteration period is read
    "$NSYS" stats --report cuda_gpu_kern_sum --report cuda_gpu_trace --format csv \
        --force-export=true --output "analysis/data/nsys-$scene-$config" "$rep.nsys-rep" > /dev/null
done

python analysis/scripts/nsys-stages.py "$ITERS"
