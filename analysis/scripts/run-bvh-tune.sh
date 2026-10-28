#!/usr/bin/env bash
# Sweeps the two settings of the BVH build, the leaf size and the depth limit, on Suzanne
# (16k triangles, 800x800) and the gray kitchen (400x400), with the BVH on in every run
# Both are the optional "BVH" keys LEAF_SIZE and MAX_DEPTH of the scene file, so each
# configuration is a temporary copy of the scene and nothing is rebuilt
#   leaf size   : 1, 2, 4, 8, 16, 32 with the depth limit at 24, which no scene reaches
#   depth limit : 6, 8, 10, 12 with the leaf size at 4
# Writes analysis/data/bvh-tune-timing.csv, with the node count the build printed

set -euo pipefail
cd "$(dirname "$0")/../.."

EXE=build/bin/Release/cis565_path_tracer.exe
SUZANNE=scenes/mesh/suzanne.json
KITCHEN=scenes/kitchen/kitchen_v0.json
ITERS=300
OUT=analysis/data/bvh-tune-timing.csv

# leaf size, depth limit
CONFIGS=(
    "4|24"
    "1|24"
    "2|24"
    "8|24"
    "16|24"
    "32|24"
    "4|6"
    "4|8"
    "4|10"
    "4|12"
)

cleanup() { rm -f scenes/mesh/tmp_bvhtune_*.json scenes/kitchen/tmp_bvhtune_*.json; }
trap cleanup EXIT

mkdir -p analysis/logs
echo "scene,leaf_size,depth_limit,bvh_nodes,ms_per_iter,fps" > "$OUT"

# a temporary copy of a scene with the BVH settings added; the real scene is not edited
make_scene() {  # source scene, output scene, file name inside the scene, leaf size, depth limit
    python - "$1" "$2" "$3" "$ITERS" "$4" "$5" <<'PY'
import json
import sys

src, out, name, iters, leaf, depth = sys.argv[1:]
d = json.load(open(src))
d['Camera']['ITERATIONS'] = int(iters)
d['Camera']['FILE'] = name
d['BVH'] = {'LEAF_SIZE': int(leaf), 'MAX_DEPTH': int(depth)}
json.dump(d, open(out, 'w'), indent=4)
PY
}

# name, scene file, leaf size, depth limit
run() {
    local name=$1 scene=$2 leaf=$3 depth=$4
    local log="analysis/logs/bvhtune-$name-l$leaf-d$depth.log"
    echo "=== $name, leaf size $leaf, depth limit $depth ==="
    # a freshly linked exe can stay locked for a moment (linker/antivirus); retry briefly
    for attempt in 1 2 3 4 5; do
        if "$EXE" "$scene" | tee "$log"; then break; fi
        [ "$attempt" -eq 5 ] && { echo "exe still locked, giving up" >&2; exit 1; }
        sleep 3
    done
    rm -f tmp_bvhtune_*.samp.png tmp_bvhtune_*samp.png
    awk -v n="$name" -v l="$leaf" -v d="$depth" -v out="$OUT" '
        /BVH nodes/     { nodes += $1 }
        /ms\/iteration/ { gsub(/[()]/,""); print n","l","d","nodes","$5","$7 >> out }
    ' "$log"
}

for c in "${CONFIGS[@]}"; do
    IFS='|' read -r leaf depth <<< "$c"
    make_scene "$SUZANNE" scenes/mesh/tmp_bvhtune_suzanne.json tmp_bvhtune_suzanne "$leaf" "$depth"
    make_scene "$KITCHEN" scenes/kitchen/tmp_bvhtune_kitchen.json tmp_bvhtune_kitchen "$leaf" "$depth"
    run suzanne_16k scenes/mesh/tmp_bvhtune_suzanne.json "$leaf" "$depth"
    run kitchen_v0 scenes/kitchen/tmp_bvhtune_kitchen.json "$leaf" "$depth"
done

echo "wrote $OUT"
cat "$OUT"
