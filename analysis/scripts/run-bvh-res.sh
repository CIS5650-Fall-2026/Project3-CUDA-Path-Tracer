#!/usr/bin/env bash
# Times the BVH ON/OFF on Suzanne and the gray kitchen at the same 800x800, and on the
# kitchen at 400x400 as well, in one run, to tell whether the kitchen's smaller speedup in
# run-bvh.sh comes from its resolution rather than from the scene
# Writes analysis/data/bvh-res-timing.csv and restores src/pathtrace.cu on exit
# The binary left in build/ is the last configuration (BVH off), so rebuild before rendering

set -euo pipefail
cd "$(dirname "$0")/../.."

SRC=src/pathtrace.cu
EXE=build/bin/Release/cis565_path_tracer.exe
SUZANNE=scenes/mesh/suzanne.json
KITCHEN=scenes/kitchen/kitchen_v0.json
ITERS=300
OUT=analysis/data/bvh-res-timing.csv

BAK=$(mktemp -d)
cp "$SRC" "$BAK/"
restore() {
    { cat "$BAK/pathtrace.cu" > "$SRC"
      rm -f scenes/mesh/tmp_bvhres_*.json scenes/kitchen/tmp_bvhres_*.json
      rm -rf "$BAK"
    } || echo "WARNING: restore incomplete, original kept in $BAK" >&2
}
trap restore EXIT

mkdir -p analysis/logs
echo "scene,res,triangles,bvh,ms_per_iter,fps" > "$OUT"

# temporary copies, so the real scenes are never edited: iteration count, resolution, output name
SUZ800=scenes/mesh/tmp_bvhres_suzanne.json
KIT800=scenes/kitchen/tmp_bvhres_kitchen800.json
KIT400=scenes/kitchen/tmp_bvhres_kitchen400.json
sed -E "s/\"ITERATIONS\": *[0-9]+/\"ITERATIONS\":$ITERS/; s/\"FILE\": *\"suzanne\"/\"FILE\":\"tmp_bvhres_suzanne\"/" "$SUZANNE" > "$SUZ800"
sed -E "s/\"ITERATIONS\": *[0-9]+/\"ITERATIONS\":$ITERS/; s/\"FILE\": *\"kitchen_v0\"/\"FILE\":\"tmp_bvhres_kitchen400\"/" "$KITCHEN" > "$KIT400"
sed -z -E "s/\"RES\": *\[[^]]*\]/\"RES\": [800, 800]/" "$KIT400" \
    | sed -E "s/tmp_bvhres_kitchen400/tmp_bvhres_kitchen800/" > "$KIT800"

# name, scene file, resolution, image prefix, bvh
run() {
    local name=$1 scene=$2 res=$3 prefix=$4 b=$5
    local log="analysis/logs/bvhres-$name-$b.log"
    echo "=== $name, ${res}x${res}, bvh=$b ==="
    # a freshly linked exe can stay locked for a moment (linker/antivirus); retry briefly
    for attempt in 1 2 3 4 5; do
        if "$EXE" "$scene" | tee "$log"; then break; fi
        [ "$attempt" -eq 5 ] && { echo "exe still locked, giving up" >&2; exit 1; }
        sleep 3
    done
    rm -f "$prefix".*samp.png
    awk -v n="$name" -v r="$res" -v b="$b" -v out="$OUT" '
        /triangles/     { tris += $3 }
        /ms\/iteration/ { gsub(/[()]/,""); print n","r","tris","b","$5","$7 >> out }
    ' "$log"
}

for b in 1 0; do
    sed -i -E "s/^(#define BVH) +[0-9]+/\1 $b/" "$SRC"
    cmake --build build --config Release > /dev/null

    run suzanne_16k    "$SUZ800" 800 tmp_bvhres_suzanne    "$b"
    run kitchen_v0_400 "$KIT400" 400 tmp_bvhres_kitchen400 "$b"
    run kitchen_v0_800 "$KIT800" 800 tmp_bvhres_kitchen800 "$b"
done

echo "wrote $OUT"
cat "$OUT"
