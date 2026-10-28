#!/usr/bin/env bash
# Times mesh scenes with the BVH ON/OFF; OFF falls back to one AABB per mesh
# Writes analysis/data/bvh-timing.csv and restores the files it edits on exit

set -euo pipefail
cd "$(dirname "$0")/../.."

SRC=src/pathtrace.cu
EXE=build/bin/Release/cis565_path_tracer.exe
SUZANNE=scenes/mesh/suzanne.json
KITCHEN=scenes/kitchen/kitchen_v0.json
ITERS=300
OUT=analysis/data/bvh-timing.csv

BAK=$(mktemp -d)
cp "$SRC" "$SUZANNE" "$KITCHEN" "$BAK/"
restore() {
    { cat "$BAK/pathtrace.cu" > "$SRC"
      cat "$BAK/suzanne.json" > "$SUZANNE"
      cat "$BAK/kitchen_v0.json" > "$KITCHEN"
      rm -rf "$BAK"
    } || echo "WARNING: restore incomplete, originals kept in $BAK" >&2
}
trap restore EXIT

mkdir -p analysis/logs
echo "scene,triangles,bvh,ms_per_iter,fps" > "$OUT"

sed -i -E "s/\"ITERATIONS\": *[0-9]+/\"ITERATIONS\":$ITERS/" "$SUZANNE" "$KITCHEN"

# name, scene file, image prefix
run() {
    local name=$1 scene=$2 prefix=$3 b=$4
    local log="analysis/logs/bvh-$name-$b.log"
    echo "=== $name, bvh=$b ==="
    # a freshly linked exe can stay locked for a moment (linker/antivirus); retry briefly
    for attempt in 1 2 3 4 5; do
        if "$EXE" "$scene" | tee "$log"; then break; fi
        [ "$attempt" -eq 5 ] && { echo "exe still locked, giving up" >&2; exit 1; }
        sleep 3
    done
    mv -f "$prefix".*samp.png "build/bvh-$name-$b-${ITERS}samp.png" 2>/dev/null || true
    awk -v n="$name" -v b="$b" -v out="$OUT" '
        /triangles/     { tris += $3 }
        /ms\/iteration/ { gsub(/[()]/,""); print n","tris","b","$5","$7 >> out }
    ' "$log"
}

for b in 1 0; do
    sed -i -E "s/^(#define BVH) +[0-9]+/\1 $b/" "$SRC"
    cmake --build build --config Release > /dev/null

    for m in suzanne_4k suzanne_16k; do
        sed -i -E "s|\"FILE\":\"\.\./models/[^\"]+\.gltf\"|\"FILE\":\"../models/$m.gltf\"|" "$SUZANNE"
        run "$m" "$SUZANNE" suzanne "$b"
    done
    run kitchen_v0 "$KITCHEN" kitchen_v0 "$b"
done

echo "wrote $OUT"
cat "$OUT"
