#!/usr/bin/env bash
# Times the Suzanne scene with mesh AABB culling ON/OFF for each model size
# Writes analysis/data/aabb-timing.csv and restores the files it edits on exit

set -euo pipefail
cd "$(dirname "$0")/../.."

SRC=src/pathtrace.cu
EXE=build/bin/Release/cis565_path_tracer.exe
SCENE=scenes/mesh/suzanne.json
MODELS=(suzanne_4k suzanne_16k)
ITERS=100

BAK=$(mktemp -d)
cp "$SRC" "$SCENE" "$BAK/"
restore() {
    { cat "$BAK/pathtrace.cu" > "$SRC"
      cat "$BAK/suzanne.json" > "$SCENE"
      rm -rf "$BAK"
    } || echo "WARNING: restore incomplete, originals kept in $BAK" >&2
}
trap restore EXIT

mkdir -p analysis/logs
echo "model,triangles,aabb_cull,ms_per_iter,fps" > analysis/data/aabb-timing.csv

sed -i -E "s/\"ITERATIONS\": *[0-9]+/\"ITERATIONS\":$ITERS/" "$SCENE"
# the BVH root test makes the per-mesh box redundant, so it is off for this comparison
sed -i -E "s/^(#define BVH) +[0-9]+/\1 0/" "$SRC"
for m in "${MODELS[@]}"; do
    sed -i -E "s|\"FILE\":\"\.\./models/[^\"]+\.gltf\"|\"FILE\":\"../models/$m.gltf\"|" "$SCENE"
    for c in 1 0; do
        sed -i -E "s/^(#define MESH_AABB_CULL) +[0-9]+/\1 $c/" "$SRC"
        cmake --build build --config Release > /dev/null

        log="analysis/logs/aabb-$m-cull-$c.log"
        echo "=== $m, aabb_cull=$c ==="
        # a freshly linked exe can stay locked for a moment (linker/antivirus); retry briefly
        for attempt in 1 2 3 4 5; do
            if "$EXE" "$SCENE" | tee "$log"; then break; fi
            [ "$attempt" -eq 5 ] && { echo "exe still locked, giving up" >&2; exit 1; }
            sleep 3
        done
        mv -f suzanne.*samp.png "build/aabb-$m-cull-$c-${ITERS}samp.png" 2>/dev/null || true
        awk -v m="$m" -v c="$c" '
            /triangles/     { tris = $3 }
            /ms\/iteration/ { gsub(/[()]/,""); print m","tris","c","$5","$7 >> "analysis/data/aabb-timing.csv" }
        ' "$log"
    done
done

echo "wrote analysis/data/aabb-timing.csv"
cat analysis/data/aabb-timing.csv
