#!/usr/bin/env bash
# Times the same tiled Cornell box with plain, image and procedural wall materials
# Writes analysis/data/texture-timing.csv, no rebuild needed

set -euo pipefail
cd "$(dirname "$0")/../.."

EXE=build/bin/Release/cis565_path_tracer.exe
ITERS=300
SCENES=(tiles_plain tiles_image tiles_procedural)

mkdir -p analysis/logs
echo "scene,ms_per_iter,fps" > analysis/data/texture-timing.csv

for scene in "${SCENES[@]}"; do
    tmp="scenes/texturing/tmp_$scene.json"
    sed -E "s/\"ITERATIONS\": *[0-9]+/\"ITERATIONS\": $ITERS/; s/\"FILE\": *\"[^\"]+\"/\"FILE\": \"tmp_$scene\"/" "scenes/texturing/$scene.json" > "$tmp"

    log="analysis/logs/texture-$scene.log"
    echo "=== $scene ==="
    "$EXE" "$tmp" | tee "$log"
    rm -f "$tmp" tmp_"$scene".*samp.png
    awk -v s="$scene" '
        /ms\/iteration/ { gsub(/[()]/,""); print s","$5","$7 >> "analysis/data/texture-timing.csv" }
    ' "$log"
done

echo "wrote analysis/data/texture-timing.csv"
cat analysis/data/texture-timing.csv
