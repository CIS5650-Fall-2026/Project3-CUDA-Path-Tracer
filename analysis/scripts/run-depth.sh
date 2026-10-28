#!/usr/bin/env bash
# Times the glass scenes at trace depth 8 and 32
# Writes analysis/data/depth-timing.csv, no rebuild needed

set -euo pipefail
cd "$(dirname "$0")/../.."

EXE=build/bin/Release/cis565_path_tracer.exe
ITERS=300
SCENES=(glass_open glass_closed)
DEPTHS=(8 32)

mkdir -p analysis/logs
echo "scene,depth,ms_per_iter,fps" > analysis/data/depth-timing.csv

for scene in "${SCENES[@]}"; do
    for depth in "${DEPTHS[@]}"; do
        tmp="scenes/refraction/tmp_${scene}_d$depth.json"
        sed -E "s/\"ITERATIONS\": *[0-9]+/\"ITERATIONS\":$ITERS/; s/\"DEPTH\": *[0-9]+/\"DEPTH\":$depth/; s/\"FILE\":\"[^\"]+\"/\"FILE\":\"tmp_${scene}_d$depth\"/" "scenes/refraction/$scene.json" > "$tmp"

        log="analysis/logs/depth-$scene-$depth.log"
        echo "=== $scene, depth $depth ==="
        "$EXE" "$tmp" | tee "$log"
        rm -f "$tmp" tmp_"${scene}"_d"$depth".*samp.png
        awk -v s="$scene" -v d="$depth" '
            /ms\/iteration/ { gsub(/[()]/,""); print s","d","$5","$7 >> "analysis/data/depth-timing.csv" }
        ' "$log"
    done
done

echo "wrote analysis/data/depth-timing.csv"
cat analysis/data/depth-timing.csv
