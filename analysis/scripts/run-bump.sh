#!/usr/bin/env bash
# Times the tiled Cornell box with the procedural color alone and with bump on top
# Writes analysis/data/bump-timing.csv, no rebuild needed

set -euo pipefail
cd "$(dirname "$0")/../.."

EXE=build/bin/Release/cis565_path_tracer.exe
ITERS=300
# scene folder, scene file, label
RUNS=(
    "texturing,tiles_procedural,flat"
    "bump,tiles_bump,bump"
)

mkdir -p analysis/logs
echo "scene,bump,ms_per_iter,fps" > analysis/data/bump-timing.csv

for run in "${RUNS[@]}"; do
    IFS=, read -r dir scene label <<< "$run"
    tmp="scenes/$dir/tmp_$scene.json"
    sed -E "s/\"ITERATIONS\": *[0-9]+/\"ITERATIONS\": $ITERS/; s/\"FILE\": *\"[^\"]+\"/\"FILE\": \"tmp_$scene\"/" "scenes/$dir/$scene.json" > "$tmp"

    log="analysis/logs/bump-$scene.log"
    echo "=== $scene ($label) ==="
    "$EXE" "$tmp" | tee "$log"
    rm -f "$tmp" tmp_"$scene".*samp.png
    awk -v s="$scene" -v b="$label" '
        /ms\/iteration/ { gsub(/[()]/,""); print s","b","$5","$7 >> "analysis/data/bump-timing.csv" }
    ' "$log"
done

echo "wrote analysis/data/bump-timing.csv"
cat analysis/data/bump-timing.csv
