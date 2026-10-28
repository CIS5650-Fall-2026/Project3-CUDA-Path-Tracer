#!/usr/bin/env bash
# Times the Cornell mirror sphere as a perfect mirror and at three Phong exponents
# Writes analysis/data/specular-timing.csv; the scenes are not edited

set -euo pipefail
cd "$(dirname "$0")/../.."

EXE=build/bin/Release/cis565_path_tracer.exe
ITERS=300
OUT=analysis/data/specular-timing.csv

# name, scene file; ITERATIONS is overridden through a temporary copy
SCENES=(
    "mirror scenes/core/cornell.json"
    "5000 scenes/specular/specular_5000.json"
    "500 scenes/specular/specular_500.json"
    "50 scenes/specular/specular_50.json"
)

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

mkdir -p analysis/logs
echo "exponent,ms_per_iter,fps" > "$OUT"

for entry in "${SCENES[@]}"; do
    name=${entry%% *}
    scene=${entry#* }
    # the copy sits in the scene's own folder so relative asset paths still resolve
    tmpScene="$(dirname "$scene")/_timing_$name.json"
    sed -E "s/\"ITERATIONS\": *[0-9]+/\"ITERATIONS\": $ITERS/" "$scene" > "$tmpScene"
    log="analysis/logs/specular-$name.log"
    echo "=== exponent=$name ==="
    "$EXE" "$tmpScene" | tee "$log"
    rm -f "$tmpScene"
    rm -f cornell.*samp.png specular_*.*samp.png
    awk -v n="$name" -v out="$OUT" '
        /ms\/iteration/ { gsub(/[()]/,""); print n","$5","$7 >> out }
    ' "$log"
done

echo "wrote $OUT"
cat "$OUT"
