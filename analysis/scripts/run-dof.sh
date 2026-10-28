#!/usr/bin/env bash
# Times the kitchen scene with the thin lens ON/OFF; OFF sets the lens radius to 0
# Writes analysis/data/dof-timing.csv and restores the files it edits on exit

set -euo pipefail
cd "$(dirname "$0")/../.."

EXE=build/bin/Release/cis565_path_tracer.exe
SCENE=scenes/kitchen/kitchen_v1.json
ITERS=300
OUT=analysis/data/dof-timing.csv

BAK=$(mktemp -d)
cp "$SCENE" "$BAK/"
restore() {
    { cat "$BAK/kitchen_v1.json" > "$SCENE"
      rm -rf "$BAK"
    } || echo "WARNING: restore incomplete, original kept in $BAK" >&2
}
trap restore EXIT

mkdir -p analysis/logs
echo "lens_radius,ms_per_iter,fps" > "$OUT"

sed -i -E "s/\"ITERATIONS\": *[0-9]+/\"ITERATIONS\": $ITERS/" "$SCENE"
for r in 0.09 0.0; do
    sed -i -E "s/\"LENS_RADIUS\": *[0-9.]+/\"LENS_RADIUS\": $r/" "$SCENE"
    log="analysis/logs/dof-$r.log"
    echo "=== lens_radius=$r ==="
    "$EXE" "$SCENE" | tee "$log"
    mv -f kitchen_v1.*samp.png "build/dof-$r-${ITERS}samp.png" 2>/dev/null || true
    awk -v r="$r" -v out="$OUT" '
        /ms\/iteration/ { gsub(/[()]/,""); print r","$5","$7 >> out }
    ' "$log"
done

echo "wrote $OUT"
cat "$OUT"
