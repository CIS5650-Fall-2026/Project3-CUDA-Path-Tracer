#!/usr/bin/env bash
# Times the small-light scenes with direct lighting ON/OFF
# Writes analysis/data/direct-timing.csv and restores the file it edits on exit

set -euo pipefail
cd "$(dirname "$0")/../.."

SRC=src/pathtrace.cu
EXE=build/bin/Release/cis565_path_tracer.exe
SCENES=(small_light_depth2 small_light_depth8)
ITERS=300

BAK=$(mktemp -d)
cp "$SRC" "$BAK/"
restore() {
    { cat "$BAK/pathtrace.cu" > "$SRC"
      rm -rf "$BAK"
    } || echo "WARNING: restore incomplete, original kept in $BAK" >&2
}
trap restore EXIT

mkdir -p analysis/logs
echo "scene,direct_lighting,ms_per_iter,fps" > analysis/data/direct-timing.csv

for d in 1 0; do
    sed -i -E "s/^(#define DIRECT_LIGHTING) +[0-9]+/\1 $d/" "$SRC"
    cmake --build build --config Release > /dev/null
    for scene in "${SCENES[@]}"; do
        tmp="scenes/lighting/tmp_$scene.json"
        sed -E "s/\"ITERATIONS\": *[0-9]+/\"ITERATIONS\": $ITERS/; s/\"FILE\": *\"[^\"]+\"/\"FILE\": \"tmp_$scene\"/" "scenes/lighting/$scene.json" > "$tmp"

        log="analysis/logs/direct-$scene-$d.log"
        echo "=== $scene, direct_lighting=$d ==="
        # a freshly linked exe can stay locked for a moment; retry briefly
        for attempt in 1 2 3 4 5; do
            if "$EXE" "$tmp" | tee "$log"; then break; fi
            [ "$attempt" -eq 5 ] && { echo "exe still locked, giving up" >&2; exit 1; }
            sleep 3
        done
        rm -f "$tmp" tmp_"$scene".*samp.png
        awk -v s="$scene" -v d="$d" '
            /ms\/iteration/ { gsub(/[()]/,""); print s","d","$5","$7 >> "analysis/data/direct-timing.csv" }
        ' "$log"
    done
done

echo "wrote analysis/data/direct-timing.csv"
cat analysis/data/direct-timing.csv
