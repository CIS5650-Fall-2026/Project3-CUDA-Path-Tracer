#!/usr/bin/env bash
# Sweeps compaction ON/OFF over the open and closed Cornell boxes, with sorting
# and AA off so compaction is the only variable. Writes analysis/data/{paths,timing}.csv
# and restores the files it edits on exit, including on Ctrl-C.

set -euo pipefail
cd "$(dirname "$0")/../.."

SRC=src/pathtrace.cu
EXE=build/bin/Release/cis565_path_tracer.exe
SCENES=(cornell cornell_closed)
ITERS=5000

BAK=$(mktemp -d)
cp "$SRC" "$BAK/"
for s in "${SCENES[@]}"; do cp "scenes/core/$s.json" "$BAK/"; done

restore() {
    { cat "$BAK/pathtrace.cu" > "$SRC"
      for s in "${SCENES[@]}"; do cat "$BAK/$s.json" > "scenes/core/$s.json"; done
      rm -rf "$BAK"
    } || echo "WARNING: restore incomplete, originals kept in $BAK" >&2
}
trap restore EXIT

mkdir -p analysis/logs
echo "scene,compaction,bounce,paths_alive" > analysis/data/paths.csv
echo "scene,compaction,ms_per_iter,fps"    > analysis/data/timing.csv

for s in "${SCENES[@]}"; do
    sed -i -E "s/\"ITERATIONS\": *[0-9]+/\"ITERATIONS\":$ITERS/" "scenes/core/$s.json"
    for c in 1 0; do
        sed -i -E "s/^(#define STREAM_COMPACTION) +[0-9]+/\1 $c/;
                   s/^(#define SORT_BY_MATERIAL) +[0-9]+/\1 0/;
                   s/^(#define ANTIALIASING) +[0-9]+/\1 0/" "$SRC"
        cmake --build build --config Release > /dev/null

        log="analysis/logs/$s-compaction-$c.log"
        echo "=== $s, compaction=$c ==="
        "$EXE" "scenes/core/$s.json" | tee "$log"
        mv -f "$s".*samp.png "build/$s-compaction-$c-${ITERS}samp.png" 2>/dev/null || true
        awk -v s="$s" -v c="$c" '
            /bounce [0-9]+:/ { print s","c","$3+0","$4 >> "analysis/data/paths.csv" }
            /ms\/iteration/  { gsub(/[()]/,""); print s","c","$5","$7 >> "analysis/data/timing.csv" }
        ' "$log"
    done
done

echo "wrote analysis/data/paths.csv and analysis/data/timing.csv"
