#!/usr/bin/env bash
# Times sorting, stream compaction and anti-aliasing together, so that every comparison
# the README makes about them comes from one build in one session
#   sort x compaction, AA on : open and closed Cornell box, four builds
#   AA off                   : open box, sort off, compaction on, one more build
# Writes analysis/data/toggles-timing.csv (one row per block of 100 iterations) and
# analysis/data/toggles-paths.csv (paths alive per bounce, iteration 1)
# Restores the files it edits on exit

set -euo pipefail
cd "$(dirname "$0")/../.."

SRC=src/pathtrace.cu
EXE=build/bin/Release/cis565_path_tracer.exe
OPEN=scenes/core/cornell.json
CLOSED=scenes/core/cornell_closed.json
ITERS=1000

BAK=$(mktemp -d)
cp "$SRC" "$OPEN" "$CLOSED" "$BAK/"
restore() {
    { cat "$BAK/pathtrace.cu" > "$SRC"
      cat "$BAK/cornell.json" > "$OPEN"
      cat "$BAK/cornell_closed.json" > "$CLOSED"
      rm -rf "$BAK"
    } || echo "WARNING: restore incomplete, originals kept in $BAK" >&2
}
trap restore EXIT

mkdir -p analysis/logs
echo "scene,sort,compaction,aa,iter,ms_per_iter" > analysis/data/toggles-timing.csv
echo "scene,sort,compaction,aa,bounce,paths_alive" > analysis/data/toggles-paths.csv
sed -i -E "s/\"ITERATIONS\": *[0-9]+/\"ITERATIONS\":$ITERS/" "$OPEN" "$CLOSED"

build() {  # sort compaction aa
    sed -i -E "s/^(#define SORT_BY_MATERIAL) +[0-9]+/\1 $1/;
               s/^(#define STREAM_COMPACTION) +[0-9]+/\1 $2/;
               s/^(#define ANTIALIASING) +[0-9]+/\1 $3/" "$SRC"
    cmake --build build --config Release > /dev/null
}

run() {  # name scene sort compaction aa
    local log="analysis/logs/toggles-$1-s$3-c$4-a$5.log"
    echo "=== $1, sort=$3 compaction=$4 aa=$5 ==="
    # a freshly linked exe can stay locked for a moment; retry briefly
    for attempt in 1 2 3 4 5; do
        if "$EXE" "$2" | tee "$log"; then break; fi
        [ "$attempt" -eq 5 ] && { echo "exe still locked, giving up" >&2; exit 1; }
        sleep 3
    done
    rm -f "$(basename "${2%.json}")".*samp.png
    awk -v n="$1" -v s="$3" -v c="$4" -v a="$5" '
        /ms\/iteration/   { gsub(/[()]/,""); print n","s","c","a","$3+0","$5 >> "analysis/data/toggles-timing.csv" }
        /bounce [0-9]+:/  { print n","s","c","a","$3+0","$4 >> "analysis/data/toggles-paths.csv" }
    ' "$log"
}

for s in 1 0; do
    for c in 1 0; do
        build $s $c 1
        run open   "$OPEN"   $s $c 1
        run closed "$CLOSED" $s $c 1
    done
done

build 0 1 0
run open "$OPEN" 0 1 0

echo "wrote analysis/data/toggles-timing.csv and analysis/data/toggles-paths.csv"
