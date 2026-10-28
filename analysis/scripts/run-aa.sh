#!/usr/bin/env bash
# Two runs on the open Cornell box to settle what anti-aliasing costs.
#   aa-only : compaction on, sort off, AA on  -> compares against the sweep's 20.40 ms
#   all-on  : compaction on, sort on,  AA on  -> the number the hero image caption needs
# Writes analysis/data/aa-timing.csv and restores src/pathtrace.cu on exit.
set -euo pipefail
cd "$(dirname "$0")/../.."

SRC=src/pathtrace.cu
EXE=build/bin/Release/cis565_path_tracer.exe

BAK=$(mktemp -d)
cp "$SRC" "$BAK/"
restore() {
    { cat "$BAK/pathtrace.cu" > "$SRC"; rm -rf "$BAK"; } \
        || echo "WARNING: restore incomplete, original kept in $BAK" >&2
}
trap restore EXIT

mkdir -p analysis/logs
echo "config,sort,aa,ms_per_iter,fps" > analysis/data/aa-timing.csv

run() {  # name sort aa
    sed -i -E "s/^(#define STREAM_COMPACTION) +[0-9]+/\1 1/;
               s/^(#define SORT_BY_MATERIAL) +[0-9]+/\1 $2/;
               s/^(#define ANTIALIASING) +[0-9]+/\1 $3/" "$SRC"
    cmake --build build --config Release > /dev/null

    log="analysis/logs/aa-$1.log"
    echo "=== $1 (sort=$2, aa=$3) ==="
    "$EXE" scenes/core/cornell.json | tee "$log"
    mv -f cornell.*samp.png "build/aa-$1-5000samp.png" 2>/dev/null || true
    awk -v n="$1" -v s="$2" -v a="$3" '/ms\/iteration/ {
        gsub(/[()]/,""); print n","s","a","$5","$7 >> "analysis/data/aa-timing.csv" }' "$log"
}

run aa-only 0 1
run all-on  1 1

echo "wrote analysis/data/aa-timing.csv"
