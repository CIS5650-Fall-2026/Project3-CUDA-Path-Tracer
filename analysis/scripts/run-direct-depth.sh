#!/usr/bin/env bash
# Renders the small-light Cornell box at several trace depths with direct lighting ON and OFF,
# 100 spp each, and measures the noise of each pair with measure-noise.py. If the noise direct
# lighting leaves at depth 8 comes from paths that hit the light by chance on earlier bounces,
# it should grow with the depth
# Writes build/direct-depth-<depth>-<0|1>-100samp.png and analysis/data/direct-depth-noise.txt
# Restores src/pathtrace.cu on exit. The binary left in build/ has direct lighting OFF, which
# is also the default, but rebuild before rendering anything else

set -euo pipefail
cd "$(dirname "$0")/../.."

SRC=src/pathtrace.cu
EXE=build/bin/Release/cis565_path_tracer.exe
BASE=scenes/lighting/small_light_depth8.json
DEPTHS=(2 3 4 5 6 8)
OUT=analysis/data/direct-depth-noise.txt

BAK=$(mktemp -d)
cp "$SRC" "$BAK/"
restore() {
    { cat "$BAK/pathtrace.cu" > "$SRC"
      rm -f scenes/lighting/tmp_directdepth_*.json
      rm -rf "$BAK"
    } || echo "WARNING: restore incomplete, original kept in $BAK" >&2
}
trap restore EXIT

mkdir -p analysis/logs

for d in 1 0; do
    sed -i -E "s/^(#define DIRECT_LIGHTING) +[0-9]+/\1 $d/" "$SRC"
    cmake --build build --config Release > /dev/null
    for depth in "${DEPTHS[@]}"; do
        name="tmp_directdepth_$depth"
        tmp="scenes/lighting/$name.json"
        sed -E "s/\"DEPTH\": *[0-9]+/\"DEPTH\": $depth/; s/\"ITERATIONS\": *[0-9]+/\"ITERATIONS\": 100/; s/\"FILE\": *\"[^\"]+\"/\"FILE\": \"$name\"/" "$BASE" > "$tmp"

        log="analysis/logs/direct-depth-$depth-$d.log"
        echo "=== depth $depth, direct_lighting=$d ==="
        # a freshly linked exe can stay locked for a moment; retry briefly
        for attempt in 1 2 3 4 5; do
            if "$EXE" "$tmp" | tee "$log"; then break; fi
            [ "$attempt" -eq 5 ] && { echo "exe still locked, giving up" >&2; exit 1; }
            sleep 3
        done
        mv -f "$name".*samp.png "build/direct-depth-$depth-$d-100samp.png"
        rm -f "$tmp"
    done
done

: > "$OUT"
for depth in "${DEPTHS[@]}"; do
    echo "depth $depth, off then on" | tee -a "$OUT"
    python analysis/scripts/measure-noise.py \
        "build/direct-depth-$depth-0-100samp.png" "build/direct-depth-$depth-1-100samp.png" | tee -a "$OUT"
done
echo "wrote $OUT"
