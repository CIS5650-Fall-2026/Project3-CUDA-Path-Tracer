#!/usr/bin/env bash
# Byte-for-byte check that a change leaves the renders alone: for refactors
# and performance work, not for changes that are meant to alter the image.
# Renders a fixed set of scenes headless with both intersection paths, at
# 400x400 and at 401x399. 400x400 is a multiple of the 128-thread block; the
# odd size runs the kernels' tail guards and CUB's partial last tile.
#
#   tests/cmp_renders.sh golden  [DIR]   render the set into DIR
#   tests/cmp_renders.sh compare [DIR]   render it again into DIR_compare and
#                                        cmp every PNG against DIR's
#
# DIR defaults to build/golden (gitignored). Render the goldens before the
# change, rebuild, then compare; the exit code is nonzero when any render
# differs or fails. EXE overrides the renderer, for example a scratch build.
# Each render's console output is kept next to its PNG as a .log.

set -u
cd "$(dirname "$0")/.."

mode=${1:-}
golden=${2:-build/golden}
exe=${EXE:-./build/bin/Release/cis565_path_tracer.exe}

# name | scene file | spp on the naive path | spp on OptiX. The naive kernel
# tests every triangle at every bounce, so the large scenes get a few samples.
scenes=(
  "cornell|scenes/cornell.json|64|64"
  "cornell_glass|scenes/cornell_glass.json|64|64"
  "cornell_boxcube|scenes/cornell_boxcube.json|64|64"
  "cornell_boxmesh|scenes/cornell_boxmesh.json|64|64"
  "cornell_suzanne|scenes/cornell_suzanne.json|16|64"
  "sphere|scenes/sphere.json|64|64"
  "cornell_duck|scenes/cornell_duck.json|16|64"
  "cornell_helmet|scenes/cornell_helmet.json|8|64"
  "sponza|scenes/sponza.json|2|16"
  "transmission_roughness|scenes/transmission_roughness.json|2|32"
  "clearcoat|scenes/clearcoat.json|4|32"
  "dragon_attenuation|scenes/dragon_attenuation.json|2|32"
  "research_cornell_box|scenes/assets/gltf-research-scenes/scenes/cornell-box/gltf/cornell_box_extended.gltf|32|32"
)
sizes=(400x400 401x399)

case "$mode" in
  golden)  out=$golden ;;
  compare) out=${golden%/}_compare ;;
  *)
    echo "usage: $0 golden|compare [DIR]" >&2
    exit 2
    ;;
esac
if [ "$mode" = compare ] && [ ! -d "$golden" ]; then
  echo "no goldens in $golden; render them first with: $0 golden $golden" >&2
  exit 2
fi
if [ ! -x "$exe" ]; then
  echo "no renderer at $exe" >&2
  exit 2
fi

mkdir -p "$out"
commit=$(git rev-parse --short HEAD)
if ! git diff --quiet HEAD -- src CMakeLists.txt; then
  commit="$commit + uncommitted changes"
fi
echo "$commit" > "$out/commit.txt"
if [ "$mode" = compare ]; then
  echo "goldens from: $(cat "$golden/commit.txt" 2>/dev/null || echo unknown)"
  echo "this build:   $commit"
fi

total=0
problems=0
for entry in "${scenes[@]}"; do
  IFS='|' read -r name file naiveSpp optixSpp <<< "$entry"
  for size in "${sizes[@]}"; do
    for path in naive optix; do
      if [ "$path" = naive ]; then
        spp=$naiveSpp
        flag=--no-optix
      else
        spp=$optixSpp
        flag=
      fi
      id=${name}_${size}_${path}
      total=$((total + 1))
      # A failed render must not leave the previous run's PNG to compare.
      rm -f "$out/$id.png"
      "$exe" "$file" --headless --spp "$spp" --res "$size" $flag --out "$out/$id.png" > "$out/$id.log" 2>&1
      if [ ! -f "$out/$id.png" ]; then
        echo "$id: render failed, see $out/$id.log"
        problems=$((problems + 1))
      elif [ "$mode" = golden ]; then
        echo "$id: rendered"
      elif [ ! -f "$golden/$id.png" ]; then
        echo "$id: no golden"
        problems=$((problems + 1))
      elif cmp -s "$out/$id.png" "$golden/$id.png"; then
        echo "$id: identical"
      else
        echo "$id: DIFFERS"
        problems=$((problems + 1))
      fi
    done
  done
done

if [ "$mode" = golden ]; then
  echo "$((total - problems)) of $total rendered into $out"
else
  echo "$((total - problems)) of $total identical"
fi
[ "$problems" -eq 0 ]
