# Toys in the Park — revised scene

## Changes

- Removed Miles and the WALL-E/EVE group.
- Increased Toothless from 75 cm to **115 cm wingspan** and moved it to the back center.
- Packed the six remaining toys into a compact central group.
- Turned the outer toys toward the camera: Iron Man +12 degrees, Kirby +13 degrees from its corrected forward direction, and WALL-E -10 degrees. Spider-Man turns -5 degrees and Spheal +2 degrees.
- Preserved the 78 cm-high wooden table, original model proportions, textures, and park HDRI setup.

## Install

1. Extract the original `SceneAssets.zip` into your project's `scenes/` directory. Its model folders must be directly inside `scenes/`, for example `scenes/Kirby/textures/`.
2. Extract this archive into `scenes/`, replacing the previous `TabletopFinal` files. Keep `tabletop.gltf` and `tabletop.bin` together.
3. In `main.cpp`, use this glTF call:

```cpp
bool loaded = loadGltf("../scenes/TabletopFinal/tabletop.gltf", loadedMeshes, loadedImages, loadedTextures);
```

4. Keep your existing park HDRI call:

```cpp
if (!loadEnvironment("../img/greenwich_park_4k.hdr", environment)) {
    return EXIT_FAILURE;
}
```

5. Set Visual Studio **Debugging > Command Arguments** to:

```text
../scenes/TabletopFinal/tabletop.json
```

Paths follow your existing working-directory convention. The JSON loads camera settings; your `main.cpp` separately loads the glTF and environment. No new mesh-loading or animation code is needed.

## Included and external files

The archive contains the combined glTF, its baked geometry buffer, camera JSON, layout metadata, preview, instructions, and model credits. It reuses the original texture files in `Ironman`, `SymbioteSpiderman`, `Kirby`, `Spheal`, `Walle`, `Toothless`, and `WoodenTable`. It has no references to Miles or WalleAndEve.

The HDRI is not in the uploaded assets and is not bundled. Use your existing `img/greenwich_park_4k.hdr`.

## Scene

| Toy | Size | Position |
|---|---:|---|
| Toothless | 115 cm wingspan | Back center |
| Iron Man | 38 cm tall | Middle left |
| Symbiote Spider-Man | 37 cm tall | Middle right |
| Kirby | 18 cm tall | Front left |
| Spheal | 17 cm tall | Front center |
| WALL-E | 28 cm tall | Front right |

Toothless keeps its source flying pose; its lowest geometry rests on the tabletop. Toy scales are uniform. Skin poses are baked into static geometry. Each named root node has a bottom-center pivot and a translation in metres, allowing individual placement changes.

`layout.json` records measured bounds and source transforms for reference; your renderer does not read it. Material appearance follows your existing renderer. Original PBR material data and UVs are retained; the uploaded `main.cpp` does not transfer emissive or normal-map fields to rendering materials.

## Camera and preview

The supplied camera remains 1600 × 1000, 4096 samples, depth 12, aperture zero. Use 128–256 samples for an initial render. `FOVY: 18` follows your current loader's half-angle convention.

`layout-preview.png` is a software-rasterized placement preview, not an OptiX render. It omits the park HDRI, shadows, reflections, and final lighting.

## Validation

Verified 7 groups (table plus six toys), 623,524 vertices, and 921,248 triangles. Checked buffer/accessor ranges, finite values, triangle indices, material indices, and external texture paths. All toy bounding boxes are separate and fit inside the tabletop. Lower-vertex intersections against the table set placement with 0.15 mm clearance.

The archive is written to a temporary path, closed, CRC-tested, extracted to a fresh directory, and all extracted file hashes are compared to their originals before publication. The CUDA/OptiX application was not executed here.
