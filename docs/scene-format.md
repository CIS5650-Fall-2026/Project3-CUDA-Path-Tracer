# Scene extensions

The original JSON `Materials`, `Objects`, and `Camera` sections remain supported. Asset paths are relative to the scene JSON, not the shell's working directory.

## Materials

`TYPE` accepts `Diffuse`, `Specular` (ideal mirror), `Glass`, and `Emitting`. `RGB` is a linear color multiplier. Emitting materials use `EMITTANCE`; glass uses `IOR`.

- `TEXTURE`: relative image filename; color samples are decoded from sRGB before filtering.
- `PROCEDURAL`: `"checker"`; generates a repeated UV checker.
- `BUMP`: relative height-map filename; height is linear.
- `BUMP_STRENGTH`: height-derivative scaling factor.

The existing STB decoder determines supported image formats. New showcase assets use original 24-bit BMP files. Textures use repeated bilinear filtering. OBJ material libraries are not imported; each object receives the JSON material named by `MATERIAL`.

## Objects

`TYPE` accepts `cube`, `sphere`, and `obj`. An OBJ object additionally requires `FILE`. `TRANS`, `ROTAT`, and `SCALE` retain the starter's translation, degree rotation, and scaling convention. Spheres have local radius 0.5 and cubes have local extent [-0.5, 0.5]. Singular transforms are rejected.

OBJ supports `v`, `vt`, `vn`, and `f`, including independently indexed positive/negative face attributes. Faces may omit normals or UVs. Convex planar polygons are triangulated; concave, nonplanar, malformed, or degenerate faces are rejected with line diagnostics. Organizational OBJ records do not change the scene-assigned material.

Top-level `BVH_MAX_DEPTH` controls CPU construction depth, bounded to the supported iterative traversal capacity. The leaf target is four primitives; depth-capped leaves may contain more. `--bvh off` selects brute-force traversal over the same primitives.

## Camera and runtime overrides

`APERTURE` is the thin-lens aperture radius; zero uses the pinhole camera. `FOCUS` is distance to the focal plane along the viewing direction. The starter's `FOVY` convention is a vertical **half-angle** in degrees. Existing `RES`, `EYE`, `LOOKAT`, `UP`, `ITERATIONS`, `DEPTH`, and `FILE` remain supported.

Headless `--resolution`, `--depth`, and `--samples` override scene rendering settings. Seed and feature toggles are explicit CLI settings. A checkpoint stores the actual camera/configuration used, independent of its extendable target sample count.

## Original assets

Run `python3 scripts/generate_assets.py` to regenerate the original closed ceramic profiles, torus, triangle light, glaze color map, height map, showcase scenes, and controlled benchmark/lighting scenes. Generated assets are checked in so rendering does not require Python.
