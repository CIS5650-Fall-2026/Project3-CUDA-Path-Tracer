# Scene extensions

The original JSON `Materials`, `Objects`, and `Camera` sections remain supported. Asset paths are relative to the scene JSON, not the shell's working directory.

## Materials

`TYPE` accepts `Diffuse`, `Specular` (ideal mirror), `Glass`, and `Emitting`. `RGB` is a linear color multiplier. Emitting materials use `EMITTANCE`; glass uses `IOR`.

- `TEXTURE`: relative image filename; color samples are decoded from sRGB before filtering.
- `PROCEDURAL`: one of `"none"`, `"checker"`, `"marble"`, or `"cells"`. Each
  name accepts exactly its lowercase, title-case, and uppercase spelling (for
  example, `"marble"`, `"Marble"`, and `"MARBLE"`). The corresponding internal
  IDs are 0, 1, 2, and 3. Omitting the property is equivalent to `"none"`.
- `CHECKER_SCALE`: positive checker frequency, defaulting to 8. This property is
  valid only when `PROCEDURAL` is the checker (ID 1); it is rejected for none,
  marble, and cells.
- `BUMP`: relative height-map filename; height is linear.
- `BUMP_STRENGTH`: height-derivative scaling factor.

The existing STB decoder determines supported image formats. The image-based
workbench uses original 24-bit BMP files; the procedural showcase materials do
not use image textures or bump maps. Textures use repeated bilinear filtering.
OBJ material libraries are not imported; each object receives the JSON material
named by `MATERIAL`.

Procedural colors are linear RGB multipliers applied to the material's `RGB`;
there is no sRGB decode. Define `fract(x) = x - floor(x)`, and let `mix(a,b,s)`
and `smoothstep(a,b,x)` have their usual component-wise linear-interpolation and
clamped cubic definitions. The checker uses
`floor(CHECKER_SCALE * u) + floor(CHECKER_SCALE * v)`: even squares multiply by
1 and odd squares by 0.18. The two new patterns wrap both UV coordinates before
evaluation:

- Marble sets `U = fract(u)`, `V = fract(v)`, then
  `q = 2 pi (8 U + 0.35 sin(2 pi V) + 0.20 sin(2 pi (3 U - 2 V)))`,
  `w = 0.5 + 0.5 sin(q)`, and `k = (1 - w)^4`. Its result is
  `mix((0.82, 0.72, 0.50), (0.045, 0.12, 0.21), k)`. Thus UV `(0,0)` is
  exactly `(0.7715625, 0.6825, 0.481875)`.
- Cells sets `row = floor(8 V)`, `x = 8 U + 0.5 (row mod 2)`,
  `fx = fract(x)`, and `fy = fract(8 V)`. The alternating color is
  `mix((0.06, 0.34, 0.31), (0.58, 0.19, 0.08),
  (floor(x) + row) mod 2)`. With
  `d = 1 - smoothstep(0.26, 0.30, abs(fx - 0.5) + abs(fy - 0.5))`, the diamond
  color is `mix(alternating, (0.86, 0.57, 0.14), d)`. Finally,
  `edge = smoothstep(0.025, 0.05, min(fx, 1-fx, fy, 1-fy))`, and the result is
  `mix((0.025, 0.035, 0.05), diamond, edge)`. UV `(0,0)` is grout and UV
  `(0.0625,0.0625)` is gold.

## Objects

`TYPE` accepts `cube`, `sphere`, and `obj`. An OBJ object additionally requires `FILE`. `TRANS`, `ROTAT`, and `SCALE` retain the starter's translation, degree rotation, and scaling convention. Spheres have local radius 0.5 and cubes have local extent [-0.5, 0.5]. Singular transforms are rejected.

OBJ supports `v`, `vt`, `vn`, and `f`, including independently indexed positive/negative face attributes. Faces may omit normals or UVs. Convex planar polygons are triangulated; concave, nonplanar, malformed, or degenerate faces are rejected with line diagnostics. Organizational OBJ records do not change the scene-assigned material.

Top-level `BVH_MAX_DEPTH` controls CPU construction depth, bounded to the supported iterative traversal capacity. The leaf target is four primitives; depth-capped leaves may contain more. `--bvh off` selects brute-force traversal over the same primitives.

## Camera and runtime overrides

`APERTURE` is the thin-lens aperture radius; zero uses the pinhole camera. `FOCUS` is distance to the focal plane along the viewing direction. The starter's `FOVY` convention is a vertical **half-angle** in degrees. Existing `RES`, `EYE`, `LOOKAT`, `UP`, `ITERATIONS`, `DEPTH`, and `FILE` remain supported.

Headless `--resolution`, `--depth`, and `--samples` override scene rendering settings. Seed and feature toggles are explicit CLI settings. A checkpoint stores the actual camera/configuration used, independent of its extendable target sample count.

## Original procedural geometry

`scripts/generate_assets.py` also generates two original analytic surfaces. In
the formulas below, `u,v` range over `[0,1]`.

For the fluted body, let

```text
theta = 2 pi u
R     = 0.50 + 0.14 sin(pi v) + 0.09 sin^2(3 pi v)
r     = R (1 + 0.18 cos(7 theta))
a     = 1 + 0.16 sin(2 pi v)
b     = 1 - 0.12 sin(2 pi v)
phi   = theta + 1.5 pi v + 0.20 sin(2 pi v)
P     = (a r cos(phi), 2.7 v, b r sin(phi))
```

The open `v` ends are closed by triangle fans. Their vertices are separate from
the smooth side attributes so that the bottom and top normals are respectively
`(0,-1,0)` and `(0,1,0)`. Each cap center has UV `(0.5,0.5)`, while a ring point
at `theta` has UV `(0.5 + 0.5 cos(theta), 0.5 + 0.5 sin(theta))`.

For the swept knot, let

```text
t     = 2 pi u
theta = 2 pi v
A     = 1 + 0.35 cos(3 t)
A'    = -1.05 sin(3 t)
A''   = -3.15 cos(3 t)
C     = (A cos(2t), A sin(2t), 0.35 sin(3t))
C'    = (A' cos(2t) - 2 A sin(2t),
         A' sin(2t) + 2 A cos(2t),
         1.05 cos(3t))
C''   = (A'' cos(2t) - 4 A' sin(2t) - 4 A cos(2t),
         A'' sin(2t) + 4 A' cos(2t) - 4 A sin(2t),
         -3.15 sin(3t))
T     = normalize(C')
N     = normalize(C'' - dot(C'',T) T)
B     = cross(T,N)
rho   = 0.10 + 0.025 sin(5t)
P     = C + rho (cos(theta) N + sin(theta) B)
```

This analytic Frenet frame is evaluated independently at every sample; it is
not an accumulated transport frame. Both surfaces compute `P_u` and `P_v` with
centered differences of step `h = 1e-5`, then use
`normalize(cross(P_v,P_u))`. Generation rejects non-finite or degenerate
positions, partial derivatives, normals, and triangles, and verifies outward
winding.

The serialized side grid duplicates seam attributes for UV continuity. At a
canonical seam, position and normal text is copied exactly from the zero end,
while the texture coordinate retains endpoint 1. The body canonicalizes its
`u=1` seam; the knot canonicalizes both `u=1` and `v=1`. Body cap-ring positions
are also textually identical to their matching side-ring positions. Exact
position welding therefore produces a closed manifold with every edge incident
to two triangles. Counts below are for that welded topology (`V/E/F`); the body
has sphere topology and the uncapped knot has torus topology.

| Mesh | Grid | Triangles | Welded V/E/F | Euler characteristic |
| --- | ---: | ---: | ---: | ---: |
| `fluted-body-low.obj` | 32 x 24 | 1,600 | 802 / 2,400 / 1,600 | 2 |
| `fluted-body-medium.obj` | 64 x 48 | 6,272 | 3,138 / 9,408 / 6,272 | 2 |
| `fluted-body-high.obj` | 128 x 96 | 24,832 | 12,418 / 37,248 / 24,832 | 2 |
| `swept-knot-low.obj` | 64 x 8 | 1,024 | 512 / 1,536 / 1,024 | 0 |
| `swept-knot-medium.obj` | 128 x 16 | 4,096 | 2,048 / 6,144 / 4,096 | 0 |
| `swept-knot-high.obj` | 256 x 32 | 16,384 | 8,192 / 24,576 / 16,384 | 0 |

## Procedural showcase scenes

The six individual high-mesh scenes are locked to these filenames:

- `procedural-fluted-body-plain.json`, `procedural-fluted-body-marble.json`,
  and `procedural-fluted-body-cells.json`
- `procedural-swept-knot-plain.json`, `procedural-swept-knot-marble.json`, and
  `procedural-swept-knot-cells.json`

Within each trio the camera and light objects are identical. Each scene is
640 x 424, 512 samples, and depth 10. The body is at `(0,0,0)` with eye
`(2.3,1.9,4.2)`, look-at `(0,1.35,0)`, and `FOVY` 18. The knot is at
`(0,1.5,0)` with eye `(2.8,2.3,5.2)`, look-at `(0,1.5,0)`, and `FOVY` 18.
The shape material has `RGB` `(0.95,0.95,0.95)` and uses none, marble, or cells
as named; it has no file texture or bump map. All six retain the workbench floor
(center `(0,-0.2,0)`, scale `(18,0.4,18)`, so its top is at `y=0`), back wall
(center `(0,3,-3.2)`, scale `(14,6,0.2)`), warm overhead light (center
`(-2,5.3,2)`, scale `(4,0.15,2)`, RGB `(1,0.82,0.62)`, emittance 16), and cool
side light (center `(4,3.8,-0.5)`, scale `(0.15,3,2)`, RGB `(0.5,0.72,1)`,
emittance 12).

`procedural-optical-stilllife.json` combines the marble body at `(-2,0,0)`, the
cells knot at `(2.3,1.5,-0.15)`, and a glass sphere at `(0,0.65,1)` with scale
`(1.3,1.3,1.3)`, arranged without overlap. It uses the same floor, wall, and
warm/cool lights, plus the original workbench camera: eye `(4.8,2.8,7.2)`,
look-at `(0,1.2,0)`, and `FOVY` 22. Its settings are 960 x 640, 2,048 samples,
and depth 10.

For reproducible renders of these seven scenes, pass `--seed 947` on the
headless renderer command line. Seed is intentionally a CLI setting, not a new
JSON scene property.

## Asset regeneration

Run `python3 scripts/generate_assets.py` to regenerate the original closed
ceramic profiles, torus, the two procedural meshes at all three LODs, triangle
light, glaze color map, height map, showcase scenes, and controlled
benchmark/lighting scenes. This is an offline CPU/Python asset-generation step;
the renderer consumes the checked-in OBJ files and does not require Python.

Use

```text
python3 scripts/generate_assets.py --profile OUTPUT_JSON --profile-only
```

to measure procedural construction, OBJ serialization, and temporary-file
writes for the six new meshes without regenerating or touching any checked-in
asset or scene. Profiling performs one warmup and three measured runs per mesh;
every run writes to a `tempfile.TemporaryDirectory` and independently validates
the serialized topology, seams, positions, normals, UVs, triangle areas, and
winding. Validation is timed separately and is excluded from the reported
generation-and-write interval. Omitting `--profile-only` first performs the
ordinary full regeneration and then performs the same temporary-file profile.

Each profile record uses `relative_path` for the canonical checked-in asset and
reports separate generated and canonical byte sizes and SHA-256 digests plus
`byte_identical_to_canonical`. Canonical assets are read only for this
comparison. The equality flag may legitimately be false on a host whose math
library produces different last-bit text while the independently validated
mesh remains correct; profiling never replaces the canonical file with the
temporary result.

The JSON purpose is `cpu-procedural-generation-profile`. Its wall-clock values
apply to the recorded hardware, Python runtime, operating system, filesystem,
cache state, host load, and process invocation. Writes are OS-buffered and do
not include an `fsync`; validation, hashing, canonical reads/comparison, and
report emission are outside the generation interval. This is CPU generation
profiling only: the generator performs and reports no GPU geometry generation
or GPU timings.
