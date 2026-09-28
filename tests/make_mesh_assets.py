"""Generate our own mesh assets and small glTF loader fixtures."""
import base64
import copy
import json
import math
from pathlib import Path
import struct

root = Path(__file__).resolve().parents[1]
assets = root / "scenes" / "meshes"
fixtures = root / "tests" / "meshes"
assets.mkdir(exist_ok=True)
fixtures.mkdir(exist_ok=True)

def write_mesh(folder, name, positions, indices=None, component=5123, stride=12, embedded=False, glb=False):
    data = b"".join(struct.pack("<3f", *p) + bytes(stride - 12) for p in positions)
    views = [{"buffer": 0, "byteOffset": 0, "byteLength": len(data), "byteStride": stride}]
    accessors = [{"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3",
                  "min": [min(p[i] for p in positions) for i in range(3)],
                  "max": [max(p[i] for p in positions) for i in range(3)]}]
    primitive = {"attributes": {"POSITION": 0}, "mode": 4}
    if indices is not None:
        offset = len(data)
        encoded = struct.pack("<" + {5121: "B", 5123: "H", 5125: "I"}[component] * len(indices), *indices)
        data += encoded
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(encoded)})
        accessors.append({"bufferView": 1, "componentType": component, "count": len(indices), "type": "SCALAR"})
        primitive["indices"] = 1
    buffer = {"byteLength": len(data)}
    if embedded:
        buffer["uri"] = "data:application/octet-stream;base64," + base64.b64encode(data).decode()
    elif not glb:
        buffer["uri"] = name + ".bin"
        (folder / buffer["uri"]).write_bytes(data)
    doc = {"asset": {"version": "2.0", "generator": "Project 3 mesh fixtures"},
           "buffers": [buffer], "bufferViews": views, "accessors": accessors,
           "meshes": [{"primitives": [primitive]}],
           "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}], "scene": 0}
    if glb:
        text = json.dumps(doc).encode()
        text += b" " * (-len(text) % 4)
        data += bytes(-len(data) % 4)
        chunks = struct.pack("<II", len(text), 0x4E4F534A) + text
        chunks += struct.pack("<II", len(data), 0x004E4942) + data
        (folder / (name + ".glb")).write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(chunks)) + chunks)
    else:
        (folder / (name + ".gltf")).write_text(json.dumps(doc, indent=2))
    return doc

positions = []
indices = []
around, tube = 40, 12
for i in range(around):
    u = 2 * math.pi * i / around
    for j in range(tube):
        v = 2 * math.pi * j / tube
        radius = 0.8 + 0.28 * math.cos(v)
        positions.append((radius * math.cos(u), radius * math.sin(u), 0.28 * math.sin(v)))
for i in range(around):
    for j in range(tube):
        a = i * tube + j
        b = ((i + 1) % around) * tube + j
        c = ((i + 1) % around) * tube + (j + 1) % tube
        d = i * tube + (j + 1) % tube
        indices.extend((a, b, c, a, c, d))
write_mesh(assets, "torus", positions, indices)

triangle = [(-1, -1, 0), (1, -1, 0), (0, 1, 0)]
for name, component, stride, embedded, glb in [
    ("external", 5123, 12, False, False),
    ("interleaved", 5125, 24, False, False),
    ("embedded", 5121, 12, True, False),
    ("binary", 5123, 12, False, True),
]:
    write_mesh(fixtures, name, triangle, [0, 1, 2], component, stride, embedded, glb)
doc = write_mesh(fixtures, "unindexed", triangle)
doc["nodes"] = [{"translation": [2, 0, 0], "children": [1]},
                {"mesh": 0, "scale": [-2, 3, 1]}]
(fixtures / "transformed.gltf").write_text(json.dumps(doc, indent=2))
bad = copy.deepcopy(doc)
bad["accessors"][0]["count"] = 999
(fixtures / "invalid-range.gltf").write_text(json.dumps(bad, indent=2))
bad = copy.deepcopy(doc)
bad["nodes"] = [{"children": [0]}]
(fixtures / "cycle.gltf").write_text(json.dumps(bad, indent=2))

scene = {
    "Materials": {
        "white": {"TYPE": "Diffuse", "RGB": [0.8, 0.8, 0.8]},
        "gold": {"TYPE": "Diffuse", "RGB": [0.85, 0.5, 0.15]},
        "blue": {"TYPE": "Diffuse", "RGB": [0.15, 0.5, 0.85]},
        "light": {"TYPE": "Emitting", "RGB": [1, 1, 1], "EMITTANCE": 8}
    },
    "Camera": {"RES": [240, 180], "FOVY": 25, "ITERATIONS": 768, "DEPTH": 6,
               "FILE": "build/mesh-preview", "EYE": [0, 2, 8], "LOOKAT": [0, 2, 0], "UP": [0, 1, 0],
               "MESH_CULLING": True},
    "Objects": [
        {"TYPE": "mesh", "FILE": "meshes/torus.gltf", "MATERIAL": "gold", "TRANS": [-1.3, 1.6, 0],
         "ROTAT": [0, 25, 0], "SCALE": [1.4, 1.4, 1.4]},
        {"TYPE": "mesh", "FILE": "meshes/torus.gltf", "MATERIAL": "blue", "TRANS": [1.5, 1.4, -1],
         "ROTAT": [0, -30, 20], "SCALE": [1.2, 1.2, 1.2]},
        {"TYPE": "cube", "MATERIAL": "white", "TRANS": [0, -0.1, 0],
         "ROTAT": [0, 0, 0], "SCALE": [20, 0.2, 20]},
        {"TYPE": "cube", "MATERIAL": "light", "TRANS": [0, 7, 2],
         "ROTAT": [0, 0, 0], "SCALE": [7, 0.1, 7]}
    ]
}
(root / "scenes/mesh-preview.json").write_text(json.dumps(scene, indent=2))
scene["Camera"].update(RES=[320, 240], ITERATIONS=100, DEPTH=6, FILE="build/mesh-benchmark")
(root / "scenes/benchmark-mesh.json").write_text(json.dumps(scene, indent=2))

dof = copy.deepcopy(scene)
dof["Camera"].update(RES=[300, 180], ITERATIONS=1024, DEPTH=1, FILE="build/dof-on",
                     EYE=[0, 0, 9], LOOKAT=[0, 0, 0], APERTURE_RADIUS=0.6, FOCAL_DISTANCE=9)
dof["Materials"] = {
    "pink": {"TYPE": "Emitting", "RGB": [1, 0.25, 0.35], "EMITTANCE": 1},
    "gold": {"TYPE": "Emitting", "RGB": [1, 0.65, 0.12], "EMITTANCE": 1},
    "blue": {"TYPE": "Emitting", "RGB": [0.2, 0.55, 1], "EMITTANCE": 1}
}
dof["Objects"] = [
    {"TYPE": "mesh", "FILE": "meshes/torus.gltf", "MATERIAL": material, "TRANS": pos,
     "ROTAT": [0, 0, 0], "SCALE": [1, 1, 1]}
    for material, pos in [("pink", [-2.2, 0, 2]), ("gold", [0, 0, 0]), ("blue", [3, 0, -3])]
]
(root / "scenes/depth-of-field.json").write_text(json.dumps(dof, indent=2))
dof["Camera"].update(APERTURE_RADIUS=0, FILE="build/dof-off")
(root / "scenes/depth-of-field-pinhole.json").write_text(json.dumps(dof, indent=2))
