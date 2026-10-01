#!/usr/bin/env python3
"""Deterministically convert the project's canonical OBJ meshes to glTF 2.0."""

import argparse
import base64
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile


ROOT = Path(__file__).resolve().parents[1]
GLTF_DIR = Path("scenes/assets/gltf")
MANIFEST = GLTF_DIR / "generation.json"

MESH_SOURCES = {
    "vase-low": Path("scenes/assets/vase-low.obj"),
    "vase-medium": Path("scenes/assets/vase-medium.obj"),
    "vase-high": Path("scenes/assets/vase-high.obj"),
    "fluted-body-medium": Path("scenes/assets/fluted-body-medium.obj"),
    "swept-knot-medium": Path("scenes/assets/swept-knot-medium.obj"),
}
SCENE_SOURCES = (
    Path("scenes/complexity-low.json"),
    Path("scenes/complexity-medium.json"),
    Path("scenes/complexity-high.json"),
    Path("scenes/hedge-motion.json"),
    Path("scenes/hedge-motion-validation.json"),
)
SCENE_OUTPUTS = (
    Path("scenes/gltf-complexity-low.json"),
    Path("scenes/gltf-complexity-medium.json"),
    Path("scenes/gltf-complexity-high.json"),
    Path("scenes/gltf-showcase.json"),
    Path("scenes/gltf-baseline.json"),
    Path("scenes/gltf-equivalence-glb.json"),
    Path("scenes/gltf-equivalence-external.json"),
    Path("scenes/gltf-equivalence-data-uri.json"),
    Path("scenes/gltf-motion-validation.json"),
)

IDENTITY = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]


@dataclass(frozen=True)
class Source:
    data: bytes
    mtime_ns: int


@dataclass(frozen=True)
class Mesh:
    binary: bytes
    vertices: tuple
    indices: tuple
    source_counts: dict
    bounds_min: tuple
    bounds_max: tuple


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _float(token, where):
    try:
        value = float(token)
    except ValueError as exc:
        raise ValueError(f"{where}: invalid floating-point value {token!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{where}: non-finite floating-point value")
    return value


def _f32(value, where):
    try:
        rounded = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError(f"{where}: value is outside float32 range") from exc
    if not math.isfinite(rounded):
        raise ValueError(f"{where}: value becomes non-finite in float32")
    return rounded


def _obj_index(token, count, where):
    try:
        raw = int(token)
    except ValueError as exc:
        raise ValueError(f"{where}: invalid OBJ index {token!r}") from exc
    if raw == 0:
        raise ValueError(f"{where}: OBJ indices cannot be zero")
    index = raw - 1 if raw > 0 else count + raw
    if not 0 <= index < count:
        raise ValueError(f"{where}: OBJ index {raw} is out of range for {count} values")
    return index


def parse_obj(data, label):
    """Parse the canonical triangle-only v/vt/vn OBJ subset without repairing it."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label}: OBJ is not UTF-8") from exc

    positions, texcoords, normals, faces = [], [], [], []
    for line_number, original in enumerate(text.splitlines(), 1):
        line = original.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        kind = fields[0]
        where = f"{label}:{line_number}"
        if kind == "v":
            if len(fields) != 4:
                raise ValueError(f"{where}: positions must have exactly three components")
            positions.append(tuple(_float(value, where) for value in fields[1:]))
        elif kind == "vt":
            if len(fields) != 3:
                raise ValueError(f"{where}: texture coordinates must have exactly two components")
            uv = tuple(_float(value, where) for value in fields[1:])
            if any(value < 0.0 or value > 1.0 for value in uv):
                raise ValueError(f"{where}: texture coordinate is outside [0, 1]")
            texcoords.append(uv)
        elif kind == "vn":
            if len(fields) != 4:
                raise ValueError(f"{where}: normals must have exactly three components")
            normal = tuple(_float(value, where) for value in fields[1:])
            if sum(value * value for value in normal) <= 0.0:
                raise ValueError(f"{where}: normal is degenerate")
            normals.append(normal)
        elif kind == "f":
            if len(fields) != 4:
                raise ValueError(f"{where}: only triangular faces are supported")
            face = []
            for corner in fields[1:]:
                parts = corner.split("/")
                if len(parts) != 3 or any(not part for part in parts):
                    raise ValueError(f"{where}: every face corner must provide v/vt/vn")
                face.append((
                    _obj_index(parts[0], len(positions), where),
                    _obj_index(parts[1], len(texcoords), where),
                    _obj_index(parts[2], len(normals), where),
                ))
            faces.append(tuple(face))
        else:
            raise ValueError(f"{where}: unsupported canonical OBJ statement {kind!r}")

    if not positions or not texcoords or not normals or not faces:
        raise ValueError(f"{label}: OBJ requires positions, texture coordinates, normals, and faces")

    tuple_indices = {}
    vertices, indices = [], []
    for face in faces:
        for key in face:
            if key not in tuple_indices:
                position = tuple(_f32(value, label) for value in positions[key[0]])
                normal = tuple(_f32(value, label) for value in normals[key[2]])
                if sum(value * value for value in normal) <= 0.0:
                    raise ValueError(f"{label}: a normal becomes degenerate in float32")
                source_uv = texcoords[key[1]]
                # The loader maps glTF (u, v) to renderer (u, 1-v), so store
                # the inverse mapping to preserve the canonical OBJ convention.
                uv = (_f32(source_uv[0], label), _f32(1.0 - source_uv[1], label))
                tuple_indices[key] = len(vertices)
                vertices.append(position + normal + uv)
            indices.append(tuple_indices[key])

    if len(vertices) > 0xFFFFFFFF or len(indices) > 0xFFFFFFFF:
        raise ValueError(f"{label}: mesh exceeds uint32 glTF limits")
    vertex_bytes = b"".join(struct.pack("<8f", *vertex) for vertex in vertices)
    index_bytes = b"".join(struct.pack("<I", index) for index in indices)
    positions32 = [vertex[:3] for vertex in vertices]
    return Mesh(
        binary=vertex_bytes + index_bytes,
        vertices=tuple(vertices),
        indices=tuple(indices),
        source_counts={
            "positions": len(positions),
            "texcoords": len(texcoords),
            "normals": len(normals),
            "triangles": len(faces),
        },
        bounds_min=tuple(min(value[axis] for value in positions32) for axis in range(3)),
        bounds_max=tuple(max(value[axis] for value in positions32) for axis in range(3)),
    )


def gltf_document(name, mesh, uri=None):
    vertex_bytes = len(mesh.vertices) * 32
    index_bytes = len(mesh.indices) * 4
    buffer = {"byteLength": len(mesh.binary)}
    if uri is not None:
        buffer["uri"] = uri
    return {
        "accessors": [
            {
                "bufferView": 0, "byteOffset": 0, "componentType": 5126,
                "count": len(mesh.vertices), "type": "VEC3",
                "min": list(mesh.bounds_min), "max": list(mesh.bounds_max),
            },
            {
                "bufferView": 0, "byteOffset": 12, "componentType": 5126,
                "count": len(mesh.vertices), "type": "VEC3",
            },
            {
                "bufferView": 0, "byteOffset": 24, "componentType": 5126,
                "count": len(mesh.vertices), "type": "VEC2",
            },
            {
                "bufferView": 1, "byteOffset": 0, "componentType": 5125,
                "count": len(mesh.indices), "type": "SCALAR",
                "min": [min(mesh.indices)], "max": [max(mesh.indices)],
            },
        ],
        "asset": {"generator": "scripts/generate_gltf_assets.py", "version": "2.0"},
        "buffers": [buffer],
        "bufferViews": [
            {
                "buffer": 0, "byteOffset": 0, "byteLength": vertex_bytes,
                "byteStride": 32, "target": 34962,
            },
            {
                "buffer": 0, "byteOffset": vertex_bytes,
                "byteLength": index_bytes, "target": 34963,
            },
        ],
        "meshes": [{
            "name": name,
            "primitives": [{
                "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                "indices": 3,
                "mode": 4,
            }],
        }],
        "nodes": [{"matrix": IDENTITY, "mesh": 0, "name": name}],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
    }


def compact_json(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"),
                      sort_keys=True).encode("utf-8")


def make_glb(document, binary):
    encoded = compact_json(document)
    json_chunk = encoded + b" " * (-len(encoded) % 4)
    bin_chunk = binary + b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    return (struct.pack("<4sII", b"glTF", 2, total)
            + struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk
            + struct.pack("<I4s", len(bin_chunk), b"BIN\0") + bin_chunk)


def _json_scene(value):
    return (json.dumps(value, ensure_ascii=True, indent=2) + "\n").encode("utf-8")


def _load_scene(sources, path):
    try:
        scene = json.loads(sources[path].data)
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid or missing source scene") from exc
    if not isinstance(scene, dict) or not isinstance(scene.get("Objects"), list):
        raise ValueError(f"{path}: expected an object with an Objects array")
    return scene


def _replace_object(scene, old_file, new_file):
    matches = [item for item in scene["Objects"]
               if item.get("TYPE") == "obj" and item.get("FILE") == old_file]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one OBJ object for {old_file}, found {len(matches)}")
    matches[0]["TYPE"] = "gltf"
    matches[0]["FILE"] = new_file
    return matches[0]


def _set_camera_file(scene, value):
    camera = scene.get("Camera")
    if not isinstance(camera, dict) or "FILE" not in camera:
        raise ValueError("source scene has no Camera.FILE")
    camera["FILE"] = value


def build_scenes(sources):
    results = {}
    complexities = {}
    for level in ("low", "medium", "high"):
        source_path = Path(f"scenes/complexity-{level}.json")
        scene = _load_scene(sources, source_path)
        _replace_object(scene, f"assets/vase-{level}.obj",
                        f"assets/gltf/vase-{level}.glb")
        _set_camera_file(scene, f"gltf-complexity-{level}")
        complexities[level] = scene
        results[Path(f"scenes/gltf-complexity-{level}.json")] = _json_scene(scene)

    showcase = _load_scene(sources, Path("scenes/hedge-motion.json"))
    _replace_object(showcase, "assets/fluted-body-high.obj",
                    "assets/gltf/fluted-body-medium.glb")
    _replace_object(showcase, "assets/swept-knot-high.obj",
                    "assets/gltf/swept-knot-medium.glb")
    _set_camera_file(showcase, "gltf-showcase")
    camera = showcase["Camera"]
    if (camera.get("RES") != [960, 640] or camera.get("ITERATIONS") != 2048
            or camera.get("DEPTH") != 10 or camera.get("SHUTTER") != [0.0, 1.0]):
        raise ValueError("hedge-motion source no longer has the locked capture camera")
    results[Path("scenes/gltf-showcase.json")] = _json_scene(showcase)

    baseline = copy.deepcopy(showcase)
    imported = {
        "assets/gltf/fluted-body-medium.glb",
        "assets/gltf/swept-knot-medium.glb",
    }
    baseline["Objects"] = [item for item in baseline["Objects"]
                           if item.get("FILE") not in imported]
    if len(showcase["Objects"]) - len(baseline["Objects"]) != 2:
        raise ValueError("gltf baseline did not remove exactly two imported shapes")
    _set_camera_file(baseline, "gltf-baseline")
    results[Path("scenes/gltf-baseline.json")] = _json_scene(baseline)

    encodings = {
        "glb": "assets/gltf/vase-low.glb",
        "external": "assets/gltf/vase-low.gltf",
        "data-uri": "assets/gltf/vase-low-data-uri.gltf",
    }
    for encoding, filename in encodings.items():
        scene = copy.deepcopy(complexities["low"])
        mesh_objects = [item for item in scene["Objects"] if item.get("TYPE") == "gltf"]
        if len(mesh_objects) != 1:
            raise ValueError("low-complexity derivation no longer has one glTF object")
        mesh_objects[0]["FILE"] = filename
        scene["Camera"]["RES"] = [17, 13]
        scene["Camera"]["ITERATIONS"] = 17
        scene["Camera"]["DEPTH"] = 7
        _set_camera_file(scene, f"gltf-equivalence-{encoding}")
        results[Path(f"scenes/gltf-equivalence-{encoding}.json")] = _json_scene(scene)

    motion = _load_scene(sources, Path("scenes/hedge-motion-validation.json"))
    moving_mesh = _replace_object(motion, "assets/vase-low.obj",
                                  "assets/gltf/vase-low.glb")
    if "MOTION_TRANSLATION" not in moving_mesh:
        raise ValueError("motion validation source mesh is not moving")
    motion["Camera"]["RES"] = [17, 13]
    motion["Camera"]["ITERATIONS"] = 17
    motion["Camera"]["DEPTH"] = 7
    _set_camera_file(motion, "gltf-motion-validation")
    results[Path("scenes/gltf-motion-validation.json")] = _json_scene(motion)

    if set(results) != set(SCENE_OUTPUTS):
        raise AssertionError("internal scene output list is inconsistent")
    return results


def _parse_container(path, files):
    encoded = files[path]
    if path.suffix == ".glb":
        if len(encoded) < 28:
            raise ValueError(f"{path}: truncated GLB")
        magic, version, total = struct.unpack_from("<4sII", encoded)
        if magic != b"glTF" or version != 2 or total != len(encoded) or total % 4:
            raise ValueError(f"{path}: invalid GLB v2 header")
        offset, chunks = 12, []
        while offset < len(encoded):
            if offset + 8 > len(encoded):
                raise ValueError(f"{path}: truncated GLB chunk header")
            length, kind = struct.unpack_from("<I4s", encoded, offset)
            offset += 8
            if length % 4 or offset + length > len(encoded):
                raise ValueError(f"{path}: invalid GLB chunk length")
            chunks.append((kind, encoded[offset:offset + length]))
            offset += length
        if [kind for kind, unused in chunks] != [b"JSON", b"BIN\0"]:
            raise ValueError(f"{path}: expected one JSON and one BIN chunk")
        document = json.loads(chunks[0][1])
        raw_buffer = chunks[1][1]
        declared = document["buffers"][0]["byteLength"]
        if declared > len(raw_buffer) or any(raw_buffer[declared:]):
            raise ValueError(f"{path}: invalid GLB BIN padding")
        if "uri" in document["buffers"][0]:
            raise ValueError(f"{path}: GLB buffer unexpectedly has a URI")
        return document, raw_buffer[:declared]

    try:
        document = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid glTF JSON") from exc
    buffer = document["buffers"][0]
    uri = buffer.get("uri")
    if not isinstance(uri, str):
        raise ValueError(f"{path}: JSON glTF buffer has no URI")
    prefix = "data:application/octet-stream;base64,"
    if uri.startswith(prefix):
        try:
            binary = base64.b64decode(uri[len(prefix):], validate=True)
        except ValueError as exc:
            raise ValueError(f"{path}: invalid base64 buffer URI") from exc
    else:
        if Path(uri).name != uri or Path(uri).is_absolute():
            raise ValueError(f"{path}: external buffer URI must be a relative basename")
        target = path.parent / uri
        if target not in files:
            raise ValueError(f"{path}: missing external buffer {target}")
        binary = files[target]
    return document, binary


def inspect_asset(path, files):
    """Decode an output independently and audit its glTF layout and values."""
    document, binary = _parse_container(path, files)
    if document.get("asset", {}).get("version") != "2.0":
        raise ValueError(f"{path}: not glTF 2.0")
    if document.get("scene") != 0 or document.get("scenes") != [{"nodes": [0]}]:
        raise ValueError(f"{path}: default scene/node is not explicit")
    nodes = document.get("nodes")
    if len(nodes) != 1 or nodes[0].get("mesh") != 0 or nodes[0].get("matrix") != IDENTITY:
        raise ValueError(f"{path}: node 0 is not an explicit identity mesh node")
    if len(document.get("buffers", [])) != 1:
        raise ValueError(f"{path}: expected one buffer")
    if document["buffers"][0].get("byteLength") != len(binary) or len(binary) % 4:
        raise ValueError(f"{path}: buffer length or alignment mismatch")

    views = document.get("bufferViews")
    accessors = document.get("accessors")
    if len(views) != 2 or len(accessors) != 4:
        raise ValueError(f"{path}: expected two buffer views and four accessors")
    primitive = document.get("meshes", [{}])[0].get("primitives", [{}])[0]
    if (primitive.get("mode") != 4 or primitive.get("indices") != 3
            or primitive.get("attributes") != {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}):
        raise ValueError(f"{path}: primitive contract mismatch")
    vertex_count = accessors[0].get("count")
    index_count = accessors[3].get("count")
    vertex_bytes = vertex_count * 32
    expected_views = [
        {"buffer": 0, "byteOffset": 0, "byteLength": vertex_bytes,
         "byteStride": 32, "target": 34962},
        {"buffer": 0, "byteOffset": vertex_bytes, "byteLength": index_count * 4,
         "target": 34963},
    ]
    if views != expected_views or vertex_bytes + index_count * 4 != len(binary):
        raise ValueError(f"{path}: buffer view layout mismatch")
    expected_accessors = ((0, 0, 5126, "VEC3"), (0, 12, 5126, "VEC3"),
                          (0, 24, 5126, "VEC2"), (1, 0, 5125, "SCALAR"))
    for accessor, expected in zip(accessors, expected_accessors):
        actual = (accessor.get("bufferView"), accessor.get("byteOffset"),
                  accessor.get("componentType"), accessor.get("type"))
        if actual != expected or accessor.get("count", 0) <= 0:
            raise ValueError(f"{path}: accessor layout mismatch")

    positions, normals, uvs = [], [], []
    for index in range(vertex_count):
        vertex = struct.unpack_from("<8f", binary, index * 32)
        positions.append(vertex[:3])
        normals.append(vertex[3:6])
        uvs.append(vertex[6:8])
    indices = struct.unpack_from(f"<{index_count}I", binary, vertex_bytes)
    if (not all(math.isfinite(value) for vertex in positions + normals + uvs for value in vertex)
            or not all(sum(value * value for value in normal) > 0.0 for normal in normals)):
        raise ValueError(f"{path}: non-finite attribute or degenerate normal")
    if not all(0.0 <= value <= 1.0 for uv in uvs for value in uv):
        raise ValueError(f"{path}: texture coordinate outside [0, 1]")
    if index_count % 3 or not all(index < vertex_count for index in indices):
        raise ValueError(f"{path}: invalid triangle indices")
    bounds_min = [min(value[axis] for value in positions) for axis in range(3)]
    bounds_max = [max(value[axis] for value in positions) for axis in range(3)]
    if accessors[0].get("min") != bounds_min or accessors[0].get("max") != bounds_max:
        raise ValueError(f"{path}: POSITION bounds mismatch")
    if accessors[3].get("min") != [min(indices)] or accessors[3].get("max") != [max(indices)]:
        raise ValueError(f"{path}: index bounds mismatch")
    return {
        "binary": binary,
        "vertices": vertex_count,
        "indices": index_count,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
    }


def asset_paths():
    paths = {GLTF_DIR / f"{name}.glb" for name in MESH_SOURCES}
    paths.update({GLTF_DIR / "vase-low.gltf", GLTF_DIR / "vase-low.bin",
                  GLTF_DIR / "vase-low-data-uri.gltf", MANIFEST})
    return paths


def build_outputs(sources):
    meshes = {name: parse_obj(sources[path].data, str(path))
              for name, path in MESH_SOURCES.items()}
    outputs = {}
    mesh_assets = {}
    for name, mesh in meshes.items():
        path = GLTF_DIR / f"{name}.glb"
        outputs[path] = make_glb(gltf_document(name, mesh), mesh.binary)
        mesh_assets[name] = {"glb": path}

    low = meshes["vase-low"]
    external_bin = GLTF_DIR / "vase-low.bin"
    external_json = GLTF_DIR / "vase-low.gltf"
    data_json = GLTF_DIR / "vase-low-data-uri.gltf"
    outputs[external_bin] = low.binary
    outputs[external_json] = compact_json(
        gltf_document("vase-low", low, external_bin.name)) + b"\n"
    data_uri = "data:application/octet-stream;base64," + base64.b64encode(low.binary).decode("ascii")
    outputs[data_json] = compact_json(gltf_document("vase-low", low, data_uri)) + b"\n"
    mesh_assets["vase-low"].update({
        "external_gltf": external_json,
        "external_bin": external_bin,
        "data_uri_gltf": data_json,
    })

    scene_outputs = build_scenes(sources)
    outputs.update(scene_outputs)

    for name, paths in mesh_assets.items():
        binaries = []
        for kind, path in paths.items():
            if kind != "external_bin":
                inspected = inspect_asset(path, outputs)
                binaries.append(inspected["binary"])
                if inspected["binary"] != meshes[name].binary:
                    raise ValueError(f"{path}: decoded binary differs from source conversion")
        if any(binary != binaries[0] for binary in binaries[1:]):
            raise ValueError(f"{name}: encodings do not contain identical attribute/index bytes")

    generator_data = Path(__file__).resolve().read_bytes()
    manifest_meshes = {}
    for name, mesh in meshes.items():
        source_path = MESH_SOURCES[name]
        manifest_meshes[name] = {
            "assets": {
                kind: {
                    "bytes": len(outputs[path]),
                    "path": path.as_posix(),
                    "sha256": sha256(outputs[path]),
                }
                for kind, path in mesh_assets[name].items()
            },
            "binary": {"bytes": len(mesh.binary), "sha256": sha256(mesh.binary)},
            "counts": {
                **mesh.source_counts,
                "exported_vertices": len(mesh.vertices),
                "indices": len(mesh.indices),
            },
            "position_bounds_float32": {
                "max": list(mesh.bounds_max), "min": list(mesh.bounds_min),
            },
            "source": {
                "path": source_path.as_posix(),
                "sha256": sha256(sources[source_path].data),
            },
        }
    manifest = {
        "format": {
            "container": "glTF 2.0; GLB v2 or JSON with external/data URI buffer",
            "indices": "little-endian uint32, TRIANGLES",
            "vertex": "interleaved little-endian float32 POSITION(3), NORMAL(3), TEXCOORD_0(2); byteStride 32",
            "uv_contract": (
                "stored glTF v = float32(1 - canonical OBJ v); the loader applies 1-v. "
                "The restored renderer v can differ from direct OBJ v by float32 rounding and is not claimed bit-exact."
            ),
        },
        "generated_scenes": {
            path.as_posix(): sha256(data) for path, data in sorted(scene_outputs.items())
        },
        "generator": {
            "path": "scripts/generate_gltf_assets.py",
            "sha256": sha256(generator_data),
        },
        "meshes": manifest_meshes,
        "scene_sources": {
            path.as_posix(): sha256(sources[path].data) for path in SCENE_SOURCES
        },
        "schema": "cis565-gltf-generation-v1",
    }
    outputs[MANIFEST] = _json_scene(manifest)
    expected = asset_paths() | set(SCENE_OUTPUTS)
    if set(outputs) != expected:
        raise AssertionError("internal output allowlist is inconsistent")
    return outputs


def read_sources():
    sources = {}
    for relative in (*MESH_SOURCES.values(), *SCENE_SOURCES):
        path = ROOT / relative
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
            raise RuntimeError(f"input changed while being read: {relative}")
        sources[relative] = Source(data, after.st_mtime_ns)
    return sources


def assert_sources_unchanged(sources):
    for relative, source in sources.items():
        path = ROOT / relative
        if path.stat().st_mtime_ns != source.mtime_ns or path.read_bytes() != source.data:
            raise RuntimeError(f"input bytes or mtime changed during generation: {relative}")


def verify_outputs(expected):
    missing, changed = [], []
    for relative, data in expected.items():
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative.as_posix())
        elif path.read_bytes() != data:
            changed.append(relative.as_posix())
    directory = ROOT / GLTF_DIR
    allowed_names = {path.name for path in asset_paths()}
    extras = sorted(path.name for path in directory.iterdir()
                    if path.name not in allowed_names) if directory.is_dir() else []
    if missing or changed or extras:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if changed:
            details.append("different: " + ", ".join(changed))
        if extras:
            details.append("unexpected glTF assets: " + ", ".join(extras))
        raise RuntimeError("verification failed (" + "; ".join(details) + ")")

    disk_files = {path: (ROOT / path).read_bytes() for path in asset_paths()
                  if path != MANIFEST}
    decoded = [inspect_asset(GLTF_DIR / name, disk_files)["binary"] for name in (
        "vase-low.glb", "vase-low.gltf", "vase-low-data-uri.gltf")]
    if not decoded[0] == decoded[1] == decoded[2]:
        raise RuntimeError("vase-low encodings do not decode to identical binary data")


def write_outputs(outputs):
    for relative, data in sorted(outputs.items()):
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not path.is_file():
            raise RuntimeError(f"refusing to replace non-file output: {relative}")
        if not path.exists() or path.read_bytes() != data:
            path.write_bytes(data)


def _expect_failure(action, description):
    try:
        action()
    except (ValueError, KeyError, IndexError, struct.error):
        return
    raise AssertionError(f"self-test expected failure: {description}")


def self_test():
    fixture = """# independently authored triangle fixture
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
vn 0 0 1
vn 0 0 1
vn 0 0 1
f -3/-3/-3 -2/-2/-2 -1/-1/-1
"""
    with tempfile.TemporaryDirectory(prefix="gltf-generator-self-test-") as temp:
        fixture_path = Path(temp) / "triangle.obj"
        fixture_path.write_text(fixture, encoding="utf-8")
        mesh = parse_obj(fixture_path.read_bytes(), str(fixture_path))
        if len(mesh.vertices) != 3 or mesh.indices != (0, 1, 2):
            raise AssertionError("tuple indexing self-test failed")
        base = Path("fixture")
        files = {
            base / "triangle.bin": mesh.binary,
            base / "triangle.glb": make_glb(gltf_document("triangle", mesh), mesh.binary),
            base / "triangle.gltf": compact_json(
                gltf_document("triangle", mesh, "triangle.bin")) + b"\n",
            base / "triangle-data.gltf": compact_json(gltf_document(
                "triangle", mesh,
                "data:application/octet-stream;base64," +
                base64.b64encode(mesh.binary).decode("ascii"))) + b"\n",
        }
        decoded = [inspect_asset(path, files)["binary"] for path in (
            base / "triangle.glb", base / "triangle.gltf", base / "triangle-data.gltf")]
        if not decoded[0] == decoded[1] == decoded[2] == mesh.binary:
            raise AssertionError("encoding equivalence self-test failed")

        corrupt_glb = bytearray(files[base / "triangle.glb"])
        struct.pack_into("<I", corrupt_glb, 8, len(corrupt_glb) + 4)
        _expect_failure(lambda: inspect_asset(base / "triangle.glb",
                                              {**files, base / "triangle.glb": bytes(corrupt_glb)}),
                        "corrupt GLB length")
        invalid = (
            fixture.replace("f -3/-3/-3 -2/-2/-2 -1/-1/-1", "f 1/1 2/2 3/3"),
            fixture.replace("f -3/-3/-3 -2/-2/-2 -1/-1/-1", "f 1/1/1 2/2/2 3/3/3 1/1/1"),
            fixture.replace("v 0 0 0", "v nan 0 0", 1),
            fixture.replace("-3/-3/-3", "0/-3/-3", 1),
            fixture.replace("-3/-3/-3", "8/-3/-3", 1),
        )
        for number, text in enumerate(invalid):
            _expect_failure(lambda text=text: parse_obj(text.encode(), f"invalid-{number}"),
                            f"invalid OBJ fixture {number}")
    print("self-test passed: OBJ rejection, GLB structure, accessors, and three encodings")


def main():
    parser = argparse.ArgumentParser(
        description="Generate deterministic glTF assets and derived scenes from canonical local OBJ/JSON inputs.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true",
                      help="recompute and compare every owned output without writing")
    mode.add_argument("--self-test", action="store_true",
                      help="run isolated tiny-fixture tests without reading production assets")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    sources = read_sources()
    outputs = build_outputs(sources)
    if args.verify:
        verify_outputs(outputs)
        action = "verified"
    else:
        write_outputs(outputs)
        verify_outputs(outputs)
        action = "generated"
    assert_sources_unchanged(sources)
    print(f"{action} {len(outputs)} deterministic files; canonical input bytes and mtimes unchanged")


if __name__ == "__main__":
    main()
