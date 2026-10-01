#!/usr/bin/env python3
"""Run the fixed glTF validation, timing, gallery, and CPU-load studies.

The experimental settings are intentionally not CLI options.  Every process and
artifact is recorded, and ``--output`` must name a new directory so prior
evidence cannot be overwritten.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import statistics
import struct
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Sequence

try:
    from validate_render import fnv1a, read_sum
except ImportError:  # pragma: no cover - module-style invocation
    from scripts.validate_render import fnv1a, read_sum


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "build/bin/cis565_path_tracer_headless"
DEFAULT_BUILD = ROOT / "build"
SCHEMA_VERSION = 1
SEED = 947
DEFAULT_SAMPLER = "counter"
SAMPLER_IDS = {"counter": 2, "halton": 3}

ENCODING_SCENES = {
    "glb": "gltf-equivalence-glb.json",
    "external": "gltf-equivalence-external.json",
    "data-uri": "gltf-equivalence-data-uri.json",
}
MOTION_SCENE = "gltf-motion-validation.json"
GLTF_COMPLEXITY = {
    level: f"gltf-complexity-{level}.json"
    for level in ("low", "medium", "high")
}
OBJ_COMPLEXITY = {
    level: f"complexity-{level}.json"
    for level in ("low", "medium", "high")
}
GALLERY_SCENES = {
    "showcase": "gltf-showcase.json",
    "baseline": "gltf-baseline.json",
}
SOURCE_DIRECTORIES = ("src", "stream_compaction", "external", "cmake")
RESUME_PHYSICAL_FIELDS = (
    "resolution", "target_samples", "seed", "sampler", "sampler_id",
    "sampler_version", "config", "camera", "fingerprints", "compatibility",
)


class EvidenceError(RuntimeError):
    """An invocation or retained artifact violated the fixed design."""


def sampler_id_for(name: str) -> int:
    try:
        return SAMPLER_IDS[name]
    except (KeyError, TypeError) as error:
        raise EvidenceError(f"unsupported sampler: {name!r}") from error


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def suffix(prefix: Path, extension: str) -> Path:
    return Path(str(prefix) + extension)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_file_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise EvidenceError(f"missing file: {path}")
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in fields):
        raise EvidenceError(f"file changed while being hashed: {path}")
    return {
        "path": str(path),
        "sha256": digest,
        "bytes": after.st_size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
    }


def same_file_identity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("path", "sha256", "bytes", "device", "inode", "mtime_ns")
    return all(left.get(key) == right.get(key) for key in keys)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def relative_path(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def file_record(path: Path, evidence_root: Path) -> dict[str, Any]:
    identity = stable_file_identity(path)
    return {
        "path": relative_path(path, evidence_root),
        "sha256": identity["sha256"],
        "bytes": identity["bytes"],
    }


def finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise EvidenceError(f"{label} is not finite and nonnegative: {value!r}")
    return result


def integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceError(f"{label} is not an integer: {value!r}")
    return value


def all_floats_finite(value: Any, label: str = "metadata") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise EvidenceError(f"{label} contains a non-finite value")
    if isinstance(value, dict):
        for key, child in value.items():
            all_floats_finite(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            all_floats_finite(child, f"{label}[{index}]")


def renderer_source_snapshot() -> dict[str, Any]:
    """Reproduce the repository's independent renderer-input hash."""
    paths = [ROOT / "CMakeLists.txt"]
    for directory in SOURCE_DIRECTORIES:
        paths.extend(path for path in (ROOT / directory).rglob("*") if path.is_file())
    paths.sort(key=lambda path: path.relative_to(ROOT).as_posix())
    digest = hashlib.sha256()
    total = 0
    for path in paths:
        name = path.relative_to(ROOT).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "little"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
        total += len(content)
    return {
        "algorithm": "scripts/source_hash.py-compatible-sha256-v1",
        "sha256": digest.hexdigest(),
        "file_count": len(paths),
        "bytes": total,
    }


def _scene_references(document: Any) -> list[str]:
    result: list[str] = []
    if not isinstance(document, dict):
        return result
    for section in (document.get("Materials"), document.get("Objects")):
        values: Iterable[Any]
        if isinstance(section, dict):
            values = section.values()
        elif isinstance(section, list):
            values = section
        else:
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            for key in ("FILE", "TEXTURE", "BUMP"):
                value = item.get(key)
                if isinstance(value, str):
                    result.append(value)
    return result


def _gltf_external_buffers(path: Path) -> list[Path]:
    if path.suffix.lower() != ".gltf":
        return []
    document = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for buffer in document.get("buffers", []):
        uri = buffer.get("uri") if isinstance(buffer, dict) else None
        if not isinstance(uri, str) or uri.startswith("data:"):
            continue
        if "://" in uri or uri.startswith("/"):
            raise EvidenceError(f"unsupported non-local glTF buffer URI in {path}: {uri}")
        result.append((path.parent / uri).resolve())
    return result


def scene_identity(filename: str) -> dict[str, Any]:
    scene = (ROOT / "scenes" / filename).resolve()
    if not scene.is_file():
        raise EvidenceError(f"scene does not exist: {scene}")
    document = json.loads(scene.read_text(encoding="utf-8"))
    paths = {scene}
    for reference in _scene_references(document):
        asset = (scene.parent / reference).resolve()
        if not asset.is_file():
            raise EvidenceError(f"scene asset does not exist: {asset}")
        paths.add(asset)
        paths.update(_gltf_external_buffers(asset))
    identities = [stable_file_identity(path) for path in sorted(paths)]
    core = {
        "scene_file": filename,
        "resolved_path": str(scene),
        "files": identities,
    }
    encoded = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**core, "manifest_sha256": hashlib.sha256(encoded).hexdigest()}


def expected_shutter(filename: str) -> list[float]:
    document = json.loads((ROOT / "scenes" / filename).read_text(encoding="utf-8"))
    value = document.get("Camera", {}).get("SHUTTER", [0.0, 1.0])
    if (not isinstance(value, list) or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, (int, float))
                   or not math.isfinite(float(item)) for item in value)):
        raise EvidenceError(f"invalid Camera.SHUTTER in {filename}")
    return [float(item) for item in value]


@dataclass(frozen=True)
class RenderSpec:
    phase: str
    purpose: str
    label: str
    scene_file: str
    samples: int
    resolution: tuple[int, int]
    depth: int
    bvh: bool
    sampler: str = DEFAULT_SAMPLER
    rr: bool = False
    repeat: int | None = None
    measurement: str | None = None
    checkpoint_mode: str | None = None
    checkpoint_relative: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sampler_id_for(self.sampler)

    @property
    def sampler_id(self) -> int:
        return sampler_id_for(self.sampler)

    @property
    def benchmark(self) -> bool:
        return self.checkpoint_mode != "resume"

    def record(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "purpose": self.purpose,
            "label": self.label,
            "scene_file": self.scene_file,
            "sample_target": self.samples,
            "resolution": list(self.resolution),
            "depth": self.depth,
            "seed": SEED,
            "sampler": self.sampler,
            "sampler_id": self.sampler_id,
            "bvh": self.bvh,
            "rr": self.rr,
            "repeat": self.repeat,
            "measurement_phase": self.measurement,
            "checkpoint_mode": self.checkpoint_mode,
            "checkpoint_relative": self.checkpoint_relative,
            "benchmark": self.benchmark,
            **self.tags,
        }


def expected_config(spec: RenderSpec) -> dict[str, Any]:
    return {
        "max_depth": spec.depth,
        "aa": True,
        "nee": True,
        "bvh": spec.bvh,
        "compact": True,
        "sort": False,
        "wavefront": False,
        "rr": spec.rr,
        "rr_start": 3,
        "sampler": spec.sampler,
        "sampler_id": spec.sampler_id,
        "motion": True,
        "debug": "none",
        "depth_semantics": "maximum surface intersections; final intersection is emission-only",
    }


def render_command(binary: Path, prefix: Path, checkpoint: Path | None,
                   spec: RenderSpec) -> list[str]:
    scene = (ROOT / "scenes" / spec.scene_file).resolve()
    if spec.checkpoint_mode == "resume":
        if checkpoint is None:
            raise EvidenceError(f"{spec.label}: resume checkpoint is missing")
        return [
            str(binary.resolve()), str(scene),
            "--samples", str(spec.samples),
            "--output", str(prefix.resolve()),
            "--resume", str(checkpoint.resolve()),
        ]
    width, height = spec.resolution
    command = [
        str(binary.resolve()), str(scene),
        "--samples", str(spec.samples),
        "--seed", str(SEED),
        "--output", str(prefix.resolve()),
        "--resolution", f"{width}x{height}",
        "--depth", str(spec.depth),
        "--benchmark",
        "--wavefront", "off",
        "--rr", "on" if spec.rr else "off",
        "--rr-start", "3",
        "--compact", "on",
        "--bvh", "on" if spec.bvh else "off",
        "--aa", "on",
        "--nee", "on",
        "--sort", "off",
        "--motion", "on",
        "--sampler", spec.sampler,
    ]
    if spec.checkpoint_mode == "write":
        if checkpoint is None:
            raise EvidenceError(f"{spec.label}: output checkpoint is missing")
        command.extend(("--checkpoint", str(checkpoint.resolve())))
    return command


def motion_group_key(sampler: str, bvh: bool) -> str:
    sampler_id_for(sampler)
    return f"{sampler}-bvh-{'on' if bvh else 'off'}"


def motion_validation_specs() -> list[RenderSpec]:
    result = []
    for sampler in SAMPLER_IDS:
        for bvh in (False, True):
            mode = "on" if bvh else "off"
            group = motion_group_key(sampler, bvh)
            checkpoint = f"validation/checkpoints/motion-{group}.checkpoint"
            common = {"motion_group": group}
            result.extend((
                RenderSpec("validation", "motion-reference",
                           f"motion-{sampler}-reference-bvh-{mode}", MOTION_SCENE,
                           17, (17, 13), 7, bvh, sampler=sampler, tags=common),
                RenderSpec("validation", "motion-partial",
                           f"motion-{sampler}-partial-bvh-{mode}", MOTION_SCENE,
                           5, (17, 13), 7, bvh, sampler=sampler,
                           checkpoint_mode="write", checkpoint_relative=checkpoint,
                           tags=common),
                RenderSpec("validation", "motion-resume",
                           f"motion-{sampler}-resume-bvh-{mode}", MOTION_SCENE,
                           17, (17, 13), 7, bvh, sampler=sampler,
                           checkpoint_mode="resume", checkpoint_relative=checkpoint,
                           tags=common),
            ))
    return result


def validation_specs() -> list[RenderSpec]:
    result = []
    for bvh in (False, True):
        mode = "on" if bvh else "off"
        for encoding, scene in ENCODING_SCENES.items():
            result.append(RenderSpec(
                "validation", "encoding-equivalence",
                f"equivalence-{encoding}-bvh-{mode}", scene, 17, (17, 13), 7, bvh,
                tags={"encoding": encoding},
            ))
    result.extend(motion_validation_specs())
    return result


def timing_specs() -> list[RenderSpec]:
    result = []
    for level, scene in GLTF_COMPLEXITY.items():
        for bvh in (False, True):
            mode = "on" if bvh else "off"
            for measurement in ("warmup", "measured"):
                for repeat in range(1, 4):
                    result.append(RenderSpec(
                        "timing", "timing",
                        f"{level}-bvh-{mode}-{measurement}-{repeat}",
                        scene, 8, (96, 64), 10, bvh,
                        repeat=repeat, measurement=measurement,
                        tags={"complexity": level},
                    ))
    return result


def gallery_specs() -> list[RenderSpec]:
    result = [
        RenderSpec("gallery", "capability", name, scene,
                   128, (512, 336), 10, True,
                   tags={"gallery_role": name, "exact_comparison": False})
        for name, scene in GALLERY_SCENES.items()
    ]
    for level, scene in GLTF_COMPLEXITY.items():
        for bvh in (False, True):
            mode = "on" if bvh else "off"
            result.append(RenderSpec(
                "gallery", "culling", f"{level}-bvh-{mode}", scene,
                16, (128, 84), 10, bvh,
                tags={"complexity": level, "exact_comparison": True},
            ))
    return result


def design() -> dict[str, Any]:
    return {
        "seed": SEED,
        "default_sampler": DEFAULT_SAMPLER,
        "sampler_ids": dict(SAMPLER_IDS),
        "renderer_process_counts": {
            "validation": len(validation_specs()),
            "timing": len(timing_specs()),
            "gallery": len(gallery_specs()),
            "all": len(validation_specs()) + len(timing_specs()) + len(gallery_specs()),
        },
        "validation": {
            "encoding_renders": 6,
            "encoding_sampler": DEFAULT_SAMPLER,
            "motion_renders": 12,
            "motion_samplers": list(SAMPLER_IDS),
            "motion_continuation": "fresh 17 vs checkpoint 5 -> minimal resume 17",
        },
        "timing": {
            "configurations": 6,
            "sampler": DEFAULT_SAMPLER,
            "warmups_per_configuration": 3,
            "measured_repeats_per_configuration": 3,
            "warmups_excluded_from_statistics": True,
        },
        "gallery": {
            "capability_renders": 2,
            "culling_renders": 6,
            "depth": 10,
            "sampler": DEFAULT_SAMPLER,
            "showcase_baseline_exactness": "not asserted: object omission is intentional",
        },
        "fixed_scheduling": {
            "wavefront": "off", "compact": "on", "aa": "on", "nee": "on",
            "sort": "off", "motion": "on", "rr": "off", "rr_start": 3,
        },
    }


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if (len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n"
            or struct.unpack_from(">I", header, 8)[0] != 13
            or header[12:16] != b"IHDR"):
        raise EvidenceError(f"invalid PNG header: {path}")
    return struct.unpack_from(">II", header, 16)


def parse_benchmark(path: Path, completed: int, max_depth: int) -> dict[str, Any]:
    gpu_columns = (
        "intersection_ms", "sort_ms", "shade_ms", "compaction_ms",
        "count_ms", "queue_ms",
    )
    shader_breakdown = (
        "diffuse_ms", "mirror_ms", "dielectric_ms", "emission_ms", "terminal_ms",
    )
    host_columns = ("queue_host_ms",)
    overall = []
    bounce = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "record_type", "host_ms", "gpu_ms", "measured_samples",
            "render_loop_wall_ms", "sample", "bounce", "active_before",
            "active_after", *gpu_columns, *shader_breakdown, *host_columns,
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise EvidenceError(f"{path}: benchmark CSV lacks {sorted(missing)}")
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise EvidenceError(f"{path}:{line}: misaligned CSV row")
            kind = row["record_type"]
            if kind == "overall":
                overall.append(row)
            elif kind == "bounce":
                bounce.append(row)
            else:
                raise EvidenceError(f"{path}:{line}: unknown record_type {kind!r}")
    if len(overall) != 1 or not bounce:
        raise EvidenceError(f"{path}: expected one overall and at least one bounce row")

    def number(row: dict[str, str], name: str, label: str) -> float:
        try:
            return finite_nonnegative(float(row[name]), label)
        except ValueError as error:
            raise EvidenceError(f"{label} is not numeric: {row[name]!r}") from error

    row = overall[0]
    try:
        measured = int(row["measured_samples"])
    except ValueError as error:
        raise EvidenceError(f"{path}: measured_samples is not an integer") from error
    if measured != completed:
        raise EvidenceError(
            f"{path}: measured_samples {measured} != completed samples {completed}"
        )

    topology: dict[int, list[tuple[int, int, int]]] = {}
    for index, item in enumerate(bounce, 2):
        parsed = []
        for name in ("sample", "bounce", "active_before", "active_after"):
            try:
                value = int(item[name])
            except ValueError as error:
                raise EvidenceError(f"{path}:{index}: {name} is not an integer") from error
            if str(value) != item[name] or value < 0:
                raise EvidenceError(f"{path}:{index}: invalid {name} {item[name]!r}")
            parsed.append(value)
        sample, bounce_index, active_before, active_after = parsed
        if sample >= completed or bounce_index >= max_depth or active_before <= 0 \
                or active_after > active_before:
            raise EvidenceError(f"{path}:{index}: invalid bounce topology {parsed}")
        topology.setdefault(sample, []).append((bounce_index, active_before, active_after))
    if set(topology) != set(range(completed)):
        raise EvidenceError(f"{path}: bounce rows do not cover every completed sample")
    for sample, rows in topology.items():
        rows.sort()
        if [row[0] for row in rows] != list(range(len(rows))):
            raise EvidenceError(f"{path}: sample {sample} has non-contiguous bounce rows")
        for current, following in zip(rows, rows[1:]):
            if current[2] == 0 or following[1] != current[2]:
                raise EvidenceError(
                    f"{path}: sample {sample} violates compact-on active-path continuity"
                )

    def column_sum(name: str) -> float:
        return math.fsum(
            number(item, name, f"benchmark.{name}") for item in bounce
        )

    return {
        "renderer_host_ms": number(row, "host_ms", "benchmark.host_ms"),
        "renderer_gpu_ms": number(row, "gpu_ms", "benchmark.gpu_ms"),
        "render_loop_wall_ms": number(
            row, "render_loop_wall_ms", "benchmark.render_loop_wall_ms"
        ),
        "measured_samples": measured,
        "bounce_records": len(bounce),
        "gpu_stage_sums_ms": {name: column_sum(name) for name in gpu_columns},
        "host_stage_sums_ms": {name: column_sum(name) for name in host_columns},
        "wavefront_shader_breakdown_ms": {
            name: column_sum(name) for name in shader_breakdown
        },
        "semantics": {
            "queue_host_ms": "host queue bookkeeping/submission; not GPU work",
            "shader_breakdown": (
                "alternative wavefront breakdown; do not add to aggregate shade_ms"
            ),
        },
    }


def checkpoint_record(path: Path, evidence_root: Path) -> dict[str, Any]:
    encoded = path.read_bytes()
    if len(encoded) < 32 or encoded[:8] != b"C565CKPT":
        raise EvidenceError(f"invalid checkpoint header: {path}")
    version, endian, payload = struct.unpack_from("<IIQ", encoded, 8)
    if version != 3 or endian != 0x01020304:
        raise EvidenceError(
            f"{path}: expected checkpoint v3/little-endian marker, got {version}/{endian:#x}"
        )
    if len(encoded) != 24 + payload + 8:
        raise EvidenceError(f"{path}: checkpoint payload length is inconsistent")
    checksum = struct.unpack_from("<Q", encoded, len(encoded) - 8)[0]
    if fnv1a(encoded[:-8]) != checksum:
        raise EvidenceError(f"{path}: checkpoint checksum mismatch")
    return {
        **file_record(path, evidence_root),
        "format_version": version,
        "payload_bytes": payload,
        "checksum_fnv1a64": checksum,
    }


def artifact_paths(prefix: Path, benchmark: bool) -> dict[str, Path]:
    result = {
        "png": suffix(prefix, ".png"),
        "raw": suffix(prefix, ".sum.bin"),
        "metadata": suffix(prefix, ".json"),
    }
    if benchmark:
        result["benchmark_csv"] = suffix(prefix, ".benchmark.csv")
    return result


def audit_render(prefix: Path, spec: RenderSpec, process_elapsed_ms: float,
                 evidence_root: Path, source_hash: str) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = artifact_paths(prefix, spec.benchmark)
    artifacts = {name: file_record(path, evidence_root) for name, path in paths.items()}
    unexpected_csv = suffix(prefix, ".benchmark.csv")
    if not spec.benchmark and unexpected_csv.exists():
        raise EvidenceError(f"{spec.label}: minimal resume unexpectedly wrote benchmark CSV")
    if png_dimensions(paths["png"]) != spec.resolution:
        raise EvidenceError(f"{spec.label}: PNG resolution differs from the design")

    raw = read_sum(paths["raw"])
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    all_floats_finite(metadata)
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, actual: Any = None) -> None:
        if not condition:
            raise EvidenceError(f"{spec.label}: {name} failed: {actual!r}")
        checks.append({"name": name, "passed": True, "actual": actual})

    width, height = spec.resolution
    check("metadata-format-version", metadata.get("format_version") == 1,
          metadata.get("format_version"))
    target = integer(metadata.get("target_samples"), "metadata.target_samples")
    completed = integer(metadata.get("completed_samples"), "metadata.completed_samples")
    check("target-samples", target == spec.samples, target)
    check("completed-samples", completed == spec.samples, completed)
    check("raw-samples", raw["samples"] == completed, raw["samples"])
    check("raw-resolution", (raw["width"], raw["height"]) == (width, height),
          [raw["width"], raw["height"]])
    check("metadata-resolution", metadata.get("resolution") == [width, height],
          metadata.get("resolution"))
    check("seed", metadata.get("seed") == SEED, metadata.get("seed"))
    check("sampler", metadata.get("sampler") == spec.sampler, metadata.get("sampler"))
    check("sampler-id", metadata.get("sampler_id") == spec.sampler_id,
          metadata.get("sampler_id"))
    check("sampler-version", metadata.get("sampler_version") == spec.sampler_id,
          metadata.get("sampler_version"))
    check("config", metadata.get("config") == expected_config(spec),
          metadata.get("config"))

    expected_scene = (ROOT / "scenes" / spec.scene_file).resolve()
    scene_value = metadata.get("scene")
    check("scene", isinstance(scene_value, str)
          and Path(scene_value).resolve() == expected_scene, scene_value)
    camera = metadata.get("camera")
    check("camera-object", isinstance(camera, dict), camera)
    check("camera-shutter", camera.get("shutter") == expected_shutter(spec.scene_file),
          camera.get("shutter"))

    checksum = fnv1a(raw["encoded"][:-8])
    linear = metadata.get("linear_sum_format")
    check("linear-sum-metadata", isinstance(linear, dict), linear)
    check("linear-sum-version", linear.get("version") == 1, linear.get("version"))
    check("raw-checksum", linear.get("checksum") == checksum, linear.get("checksum"))
    outputs = metadata.get("outputs")
    check("outputs-object", isinstance(outputs, dict), outputs)
    check("output-png", isinstance(outputs.get("png"), str)
          and Path(outputs["png"]).resolve() == paths["png"].resolve(), outputs.get("png"))
    check("output-raw", isinstance(outputs.get("linear_sum"), str)
          and Path(outputs["linear_sum"]).resolve() == paths["raw"].resolve(),
          outputs.get("linear_sum"))
    checkpoint = (evidence_root / spec.checkpoint_relative).resolve() \
        if spec.checkpoint_relative is not None else None
    expected_resume = str(checkpoint) if spec.checkpoint_mode == "resume" else None
    expected_checkpoint = str(checkpoint) if spec.checkpoint_mode == "write" else None
    check("resume-metadata", metadata.get("resumed_from") == expected_resume,
          metadata.get("resumed_from"))
    check("checkpoint-metadata", metadata.get("checkpoint") == expected_checkpoint,
          metadata.get("checkpoint"))
    check("checkpoint-every", metadata.get("checkpoint_every") is None,
          metadata.get("checkpoint_every"))

    fingerprints = metadata.get("fingerprints")
    check("fingerprints-object", isinstance(fingerprints, dict), fingerprints)
    for name in ("scene_file", "assets", "materials", "geometry", "render_config"):
        value = fingerprints.get(name)
        check(f"fingerprint-{name}", isinstance(value, int)
              and not isinstance(value, bool) and value > 0, value)

    compatibility = metadata.get("compatibility")
    check("compatibility-object", isinstance(compatibility, dict), compatibility)
    build_id = compatibility.get("build_identifier")
    check("build-identifier", isinstance(build_id, str) and bool(build_id), build_id)
    check("build-source-hash", source_hash in build_id, build_id)
    starter = compatibility.get("baseline_starter_revision")
    check("starter-revision", isinstance(starter, str) and bool(starter), starter)
    check("device", isinstance(compatibility.get("device"), str)
          and bool(compatibility.get("device")), compatibility.get("device"))

    render_loop = finite_nonnegative(
        metadata.get("render_loop_wall_ms"), "metadata.render_loop_wall_ms"
    )
    policy = metadata.get("run_policy")
    check("run-policy-object", isinstance(policy, dict), policy)
    check("run-policy-fields",
          set(policy) == {"time_budget_ms", "stop_reason", "last_sample_wall_ms"},
          sorted(policy))
    check("run-policy-budget", policy.get("time_budget_ms") is None, policy)
    check("run-policy-stop", policy.get("stop_reason") == "sample_target", policy)
    last_sample = finite_nonnegative(
        policy.get("last_sample_wall_ms"), "run_policy.last_sample_wall_ms"
    )
    check("last-sample-within-loop", last_sample <= render_loop,
          {"last_sample_wall_ms": last_sample, "render_loop_wall_ms": render_loop})

    benchmark = None
    if spec.benchmark:
        check("benchmark-enabled", metadata.get("benchmark_enabled") is True,
              metadata.get("benchmark_enabled"))
        benchmark_metadata = metadata.get("benchmark")
        check("benchmark-object", isinstance(benchmark_metadata, dict), benchmark_metadata)
        check("benchmark-csv-path", isinstance(benchmark_metadata.get("csv"), str)
              and Path(benchmark_metadata["csv"]).resolve() == paths["benchmark_csv"].resolve(),
              benchmark_metadata.get("csv"))
        check("benchmark-samples", benchmark_metadata.get("measured_samples") == completed,
              benchmark_metadata.get("measured_samples"))
        host_ms = finite_nonnegative(
            benchmark_metadata.get("renderer_host_ms"), "benchmark.renderer_host_ms"
        )
        gpu_ms = finite_nonnegative(
            benchmark_metadata.get("renderer_gpu_ms"), "benchmark.renderer_gpu_ms"
        )
        benchmark = parse_benchmark(paths["benchmark_csv"], completed, spec.depth)
        for name, expected in (
            ("renderer_host_ms", host_ms),
            ("renderer_gpu_ms", gpu_ms),
            ("render_loop_wall_ms", render_loop),
        ):
            check(f"benchmark-csv-{name}",
                  math.isclose(benchmark[name], expected, rel_tol=2e-6, abs_tol=1e-5),
                  {"csv": benchmark[name], "metadata": expected})
    else:
        check("benchmark-disabled", metadata.get("benchmark_enabled") is False,
              metadata.get("benchmark_enabled"))
        check("benchmark-absent", "benchmark" not in metadata, metadata.get("benchmark"))
        host_ms = None
        gpu_ms = None

    elapsed = finite_nonnegative(process_elapsed_ms, "process_elapsed_ms")
    check("process-covers-render-loop", elapsed + 1e-6 >= render_loop,
          {"process_elapsed_ms": elapsed, "render_loop_wall_ms": render_loop})
    return artifacts, {
        "audit": {
            "passed": True,
            "checks": checks,
            "raw_parser": "validate_render.read_sum",
            "raw_values_finite_nonnegative": True,
        },
        "actual": {
            "resolution": [width, height],
            "target_samples": target,
            "completed_samples": completed,
            "seed": metadata["seed"],
            "sampler": metadata["sampler"],
            "sampler_id": metadata["sampler_id"],
            "sampler_version": metadata["sampler_version"],
            "config": metadata["config"],
            "camera": camera,
            "raw_checksum_fnv1a64": checksum,
            "raw_sha256": raw["sha256"],
            "render_loop_wall_ms": render_loop,
            "renderer_host_ms": host_ms,
            "renderer_gpu_ms": gpu_ms,
            "process_elapsed_ms": elapsed,
            "run_policy": policy,
            "benchmark_csv": benchmark,
            "compatibility": compatibility,
            "fingerprints": fingerprints,
        },
    }


def existing_artifacts(prefix: Path, benchmark: bool,
                       evidence_root: Path) -> dict[str, Any]:
    result = {}
    candidates = artifact_paths(prefix, benchmark)
    if not benchmark:
        candidates["benchmark_csv"] = suffix(prefix, ".benchmark.csv")
    for name, path in candidates.items():
        if path.is_file():
            result[name] = file_record(path, evidence_root)
    return result


def terminate(process: subprocess.Popen[Any]) -> int | None:
    if process.poll() is not None:
        return process.returncode
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait(timeout=5)


class MeasurementSession:
    def __init__(self, output: Path, binary: Path, build: Path, requested: str):
        self.output = output.resolve()
        self.binary = binary.resolve()
        self.build = build.resolve()
        self.records: list[dict[str, Any]] = []
        self.initial_binary: dict[str, Any] = {}
        self.initial_source: dict[str, Any] = {}
        self.initial_driver: dict[str, Any] = {}
        self.initial_scenes: dict[str, dict[str, Any]] = {}
        self.state: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "tool": "scripts/measure_gltf.py",
            "status": "running",
            "phase_requested": requested,
            "started_utc": utc_now(),
            "completed_utc": None,
            "design": design(),
            "identity": {},
            "input_scenes": {},
            "phases": {},
            "run_count": 0,
            "errors": [],
        }

    def persist(self) -> None:
        self.state["run_count"] = len(self.records)
        atomic_json(self.output / "runs.json", {
            "schema_version": SCHEMA_VERSION,
            "runs": self.records,
        })
        atomic_json(self.output / "summary.json", self.state)
        for phase, summary in self.state["phases"].items():
            directory = self.output / phase
            if not directory.is_dir():
                continue
            atomic_json(directory / "runs.json", {
                "schema_version": SCHEMA_VERSION,
                "runs": [item for item in self.records if item["phase"] == phase],
            })
            atomic_json(directory / "summary.json", summary)

    def initialize(self, scene_files: Iterable[str]) -> None:
        if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise EvidenceError(f"renderer binary is absent or not executable: {self.binary}")
        self.initial_binary = stable_file_identity(self.binary)
        self.initial_source = renderer_source_snapshot()
        self.initial_driver = stable_file_identity(Path(__file__))
        self.initial_scenes = {
            filename: scene_identity(filename) for filename in sorted(set(scene_files))
        }
        manifest = ROOT / "scenes/assets/gltf/generation.json"
        self.state["identity"] = {
            "binary": self.initial_binary,
            "renderer_source": self.initial_source,
            "driver": self.initial_driver,
            "build_directory": str(self.build),
            "asset_generation_manifest": (
                stable_file_identity(manifest) if manifest.is_file() else None
            ),
        }
        self.state["input_scenes"] = self.initial_scenes
        self.persist()

    def prepare_phase(self, phase: str) -> None:
        directory = self.output / phase
        directory.mkdir(exist_ok=False)
        (directory / "renders").mkdir()
        if phase == "validation":
            (directory / "checkpoints").mkdir()
        self.state["phases"][phase] = {
            "schema_version": SCHEMA_VERSION,
            "phase": phase,
            "status": "running",
            "started_utc": utc_now(),
            "completed_utc": None,
            "run_count": 0,
        }
        self.persist()

    def run(self, spec: RenderSpec) -> dict[str, Any]:
        prefix = self.output / spec.phase / "renders" / spec.label
        checkpoint = (
            self.output / spec.checkpoint_relative
            if spec.checkpoint_relative is not None else None
        )
        command = render_command(self.binary, prefix, checkpoint, spec)
        stdout_path = suffix(prefix, ".stdout.txt")
        stderr_path = suffix(prefix, ".stderr.txt")
        command_path = suffix(prefix, ".command.json")
        candidates = [stdout_path, stderr_path, command_path]
        candidates.extend(artifact_paths(prefix, spec.benchmark).values())
        if checkpoint is not None and spec.checkpoint_mode == "write":
            candidates.append(checkpoint)
        existing = [str(path) for path in candidates if path.exists()]
        if existing:
            raise EvidenceError(f"{spec.label}: refusing to overwrite {existing}")

        run_id = len(self.records) + 1
        started = utc_now()
        invocation: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "phase": spec.phase,
            "purpose": spec.purpose,
            "label": spec.label,
            "status": "running",
            "design": spec.record(),
            "command": command,
            "cwd": str(ROOT),
            "started_utc": started,
            "completed_utc": None,
            "exit_code": None,
            "process_elapsed_ms": None,
            "pid": None,
        }
        atomic_json(command_path, invocation)

        binary_before = scene_before = resume_before = None
        binary_after = scene_after = resume_after = None
        identity_errors: list[str] = []
        launch_error: Exception | None = None
        interrupted = False
        exit_code: int | None = None
        elapsed_ms: float | None = None
        process: subprocess.Popen[Any] | None = None
        try:
            binary_before = stable_file_identity(self.binary)
            if not same_file_identity(binary_before, self.initial_binary):
                identity_errors.append("binary differs from the session-start identity")
            scene_before = scene_identity(spec.scene_file)
            if scene_before != self.initial_scenes[spec.scene_file]:
                identity_errors.append("scene inputs differ from the session-start identity")
            if spec.checkpoint_mode == "resume":
                assert checkpoint is not None
                resume_before = checkpoint_record(checkpoint, self.output)
        except Exception as error:
            launch_error = error

        invocation["binary_identity_before"] = binary_before
        invocation["scene_identity_before"] = scene_before
        invocation["resume_checkpoint_before"] = resume_before
        atomic_json(command_path, invocation)

        began: float | None = None
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            if launch_error is None and not identity_errors:
                try:
                    began = time.perf_counter()
                    process = subprocess.Popen(
                        command, cwd=ROOT, stdout=stdout, stderr=stderr,
                        start_new_session=True,
                    )
                    invocation["pid"] = process.pid
                    atomic_json(command_path, invocation)
                    exit_code = process.wait()
                except KeyboardInterrupt:
                    interrupted = True
                    if process is not None:
                        exit_code = terminate(process)
                    stderr.write(b"\nmeasurement interrupted; child process terminated\n")
                except Exception as error:
                    launch_error = error
                    if process is not None and process.poll() is None:
                        exit_code = terminate(process)
                    stderr.write((str(error) + "\n").encode("utf-8", "replace"))
                finally:
                    if began is not None:
                        elapsed_ms = (time.perf_counter() - began) * 1000.0
            else:
                message = launch_error or EvidenceError("; ".join(identity_errors))
                stderr.write((str(message) + "\n").encode("utf-8", "replace"))

        try:
            binary_after = stable_file_identity(self.binary)
            if binary_before is None or not same_file_identity(binary_before, binary_after):
                identity_errors.append("binary changed during the invocation")
        except Exception as error:
            identity_errors.append(f"cannot hash binary after invocation: {error}")
        try:
            scene_after = scene_identity(spec.scene_file)
            if scene_before is None or scene_after != scene_before:
                identity_errors.append("scene inputs changed during the invocation")
        except Exception as error:
            identity_errors.append(f"cannot hash scene after invocation: {error}")
        if spec.checkpoint_mode == "resume" and checkpoint is not None:
            try:
                resume_after = checkpoint_record(checkpoint, self.output)
                if resume_after != resume_before:
                    identity_errors.append("resume checkpoint changed during invocation")
            except Exception as error:
                identity_errors.append(f"cannot audit resume checkpoint after invocation: {error}")

        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "phase": spec.phase,
            "purpose": spec.purpose,
            "label": spec.label,
            "status": "failed",
            "design": spec.record(),
            "command": command,
            "cwd": str(ROOT),
            "started_utc": started,
            "completed_utc": utc_now(),
            "exit_code": exit_code,
            "process_elapsed_ms": elapsed_ms,
            "stdout": file_record(stdout_path, self.output),
            "stderr": file_record(stderr_path, self.output),
            "artifacts": existing_artifacts(prefix, spec.benchmark, self.output),
            "binary_identity": {
                "before": binary_before, "after": binary_after,
                "unchanged": binary_before is not None and binary_after is not None
                and same_file_identity(binary_before, binary_after),
            },
            "scene_input_identity": {
                "before": scene_before, "after": scene_after,
                "unchanged": scene_before is not None and scene_after == scene_before,
            },
            "source_input_sha256": self.initial_source.get("sha256"),
        }
        if checkpoint is not None:
            record["checkpoint"] = {
                "mode": spec.checkpoint_mode,
                "path": relative_path(checkpoint, self.output),
                "before": resume_before,
                "after": resume_after,
            }

        if interrupted:
            record["status"] = "interrupted"
            record["failure"] = {"kind": "interrupted", "message": "child terminated"}
        elif launch_error is not None:
            record["failure"] = {
                "kind": "launch", "message": f"{type(launch_error).__name__}: {launch_error}"
            }
        elif identity_errors:
            record["failure"] = {"kind": "identity", "message": "; ".join(identity_errors)}
        elif exit_code != 0:
            record["failure"] = {
                "kind": "renderer-exit", "message": f"renderer exited with {exit_code}"
            }
        else:
            try:
                if elapsed_ms is None:
                    raise EvidenceError("missing process duration")
                artifacts, audit = audit_render(
                    prefix, spec, elapsed_ms, self.output, self.initial_source["sha256"]
                )
                record["artifacts"] = artifacts
                if spec.checkpoint_mode == "write":
                    assert checkpoint is not None
                    record["checkpoint"]["after"] = checkpoint_record(checkpoint, self.output)
                record.update(audit)
                record["status"] = "passed"
            except Exception as error:
                record["audit"] = {
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                }

        invocation.update(record)
        invocation["status"] = record["status"]
        atomic_json(command_path, invocation)
        record["invocation_record"] = file_record(command_path, self.output)
        self.records.append(record)
        phase_state = self.state["phases"][spec.phase]
        phase_state["run_count"] = sum(
            item["phase"] == spec.phase for item in self.records
        )
        phase_state["last_run"] = {
            "run_id": run_id, "label": spec.label, "status": record["status"]
        }
        self.persist()
        print(f"[{run_id}] {spec.phase}/{spec.label}: {record['status']}", flush=True)
        if interrupted:
            raise KeyboardInterrupt
        if record["status"] != "passed":
            raise EvidenceError(f"{spec.label} failed: {record.get('failure', record.get('audit'))}")
        return record

    def finish_identity(self) -> dict[str, Any]:
        binary = stable_file_identity(self.binary)
        source = renderer_source_snapshot()
        driver = stable_file_identity(Path(__file__))
        manifest_start = self.state["identity"].get("asset_generation_manifest")
        manifest_path = ROOT / "scenes/assets/gltf/generation.json"
        manifest_end = stable_file_identity(manifest_path) if manifest_start is not None else None
        if not same_file_identity(binary, self.initial_binary):
            raise EvidenceError("renderer binary changed during the session")
        if source != self.initial_source:
            raise EvidenceError("renderer source inputs changed during the session")
        if not same_file_identity(driver, self.initial_driver):
            raise EvidenceError("measurement driver changed during the session")
        if manifest_end != manifest_start:
            raise EvidenceError("glTF generation manifest changed during the session")
        builds = {
            item["actual"]["compatibility"]["build_identifier"]
            for item in self.records if item.get("status") == "passed"
        }
        sources = {
            item["actual"]["compatibility"]["baseline_starter_revision"]
            for item in self.records if item.get("status") == "passed"
        }
        devices = {
            item["actual"]["compatibility"]["device"]
            for item in self.records if item.get("status") == "passed"
        }
        if len(builds) != 1 or len(sources) != 1 or len(devices) != 1:
            raise EvidenceError("renderer metadata identity changed between invocations")
        return {
            "binary_end": binary,
            "renderer_source_end": source,
            "driver_end": driver,
            "asset_generation_manifest_end": manifest_end,
            "binary_unchanged": True,
            "renderer_source_unchanged": True,
            "driver_unchanged": True,
            "asset_generation_manifest_unchanged": True,
            "metadata_build_identifiers": sorted(builds),
            "metadata_starter_revisions": sorted(sources),
            "metadata_devices": sorted(devices),
            "metadata_build_matches_source_hash": all(
                self.initial_source["sha256"] in value for value in builds
            ),
        }


def artifact_path(record: dict[str, Any], name: str, root: Path) -> Path:
    artifact = record.get("artifacts", {}).get(name)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
        raise EvidenceError(f"{record.get('label')}: missing {name} artifact record")
    path = Path(artifact["path"])
    return path if path.is_absolute() else root / path


def byte_exact(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as first, right.open("rb") as second:
        while True:
            a = first.read(1024 * 1024)
            b = second.read(1024 * 1024)
            if a != b:
                return False
            if not a:
                return True


def exact_artifacts(records: Sequence[dict[str, Any]], names: Sequence[str],
                    root: Path, label: str) -> dict[str, Any]:
    if not records:
        raise EvidenceError(f"{label}: cannot compare an empty record set")
    result = {"name": label, "passed": True, "run_ids": [item["run_id"] for item in records]}
    reference = records[0]
    for name in names:
        first = artifact_path(reference, name, root)
        for candidate in records[1:]:
            other = artifact_path(candidate, name, root)
            if not byte_exact(first, other):
                raise EvidenceError(
                    f"{label}: {name} differs between {reference['label']} and {candidate['label']}"
                )
        result[name + "_sha256"] = sha256_file(first)
        result[name + "_bytes"] = first.stat().st_size
    return result


def physical_setting_differences(reference: dict[str, Any],
                                 resumed: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {"reference": reference.get(name), "resume": resumed.get(name)}
        for name in RESUME_PHYSICAL_FIELDS
        if reference.get(name) != resumed.get(name)
    }


def describe(values: Sequence[float]) -> dict[str, Any]:
    converted = [finite_nonnegative(value, "timing value") for value in values]
    if len(converted) != 3:
        raise EvidenceError(f"expected exactly three measured values, got {len(converted)}")
    return {
        "values": converted,
        "minimum": min(converted),
        "median": statistics.median(converted),
        "maximum": max(converted),
    }


def analyze_validation(records: Sequence[dict[str, Any]], root: Path) -> dict[str, Any]:
    if len(records) != 18:
        raise EvidenceError(f"validation produced {len(records)} runs, expected 18")
    encoding = [item for item in records if item["purpose"] == "encoding-equivalence"]
    if len(encoding) != 6 or any(
            item["design"]["sampler"] != DEFAULT_SAMPLER for item in encoding):
        raise EvidenceError("encoding validation must contain six Counter runs")

    def motion_index(purpose: str) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for item in records:
            if item["purpose"] != purpose:
                continue
            key = item["design"].get("motion_group")
            if not isinstance(key, str) or key in indexed:
                raise EvidenceError(f"{purpose} has a missing or duplicate motion group")
            indexed[key] = item
        return indexed

    motion_reference = motion_index("motion-reference")
    motion_partial = motion_index("motion-partial")
    motion_resume = motion_index("motion-resume")
    expected_groups = {
        motion_group_key(sampler, bvh)
        for sampler in SAMPLER_IDS for bvh in (False, True)
    }
    if set(motion_reference) != expected_groups \
            or set(motion_partial) != expected_groups \
            or set(motion_resume) != expected_groups:
        raise EvidenceError("validation records do not match the fixed matrix")

    checks = [exact_artifacts(
        encoding, ("raw", "png"), root,
        "three encodings and both BVH modes are byte-exact",
    )]
    group_summaries = []
    for sampler in SAMPLER_IDS:
        for bvh in (False, True):
            group = motion_group_key(sampler, bvh)
            reference = motion_reference[group]
            partial = motion_partial[group]
            resumed = motion_resume[group]
            checks.append(exact_artifacts(
                (reference, resumed), ("raw", "png"), root,
                f"motion {group} fresh/resume is byte-exact",
            ))
            differences = physical_setting_differences(
                reference["actual"], resumed["actual"]
            )
            if differences:
                raise EvidenceError(
                    f"motion resume {group} did not inherit physical settings: {differences}"
                )
            options = resumed["command"][2:]
            expected = [
                "--samples", "17", "--output",
                str((root / "validation/renders" / resumed["label"]).resolve()),
                "--resume",
                str((root / resumed["design"]["checkpoint_relative"]).resolve()),
            ]
            if options != expected:
                raise EvidenceError(f"motion resume {group} is not minimal: {options}")
            checkpoint = partial.get("checkpoint", {}).get("after")
            if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 3:
                raise EvidenceError(f"motion partial {group} lacks a valid v3 checkpoint")
            if checkpoint != resumed.get("checkpoint", {}).get("before"):
                raise EvidenceError(
                    f"motion {group} resume input is not the recorded partial checkpoint"
                )
            group_summaries.append({
                "group_key": group,
                "sampler": sampler,
                "sampler_id": sampler_id_for(sampler),
                "bvh": bvh,
                "reference_run_id": reference["run_id"],
                "partial_run_id": partial["run_id"],
                "resume_run_id": resumed["run_id"],
                "checkpoint": resumed["design"]["checkpoint_relative"],
                "inherited_physical_fields": list(RESUME_PHYSICAL_FIELDS),
            })
        checks.append(exact_artifacts(
            tuple(motion_reference[motion_group_key(sampler, bvh)]
                  for bvh in (False, True)),
            ("raw", "png"), root,
            f"motion {sampler} BVH reference modes are byte-exact",
        ))
    return {
        "status": "completed",
        "completed_utc": utc_now(),
        "run_count": len(records),
        "checks": checks,
        "motion_groups": group_summaries,
        "counter_halton_outputs_compared": False,
        "encoding_metadata_fingerprints_compared": False,
        "motion_resume_physical_metadata_compared": list(RESUME_PHYSICAL_FIELDS),
        "metadata_note": (
            "Encoding-specific fingerprints may differ and are not compared across encodings. "
            "Within each motion fresh/resume group, fingerprints and compatibility are exact."
        ),
        "minimal_resume_benchmark_csv_expected": False,
    }


def analyze_timing(records: Sequence[dict[str, Any]], root: Path) -> dict[str, Any]:
    if len(records) != 36:
        raise EvidenceError(f"timing produced {len(records)} runs, expected 36")
    groups = []
    exactness = []
    for level in GLTF_COMPLEXITY:
        level_records = [item for item in records if item["design"]["complexity"] == level]
        if len(level_records) != 12:
            raise EvidenceError(f"timing {level} has {len(level_records)} runs, expected 12")
        exactness.append(exact_artifacts(
            level_records, ("raw", "png"), root,
            f"timing {level}: all repeats and BVH modes are byte-exact",
        ))
        for bvh in (False, True):
            group = [item for item in level_records if item["design"]["bvh"] is bvh]
            warmups = sorted(
                (item for item in group if item["design"]["measurement_phase"] == "warmup"),
                key=lambda item: item["design"]["repeat"],
            )
            measured = sorted(
                (item for item in group if item["design"]["measurement_phase"] == "measured"),
                key=lambda item: item["design"]["repeat"],
            )
            if len(warmups) != 3 or len(measured) != 3:
                raise EvidenceError(f"timing {level}/bvh-{bvh} repeat count is wrong")
            metrics = {
                name: describe([item["actual"][name] for item in measured])
                for name in (
                    "renderer_gpu_ms", "renderer_host_ms",
                    "render_loop_wall_ms", "process_elapsed_ms",
                )
            }
            stage_metrics = {}
            for category in (
                "gpu_stage_sums_ms", "host_stage_sums_ms",
                "wavefront_shader_breakdown_ms",
            ):
                names = measured[0]["actual"]["benchmark_csv"][category]
                stage_metrics[category] = {
                    name: describe([
                        item["actual"]["benchmark_csv"][category][name]
                        for item in measured
                    ])
                    for name in names
                }
            groups.append({
                "complexity": level,
                "bvh": bvh,
                "warmup_run_ids": [item["run_id"] for item in warmups],
                "measured_run_ids": [item["run_id"] for item in measured],
                "warmups_excluded_from_statistics": True,
                "summary_statistic": "median, minimum, and maximum of exactly three measured fresh processes",
                "metrics_ms": metrics,
                "benchmark_stage_metrics_ms": stage_metrics,
                "stage_semantics": measured[0]["actual"]["benchmark_csv"]["semantics"],
            })
    return {
        "status": "renderer-completed",
        "completed_utc": utc_now(),
        "run_count": len(records),
        "groups": groups,
        "exactness": exactness,
    }


def analyze_gallery(records: Sequence[dict[str, Any]], root: Path) -> dict[str, Any]:
    if len(records) != 8:
        raise EvidenceError(f"gallery produced {len(records)} runs, expected 8")
    checks = []
    for level in GLTF_COMPLEXITY:
        pair = [item for item in records if item["design"].get("complexity") == level]
        if len(pair) != 2:
            raise EvidenceError(f"gallery {level} has {len(pair)} BVH renders")
        checks.append(exact_artifacts(
            pair, ("raw", "png"), root,
            f"gallery {level} BVH modes are byte-exact",
        ))
    capability = [item for item in records if item["purpose"] == "capability"]
    if {item["label"] for item in capability} != {"showcase", "baseline"}:
        raise EvidenceError("gallery capability pair is incomplete")
    return {
        "status": "completed",
        "completed_utc": utc_now(),
        "run_count": len(records),
        "checks": checks,
        "capability_run_ids": [item["run_id"] for item in capability],
        "showcase_baseline_exactness": "not asserted: baseline intentionally omits imported objects",
    }


def logged_process(command: Sequence[str], directory: Path, name: str) -> dict[str, Any]:
    stdout_path = directory / f"{name}.stdout.txt"
    stderr_path = directory / f"{name}.stderr.txt"
    command_path = directory / f"{name}.command.json"
    for path in (stdout_path, stderr_path, command_path):
        if path.exists():
            raise EvidenceError(f"refusing to overwrite command evidence: {path}")
    record: dict[str, Any] = {
        "name": name,
        "status": "running",
        "command": list(command),
        "cwd": str(ROOT),
        "started_utc": utc_now(),
        "completed_utc": None,
        "exit_code": None,
        "process_elapsed_ms": None,
        "pid": None,
    }
    atomic_json(command_path, record)
    process: subprocess.Popen[Any] | None = None
    began: float | None = None
    error: Exception | None = None
    interrupted = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            began = time.perf_counter()
            process = subprocess.Popen(
                list(command), cwd=ROOT, stdout=stdout, stderr=stderr,
                start_new_session=True,
            )
            record["pid"] = process.pid
            atomic_json(command_path, record)
            record["exit_code"] = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            if process is not None:
                record["exit_code"] = terminate(process)
            stderr.write(b"\nmeasurement interrupted; child process terminated\n")
        except Exception as caught:
            error = caught
            if process is not None and process.poll() is None:
                record["exit_code"] = terminate(process)
            stderr.write((str(caught) + "\n").encode("utf-8", "replace"))
        finally:
            if began is not None:
                record["process_elapsed_ms"] = (time.perf_counter() - began) * 1000.0
    record["completed_utc"] = utc_now()
    record["stdout"] = file_record(stdout_path, directory)
    record["stderr"] = file_record(stderr_path, directory)
    if interrupted:
        record["status"] = "interrupted"
        record["failure"] = "measurement interrupted"
    elif error is not None:
        record["status"] = "failed"
        record["failure"] = f"{type(error).__name__}: {error}"
    elif record["exit_code"] == 0:
        record["status"] = "passed"
    else:
        record["status"] = "failed"
        record["failure"] = f"process exited with {record['exit_code']}"
    atomic_json(command_path, record)
    record["command_record"] = file_record(command_path, directory)
    if interrupted:
        raise KeyboardInterrupt
    return record


def validate_profile_records(records: Any, expected_scenes: Sequence[Path]) -> list[dict[str, Any]]:
    if not isinstance(records, list) or len(records) != len(expected_scenes):
        raise EvidenceError("CPU profile did not return exactly six scene records")
    by_scene: dict[Path, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("scene"), str):
            raise EvidenceError("CPU profile returned a malformed record")
        path = Path(record["scene"]).resolve()
        if path in by_scene:
            raise EvidenceError(f"CPU profile duplicated scene: {path}")
        for name in ("primitives", "triangles", "bvh_nodes"):
            value = record.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise EvidenceError(f"CPU profile {path}: invalid {name}: {value!r}")
        for name in ("load_including_bvh_ms", "bvh_rebuild_ms"):
            values = record.get(name)
            if not isinstance(values, list) or len(values) != 3:
                raise EvidenceError(f"CPU profile {path}: {name} must have three repeats")
            converted = [finite_nonnegative(value, f"CPU profile {path} {name}")
                         for value in values]
            record[name] = converted
            median_name = name.removesuffix("_ms") + "_median_ms"
            median = finite_nonnegative(record.get(median_name), median_name)
            if not math.isclose(median, statistics.median(converted),
                                rel_tol=1e-12, abs_tol=1e-12):
                raise EvidenceError(f"CPU profile {path}: inconsistent {median_name}")
        by_scene[path] = record
    expected = {path.resolve() for path in expected_scenes}
    if set(by_scene) != expected:
        raise EvidenceError(
            f"CPU profile scene set differs: expected {sorted(map(str, expected))}, "
            f"got {sorted(map(str, by_scene))}"
        )
    return [by_scene[path.resolve()] for path in expected_scenes]


def run_cpu_profile(session: MeasurementSession) -> dict[str, Any]:
    directory = session.output / "timing/cpu-profile"
    directory.mkdir()
    summary_path = directory / "summary.json"
    source = ROOT / "scripts/profile_scene_load.cpp"
    archive = session.build / "lib/libcis565_core.a"
    executable = directory / "private-profile-scene-load"
    scene_paths = [
        *(ROOT / "scenes" / GLTF_COMPLEXITY[level] for level in GLTF_COMPLEXITY),
        *(ROOT / "scenes" / OBJ_COMPLEXITY[level] for level in OBJ_COMPLEXITY),
    ]
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    commands: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_utc": utc_now(),
        "completed_utc": None,
        "warmups": 1,
        "measured_repeats": 3,
        "units": "milliseconds",
        "helper_source_path": str(source),
        "core_archive_path": str(archive),
        "scene_paths": [str(path) for path in scene_paths],
        "commands": commands,
        "private_platform_executable": {
            "path": relative_path(executable, session.output),
            "retained": False,
            "deleted": False,
        },
        "note": (
            "CPU scene load includes parsing, transforms, camera setup, and initial BVH. "
            "The separate BVH measurement rebuilds from already loaded primitives. "
            "These are distinct operations; no subtraction or GPU-work claim is made. "
            "The helper initializes no GPU. One warmup precedes three measured repeats."
        ),
    }
    atomic_json(summary_path, summary)
    try:
        if not compiler:
            raise EvidenceError("CXX is empty")
        summary.update({
            "helper_source": stable_file_identity(source),
            "core_archive_before": stable_file_identity(archive),
            "scene_inputs_before": {
                path.name: scene_identity(path.name) for path in scene_paths
            },
        })
        atomic_json(summary_path, summary)
        version = logged_process([*compiler, "--version"], directory, "compiler-version")
        commands.append(version)
        atomic_json(summary_path, summary)
        if version["status"] != "passed":
            raise EvidenceError(f"compiler version command failed: {version.get('failure')}")

        compile_command = [
            *compiler, "-std=c++17", "-O3", "-DNDEBUG",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "external/include"),
            str(source), str(archive), "-o", str(executable),
        ]
        compile_record = logged_process(compile_command, directory, "compile")
        commands.append(compile_record)
        atomic_json(summary_path, summary)
        if compile_record["status"] != "passed":
            raise EvidenceError(f"CPU profile compilation failed: {compile_record.get('failure')}")
        summary["private_platform_executable"]["identity_before_measurement"] = \
            stable_file_identity(executable)

        measure = logged_process(
            [str(executable), *map(str, scene_paths)], directory, "measure"
        )
        commands.append(measure)
        atomic_json(summary_path, summary)
        if measure["status"] != "passed":
            raise EvidenceError(f"CPU profile helper failed: {measure.get('failure')}")
        parsed = json.loads((directory / "measure.stdout.txt").read_text(encoding="utf-8"))
        records = validate_profile_records(parsed, scene_paths)
        atomic_json(directory / "records.json", records)

        indexed = {Path(item["scene"]).resolve(): item for item in records}
        comparisons = []
        for level in GLTF_COMPLEXITY:
            gltf = indexed[(ROOT / "scenes" / GLTF_COMPLEXITY[level]).resolve()]
            obj = indexed[(ROOT / "scenes" / OBJ_COMPLEXITY[level]).resolve()]
            counts = ("primitives", "triangles", "bvh_nodes")
            if any(gltf[name] != obj[name] for name in counts):
                raise EvidenceError(
                    f"CPU profile {level}: glTF/OBJ workload counts differ: "
                    f"{ {name: [gltf[name], obj[name]] for name in counts} }"
                )
            comparison = {
                "complexity": level,
                "gltf_scene": gltf["scene"],
                "obj_scene": obj["scene"],
                "matched_counts": {name: gltf[name] for name in counts},
                "ratios": {},
            }
            for name in ("load_including_bvh_median_ms", "bvh_rebuild_median_ms"):
                denominator = float(obj[name])
                comparison["ratios"]["gltf_over_obj_" + name] = (
                    float(gltf[name]) / denominator if denominator > 0.0 else None
                )
            comparisons.append(comparison)
        summary.update({
            "status": "completed",
            "completed_utc": utc_now(),
            "records_artifact": file_record(directory / "records.json", session.output),
            "comparisons": comparisons,
            "helper_source_after": stable_file_identity(source),
            "core_archive_after": stable_file_identity(archive),
            "scene_inputs_after": {
                path.name: scene_identity(path.name) for path in scene_paths
            },
        })
        if not same_file_identity(summary["core_archive_before"], summary["core_archive_after"]):
            raise EvidenceError("core archive changed during CPU profiling")
        if not same_file_identity(summary["helper_source"], summary["helper_source_after"]):
            raise EvidenceError("CPU profile helper source changed during profiling")
        if summary["scene_inputs_before"] != summary["scene_inputs_after"]:
            raise EvidenceError("profile scene inputs changed during CPU profiling")
    except BaseException as error:
        summary["status"] = "failed"
        summary["completed_utc"] = utc_now()
        summary["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if executable.exists():
            try:
                executable.unlink()
            except OSError as deletion_error:
                summary["private_platform_executable"]["deletion_error"] = str(deletion_error)
        summary["private_platform_executable"]["deleted"] = not executable.exists()
        summary["private_platform_executable"]["retained"] = executable.exists()
        atomic_json(summary_path, summary)
    return summary


def selected_phases(requested: str) -> tuple[str, ...]:
    return ("validation", "timing", "gallery") if requested == "all" else (requested,)


def scene_files_for(requested: str) -> set[str]:
    result: set[str] = set()
    phases = selected_phases(requested)
    if "validation" in phases:
        result.update(ENCODING_SCENES.values())
        result.add(MOTION_SCENE)
    if "timing" in phases:
        result.update(GLTF_COMPLEXITY.values())
        result.update(OBJ_COMPLEXITY.values())
    if "gallery" in phases:
        result.update(GALLERY_SCENES.values())
        result.update(GLTF_COMPLEXITY.values())
    return result


def run_validation_phase(session: MeasurementSession) -> None:
    session.prepare_phase("validation")
    records = [session.run(spec) for spec in validation_specs()]
    session.state["phases"]["validation"].update(
        analyze_validation(records, session.output)
    )
    session.persist()


def run_timing_phase(session: MeasurementSession) -> None:
    session.prepare_phase("timing")
    records = [session.run(spec) for spec in timing_specs()]
    summary = analyze_timing(records, session.output)
    session.state["phases"]["timing"].update(summary)
    session.state["phases"]["timing"]["cpu_profile"] = {
        "status": "running", "summary": "cpu-profile/summary.json"
    }
    session.persist()
    try:
        profile = run_cpu_profile(session)
    except BaseException:
        profile_path = session.output / "timing/cpu-profile/summary.json"
        session.state["phases"]["timing"]["cpu_profile"] = {
            "status": "failed",
            "summary": "cpu-profile/summary.json",
            "summary_artifact": (
                file_record(profile_path, session.output) if profile_path.is_file() else None
            ),
        }
        session.persist()
        raise
    session.state["phases"]["timing"]["cpu_profile"] = {
        "status": profile["status"],
        "summary": "cpu-profile/summary.json",
        "summary_artifact": file_record(
            session.output / "timing/cpu-profile/summary.json", session.output
        ),
    }
    session.state["phases"]["timing"]["status"] = "completed"
    session.state["phases"]["timing"]["completed_utc"] = utc_now()
    session.persist()


def run_gallery_phase(session: MeasurementSession) -> None:
    session.prepare_phase("gallery")
    records = [session.run(spec) for spec in gallery_specs()]
    session.state["phases"]["gallery"].update(
        analyze_gallery(records, session.output)
    )
    session.persist()


def run_measurement(args: argparse.Namespace, output: Path) -> int:
    session = MeasurementSession(output, args.binary, args.build, args.phase)
    session.persist()
    try:
        session.initialize(scene_files_for(args.phase))
        for phase in selected_phases(args.phase):
            if phase == "validation":
                run_validation_phase(session)
            elif phase == "timing":
                run_timing_phase(session)
            else:
                run_gallery_phase(session)
        session.state["identity"]["final"] = session.finish_identity()
        session.state["status"] = "completed"
        session.state["completed_utc"] = utc_now()
        session.persist()
        print(output)
        return 0
    except BaseException as error:
        session.state["status"] = "failed"
        session.state["completed_utc"] = utc_now()
        for phase in session.state["phases"].values():
            if phase.get("status") in ("running", "renderer-completed"):
                phase["status"] = "failed"
                phase["completed_utc"] = utc_now()
                phase["error"] = f"{type(error).__name__}: {error}"
        session.state["errors"].append({
            "time_utc": utc_now(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        })
        try:
            session.persist()
        except Exception as persist_error:
            print(f"also failed to persist state: {persist_error}", file=sys.stderr)
        if isinstance(error, KeyboardInterrupt):
            print("measurement interrupted; partial evidence was retained", file=sys.stderr)
            return 130
        print(f"measurement failed: {error}", file=sys.stderr)
        return 1


def run_self_test(output: Path) -> int:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        if not condition:
            raise AssertionError(f"{name}: {detail!r}")
        checks.append({"name": name, "passed": True, "detail": detail})

    validation = validation_specs()
    timing = timing_specs()
    gallery = gallery_specs()
    check("validation-count", len(validation) == 18, len(validation))
    check("timing-count", len(timing) == 36, len(timing))
    check("gallery-count", len(gallery) == 8, len(gallery))
    every = validation + timing + gallery
    check("all-count", len(every) == 62, len(every))
    check("unique-phase-labels", len({(item.phase, item.label) for item in every}) == 62)
    check("encoding-matrix", {
        (item.tags["encoding"], item.bvh, item.sampler)
        for item in validation if item.purpose == "encoding-equivalence"
    } == {(encoding, bvh, DEFAULT_SAMPLER)
          for encoding in ENCODING_SCENES for bvh in (False, True)})
    motion = [item for item in validation if item.purpose.startswith("motion-")]
    expected_motion = {
        (sampler, bvh, purpose, samples, checkpoint_mode)
        for sampler in SAMPLER_IDS
        for bvh in (False, True)
        for purpose, samples, checkpoint_mode in (
            ("motion-reference", 17, None),
            ("motion-partial", 5, "write"),
            ("motion-resume", 17, "resume"),
        )
    }
    check("motion-matrix", {
        (item.sampler, item.bvh, item.purpose, item.samples, item.checkpoint_mode)
        for item in motion
    } == expected_motion)
    check("motion-group-keys", {
        item.tags.get("motion_group") for item in motion
    } == {motion_group_key(sampler, bvh)
          for sampler in SAMPLER_IDS for bvh in (False, True)})
    check("motion-checkpoint-count", len({
        item.checkpoint_relative for item in motion if item.checkpoint_mode == "write"
    }) == 4)
    check("validation-settings", all(
        item.resolution == (17, 13) and item.depth == 7
        and item.samples in (5, 17) and not item.rr for item in validation
    ))
    check("timing-settings", all(
        item.samples == 8 and item.resolution == (96, 64) and item.depth == 10
        and not item.rr and item.sampler == DEFAULT_SAMPLER for item in timing
    ))
    check("gallery-settings", all(
        (item.samples, item.resolution) == (
            (16, (128, 84)) if item.purpose == "culling" else (128, (512, 336))
        ) and item.depth == 10 and item.sampler == DEFAULT_SAMPLER for item in gallery
    ))

    regular = validation[0]
    command = render_command(Path("/synthetic/headless"), output / "regular", None, regular)
    check("no-validate-flag", "--validate" not in command)
    check("benchmark-flag", "--benchmark" in command)
    expected_options = {
        "--samples": "17", "--seed": "947", "--resolution": "17x13",
        "--depth": "7", "--wavefront": "off", "--rr": "off",
        "--rr-start": "3", "--compact": "on", "--bvh": "off",
        "--aa": "on", "--nee": "on", "--sort": "off", "--motion": "on",
        "--sampler": "counter",
    }
    for option, expected in expected_options.items():
        actual = command[command.index(option) + 1]
        check("command-" + option[2:], actual == expected,
              {"actual": actual, "expected": expected})

    for sampler in SAMPLER_IDS:
        fixture = next(item for item in motion
                       if item.purpose == "motion-reference"
                       and item.sampler == sampler and not item.bvh)
        fixture_command = render_command(
            Path("/synthetic/headless"), output / f"regular-{sampler}", None, fixture
        )
        check(f"{sampler}-command-sampler",
              fixture_command[fixture_command.index("--sampler") + 1] == sampler,
              fixture_command)
        check(f"{sampler}-command-rr-off",
              fixture_command[fixture_command.index("--rr") + 1] == "off",
              fixture_command)
        check(f"{sampler}-command-rr-start",
              fixture_command[fixture_command.index("--rr-start") + 1] == "3",
              fixture_command)

    resume_specs = [item for item in motion if item.checkpoint_mode == "resume"]
    check("resume-matrix-count", len(resume_specs) == 4, len(resume_specs))
    for resume_spec in resume_specs:
        checkpoint = output / resume_spec.checkpoint_relative
        resume_prefix = output / f"resume-{resume_spec.tags['motion_group']}"
        resume = render_command(
            Path("/synthetic/headless"), resume_prefix, checkpoint, resume_spec
        )
        expected_resume = [
            str((ROOT / "scenes" / resume_spec.scene_file).resolve()),
            "--samples", "17", "--output", str(resume_prefix.resolve()),
            "--resume", str(checkpoint.resolve()),
        ]
        check(f"minimal-resume-{resume_spec.tags['motion_group']}",
              resume[1:] == expected_resume, resume)
        check(f"resume-no-benchmark-{resume_spec.tags['motion_group']}",
              "--benchmark" not in resume, resume)
    check("motion-rr-off", all(not item.rr for item in motion))
    check("sampler-ids", {name: sampler_id_for(name) for name in SAMPLER_IDS}
          == {"counter": 2, "halton": 3})
    try:
        sampler_id_for("Counter")
    except EvidenceError:
        rejected_sampler = True
    else:
        rejected_sampler = False
    check("unsupported-sampler-rejected", rejected_sampler)
    inherited_fixture = {
        name: {"value": index} for index, name in enumerate(RESUME_PHYSICAL_FIELDS)
    }
    check("resume-physical-fields-equal",
          not physical_setting_differences(inherited_fixture, dict(inherited_fixture)))
    changed_fixture = dict(inherited_fixture)
    changed_fixture["camera"] = {"value": "changed"}
    changed_fixture["fingerprints"] = {"value": "changed"}
    check("resume-physical-fields-differences",
          set(physical_setting_differences(inherited_fixture, changed_fixture))
          == {"camera", "fingerprints"})
    check("timing-repeat-design", sum(item.measurement == "warmup" for item in timing) == 18
          and sum(item.measurement == "measured" for item in timing) == 18)
    check("gallery-depths", all(item.depth == 10 for item in gallery))

    raw_path = output / "synthetic.sum.bin"
    width, height, samples = 2, 1, 4
    values = (0.0, 4.0, 8.0, 12.0, 16.0, 20.0)
    encoded = bytearray(struct.pack("<8sIIIIQ", b"C565SUM\0", 1,
                                    width, height, 3, samples))
    encoded.extend(struct.pack("<6f", *values))
    encoded.extend(struct.pack("<Q", fnv1a(encoded)))
    raw_path.write_bytes(encoded)
    parsed = read_sum(raw_path)
    check("authoritative-raw-parser", parsed["samples"] == samples
          and parsed["width"] == width and parsed["height"] == height)

    benchmark_path = output / "synthetic.benchmark.csv"
    fields = [
        "record_type", "sample", "bounce", "active_before", "active_after",
        "intersection_ms", "sort_ms", "shade_ms", "compaction_ms", "count_ms",
        "host_ms", "gpu_ms", "measured_samples", "render_loop_wall_ms",
        "queue_ms", "queue_host_ms", "diffuse_ms", "mirror_ms", "dielectric_ms",
        "emission_ms", "terminal_ms",
    ]
    with benchmark_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample in range(2):
            row = {name: "0" for name in fields}
            row.update(record_type="bounce", sample=str(sample), bounce="0",
                       active_before="2", active_after="0")
            writer.writerow(row)
        row = {name: "" for name in fields}
        row.update(record_type="overall", host_ms="1", gpu_ms="2",
                   measured_samples="2", render_loop_wall_ms="3")
        writer.writerow(row)
    benchmark = parse_benchmark(benchmark_path, 2, 7)
    check("benchmark-parser", benchmark["bounce_records"] == 2
          and benchmark["measured_samples"] == 2, benchmark)

    checkpoint_path = output / "synthetic.checkpoint"
    header = struct.pack("<8sIIQ", b"C565CKPT", 3, 0x01020304, 0)
    checkpoint_path.write_bytes(header + struct.pack("<Q", fnv1a(header)))
    checkpoint_info = checkpoint_record(checkpoint_path, output)
    check("checkpoint-v3-parser", checkpoint_info["format_version"] == 3,
          checkpoint_info)
    rejected_path = output / "synthetic-v2.checkpoint"
    rejected_header = struct.pack("<8sIIQ", b"C565CKPT", 2, 0x01020304, 0)
    rejected_path.write_bytes(rejected_header + struct.pack("<Q", fnv1a(rejected_header)))
    try:
        checkpoint_record(rejected_path, output)
    except EvidenceError:
        rejected_checkpoint = True
    else:
        rejected_checkpoint = False
    check("checkpoint-non-v3-rejected", rejected_checkpoint)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "tool": "scripts/measure_gltf.py",
        "mode": "self-test",
        "status": "passed",
        "completed_utc": utc_now(),
        "renderer_executed": False,
        "compiler_executed": False,
        "checks": checks,
        "design": design(),
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", type=Path, default=DEFAULT_BINARY,
        help=f"headless renderer (default: {DEFAULT_BINARY})",
    )
    parser.add_argument(
        "--build", type=Path, default=DEFAULT_BUILD,
        help=f"build tree containing lib/libcis565_core.a (default: {DEFAULT_BUILD})",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="brand-new evidence directory (required; existing paths are rejected)",
    )
    parser.add_argument(
        "--phase", choices=("validation", "timing", "gallery", "all"), default="all",
        help="fixed subdesign to execute (default: all)",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="test design, commands, and parsers without executing renderer or compiler",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not __debug__:
        parser.error("optimized Python is unsupported because validate_render.read_sum uses assertions")
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"--output must be a brand-new path: {output}")
    except OSError as error:
        parser.error(f"cannot create --output directory {output}: {error}")
    if args.self_test:
        return run_self_test(output)
    return run_measurement(args, output)


if __name__ == "__main__":
    raise SystemExit(main())
