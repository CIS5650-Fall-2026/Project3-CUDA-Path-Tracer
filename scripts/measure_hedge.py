#!/usr/bin/env python3
"""Run and analyze the fixed hedge sampling, timing, and gallery designs.

This driver intentionally exposes no knobs for seeds, sample counts, time
budgets, or scheduling.  Those values are part of the experimental design.
Every renderer invocation is retained as independently auditable evidence.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass, field
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import struct
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Sequence

try:
    # Running ``python3 scripts/measure_hedge.py`` puts scripts/ on sys.path.
    from validate_render import fnv1a, read_sum
except ImportError:  # pragma: no cover - supports module-style invocation.
    from scripts.validate_render import fnv1a, read_sum


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "build/bin/cis565_path_tracer_headless"

QUALITY_SCENES = (
    ("diffuse", "hedge-sampling-diffuse.json"),
    ("glass-dof", "hedge-sampling-glass-dof.json"),
)
MOTION_SCENES = {
    "start": "hedge-motion-frozen-start.json",
    "midpoint": "hedge-motion-frozen-midpoint.json",
    "end": "hedge-motion-frozen-end.json",
    "full": "hedge-motion.json",
}
MOTION_SHUTTERS = {
    "start": (0.0, 0.0),
    "midpoint": (0.5, 0.5),
    "end": (1.0, 1.0),
    "full": (0.0, 1.0),
}
EXPECTED_SCENE_SHUTTERS = {
    **{filename: MOTION_SHUTTERS[role] for role, filename in MOTION_SCENES.items()},
    **{filename: (0.0, 1.0) for _, filename in QUALITY_SCENES},
    "hedge-obj-baseline.json": (0.0, 1.0),
    "procedural-optical-stilllife.json": (0.0, 1.0),
    "complexity-medium.json": (0.0, 1.0),
    "optical-workbench.json": (0.0, 1.0),
}

QUALITY_SEEDS = (947, 104729, 130363, 155921, 196613, 262147, 314159, 524287)
REFERENCE_SEEDS = (2718281, 3141593, 1618033, 1414213)
QUALITY_SAMPLE_COUNTS = (16, 64, 256)
SAMPLERS = ("counter", "halton")
SAMPLER_IDS = {"counter": 2, "halton": 3}

QUALITY_RESOLUTION = (128, 84)
QUALITY_DEPTH = 10
REFERENCE_SAMPLES = 8192
REFERENCE_RETRY_SAMPLES = 16384
REFERENCE_PRECISION_FRACTION = 0.25
EQUAL_TIME_BUDGET_MS = 4000
EQUAL_TIME_TARGET_SAMPLES = 65536

TIMING_SAMPLES = 256
TIMING_SEED = 947
TIMING_WARMUPS = 3
TIMING_MEASURED = 3

GALLERY_STATIC_RESOLUTION = (512, 336)
GALLERY_STATIC_SAMPLES = 64
GALLERY_MOTION_SAMPLES = 512
GALLERY_SEED = 947
GALLERY_COMPARISON_RESOLUTION = (512, 336)
GALLERY_COMPARISON_SAMPLES = 128
GALLERY_BVH_RESOLUTION = (192, 128)
GALLERY_BVH_SAMPLES = 32

COMMON_SCHEDULING = {
    "wavefront": "on",
    "rr": "on",
    "rr-start": "3",
    "compact": "on",
    "bvh": "on",
    "aa": "on",
    "nee": "on",
    "sort": "off",
    "motion": "on",
}

SCHEMA_VERSION = 1


class EvidenceError(RuntimeError):
    """Raised when execution or retained evidence violates the fixed design."""


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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def files_byte_exact(left: Path, right: Path) -> bool:
    """Compare retained artifacts byte-for-byte without loading them whole."""
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        while True:
            left_block = left_stream.read(1024 * 1024)
            right_block = right_stream.read(1024 * 1024)
            if left_block != right_block:
                return False
            if not left_block:
                return True


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256_bytes(encoded)


def binary_file_identity(path: Path) -> dict[str, Any]:
    """Hash a binary with stable before/after stat data around the read."""
    path = path.resolve()
    if not path.is_file():
        raise EvidenceError(f"renderer binary does not exist: {path}")
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise EvidenceError(f"renderer binary changed while it was being hashed: {path}")
    return {
        "path": str(path),
        "sha256": digest,
        "bytes": after.st_size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
    }


def same_binary_identity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    fields = ("path", "sha256", "bytes", "device", "inode", "mtime_ns")
    return all(left.get(field) == right.get(field) for field in fields)


def atomic_json(path: Path, value: Any) -> None:
    """Replace a JSON record atomically within its already-created directory."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def relative_to(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def require_nonnegative_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise EvidenceError(f"{label} is not finite and nonnegative: {value!r}")
    return result


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceError(f"{label} is not an integer: {value!r}")
    return value


def recursive_floats_are_finite(value: Any, label: str = "metadata") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvidenceError(f"{label} contains a non-finite float")
    elif isinstance(value, dict):
        for key, child in value.items():
            recursive_floats_are_finite(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            recursive_floats_are_finite(child, f"{label}[{index}]")


def file_record(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceError(f"expected artifact is missing: {path}")
    return {
        "path": relative_to(path, root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def describe(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise EvidenceError("cannot summarize an empty numeric series")
    converted = [float(value) for value in values]
    if not all(math.isfinite(value) for value in converted):
        raise EvidenceError("numeric series contains a non-finite value")
    return {
        "values": converted,
        "mean": math.fsum(converted) / len(converted),
        "median": statistics.median(converted),
        "min": min(converted),
        "max": max(converted),
        "sample_variance": (
            statistics.variance(converted) if len(converted) > 1 else 0.0
        ),
        "sample_standard_deviation": (
            statistics.stdev(converted) if len(converted) > 1 else 0.0
        ),
        "count": len(converted),
    }


def mean_and_sample_variance(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise EvidenceError("cannot compute a mean for an empty sequence")
    mean = math.fsum(values) / len(values)
    variance = 0.0
    if len(values) > 1:
        variance = math.fsum((value - mean) ** 2 for value in values) / (
            len(values) - 1
        )
    return mean, variance


@dataclass(frozen=True)
class RenderSpec:
    phase: str
    purpose: str
    label: str
    relative_prefix: str
    scene_key: str
    scene_file: str
    sampler: str
    seed: int
    samples: int
    resolution: tuple[int, int]
    depth: int
    time_budget_ms: int | None = None
    scheduling_overrides: dict[str, str] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)

    def design_record(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "purpose": self.purpose,
            "label": self.label,
            "scene": self.scene_key,
            "scene_file": self.scene_file,
            "sampler": self.sampler,
            "seed": self.seed,
            "sample_target": self.samples,
            "resolution": list(self.resolution),
            "depth": self.depth,
            "time_budget_ms": self.time_budget_ms,
            "scheduling_overrides": dict(self.scheduling_overrides),
            "effective_scheduling": effective_scheduling(self),
            **self.tags,
        }


def effective_scheduling(spec: RenderSpec) -> dict[str, str]:
    """Return the fixed schedule plus explicit gallery-only overrides."""
    overrides = dict(spec.scheduling_overrides)
    unknown = sorted(set(overrides) - set(COMMON_SCHEDULING))
    if unknown:
        raise EvidenceError(
            f"{spec.label}: unknown scheduling override(s): {', '.join(unknown)}"
        )
    if overrides and spec.phase != "gallery":
        raise EvidenceError(
            f"{spec.label}: scheduling overrides are gallery-only"
        )
    effective = dict(COMMON_SCHEDULING)
    effective.update(overrides)
    for name in ("wavefront", "rr", "compact", "bvh", "aa", "nee", "sort", "motion"):
        if effective[name] not in ("on", "off"):
            raise EvidenceError(
                f"{spec.label}: --{name} must be on or off, got {effective[name]!r}"
            )
    try:
        rr_start = int(effective["rr-start"])
    except (TypeError, ValueError) as error:
        raise EvidenceError(
            f"{spec.label}: --rr-start is not an integer: {effective['rr-start']!r}"
        ) from error
    if rr_start < 0 or str(rr_start) != effective["rr-start"]:
        raise EvidenceError(
            f"{spec.label}: --rr-start must be a canonical nonnegative integer"
        )
    return effective


def expected_config_for(spec: RenderSpec) -> dict[str, Any]:
    """Translate the effective CLI schedule into the renderer metadata schema."""
    scheduling = effective_scheduling(spec)
    expected_sampler_id = SAMPLER_IDS[spec.sampler]
    return {
        "max_depth": spec.depth,
        "aa": scheduling["aa"] == "on",
        "nee": scheduling["nee"] == "on",
        "bvh": scheduling["bvh"] == "on",
        "compact": scheduling["compact"] == "on",
        "sort": scheduling["sort"] == "on",
        "wavefront": scheduling["wavefront"] == "on",
        "rr": scheduling["rr"] == "on",
        "rr_start": int(scheduling["rr-start"]),
        "sampler": spec.sampler,
        "sampler_id": expected_sampler_id,
        "motion": scheduling["motion"] == "on",
        "debug": "none",
        "depth_semantics": (
            "maximum surface intersections; final intersection is emission-only"
        ),
    }


def command_for(binary: Path, prefix: Path, spec: RenderSpec) -> list[str]:
    if spec.sampler not in SAMPLERS:
        raise EvidenceError(f"unsupported fixed sampler: {spec.sampler}")
    width, height = spec.resolution
    command = [
        str(binary.resolve()),
        str((ROOT / "scenes" / spec.scene_file).resolve()),
        "--samples",
        str(spec.samples),
        "--seed",
        str(spec.seed),
        "--output",
        str(prefix.resolve()),
        "--resolution",
        f"{width}x{height}",
        "--depth",
        str(spec.depth),
        "--benchmark",
    ]
    for name, value in effective_scheduling(spec).items():
        command.extend(("--" + name, value))
    command.extend(("--sampler", spec.sampler))
    if spec.time_budget_ms is not None:
        command.extend(("--time-budget-ms", str(spec.time_budget_ms)))
    return command


def quality_candidate_specs() -> list[RenderSpec]:
    result = []
    for scene_key, scene_file in QUALITY_SCENES:
        for samples in QUALITY_SAMPLE_COUNTS:
            for seed in QUALITY_SEEDS:
                for sampler in SAMPLERS:
                    label = f"candidate-{scene_key}-s{samples}-seed{seed}-{sampler}"
                    result.append(RenderSpec(
                        phase="quality",
                        purpose="candidate",
                        label=label,
                        relative_prefix="quality/candidates/" + label,
                        scene_key=scene_key,
                        scene_file=scene_file,
                        sampler=sampler,
                        seed=seed,
                        samples=samples,
                        resolution=QUALITY_RESOLUTION,
                        depth=QUALITY_DEPTH,
                        tags={"quality_samples": samples},
                    ))
    return result


def reference_specs(samples: int, scene_keys: Iterable[str] | None = None) -> list[RenderSpec]:
    selected = set(scene_keys) if scene_keys is not None else None
    result = []
    for scene_key, scene_file in QUALITY_SCENES:
        if selected is not None and scene_key not in selected:
            continue
        for seed in REFERENCE_SEEDS:
            label = f"reference-{scene_key}-s{samples}-seed{seed}-counter"
            result.append(RenderSpec(
                phase="quality",
                purpose="reference",
                label=label,
                relative_prefix="quality/references/" + label,
                scene_key=scene_key,
                scene_file=scene_file,
                sampler="counter",
                seed=seed,
                samples=samples,
                resolution=QUALITY_RESOLUTION,
                depth=QUALITY_DEPTH,
                tags={"reference_samples_per_seed": samples},
            ))
    return result


def equal_time_specs() -> list[RenderSpec]:
    result = []
    for scene_key, scene_file in QUALITY_SCENES:
        for seed in QUALITY_SEEDS:
            for sampler in SAMPLERS:
                label = f"equal-time-{scene_key}-seed{seed}-{sampler}"
                result.append(RenderSpec(
                    phase="quality",
                    purpose="equal-time",
                    label=label,
                    relative_prefix="quality/equal-time/" + label,
                    scene_key=scene_key,
                    scene_file=scene_file,
                    sampler=sampler,
                    seed=seed,
                    samples=EQUAL_TIME_TARGET_SAMPLES,
                    resolution=QUALITY_RESOLUTION,
                    depth=QUALITY_DEPTH,
                    time_budget_ms=EQUAL_TIME_BUDGET_MS,
                    tags={"time_budget_ms": EQUAL_TIME_BUDGET_MS},
                ))
    return result


def timing_specs() -> list[RenderSpec]:
    configurations = [
        ("static", scene_key, scene_file, sampler, scene_key)
        for scene_key, scene_file in QUALITY_SCENES
        for sampler in SAMPLERS
    ]
    configurations.extend((
        ("motion", "motion-midpoint", MOTION_SCENES["midpoint"], "halton", "midpoint"),
        ("motion", "motion-full", MOTION_SCENES["full"], "halton", "full-shutter"),
    ))
    result = []
    for kind, scene_key, scene_file, sampler, comparison_role in configurations:
        for measurement_phase, repeats in (
            ("warmup", TIMING_WARMUPS),
            ("measured", TIMING_MEASURED),
        ):
            for repeat in range(1, repeats + 1):
                label = (
                    f"timing-{scene_key}-{sampler}-{measurement_phase}-{repeat}"
                )
                result.append(RenderSpec(
                    phase="timing",
                    purpose="timing",
                    label=label,
                    relative_prefix="timing/runs/" + label,
                    scene_key=scene_key,
                    scene_file=scene_file,
                    sampler=sampler,
                    seed=TIMING_SEED,
                    samples=TIMING_SAMPLES,
                    resolution=QUALITY_RESOLUTION,
                    depth=QUALITY_DEPTH,
                    tags={
                        "kind": kind,
                        "comparison_role": comparison_role,
                        "measurement_phase": measurement_phase,
                        "repeat": repeat,
                    },
                ))
    return result


def gallery_specs() -> list[RenderSpec]:
    result = []
    for scene_key, scene_file in QUALITY_SCENES:
        for sampler in SAMPLERS:
            label = f"gallery-static-{scene_key}-{sampler}"
            result.append(RenderSpec(
                phase="gallery",
                purpose="gallery",
                label=label,
                relative_prefix="gallery/renders/" + label,
                scene_key=scene_key,
                scene_file=scene_file,
                sampler=sampler,
                seed=GALLERY_SEED,
                samples=GALLERY_STATIC_SAMPLES,
                resolution=GALLERY_STATIC_RESOLUTION,
                depth=QUALITY_DEPTH,
                tags={"kind": "static", "gallery_role": scene_key},
            ))
    for role in ("start", "midpoint", "end", "full"):
        scene_key = "motion-" + role
        label = f"gallery-{scene_key}-halton"
        result.append(RenderSpec(
            phase="gallery",
            purpose="gallery",
            label=label,
            relative_prefix="gallery/renders/" + label,
            scene_key=scene_key,
            scene_file=MOTION_SCENES[role],
            sampler="halton",
            seed=GALLERY_SEED,
            samples=GALLERY_MOTION_SAMPLES,
            resolution=GALLERY_STATIC_RESOLUTION,
            depth=QUALITY_DEPTH,
            tags={"kind": "motion", "gallery_role": role},
        ))

    comparison_designs = (
        {
            "pair_id": "obj-capability",
            "kind": "capability",
            "semantics": "capability-illustration-not-performance-comparison",
            "exactness": "not-asserted",
            "members": (
                ("baseline-without-obj", "obj-baseline", "hedge-obj-baseline.json", {}),
                (
                    "with-obj",
                    "procedural-optical-stilllife",
                    "procedural-optical-stilllife.json",
                    {},
                ),
            ),
            "resolution": GALLERY_COMPARISON_RESOLUTION,
            "samples": GALLERY_COMPARISON_SAMPLES,
        },
        {
            "pair_id": "bvh-schedule",
            "kind": "schedule-equivalence",
            "semantics": "exact-bvh-schedule-equivalence",
            "exactness": "raw-and-png-required",
            "members": (
                ("bvh-off", "complexity-medium", "complexity-medium.json", {"bvh": "off"}),
                ("bvh-on", "complexity-medium", "complexity-medium.json", {"bvh": "on"}),
            ),
            "resolution": GALLERY_BVH_RESOLUTION,
            "samples": GALLERY_BVH_SAMPLES,
        },
        {
            "pair_id": "wavefront-schedule",
            "kind": "schedule-equivalence",
            "semantics": "exact-unified-wavefront-equivalence",
            "exactness": "raw-and-png-required",
            "members": (
                (
                    "wavefront-off",
                    "optical-workbench",
                    "optical-workbench.json",
                    {"wavefront": "off", "rr": "on", "sort": "off"},
                ),
                (
                    "wavefront-on",
                    "optical-workbench",
                    "optical-workbench.json",
                    {"wavefront": "on", "rr": "on", "sort": "off"},
                ),
            ),
            "resolution": GALLERY_COMPARISON_RESOLUTION,
            "samples": GALLERY_COMPARISON_SAMPLES,
        },
        {
            "pair_id": "rr-illustration",
            "kind": "rr-illustration",
            "semantics": "single-image-illustration-not-statistical-validation",
            "exactness": "not-asserted",
            "members": (
                (
                    "rr-off",
                    "optical-workbench",
                    "optical-workbench.json",
                    {"rr": "off", "wavefront": "on", "sort": "off"},
                ),
                (
                    "rr-on",
                    "optical-workbench",
                    "optical-workbench.json",
                    {"rr": "on", "wavefront": "on", "sort": "off"},
                ),
            ),
            "resolution": GALLERY_COMPARISON_RESOLUTION,
            "samples": GALLERY_COMPARISON_SAMPLES,
        },
    )
    for design in comparison_designs:
        pair_id = design["pair_id"]
        for pair_role, scene_key, scene_file, overrides in design["members"]:
            label = f"gallery-{pair_id}-{pair_role}"
            result.append(RenderSpec(
                phase="gallery",
                purpose="gallery",
                label=label,
                relative_prefix="gallery/renders/" + label,
                scene_key=scene_key,
                scene_file=scene_file,
                sampler="counter",
                seed=GALLERY_SEED,
                samples=design["samples"],
                resolution=design["resolution"],
                depth=QUALITY_DEPTH,
                scheduling_overrides=overrides,
                tags={
                    "kind": design["kind"],
                    "gallery_role": pair_role,
                    "gallery_pair_id": pair_id,
                    "gallery_pair_semantics": design["semantics"],
                    "gallery_exactness_requirement": design["exactness"],
                },
            ))
    return result


def fixed_design() -> dict[str, Any]:
    return {
        "scheduling": {
            **COMMON_SCHEDULING,
            "benchmark": True,
            "note": (
                "fixed for quality and timing; gallery comparison members record "
                "explicit per-render overrides"
            ),
        },
        "quality": {
            "fixtures": [key for key, _ in QUALITY_SCENES],
            "resolution": list(QUALITY_RESOLUTION),
            "depth": QUALITY_DEPTH,
            "candidate_seeds": list(QUALITY_SEEDS),
            "candidate_samples": list(QUALITY_SAMPLE_COUNTS),
            "samplers": list(SAMPLERS),
            "candidate_render_count": len(quality_candidate_specs()),
            "reference_sampler": "counter",
            "reference_seeds": list(REFERENCE_SEEDS),
            "reference_samples_per_seed": REFERENCE_SAMPLES,
            "reference_render_count": len(reference_specs(REFERENCE_SAMPLES)),
            "reference_retry_samples_per_seed": REFERENCE_RETRY_SAMPLES,
            "reference_retry_policy": "exactly once, same four seeds, failed scenes only",
            "reference_precision_criterion": (
                "split_reference_rmse <= 0.25 * min(mean candidate RMSE at "
                "256 spp for counter, mean candidate RMSE at 256 spp for halton)"
            ),
            "equal_time_budget_ms": EQUAL_TIME_BUDGET_MS,
            "equal_time_target_samples": EQUAL_TIME_TARGET_SAMPLES,
            "equal_time_render_count": len(equal_time_specs()),
            "base_render_count": (
                len(quality_candidate_specs())
                + len(reference_specs(REFERENCE_SAMPLES))
                + len(equal_time_specs())
            ),
            "maximum_adaptive_render_count": (
                len(quality_candidate_specs())
                + len(reference_specs(REFERENCE_SAMPLES))
                + len(reference_specs(REFERENCE_RETRY_SAMPLES))
                + len(equal_time_specs())
            ),
        },
        "timing": {
            "resolution": list(QUALITY_RESOLUTION),
            "depth": QUALITY_DEPTH,
            "samples": TIMING_SAMPLES,
            "seed": TIMING_SEED,
            "warmups_per_configuration": TIMING_WARMUPS,
            "measured_repeats_per_configuration": TIMING_MEASURED,
            "configuration_count": 6,
            "render_count": len(timing_specs()),
        },
        "gallery": {
            "seed": GALLERY_SEED,
            "original_resolution": list(GALLERY_STATIC_RESOLUTION),
            "static_samples": GALLERY_STATIC_SAMPLES,
            "motion_samples": GALLERY_MOTION_SAMPLES,
            "comparison_samples": GALLERY_COMPARISON_SAMPLES,
            "bvh_resolution": list(GALLERY_BVH_RESOLUTION),
            "bvh_samples": GALLERY_BVH_SAMPLES,
            "comparison_pairs": [
                "obj-capability",
                "bvh-schedule",
                "wavefront-schedule",
                "rr-illustration",
            ],
            "comparison_runs": [
                spec.design_record()
                for spec in gallery_specs()
                if spec.tags.get("gallery_pair_id") is not None
            ],
            "render_count": len(gallery_specs()),
        },
        "counts": {
            "quality_base": 136,
            "quality_maximum_after_reference_retry": 144,
            "timing": 36,
            "gallery": 16,
            "all_base": 188,
            "all_maximum_after_reference_retry": 196,
        },
    }


def referenced_scene_assets(document: dict[str, Any]) -> list[str]:
    paths = []
    materials = document.get("Materials", {})
    if isinstance(materials, dict):
        for material in materials.values():
            if isinstance(material, dict):
                for key in ("TEXTURE", "BUMP"):
                    value = material.get(key)
                    if isinstance(value, str):
                        paths.append(value)
    objects = document.get("Objects", [])
    if isinstance(objects, list):
        for item in objects:
            if isinstance(item, dict) and isinstance(item.get("FILE"), str):
                paths.append(item["FILE"])
    return sorted(set(paths))


def inspect_scene_inputs(scene_files: Iterable[str]) -> dict[str, Any]:
    result = {}
    for filename in sorted(set(scene_files)):
        path = ROOT / "scenes" / filename
        if not path.is_file():
            raise EvidenceError(f"required authored scene does not exist: {path}")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EvidenceError(f"cannot parse scene {path}: {error}") from error
        assets = []
        for logical in referenced_scene_assets(document):
            asset = Path(logical)
            if not asset.is_absolute():
                asset = path.parent / asset
            asset = asset.resolve()
            if not asset.is_file():
                raise EvidenceError(
                    f"scene {filename} references missing asset {logical!r}"
                )
            assets.append({
                "logical_path": logical,
                "resolved_path": str(asset),
                "sha256": sha256_file(asset),
                "bytes": asset.stat().st_size,
            })
        result[filename] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "referenced_assets": assets,
            "native_camera": document.get("Camera"),
        }
    return result


def compact_scene_input_identity(
    filename: str, inspected: dict[str, Any]
) -> dict[str, Any]:
    assets = [
        {
            "logical_path": asset["logical_path"],
            "resolved_path": asset["resolved_path"],
            "sha256": asset["sha256"],
            "bytes": asset["bytes"],
        }
        for asset in inspected.get("referenced_assets", [])
    ]
    core = {
        "scene_file": filename,
        "resolved_path": inspected["path"],
        "scene_sha256": inspected["sha256"],
        "scene_bytes": inspected["bytes"],
        "assets": assets,
    }
    return {**core, "manifest_sha256": canonical_json_sha256(core)}


def inspect_scene_identity(filename: str) -> dict[str, Any]:
    inspected = inspect_scene_inputs((filename,))[filename]
    return compact_scene_input_identity(filename, inspected)


def expected_camera_metadata(scene_path: Path) -> dict[str, Any]:
    """Derive the camera fields emitted by headless metadata from authored JSON."""
    try:
        document = json.loads(scene_path.read_text(encoding="utf-8"))
        camera = document["Camera"]
        position = [float(value) for value in camera["EYE"]]
        look_at = [float(value) for value in camera["LOOKAT"]]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise EvidenceError(
            f"cannot derive expected camera metadata from {scene_path}: {error}"
        ) from error
    if len(position) != 3 or len(look_at) != 3:
        raise EvidenceError(f"{scene_path}: camera vectors must have three elements")
    aperture = float(camera.get("APERTURE", 0.0))
    if "FOCUS" in camera and "FOCUS_DISTANCE" in camera:
        raise EvidenceError(f"{scene_path}: camera has both focus spellings")
    if "FOCUS" in camera:
        focus_distance = float(camera["FOCUS"])
    elif "FOCUS_DISTANCE" in camera:
        focus_distance = float(camera["FOCUS_DISTANCE"])
    else:
        focus_distance = math.sqrt(math.fsum(
            (target - eye) ** 2 for target, eye in zip(look_at, position)
        ))
    shutter = [float(value) for value in camera.get("SHUTTER", [0.0, 1.0])]
    expected = {
        "position": position,
        "look_at": look_at,
        "aperture": aperture,
        "focus_distance": focus_distance,
        "shutter": shutter,
    }
    recursive_floats_are_finite(expected, f"{scene_path}.Camera")
    return expected


def initial_scene_identity(
    input_scenes: dict[str, Any], filename: str
) -> dict[str, Any]:
    inspected = input_scenes.get(filename)
    if not isinstance(inspected, dict):
        raise EvidenceError(f"initial input manifest lacks scene {filename}")
    return compact_scene_input_identity(filename, inspected)


def motion_design_signature(path: Path) -> tuple[str, dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    normalized = json.loads(json.dumps(document))
    camera = normalized.get("Camera")
    if not isinstance(camera, dict):
        raise EvidenceError(f"motion scene has no Camera object: {path}")
    camera.pop("FILE", None)
    camera.pop("SHUTTER", None)
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256_bytes(encoded), normalized


def verify_motion_variants() -> dict[str, Any]:
    variants = {}
    normalized_documents = {}
    camera_files = []
    for role in ("start", "midpoint", "end", "full"):
        filename = MOTION_SCENES[role]
        path = ROOT / "scenes" / filename
        document = json.loads(path.read_text(encoding="utf-8"))
        camera = document.get("Camera")
        if not isinstance(camera, dict):
            raise EvidenceError(f"motion scene has no Camera object: {path}")
        expected_shutter = list(MOTION_SHUTTERS[role])
        if camera.get("SHUTTER") != expected_shutter:
            raise EvidenceError(
                f"{filename}: expected SHUTTER {expected_shutter}, "
                f"found {camera.get('SHUTTER')!r}"
            )
        expected_native = {"RES": [960, 640], "ITERATIONS": 2048, "DEPTH": 10}
        for key, expected in expected_native.items():
            if camera.get(key) != expected:
                raise EvidenceError(
                    f"{filename}: expected Camera.{key}={expected!r}, "
                    f"found {camera.get(key)!r}"
                )
        camera_file = camera.get("FILE")
        if not isinstance(camera_file, str) or not camera_file:
            raise EvidenceError(f"{filename}: Camera.FILE must be nonempty")
        camera_files.append(camera_file)
        normalized_hash, normalized = motion_design_signature(path)
        normalized_documents[role] = normalized
        variants[role] = {
            "scene_file": filename,
            "camera_file": camera_file,
            "shutter": expected_shutter,
            "native_resolution": [960, 640],
            "native_iterations": 2048,
            "native_depth": 10,
            "normalized_scene_sha256": normalized_hash,
        }
    if len(set(camera_files)) != len(camera_files):
        raise EvidenceError(f"motion Camera.FILE values are not unique: {camera_files}")
    baseline = normalized_documents["start"]
    unequal = [
        role for role, document in normalized_documents.items() if document != baseline
    ]
    if unequal:
        raise EvidenceError(
            "motion variants differ outside Camera.FILE/SHUTTER: " + ", ".join(unequal)
        )
    normalized_hashes = {
        entry["normalized_scene_sha256"] for entry in variants.values()
    }
    if len(normalized_hashes) != 1:
        raise EvidenceError("equal normalized motion documents produced different hashes")
    normalized_hash = next(iter(normalized_hashes))
    evidence = {
        "equal": True,
        "all_four_variants_equal": True,
        "normalized_scene_sha256": normalized_hash,
        "excluded_variant_fields": ["Camera.FILE", "Camera.SHUTTER"],
        "camera_files_unique": True,
        "variants": variants,
        # Stable compatibility fields consumed by plot_hedge.py.
        "midpoint_normalized_scene_sha256": normalized_hash,
        "full_shutter_normalized_scene_sha256": normalized_hash,
        "midpoint_scene": MOTION_SCENES["midpoint"],
        "full_shutter_scene": MOTION_SCENES["full"],
    }
    validate_motion_design_evidence(evidence)
    return evidence


def validate_motion_design_evidence(evidence: Any) -> None:
    """Validate the captured, renderer-independent four-scene motion proof."""
    if not isinstance(evidence, dict):
        raise EvidenceError("motion design evidence is not an object")
    if evidence.get("equal") is not True or evidence.get(
        "all_four_variants_equal"
    ) is not True:
        raise EvidenceError("motion variants were not recorded as normalized-equal")
    if evidence.get("camera_files_unique") is not True:
        raise EvidenceError("motion Camera.FILE values were not recorded as unique")
    if evidence.get("excluded_variant_fields") != [
        "Camera.FILE",
        "Camera.SHUTTER",
    ]:
        raise EvidenceError("motion normalization excluded unexpected fields")
    top_hash = evidence.get("normalized_scene_sha256")
    if (
        not isinstance(top_hash, str)
        or len(top_hash) != 64
        or any(character not in "0123456789abcdef" for character in top_hash)
    ):
        raise EvidenceError("motion normalized design hash is invalid")
    variants = evidence.get("variants")
    if not isinstance(variants, dict) or set(variants) != set(MOTION_SCENES):
        raise EvidenceError("motion evidence does not contain exactly four variants")
    camera_files = []
    normalized_hashes = set()
    for role, filename in MOTION_SCENES.items():
        variant = variants.get(role)
        if not isinstance(variant, dict):
            raise EvidenceError(f"motion evidence lacks the {role} variant")
        expected = {
            "scene_file": filename,
            "shutter": list(MOTION_SHUTTERS[role]),
            "native_resolution": [960, 640],
            "native_iterations": 2048,
            "native_depth": 10,
        }
        for key, value in expected.items():
            if variant.get(key) != value:
                raise EvidenceError(
                    f"motion evidence {role}.{key} is {variant.get(key)!r}, "
                    f"expected {value!r}"
                )
        camera_file = variant.get("camera_file")
        if not isinstance(camera_file, str) or not camera_file:
            raise EvidenceError(f"motion evidence {role} has no Camera.FILE")
        camera_files.append(camera_file)
        normalized_hashes.add(variant.get("normalized_scene_sha256"))
    if len(set(camera_files)) != 4:
        raise EvidenceError("captured motion Camera.FILE values are not unique")
    if normalized_hashes != {top_hash}:
        raise EvidenceError("captured motion normalized hashes disagree")
    if evidence.get("midpoint_normalized_scene_sha256") != top_hash:
        raise EvidenceError("captured midpoint normalized hash disagrees")
    if evidence.get("full_shutter_normalized_scene_sha256") != top_hash:
        raise EvidenceError("captured full-shutter normalized hash disagrees")
    if evidence.get("midpoint_scene") != MOTION_SCENES["midpoint"]:
        raise EvidenceError("captured midpoint scene filename disagrees")
    if evidence.get("full_shutter_scene") != MOTION_SCENES["full"]:
        raise EvidenceError("captured full-shutter scene filename disagrees")


def verify_motion_pair() -> dict[str, Any]:
    """Compatibility name for the now stronger four-variant audit."""
    return verify_motion_variants()


def parse_benchmark(path: Path, completed_samples: int) -> dict[str, Any]:
    timing_columns = (
        "intersection_ms",
        "sort_ms",
        "shade_ms",
        "compaction_ms",
        "count_ms",
        "queue_ms",
        "queue_host_ms",
        "diffuse_ms",
        "mirror_ms",
        "dielectric_ms",
        "emission_ms",
        "terminal_ms",
    )
    overall = []
    bounce_rows = 0
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        required = {
            "record_type",
            "host_ms",
            "gpu_ms",
            "measured_samples",
            "render_loop_wall_ms",
        }
        if not required <= fields:
            raise EvidenceError(
                f"{path}: benchmark CSV lacks columns {sorted(required - fields)}"
            )
        for index, row in enumerate(reader):
            if None in row or any(value is None for value in row.values()):
                raise EvidenceError(
                    f"{path}: benchmark CSV contains a misaligned row at {index + 2}"
                )
            if row["record_type"] == "overall":
                overall.append(row)
            elif row["record_type"] == "bounce":
                bounce_rows += 1
                for column in timing_columns:
                    value = row.get(column, "")
                    if value not in (None, ""):
                        require_nonnegative_finite(
                            float(value), f"CSV bounce[{index}].{column}"
                        )
    if len(overall) != 1:
        raise EvidenceError(f"{path}: expected one overall benchmark row")
    row = overall[0]
    host_ms = require_nonnegative_finite(float(row["host_ms"]), "CSV host_ms")
    gpu_ms = require_nonnegative_finite(float(row["gpu_ms"]), "CSV gpu_ms")
    render_loop_ms = require_nonnegative_finite(
        float(row["render_loop_wall_ms"]), "CSV render_loop_wall_ms"
    )
    measured = int(row["measured_samples"])
    if measured != completed_samples:
        raise EvidenceError(
            f"{path}: CSV measured_samples {measured} != raw {completed_samples}"
        )
    if bounce_rows == 0:
        raise EvidenceError(f"{path}: benchmark CSV contains no bounce rows")
    return {
        "renderer_host_ms": host_ms,
        "renderer_gpu_ms": gpu_ms,
        "measured_samples": measured,
        "render_loop_wall_ms": render_loop_ms,
        "bounce_records": bounce_rows,
    }


def audit_artifact_paths(
    paths: dict[str, Path],
    spec: RenderSpec,
    process_elapsed_ms: float,
    evidence_root: Path,
    binary_identity: dict[str, Any],
    scene_input_identity: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    required_paths = {"png", "raw", "metadata", "benchmark_csv"}
    if set(paths) != required_paths:
        raise EvidenceError(
            f"{spec.label}: artifact path set differs from {sorted(required_paths)}"
        )
    artifacts = {name: file_record(path, evidence_root) for name, path in paths.items()}
    if artifacts["png"]["bytes"] <= 0:
        raise EvidenceError(f"{paths['png']}: empty PNG")

    raw = read_sum(paths["raw"])
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    recursive_floats_are_finite(metadata)
    checks = []

    def check(name: str, condition: bool, detail: Any) -> None:
        if not condition:
            raise EvidenceError(f"{spec.label}: audit {name} failed: {detail}")
        checks.append({"name": name, "passed": True, "detail": detail})

    check(
        "binary-sha256",
        isinstance(binary_identity.get("sha256"), str)
        and len(binary_identity["sha256"]) == 64,
        binary_identity,
    )
    check(
        "scene-input-hashes",
        isinstance(scene_input_identity.get("scene_sha256"), str)
        and len(scene_input_identity["scene_sha256"]) == 64
        and isinstance(scene_input_identity.get("manifest_sha256"), str)
        and len(scene_input_identity["manifest_sha256"]) == 64,
        scene_input_identity,
    )
    scene_identity_core = {
        key: value
        for key, value in scene_input_identity.items()
        if key != "manifest_sha256"
    }
    check(
        "scene-input-manifest-hash",
        scene_input_identity.get("manifest_sha256")
        == canonical_json_sha256(scene_identity_core),
        scene_input_identity.get("manifest_sha256"),
    )

    width, height = spec.resolution
    completed = require_int(metadata.get("completed_samples"), "completed_samples")
    target = require_int(metadata.get("target_samples"), "target_samples")
    check("raw-completed-samples", raw["samples"] == completed, completed)
    check(
        "raw-resolution",
        (raw["width"], raw["height"]) == (width, height)
        and metadata.get("resolution") == [width, height],
        {"raw": [raw["width"], raw["height"]], "metadata": metadata.get("resolution")},
    )
    check("target-samples", target == spec.samples, target)
    if spec.time_budget_ms is None:
        check("completed-fixed-target", completed == spec.samples, completed)
    else:
        check("completed-time-budget-range", 0 < completed <= spec.samples, completed)
    check("seed", metadata.get("seed") == spec.seed, metadata.get("seed"))

    expected_sampler_id = SAMPLER_IDS[spec.sampler]
    check(
        "sampler-version",
        metadata.get("sampler_version") == expected_sampler_id,
        metadata.get("sampler_version"),
    )
    check("sampler-name", metadata.get("sampler") == spec.sampler, metadata.get("sampler"))
    check("sampler-id", metadata.get("sampler_id") == expected_sampler_id, metadata.get("sampler_id"))

    config = metadata.get("config")
    if not isinstance(config, dict):
        raise EvidenceError(f"{spec.label}: metadata config is missing")
    expected_config = expected_config_for(spec)
    check(
        "config-fields",
        set(config) == set(expected_config),
        {"actual": sorted(config), "expected": sorted(expected_config)},
    )
    for key, expected in expected_config.items():
        check(f"config-{key}", config.get(key) == expected, config.get(key))
    check("benchmark-enabled", metadata.get("benchmark_enabled") is True, True)

    # read_sum is the authoritative structural parser and already verifies the
    # trailer.  Recompute the documented FNV value without independently
    # decoding trailer bytes or guessing payload offsets.
    raw_checksum = fnv1a(raw["encoded"][:-8])
    metadata_checksum = metadata.get("linear_sum_format", {}).get("checksum")
    check("raw-checksum", metadata_checksum == raw_checksum, metadata_checksum)

    compatibility = metadata.get("compatibility")
    fingerprints = metadata.get("fingerprints")
    if not isinstance(compatibility, dict):
        raise EvidenceError(f"{spec.label}: compatibility metadata is missing")
    if not isinstance(fingerprints, dict):
        raise EvidenceError(f"{spec.label}: fingerprint metadata is missing")
    for key in ("baseline_starter_revision", "build_identifier"):
        value = compatibility.get(key)
        check(f"compatibility-{key}", isinstance(value, str) and bool(value), value)
    check(
        "compatibility-device",
        isinstance(compatibility.get("device"), str)
        and bool(compatibility.get("device")),
        compatibility.get("device"),
    )
    for key in ("scene_file", "assets", "materials", "geometry", "render_config"):
        value = fingerprints.get(key)
        check(
            f"fingerprint-{key}",
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            value,
        )
    check(
        "scene-input-filename",
        scene_input_identity.get("scene_file") == spec.scene_file,
        scene_input_identity,
    )
    scene_path_value = scene_input_identity.get("resolved_path")
    if not isinstance(scene_path_value, str) or not scene_path_value:
        raise EvidenceError(f"{spec.label}: scene input identity lacks resolved_path")
    scene_path = Path(scene_path_value).resolve()
    metadata_scene = metadata.get("scene")
    check(
        "metadata-scene-path",
        isinstance(metadata_scene, str)
        and Path(metadata_scene).resolve() == scene_path,
        metadata_scene,
    )
    camera = metadata.get("camera")
    if not isinstance(camera, dict):
        raise EvidenceError(f"{spec.label}: metadata camera is missing")
    expected_camera = expected_camera_metadata(scene_path)
    expected_shutter = EXPECTED_SCENE_SHUTTERS.get(spec.scene_file)
    if expected_shutter is None:
        raise EvidenceError(f"{spec.label}: no authored shutter expectation")
    check(
        "authored-camera-shutter-design",
        expected_camera["shutter"] == list(expected_shutter),
        {"scene": expected_camera["shutter"], "design": list(expected_shutter)},
    )
    check(
        "camera-fields",
        set(camera) == {
            "position", "look_at", "aperture", "focus_distance", "shutter"
        },
        sorted(camera),
    )
    for key in ("position", "look_at", "shutter"):
        actual_values = camera.get(key)
        expected_values = expected_camera[key]
        check(
            "camera-" + key.replace("_", "-"),
            isinstance(actual_values, list)
            and len(actual_values) == len(expected_values)
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and math.isclose(
                    float(value), expected,
                    rel_tol=1e-6, abs_tol=1e-6,
                )
                for value, expected in zip(actual_values, expected_values)
            ),
            {"actual": actual_values, "expected": expected_values},
        )
    for key in ("aperture", "focus_distance"):
        value = camera.get(key)
        check(
            "camera-" + key.replace("_", "-"),
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and math.isclose(
                float(value), expected_camera[key], rel_tol=1e-6, abs_tol=1e-6
            ),
            {"actual": value, "expected": expected_camera[key]},
        )
    check(
        "camera-shutter",
        camera.get("shutter") == list(expected_shutter),
        {"actual": camera.get("shutter"), "expected": list(expected_shutter)},
    )

    render_loop_ms = require_nonnegative_finite(
        metadata.get("render_loop_wall_ms"), "metadata.render_loop_wall_ms"
    )
    benchmark = metadata.get("benchmark")
    if not isinstance(benchmark, dict):
        raise EvidenceError(f"{spec.label}: benchmark metadata is missing")
    gpu_ms = require_nonnegative_finite(
        benchmark.get("renderer_gpu_ms"), "metadata.benchmark.renderer_gpu_ms"
    )
    host_ms = require_nonnegative_finite(
        benchmark.get("renderer_host_ms"), "metadata.benchmark.renderer_host_ms"
    )
    check(
        "benchmark-measured-samples",
        benchmark.get("measured_samples") == completed,
        benchmark.get("measured_samples"),
    )
    csv_benchmark = parse_benchmark(paths["benchmark_csv"], completed)
    for key, metadata_value in (
        ("renderer_gpu_ms", gpu_ms),
        ("renderer_host_ms", host_ms),
        ("render_loop_wall_ms", render_loop_ms),
    ):
        check(
            "benchmark-csv-" + key,
            math.isclose(
                csv_benchmark[key], metadata_value, rel_tol=2e-6, abs_tol=1e-5
            ),
            {"csv": csv_benchmark[key], "metadata": metadata_value},
        )

    policy = metadata.get("run_policy")
    if not isinstance(policy, dict):
        raise EvidenceError(f"{spec.label}: run_policy is missing")
    check(
        "policy-fields",
        set(policy) == {"time_budget_ms", "stop_reason", "last_sample_wall_ms"},
        sorted(policy),
    )
    check(
        "policy-stop-reason-enum",
        policy.get("stop_reason") in ("sample_target", "time_budget"),
        policy.get("stop_reason"),
    )
    last_sample_wall_ms = require_nonnegative_finite(
        policy.get("last_sample_wall_ms"), "run_policy.last_sample_wall_ms"
    )
    check(
        "policy-last-sample-within-loop",
        last_sample_wall_ms <= render_loop_ms,
        {"last_sample_wall_ms": last_sample_wall_ms, "render_loop_wall_ms": render_loop_ms},
    )
    if spec.time_budget_ms is None:
        check("policy-budget-null", policy.get("time_budget_ms") is None, policy)
        check("policy-stop-reason", policy.get("stop_reason") == "sample_target", policy)
    else:
        check(
            "policy-budget-positive-integer",
            isinstance(policy.get("time_budget_ms"), int)
            and not isinstance(policy.get("time_budget_ms"), bool)
            and policy.get("time_budget_ms") > 0
            and policy.get("time_budget_ms") == spec.time_budget_ms,
            policy,
        )
        expected_reason = "time_budget" if completed < target else "sample_target"
        check("policy-stop-reason", policy.get("stop_reason") == expected_reason, policy)
        if expected_reason == "time_budget":
            check(
                "policy-whole-sample-overshoot",
                render_loop_ms >= spec.time_budget_ms,
                {"elapsed_ms": render_loop_ms, "budget_ms": spec.time_budget_ms},
            )

    process_elapsed_ms = require_nonnegative_finite(
        process_elapsed_ms, "process elapsed milliseconds"
    )
    non_render_overhead_ms = process_elapsed_ms - render_loop_ms
    if non_render_overhead_ms < 0.0:
        raise EvidenceError(
            f"{spec.label}: process duration is below renderer loop duration"
        )
    inv_samples = 1.0 / completed
    raw_linear_mean = math.fsum(raw["values"]) / len(raw["values"]) * inv_samples
    time_delta = (
        render_loop_ms - spec.time_budget_ms
        if spec.time_budget_ms is not None
        else None
    )
    actual = {
        "resolution": [raw["width"], raw["height"]],
        "target_samples": target,
        "completed_samples": completed,
        "seed": metadata["seed"],
        "sampler": metadata["sampler"],
        "sampler_id": metadata["sampler_id"],
        "sampler_version": metadata["sampler_version"],
        "config": config,
        "camera": camera,
        "linear_rgb_mean": raw_linear_mean,
        "raw_checksum_fnv1a64": raw_checksum,
        "render_loop_wall_ms": render_loop_ms,
        "renderer_gpu_ms": gpu_ms,
        "renderer_host_ms": host_ms,
        "process_elapsed_ms": process_elapsed_ms,
        "non_render_process_overhead_ms": non_render_overhead_ms,
        "renderer_gpu_ms_per_completed_sample": gpu_ms * inv_samples,
        "render_loop_wall_ms_per_completed_sample": render_loop_ms * inv_samples,
        "process_elapsed_ms_per_completed_sample": process_elapsed_ms * inv_samples,
        "run_policy": policy,
        "last_sample_wall_ms": last_sample_wall_ms,
        "render_loop_minus_budget_ms": time_delta,
        "time_budget_overshoot_ms": (
            max(0.0, time_delta) if time_delta is not None else None
        ),
        "benchmark_csv": csv_benchmark,
        "identity": {
            "binary_sha256": binary_identity.get("sha256"),
            "binary_file_identity": binary_identity,
            "metadata_scene": metadata_scene,
            "scene_file_sha256": scene_input_identity.get("scene_sha256"),
            "scene_input_identity": scene_input_identity,
            "compatibility": compatibility,
            "fingerprints": fingerprints,
        },
    }
    audit = {
        "passed": True,
        "checks": checks,
        "raw_parser": "scripts.validate_render.read_sum",
        "raw_values_finite_nonnegative": True,
    }
    return artifacts, {"actual": actual, "audit": audit}


def audit_render(
    prefix: Path,
    spec: RenderSpec,
    process_elapsed_ms: float,
    evidence_root: Path,
    binary_identity: dict[str, Any],
    scene_input_identity: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "png": suffix(prefix, ".png"),
        "raw": suffix(prefix, ".sum.bin"),
        "metadata": suffix(prefix, ".json"),
        "benchmark_csv": suffix(prefix, ".benchmark.csv"),
    }
    return audit_artifact_paths(
        paths,
        spec,
        process_elapsed_ms,
        evidence_root,
        binary_identity,
        scene_input_identity,
    )


RENDER_ARTIFACT_EXTENSIONS = {
    "png": ".png",
    "raw": ".sum.bin",
    "metadata": ".json",
    "benchmark_csv": ".benchmark.csv",
}


def existing_render_artifacts(prefix: Path, evidence_root: Path) -> dict[str, Any]:
    result = {}
    for name, extension in RENDER_ARTIFACT_EXTENSIONS.items():
        path = suffix(prefix, extension)
        if path.is_file():
            result[name] = file_record(path, evidence_root)
    return result


def terminate_process_group(process: subprocess.Popen[Any]) -> int | None:
    """Terminate one renderer process group and synchronously reap its child."""
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
        try:
            return process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Usually redundant after SIGKILL to the process group, but this
            # explicitly guarantees that the direct child is killed and reaped.
            process.kill()
            return process.wait(timeout=5)


class MeasurementSession:
    def __init__(self, output: Path, binary: Path, phase: str):
        self.output = output.resolve()
        self.binary = binary.resolve()
        self.records: list[dict[str, Any]] = []
        self.state: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "tool": "scripts/measure_hedge.py",
            "status": "running",
            "execution_status": "running",
            "phase_requested": phase,
            "started_utc": utc_now(),
            "completed_utc": None,
            "design": fixed_design(),
            "identity": {},
            "input_scenes": {},
            "run_count": 0,
            "phases": {},
            "errors": [],
        }
        self.binary_sha256 = ""
        self.initial_binary_identity: dict[str, Any] = {}

    def initialize_identity(self) -> None:
        if not self.binary.is_file():
            raise EvidenceError(f"renderer binary does not exist: {self.binary}")
        if not os.access(self.binary, os.X_OK):
            raise EvidenceError(f"renderer binary is not executable: {self.binary}")
        self.initial_binary_identity = binary_file_identity(self.binary)
        self.binary_sha256 = self.initial_binary_identity["sha256"]
        self.state["identity"] = {
            "binary_path": str(self.binary),
            "binary_sha256": self.binary_sha256,
            "binary_bytes": self.initial_binary_identity["bytes"],
            "initial_binary_file_identity": self.initial_binary_identity,
            "driver_path": str(Path(__file__).resolve()),
            "driver_sha256": sha256_file(Path(__file__).resolve()),
        }

    def persist(self) -> None:
        self.state["run_count"] = len(self.records)
        atomic_json(
            self.output / "runs.json",
            {"schema_version": SCHEMA_VERSION, "runs": self.records},
        )
        atomic_json(self.output / "summary.json", self.state)
        for phase in ("quality", "timing", "gallery"):
            phase_dir = self.output / phase
            if not phase_dir.is_dir():
                continue
            phase_records = [record for record in self.records if record["phase"] == phase]
            atomic_json(
                phase_dir / "runs.json",
                {"schema_version": SCHEMA_VERSION, "runs": phase_records},
            )
            if phase in self.state["phases"]:
                atomic_json(phase_dir / "summary.json", self.state["phases"][phase])

    def prepare_phase(self, phase: str) -> None:
        phase_dir = self.output / phase
        phase_dir.mkdir(parents=False, exist_ok=False)
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
        prefix = self.output / spec.relative_prefix
        prefix.parent.mkdir(parents=True, exist_ok=True)
        output_candidates = [
            suffix(prefix, extension)
            for extension in (
                ".png",
                ".sum.bin",
                ".json",
                ".benchmark.csv",
                ".stdout.txt",
                ".stderr.txt",
                ".command.json",
            )
        ]
        existing = [str(path) for path in output_candidates if path.exists()]
        if existing:
            raise EvidenceError(
                f"{spec.label}: refusing to overwrite invocation artifacts: {existing}"
            )
        command = command_for(self.binary, prefix, spec)
        command_path = suffix(prefix, ".command.json")
        stdout_path = suffix(prefix, ".stdout.txt")
        stderr_path = suffix(prefix, ".stderr.txt")
        run_id = len(self.records) + 1
        started = utc_now()
        invocation = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "status": "running",
            "phase": spec.phase,
            "label": spec.label,
            "design": spec.design_record(),
            "command": command,
            "cwd": str(ROOT),
            "started_utc": started,
            "completed_utc": None,
            "exit_code": None,
            "process_elapsed_ms": None,
            "binary_identity": {"before": None, "after": None},
            "scene_input_identity": {"before": None, "after": None},
            "pid": None,
        }
        atomic_json(command_path, invocation)

        envelope_began = time.perf_counter()
        process_began: float | None = None
        process_elapsed_ms: float | None = None
        exit_code: int | None = None
        process: subprocess.Popen[Any] | None = None
        interrupted = False
        launch_error: Exception | None = None
        identity_errors: list[str] = []
        binary_before: dict[str, Any] | None = None
        binary_after: dict[str, Any] | None = None
        scene_before: dict[str, Any] | None = None
        scene_after: dict[str, Any] | None = None
        try:
            binary_before = binary_file_identity(self.binary)
            initial_binary = self.initial_binary_identity
            if not same_binary_identity(binary_before, initial_binary):
                identity_errors.append(
                    "renderer binary differs from the initial session identity before launch"
                )
            scene_before = inspect_scene_identity(spec.scene_file)
            expected_scene = initial_scene_identity(
                self.state["input_scenes"], spec.scene_file
            )
            if scene_before != expected_scene:
                identity_errors.append(
                    "scene or referenced assets differ from the initial input manifest before launch"
                )
        except KeyboardInterrupt:
            interrupted = True
            launch_error = EvidenceError("driver interrupted during invocation preflight")
        except Exception as error:
            launch_error = error

        invocation["binary_identity"]["before"] = binary_before
        invocation["scene_input_identity"]["before"] = scene_before
        atomic_json(command_path, invocation)

        with stdout_path.open("wb") as stdout_stream, stderr_path.open("wb") as stderr_stream:
            if launch_error is None and not identity_errors:
                try:
                    process_began = time.perf_counter()
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        stdout=stdout_stream,
                        stderr=stderr_stream,
                        start_new_session=True,
                    )
                    invocation["pid"] = process.pid
                    atomic_json(command_path, invocation)
                    exit_code = process.wait()
                except KeyboardInterrupt:
                    interrupted = True
                    if process is not None:
                        exit_code = terminate_process_group(process)
                    stderr_stream.write(
                        b"\nmeasurement driver interrupted; renderer process group terminated\n"
                    )
                    stderr_stream.flush()
                except Exception as error:
                    launch_error = error
                    if process is not None and process.poll() is None:
                        exit_code = terminate_process_group(process)
                    stderr_stream.write((str(error) + "\n").encode("utf-8", "replace"))
                    stderr_stream.flush()
                finally:
                    if process_began is not None:
                        process_elapsed_ms = (
                            time.perf_counter() - process_began
                        ) * 1000.0
            else:
                reason = launch_error or EvidenceError("; ".join(identity_errors))
                stderr_stream.write((str(reason) + "\n").encode("utf-8", "replace"))
                stderr_stream.flush()
        try:
            binary_after = binary_file_identity(self.binary)
            if binary_before is None or not same_binary_identity(
                binary_before, binary_after
            ):
                identity_errors.append(
                    "renderer binary was replaced or modified during the invocation"
                )
        except KeyboardInterrupt:
            interrupted = True
            identity_errors.append("driver interrupted during post-invocation binary audit")
        except Exception as error:
            identity_errors.append(f"cannot hash renderer binary after invocation: {error}")
        try:
            scene_after = inspect_scene_identity(spec.scene_file)
            if scene_before is None or scene_after != scene_before:
                identity_errors.append(
                    "scene or referenced assets changed during the invocation"
                )
        except KeyboardInterrupt:
            interrupted = True
            identity_errors.append("driver interrupted during post-invocation scene audit")
        except Exception as error:
            identity_errors.append(f"cannot hash scene inputs after invocation: {error}")
        invocation_envelope_elapsed_ms = (
            time.perf_counter() - envelope_began
        ) * 1000.0

        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "phase": spec.phase,
            "purpose": spec.purpose,
            "label": spec.label,
            "status": "failed",
            "design": spec.design_record(),
            "command": command,
            "cwd": str(ROOT),
            "started_utc": started,
            "completed_utc": utc_now(),
            "exit_code": exit_code,
            "process_elapsed_ms": process_elapsed_ms,
            "invocation_envelope_elapsed_ms": invocation_envelope_elapsed_ms,
            "stdout": file_record(stdout_path, self.output),
            "stderr": file_record(stderr_path, self.output),
            "artifacts": existing_render_artifacts(prefix, self.output),
            "binary_identity": {
                "before": binary_before,
                "after": binary_after,
                "unchanged": (
                    binary_before is not None
                    and binary_after is not None
                    and same_binary_identity(binary_before, binary_after)
                ),
            },
            "scene_input_identity": {
                "before": scene_before,
                "after": scene_after,
                "unchanged": scene_before is not None and scene_after == scene_before,
            },
        }
        if interrupted:
            record["status"] = "interrupted"
            record["failure"] = {
                "kind": "interrupted",
                "message": "driver interrupted; renderer process group terminated and reaped",
            }
        elif launch_error is not None:
            record["failure"] = {
                "kind": "renderer-launch",
                "message": f"{type(launch_error).__name__}: {launch_error}",
            }
        elif identity_errors:
            record["failure"] = {
                "kind": "identity-change",
                "message": "; ".join(identity_errors),
            }
        elif exit_code == 0:
            try:
                if binary_before is None or scene_before is None:
                    raise EvidenceError("missing pre-invocation identities")
                artifacts, audited = audit_render(
                    prefix,
                    spec,
                    process_elapsed_ms,
                    self.output,
                    binary_before,
                    scene_before,
                )
                record["artifacts"] = artifacts
                record.update(audited)
                record["status"] = "passed"
            except Exception as error:
                record["audit"] = {
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        else:
            record["failure"] = {
                "kind": "renderer-exit",
                "message": f"renderer exited with code {exit_code}",
            }

        invocation.update(record)
        invocation["status"] = record["status"]
        atomic_json(command_path, invocation)
        record["invocation_record"] = file_record(command_path, self.output)
        self.records.append(record)
        phase_state = self.state["phases"][spec.phase]
        phase_state["run_count"] = sum(
            candidate["phase"] == spec.phase for candidate in self.records
        )
        phase_state["last_run"] = {
            "run_id": run_id,
            "label": spec.label,
            "status": record["status"],
        }
        self.persist()
        print(
            f"[{run_id}] {spec.phase}/{spec.label}: {record['status']} "
            f"({invocation_envelope_elapsed_ms / 1000.0:.3f}s envelope)",
            flush=True,
        )
        if interrupted:
            raise KeyboardInterrupt
        if record["status"] != "passed":
            error = record.get("audit", record.get("failure"))
            raise EvidenceError(f"{spec.label} failed: {error}")
        return record


def matching_records(
    records: Sequence[dict[str, Any]], purpose: str, **values: Any
) -> list[dict[str, Any]]:
    result = []
    for record in records:
        if record.get("status") != "passed" or record.get("purpose") != purpose:
            continue
        design = record.get("design", {})
        if all(design.get(key) == value for key, value in values.items()):
            result.append(record)
    return result


def artifact_path(record: dict[str, Any], name: str, evidence_root: Path) -> Path:
    try:
        artifact = record["artifacts"][name]
        relative = artifact["path"]
        if not isinstance(relative, str) or not relative:
            raise EvidenceError("artifact path is not a nonempty string")
        path = (evidence_root / relative).resolve()
        path.relative_to(evidence_root.resolve())
    except (KeyError, TypeError) as error:
        raise EvidenceError(
            f"run {record.get('label')} has no retained {name} artifact"
        ) from error
    except ValueError as error:
        raise EvidenceError(
            f"run {record.get('label')} {name} artifact escapes evidence root"
        ) from error
    if not path.is_file():
        raise EvidenceError(f"retained artifact is missing: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != artifact.get("sha256"):
        raise EvidenceError(
            f"retained artifact hash mismatch for {path}: "
            f"{actual_hash} != {artifact.get('sha256')}"
        )
    if path.stat().st_size != artifact.get("bytes"):
        raise EvidenceError(
            f"retained artifact size mismatch for {path}: "
            f"{path.stat().st_size} != {artifact.get('bytes')}"
        )
    return path


def sidecar_path(record: dict[str, Any], name: str, evidence_root: Path) -> Path:
    try:
        reference = record[name]
        relative = reference["path"]
        if not isinstance(relative, str) or not relative:
            raise EvidenceError("sidecar path is not a nonempty string")
        path = (evidence_root / relative).resolve()
        path.relative_to(evidence_root.resolve())
    except (KeyError, TypeError) as error:
        raise EvidenceError(
            f"run {record.get('label')} has no retained {name} sidecar"
        ) from error
    except ValueError as error:
        raise EvidenceError(
            f"run {record.get('label')} {name} sidecar escapes evidence root"
        ) from error
    if not path.is_file():
        raise EvidenceError(f"retained sidecar is missing: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != reference.get("sha256"):
        raise EvidenceError(
            f"retained sidecar hash mismatch for {path}: "
            f"{actual_hash} != {reference.get('sha256')}"
        )
    if path.stat().st_size != reference.get("bytes"):
        raise EvidenceError(f"retained sidecar size mismatch for {path}")
    return path


def spec_from_record(record: dict[str, Any]) -> RenderSpec:
    design = record.get("design")
    if not isinstance(design, dict):
        raise EvidenceError(f"run {record.get('label')} lacks a design object")
    resolution = design.get("resolution")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in resolution)
    ):
        raise EvidenceError(f"run {record.get('label')} has invalid design resolution")
    standard = {
        "phase",
        "purpose",
        "label",
        "scene",
        "scene_file",
        "sampler",
        "seed",
        "sample_target",
        "resolution",
        "depth",
        "time_budget_ms",
        "scheduling_overrides",
        "effective_scheduling",
    }
    tags = {key: value for key, value in design.items() if key not in standard}
    scheduling_overrides = design.get("scheduling_overrides")
    if not isinstance(scheduling_overrides, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in scheduling_overrides.items()
    ):
        raise EvidenceError(
            f"run {record.get('label')} has invalid scheduling_overrides"
        )
    spec = RenderSpec(
        phase=design.get("phase"),
        purpose=design.get("purpose"),
        label=design.get("label"),
        relative_prefix="reanalysis/" + str(design.get("label")),
        scene_key=design.get("scene"),
        scene_file=design.get("scene_file"),
        sampler=design.get("sampler"),
        seed=design.get("seed"),
        samples=design.get("sample_target"),
        resolution=(resolution[0], resolution[1]),
        depth=design.get("depth"),
        time_budget_ms=design.get("time_budget_ms"),
        scheduling_overrides=dict(scheduling_overrides),
        tags=tags,
    )
    if spec.design_record() != design:
        raise EvidenceError(
            f"run {record.get('label')} design cannot be reconstructed exactly"
        )
    for key, expected in (
        ("phase", spec.phase),
        ("purpose", spec.purpose),
        ("label", spec.label),
    ):
        if record.get(key) != expected:
            raise EvidenceError(
                f"run {record.get('label')} top-level {key} disagrees with design"
            )
    return spec


def reaudit_record(record: dict[str, Any], evidence_root: Path) -> dict[str, Any]:
    """Rebuild one successful run solely from its hashed retained artifacts."""
    # Verify every hash before parsing any artifact-derived value.
    paths = {
        name: artifact_path(record, name, evidence_root)
        for name in RENDER_ARTIFACT_EXTENSIONS
    }
    sidecar_path(record, "stdout", evidence_root)
    sidecar_path(record, "stderr", evidence_root)
    invocation_path = sidecar_path(record, "invocation_record", evidence_root)

    invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
    if invocation.get("status") != "passed" or invocation.get("exit_code") != 0:
        raise EvidenceError(
            f"run {record.get('label')} invocation was not a successful exit-zero run"
        )
    if invocation.get("command") != record.get("command"):
        raise EvidenceError(f"run {record.get('label')} command record disagrees")
    if invocation.get("design") != record.get("design"):
        raise EvidenceError(f"run {record.get('label')} design record disagrees")
    for key in ("run_id", "phase", "purpose", "label"):
        if invocation.get(key) != record.get(key):
            raise EvidenceError(
                f"run {record.get('label')} invocation {key} disagrees"
            )
    process_elapsed_ms = require_nonnegative_finite(
        invocation.get("process_elapsed_ms"), "invocation.process_elapsed_ms"
    )
    binary = invocation.get("binary_identity")
    scene = invocation.get("scene_input_identity")
    if not isinstance(binary, dict) or not isinstance(scene, dict):
        raise EvidenceError(f"run {record.get('label')} invocation lacks input identities")
    binary_before = binary.get("before")
    binary_after = binary.get("after")
    scene_before = scene.get("before")
    scene_after = scene.get("after")
    if not isinstance(binary_before, dict) or not isinstance(binary_after, dict):
        raise EvidenceError(f"run {record.get('label')} lacks binary before/after hashes")
    if not same_binary_identity(binary_before, binary_after):
        raise EvidenceError(f"run {record.get('label')} binary identity changed")
    if not isinstance(scene_before, dict) or scene_before != scene_after:
        raise EvidenceError(f"run {record.get('label')} scene input identity changed")
    spec = spec_from_record(record)
    artifacts, audited = audit_artifact_paths(
        paths,
        spec,
        process_elapsed_ms,
        evidence_root,
        binary_before,
        scene_before,
    )
    reconstructed = copy.deepcopy(record)
    cached_actual = reconstructed.get("actual")
    if cached_actual is not None and cached_actual != audited["actual"]:
        raise EvidenceError(
            f"run {record.get('label')} cached actual values disagree with artifacts"
        )
    if reconstructed.get("artifacts") != artifacts:
        raise EvidenceError(f"run {record.get('label')} artifact records disagree")
    if invocation.get("artifacts") != artifacts:
        raise EvidenceError(
            f"run {record.get('label')} hashed invocation artifact records disagree"
        )
    if invocation.get("actual") != audited["actual"]:
        raise EvidenceError(
            f"run {record.get('label')} hashed invocation actual values disagree"
        )
    expected_binary_record = {
        "before": binary_before,
        "after": binary_after,
        "unchanged": True,
    }
    expected_scene_record = {
        "before": scene_before,
        "after": scene_after,
        "unchanged": True,
    }
    if record.get("binary_identity") != expected_binary_record:
        raise EvidenceError(
            f"run {record.get('label')} cached binary identity disagrees"
        )
    if record.get("scene_input_identity") != expected_scene_record:
        raise EvidenceError(
            f"run {record.get('label')} cached scene input identity disagrees"
        )
    reconstructed.update(audited)
    reconstructed["actual"] = audited["actual"]
    reconstructed["process_elapsed_ms"] = process_elapsed_ms
    reconstructed["binary_identity"] = expected_binary_record
    reconstructed["scene_input_identity"] = expected_scene_record
    reconstructed["reanalysis"] = {
        "independently_reconstructed": True,
        "verified_artifact_hashes": sorted(paths),
        "verified_sidecar_hashes": ["stdout", "stderr", "invocation_record"],
        "cached_actual_matched": cached_actual == audited["actual"],
    }
    return reconstructed


def reaudit_records(
    records: Sequence[dict[str, Any]],
    evidence_root: Path,
    phases: Sequence[str],
) -> list[dict[str, Any]]:
    selected = set(phases)
    rebuilt = []
    for record in records:
        if record.get("phase") in selected and record.get("status") == "passed":
            rebuilt.append(reaudit_record(record, evidence_root))
        else:
            rebuilt.append(copy.deepcopy(record))
    return rebuilt


def normalized_image(
    record: dict[str, Any], evidence_root: Path
) -> tuple[tuple[int, int], list[float]]:
    raw = read_sum(artifact_path(record, "raw", evidence_root))
    completed = raw["samples"]
    if completed != record["actual"]["completed_samples"]:
        raise EvidenceError(f"{record['label']}: retained raw sample count changed")
    inverse = 1.0 / completed
    return (raw["width"], raw["height"]), [value * inverse for value in raw["values"]]


def average_images(
    records: Sequence[dict[str, Any]], evidence_root: Path
) -> tuple[tuple[int, int], list[float]]:
    if not records:
        raise EvidenceError("cannot average an empty image set")
    dimensions: tuple[int, int] | None = None
    accumulation: list[float] | None = None
    for record in records:
        current_dimensions, values = normalized_image(record, evidence_root)
        if dimensions is None:
            dimensions = current_dimensions
            accumulation = list(values)
        else:
            if current_dimensions != dimensions or len(values) != len(accumulation or ()):
                raise EvidenceError("reference images do not have identical dimensions")
            assert accumulation is not None
            for index, value in enumerate(values):
                accumulation[index] += value
    assert dimensions is not None and accumulation is not None
    inverse = 1.0 / len(records)
    return dimensions, [value * inverse for value in accumulation]


def image_metrics(
    record: dict[str, Any],
    evidence_root: Path,
    reference_dimensions: tuple[int, int],
    reference: Sequence[float],
) -> dict[str, float]:
    dimensions, values = normalized_image(record, evidence_root)
    if dimensions != reference_dimensions or len(values) != len(reference):
        raise EvidenceError(f"{record['label']}: image/reference dimensions differ")
    mse = math.fsum(
        (value - reference_value) ** 2
        for value, reference_value in zip(values, reference)
    ) / len(values)
    linear_mean = math.fsum(values) / len(values)
    return {"mse": mse, "rmse": math.sqrt(mse), "linear_rgb_mean": linear_mean}


def image_rmse(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise EvidenceError("RMSE inputs must be nonempty and equally sized")
    mse = math.fsum((a - b) ** 2 for a, b in zip(left, right)) / len(left)
    return math.sqrt(mse)


def unique_record(
    records: Sequence[dict[str, Any]], purpose: str, **values: Any
) -> dict[str, Any]:
    matches = matching_records(records, purpose, **values)
    if len(matches) != 1:
        raise EvidenceError(
            f"expected one {purpose} record matching {values}, found {len(matches)}"
        )
    return matches[0]


def reference_attempt(
    records: Sequence[dict[str, Any]],
    evidence_root: Path,
    scene_key: str,
    samples_per_seed: int,
) -> tuple[dict[str, Any], tuple[int, int], list[float], list[dict[str, Any]]]:
    references = [
        unique_record(
            records,
            "reference",
            scene=scene_key,
            seed=seed,
            sampler="counter",
            reference_samples_per_seed=samples_per_seed,
        )
        for seed in REFERENCE_SEEDS
    ]
    first_half = references[:2]
    second_half = references[2:]
    dimensions_a, average_a = average_images(first_half, evidence_root)
    dimensions_b, average_b = average_images(second_half, evidence_root)
    if dimensions_a != dimensions_b:
        raise EvidenceError(f"{scene_key}: split reference dimensions differ")
    split_rmse = image_rmse(average_a, average_b)
    full_dimensions, full_average = average_images(references, evidence_root)

    candidate_means = {}
    for sampler in SAMPLERS:
        rmses = []
        for seed in QUALITY_SEEDS:
            candidate = unique_record(
                records,
                "candidate",
                scene=scene_key,
                seed=seed,
                sampler=sampler,
                quality_samples=256,
            )
            metrics = image_metrics(
                candidate, evidence_root, full_dimensions, full_average
            )
            rmses.append(metrics["rmse"])
        candidate_means[sampler] = math.fsum(rmses) / len(rmses)
    candidate_scale = min(candidate_means.values())
    threshold = REFERENCE_PRECISION_FRACTION * candidate_scale
    adequate = split_rmse <= threshold
    attempt = {
        "samples_per_seed": samples_per_seed,
        "seeds": list(REFERENCE_SEEDS),
        "first_half_seeds": list(REFERENCE_SEEDS[:2]),
        "second_half_seeds": list(REFERENCE_SEEDS[2:]),
        "split_reference_rmse": split_rmse,
        "candidate_256_mean_rmse_by_sampler": candidate_means,
        "candidate_rmse_scale": candidate_scale,
        "maximum_allowed_split_rmse": threshold,
        "precision_fraction": REFERENCE_PRECISION_FRACTION,
        "adequate": adequate,
        "criterion": (
            "split_reference_rmse <= 0.25 * min(mean(candidate RMSE at 256 "
            "spp, counter), mean(candidate RMSE at 256 spp, halton))"
        ),
        "reference_run_ids": [record["run_id"] for record in references],
    }
    return attempt, full_dimensions, full_average, references


def paired_intensity_check(
    counter_means: Sequence[float], halton_means: Sequence[float]
) -> dict[str, Any]:
    if len(counter_means) != len(halton_means) or len(counter_means) < 2:
        raise EvidenceError("paired intensity test needs equal paired samples")
    differences = [
        halton - counter for counter, halton in zip(counter_means, halton_means)
    ]
    counter_mean = math.fsum(counter_means) / len(counter_means)
    halton_mean = math.fsum(halton_means) / len(halton_means)
    mean_difference, difference_variance = mean_and_sample_variance(differences)
    paired_standard_error = math.sqrt(difference_variance / len(differences))
    allowed = max(4.0 * paired_standard_error, 0.01 * counter_mean) + 1e-6
    absolute = abs(mean_difference)
    return {
        "counter_mean": counter_mean,
        "halton_mean": halton_mean,
        "paired_differences_halton_minus_counter": differences,
        "paired_mean_difference": mean_difference,
        "paired_sample_variance": difference_variance,
        "paired_standard_error": paired_standard_error,
        "absolute_mean_difference": absolute,
        "maximum_allowed_difference": allowed,
        "passed": absolute <= allowed,
        "criterion": (
            "abs(mean(halton_linear_mean - counter_linear_mean)) <= "
            "max(4 * paired_standard_error, 0.01 * counter_mean) + 1e-6"
        ),
    }


def candidate_summary(
    records: Sequence[dict[str, Any]],
    evidence_root: Path,
    scene_key: str,
    samples: int,
    reference_dimensions: tuple[int, int],
    reference: Sequence[float],
) -> tuple[dict[str, Any], list[float], list[float]]:
    per_seed = []
    mse_by_sampler: dict[str, list[float]] = {sampler: [] for sampler in SAMPLERS}
    rmse_by_sampler: dict[str, list[float]] = {sampler: [] for sampler in SAMPLERS}
    mean_by_sampler: dict[str, list[float]] = {sampler: [] for sampler in SAMPLERS}
    for seed in QUALITY_SEEDS:
        entry: dict[str, Any] = {"seed": seed}
        for sampler in SAMPLERS:
            record = unique_record(
                records,
                "candidate",
                scene=scene_key,
                seed=seed,
                sampler=sampler,
                quality_samples=samples,
            )
            metrics = image_metrics(
                record, evidence_root, reference_dimensions, reference
            )
            mse_by_sampler[sampler].append(metrics["mse"])
            rmse_by_sampler[sampler].append(metrics["rmse"])
            mean_by_sampler[sampler].append(metrics["linear_rgb_mean"])
            entry[sampler] = {
                **metrics,
                "completed_samples": record["actual"]["completed_samples"],
                "run_id": record["run_id"],
            }
        per_seed.append(entry)

    sampler_summaries = {}
    for sampler in SAMPLERS:
        mse_description = describe(mse_by_sampler[sampler])
        sampler_summaries[sampler] = {
            "mean_mse": mse_description["mean"],
            "sample_variance_mse": mse_description["sample_variance"],
            "sample_standard_deviation_mse": mse_description[
                "sample_standard_deviation"
            ],
            "min_mse": mse_description["min"],
            "max_mse": mse_description["max"],
            "mean_rmse": math.fsum(rmse_by_sampler[sampler]) / len(
                rmse_by_sampler[sampler]
            ),
            "mean_linear_rgb": math.fsum(mean_by_sampler[sampler]) / len(
                mean_by_sampler[sampler]
            ),
        }
    counter_mse = sampler_summaries["counter"]["mean_mse"]
    halton_mse = sampler_summaries["halton"]["mean_mse"]
    reduction = (
        (counter_mse - halton_mse) / counter_mse
        if counter_mse > 0.0
        else (0.0 if halton_mse == 0.0 else -math.inf)
    )
    if not math.isfinite(reduction):
        raise EvidenceError(f"{scene_key}/{samples}: undefined MSE reduction")
    intensity = paired_intensity_check(
        mean_by_sampler["counter"], mean_by_sampler["halton"]
    )
    return ({
        "scene": scene_key,
        "samples": samples,
        "seeds": list(QUALITY_SEEDS),
        "counter": sampler_summaries["counter"],
        "halton": sampler_summaries["halton"],
        "relative_mse_reduction": reduction,
        "relative_mse_reduction_formula": "(counter_mean_mse - halton_mean_mse) / counter_mean_mse",
        "strict_improvement": halton_mse < counter_mse,
        "intensity_paired": intensity,
        "per_seed": per_seed,
    }, mse_by_sampler["counter"], mse_by_sampler["halton"])


def metrics_from_actual(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    field_names = (
        "renderer_gpu_ms",
        "render_loop_wall_ms",
        "process_elapsed_ms",
        "renderer_gpu_ms_per_completed_sample",
        "render_loop_wall_ms_per_completed_sample",
        "process_elapsed_ms_per_completed_sample",
        "completed_samples",
    )
    return {
        name: describe([record["actual"][name] for record in records])
        for name in field_names
    }


def analyze_equal_time(
    records: Sequence[dict[str, Any]],
    evidence_root: Path,
    references: dict[str, tuple[tuple[int, int], list[float], bool]],
) -> dict[str, Any]:
    individual = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for scene_key, _ in QUALITY_SCENES:
        dimensions, reference, adequate = references[scene_key]
        for seed in QUALITY_SEEDS:
            for sampler in SAMPLERS:
                record = unique_record(
                    records,
                    "equal-time",
                    scene=scene_key,
                    seed=seed,
                    sampler=sampler,
                )
                metrics = image_metrics(record, evidence_root, dimensions, reference)
                actual = record["actual"]
                entry = {
                    "run_id": record["run_id"],
                    "scene": scene_key,
                    "sampler": sampler,
                    "seed": seed,
                    "reference_adequate": adequate,
                    "completed_samples": actual["completed_samples"],
                    **metrics,
                    "renderer_gpu_ms": actual["renderer_gpu_ms"],
                    "render_loop_wall_ms": actual["render_loop_wall_ms"],
                    "process_elapsed_ms": actual["process_elapsed_ms"],
                    "non_render_process_overhead_ms": actual[
                        "non_render_process_overhead_ms"
                    ],
                    "renderer_gpu_ms_per_completed_sample": actual[
                        "renderer_gpu_ms_per_completed_sample"
                    ],
                    "render_loop_wall_ms_per_completed_sample": actual[
                        "render_loop_wall_ms_per_completed_sample"
                    ],
                    "process_elapsed_ms_per_completed_sample": actual[
                        "process_elapsed_ms_per_completed_sample"
                    ],
                    "render_loop_minus_budget_ms": actual[
                        "render_loop_minus_budget_ms"
                    ],
                    "time_budget_overshoot_ms": actual[
                        "time_budget_overshoot_ms"
                    ],
                    "stop_reason": actual["run_policy"]["stop_reason"],
                    "last_sample_wall_ms": actual["last_sample_wall_ms"],
                }
                individual.append(entry)
                grouped.setdefault((scene_key, sampler), []).append(entry)

    summaries = []
    for (scene_key, sampler), entries in grouped.items():
        source_records = [
            unique_record(
                records,
                "equal-time",
                scene=scene_key,
                seed=entry["seed"],
                sampler=sampler,
            )
            for entry in entries
        ]
        timing_metrics = metrics_from_actual(source_records)
        timing_metrics.update({
            "mse": describe([entry["mse"] for entry in entries]),
            "rmse": describe([entry["rmse"] for entry in entries]),
            "linear_rgb_mean": describe(
                [entry["linear_rgb_mean"] for entry in entries]
            ),
            "non_render_process_overhead_ms": describe(
                [entry["non_render_process_overhead_ms"] for entry in entries]
            ),
            "render_loop_minus_budget_ms": describe(
                [entry["render_loop_minus_budget_ms"] for entry in entries]
            ),
            "time_budget_overshoot_ms": describe(
                [entry["time_budget_overshoot_ms"] for entry in entries]
            ),
            "last_sample_wall_ms": describe(
                [entry["last_sample_wall_ms"] for entry in entries]
            ),
        })
        summaries.append({
            "scene": scene_key,
            "sampler": sampler,
            "seeds": [entry["seed"] for entry in entries],
            "reference_adequate": all(entry["reference_adequate"] for entry in entries),
            "completed_sample_counts": [
                entry["completed_samples"] for entry in entries
            ],
            "metrics": timing_metrics,
        })
    return {
        "budget_ms": EQUAL_TIME_BUDGET_MS,
        "target_samples": EQUAL_TIME_TARGET_SAMPLES,
        "run_count": len(individual),
        "runs": individual,
        "summaries": summaries,
        "process_overhead_definition": (
            "process_elapsed_ms - render_loop_wall_ms; non-render process overhead "
            "includes startup, scene/asset loading, upload, download, image/raw/metadata "
            "output, and process teardown; it is not labeled renderer setup time"
        ),
        "time_budget_note": (
            "the budget is checked between whole samples; no upper-bound assertion "
            "excludes one-sample overshoot"
        ),
    }


def analyze_quality(
    records: Sequence[dict[str, Any]], evidence_root: Path
) -> dict[str, Any]:
    precision = []
    selected_references: dict[
        str, tuple[tuple[int, int], list[float], bool]
    ] = {}
    for scene_key, _ in QUALITY_SCENES:
        attempts = []
        initial, dimensions, average, _ = reference_attempt(
            records, evidence_root, scene_key, REFERENCE_SAMPLES
        )
        attempts.append(initial)
        selected_samples = REFERENCE_SAMPLES
        retry_records = matching_records(
            records,
            "reference",
            scene=scene_key,
            reference_samples_per_seed=REFERENCE_RETRY_SAMPLES,
        )
        if not initial["adequate"] and not retry_records:
            raise EvidenceError(
                f"{scene_key}: inadequate 8192-spp references require the fixed 16384-spp retry"
            )
        if initial["adequate"] and retry_records:
            raise EvidenceError(
                f"{scene_key}: retry evidence exists despite an adequate initial reference"
            )
        if retry_records:
            if len(retry_records) != len(REFERENCE_SEEDS):
                raise EvidenceError(
                    f"{scene_key}: partial 16384-spp reference retry is not analyzable"
                )
            retry, dimensions, average, _ = reference_attempt(
                records, evidence_root, scene_key, REFERENCE_RETRY_SAMPLES
            )
            attempts.append(retry)
            selected_samples = REFERENCE_RETRY_SAMPLES
        selected_adequate = attempts[-1]["adequate"]
        precision.append({
            "scene": scene_key,
            "attempts": attempts,
            "retry_performed": len(attempts) == 2,
            "selected_samples_per_seed": selected_samples,
            "adequate": selected_adequate,
        })
        selected_references[scene_key] = (
            dimensions,
            average,
            selected_adequate,
        )

    convergence = []
    pooled_counter = []
    pooled_halton = []
    for scene_key, _ in QUALITY_SCENES:
        dimensions, reference, _ = selected_references[scene_key]
        for samples in QUALITY_SAMPLE_COUNTS:
            summary, counter_mse, halton_mse = candidate_summary(
                records,
                evidence_root,
                scene_key,
                samples,
                dimensions,
                reference,
            )
            convergence.append(summary)
            pooled_counter.extend(counter_mse)
            pooled_halton.extend(halton_mse)

    expected_per_sampler = (
        len(QUALITY_SCENES) * len(QUALITY_SAMPLE_COUNTS) * len(QUALITY_SEEDS)
    )
    if len(pooled_counter) != expected_per_sampler or len(pooled_halton) != expected_per_sampler:
        raise EvidenceError("candidate pooling did not include the complete fixed design")
    pooled_counter_mse = math.fsum(pooled_counter) / len(pooled_counter)
    pooled_halton_mse = math.fsum(pooled_halton) / len(pooled_halton)
    if pooled_counter_mse <= 0.0:
        raise EvidenceError("pooled counter MSE is not positive")
    pooled_reduction = (
        pooled_counter_mse - pooled_halton_mse
    ) / pooled_counter_mse
    improved = sum(entry["strict_improvement"] for entry in convergence)
    intensity_passed = sum(entry["intensity_paired"]["passed"] for entry in convergence)
    adequate_references = all(entry["adequate"] for entry in precision)
    criteria = {
        "adequate_references": adequate_references,
        "pooled_mse_reduction_at_least_10_percent": pooled_reduction >= 0.10,
        "at_least_4_of_6_scene_budget_improvements": improved >= 4,
        "all_6_intensity_criteria_pass": intensity_passed == len(convergence),
    }
    if not adequate_references:
        qualification_status = "inconclusive"
    elif all(criteria.values()):
        qualification_status = "qualified"
    else:
        qualification_status = "not-qualified"
    failed_criteria = [name for name, passed in criteria.items() if not passed]

    equal_time = analyze_equal_time(records, evidence_root, selected_references)
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "quality",
        "status": "completed",
        "completed_utc": utc_now(),
        "run_count": sum(record.get("phase") == "quality" for record in records),
        "reference_precision": precision,
        "convergence": convergence,
        "pooled_mse": {
            "counter": pooled_counter_mse,
            "halton": pooled_halton_mse,
            "relative_reduction": pooled_reduction,
            "formula": "(counter_pooled_mse - halton_pooled_mse) / counter_pooled_mse",
            "matched_pairs": expected_per_sampler,
            "candidate_renders": 2 * expected_per_sampler,
            "pooling": (
                "arithmetic mean of per-render linear-RGB per-channel pixel MSE; "
                "all fixed candidate images have identical dimensions"
            ),
        },
        "comparisons": {
            "strict_halton_mse_improvements": improved,
            "total": len(convergence),
            "criterion": "halton mean MSE is strictly less than counter mean MSE",
        },
        "intensity_criteria": {
            "passed": intensity_passed,
            "total": len(convergence),
            "all_passed": intensity_passed == len(convergence),
        },
        "qualification": {
            "status": qualification_status,
            "criteria": criteria,
            "failed_criteria": failed_criteria,
            "required_pooled_relative_reduction": 0.10,
            "required_strict_improvements": 4,
            "required_intensity_checks": len(convergence),
            "note": (
                "inadequate references force an inconclusive result regardless "
                "of the observed candidate ordering"
            ),
        },
        "equal_time": equal_time,
    }


def analyze_timing(
    records: Sequence[dict[str, Any]], motion_pair: dict[str, Any]
) -> dict[str, Any]:
    configurations = [
        ("static", scene_key, sampler, scene_key)
        for scene_key, _ in QUALITY_SCENES
        for sampler in SAMPLERS
    ]
    configurations.extend((
        ("motion", "motion-midpoint", "halton", "midpoint"),
        ("motion", "motion-full", "halton", "full-shutter"),
    ))
    groups = []
    for kind, scene_key, sampler, comparison_role in configurations:
        warmups = matching_records(
            records,
            "timing",
            scene=scene_key,
            sampler=sampler,
            measurement_phase="warmup",
        )
        measured = matching_records(
            records,
            "timing",
            scene=scene_key,
            sampler=sampler,
            measurement_phase="measured",
        )
        warmups.sort(key=lambda record: record["design"]["repeat"])
        measured.sort(key=lambda record: record["design"]["repeat"])
        if len(warmups) != TIMING_WARMUPS or len(measured) != TIMING_MEASURED:
            raise EvidenceError(
                f"timing group {scene_key}/{sampler} has "
                f"{len(warmups)} warmups and {len(measured)} measured runs"
            )
        identities = [record["actual"]["identity"] for record in measured]
        groups.append({
            "group_id": f"{scene_key}-{sampler}",
            "kind": kind,
            "scene": scene_key,
            "sampler": sampler,
            "comparison_role": comparison_role,
            "samples": TIMING_SAMPLES,
            "seed": TIMING_SEED,
            "warmup_run_ids": [record["run_id"] for record in warmups],
            "measured_run_ids": [record["run_id"] for record in measured],
            "warmups_excluded_from_summary": True,
            "metrics": metrics_from_actual(measured),
            "metadata_identities": identities,
        })

    by_id = {group["group_id"]: group for group in groups}
    midpoint = by_id["motion-midpoint-halton"]
    full = by_id["motion-full-halton"]
    ratio_fields = (
        "renderer_gpu_ms",
        "render_loop_wall_ms",
        "process_elapsed_ms",
        "renderer_gpu_ms_per_completed_sample",
        "render_loop_wall_ms_per_completed_sample",
        "process_elapsed_ms_per_completed_sample",
    )
    ratios = {}
    for name in ratio_fields:
        denominator = midpoint["metrics"][name]["median"]
        ratios[name + "_full_over_midpoint"] = (
            full["metrics"][name]["median"] / denominator
            if denominator > 0.0
            else None
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "timing",
        "status": "completed",
        "completed_utc": utc_now(),
        "run_count": sum(record.get("phase") == "timing" for record in records),
        "warmups_per_configuration": TIMING_WARMUPS,
        "measured_repeats_per_configuration": TIMING_MEASURED,
        "summary_statistic": "median with minimum and maximum of exactly three measured repeats",
        "groups": groups,
        "motion_cost": {
            "sampler": "halton",
            "midpoint_group_id": midpoint["group_id"],
            "full_shutter_group_id": full["group_id"],
            "midpoint": midpoint,
            "full_shutter": full,
            "median_ratios": ratios,
            "swept_geometry": motion_pair,
            "note": (
                "the authored scene designs are identical after excluding only "
                "Camera.FILE and Camera.SHUTTER; renderer geometry fingerprints may "
                "still differ because shutter is intentionally part of parsed scene identity"
            ),
        },
    }


def analyze_gallery(
    records: Sequence[dict[str, Any]], evidence_root: Path
) -> dict[str, Any]:
    gallery_records = matching_records(records, "gallery")
    if len(gallery_records) != len(gallery_specs()):
        raise EvidenceError(
            f"gallery requires {len(gallery_specs())} renders, found {len(gallery_records)}"
        )
    items = []
    for record in gallery_records:
        design = record["design"]
        png_path = artifact_path(record, "png", evidence_root)
        raw_path = artifact_path(record, "raw", evidence_root)
        metadata_path = artifact_path(record, "metadata", evidence_root)
        benchmark_path = artifact_path(record, "benchmark_csv", evidence_root)
        items.append({
            "run_id": record["run_id"],
            "label": record["label"],
            "kind": design["kind"],
            "gallery_role": design["gallery_role"],
            "scene": design["scene"],
            "scene_file": design["scene_file"],
            "sampler": design["sampler"],
            "seed": design["seed"],
            "samples": record["actual"]["completed_samples"],
            "resolution": record["actual"]["resolution"],
            "scheduling_overrides": design["scheduling_overrides"],
            "effective_scheduling": design["effective_scheduling"],
            "actual_config": record["actual"]["config"],
            "linear_rgb_mean": record["actual"]["linear_rgb_mean"],
            "gallery_pair_id": design.get("gallery_pair_id"),
            "gallery_pair_semantics": design.get("gallery_pair_semantics"),
            "gallery_exactness_requirement": design.get(
                "gallery_exactness_requirement"
            ),
            "files": {
                "png": str(png_path),
                "raw": str(raw_path),
                "metadata": str(metadata_path),
                "benchmark_csv": str(benchmark_path),
            },
            "hashes": {
                name: record["artifacts"][name]["sha256"]
                for name in ("png", "raw", "metadata", "benchmark_csv")
            },
            "camera": record["actual"]["camera"],
            "identity": record["actual"]["identity"],
        })
    items.sort(key=lambda item: item["run_id"])

    pair_results = []
    for pair_id in (
        "obj-capability",
        "bvh-schedule",
        "wavefront-schedule",
        "rr-illustration",
    ):
        members = [item for item in items if item["gallery_pair_id"] == pair_id]
        if len(members) != 2:
            raise EvidenceError(
                f"gallery pair {pair_id} requires two members, found {len(members)}"
            )
        members.sort(key=lambda item: item["gallery_role"])
        semantics = {item["gallery_pair_semantics"] for item in members}
        requirements = {item["gallery_exactness_requirement"] for item in members}
        if len(semantics) != 1 or len(requirements) != 1:
            raise EvidenceError(f"gallery pair {pair_id} has inconsistent tags")
        requirement = next(iter(requirements))
        raw_hashes = {
            item["gallery_role"]: item["hashes"]["raw"] for item in members
        }
        png_hashes = {
            item["gallery_role"]: item["hashes"]["png"] for item in members
        }
        raw_hash_exact = len(set(raw_hashes.values())) == 1
        png_hash_exact = len(set(png_hashes.values())) == 1
        raw_exact = files_byte_exact(
            Path(members[0]["files"]["raw"]),
            Path(members[1]["files"]["raw"]),
        )
        png_exact = files_byte_exact(
            Path(members[0]["files"]["png"]),
            Path(members[1]["files"]["png"]),
        )
        if raw_exact != raw_hash_exact or png_exact != png_hash_exact:
            raise EvidenceError(
                f"gallery pair {pair_id} byte comparison disagrees with SHA-256"
            )
        exactness_required = requirement == "raw-and-png-required"
        if requirement not in ("raw-and-png-required", "not-asserted"):
            raise EvidenceError(
                f"gallery pair {pair_id} has unknown exactness requirement {requirement!r}"
            )
        if exactness_required and not (raw_exact and png_exact):
            raise EvidenceError(
                f"gallery pair {pair_id} violated exact raw/PNG equivalence: "
                f"raw={raw_hashes}, png={png_hashes}"
            )
        role_order = [item["gallery_role"] for item in members]
        means = {
            item["gallery_role"]: item["linear_rgb_mean"] for item in members
        }
        pair = {
            "pair_id": pair_id,
            "semantics": next(iter(semantics)),
            "role_order": role_order,
            "members": [
                {
                    "role": item["gallery_role"],
                    "run_id": item["run_id"],
                    "label": item["label"],
                    "scene": item["scene"],
                    "scene_file": item["scene_file"],
                    "scheduling_overrides": item["scheduling_overrides"],
                    "effective_scheduling": item["effective_scheduling"],
                    "raw_sha256": item["hashes"]["raw"],
                    "png_sha256": item["hashes"]["png"],
                    "linear_rgb_mean": item["linear_rgb_mean"],
                }
                for item in members
            ],
            "raw_sha256_by_role": raw_hashes,
            "png_sha256_by_role": png_hashes,
            "raw_byte_exact_observed": raw_exact,
            "png_byte_exact_observed": png_exact,
            "raw_exact_observed": raw_exact,
            "png_exact_observed": png_exact,
            "raw_sha256_equal_observed": raw_hash_exact,
            "png_sha256_equal_observed": png_hash_exact,
            "exactness_requirement": requirement,
            "exactness_asserted": exactness_required,
            "exactness_assertion_passed": (
                raw_exact and png_exact if exactness_required else None
            ),
            "linear_rgb_mean_by_role": means,
            "linear_rgb_mean_second_minus_first": (
                means[role_order[1]] - means[role_order[0]]
            ),
        }
        if pair_id == "obj-capability":
            pair["interpretation"] = (
                "Capability illustration using intentionally different scene content; "
                "it is not an isolated performance comparison and exact equality is "
                "neither expected nor asserted."
            )
            pair["members_intentionally_non_equivalent"] = True
        elif pair_id == "rr-illustration":
            pair["interpretation"] = (
                "Single-image RR illustration using intentionally different RR "
                "settings; exact equality is not asserted and this pair does not "
                "replace the fixed eight-seed statistical RR validation."
            )
            pair["members_intentionally_non_equivalent"] = True
            pair["replaces_statistical_rr_validation"] = False
        else:
            pair["interpretation"] = (
                "Scheduling-equivalence check; both the accumulated linear sum and "
                "the rendered PNG are required to be byte-exact."
            )
            pair["members_intentionally_non_equivalent"] = False
        pair_results.append(pair)

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "gallery",
        "status": "completed",
        "completed_utc": utc_now(),
        "run_count": len(items),
        "items": items,
        "static_render_count": sum(item["kind"] == "static" for item in items),
        "motion_render_count": sum(item["kind"] == "motion" for item in items),
        "comparison_render_count": sum(
            item["gallery_pair_id"] is not None for item in items
        ),
        "pair_results": pair_results,
        "required_exact_pairs_passed": all(
            pair["exactness_assertion_passed"] is True
            for pair in pair_results
            if pair["exactness_asserted"]
        ),
    }


def observed_identities(
    records: Sequence[dict[str, Any]],
    input_scenes: dict[str, Any],
    initial_binary: dict[str, Any],
) -> dict[str, Any]:
    successful = [record for record in records if record.get("status") == "passed"]
    builds = set()
    sources = set()
    devices = set()
    binary_hashes = set()
    per_scene: dict[str, dict[str, Any]] = {}
    binary_matches_initial = True
    scene_inputs_match_initial = True
    for record in successful:
        identity = record["actual"]["identity"]
        compatibility = identity["compatibility"]
        builds.add(compatibility["build_identifier"])
        sources.add(compatibility["baseline_starter_revision"])
        devices.add(compatibility.get("device", ""))
        binary_record = record.get("binary_identity", {})
        before = binary_record.get("before")
        after = binary_record.get("after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            binary_matches_initial = False
        else:
            binary_hashes.update((before.get("sha256"), after.get("sha256")))
            binary_matches_initial = binary_matches_initial and same_binary_identity(
                before, after
            ) and same_binary_identity(before, initial_binary)

        scene_file = record["design"]["scene_file"]
        expected_scene = initial_scene_identity(input_scenes, scene_file)
        scene_record = record.get("scene_input_identity", {})
        scene_before = scene_record.get("before")
        scene_after = scene_record.get("after")
        scene_inputs_match_initial = scene_inputs_match_initial and (
            isinstance(scene_before, dict)
            and scene_before == scene_after
            and scene_before == expected_scene
        )
        fingerprints = identity["fingerprints"]
        entry = per_scene.setdefault(scene_file, {
            "initial_scene_sha256": expected_scene["scene_sha256"],
            "initial_input_manifest_sha256": expected_scene["manifest_sha256"],
            "observed_scene_sha256": set(),
            "observed_input_manifest_sha256": set(),
            "metadata_scene_file_fingerprints": set(),
            "metadata_asset_fingerprints": set(),
            "metadata_material_fingerprints": set(),
            "metadata_geometry_fingerprints": set(),
            "metadata_render_config_fingerprints": set(),
        })
        if isinstance(scene_before, dict):
            entry["observed_scene_sha256"].add(scene_before.get("scene_sha256"))
            entry["observed_input_manifest_sha256"].add(
                scene_before.get("manifest_sha256")
            )
        entry["metadata_scene_file_fingerprints"].add(fingerprints["scene_file"])
        entry["metadata_asset_fingerprints"].add(fingerprints["assets"])
        entry["metadata_material_fingerprints"].add(fingerprints["materials"])
        entry["metadata_geometry_fingerprints"].add(fingerprints["geometry"])
        entry["metadata_render_config_fingerprints"].add(
            fingerprints["render_config"]
        )

    scene_fingerprint_consistent = True
    serialized_scenes = {}
    for scene_file, entry in per_scene.items():
        stable_keys = (
            "observed_scene_sha256",
            "observed_input_manifest_sha256",
            "metadata_scene_file_fingerprints",
            "metadata_asset_fingerprints",
            "metadata_material_fingerprints",
            "metadata_geometry_fingerprints",
        )
        if any(len(entry[key]) != 1 for key in stable_keys):
            scene_fingerprint_consistent = False
        if entry["observed_scene_sha256"] != {entry["initial_scene_sha256"]}:
            scene_fingerprint_consistent = False
        if entry["observed_input_manifest_sha256"] != {
            entry["initial_input_manifest_sha256"]
        }:
            scene_fingerprint_consistent = False
        serialized_scenes[scene_file] = {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in entry.items()
        }
    checks = {
        "has_successful_runs": bool(successful),
        "single_binary": len(binary_hashes) == 1,
        "binary_matches_initial_before_and_after_every_run": binary_matches_initial,
        "single_build": len(builds) == 1,
        "single_baseline_source_revision": len(sources) == 1,
        "single_nonempty_device": len(devices) == 1 and "" not in devices,
        "scene_inputs_match_initial_manifest": scene_inputs_match_initial,
        "per_scene_metadata_fingerprints_consistent": scene_fingerprint_consistent,
    }
    return {
        "successful_run_count": len(successful),
        "binary_sha256_values": sorted(
            value for value in binary_hashes if isinstance(value, str)
        ),
        "build_identifiers": sorted(builds),
        "baseline_source_revisions": sorted(sources),
        "devices": sorted(devices),
        "per_scene": serialized_scenes,
        "checks": checks,
        "passed": all(checks.values()),
    }


def phase_scene_files(phase: str) -> set[str]:
    result: set[str] = set()
    if phase in ("quality", "all"):
        result.update(filename for _, filename in QUALITY_SCENES)
    if phase in ("timing", "all"):
        result.update(filename for _, filename in QUALITY_SCENES)
        result.update(MOTION_SCENES.values())
    if phase in ("gallery", "all"):
        result.update(spec.scene_file for spec in gallery_specs())
    return result


def run_quality(session: MeasurementSession) -> None:
    session.prepare_phase("quality")
    for spec in quality_candidate_specs():
        session.run(spec)
    for spec in reference_specs(REFERENCE_SAMPLES):
        session.run(spec)

    initial_precision = []
    failed_scenes = []
    for scene_key, _ in QUALITY_SCENES:
        attempt, _, _, _ = reference_attempt(
            session.records, session.output, scene_key, REFERENCE_SAMPLES
        )
        initial_precision.append({"scene": scene_key, "attempts": [attempt]})
        if not attempt["adequate"]:
            failed_scenes.append(scene_key)
    session.state["phases"]["quality"]["reference_precision"] = initial_precision
    session.state["phases"]["quality"]["reference_retry_scenes"] = failed_scenes
    session.persist()

    if failed_scenes:
        for spec in reference_specs(REFERENCE_RETRY_SAMPLES, failed_scenes):
            session.run(spec)
    for spec in equal_time_specs():
        session.run(spec)
    started = session.state["phases"]["quality"].get("started_utc")
    summary = analyze_quality(session.records, session.output)
    summary["started_utc"] = started
    session.state["phases"]["quality"] = summary
    session.persist()


def run_timing(session: MeasurementSession, motion_pair: dict[str, Any]) -> None:
    session.prepare_phase("timing")
    for spec in timing_specs():
        session.run(spec)
    started = session.state["phases"]["timing"].get("started_utc")
    summary = analyze_timing(session.records, motion_pair)
    summary["started_utc"] = started
    session.state["phases"]["timing"] = summary
    session.persist()


def run_gallery(session: MeasurementSession) -> None:
    session.prepare_phase("gallery")
    for spec in gallery_specs():
        session.run(spec)
    started = session.state["phases"]["gallery"].get("started_utc")
    summary = analyze_gallery(session.records, session.output)
    summary["started_utc"] = started
    session.state["phases"]["gallery"] = summary
    session.persist()


def selected_phases(phase: str) -> tuple[str, ...]:
    return ("quality", "timing", "gallery") if phase == "all" else (phase,)


def final_gate_status(state: dict[str, Any]) -> str:
    quality = state.get("phases", {}).get("quality")
    if quality and quality.get("status") == "completed":
        qualification = quality.get("qualification", {}).get("status")
        if qualification == "not-qualified":
            return "not-qualified"
        if qualification == "inconclusive":
            return "inconclusive"
    return "passed"


def run_measurement(args: argparse.Namespace, output: Path) -> int:
    session = MeasurementSession(output, args.binary, args.phase)
    session.persist()
    try:
        session.initialize_identity()
        session.state["input_scenes"] = inspect_scene_inputs(
            phase_scene_files(args.phase)
        )
        motion_pair = None
        if args.phase in ("timing", "gallery", "all"):
            motion_pair = verify_motion_variants()
            session.state["design"]["motion_variants"] = motion_pair
            if args.phase in ("timing", "all"):
                session.state["design"]["timing"]["motion_swept_geometry"] = motion_pair
        session.persist()

        for phase in selected_phases(args.phase):
            if phase == "quality":
                run_quality(session)
            elif phase == "timing":
                assert motion_pair is not None
                run_timing(session, motion_pair)
            elif phase == "gallery":
                run_gallery(session)
        observed = observed_identities(
            session.records,
            session.state["input_scenes"],
            session.initial_binary_identity,
        )
        if not observed["passed"]:
            raise EvidenceError(
                f"renderer/source identity changed within the measurement: {observed}"
            )
        session.state["identity"]["observed_metadata"] = observed
        session.state["execution_status"] = "completed"
        session.state["status"] = final_gate_status(session.state)
        session.state["completed_utc"] = utc_now()
        session.persist()
        print(output)
        # Code 2 reports a completed negative or inconclusive scientific outcome,
        # not an execution failure; the retained execution_status stays completed.
        # A passed outcome returns 0; execution/evidence failures use the branch below.
        return 2 if session.state["status"] in ("not-qualified", "inconclusive") else 0
    except BaseException as error:
        session.state["status"] = "failed"
        session.state["execution_status"] = "failed"
        session.state["completed_utc"] = utc_now()
        for phase_state in session.state["phases"].values():
            if phase_state.get("status") == "running":
                phase_state["status"] = "failed"
                phase_state["completed_utc"] = utc_now()
                phase_state["error"] = f"{type(error).__name__}: {error}"
        session.state["errors"].append({
            "time_utc": utc_now(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        })
        try:
            session.persist()
        except Exception as persist_error:
            print(f"also failed to persist final state: {persist_error}", file=sys.stderr)
        if isinstance(error, KeyboardInterrupt):
            print("measurement interrupted; partial evidence was retained", file=sys.stderr)
            return 130
        print(f"measurement failed: {error}", file=sys.stderr)
        return 1


def load_source_evidence(source: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = source.resolve()
    summary_path = source / "summary.json"
    runs_path = source / "runs.json"
    if not summary_path.is_file() or not runs_path.is_file():
        raise EvidenceError(f"analysis source lacks summary.json or runs.json: {source}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs_document = json.loads(runs_path.read_text(encoding="utf-8"))
    records = runs_document.get("runs")
    if not isinstance(records, list):
        raise EvidenceError(f"analysis source has an invalid runs.json: {runs_path}")
    return summary, records


def validate_analysis_source(
    summary: dict[str, Any],
    records: Sequence[dict[str, Any]],
    phases: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Require completed source evidence and return captured identity inputs."""
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceError("analysis source has an unsupported schema_version")
    if summary.get("execution_status") != "completed":
        raise EvidenceError("analysis source did not complete execution")
    phase_summaries = summary.get("phases")
    if not isinstance(phase_summaries, dict):
        raise EvidenceError("analysis source lacks phase summaries")
    for phase in phases:
        phase_summary = phase_summaries.get(phase)
        if not isinstance(phase_summary, dict) or phase_summary.get("status") != "completed":
            raise EvidenceError(f"analysis source phase {phase!r} is not completed")
        phase_records = [record for record in records if record.get("phase") == phase]
        if not phase_records:
            raise EvidenceError(f"analysis source contains no {phase} runs")
        failed = [
            record.get("label")
            for record in phase_records
            if record.get("status") != "passed"
        ]
        if failed:
            raise EvidenceError(
                f"analysis source phase {phase!r} contains failed runs: {failed}"
            )
        if phase_summary.get("run_count") != len(phase_records):
            raise EvidenceError(
                f"analysis source phase {phase!r} run count disagrees with runs.json"
            )

    identity = summary.get("identity")
    if not isinstance(identity, dict):
        raise EvidenceError("analysis source lacks renderer identity")
    initial_binary = identity.get("initial_binary_file_identity")
    if not isinstance(initial_binary, dict):
        raise EvidenceError("analysis source lacks the initial binary file identity")
    input_scenes = summary.get("input_scenes")
    if not isinstance(input_scenes, dict):
        raise EvidenceError("analysis source lacks the initial scene/asset manifest")

    design = summary.get("design")
    if not isinstance(design, dict):
        raise EvidenceError("analysis source lacks its fixed design")
    motion_evidence = None
    if any(phase in ("timing", "gallery") for phase in phases):
        motion_evidence = design.get("motion_variants")
        validate_motion_design_evidence(motion_evidence)
        if "timing" in phases:
            captured_timing = design.get("timing", {}).get("motion_swept_geometry")
            validate_motion_design_evidence(captured_timing)
            if captured_timing != motion_evidence:
                raise EvidenceError(
                    "top-level and timing captured motion design evidence disagree"
                )
    return initial_binary, input_scenes, motion_evidence


def run_analysis_only(args: argparse.Namespace, output: Path) -> int:
    source = args.analyze_only.resolve()
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": "scripts/measure_hedge.py",
        "mode": "analyze-only",
        "status": "running",
        "execution_status": "running",
        "phase_requested": args.phase,
        "started_utc": utc_now(),
        "completed_utc": None,
        "source_evidence": str(source),
        "phases": {},
        "errors": [],
    }
    atomic_json(output / "summary.json", state)
    try:
        source_summary_path = source / "summary.json"
        source_runs_path = source / "runs.json"
        source_summary_sha256 = sha256_file(source_summary_path)
        source_runs_sha256 = sha256_file(source_runs_path)
        source_summary, source_records = load_source_evidence(source)
        phases = selected_phases(args.phase)
        initial_binary, input_scenes, motion_evidence = validate_analysis_source(
            source_summary, source_records, phases
        )

        # Rebuild every selected successful run from the retained files.  This
        # deliberately happens before any summary calculation and does not use
        # cached record.actual timing, sample-count, or image-mean values.
        records = reaudit_records(source_records, source, phases)
        selected_records = [
            record for record in records if record.get("phase") in phases
        ]
        observed = observed_identities(
            selected_records, input_scenes, initial_binary
        )
        if not observed["passed"]:
            raise EvidenceError(
                "reconstructed renderer/source identity audit failed: "
                f"{observed}"
            )

        state["source_summary_sha256"] = source_summary_sha256
        state["source_runs_sha256"] = source_runs_sha256
        state["source_identity"] = source_summary.get("identity")
        state["design"] = source_summary["design"]
        state["reanalysis"] = {
            "selected_phases": list(phases),
            "successful_runs_reconstructed": sum(
                bool(record.get("reanalysis", {}).get(
                    "independently_reconstructed"
                ))
                for record in selected_records
            ),
            "cached_actual_values_used_for_analysis": False,
            "artifact_hashes_verified_before_parsing": True,
            "observed_identity": observed,
        }
        atomic_json(
            output / "runs.json",
            {
                "schema_version": SCHEMA_VERSION,
                "source_evidence": str(source),
                "source_runs_sha256": state["source_runs_sha256"],
                "runs": records,
            },
        )
        for phase in phases:
            phase_dir = output / phase
            phase_dir.mkdir(parents=False, exist_ok=False)
            state["phases"][phase] = {
                "schema_version": SCHEMA_VERSION,
                "phase": phase,
                "status": "running",
                "started_utc": utc_now(),
                "completed_utc": None,
            }
            analysis_started = state["phases"][phase]["started_utc"]
            atomic_json(output / "summary.json", state)
            if phase == "quality":
                summary = analyze_quality(records, source)
            elif phase == "timing":
                assert motion_evidence is not None
                summary = analyze_timing(records, motion_evidence)
            else:
                summary = analyze_gallery(records, source)
            summary["analysis_source"] = str(source)
            summary["started_utc"] = analysis_started
            state["phases"][phase] = summary
            atomic_json(phase_dir / "summary.json", summary)
            phase_records = [record for record in records if record.get("phase") == phase]
            atomic_json(
                phase_dir / "runs.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "source_evidence": str(source),
                    "runs": phase_records,
                },
            )
            atomic_json(output / "summary.json", state)
        if sha256_file(source_summary_path) != source_summary_sha256:
            raise EvidenceError("source summary.json changed during reanalysis")
        if sha256_file(source_runs_path) != source_runs_sha256:
            raise EvidenceError("source runs.json changed during reanalysis")
        state["execution_status"] = "completed"
        state["status"] = final_gate_status(state)
        state["completed_utc"] = utc_now()
        state["run_count"] = len(records)
        atomic_json(output / "summary.json", state)
        print(output)
        return 0
    except BaseException as error:
        state["status"] = "failed"
        state["execution_status"] = "failed"
        state["completed_utc"] = utc_now()
        for phase_state in state["phases"].values():
            if phase_state.get("status") == "running":
                phase_state["status"] = "failed"
                phase_state["completed_utc"] = utc_now()
                phase_state["error"] = f"{type(error).__name__}: {error}"
        state["errors"].append({
            "time_utc": utc_now(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        })
        atomic_json(output / "summary.json", state)
        for phase_name, phase_state in state["phases"].items():
            phase_dir = output / phase_name
            if phase_dir.is_dir():
                atomic_json(phase_dir / "summary.json", phase_state)
        if isinstance(error, KeyboardInterrupt):
            return 130
        print(f"analysis failed: {error}", file=sys.stderr)
        return 1


def run_self_test(output: Path) -> int:
    checks = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        if not condition:
            raise AssertionError(f"self-test failed: {name}: {detail}")
        checks.append({"name": name, "passed": True, "detail": detail})

    check("candidate-count", len(quality_candidate_specs()) == 96, 96)
    check("initial-reference-count", len(reference_specs(REFERENCE_SAMPLES)) == 8, 8)
    check(
        "maximum-reference-retry-count",
        len(reference_specs(REFERENCE_RETRY_SAMPLES)) == 8,
        8,
    )
    check("equal-time-count", len(equal_time_specs()) == 32, 32)
    check("quality-base-count", 96 + 8 + 32 == 136, 136)
    check("quality-maximum-count", 96 + 8 + 8 + 32 == 144, 144)
    check("timing-count", len(timing_specs()) == 36, 36)
    check("gallery-count", len(gallery_specs()) == 16, 16)
    check("all-base-count", 136 + 36 + 16 == 188, 188)
    check("all-maximum-count", 144 + 36 + 16 == 196, 196)

    every_spec = (
        quality_candidate_specs()
        + reference_specs(REFERENCE_SAMPLES)
        + reference_specs(REFERENCE_RETRY_SAMPLES)
        + equal_time_specs()
        + timing_specs()
        + gallery_specs()
    )
    identities = {(item.phase, item.label) for item in every_spec}
    prefixes = {item.relative_prefix for item in every_spec}
    check("all-labels-unique", len(identities) == len(every_spec), len(identities))
    check("all-prefixes-unique", len(prefixes) == len(every_spec), len(prefixes))
    check(
        "time-budget-only-equal-time",
        all(
            (item.time_budget_ms is not None) == (item.purpose == "equal-time")
            for item in every_spec
        ),
    )
    check(
        "all-fixed-scheduling-valid",
        all(
            "--benchmark" in command_for(
                Path("/synthetic/headless"), output / item.label, item
            )
            for item in every_spec
        ),
        len(every_spec),
    )
    check(
        "quality-timing-no-scheduling-overrides",
        all(
            not item.scheduling_overrides
            for item in every_spec
            if item.phase in ("quality", "timing")
        ),
    )
    gallery_pairs = {
        item.tags.get("gallery_pair_id"): item
        for item in gallery_specs()
        if item.tags.get("gallery_pair_id")
    }
    check(
        "gallery-pair-identifiers",
        set(gallery_pairs) == {
            "obj-capability",
            "bvh-schedule",
            "wavefront-schedule",
            "rr-illustration",
        },
        sorted(gallery_pairs),
    )
    bvh_commands = {
        item.tags["gallery_role"]: command_for(
            Path("/synthetic/headless"), output / item.label, item
        )
        for item in gallery_specs()
        if item.tags.get("gallery_pair_id") == "bvh-schedule"
    }
    check(
        "gallery-bvh-overrides",
        len(bvh_commands) == 2
        and bvh_commands["bvh-off"][
            bvh_commands["bvh-off"].index("--bvh") + 1
        ] == "off"
        and bvh_commands["bvh-on"][
            bvh_commands["bvh-on"].index("--bvh") + 1
        ] == "on",
        bvh_commands,
    )
    comparison_specs = {
        (item.tags.get("gallery_pair_id"), item.tags.get("gallery_role")): item
        for item in gallery_specs()
        if item.tags.get("gallery_pair_id")
    }
    check(
        "gallery-obj-capability-design",
        {
            comparison_specs[("obj-capability", role)].scene_file
            for role in ("baseline-without-obj", "with-obj")
        } == {"hedge-obj-baseline.json", "procedural-optical-stilllife.json"}
        and all(
            comparison_specs[("obj-capability", role)].resolution
            == GALLERY_COMPARISON_RESOLUTION
            and comparison_specs[("obj-capability", role)].samples
            == GALLERY_COMPARISON_SAMPLES
            and comparison_specs[("obj-capability", role)].sampler == "counter"
            and comparison_specs[("obj-capability", role)].tags[
                "gallery_exactness_requirement"
            ] == "not-asserted"
            for role in ("baseline-without-obj", "with-obj")
        ),
    )
    check(
        "gallery-wavefront-design",
        effective_scheduling(
            comparison_specs[("wavefront-schedule", "wavefront-off")]
        )["wavefront"] == "off"
        and effective_scheduling(
            comparison_specs[("wavefront-schedule", "wavefront-on")]
        )["wavefront"] == "on"
        and all(
            effective_scheduling(item)["rr"] == "on"
            and effective_scheduling(item)["sort"] == "off"
            for (pair_id, _), item in comparison_specs.items()
            if pair_id == "wavefront-schedule"
        ),
    )
    check(
        "gallery-rr-design",
        effective_scheduling(comparison_specs[("rr-illustration", "rr-off")])[
            "rr"
        ] == "off"
        and effective_scheduling(
            comparison_specs[("rr-illustration", "rr-on")]
        )["rr"] == "on"
        and all(
            effective_scheduling(item)["wavefront"] == "on"
            and effective_scheduling(item)["sort"] == "off"
            and item.tags["gallery_exactness_requirement"] == "not-asserted"
            for (pair_id, _), item in comparison_specs.items()
            if pair_id == "rr-illustration"
        ),
    )
    gallery_inputs = inspect_scene_inputs(phase_scene_files("gallery"))
    check(
        "gallery-scenes-and-assets-resolve",
        "hedge-obj-baseline.json" in gallery_inputs
        and all(
            Path(scene["path"]).is_file()
            and all(Path(asset["resolved_path"]).is_file() for asset in scene["referenced_assets"])
            for scene in gallery_inputs.values()
        ),
        sorted(gallery_inputs),
    )

    spec = equal_time_specs()[0]
    command = command_for(Path("/synthetic/headless"), output / "synthetic", spec)
    pairs = {}
    cursor = 2
    while cursor < len(command):
        option = command[cursor]
        if option == "--benchmark":
            pairs[option] = True
            cursor += 1
        else:
            if cursor + 1 >= len(command):
                raise AssertionError(f"self-test found an option without a value: {option}")
            pairs[option] = command[cursor + 1]
            cursor += 2
    check("command-benchmark", "--benchmark" in command, command)
    for key, expected in COMMON_SCHEDULING.items():
        check("command-" + key, pairs.get("--" + key) == expected, command)
    check("command-sampler", pairs.get("--sampler") == "counter", command)
    check(
        "command-time-budget",
        pairs.get("--time-budget-ms") == str(EQUAL_TIME_BUDGET_MS),
        command,
    )
    check(
        "command-target-samples",
        pairs.get("--samples") == str(EQUAL_TIME_TARGET_SAMPLES),
        command,
    )

    synthetic = output / "synthetic.sum.bin"
    width, height, samples = 2, 1, 4
    values = (4.0, 8.0, 12.0, 0.0, 4.0, 8.0)
    encoded = bytearray(struct.pack("<8sIIIIQ", b"C565SUM\0", 1, width, height, 3, samples))
    encoded.extend(struct.pack("<" + str(len(values)) + "f", *values))
    encoded.extend(struct.pack("<Q", fnv1a(encoded)))
    synthetic.write_bytes(encoded)
    parsed = read_sum(synthetic)
    check("authoritative-read-sum", parsed["samples"] == samples, parsed["samples"])
    check(
        "synthetic-linear-mean",
        math.isclose(
            math.fsum(parsed["values"]) / len(parsed["values"]) / samples,
            1.5,
        ),
        parsed["values"],
    )

    # Exercise the complete time-budget metadata/range audit with a clearly
    # labeled synthetic self-test fixture.  No production result is fabricated.
    audit_prefix = output / "synthetic-time-budget-audit"
    audit_spec = RenderSpec(
        phase="quality",
        purpose="equal-time",
        label="synthetic-time-budget-audit",
        relative_prefix="synthetic-time-budget-audit",
        scene_key="diffuse",
        scene_file=QUALITY_SCENES[0][1],
        sampler="counter",
        seed=947,
        samples=EQUAL_TIME_TARGET_SAMPLES,
        resolution=(width, height),
        depth=QUALITY_DEPTH,
        time_budget_ms=EQUAL_TIME_BUDGET_MS,
    )
    suffix(audit_prefix, ".sum.bin").write_bytes(encoded)
    suffix(audit_prefix, ".png").write_bytes(b"\x89PNG\r\n\x1a\n")
    benchmark_header = (
        "record_type,host_ms,gpu_ms,measured_samples,render_loop_wall_ms\n"
    )
    suffix(audit_prefix, ".benchmark.csv").write_text(
        benchmark_header + "bounce,,,,\n" + "overall,3,2,4,4005\n",
        encoding="utf-8",
    )
    synthetic_metadata = {
        "format_version": 1,
        "scene": str((ROOT / "scenes" / audit_spec.scene_file).resolve()),
        "resolution": [width, height],
        "target_samples": EQUAL_TIME_TARGET_SAMPLES,
        "completed_samples": samples,
        "seed": audit_spec.seed,
        "sampler_version": SAMPLER_IDS["counter"],
        "sampler": "counter",
        "sampler_id": SAMPLER_IDS["counter"],
        "config": {
            "max_depth": QUALITY_DEPTH,
            "aa": True,
            "nee": True,
            "bvh": True,
            "compact": True,
            "sort": False,
            "wavefront": True,
            "rr": True,
            "rr_start": 3,
            "sampler": "counter",
            "sampler_id": SAMPLER_IDS["counter"],
            "motion": True,
            "debug": "none",
            "depth_semantics": (
                "maximum surface intersections; final intersection is emission-only"
            ),
        },
        "camera": {
            "position": [0.0, 5.0, 10.5],
            "look_at": [0.0, 5.0, 0.0],
            "aperture": 0.0,
            "focus_distance": 10.5,
            "shutter": [0.0, 1.0],
        },
        "linear_sum_format": {
            "checksum": fnv1a(parsed["encoded"][:-8])
        },
        "fingerprints": {
            "scene_file": 1,
            "assets": 2,
            "materials": 3,
            "geometry": 4,
            "render_config": 5,
        },
        "compatibility": {
            "baseline_starter_revision": "synthetic-self-test",
            "build_identifier": "synthetic-self-test",
            "device": "synthetic-self-test-device",
        },
        "benchmark_enabled": True,
        "render_loop_wall_ms": 4005.0,
        "run_policy": {
            "time_budget_ms": EQUAL_TIME_BUDGET_MS,
            "stop_reason": "time_budget",
            "last_sample_wall_ms": 5.0,
        },
        "benchmark": {
            "measured_samples": samples,
            "renderer_host_ms": 3.0,
            "renderer_gpu_ms": 2.0,
        },
    }
    atomic_json(suffix(audit_prefix, ".json"), synthetic_metadata)
    self_binary_identity = binary_file_identity(Path(sys.executable))
    self_binary_identity_again = binary_file_identity(Path(sys.executable))
    check(
        "binary-identity-stable",
        same_binary_identity(self_binary_identity, self_binary_identity_again),
        self_binary_identity,
    )
    synthetic_scene_identity = inspect_scene_identity(audit_spec.scene_file)
    check(
        "scene-input-manifest-hash",
        synthetic_scene_identity["manifest_sha256"]
        == canonical_json_sha256({
            key: value
            for key, value in synthetic_scene_identity.items()
            if key != "manifest_sha256"
        }),
        synthetic_scene_identity,
    )
    synthetic_artifacts, synthetic_audit = audit_render(
        audit_prefix,
        audit_spec,
        4010.0,
        output,
        self_binary_identity,
        synthetic_scene_identity,
    )
    check("synthetic-metadata-audit", synthetic_audit["audit"]["passed"])
    check(
        "synthetic-time-budget-range",
        synthetic_audit["actual"]["completed_samples"] == samples
        and synthetic_audit["actual"]["time_budget_overshoot_ms"] == 5.0,
        synthetic_audit["actual"],
    )
    check(
        "synthetic-run-policy",
        synthetic_audit["actual"]["run_policy"] == synthetic_metadata["run_policy"]
        and synthetic_audit["actual"]["last_sample_wall_ms"] == 5.0,
        synthetic_audit["actual"]["run_policy"],
    )

    fixed_spec = RenderSpec(
        phase="quality",
        purpose="candidate",
        label="synthetic-fixed-sample-audit",
        relative_prefix="synthetic-time-budget-audit",
        scene_key="diffuse",
        scene_file=QUALITY_SCENES[0][1],
        sampler="counter",
        seed=947,
        samples=samples,
        resolution=(width, height),
        depth=QUALITY_DEPTH,
    )
    fixed_metadata = copy.deepcopy(synthetic_metadata)
    fixed_metadata["target_samples"] = samples
    fixed_metadata["run_policy"] = {
        "time_budget_ms": None,
        "stop_reason": "sample_target",
        "last_sample_wall_ms": 5.0,
    }
    atomic_json(suffix(audit_prefix, ".json"), fixed_metadata)
    _, fixed_audit = audit_render(
        audit_prefix,
        fixed_spec,
        4010.0,
        output,
        self_binary_identity,
        synthetic_scene_identity,
    )
    check(
        "synthetic-fixed-run-policy",
        fixed_audit["actual"]["run_policy"] == fixed_metadata["run_policy"],
        fixed_audit["actual"]["run_policy"],
    )

    ceiling_spec = RenderSpec(
        phase="quality",
        purpose="equal-time",
        label="synthetic-budget-reaches-target-audit",
        relative_prefix="synthetic-time-budget-audit",
        scene_key="diffuse",
        scene_file=QUALITY_SCENES[0][1],
        sampler="counter",
        seed=947,
        samples=samples,
        resolution=(width, height),
        depth=QUALITY_DEPTH,
        time_budget_ms=EQUAL_TIME_BUDGET_MS,
    )
    ceiling_metadata = copy.deepcopy(fixed_metadata)
    ceiling_metadata["run_policy"] = {
        "time_budget_ms": EQUAL_TIME_BUDGET_MS,
        "stop_reason": "sample_target",
        "last_sample_wall_ms": 5.0,
    }
    atomic_json(suffix(audit_prefix, ".json"), ceiling_metadata)
    _, ceiling_audit = audit_render(
        audit_prefix,
        ceiling_spec,
        4010.0,
        output,
        self_binary_identity,
        synthetic_scene_identity,
    )
    check(
        "synthetic-budget-target-stop-reason",
        ceiling_audit["actual"]["completed_samples"] == samples
        and ceiling_audit["actual"]["run_policy"]["stop_reason"]
        == "sample_target",
        ceiling_audit["actual"]["run_policy"],
    )
    atomic_json(suffix(audit_prefix, ".json"), synthetic_metadata)

    # Construct the same hashed sidecars as a completed production invocation,
    # then prove analyze-only reconstruction ignores cached timing as a source.
    stdout_path = suffix(audit_prefix, ".stdout.txt")
    stderr_path = suffix(audit_prefix, ".stderr.txt")
    invocation_path = suffix(audit_prefix, ".command.json")
    stdout_path.write_text("synthetic self-test stdout\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    synthetic_command = command_for(
        Path(sys.executable), audit_prefix, audit_spec
    )
    synthetic_record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": 1,
        "phase": audit_spec.phase,
        "purpose": audit_spec.purpose,
        "label": audit_spec.label,
        "status": "passed",
        "design": audit_spec.design_record(),
        "command": synthetic_command,
        "cwd": str(ROOT),
        "started_utc": utc_now(),
        "completed_utc": utc_now(),
        "exit_code": 0,
        "process_elapsed_ms": 4010.0,
        "invocation_envelope_elapsed_ms": 4011.0,
        "stdout": file_record(stdout_path, output),
        "stderr": file_record(stderr_path, output),
        "artifacts": synthetic_artifacts,
        "binary_identity": {
            "before": self_binary_identity,
            "after": self_binary_identity,
            "unchanged": True,
        },
        "scene_input_identity": {
            "before": synthetic_scene_identity,
            "after": synthetic_scene_identity,
            "unchanged": True,
        },
        **synthetic_audit,
    }
    invocation = copy.deepcopy(synthetic_record)
    invocation["pid"] = 0
    atomic_json(invocation_path, invocation)
    synthetic_record["invocation_record"] = file_record(invocation_path, output)
    reconstructed = reaudit_record(synthetic_record, output)
    check(
        "independent-artifact-reanalysis",
        reconstructed["reanalysis"]["independently_reconstructed"]
        and reconstructed["actual"] == synthetic_audit["actual"],
        reconstructed["reanalysis"],
    )

    invalid_metadata = copy.deepcopy(synthetic_metadata)
    invalid_metadata["fingerprints"]["geometry"] = 0
    atomic_json(suffix(audit_prefix, ".json"), invalid_metadata)
    rejected_zero_fingerprint = False
    try:
        audit_render(
            audit_prefix,
            audit_spec,
            4010.0,
            output,
            self_binary_identity,
            synthetic_scene_identity,
        )
    except EvidenceError:
        rejected_zero_fingerprint = True
    finally:
        atomic_json(suffix(audit_prefix, ".json"), synthetic_metadata)
    check("zero-fingerprint-rejected", rejected_zero_fingerprint)

    motion_evidence = verify_motion_variants()
    validate_motion_design_evidence(motion_evidence)
    check(
        "motion-four-variant-design",
        motion_evidence["all_four_variants_equal"]
        and motion_evidence["camera_files_unique"]
        and {
            role: variant["shutter"]
            for role, variant in motion_evidence["variants"].items()
        } == {role: list(value) for role, value in MOTION_SHUTTERS.items()},
        motion_evidence,
    )

    intensity = paired_intensity_check([1.0, 1.1, 0.9], [1.0, 1.1, 0.9])
    check("paired-intensity-identical", intensity["passed"], intensity)
    check("paired-intensity-zero-difference", intensity["absolute_mean_difference"] == 0.0)
    check("rmse", math.isclose(image_rmse([0.0, 2.0], [0.0, 0.0]), math.sqrt(2.0)))
    described = describe([1.0, 2.0, 3.0])
    check("describe-median", described["median"] == 2.0, described)
    check("fixed-sampler-ids", SAMPLER_IDS == {"counter": 2, "halton": 3})

    summary = {
        "schema_version": SCHEMA_VERSION,
        "tool": "scripts/measure_hedge.py",
        "mode": "self-test",
        "status": "passed",
        "completed_utc": utc_now(),
        "checks": checks,
        "fixed_design": fixed_design(),
        "renderer_executed": False,
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=DEFAULT_BINARY,
        help=f"headless renderer (default: {DEFAULT_BINARY})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="brand-new evidence directory (required; existing paths are rejected)",
    )
    parser.add_argument(
        "--phase",
        choices=("quality", "timing", "gallery", "all"),
        default="all",
        help="fixed subdesign to run (default: all)",
    )
    parser.add_argument(
        "--analyze-only",
        type=Path,
        metavar="EVIDENCE_DIR",
        help="recompute summaries from retained evidence without executing a renderer",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="exercise fixed counts, command construction, raw parsing, and statistics only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test and args.analyze_only is not None:
        parser.error("--self-test and --analyze-only are mutually exclusive")
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"--output must be a brand-new path: {output}")
    except OSError as error:
        parser.error(f"cannot create --output directory {output}: {error}")
    if args.self_test:
        return run_self_test(output)
    if args.analyze_only is not None:
        return run_analysis_only(args, output)
    return run_measurement(args, output)


if __name__ == "__main__":
    raise SystemExit(main())
