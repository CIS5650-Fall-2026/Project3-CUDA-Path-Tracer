#!/usr/bin/env python3
"""Validate sampler and motion-blur hedge claims with retained local evidence.

This driver intentionally invokes only the supplied local headless renderer.  It
does not provision machines, contact remote services, or modify canonical scene
or asset files.  Every renderer process receives its own evidence directory.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence

try:
    # This is the authoritative parser for the repository's raw-sum format.
    from validate_render import fnv1a, read_sum
except ModuleNotFoundError:  # Allows ``python -m scripts.validate_hedge``.
    from scripts.validate_render import fnv1a, read_sum


ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT / "scenes"
DEFAULT_BINARY = ROOT / "build/bin/cis565_path_tracer_headless"

VALIDATION_SCENE = SCENES / "hedge-motion-validation.json"
STATIC_SCENE = SCENES / "hedge-sampling-diffuse.json"

WIDTH = 17
HEIGHT = 13
DEPTH = 7
SEED = 947
TARGET_SAMPLES = 17
PARTIAL_SAMPLES = 5
RR_START = 3
TIME_BUDGET_MS = 20
TIME_BUDGET_TARGET_SAMPLES = 65536
SAMPLER_IDS = {"counter": 2, "halton": 3}
DEPTH_SEMANTICS = (
    "maximum surface intersections; final intersection is emission-only"
)

EXPECTED_COMPLETE_INVOCATIONS = 108


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Return a JSON-serializable representation without non-standard NaN."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return {"non_finite_float": repr(value)}
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def relative_name(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": relative_name(path, root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def all_json_floats_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(all_json_floats_finite(item) for item in value)
    if isinstance(value, dict):
        return all(all_json_floats_finite(item) for item in value.values())
    return False


def finite_nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def binary32(value: Any) -> float:
    """Match the scene loader's JSON-number -> C++ float conversion exactly."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


@dataclass(frozen=True)
class PhysicalConfig:
    sampler: str
    motion: bool
    wavefront: bool
    rr: bool
    bvh: bool = True
    compact: bool = True
    sort: bool = False
    aa: bool = True
    nee: bool = True
    rr_start: int = RR_START
    seed: int = SEED
    width: int = WIDTH
    height: int = HEIGHT
    depth: int = DEPTH

    @property
    def sampler_id(self) -> int:
        return SAMPLER_IDS[self.sampler]

    @property
    def slug(self) -> str:
        return (
            f"sampler-{self.sampler}-motion-{'on' if self.motion else 'off'}-"
            f"wavefront-{'on' if self.wavefront else 'off'}-"
            f"rr-{'on' if self.rr else 'off'}"
        )

    def metadata_config(self) -> dict[str, Any]:
        return {
            "max_depth": self.depth,
            "aa": self.aa,
            "nee": self.nee,
            "bvh": self.bvh,
            "compact": self.compact,
            "sort": self.sort,
            "wavefront": self.wavefront,
            "rr": self.rr,
            "rr_start": self.rr_start,
            "sampler": self.sampler,
            "sampler_id": self.sampler_id,
            "motion": self.motion,
            "debug": "none",
            "depth_semantics": DEPTH_SEMANTICS,
        }


CONFIGURATIONS = tuple(
    PhysicalConfig(sampler=sampler, motion=motion, wavefront=wavefront, rr=rr)
    for sampler, motion, wavefront, rr in itertools.product(
        ("counter", "halton"), (False, True), (False, True), (False, True)
    )
)


def switch(value: bool) -> str:
    return "on" if value else "off"


def fresh_command(
    binary: Path,
    scene: Path,
    prefix: Path,
    samples: int,
    config: PhysicalConfig,
    *,
    checkpoint: Path | None = None,
    checkpoint_every: int | None = None,
    benchmark: bool = False,
    time_budget_ms: str | int | None = None,
) -> list[str]:
    command = [
        str(binary),
        str(scene),
        "--samples",
        str(samples),
        "--seed",
        str(config.seed),
        "--resolution",
        f"{config.width}x{config.height}",
        "--depth",
        str(config.depth),
        "--compact",
        switch(config.compact),
        "--sort",
        switch(config.sort),
        "--bvh",
        switch(config.bvh),
        "--aa",
        switch(config.aa),
        "--nee",
        switch(config.nee),
        "--wavefront",
        switch(config.wavefront),
        "--rr",
        switch(config.rr),
        "--rr-start",
        str(config.rr_start),
        "--sampler",
        config.sampler,
        "--motion",
        switch(config.motion),
        "--output",
        str(prefix),
    ]
    if checkpoint is not None:
        command.extend(("--checkpoint", str(checkpoint)))
    if checkpoint_every is not None:
        command.extend(("--checkpoint-every", str(checkpoint_every)))
    if time_budget_ms is not None:
        command.extend(("--time-budget-ms", str(time_budget_ms)))
    if benchmark:
        command.append("--benchmark")
    return command


def minimal_resume_command(
    binary: Path,
    scene: Path,
    prefix: Path,
    samples: int,
    checkpoint: Path,
) -> list[str]:
    """The deliberately minimal continuation invocation required by the design."""
    return [
        str(binary),
        str(scene),
        "--samples",
        str(samples),
        "--output",
        str(prefix),
        "--resume",
        str(checkpoint),
    ]


def scene_shutter(scene: Path) -> list[float]:
    document = json.loads(scene.read_text(encoding="utf-8"))
    value = document.get("Camera", {}).get("SHUTTER", [0.0, 1.0])
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{scene}: Camera.SHUTTER is not a pair")
    result = [float(value[0]), float(value[1])]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{scene}: Camera.SHUTTER is non-finite")
    return result


def scene_camera_metadata(scene: Path) -> dict[str, Any]:
    document = json.loads(scene.read_text(encoding="utf-8"))
    camera = document.get("Camera")
    if not isinstance(camera, dict):
        raise ValueError(f"{scene}: Camera is not an object")
    eye = camera.get("EYE")
    look_at = camera.get("LOOKAT")
    if not (isinstance(eye, list) and len(eye) == 3
            and isinstance(look_at, list) and len(look_at) == 3):
        raise ValueError(f"{scene}: camera EYE/LOOKAT are not triplets")
    aperture = binary32(camera.get("APERTURE", 0.0))
    focus_value = camera.get("FOCUS", camera.get("FOCUS_DISTANCE"))
    if focus_value is None:
        # None of the authored hedge scenes takes this path, but retain a
        # binary32 result to match Camera::focusDistance if a future fixture
        # relies on the loader's computed default.
        eye32 = [binary32(value) for value in eye]
        look_at32 = [binary32(value) for value in look_at]
        focus_value = math.sqrt(math.fsum(
            (look_at32[index] - eye32[index]) ** 2 for index in range(3)
        ))
    result = {
        "position": [binary32(value) for value in eye],
        "look_at": [binary32(value) for value in look_at],
        "aperture": aperture,
        "focus_distance": binary32(focus_value),
        "shutter": [binary32(value) for value in scene_shutter(scene)],
    }
    if not all_json_floats_finite(result):
        raise ValueError(f"{scene}: camera metadata contains non-finite values")
    return result


def expected_render(
    scene: Path,
    config: PhysicalConfig,
    target_samples: int,
    *,
    completed_samples: int | None = None,
    completed_range: tuple[int, int] | None = None,
    benchmark: bool = False,
    time_budget_ms: int | None = None,
    shutter: Sequence[float] | None = None,
    camera: Mapping[str, Any] | None = None,
    checkpoint: Path | None = None,
    resumed_from: Path | None = None,
) -> dict[str, Any]:
    camera_expected = dict(camera) if camera is not None else scene_camera_metadata(scene)
    if shutter is not None:
        camera_expected["shutter"] = list(shutter)
    return {
        "scene": str(scene),
        "target_samples": target_samples,
        "completed_samples": completed_samples,
        "completed_range": list(completed_range) if completed_range else None,
        "seed": config.seed,
        "resolution": [config.width, config.height],
        "sampler": config.sampler,
        "sampler_id": config.sampler_id,
        "config": config.metadata_config(),
        "camera": camera_expected,
        "benchmark": benchmark,
        "time_budget_ms": time_budget_ms,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "resumed_from": str(resumed_from) if resumed_from is not None else None,
    }


def append_audit(
    audits: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    actual: Any = None,
    expected: Any = None,
    detail: str | None = None,
) -> None:
    item: dict[str, Any] = {"name": name, "passed": bool(passed)}
    if actual is not None:
        item["actual"] = json_safe(actual)
    if expected is not None:
        item["expected"] = json_safe(expected)
    if detail is not None:
        item["detail"] = detail
    audits.append(item)


def audit_loaded_render(
    metadata: Any,
    raw: Mapping[str, Any],
    expected: Mapping[str, Any],
    process_elapsed_seconds: float,
) -> list[dict[str, Any]]:
    """Audit already-loaded metadata/raw data; pure enough for self-tests."""
    audits: list[dict[str, Any]] = []
    append_audit(audits, "metadata-is-object", isinstance(metadata, dict))
    if not isinstance(metadata, dict):
        return audits

    append_audit(audits, "metadata-json-floats-finite", all_json_floats_finite(metadata))

    completed = metadata.get("completed_samples")
    target = metadata.get("target_samples")
    append_audit(
        audits,
        "target-samples",
        target == expected["target_samples"],
        actual=target,
        expected=expected["target_samples"],
    )
    if expected.get("completed_samples") is not None:
        completed_ok = completed == expected["completed_samples"]
        completed_expectation: Any = expected["completed_samples"]
    else:
        bounds = expected.get("completed_range")
        completed_ok = (
            isinstance(completed, int)
            and not isinstance(completed, bool)
            and bounds is not None
            and bounds[0] <= completed <= bounds[1]
        )
        completed_expectation = bounds
    append_audit(
        audits,
        "completed-samples",
        completed_ok,
        actual=completed,
        expected=completed_expectation,
    )

    resolution = metadata.get("resolution")
    append_audit(
        audits,
        "metadata-resolution",
        resolution == expected["resolution"],
        actual=resolution,
        expected=expected["resolution"],
    )
    raw_resolution = [raw.get("width"), raw.get("height")]
    append_audit(
        audits,
        "raw-resolution-matches-metadata",
        raw_resolution == resolution,
        actual=raw_resolution,
        expected=resolution,
    )
    append_audit(
        audits,
        "raw-samples-match-metadata",
        raw.get("samples") == completed,
        actual=raw.get("samples"),
        expected=completed,
    )
    values = raw.get("values", ())
    raw_values_ok = bool(values) and all(
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
        for value in values
    )
    append_audit(
        audits,
        "raw-values-finite-nonnegative",
        raw_values_ok,
        actual={"value_count": len(values)},
    )

    encoded = raw.get("encoded")
    computed_checksum = fnv1a(encoded[:-8]) if isinstance(encoded, bytes) else None
    checksum = metadata.get("linear_sum_format", {}).get("checksum") \
        if isinstance(metadata.get("linear_sum_format"), dict) else None
    append_audit(
        audits,
        "raw-checksum-matches-metadata",
        computed_checksum is not None and checksum == computed_checksum,
        actual=checksum,
        expected=computed_checksum,
    )

    append_audit(
        audits,
        "scene-path",
        metadata.get("scene") == expected["scene"],
        actual=metadata.get("scene"),
        expected=expected["scene"],
    )
    append_audit(
        audits,
        "seed",
        metadata.get("seed") == expected["seed"],
        actual=metadata.get("seed"),
        expected=expected["seed"],
    )
    for key in ("checkpoint", "resumed_from"):
        append_audit(
            audits,
            "metadata-" + key.replace("_", "-"),
            metadata.get(key) == expected.get(key),
            actual=metadata.get(key),
            expected=expected.get(key),
        )
    for key in ("sampler", "sampler_id", "sampler_version"):
        wanted = expected["sampler"] if key == "sampler" else expected["sampler_id"]
        append_audit(
            audits,
            "metadata-" + key.replace("_", "-"),
            metadata.get(key) == wanted,
            actual=metadata.get(key),
            expected=wanted,
        )

    actual_config = metadata.get("config")
    append_audit(audits, "config-is-object", isinstance(actual_config, dict))
    if isinstance(actual_config, dict):
        append_audit(
            audits,
            "config-exact-fields",
            set(actual_config) == set(expected["config"]),
            actual=sorted(actual_config),
            expected=sorted(expected["config"]),
        )
        for key, wanted in expected["config"].items():
            append_audit(
                audits,
                "config-" + key.replace("_", "-"),
                actual_config.get(key) == wanted,
                actual=actual_config.get(key),
                expected=wanted,
            )

    camera = metadata.get("camera")
    append_audit(audits, "camera-is-object", isinstance(camera, dict))
    if isinstance(camera, dict):
        append_audit(
            audits,
            "camera-exact-fields",
            set(camera) == set(expected["camera"]),
            actual=sorted(camera),
            expected=sorted(expected["camera"]),
        )
        for key, wanted in expected["camera"].items():
            append_audit(
                audits,
                "camera-" + key.replace("_", "-"),
                camera.get(key) == wanted,
                actual=camera.get(key),
                expected=wanted,
            )
        shutter = camera.get("shutter")
        shutter_range_ok = (
            isinstance(shutter, list)
            and len(shutter) == 2
            and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in shutter)
            and 0.0 <= float(shutter[0]) <= float(shutter[1]) <= 1.0
        )
        append_audit(audits, "camera-shutter-range", shutter_range_ok, actual=shutter)
        append_audit(
            audits,
            "camera-aperture-nonnegative",
            finite_nonnegative_number(camera.get("aperture")),
            actual=camera.get("aperture"),
        )
        focus = camera.get("focus_distance")
        append_audit(
            audits,
            "camera-focus-distance-positive",
            finite_nonnegative_number(focus) and float(focus) > 0.0,
            actual=focus,
        )

    fingerprints = metadata.get("fingerprints")
    required_fingerprints = ("scene_file", "assets", "materials", "geometry", "render_config")
    fingerprint_ok = isinstance(fingerprints, dict) and all(
        isinstance(fingerprints.get(key), int)
        and not isinstance(fingerprints.get(key), bool)
        and fingerprints[key] > 0
        for key in required_fingerprints
    )
    append_audit(
        audits,
        "scene-asset-and-render-fingerprints-retained",
        fingerprint_ok,
        actual=fingerprints,
        expected=list(required_fingerprints),
    )

    compatibility = metadata.get("compatibility")
    build_identifier = compatibility.get("build_identifier") \
        if isinstance(compatibility, dict) else None
    source_identifier = None
    if isinstance(compatibility, dict):
        source_identifier = compatibility.get(
            "source_revision", compatibility.get("baseline_starter_revision")
        )
    append_audit(
        audits,
        "build-identifier-retained",
        isinstance(build_identifier, str) and bool(build_identifier),
        actual=build_identifier,
    )
    append_audit(
        audits,
        "source-identifier-retained",
        isinstance(source_identifier, str) and bool(source_identifier),
        actual=source_identifier,
    )

    render_loop_ms = metadata.get("render_loop_wall_ms")
    append_audit(
        audits,
        "render-loop-wall-ms-nonnegative",
        finite_nonnegative_number(render_loop_ms),
        actual=render_loop_ms,
    )
    append_audit(
        audits,
        "render-loop-contained-in-process-elapsed",
        finite_nonnegative_number(render_loop_ms)
        and float(render_loop_ms) <= process_elapsed_seconds * 1000.0 + 100.0,
        actual={
            "render_loop_wall_ms": render_loop_ms,
            "process_elapsed_ms": process_elapsed_seconds * 1000.0,
            "timer_tolerance_ms": 100.0,
        },
    )

    policy = metadata.get("run_policy")
    append_audit(audits, "run-policy-is-object", isinstance(policy, dict))
    append_audit(
        audits,
        "obsolete-policy-alias-absent",
        "diagnostic_run_policy" not in metadata,
        actual="diagnostic_run_policy" in metadata,
        expected=False,
    )
    if isinstance(policy, dict):
        required_policy_fields = {
            "time_budget_ms", "stop_reason", "last_sample_wall_ms"
        }
        append_audit(
            audits,
            "run-policy-exact-fields",
            set(policy) == required_policy_fields,
            actual=sorted(policy),
            expected=sorted(required_policy_fields),
        )
        requested = expected.get("time_budget_ms")
        recorded_budget = policy.get("time_budget_ms")
        budget_type_ok = (
            recorded_budget is None
            if requested is None
            else isinstance(recorded_budget, int)
            and not isinstance(recorded_budget, bool)
            and recorded_budget > 0
        )
        append_audit(
            audits,
            "time-budget-type",
            budget_type_ok,
            actual=recorded_budget,
            expected="null" if requested is None else "positive integer",
        )
        append_audit(
            audits,
            "time-budget-request-recorded",
            recorded_budget == requested,
            actual=recorded_budget,
            expected=requested,
        )
        last_sample_ms = policy.get("last_sample_wall_ms")
        append_audit(
            audits,
            "last-sample-wall-ms-nonnegative",
            finite_nonnegative_number(last_sample_ms),
            actual=last_sample_ms,
        )
        last_sample_contained = (
            finite_nonnegative_number(last_sample_ms)
            and finite_nonnegative_number(render_loop_ms)
            and float(last_sample_ms) <= float(render_loop_ms) + 1e-6
        )
        append_audit(
            audits,
            "last-sample-contained-in-render-loop",
            last_sample_contained,
            actual=last_sample_ms,
            expected={"at_most_render_loop_wall_ms": render_loop_ms},
        )
        wanted_stop = (
            "time_budget"
            if requested is not None
            and isinstance(completed, int)
            and not isinstance(completed, bool)
            and isinstance(target, int)
            and not isinstance(target, bool)
            and completed < target
            else "sample_target"
        )
        append_audit(
            audits,
            "stop-reason-consistent",
            policy.get("stop_reason") == wanted_stop,
            actual=policy.get("stop_reason"),
            expected=wanted_stop,
        )
        if wanted_stop == "time_budget":
            append_audit(
                audits,
                "time-budget-reached-before-stop",
                finite_nonnegative_number(render_loop_ms)
                and float(render_loop_ms) >= float(requested),
                actual=render_loop_ms,
                expected={"at_least_ms": requested},
                detail=(
                    "The budget is checked only between whole samples; no upper "
                    "elapsed-time bound is asserted."
                ),
            )

    benchmark_enabled = metadata.get("benchmark_enabled")
    append_audit(
        audits,
        "benchmark-enabled-flag",
        benchmark_enabled is expected["benchmark"],
        actual=benchmark_enabled,
        expected=expected["benchmark"],
    )
    if expected["benchmark"]:
        benchmark = metadata.get("benchmark")
        benchmark_ok = isinstance(benchmark, dict)
        append_audit(audits, "benchmark-metadata-is-object", benchmark_ok)
        if benchmark_ok:
            append_audit(
                audits,
                "benchmark-measured-samples",
                benchmark.get("measured_samples") == completed,
                actual=benchmark.get("measured_samples"),
                expected=completed,
            )
            for key in ("renderer_host_ms", "renderer_gpu_ms"):
                append_audit(
                    audits,
                    "benchmark-" + key.replace("_", "-"),
                    finite_nonnegative_number(benchmark.get(key)),
                    actual=benchmark.get(key),
                )
    return audits


def audit_benchmark_csv(
    path: Path,
    *,
    wavefront: bool,
    rr: bool,
    completed_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    rows: list[dict[str, str]] = []
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except Exception as error:  # Evidence should retain malformed CSV failures.
        append_audit(audits, "benchmark-csv-readable", False, detail=repr(error))
        return audits, {}
    append_audit(audits, "benchmark-csv-readable", True)
    bounce_rows = [row for row in rows if row.get("record_type") == "bounce"]
    overall_rows = [row for row in rows if row.get("record_type") == "overall"]
    append_audit(audits, "benchmark-has-bounce-rows", bool(bounce_rows), actual=len(bounce_rows))
    append_audit(
        audits, "benchmark-has-one-overall-row", len(overall_rows) == 1, actual=len(overall_rows)
    )

    queue_fields = (
        "queued_diffuse",
        "queued_mirror",
        "queued_dielectric",
        "queued_emission",
        "queued_terminal",
    )
    totals = {
        "bounce_records": len(bounce_rows),
        "active_before": 0,
        "queued_paths": 0,
        "kernel_dispatches": 0,
        "rr_tested": 0,
        "rr_terminated": 0,
        "overall_measured_samples": None,
        "overall_host_ms": None,
        "overall_gpu_ms": None,
        "overall_render_loop_wall_ms": None,
    }
    row_errors: list[str] = []
    time_fields = (
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
    for index, row in enumerate(bounce_rows):
        try:
            active_before = int(row["active_before"])
            active_after = int(row["active_after"])
            queued = sum(int(row[field]) for field in queue_fields)
            dispatches = int(row["kernel_dispatches"])
            tested = int(row["rr_tested"])
            terminated = int(row["rr_terminated"])
            if not (0 <= active_after <= active_before):
                row_errors.append(f"row {index}: invalid active counts")
            if not (0 <= terminated <= tested <= active_before):
                row_errors.append(f"row {index}: invalid RR counts")
            if any(not finite_nonnegative_number(float(row[field])) for field in time_fields):
                row_errors.append(f"row {index}: invalid timing")
            if wavefront:
                if queued != active_before:
                    row_errors.append(f"row {index}: queues do not partition active paths")
                if active_before > 0 and dispatches <= 0:
                    row_errors.append(f"row {index}: no wavefront dispatch")
            elif queued != 0 or dispatches != 0:
                row_errors.append(f"row {index}: unified mode reported queue work")
            totals["active_before"] += active_before
            totals["queued_paths"] += queued
            totals["kernel_dispatches"] += dispatches
            totals["rr_tested"] += tested
            totals["rr_terminated"] += terminated
        except (KeyError, TypeError, ValueError) as error:
            row_errors.append(f"row {index}: {error}")
    append_audit(
        audits,
        "benchmark-per-bounce-invariants",
        not row_errors,
        actual=row_errors[:20],
        detail=(f"{len(row_errors)} errors" if row_errors else None),
    )
    if rr:
        append_audit(
            audits,
            "benchmark-rr-enabled-observed-tests",
            totals["rr_tested"] > 0,
            actual=totals["rr_tested"],
        )
    else:
        append_audit(
            audits,
            "benchmark-rr-disabled-zero-counters",
            totals["rr_tested"] == 0 and totals["rr_terminated"] == 0,
            actual={
                "rr_tested": totals["rr_tested"],
                "rr_terminated": totals["rr_terminated"],
            },
        )
    if len(overall_rows) == 1:
        try:
            measured = int(overall_rows[0]["measured_samples"])
        except (KeyError, TypeError, ValueError):
            measured = None
        append_audit(
            audits,
            "benchmark-overall-measured-samples",
            measured == completed_samples,
            actual=measured,
            expected=completed_samples,
        )
        totals["overall_measured_samples"] = measured
        for key in ("host_ms", "gpu_ms", "render_loop_wall_ms"):
            try:
                value: Any = float(overall_rows[0][key])
            except (KeyError, TypeError, ValueError):
                value = None
            append_audit(
                audits,
                "benchmark-overall-" + key.replace("_", "-"),
                finite_nonnegative_number(value),
                actual=value,
            )
            totals["overall_" + key] = value
    return audits, totals


def audit_benchmark_metadata_consistency(
    metadata: Any, totals: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Cross-check the CSV overall record against independently written JSON."""
    audits: list[dict[str, Any]] = []
    benchmark_metadata = metadata.get("benchmark") \
        if isinstance(metadata, dict) else None
    metadata_pairs = (
        (
            "benchmark-csv-measured-samples-match-metadata",
            totals.get("overall_measured_samples"),
            benchmark_metadata.get("measured_samples")
            if isinstance(benchmark_metadata, dict) else None,
            True,
        ),
        (
            "benchmark-csv-host-ms-match-metadata",
            totals.get("overall_host_ms"),
            benchmark_metadata.get("renderer_host_ms")
            if isinstance(benchmark_metadata, dict) else None,
            False,
        ),
        (
            "benchmark-csv-gpu-ms-match-metadata",
            totals.get("overall_gpu_ms"),
            benchmark_metadata.get("renderer_gpu_ms")
            if isinstance(benchmark_metadata, dict) else None,
            False,
        ),
        (
            "benchmark-csv-render-loop-ms-match-metadata",
            totals.get("overall_render_loop_wall_ms"),
            metadata.get("render_loop_wall_ms")
            if isinstance(metadata, dict) else None,
            False,
        ),
    )
    for audit_name, csv_value, metadata_value, exact in metadata_pairs:
        if exact:
            consistent = csv_value == metadata_value
        else:
            consistent = (
                finite_nonnegative_number(csv_value)
                and finite_nonnegative_number(metadata_value)
                and math.isclose(
                    float(csv_value),
                    float(metadata_value),
                    rel_tol=1e-7,
                    abs_tol=1e-6,
                )
            )
        append_audit(
            audits,
            audit_name,
            consistent,
            actual=csv_value,
            expected=metadata_value,
            detail=(
                None if exact else
                "CSV uses nine significant digits; compared with "
                "rel_tol=1e-7 and abs_tol=1e-6."
            ),
        )
    return audits


class Evidence:
    def __init__(self, output: Path, binary: Path, benchmark: bool) -> None:
        self.output = output
        self.binary = binary
        self.benchmark = benchmark
        self.records: list[dict[str, Any]] = []
        self.checks: list[dict[str, Any]] = []
        self.raw: dict[str, dict[str, Any]] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.started_utc = utc_now()
        self.finished_utc: str | None = None
        self.status = "running"
        self.next_index = 1
        self.binary_identity: dict[str, Any] = {"path": str(self.binary)}
        if self.binary.is_file():
            try:
                self.binary_identity.update(
                    {"bytes": self.binary.stat().st_size, "sha256": sha256_file(self.binary)}
                )
            except OSError as error:
                self.binary_identity["hash_error"] = repr(error)
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "runs").mkdir()
        self.persist()

    def summary(self) -> dict[str, Any]:
        failed = sum(not item["passed"] for item in self.checks)
        return {
            "schema_version": 1,
            "tool": "validate_hedge.py",
            "status": self.status,
            "passed": self.status == "passed",
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "binary": self.binary_identity,
            "output": str(self.output),
            "fixed_design": {
                "scene": str(VALIDATION_SCENE),
                "static_scene": str(STATIC_SCENE),
                "resolution": [WIDTH, HEIGHT],
                "depth": DEPTH,
                "seed": SEED,
                "target_samples": TARGET_SAMPLES,
                "partial_samples": PARTIAL_SAMPLES,
                "rr_start": RR_START,
                "samplers": SAMPLER_IDS,
                "matrix_configurations": len(CONFIGURATIONS),
                "time_budget_ms": TIME_BUDGET_MS,
                "time_budget_target_samples": TIME_BUDGET_TARGET_SAMPLES,
                "benchmark_reference_runs": self.benchmark,
                "time_budget_benchmark": True,
                "expected_complete_invocations": EXPECTED_COMPLETE_INVOCATIONS,
            },
            "counts": {
                "invocations": len(self.records),
                "checks": len(self.checks),
                "passed_checks": len(self.checks) - failed,
                "failed_checks": failed,
            },
            "records": self.records,
            "checks": self.checks,
        }

    def persist(self) -> None:
        write_json_atomic(self.output / "summary.json", self.summary())

    def add_check(self, name: str, passed: bool, **detail: Any) -> bool:
        item = {"name": name, "passed": bool(passed), **json_safe(detail)}
        self.checks.append(item)
        self.persist()
        return bool(passed)

    def allocate(self, label: str) -> tuple[Path, Path]:
        safe = "".join(character if character.isalnum() or character in "-_" else "-"
                       for character in label)
        run_dir = self.output / "runs" / f"{self.next_index:03d}-{safe}"
        self.next_index += 1
        run_dir.mkdir(exist_ok=False)
        return run_dir, run_dir / "render"

    def _record_inputs(self, command: Sequence[str]) -> list[dict[str, Any]]:
        candidates: list[tuple[str, Path]] = []
        if command:
            candidates.append(("binary", Path(command[0])))
        if len(command) >= 2:
            candidates.append(("scene", Path(command[1])))
        for option in ("--resume",):
            if option in command:
                index = command.index(option)
                if index + 1 < len(command):
                    candidates.append((option.removeprefix("--"), Path(command[index + 1])))
        result = []
        for role, path in candidates:
            item: dict[str, Any] = {"role": role, "path": str(path)}
            if role == "binary" and path == self.binary and "sha256" in self.binary_identity:
                item.update(
                    {
                        "bytes": self.binary_identity["bytes"],
                        "sha256": self.binary_identity["sha256"],
                    }
                )
            elif path.is_file():
                item.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
            else:
                item["missing"] = True
            result.append(item)
        return result

    def invoke(
        self,
        label: str,
        run_dir: Path,
        prefix: Path,
        command: Sequence[str],
        *,
        expect_success: bool,
        expected_rejection_terms: Sequence[str] = (),
    ) -> dict[str, Any]:
        command = [str(item) for item in command]
        command_document = {
            "label": label,
            "argv": command,
            "shell": False,
            "cwd": str(ROOT),
            "input_files_before_invocation": self._record_inputs(command),
        }
        command_path = run_dir / "command.json"
        write_json_atomic(command_path, command_document)
        started = utc_now()
        before = time.perf_counter()
        exception = None
        try:
            process = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            returncode: int | None = process.returncode
            stdout = process.stdout
            stderr = process.stderr
        except Exception as error:  # Preserve launch failures as evidence.
            returncode = None
            stdout = ""
            stderr = ""
            exception = repr(error)
        elapsed = time.perf_counter() - before
        finished = utc_now()
        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        record: dict[str, Any] = {
            "index": len(self.records) + 1,
            "label": label,
            "run_directory": relative_name(run_dir, self.output),
            "output_prefix": relative_name(prefix, self.output),
            "command": command,
            "command_json": file_record(command_path, self.output),
            "inputs": command_document["input_files_before_invocation"],
            "started_utc": started,
            "finished_utc": finished,
            "elapsed_seconds": elapsed,
            "exit_code": returncode,
            "expected_exit": "zero" if expect_success else "nonzero",
            "launch_exception": exception,
            "stdout": file_record(stdout_path, self.output),
            "stderr": file_record(stderr_path, self.output),
            "artifacts": {},
            "audits": [],
        }
        if not expect_success:
            record["rejection_text"] = (stdout + "\n" + stderr).strip()
        self.records.append(record)
        self._refresh_artifacts(record, run_dir, prefix, command)
        self._write_invocation(record, run_dir)

        if expect_success:
            accepted = returncode == 0
            self.add_check(
                "invocation/" + label,
                accepted,
                expected="exit code 0",
                actual=returncode,
                launch_exception=exception,
            )
        else:
            rejection = (stdout + "\n" + stderr).lower()
            terms_ok = not expected_rejection_terms or any(
                term.lower() in rejection for term in expected_rejection_terms
            )
            accepted = returncode is not None and returncode != 0 and terms_ok
            self.add_check(
                "invocation/" + label,
                accepted,
                expected={
                    "exit": "nonzero",
                    "rejection_contains_any": list(expected_rejection_terms),
                },
                actual={
                    "exit_code": returncode,
                    "matched_rejection_term": terms_ok,
                    "launch_exception": exception,
                },
            )
        record["expectation_met"] = accepted
        record["expected_rejection_terms"] = list(expected_rejection_terms)
        self._write_invocation(record, run_dir)
        return record

    def _refresh_artifacts(
        self,
        record: dict[str, Any],
        run_dir: Path,
        prefix: Path,
        command: Sequence[str],
    ) -> None:
        artifacts: dict[str, Any] = {}
        for name, suffix in (
            ("metadata", ".json"),
            ("raw_sum", ".sum.bin"),
            ("png", ".png"),
            ("benchmark_csv", ".benchmark.csv"),
        ):
            path = Path(str(prefix) + suffix)
            if path.is_file():
                artifacts[name] = file_record(path, self.output)
        if "--checkpoint" in command:
            index = command.index("--checkpoint")
            if index + 1 < len(command):
                checkpoint = Path(command[index + 1])
                if checkpoint.is_file():
                    artifacts["checkpoint"] = file_record(checkpoint, self.output)
        record["artifacts"] = artifacts

    def _write_invocation(self, record: dict[str, Any], run_dir: Path) -> None:
        write_json_atomic(run_dir / "invocation.json", record)
        self.persist()

    def audit_render(
        self,
        record: dict[str, Any],
        run_dir: Path,
        prefix: Path,
        expected: Mapping[str, Any],
    ) -> None:
        label = record["label"]
        audits: list[dict[str, Any]] = []
        raw_path = Path(str(prefix) + ".sum.bin")
        metadata_path = Path(str(prefix) + ".json")
        png_path = Path(str(prefix) + ".png")
        raw = None
        metadata = None
        try:
            raw = read_sum(raw_path)
            append_audit(audits, "raw-sum-parsed-by-validate-render", True)
        except Exception as error:
            append_audit(
                audits,
                "raw-sum-parsed-by-validate-render",
                False,
                detail=repr(error),
            )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            append_audit(audits, "metadata-json-readable", True)
        except Exception as error:
            append_audit(audits, "metadata-json-readable", False, detail=repr(error))
        png_ok = False
        try:
            png_data = png_path.read_bytes()
            png_ok = len(png_data) > 8 and png_data.startswith(b"\x89PNG\r\n\x1a\n")
            append_audit(
                audits,
                "png-present-and-signature-valid",
                png_ok,
                actual={"bytes": len(png_data), "sha256": sha256_bytes(png_data)},
            )
        except Exception as error:
            append_audit(audits, "png-present-and-signature-valid", False, detail=repr(error))

        if raw is not None:
            self.raw[label] = raw
            values = raw["values"]
            record["raw_summary"] = {
                "width": raw["width"],
                "height": raw["height"],
                "samples": raw["samples"],
                "value_count": len(values),
                "minimum": min(values),
                "maximum": max(values),
                "sha256": raw["sha256"],
            }
        if isinstance(metadata, dict):
            self.metadata[label] = metadata
            record["metadata"] = metadata
        if raw is not None and metadata is not None:
            audits.extend(
                audit_loaded_render(
                    metadata, raw, expected, float(record["elapsed_seconds"])
                )
            )

        if expected["benchmark"]:
            benchmark_path = Path(str(prefix) + ".benchmark.csv")
            if benchmark_path.is_file() and raw is not None:
                benchmark_audits, totals = audit_benchmark_csv(
                    benchmark_path,
                    wavefront=bool(expected["config"]["wavefront"]),
                    rr=bool(expected["config"]["rr"]),
                    completed_samples=int(raw["samples"]),
                )
                audits.extend(benchmark_audits)
                record["benchmark_counter_totals"] = totals
                audits.extend(
                    audit_benchmark_metadata_consistency(metadata, totals)
                )
            else:
                append_audit(
                    audits,
                    "benchmark-csv-present",
                    False,
                    actual=benchmark_path.is_file(),
                )

        record["audits"] = audits
        self._refresh_artifacts(record, run_dir, prefix, record["command"])
        self._write_invocation(record, run_dir)
        self.add_check(
            "artifact-audit/" + label,
            bool(audits) and all(item["passed"] for item in audits),
            audit_count=len(audits),
            failed_audits=[item for item in audits if not item["passed"]],
        )

    def render_fresh(
        self,
        label: str,
        scene: Path,
        config: PhysicalConfig,
        samples: int,
        *,
        checkpoint: bool = False,
        benchmark: bool = False,
        time_budget_ms: int | None = None,
        completed_range: tuple[int, int] | None = None,
    ) -> tuple[dict[str, Any], Path | None]:
        run_dir, prefix = self.allocate(label)
        checkpoint_path = run_dir / "state.checkpoint" if checkpoint else None
        command = fresh_command(
            self.binary,
            scene,
            prefix,
            samples,
            config,
            checkpoint=checkpoint_path,
            benchmark=benchmark,
            time_budget_ms=time_budget_ms,
        )
        record = self.invoke(
            label, run_dir, prefix, command, expect_success=True
        )
        expected = expected_render(
            scene,
            config,
            samples,
            completed_samples=None if completed_range else samples,
            completed_range=completed_range,
            benchmark=benchmark,
            time_budget_ms=time_budget_ms,
            checkpoint=checkpoint_path,
        )
        self.audit_render(record, run_dir, prefix, expected)
        if checkpoint_path is not None:
            checkpoint_ok = checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0
            self.add_check(
                "checkpoint-created/" + label,
                checkpoint_ok,
                checkpoint=(
                    file_record(checkpoint_path, self.output) if checkpoint_ok else str(checkpoint_path)
                ),
            )
        return record, checkpoint_path

    def render_resume(
        self,
        label: str,
        scene: Path,
        checkpoint: Path,
        config: PhysicalConfig,
        samples: int,
    ) -> dict[str, Any]:
        run_dir, prefix = self.allocate(label)
        command = minimal_resume_command(self.binary, scene, prefix, samples, checkpoint)
        record = self.invoke(label, run_dir, prefix, command, expect_success=True)
        wanted = minimal_resume_command(self.binary, scene, prefix, samples, checkpoint)
        self.add_check(
            "minimal-resume-arguments/" + label,
            command == wanted and command[2:] == [
                "--samples", str(samples), "--output", str(prefix), "--resume", str(checkpoint)
            ],
            actual=command,
            expected=wanted,
        )
        self.audit_render(
            record,
            run_dir,
            prefix,
            expected_render(
                scene,
                config,
                samples,
                completed_samples=samples,
                benchmark=False,
                resumed_from=checkpoint,
            ),
        )
        return record

    def reject(
        self,
        label: str,
        command_builder: Any,
        *,
        expected_terms: Sequence[str] = (),
    ) -> dict[str, Any]:
        run_dir, prefix = self.allocate(label)
        command = command_builder(prefix, run_dir)
        return self.invoke(
            label,
            run_dir,
            prefix,
            command,
            expect_success=False,
            expected_rejection_terms=expected_terms,
        )

    def exact_raw(self, name: str, left_label: str, right_label: str, detail: str) -> bool:
        left = self.raw.get(left_label)
        right = self.raw.get(right_label)
        available = left is not None and right is not None
        passed = available and left["encoded"] == right["encoded"]
        return self.add_check(
            name,
            passed,
            detail=detail,
            left={
                "label": left_label,
                "sha256": left.get("sha256") if left else None,
                "samples": left.get("samples") if left else None,
            },
            right={
                "label": right_label,
                "sha256": right.get("sha256") if right else None,
                "samples": right.get("samples") if right else None,
            },
        )

    def inherited_metadata(self, name: str, reference_label: str, resumed_label: str) -> bool:
        reference = self.metadata.get(reference_label)
        resumed = self.metadata.get(resumed_label)
        keys = (
            "config",
            "camera",
            "seed",
            "resolution",
            "sampler",
            "sampler_id",
            "sampler_version",
            "fingerprints",
            "compatibility",
        )
        comparisons = {
            key: reference is not None
            and resumed is not None
            and reference.get(key) == resumed.get(key)
            for key in keys
        }
        return self.add_check(
            name,
            bool(reference is not None and resumed is not None and all(comparisons.values())),
            compared_fields=list(keys),
            field_equality=comparisons,
            reference={key: reference.get(key) for key in keys} if reference else None,
            resumed={key: resumed.get(key) for key in keys} if resumed else None,
        )

    def finish(self) -> bool:
        self.finished_utc = utc_now()
        self.status = "passed" if self.checks and all(item["passed"] for item in self.checks) else "failed"
        self.persist()
        return self.status == "passed"


def absolute_asset_scene(source: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = json.loads(source.read_text(encoding="utf-8"))
    assets: list[dict[str, Any]] = []

    materials = document.get("Materials", {})
    if isinstance(materials, dict):
        for material_name, material in materials.items():
            if not isinstance(material, dict):
                continue
            for key in ("TEXTURE", "BUMP"):
                value = material.get(key)
                if isinstance(value, str):
                    original = Path(value)
                    resolved = (source.parent / original).resolve() \
                        if not original.is_absolute() else original.resolve()
                    material[key] = str(resolved)
                    assets.append(
                        {
                            "location": f"Materials.{material_name}.{key}",
                            "original": value,
                            "absolute": str(resolved),
                            "exists": resolved.is_file(),
                            "sha256": sha256_file(resolved) if resolved.is_file() else None,
                        }
                    )

    objects = document.get("Objects", [])
    if isinstance(objects, list):
        for index, object_definition in enumerate(objects):
            if not isinstance(object_definition, dict):
                continue
            value = object_definition.get("FILE")
            if isinstance(value, str):
                original = Path(value)
                resolved = (source.parent / original).resolve() \
                    if not original.is_absolute() else original.resolve()
                object_definition["FILE"] = str(resolved)
                assets.append(
                    {
                        "location": f"Objects[{index}].FILE",
                        "original": value,
                        "absolute": str(resolved),
                        "exists": resolved.is_file(),
                        "sha256": sha256_file(resolved) if resolved.is_file() else None,
                    }
                )
    return document, assets


def encoded_scene(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=False) + "\n").encode("utf-8")


def changed_shutter(document: Mapping[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(document))
    camera = clone.get("Camera")
    if not isinstance(camera, dict):
        raise ValueError("temporary scene has no Camera object")
    old = camera.get("SHUTTER", [0.0, 1.0])
    if old == [0.25, 0.75]:
        camera["SHUTTER"] = [0.0, 0.0]
    else:
        camera["SHUTTER"] = [0.25, 0.75]
    return clone


def changed_motion_translation(document: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    clone = json.loads(json.dumps(document))
    objects = clone.get("Objects")
    if not isinstance(objects, list):
        raise ValueError("temporary scene has no Objects array")
    for index, item in enumerate(objects):
        if isinstance(item, dict) and isinstance(item.get("MOTION_TRANSLATION"), list):
            motion = item["MOTION_TRANSLATION"]
            if len(motion) != 3:
                raise ValueError(f"Objects[{index}].MOTION_TRANSLATION is not a triplet")
            motion[0] = float(motion[0]) + 0.375
            return clone, index
    raise ValueError("temporary scene has no moving object")


def run_scene_mutation_checks(evidence: Evidence, config: PhysicalConfig) -> None:
    directory = evidence.output / "temporary-scene-mutations"
    directory.mkdir(exist_ok=False)
    canonical_bytes = VALIDATION_SCENE.read_bytes()
    canonical_copy = directory / "canonical-source-byte-copy.json"
    canonical_copy.write_bytes(canonical_bytes)
    document, assets = absolute_asset_scene(VALIDATION_SCENE)
    baseline_bytes = encoded_scene(document)
    baseline_preserved = directory / "absolute-baseline-preserved.json"
    active = directory / "active-scene.json"
    baseline_preserved.write_bytes(baseline_bytes)
    active.write_bytes(baseline_bytes)
    manifest: dict[str, Any] = {
        "canonical_source": {
            "path": str(VALIDATION_SCENE),
            "bytes": len(canonical_bytes),
            "sha256": sha256_bytes(canonical_bytes),
        },
        "canonical_source_byte_copy": file_record(canonical_copy, evidence.output),
        "absolute_baseline": file_record(baseline_preserved, evidence.output),
        "active_baseline_sha256": sha256_file(active),
        "absolutized_assets": assets,
        "mutations": [],
    }
    write_json_atomic(directory / "manifest.json", manifest)
    evidence.add_check(
        "temporary-scene/canonical-source-byte-copy-exact",
        canonical_copy.read_bytes() == canonical_bytes,
        source_sha256=sha256_bytes(canonical_bytes),
        copy_sha256=sha256_file(canonical_copy),
    )
    evidence.add_check(
        "temporary-scene/all-assets-absolutized-and-resolve",
        bool(assets)
        and all(item["exists"] and Path(item["absolute"]).is_absolute() for item in assets),
        assets=assets,
    )

    # Each semantic mutation gets a newly authored baseline checkpoint.  The
    # active scene path is then changed in place, which isolates scene identity
    # checking while all canonical inputs remain read-only.
    for mutation_name in ("shutter", "displacement"):
        active.write_bytes(baseline_bytes)
        evidence.add_check(
            f"temporary-scene/{mutation_name}-baseline-restored-before-checkpoint",
            active.read_bytes() == baseline_bytes,
            baseline_sha256=sha256_bytes(baseline_bytes),
            active_sha256=sha256_file(active),
        )
        baseline_label = f"scene-mutation-{mutation_name}-baseline"
        _, checkpoint = evidence.render_fresh(
            baseline_label,
            active,
            config,
            PARTIAL_SAMPLES,
            checkpoint=True,
        )
        if mutation_name == "shutter":
            mutated_document = changed_shutter(document)
            mutation_detail: dict[str, Any] = {
                "old_shutter": document.get("Camera", {}).get("SHUTTER", [0.0, 1.0]),
                "new_shutter": mutated_document["Camera"]["SHUTTER"],
            }
        else:
            mutated_document, object_index = changed_motion_translation(document)
            mutation_detail = {
                "object_index": object_index,
                "old_motion_translation": document["Objects"][object_index]["MOTION_TRANSLATION"],
                "new_motion_translation": mutated_document["Objects"][object_index]["MOTION_TRANSLATION"],
            }
        mutated_bytes = encoded_scene(mutated_document)
        variant_path = directory / f"mutated-{mutation_name}-preserved.json"
        variant_path.write_bytes(mutated_bytes)
        active.write_bytes(mutated_bytes)
        mutation_detail.update(
            {
                "baseline_sha256": sha256_bytes(baseline_bytes),
                "mutated_sha256": sha256_bytes(mutated_bytes),
                "preserved_variant": file_record(variant_path, evidence.output),
            }
        )
        manifest["mutations"].append({"kind": mutation_name, **mutation_detail})
        write_json_atomic(directory / "manifest.json", manifest)
        evidence.add_check(
            f"temporary-scene/{mutation_name}-changed-only-temporary-copy",
            mutated_bytes != baseline_bytes
            and VALIDATION_SCENE.read_bytes() == canonical_bytes
            and active.read_bytes() == mutated_bytes,
            **mutation_detail,
        )
        if checkpoint is not None and checkpoint.is_file():
            rejection_terms = (
                ("checkpoint camera does not exactly match",)
                if mutation_name == "shutter"
                else ("checkpoint scene-file fingerprint mismatch",)
            )
            evidence.reject(
                f"reject-scene-{mutation_name}-change",
                lambda prefix, run_dir, checkpoint=checkpoint: minimal_resume_command(
                    evidence.binary, active, prefix, TARGET_SAMPLES, checkpoint
                ),
                expected_terms=rejection_terms,
            )
        else:
            evidence.add_check(
                f"invocation/reject-scene-{mutation_name}-change",
                False,
                detail="baseline checkpoint unavailable; rejection was not attempted",
            )
        active.write_bytes(baseline_bytes)
        evidence.add_check(
            f"temporary-scene/{mutation_name}-baseline-restored-after-rejection",
            active.read_bytes() == baseline_bytes
            and VALIDATION_SCENE.read_bytes() == canonical_bytes,
            baseline_sha256=sha256_bytes(baseline_bytes),
            active_sha256=sha256_file(active),
            canonical_sha256=sha256_file(VALIDATION_SCENE),
        )


def run_validation(evidence: Evidence) -> None:
    executable_ok = evidence.binary.is_file() and os.access(evidence.binary, os.X_OK)
    evidence.add_check(
        "preflight/binary-is-executable-file",
        executable_ok,
        binary=str(evidence.binary),
    )
    for scene in (VALIDATION_SCENE, STATIC_SCENE):
        evidence.add_check(
            "preflight/scene-" + scene.stem,
            scene.is_file(),
            scene=str(scene),
            sha256=sha256_file(scene) if scene.is_file() else None,
        )
    if not executable_ok or not VALIDATION_SCENE.is_file() or not STATIC_SCENE.is_file():
        return

    validation_document = json.loads(VALIDATION_SCENE.read_text(encoding="utf-8"))
    validation_objects = validation_document.get("Objects", [])
    moving_objects = [
        item for item in validation_objects
        if isinstance(item, dict) and "MOTION_TRANSLATION" in item
    ] if isinstance(validation_objects, list) else []
    evidence.add_check(
        "preflight/motion-fixture-full-shutter",
        scene_shutter(VALIDATION_SCENE) == [0.0, 1.0],
        actual=scene_shutter(VALIDATION_SCENE),
        expected=[0.0, 1.0],
    )
    evidence.add_check(
        "preflight/motion-fixture-has-moving-geometry",
        bool(moving_objects),
        moving_object_count=len(moving_objects),
    )
    validation_materials = validation_document.get("Materials", {})
    moving_types = {
        str(item.get("TYPE", "")).lower() for item in moving_objects
        if isinstance(item, dict)
    }
    moving_emitter_types = {
        str(item.get("TYPE", "")).lower() for item in moving_objects
        if isinstance(item, dict)
        and isinstance(validation_materials, dict)
        and isinstance(validation_materials.get(item.get("MATERIAL")), dict)
        and validation_materials[item["MATERIAL"]].get("TYPE") == "Emitting"
    }
    moving_diffuse = [
        item for item in moving_objects
        if isinstance(item, dict)
        and isinstance(validation_materials, dict)
        and isinstance(validation_materials.get(item.get("MATERIAL")), dict)
        and validation_materials[item["MATERIAL"]].get("TYPE") == "Diffuse"
    ]
    lateral_motion_ok = all(
        isinstance(item.get("MOTION_TRANSLATION"), list)
        and len(item["MOTION_TRANSLATION"]) == 3
        and any(float(item["MOTION_TRANSLATION"][axis]) != 0.0 for axis in (0, 2))
        for item in moving_objects
    )
    evidence.add_check(
        "preflight/motion-fixture-representative-moving-shapes",
        {"sphere", "cube", "obj"}.issubset(moving_types)
        and {"sphere", "cube", "obj"}.issubset(moving_emitter_types)
        and bool(moving_diffuse)
        and lateral_motion_ok,
        moving_types=sorted(moving_types),
        moving_emitter_types=sorted(moving_emitter_types),
        moving_diffuse_count=len(moving_diffuse),
        all_moving_objects_have_lateral_component=lateral_motion_ok,
    )
    validation_camera = validation_document.get("Camera", {})
    evidence.add_check(
        "preflight/motion-fixture-native-design",
        isinstance(validation_camera, dict)
        and validation_camera.get("RES") == [WIDTH, HEIGHT]
        and validation_camera.get("DEPTH") == DEPTH,
        resolution=(validation_camera.get("RES") if isinstance(validation_camera, dict) else None),
        depth=(validation_camera.get("DEPTH") if isinstance(validation_camera, dict) else None),
    )
    static_document = json.loads(STATIC_SCENE.read_text(encoding="utf-8"))
    static_objects = static_document.get("Objects", [])
    static_motion_count = sum(
        isinstance(item, dict) and "MOTION_TRANSLATION" in item
        for item in static_objects
    ) if isinstance(static_objects, list) else -1
    evidence.add_check(
        "preflight/static-fixture-has-no-motion-translation",
        static_motion_count == 0,
        moving_object_count=static_motion_count,
    )
    static_materials = static_document.get("Materials", {})
    static_material_types = {
        value.get("TYPE") for value in static_materials.values()
        if isinstance(value, dict)
    } if isinstance(static_materials, dict) else set()
    evidence.add_check(
        "preflight/static-fixture-diffuse-transport-only",
        bool(static_material_types)
        and static_material_types.issubset({"Diffuse", "Emitting"}),
        material_types=sorted(str(item) for item in static_material_types),
    )

    references: dict[tuple[str, bool, bool, bool], str] = {}
    checkpoints: dict[tuple[str, bool, bool, bool], Path] = {}

    for config in CONFIGURATIONS:
        key = (config.sampler, config.motion, config.wavefront, config.rr)
        reference_label = "reference-" + config.slug
        evidence.render_fresh(
            reference_label,
            VALIDATION_SCENE,
            config,
            TARGET_SAMPLES,
            benchmark=evidence.benchmark,
        )
        references[key] = reference_label

        partial_label = "partial-" + config.slug
        _, checkpoint = evidence.render_fresh(
            partial_label,
            VALIDATION_SCENE,
            config,
            PARTIAL_SAMPLES,
            checkpoint=True,
        )
        if checkpoint is not None:
            checkpoints[key] = checkpoint
        resumed_label = "resumed-" + config.slug
        if checkpoint is not None and checkpoint.is_file():
            evidence.render_resume(
                resumed_label,
                VALIDATION_SCENE,
                checkpoint,
                config,
                TARGET_SAMPLES,
            )
            evidence.exact_raw(
                "exact-resume/" + config.slug,
                reference_label,
                resumed_label,
                "fresh-process 5-to-17 continuation must equal uninterrupted 17-sample bytes",
            )
            evidence.inherited_metadata(
                "inherited-physical-state/" + config.slug,
                reference_label,
                resumed_label,
            )
        else:
            evidence.add_check(
                "minimal-resume-arguments/" + resumed_label,
                False,
                detail="partial checkpoint unavailable",
            )
            evidence.add_check("exact-resume/" + config.slug, False, detail="resume not run")
            evidence.add_check(
                "inherited-physical-state/" + config.slug, False, detail="resume not run"
            )

        bvh_config = replace(config, bvh=False)
        bvh_label = "bvh-off-" + config.slug
        evidence.render_fresh(
            bvh_label,
            VALIDATION_SCENE,
            bvh_config,
            TARGET_SAMPLES,
            benchmark=evidence.benchmark,
        )
        evidence.exact_raw(
            "schedule-bvh/" + config.slug,
            reference_label,
            bvh_label,
            "BVH and brute-force intersection schedules must produce identical raw sums",
        )

        compact_config = replace(config, compact=False)
        compact_label = "compact-off-" + config.slug
        evidence.render_fresh(
            compact_label,
            VALIDATION_SCENE,
            compact_config,
            TARGET_SAMPLES,
            benchmark=evidence.benchmark,
        )
        evidence.exact_raw(
            "schedule-compact/" + config.slug,
            reference_label,
            compact_label,
            "path compaction on/off must produce identical raw sums",
        )

        if not config.wavefront:
            sort_config = replace(config, sort=True)
            sort_label = "sort-on-" + config.slug
            evidence.render_fresh(
                sort_label,
                VALIDATION_SCENE,
                sort_config,
                TARGET_SAMPLES,
                benchmark=evidence.benchmark,
            )
            evidence.exact_raw(
                "schedule-sort/" + config.slug,
                reference_label,
                sort_label,
                "material sorting is tested only in unified mode and must preserve raw sums",
            )

    for sampler, motion, rr in itertools.product(
        ("counter", "halton"), (False, True), (False, True)
    ):
        unified = references[(sampler, motion, False, rr)]
        wavefront = references[(sampler, motion, True, rr)]
        slug = (
            f"sampler-{sampler}-motion-{'on' if motion else 'off'}-"
            f"rr-{'on' if rr else 'off'}"
        )
        evidence.exact_raw(
            "schedule-unified-wavefront/" + slug,
            unified,
            wavefront,
            "unified and wavefront material scheduling must be byte-identical at fixed RR",
        )

    for sampler in ("counter", "halton"):
        labels: dict[bool, str] = {}
        for motion in (False, True):
            config = PhysicalConfig(
                sampler=sampler, motion=motion, wavefront=True, rr=True
            )
            label = f"static-motion-{'on' if motion else 'off'}-sampler-{sampler}"
            evidence.render_fresh(label, STATIC_SCENE, config, TARGET_SAMPLES)
            labels[motion] = label
        evidence.exact_raw(
            "static-motion-invariance/sampler-" + sampler,
            labels[False],
            labels[True],
            "motion toggle must be inert for a scene containing no moving objects",
        )

    checkpoint_key = ("counter", True, True, True)
    checkpoint = checkpoints.get(checkpoint_key)
    mismatch_config = PhysicalConfig(
        sampler="counter", motion=True, wavefront=True, rr=True
    )
    if checkpoint is not None and checkpoint.is_file():
        evidence.reject(
            "reject-checkpoint-sampler-mismatch",
            lambda prefix, run_dir: minimal_resume_command(
                evidence.binary, VALIDATION_SCENE, prefix, TARGET_SAMPLES, checkpoint
            ) + ["--sampler", "halton"],
            expected_terms=("checkpoint sampler ID",),
        )
        evidence.reject(
            "reject-checkpoint-motion-mismatch",
            lambda prefix, run_dir: minimal_resume_command(
                evidence.binary, VALIDATION_SCENE, prefix, TARGET_SAMPLES, checkpoint
            ) + ["--motion", "off"],
            expected_terms=("checkpoint motion setting",),
        )
    else:
        evidence.add_check(
            "invocation/reject-checkpoint-sampler-mismatch",
            False,
            detail="matrix checkpoint unavailable",
        )
        evidence.add_check(
            "invocation/reject-checkpoint-motion-mismatch",
            False,
            detail="matrix checkpoint unavailable",
        )

    evidence.reject(
        "reject-unknown-sampler",
        lambda prefix, run_dir: [
            str(evidence.binary), str(VALIDATION_SCENE), "--samples", "1",
            "--output", str(prefix), "--sampler", "not-a-sampler",
        ],
        expected_terms=("--sampler expects exactly",),
    )
    evidence.reject(
        "reject-unknown-motion",
        lambda prefix, run_dir: [
            str(evidence.binary), str(VALIDATION_SCENE), "--samples", "1",
            "--output", str(prefix), "--motion", "maybe",
        ],
        expected_terms=("--motion expects exactly",),
    )

    run_scene_mutation_checks(evidence, mismatch_config)

    time_label = "time-budget-positive-diagnostic"
    time_record, _ = evidence.render_fresh(
        time_label,
        VALIDATION_SCENE,
        mismatch_config,
        TIME_BUDGET_TARGET_SAMPLES,
        benchmark=True,
        time_budget_ms=TIME_BUDGET_MS,
        completed_range=(1, TIME_BUDGET_TARGET_SAMPLES),
    )
    time_metadata = evidence.metadata.get(time_label, {})
    completed = time_metadata.get("completed_samples")
    render_loop_ms = time_metadata.get("render_loop_wall_ms")
    evidence.add_check(
        "time-budget/whole-sample-range-and-overshoot-recorded",
        isinstance(completed, int)
        and 0 < completed <= TIME_BUDGET_TARGET_SAMPLES
        and finite_nonnegative_number(render_loop_ms)
        and (
            completed == TIME_BUDGET_TARGET_SAMPLES
            or float(render_loop_ms) >= TIME_BUDGET_MS
        ),
        completed_samples=completed,
        target_samples=TIME_BUDGET_TARGET_SAMPLES,
        requested_budget_ms=TIME_BUDGET_MS,
        render_loop_wall_ms=render_loop_ms,
        time_budget_overshoot_ms=(
            max(0.0, float(render_loop_ms) - TIME_BUDGET_MS)
            if finite_nonnegative_number(render_loop_ms) else None
        ),
        process_elapsed_ms=float(time_record["elapsed_seconds"]) * 1000.0,
        detail="No upper bound is asserted: the renderer may finish one whole sample after the budget.",
    )

    evidence.reject(
        "reject-time-budget-without-explicit-target",
        lambda prefix, run_dir: [
            str(evidence.binary), str(VALIDATION_SCENE), "--output", str(prefix),
            "--time-budget-ms", str(TIME_BUDGET_MS),
        ],
        expected_terms=("requires an explicit --samples safety ceiling",),
    )
    for label, value, expected_rejection in (
        ("zero", "0", "--time-budget-ms must be in"),
        ("negative", "-1", "--time-budget-ms expects an unsigned integer"),
        ("invalid", "twenty", "--time-budget-ms expects an unsigned integer"),
    ):
        evidence.reject(
            "reject-time-budget-" + label,
            lambda prefix, run_dir, value=value: [
                str(evidence.binary), str(VALIDATION_SCENE), "--samples",
                str(TIME_BUDGET_TARGET_SAMPLES), "--output", str(prefix),
                "--time-budget-ms", value,
            ],
            expected_terms=(expected_rejection,),
        )
    if checkpoint is not None and checkpoint.is_file():
        evidence.reject(
            "reject-time-budget-with-resume",
            lambda prefix, run_dir: minimal_resume_command(
                evidence.binary, VALIDATION_SCENE, prefix, TARGET_SAMPLES, checkpoint
            ) + ["--time-budget-ms", str(TIME_BUDGET_MS)],
            expected_terms=("cannot be combined with resume or checkpoint options",),
        )
    else:
        evidence.add_check(
            "invocation/reject-time-budget-with-resume",
            False,
            detail="matrix checkpoint unavailable",
        )
    evidence.reject(
        "reject-time-budget-with-checkpoint",
        lambda prefix, run_dir: fresh_command(
            evidence.binary,
            VALIDATION_SCENE,
            prefix,
            TIME_BUDGET_TARGET_SAMPLES,
            mismatch_config,
            checkpoint=run_dir / "forbidden.checkpoint",
            time_budget_ms=TIME_BUDGET_MS,
        ),
        expected_terms=("cannot be combined with resume or checkpoint options",),
    )
    evidence.reject(
        "reject-time-budget-with-checkpoint-every",
        lambda prefix, run_dir: fresh_command(
            evidence.binary,
            VALIDATION_SCENE,
            prefix,
            TIME_BUDGET_TARGET_SAMPLES,
            mismatch_config,
            checkpoint_every=5,
            time_budget_ms=TIME_BUDGET_MS,
        ),
        expected_terms=("cannot be combined with resume or checkpoint options",),
    )

    evidence.add_check(
        "fixed-design/invocation-count",
        len(evidence.records) == EXPECTED_COMPLETE_INVOCATIONS,
        actual=len(evidence.records),
        expected=EXPECTED_COMPLETE_INVOCATIONS,
    )


def synthetic_sum(path: Path, width: int, height: int, samples: int, values: Sequence[float]) -> None:
    header = struct.pack("<8sIIIIQ", b"C565SUM\0", 1, width, height, 3, samples)
    payload = struct.pack("<" + str(len(values)) + "f", *values)
    body = header + payload
    path.write_bytes(body + struct.pack("<Q", fnv1a(body)))


def self_test() -> int:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, **details: Any) -> None:
        checks.append({"name": name, "passed": bool(condition), **json_safe(details)})

    binary = Path("/synthetic/headless")
    scene = Path("/synthetic/scene.json")
    prefix = Path("/synthetic/output")
    checkpoint = Path("/synthetic/state.checkpoint")
    config = PhysicalConfig(sampler="halton", motion=False, wavefront=False, rr=True)
    computed_invocations = (
        len(CONFIGURATIONS) * 5
        + sum(not item.wavefront for item in CONFIGURATIONS)
        + 4  # static motion on/off for two samplers
        + 2  # checkpoint sampler/motion mismatch
        + 2  # unknown sampler/motion values
        + 4  # two copied-scene baselines and two mutation rejections
        + 1  # valid time-budget diagnostic
        + 7  # invalid time-budget policies
    )
    check(
        "fixed-design-counts",
        len(CONFIGURATIONS) == 16
        and computed_invocations == EXPECTED_COMPLETE_INVOCATIONS == 108,
        matrix_configurations=len(CONFIGURATIONS),
        computed_invocations=computed_invocations,
        expected_invocations=EXPECTED_COMPLETE_INVOCATIONS,
    )
    command = fresh_command(binary, scene, prefix, TARGET_SAMPLES, config)
    check(
        "fresh-command-fixed-options",
        command[:2] == [str(binary), str(scene)]
        and command[command.index("--sampler") + 1] == "halton"
        and command[command.index("--motion") + 1] == "off"
        and command[command.index("--resolution") + 1] == "17x13",
        command=command,
    )
    resume = minimal_resume_command(binary, scene, prefix, TARGET_SAMPLES, checkpoint)
    check(
        "minimal-resume-command",
        resume == [
            str(binary), str(scene), "--samples", "17", "--output", str(prefix),
            "--resume", str(checkpoint),
        ],
        command=resume,
    )

    with tempfile.TemporaryDirectory(prefix="validate-hedge-self-test-") as temporary:
        temporary_path = Path(temporary)
        raw_path = temporary_path / "synthetic.sum.bin"
        values = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
        synthetic_sum(raw_path, 2, 1, 3, values)
        raw = read_sum(raw_path)
        metadata = {
            "scene": str(scene),
            "resolution": [2, 1],
            "target_samples": 3,
            "completed_samples": 3,
            "seed": SEED,
            "sampler": "counter",
            "sampler_id": 2,
            "sampler_version": 2,
            "config": {
                **PhysicalConfig(
                    sampler="counter", motion=True, wavefront=True, rr=True,
                    width=2, height=1,
                ).metadata_config()
            },
            "camera": {
                "position": [0.0, 0.0, 1.0],
                "look_at": [0.0, 0.0, 0.0],
                "aperture": 0.0,
                "focus_distance": 1.0,
                "shutter": [0.0, 1.0],
            },
            "linear_sum_format": {"checksum": fnv1a(raw["encoded"][:-8])},
            "fingerprints": {
                "scene_file": 1,
                "assets": 2,
                "materials": 3,
                "geometry": 4,
                "render_config": 5,
            },
            "compatibility": {
                "baseline_starter_revision": "synthetic-source",
                "build_identifier": "synthetic-build",
            },
            "render_loop_wall_ms": 1.0,
            "run_policy": {
                "time_budget_ms": None,
                "stop_reason": "sample_target",
                "last_sample_wall_ms": 0.4,
            },
            "benchmark_enabled": False,
        }
        expected = expected_render(
            scene,
            PhysicalConfig(
                sampler="counter", motion=True, wavefront=True, rr=True,
                width=2, height=1,
            ),
            3,
            completed_samples=3,
            shutter=(0.0, 1.0),
            camera={
                "position": [0.0, 0.0, 1.0],
                "look_at": [0.0, 0.0, 0.0],
                "aperture": 0.0,
                "focus_distance": 1.0,
                "shutter": [0.0, 1.0],
            },
        )
        audits = audit_loaded_render(metadata, raw, expected, 0.010)
        check(
            "synthetic-metadata-and-range-audits",
            bool(audits) and all(item["passed"] for item in audits),
            failed=[item for item in audits if not item["passed"]],
        )

        ceiling_metadata = json.loads(json.dumps(metadata))
        ceiling_metadata["render_loop_wall_ms"] = 0.5
        ceiling_metadata["run_policy"] = {
            "time_budget_ms": 1,
            "stop_reason": "sample_target",
            "last_sample_wall_ms": 0.4,
        }
        ceiling_expected = expected_render(
            scene,
            PhysicalConfig(
                sampler="counter", motion=True, wavefront=True, rr=True,
                width=2, height=1,
            ),
            3,
            completed_samples=3,
            time_budget_ms=1,
            camera={
                "position": [0.0, 0.0, 1.0],
                "look_at": [0.0, 0.0, 0.0],
                "aperture": 0.0,
                "focus_distance": 1.0,
                "shutter": [0.0, 1.0],
            },
        )
        ceiling_audits = audit_loaded_render(
            ceiling_metadata, raw, ceiling_expected, 0.010
        )
        check(
            "synthetic-budgeted-target-completion-uses-sample-target",
            bool(ceiling_audits)
            and all(item["passed"] for item in ceiling_audits),
            failed=[item for item in ceiling_audits if not item["passed"]],
        )

        bad_metadata = json.loads(json.dumps(metadata))
        bad_metadata["completed_samples"] = 0
        bad_audits = audit_loaded_render(bad_metadata, raw, expected, 0.010)
        check(
            "synthetic-audit-detects-sample-mismatch",
            any(
                item["name"] == "completed-samples" and not item["passed"]
                for item in bad_audits
            )
            and any(
                item["name"] == "raw-samples-match-metadata" and not item["passed"]
                for item in bad_audits
            ),
        )

        time_metadata = json.loads(json.dumps(metadata))
        time_metadata["target_samples"] = 4
        time_metadata["render_loop_wall_ms"] = 1.5
        time_metadata["run_policy"] = {
            "time_budget_ms": 1,
            "stop_reason": "time_budget",
            "last_sample_wall_ms": 0.4,
        }
        time_expected = expected_render(
            scene,
            PhysicalConfig(
                sampler="counter", motion=True, wavefront=True, rr=True,
                width=2, height=1,
            ),
            4,
            completed_range=(1, 4),
            time_budget_ms=1,
            camera={
                "position": [0.0, 0.0, 1.0],
                "look_at": [0.0, 0.0, 0.0],
                "aperture": 0.0,
                "focus_distance": 1.0,
                "shutter": [0.0, 1.0],
            },
        )
        time_audits = audit_loaded_render(
            time_metadata, raw, time_expected, 0.010
        )
        check(
            "synthetic-time-budget-range-and-stop-audits",
            bool(time_audits) and all(item["passed"] for item in time_audits),
            failed=[item for item in time_audits if not item["passed"]],
        )
        premature_metadata = json.loads(json.dumps(time_metadata))
        premature_metadata["render_loop_wall_ms"] = 0.5
        premature_audits = audit_loaded_render(
            premature_metadata, raw, time_expected, 0.010
        )
        check(
            "synthetic-time-budget-audit-detects-premature-stop",
            any(
                item["name"] == "time-budget-reached-before-stop"
                and not item["passed"]
                for item in premature_audits
            ),
        )
        stale_policy_metadata = json.loads(json.dumps(time_metadata))
        stale_policy_metadata["diagnostic_run_policy"] = {}
        stale_policy_metadata["run_policy"]["legacy_extra_field"] = 4
        stale_policy_audits = audit_loaded_render(
            stale_policy_metadata, raw, time_expected, 0.010
        )
        check(
            "synthetic-audit-rejects-stale-or-extended-policy-schema",
            any(
                item["name"] == "obsolete-policy-alias-absent"
                and not item["passed"]
                for item in stale_policy_audits
            )
            and any(
                item["name"] == "run-policy-exact-fields"
                and not item["passed"]
                for item in stale_policy_audits
            ),
        )

        benchmark_path = temporary_path / "synthetic.benchmark.csv"
        benchmark_fields = [
            "record_type", "sample", "bounce", "active_before", "active_after",
            "intersection_ms", "sort_ms", "shade_ms", "compaction_ms", "count_ms",
            "host_ms", "gpu_ms", "measured_samples", "render_loop_wall_ms",
            "queue_ms", "queue_host_ms", "diffuse_ms", "mirror_ms", "dielectric_ms",
            "emission_ms", "terminal_ms", "queued_diffuse", "queued_mirror",
            "queued_dielectric", "queued_emission", "queued_terminal",
            "kernel_dispatches", "rr_tested", "rr_terminated",
        ]
        with benchmark_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=benchmark_fields)
            writer.writeheader()
            bounce = {field: "0" for field in benchmark_fields}
            bounce.update(
                {
                    "record_type": "bounce",
                    "active_before": "10",
                    "active_after": "5",
                    "queued_diffuse": "4",
                    "queued_terminal": "6",
                    "kernel_dispatches": "2",
                    "rr_tested": "2",
                    "rr_terminated": "1",
                }
            )
            writer.writerow(bounce)
            overall = {field: "" for field in benchmark_fields}
            overall.update(
                {
                    "record_type": "overall",
                    "host_ms": "1.0",
                    "gpu_ms": "0.5",
                    "measured_samples": "3",
                    "render_loop_wall_ms": "1.5",
                }
            )
            writer.writerow(overall)
        benchmark_audits, totals = audit_benchmark_csv(
            benchmark_path, wavefront=True, rr=True, completed_samples=3
        )
        check(
            "synthetic-benchmark-counter-audits",
            bool(benchmark_audits) and all(item["passed"] for item in benchmark_audits),
            totals=totals,
            failed=[item for item in benchmark_audits if not item["passed"]],
        )
        benchmark_metadata = {
            "render_loop_wall_ms": 1.5,
            "benchmark": {
                "measured_samples": 3,
                "renderer_host_ms": 1.0,
                "renderer_gpu_ms": 0.5,
            },
        }
        consistency_audits = audit_benchmark_metadata_consistency(
            benchmark_metadata, totals
        )
        check(
            "synthetic-benchmark-csv-metadata-consistency",
            all(item["passed"] for item in consistency_audits),
            failed=[item for item in consistency_audits if not item["passed"]],
        )

        source = temporary_path / "scene.json"
        asset = temporary_path / "mesh.obj"
        asset.write_text("# synthetic\n", encoding="utf-8")
        source.write_text(
            json.dumps(
                {
                    "Materials": {},
                    "Objects": [{"TYPE": "obj", "FILE": "mesh.obj"}],
                    "Camera": {"SHUTTER": [0, 1]},
                }
            ),
            encoding="utf-8",
        )
        absolute, assets = absolute_asset_scene(source)
        check(
            "temporary-scene-assets-become-absolute",
            Path(absolute["Objects"][0]["FILE"]).is_absolute()
            and assets[0]["exists"],
            assets=assets,
        )

    result = {
        "tool": "validate_hedge.py",
        "mode": "self-test",
        "renderer_executed": False,
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=DEFAULT_BINARY,
        help=f"local headless renderer (default: {DEFAULT_BINARY})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="REQUIRED except for --self-test; must name a directory that does not exist",
    )
    parser.add_argument(
        "--benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "benchmark matrix references and audit queue/RR counters "
            "(default: enabled; --no-benchmark disables reference instrumentation)"
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run standard-library synthetic tests without executing the renderer",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.output is None:
        parser.error("--output is required unless --self-test is used")

    output = args.output.expanduser().resolve()
    binary = args.binary.expanduser().resolve()
    if output.exists():
        parser.error(f"--output must be a brand-new directory; already exists: {output}")

    evidence: Evidence | None = None
    try:
        evidence = Evidence(output, binary, args.benchmark)
        run_validation(evidence)
    except Exception as error:
        if evidence is None:
            print(f"error: could not initialize evidence directory: {error}", file=sys.stderr)
            return 2
        evidence.add_check(
            "harness/unhandled-exception",
            False,
            exception=repr(error),
            traceback=traceback.format_exc(),
        )
    passed = evidence.finish()
    summary = evidence.summary()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
