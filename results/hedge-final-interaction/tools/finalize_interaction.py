#!/usr/bin/env python3
"""Validate real capture evidence before atomically writing its public wrapper.

This original helper performs no render, capture, conversion, or remote action.
Use --check-only to run all checks without writing the public summary. A missing
or failed input never produces a successful report.
"""

from __future__ import annotations

import argparse
import datetime as dt
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import subprocess
import sys
import zlib


SOURCE_SHA = "a16c1aba9a7cf17d4d2ffd902e5f78e0967acd06942d399ff3af256029227e86"
DECODER_SHA = "8e820815011447b44e0d3100b002b2279954fa2479371c14ec61218ef505a4e0"
RENDERER_OPTIONS = ["--sort", "off", "--wavefront", "on", "--rr", "on",
                    "--rr-start", "3", "--sampler", "halton", "--motion", "on"]
REFERENCE_CONFIG = {"max_depth": 10, "aa": True, "nee": True, "bvh": True,
                    "compact": True, "sort": False, "wavefront": True,
                    "rr": True, "rr_start": 3, "sampler": "halton",
                    "sampler_id": 3, "motion": True, "debug": "none"}


class EvidenceError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), f"expected a JSON object in {path}")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def file_record(path, root):
    require(path.is_file() and path.stat().st_size > 0, f"missing or empty artifact: {path}")
    return {"path": path.resolve().relative_to(root).as_posix(),
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def local_path(name, root, base=None):
    require(isinstance(name, str) and bool(name), "artifact path must be a nonempty string")
    path = Path(name)
    if not path.is_absolute():
        path = (base or root) / path
    path = path.resolve()
    require(path.is_relative_to(root), f"artifact is outside repository: {name}")
    return path


def asset_key(name):
    require(isinstance(name, str) and bool(name), "scene asset path must be nonempty")
    normalized = name.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if "scenes" in parts:
        normalized = "/".join(parts[parts.index("scenes"):])
    require(not normalized.startswith("/") and ".." not in PurePosixPath(normalized).parts,
            f"unsupported scene asset path: {name}")
    return normalized.removeprefix("./")


def utc(value):
    if isinstance(value, dict):
        value = value.get("utc")
    require(isinstance(value, str), "missing UTC evidence timestamp")
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "evidence timestamp must include timezone")
    return parsed.astimezone(dt.timezone.utc)


def proof_time(document):
    for key in ("timestamp", "utc", "captured_at_utc", "captured_at", "created_at_utc", "created"):
        if key in document:
            return utc(document[key])
    raise EvidenceError("provenance proof has no timestamp")


def renderer_source_hash(root):
    # Reproduce the documented length-delimited build-input protocol. This is
    # separate from the capture, whose before/after proofs are also required.
    files = [root / "CMakeLists.txt"]
    for folder in ("src", "stream_compaction", "external", "cmake"):
        files.extend(path for path in (root / folder).rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        name = path.relative_to(root).as_posix().encode("utf-8")
        size = path.stat().st_size
        digest.update(struct.pack("<Q", len(name)))
        digest.update(name)
        digest.update(struct.pack("<Q", size))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def inspect_png(path):
    data = path.read_bytes()
    require(data[:8] == b"\x89PNG\r\n\x1a\n", f"invalid PNG signature: {path}")
    offset, dimensions, ended = 8, None, False
    while offset < len(data):
        require(offset + 12 <= len(data), f"truncated PNG chunk: {path}")
        size = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4:offset + 8]
        end = offset + 8 + size
        require(end + 4 <= len(data), f"truncated PNG payload: {path}")
        checksum = zlib.crc32(data[offset + 4:end]) & 0xffffffff
        require(checksum == struct.unpack_from(">I", data, end)[0], f"PNG CRC mismatch: {path}")
        if kind == b"IHDR":
            require(dimensions is None and size == 13, f"invalid PNG IHDR: {path}")
            dimensions = struct.unpack_from(">II", data, offset + 8)
            require(data[offset + 16:offset + 18] == bytes([8, 2]),
                    f"PNG must be RGB8: {path}")
        offset = end + 4
        if kind == b"IEND":
            require(size == 0 and offset == len(data), f"invalid PNG ending: {path}")
            ended = True
            break
    require(ended and dimensions == (960, 640), f"PNG must be complete native960x640: {path}")


def inspect_sums(path, samples, metadata):
    data = path.read_bytes()
    require(len(data) == 32 + 960 * 640 * 12 + 8, "reference linear-sum size mismatch")
    header = struct.unpack_from("<8sIIIIQ", data)
    require(header == (b"C565SUM\0", 1, 960, 640, 3, samples),
            "reference linear-sum header mismatch")
    digest = 14695981039346656037
    for byte in memoryview(data)[:-8]:
        digest = ((digest ^ byte) * 1099511628211) & 0xffffffffffffffff
    stored = struct.unpack_from("<Q", data, len(data) - 8)[0]
    require(digest == stored == metadata["linear_sum_format"]["checksum"],
            "reference linear-sum FNV checksum mismatch")
    require(all(math.isfinite(value[0]) for value in struct.iter_unpack("<f", data[32:-8])),
            "reference linear sums contain nonfinite values")
    return {"passed": True, "completed_samples": samples, "checksum": stored,
            "checksum_security": "change/corruption detection only; not cryptographic",
            "preview_raw_sums_available": False, "raw_sum_equality_claimed": False}


def close_numbers(actual, expected):
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            close_numbers(a, b) for a, b in zip(actual, expected))
    return isinstance(actual, (int, float)) and math.isfinite(actual) and math.isclose(
        actual, expected, rel_tol=1e-6, abs_tol=1e-6)


def probe(path, ffprobe):
    command = [ffprobe, "-v", "error", "-count_frames", "-show_streams",
               "-show_format", "-of", "json", str(path)]
    process = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    require(process.returncode == 0, f"ffprobe failed for {path}: {process.stderr.strip()}")
    document = json.loads(process.stdout)
    video = [entry for entry in document.get("streams", []) if entry.get("codec_type") == "video"]
    require(len(video) == 1, f"expected exactly one video stream in {path}")
    stream = video[0]
    duration = float(document["format"]["duration"])
    require(math.isfinite(duration) and duration > 0, f"invalid media duration: {path}")
    count = int(stream["nb_read_frames"])
    require(count > 0, f"no decoded video frames: {path}")
    return {"command": command, "exit_code": process.returncode,
            "codec_name": stream.get("codec_name"), "width": stream.get("width"),
            "height": stream.get("height"), "pix_fmt": stream.get("pix_fmt"),
            "r_frame_rate": stream["r_frame_rate"], "avg_frame_rate": stream["avg_frame_rate"],
            "decoded_frame_count": count, "media_duration_seconds": duration}


def compile_helper(root, source, compiler, build):
    header = root / "external/include/stb_image.h"
    require(sha256(header) == DECODER_SHA, "approved starter PNG decoder header changed")
    build.mkdir(parents=True, exist_ok=True)
    binary = build / "interaction_pixel_compare"
    command = [compiler, "-std=c++17", "-O2", "-I", str(header.parent), str(source), "-o", str(binary)]
    process = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    require(process.returncode == 0, f"pixel helper compilation failed: {process.stderr.strip()}")
    return binary, {"command": command, "exit_code": process.returncode,
                    "source_sha256": sha256(source),
                    "decoder_header_sha256": DECODER_SHA,
                    "external_decoder_implementation_copied": False,
                    "temporary_executable": {**file_record(binary, root),
                                             "purpose": "temporary local verification only",
                                             "delivery_required": False},
                    "executable_delivery_required": False}


def finalize(args):
    root = Path(args.root).resolve()
    result = local_path(args.result_dir, root)
    capture = result / "capture"
    summary = read_json(capture / "summary.json")
    actions = read_json(capture / "actions.json")
    require(summary.get("passed") is True and summary.get("status") == "succeeded",
            "capture summary is not a successful real recording")
    require(actions.get("status") == "succeeded", "capture actions did not succeed")
    require(summary.get("cleanup", {}).get("errors") == [], "capture cleanup recorded errors")
    selected = [item for item in actions["png_checks"] if item.get("created_by") == "scheduled-S-save"]
    require(len(selected) == 1, "expected exactly one PNG created by scheduled-S-save")
    saved_check = selected[0]
    name = saved_check["name"]
    require(Path(name).name == name and name.endswith(".png"), "unsafe scheduled-save PNG name")
    sample_match = re.search(r"\.([1-9][0-9]*)samp\.save[0-9]+\.png$", name)
    require(sample_match is not None, "scheduled-save filename has no actual sample count")
    samples = int(sample_match.group(1))
    saved = capture / name
    saved_record = file_record(saved, root)
    inventory = [item for item in summary["png_files"] if item.get("name") == name]
    require(len(inventory) == 1, "scheduled save missing or repeated in capture PNG inventory")
    for item in (saved_check, inventory[0]):
        require(item.get("sha256") == saved_record["sha256"] and item.get("bytes") == saved_record["bytes"],
                "scheduled-save file disagrees with recorded SHA256/bytes")
        require(item.get("png_complete") is True and (item.get("width"), item.get("height")) == (960, 640),
                "scheduled-save evidence lacks a complete native PNG")
    inspect_png(saved)
    stdout_name = summary["renderer"]["stdout_log"]
    require(Path(stdout_name).name == stdout_name, "unsafe renderer stdout path")
    stdout = (capture / stdout_name).read_text(encoding="utf-8")
    require(stdout.splitlines().count(f"Saved {name}.") == 1,
            "renderer stdout does not uniquely confirm scheduled-save filename/sample count")
    for label in ("restore-original-camera", "scheduled-S-save"):
        events = [item for item in actions["events"] if item.get("label") == label]
        require(len(events) == 1 and events[0].get("status") == "completed", f"missing completed {label} event")
    require(summary["capture_geometry"].get("native_client_verified") is True,
            "capture did not verify native client dimensions")

    before_path, after_path = result / "capture-provenance-before.json", result / "capture-provenance-after.json"
    before, after = read_json(before_path), read_json(after_path)
    require(before.get("renderer_source_sha256") == after.get("renderer_source_sha256") == SOURCE_SHA,
            "capture provenance source identities differ from frozen renderer")
    require(renderer_source_hash(root) == SOURCE_SHA, "local renderer inputs differ from frozen source identity")
    require(before.get("binary_sha256") == after.get("binary_sha256"), "capture provenance binary hashes changed")
    binaries = before["binary_sha256"]
    require(all(valid_hash(binaries.get(kind)) for kind in ("headless", "preview")), "invalid provenance binary SHA256")
    assets = {}
    for proof in (before, after):
        current = {}
        require(isinstance(proof.get("scene_assets"), list) and proof["scene_assets"], "missing scene asset provenance")
        for item in proof["scene_assets"]:
            key = asset_key(item["path"])
            require(key not in current and valid_hash(item.get("sha256")), "duplicate or invalid scene asset provenance")
            actual = file_record(local_path(key, root), root)
            require(actual["bytes"] == item.get("bytes") and actual["sha256"] == item["sha256"],
                    f"local scene asset differs from provenance: {key}")
            current[key] = {"bytes": item["bytes"], "sha256": item["sha256"]}
        if assets:
            require(current == assets, "capture provenance scene asset identities changed")
        assets = current
    require(proof_time(before) <= utc(summary["created"]) <= utc(summary["completed"]) <= proof_time(after),
            "capture provenance timestamps do not bracket the real recording")
    inputs = summary["inputs"]
    scene_key = asset_key(inputs["scene"])
    require(scene_key in assets, "recorded scene is absent from provenance assets")
    actual_inputs = {"binary_sha256": binaries["preview"], "scene_sha256": assets[scene_key]["sha256"]}
    require(all(inputs.get(key) == value for key, value in actual_inputs.items()),
            "actual capture binary/scene hashes differ from provenance")
    require(inputs.get("hashes_after") == inputs.get("hashes_after_cleanup") == actual_inputs,
            "capture input hashes changed during recording or cleanup")
    require(inputs.get("unchanged_during_recording") is True and inputs.get("unchanged_after_cleanup") is True,
            "capture input immutability checks did not pass")
    require(inputs.get("renderer_options") == RENDERER_OPTIONS
            and summary["renderer"]["command"][2:] == RENDERER_OPTIONS,
            "capture renderer options differ from required final settings")
    require(summary["arguments"].get("sampler") == "halton" and summary["arguments"].get("motion") == "on",
            "capture selected sampler/motion settings differ")
    scene_path = local_path(scene_key, root)
    scene_document = read_json(scene_path)
    scene_camera = scene_document["Camera"]
    for item in scene_document.get("Objects", []):
        if "FILE" in item:
            dependency = local_path(item["FILE"], root, scene_path.parent).relative_to(root).as_posix()
            require(dependency in assets, f"scene file dependency absent from provenance: {dependency}")
    require(inputs["scene_camera"] == {"resolution": scene_camera["RES"], "iterations": scene_camera["ITERATIONS"],
            "depth": scene_camera["DEPTH"], "image_name": scene_camera["FILE"]}, "recorded scene camera differs from local scene")
    require(scene_camera["RES"] == [960, 640] and scene_camera["DEPTH"] == 10,
            "capture camera is not native960x640 depth10")

    reference = result / "post-space-reference.png"
    metadata_path = result / "post-space-reference.json"
    sum_path = result / "post-space-reference.sum.bin"
    metadata = read_json(metadata_path)
    require(metadata.get("resolution") == [960, 640] and metadata.get("seed") == 1,
            "headless reference resolution/seed mismatch")
    require(metadata.get("completed_samples") == metadata.get("target_samples") == samples,
            "headless reference did not render the actual scheduled-save sample count")
    require(metadata.get("sampler") == "halton" and metadata.get("sampler_id") == metadata.get("sampler_version") == 3,
            "headless reference Halton identity mismatch")
    require(all(type(metadata["config"].get(key)) is type(value) and metadata["config"][key] == value
                for key, value in REFERENCE_CONFIG.items()), "headless reference renderer settings mismatch")
    require(metadata.get("resumed_from") is None and metadata["run_policy"].get("stop_reason") == "sample_target",
            "headless reference must be an independent fresh sample-target run")
    require(asset_key(metadata["scene"]) == scene_key, "headless reference scene differs from captured scene")
    build_id = metadata["compatibility"]["build_identifier"]
    require(isinstance(build_id, str) and f"_source-{SOURCE_SHA}_" in build_id,
            "reference metadata build does not identify the frozen source")
    for field, scene_field in (("position", "EYE"), ("look_at", "LOOKAT"), ("aperture", "APERTURE"),
                               ("focus_distance", "FOCUS"), ("shutter", "SHUTTER")):
        require(close_numbers(metadata["camera"][field], scene_camera[scene_field]),
                f"headless reference camera {field} differs from original scene")
    inspect_png(reference)
    sum_check = inspect_sums(sum_path, samples, metadata)
    helper_source = local_path(args.pixel_source, root)
    helper, helper_build = compile_helper(root, helper_source, args.compiler, local_path(args.build_dir, root))
    png_bytes_equal = saved.read_bytes() == reference.read_bytes()
    helper_stdout = None
    if png_bytes_equal:
        comparison = {"passed": True, "method": "exact PNG byte comparison", "outcome": "PNG bytes identical",
                      "details": {"png_bytes_equal": True, "rgb8_equal": True,
                                  "maximum_absolute_channel_difference": 0,
                                  "raw_sum_equality_claimed": False}}
    else:
        process = subprocess.run([str(helper), str(saved), str(reference)], capture_output=True,
                                 text=True, timeout=60, check=False)
        require(process.returncode in (0, 1), f"RGB decoder helper failed: {process.stderr.strip()}")
        decoded = json.loads(process.stdout)
        helper_stdout = process.stdout
        require(process.returncode == 0 and decoded.get("passed") is True
                and decoded.get("maximum_absolute_channel_difference", 256) <= 1,
                "post-Space saved image exceeds the RGB8 tolerance of one channel level")
        note = ("PNG bytes differ; decoded RGB8 pixels match exactly." if decoded["rgb8_equal"] else
                "PNG bytes and some RGB8 channels differ. Every channel differs by at most 1 of 255 levels; "
                "this is consistent with the known preview division versus headless reciprocal multiplication "
                "normalization difference. The cause has not been independently reproduced, and neither "
                "exact image equality nor raw accumulation equality is claimed.")
        comparison = {"passed": True, "method": "PNG bytes first, then independently authored RGB8 decoder wrapper",
                      "outcome": "decoded RGB8 identical" if decoded["rgb8_equal"] else "bounded RGB8 differences",
                      "details": {**decoded, "png_bytes_equal": False, "disclosure": note,
                                  "raw_sum_equality_claimed": False}}

    conversion_path = result / "conversion.json"
    conversion = read_json(conversion_path)
    require(conversion.get("passed") is True, "media conversion evidence did not pass")
    visual = conversion.get("root_visual_verification", {})
    require(visual.get("passed") is True and visual.get("inspector") == "root"
            and isinstance(visual.get("frame_files"), list) and bool(visual["frame_files"])
            and bool(visual.get("statement")), "root visual/privacy frame verification is missing")
    inspected_frames = [file_record(local_path(name, root, result), root) for name in visual["frame_files"]]
    command = conversion.get("ffmpeg_command")
    require(isinstance(command, list) and command and all(isinstance(item, str) for item in command)
            and Path(command[0]).name == "ffmpeg", "conversion lacks its actual ffmpeg argv")
    gif = root / "img/interaction.gif"
    published_mp4 = root / "img/interaction.mp4"
    captured_mp4 = capture / "interaction.mp4"
    require(captured_mp4.read_bytes() == published_mp4.read_bytes(), "published MP4 differs from actual capture MP4")
    require(local_path(conversion["gif"]["path"], root) == gif
            and local_path(conversion["mp4"]["path"], root) == published_mp4, "conversion output paths mismatch")
    conversion_cwd = local_path(conversion.get("cwd", "."), root)
    input_indices = [index + 1 for index, word in enumerate(command) if word == "-i"]
    require(input_indices and all(index < len(command) for index in input_indices),
            "conversion ffmpeg argv has no input file")
    require(local_path(command[input_indices[0]], root, conversion_cwd) in (captured_mp4, published_mp4),
            "conversion ffmpeg argv does not use the actual capture MP4")
    require(local_path(command[-1], root, conversion_cwd) == gif,
            "conversion ffmpeg argv does not produce the GIF artifact")
    capture_probe, gif_probe = probe(captured_mp4, args.ffprobe), probe(gif, args.ffprobe)
    require((capture_probe["codec_name"], capture_probe["width"], capture_probe["height"], capture_probe["pix_fmt"])
            == ("h264", 1280, 720, "yuv420p"), "actual capture MP4 format mismatch")
    capture_fps = summary["arguments"]["capture_fps"]
    require(type(capture_fps) is int and 1 <= capture_fps <= 240, "invalid recorded capture-clock rate")
    for field in ("r_frame_rate", "avg_frame_rate"):
        require(abs(float(Fraction(capture_probe[field])) - capture_fps) <= max(.05, capture_fps * .01),
                f"actual capture {field} differs from capture-clock setting")
    recorded_duration = float(summary["ffprobe"]["validated"]["media_duration_seconds"])
    require(abs(capture_probe["media_duration_seconds"] - recorded_duration) <= .01,
            "actual MP4 duration differs from capture ffprobe evidence")
    require(gif_probe["codec_name"] == "gif", "GIF artifact has another codec")
    gif_fps = float(conversion["gif"]["fps"])
    require(math.isfinite(gif_fps) and gif_fps > 0, "invalid GIF conversion fps")
    require(abs(float(Fraction(gif_probe["avg_frame_rate"])) - gif_fps) <= .1,
            "actual GIF average rate differs from conversion evidence")
    require(abs(gif_probe["media_duration_seconds"] - float(conversion["gif"]["duration_seconds"])) <= .01,
            "actual GIF duration differs from conversion evidence")
    log_dir = local_path(args.verification_log_dir, root) if args.verification_log_dir else result / "tools/verification-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    comparison_log = log_dir / "post-space-comparison.json"
    comparison_log.write_text(json.dumps({"comparison": comparison, "saved_image": saved_record,
                                         "reference_image": file_record(reference, root),
                                         "helper_source": file_record(helper_source, root),
                                         "helper_build": helper_build}, indent=2, sort_keys=True, allow_nan=False) + "\n",
                              encoding="utf-8")
    helper_stdout_log = None
    if helper_stdout is not None:
        helper_stdout_log = log_dir / "pixel-compare.stdout.json"
        helper_stdout_log.write_text(helper_stdout, encoding="utf-8")
    artifacts = {"scheduled_S_save": saved_record, "headless_reference_png": file_record(reference, root),
                 "headless_reference_metadata": file_record(metadata_path, root),
                 "headless_reference_linear_sum": file_record(sum_path, root),
                 "GIF": file_record(gif, root), "MP4": file_record(published_mp4, root),
                 "capture_MP4": file_record(captured_mp4, root),
                 "finalizer_source": file_record(Path(__file__).resolve(), root),
                 "pixel_compare_source": file_record(helper_source, root),
                 "comparison_result_log": file_record(comparison_log, root),
                 "approved_starter_decoder_header": file_record(root / "external/include/stb_image.h", root),
                 "recording_script": file_record(root / "scripts/record_interaction_video.py", root),
                 "provenance_before": file_record(before_path, root), "provenance_after": file_record(after_path, root),
                 "conversion_evidence": file_record(conversion_path, root),
                 "capture_summary": file_record(capture / "summary.json", root),
                 "capture_actions": file_record(capture / "actions.json", root),
                 "renderer_stdout": file_record(capture / stdout_name, root)}
    if helper_stdout_log is not None:
        artifacts["pixel_compare_stdout_log"] = file_record(helper_stdout_log, root)
    return {"schema_version": 1, "passed": True, "status": "passed",
            "renderer_source_sha256": SOURCE_SHA, "headless_binary_sha256": binaries["headless"],
            "preview_binary_sha256": binaries["preview"], "capture_summary": "capture/summary.json",
            "post_space_comparison": comparison, "post_space_saved_samples": samples,
            "reference_build_identifier": build_id, "reference_linear_sum_validation": sum_check,
            "capture_settings": {"renderer_argv": summary["renderer"]["command"],
                                 "scene_camera": inputs["scene_camera"], "capture_clock_fps": capture_fps,
                                 "capture_argv": summary["capture"]["command"],
                                 "planned_interaction_duration_seconds": summary["arguments"]["duration_seconds"],
                                 "rate_disclosure": "capture_clock_fps is screen sampling; no renderer FPS claim is made",
                                 "sampler": "halton", "motion": "on"},
            "headless_reference_settings": {"resolution": metadata["resolution"], "seed": metadata["seed"],
                                            "samples": samples, "sampler_version": 3, "config": metadata["config"]},
            "media": {"capture_ffprobe": capture_probe, "gif_ffprobe": gif_probe,
                      "conversion_ffmpeg_argv": command, "root_visual_verification": visual,
                      "inspected_frame_artifacts": inspected_frames},
            "provenance": {"before_after_identity_verified": True,
                           "recording_bracketed_by_proof_timestamps": True,
                           "local_renderer_inputs_verified": True, "scene_assets": assets},
            "helper_build": helper_build, "artifacts": artifacts,
            "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path.cwd()))
    parser.add_argument("--result-dir", default="results/hedge-final-interaction")
    parser.add_argument("--pixel-source", default=".local/interaction_pixel_compare.cpp")
    parser.add_argument("--build-dir", default=".local/interaction-helper-build")
    parser.add_argument("--verification-log-dir", help="comparison/stdout logs (default RESULT/tools/verification-logs)")
    parser.add_argument("--compiler", default=shutil.which("c++") or "c++")
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe") or "ffprobe")
    parser.add_argument("--check-only", action="store_true", help="validate without writing the public report")
    args = parser.parse_args()
    try:
        report = finalize(args)
        if not args.check_only:
            root = Path(args.root).resolve()
            destination = local_path(args.result_dir, root) / "summary.json"
            temporary = destination.with_name(f".summary.{os.getpid()}.tmp")
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        print(json.dumps({"passed": True, "check_only": args.check_only,
                          "saved_samples": report["post_space_saved_samples"],
                          "comparison": report["post_space_comparison"],
                          "capture_duration_seconds": report["media"]["capture_ffprobe"]["media_duration_seconds"],
                          "gif_duration_seconds": report["media"]["gif_ffprobe"]["media_duration_seconds"]}, indent=2))
        return 0
    except (EvidenceError, OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        print(f"interaction finalization failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
