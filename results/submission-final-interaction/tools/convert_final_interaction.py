#!/usr/bin/env python3
"""Prepare verified new media and eight real frames, then publish after root review.

prepare leaves existing img files untouched. publish revalidates the new capture
and independent reference and requires a review bound to all eight frame hashes.
No expected renderer hash is inferred from prior evidence.
"""
from __future__ import annotations
import argparse
import datetime as dt
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import zlib

FRAME_TIMES = (2.0, 12.0, 16.0, 23.0, 30.0, 36.0, 44.5, 46.5)

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def artifact(path, root):
    require(path.is_file() and path.stat().st_size > 0, f"missing/empty artifact: {path}")
    return {"path": path.resolve().relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}

def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value

def path_inside(root, name):
    value = Path(name)
    result = (value if value.is_absolute() else root / value).resolve()
    require(result.is_relative_to(root), f"artifact is outside repository: {name}")
    return result

def write_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)

def configuration(args):
    root = Path(args.root).resolve()
    require(re.fullmatch(r"[0-9a-f]{64}", args.expected_source_sha256) is not None, "explicit frozen source SHA256 required")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.prefix) is not None, "invalid evidence prefix")
    return root, path_inside(root, args.result_dir or f"results/{args.prefix}-interaction"), path_inside(root, args.finalizer)

def verify_new_capture(args):
    root, result, finalizer = configuration(args)
    module = runpy.run_path(str(finalizer), run_name="conversion_capture_verifier")
    parameters = SimpleNamespace(root=str(root), result_dir=str(result), prefix=args.prefix,
        expected_source_sha256=args.expected_source_sha256, renderer_build_dir=args.renderer_build_dir,
        pixel_source=args.pixel_source, build_dir=args.verification_build_dir, compiler=args.compiler,
        ffprobe=args.ffprobe, verification_log_dir=None, capture_reference_only=True, check_only=True)
    proof = module["finalize"](parameters)
    require(proof.get("passed") is True and proof.get("renderer_source_sha256") == args.expected_source_sha256
            and proof.get("evidence_prefix") == args.prefix, "new capture/reference verification failed")
    return proof

def run_logged(command, root, result, stem):
    started = now()
    process = subprocess.run(command, cwd=str(root), stdin=subprocess.DEVNULL, capture_output=True,
                             text=True, timeout=300, check=False, shell=False)
    for channel in ("stdout", "stderr"):
        (result / f"{stem}.{channel}.log").write_text(getattr(process, channel), encoding="utf-8")
    require(process.returncode == 0, f"{stem} failed: {process.stderr.strip()}")
    return process.stdout, {"argv": command, "cwd": str(root), "shell": False, "started_at_utc": started,
        "ended_at_utc": now(), "exit_code": process.returncode, "stdout_log": f"{stem}.stdout.log", "stderr_log": f"{stem}.stderr.log"}

def probe_gif(path, args, root, result):
    stdout, receipt = run_logged([args.ffprobe, "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)], root, result, "gif-ffprobe")
    document = json.loads(stdout)
    streams = [item for item in document.get("streams", []) if item.get("codec_type") == "video"]
    require(len(streams) == 1, "expected one GIF video stream")
    stream = streams[0]; duration = float(document["format"]["duration"])
    require(math.isfinite(duration) and duration > 0 and int(stream["nb_read_frames"]) > 0, "incomplete GIF")
    require((stream["codec_name"], stream["width"], stream["height"]) == ("gif", 640, 360), "GIF format mismatch")
    require(abs(float(Fraction(stream["avg_frame_rate"])) - args.gif_fps) <= .1, "GIF media rate mismatch")
    write_json(result / "gif-ffprobe.json", document)
    return {"duration_seconds": duration, "avg_frame_rate": stream["avg_frame_rate"], "decoded_frames": int(stream["nb_read_frames"]), "receipt": receipt}

def require_frame_png(path):
    data = path.read_bytes(); offset = 8; dimensions = None; complete = False
    require(data[:8] == b"\x89PNG\r\n\x1a\n", "privacy frame PNG signature mismatch")
    while offset < len(data):
        require(offset + 12 <= len(data), "privacy frame PNG chunk truncated")
        size = struct.unpack_from(">I", data, offset)[0]; kind = data[offset + 4:offset + 8]; end = offset + 8 + size
        require(end + 4 <= len(data), "privacy frame PNG payload truncated")
        require(zlib.crc32(data[offset + 4:end]) & 0xffffffff == struct.unpack_from(">I", data, end)[0], "privacy frame PNG CRC mismatch")
        if kind == b"IHDR":
            require(dimensions is None and size == 13, "privacy frame PNG IHDR invalid")
            dimensions = struct.unpack_from(">II", data, offset + 8)
        offset = end + 4
        if kind == b"IEND":
            complete = size == 0 and offset == len(data); break
    require(complete and dimensions == (1280, 720), "privacy frame must be a complete1280x720 capture extraction")

def check_artifact(record, root):
    require(artifact(path_inside(root, record["path"]), root) == record, f"artifact changed: {record['path']}")

def prepare(args):
    root, result, finalizer = configuration(args)
    require(not any((result / name).exists() for name in ("conversion-preparation.json", "media-pending", "inspected-frames")), "fresh preparation outputs required")
    proof = verify_new_capture(args); capture = result / "capture/interaction.mp4"
    require(artifact(capture, root) == proof["artifacts"]["capture_MP4"], "capture changed during verification")
    duration = proof["capture_ffprobe"]["media_duration_seconds"]
    require(all(0 <= time < duration - .01 for time in FRAME_TIMES), "capture is too short for eight inspection frames")
    require(math.isfinite(args.gif_fps) and 0 < args.gif_fps <= 30, "GIF fps must be in (0,30]")
    pending, frames = result / "media-pending", result / "inspected-frames"; pending.mkdir(); frames.mkdir()
    gif, mp4 = pending / "interaction.gif", pending / "interaction.mp4"
    shutil.copyfile(capture, mp4); require(digest(capture) == digest(mp4), "pending MP4 copy mismatch")
    command = [args.ffmpeg, "-nostdin", "-v", "error", "-n", "-i", str(capture), "-filter_complex",
        f"fps={args.gif_fps:g},scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3", "-loop", "0", str(gif)]
    _, gif_receipt = run_logged(command, root, result, "gif-conversion"); gif_probe = probe_gif(gif, args, root, result)
    require(abs(gif_probe["duration_seconds"] - duration) <= 1 / args.gif_fps + .05,
            "GIF duration is inconsistent with the full actual capture")
    frame_artifacts, frame_commands = [], []
    for index, time in enumerate(FRAME_TIMES):
        frame = frames / f"frame-{index + 1:02d}-t{time:g}.png"
        argv = [args.ffmpeg, "-nostdin", "-v", "error", "-n", "-ss", f"{time:g}", "-i", str(capture), "-frames:v", "1", str(frame)]
        _, receipt = run_logged(argv, root, result, f"privacy-frame-{index + 1:02d}")
        require_frame_png(frame); frame_artifacts.append(artifact(frame, root)); frame_commands.append({"media_time_seconds": time, **receipt})
    require(artifact(capture, root) == proof["artifacts"]["capture_MP4"], "capture changed during conversion")
    receipt = {"schema_version": 1, "status": "awaiting_root_visual_verification", "passed": False,
        "evidence_prefix": args.prefix, "renderer_source_sha256": args.expected_source_sha256, "renderer_build_dir": args.renderer_build_dir,
        "prepared_at_utc": now(), "capture_and_reference_verified": proof, "capture_MP4": artifact(capture, root),
        "converter_source": artifact(Path(__file__), root), "finalizer_source": artifact(finalizer, root),
        "pending_GIF": artifact(gif, root), "pending_MP4": artifact(mp4, root), "ffmpeg_command": command, "cwd": str(root),
        "gif_conversion": gif_receipt, "gif_probe": gif_probe, "gif_fps": args.gif_fps,
        "frame_artifacts": frame_artifacts, "frame_extraction_commands": frame_commands,
        "published_img_files_changed": False, "root_visual_verification_performed": False}
    write_json(result / "conversion-preparation.json", receipt)
    return receipt

def validate_review(review, prepared, args):
    require(review.get("passed") is True and review.get("inspector") == "root" and bool(review.get("statement")), "eight-frame root visual/privacy review required")
    require(review.get("renderer_source_sha256") == args.expected_source_sha256 and review.get("capture_mp4_sha256") == prepared["capture_MP4"]["sha256"], "review source/capture binding mismatch")
    require(review.get("frame_artifacts") == prepared["frame_artifacts"] and len(review["frame_artifacts"]) == 8, "review must bind the exact eight frame paths/bytes/SHA256 values")

def replace_media(root, sources):
    images, private = root / "img", root / ".local"; images.mkdir(exist_ok=True); private.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="interaction-publication-", dir=private) as temporary:
        backups, staged = {}, {}
        try:
            for name, source in sources.items():
                destination = images / name
                if destination.exists():
                    backups[name] = Path(temporary) / name; shutil.copyfile(destination, backups[name])
                target = images / f".{name}.{os.getpid()}.tmp"; require(not target.exists(), "stale publication temporary")
                shutil.copyfile(source, target); require(digest(target) == digest(source), "publication copy mismatch"); staged[name] = target
            for name, target in staged.items():
                os.replace(target, images / name)
        except Exception:
            for name in staged:
                if name in backups:
                    shutil.copyfile(backups[name], images / name)
                else:
                    (images / name).unlink(missing_ok=True)
            raise
        finally:
            for target in staged.values(): target.unlink(missing_ok=True)

def publish(args):
    root, result, _ = configuration(args)
    require(args.visual_verification is not None, "publish requires --visual-verification with root's bound review JSON")
    prepared = read_json(result / "conversion-preparation.json")
    require(prepared.get("renderer_source_sha256") == args.expected_source_sha256 and prepared.get("evidence_prefix") == args.prefix, "prepared media belongs to another source/prefix")
    review_path = path_inside(root, args.visual_verification); review = read_json(review_path); validate_review(review, prepared, args)
    proof = verify_new_capture(args); require(proof["artifacts"]["capture_MP4"] == prepared["capture_MP4"], "capture changed since preparation")
    for key in ("capture_MP4", "pending_GIF", "pending_MP4"): check_artifact(prepared[key], root)
    for frame in prepared["frame_artifacts"]: check_artifact(frame, root); require_frame_png(path_inside(root, frame["path"]))
    pending_mp4, pending_gif = path_inside(root, prepared["pending_MP4"]["path"]), path_inside(root, prepared["pending_GIF"]["path"])
    require(digest(pending_mp4) == prepared["capture_MP4"]["sha256"], "pending MP4 differs from actual new capture")
    require(not (result / "conversion.json").exists(), "publication receipt already exists")
    replace_media(root, {"interaction.mp4": pending_mp4, "interaction.gif": pending_gif})
    receipt = {"schema_version": 1, "passed": True, "status": "published", "published_at_utc": now(),
        "evidence_prefix": args.prefix, "renderer_source_sha256": args.expected_source_sha256, "renderer_build_dir": args.renderer_build_dir,
        "converter_source": artifact(Path(__file__), root), "ffmpeg_command": prepared["ffmpeg_command"], "cwd": prepared["cwd"],
        "capture_MP4": prepared["capture_MP4"], "gif": {"path": "img/interaction.gif", "generation_path": prepared["pending_GIF"]["path"],
            "duration_seconds": prepared["gif_probe"]["duration_seconds"], "fps": prepared["gif_fps"], "sha256": prepared["pending_GIF"]["sha256"], "bytes": prepared["pending_GIF"]["bytes"]},
        "mp4": {"path": "img/interaction.mp4", "sha256": prepared["pending_MP4"]["sha256"], "bytes": prepared["pending_MP4"]["bytes"]},
        "root_visual_verification": {**review, "frame_files": [frame["path"] for frame in prepared["frame_artifacts"]]},
        "root_review_artifact": artifact(review_path, root), "preparation_artifact": artifact(result / "conversion-preparation.json", root),
        "frame_extraction_commands": prepared["frame_extraction_commands"], "rate_disclosure": "media sampling rates only; no renderer FPS claim"}
    write_json(result / "conversion.json", receipt)
    return receipt

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "publish")); parser.add_argument("--root", default=str(Path.cwd()))
    parser.add_argument("--expected-source-sha256", required=True); parser.add_argument("--prefix", default="submission-final")
    parser.add_argument("--result-dir", help="default results/PREFIX-interaction"); parser.add_argument("--renderer-build-dir", default="build-submission")
    parser.add_argument("--finalizer", default=".local/finalize_interaction.py"); parser.add_argument("--pixel-source", default=".local/interaction_pixel_compare.cpp")
    parser.add_argument("--verification-build-dir", default=".local/interaction-helper-build"); parser.add_argument("--compiler", default=shutil.which("c++") or "c++")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg"); parser.add_argument("--ffprobe", default=shutil.which("ffprobe") or "ffprobe")
    parser.add_argument("--gif-fps", type=float, default=4); parser.add_argument("--visual-verification")
    args = parser.parse_args()
    try:
        receipt = prepare(args) if args.phase == "prepare" else publish(args)
        print(json.dumps({"status": receipt["status"], "evidence_prefix": args.prefix, "published": args.phase == "publish",
            "frame_artifacts": receipt.get("frame_artifacts", receipt.get("root_visual_verification", {}).get("frame_artifacts"))}, indent=2))
        return 0
    except Exception as error:
        print(f"interaction conversion failed: {error}", file=sys.stderr); return 1

if __name__ == "__main__": raise SystemExit(main())
