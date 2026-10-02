#!/usr/bin/env python3
"""Run the authorized final recording and its independent sample-matched render.

This original controller uses the existing recorder and renderer unchanged.
All subprocesses use argv arrays and shell=False. Invoke only after the intended
X11 display is ready. It never converts or publishes media or writes summary.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time


EXPECTED_SOURCE_SHA256 = "a16c1aba9a7cf17d4d2ffd902e5f78e0967acd06942d399ff3af256029227e86"
EXPECTED_CONFIG = {"max_depth": 10, "aa": True, "nee": True, "bvh": True,
                   "compact": True, "sort": False, "wavefront": True,
                   "rr": True, "rr_start": 3, "sampler": "halton",
                   "sampler_id": 3, "motion": True, "debug": "none"}


class ControllerError(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise ControllerError(message)


def now():
    return {"utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "monotonic_ns": time.monotonic_ns()}


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_file(path, root):
    need(path.is_file(), f"required file missing: {path}")
    resolved = path.resolve()
    name = resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else str(resolved)
    return {"path": name, "bytes": path.stat().st_size, "sha256": digest_file(path)}


def load_document(path):
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    need(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def write_document(path, document, *, fresh=False):
    if fresh:
        need(not path.exists(), f"refusing to overwrite evidence: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def inside_root(root, name):
    value = Path(name)
    resolved = (value if value.is_absolute() else root / value).resolve()
    need(resolved.is_relative_to(root), f"path is outside ROOT: {name}")
    return resolved


def capture_command(python, root, preview, scene, output):
    return [python, str(root / "scripts/record_interaction_video.py"),
            "--binary", str(preview), "--scene", str(scene), "--output", str(output),
            "--sampler", "halton", "--motion", "on", "--duration", "48", "--fps", "30"]


def reference_command(headless, scene, samples, prefix):
    need(type(samples) is int and samples > 0, "reference needs the actual positive scheduled-save sample count")
    return [str(headless), str(scene), "--seed", "1", "--resolution", "960x640", "--depth", "10",
            "--aa", "on", "--nee", "on", "--bvh", "on", "--compact", "on",
            "--sort", "off", "--wavefront", "on", "--rr", "on", "--rr-start", "3",
            "--sampler", "halton", "--motion", "on", "--samples", str(samples), "--output", str(prefix)]


def take_snapshot(root, python, preview, headless, scene, kind):
    command = [python, str(root / "scripts/source_hash.py")]
    process = subprocess.run(command, cwd=str(root), capture_output=True, text=True,
                             timeout=120, check=False, shell=False)
    need(process.returncode == 0, f"renderer input hashing failed: {process.stderr.strip()}")
    source_hash = process.stdout.strip()
    need(re.fullmatch(r"[0-9a-f]{64}", source_hash) is not None, "renderer input hasher did not produce a SHA256 identity")
    files = sorted((path for path in (root / "scenes").rglob("*") if path.is_file()),
                   key=lambda path: path.relative_to(root).as_posix())
    need(scene in files, "the target scene must be among the hashed scenes/assets files")
    snapshot = {"schema_version": 1, "kind": kind,
                "renderer_source_sha256": source_hash,
                "binary_sha256": {"headless": digest_file(headless), "preview": digest_file(preview)},
                "binary_paths": {"headless": str(headless), "preview": str(preview)},
                "scene": str(scene), "scene_assets": [record_file(path, root) for path in files],
                "scene_assets_scope": "all regular files recursively under scenes, including the selected scene and all assets",
                "recorder_script": record_file(root / "scripts/record_interaction_video.py", root),
                "source_hash_script": record_file(root / "scripts/source_hash.py", root),
                "source_hash_invocation": {"argv": command, "cwd": str(root), "exit_code": process.returncode,
                                           "stdout": process.stdout, "stderr": process.stderr}}
    # The before timestamp is after its hashing and before recorder startup;
    # the after timestamp is after recorder completion and its hashing.
    snapshot["timestamp"] = now()
    return snapshot


def identical_inputs(before, after):
    for key in ("renderer_source_sha256", "binary_sha256", "binary_paths", "scene_assets",
                "recorder_script", "source_hash_script"):
        need(before[key] == after[key], f"input identity changed: {key}")


def select_scheduled_save(capture):
    summary = load_document(capture / "summary.json")
    actions = load_document(capture / "actions.json")
    need(summary.get("passed") is True and summary.get("status") == "succeeded",
         "existing recorder did not report a successful recording")
    need(actions.get("status") == "succeeded", "existing recorder actions did not succeed")
    candidates = [item for item in actions.get("png_checks", []) if item.get("created_by") == "scheduled-S-save"]
    need(len(candidates) == 1, "capture must contain exactly one scheduled-S-save PNG check")
    item = candidates[0]
    name = item["name"]
    need(isinstance(name, str) and Path(name).name == name, "scheduled-save PNG name is unsafe")
    match = re.search(r"\.([1-9][0-9]*)samp\.save[0-9]+\.png$", name)
    need(match is not None, "scheduled-save PNG filename has no actual sample count")
    actual = record_file(capture / name, capture)
    inventory = [entry for entry in summary.get("png_files", []) if entry.get("name") == name]
    need(len(inventory) == 1, "scheduled-save PNG missing or duplicated in recorder inventory")
    for entry in (item, inventory[0]):
        need(entry.get("sha256") == actual["sha256"] and entry.get("bytes") == actual["bytes"],
             "scheduled-save PNG bytes/SHA256 differ from recorder evidence")
        need(entry.get("png_complete") is True and entry.get("width") == 960 and entry.get("height") == 640,
             "scheduled-save PNG evidence is not complete native960x640")
    log = summary["renderer"]["stdout_log"]
    need(isinstance(log, str) and Path(log).name == log, "renderer stdout-log path is unsafe")
    need((capture / log).read_text(encoding="utf-8").splitlines().count(f"Saved {name}.") == 1,
         "renderer stdout does not uniquely confirm the scheduled save")
    return {"samples": int(match.group(1)), "name": name, "bytes": actual["bytes"], "sha256": actual["sha256"],
            "created_by": "scheduled-S-save", "renderer_stdout_log": log}


def confirm_capture_inputs(capture, snapshot):
    summary = load_document(capture / "summary.json")
    inputs = summary["inputs"]
    scene_path = Path(snapshot["scene"])
    assets = {entry["path"]: entry for entry in snapshot["scene_assets"]}
    # Snapshot paths are repository-relative, while the recorder stores absolute
    # native paths. Match exactly by the unique scenes-relative suffix.
    matches = [entry for path, entry in assets.items() if scene_path.as_posix().endswith("/" + path)]
    need(len(matches) == 1, "selected scene is ambiguous in the provenance snapshot")
    expected = {"binary_sha256": snapshot["binary_sha256"]["preview"], "scene_sha256": matches[0]["sha256"]}
    need(all(inputs.get(key) == value for key, value in expected.items()), "capture input hashes disagree with provenance")
    need(inputs.get("hashes_after") == inputs.get("hashes_after_cleanup") == expected,
         "capture input hashes changed during recording/cleanup")
    need(inputs.get("unchanged_during_recording") is True and inputs.get("unchanged_after_cleanup") is True,
         "capture input immutability checks did not pass")


def confirm_reference(prefix, samples, snapshot):
    metadata = load_document(prefix.with_suffix(".json"))
    need(metadata.get("completed_samples") == metadata.get("target_samples") == samples,
         "reference sample count does not match actual scheduled save")
    need(metadata.get("resolution") == [960, 640] and metadata.get("seed") == 1,
         "reference resolution/seed differs from required settings")
    need(all(type(metadata["config"].get(key)) is type(value) and metadata["config"][key] == value
             for key, value in EXPECTED_CONFIG.items()), "reference renderer configuration mismatch")
    need(metadata.get("sampler_version") == metadata.get("sampler_id") == 3 and metadata.get("sampler") == "halton",
         "reference Halton identity mismatch")
    need(metadata.get("resumed_from") is None and metadata["run_policy"].get("stop_reason") == "sample_target",
         "reference must be an independent fresh sample-target render")
    need(Path(metadata["scene"]).resolve() == Path(snapshot["scene"]).resolve(), "reference scene differs from capture")
    need(f"_source-{EXPECTED_SOURCE_SHA256}_" in metadata["compatibility"]["build_identifier"],
         "reference metadata build does not identify the frozen renderer")
    png, sums = prefix.with_suffix(".png"), prefix.with_suffix(".sum.bin")
    need(png.is_file() and png.stat().st_size > 0 and sums.is_file(), "reference PNG or raw sum is missing")
    with sums.open("rb") as stream:
        header = stream.read(32)
    need(len(header) == 32 and struct.unpack("<8sIIIIQ", header) == (b"C565SUM\0", 1, 960, 640, 3, samples),
         "reference raw-sum header mismatch")
    need(sums.stat().st_size == 32 + 960 * 640 * 12 + 8, "reference raw-sum length mismatch")
    return metadata


def run_logged(command, root, out, err):
    record = {"argv": list(command), "cwd": str(root), "shell": False,
              "started": now(), "stdout_log": out.name, "stderr_log": err.name}
    with out.open("x", encoding="utf-8") as stdout, err.open("x", encoding="utf-8") as stderr:
        process = subprocess.run(command, cwd=str(root), stdout=stdout, stderr=stderr,
                                 stdin=subprocess.DEVNULL, check=False, shell=False)
    record.update({"ended": now(), "exit_code": process.returncode})
    return record


def execute(args):
    root = Path(args.root).resolve(strict=True)
    result = inside_root(root, args.result_dir)
    scene = inside_root(root, args.scene)
    preview = inside_root(root, args.preview)
    headless = inside_root(root, args.headless)
    capture, prefix = result / "capture", result / "post-space-reference"
    recorder = root / "scripts/record_interaction_video.py"
    for binary in (preview, headless):
        need(binary.is_file() and os.access(binary, os.X_OK), f"missing executable: {binary}")
    need(recorder.is_file() and scene.is_file(), "recorder or scene file missing")
    need(bool(os.environ.get("DISPLAY")), "DISPLAY is unset; run only after the intended X11 display is ready")
    camera = load_document(scene)["Camera"]
    need(camera.get("RES") == [960, 640] and camera.get("DEPTH") == 10 and camera.get("ITERATIONS") == 2048,
         "scene must retain the native960x640 depth10 2048-iteration capture contract")
    reserved = [capture, result / "capture-run.json", result / "capture-provenance-before.json",
                result / "capture-provenance-after.json", result / "reference-provenance-before.json",
                result / "reference-provenance-after.json", result / "post-space-reference.command.json",
                result / "recorder-controller.stdout.log", result / "recorder-controller.stderr.log",
                result / "post-space-reference.stdout.log", result / "post-space-reference.stderr.log",
                prefix.with_suffix(".png"), prefix.with_suffix(".json"), prefix.with_suffix(".sum.bin")]
    need(not any(path.exists() for path in reserved), "capture/reference evidence already exists; refusing to overwrite or reuse it")
    result.mkdir(parents=True, exist_ok=True)
    run_path = result / "capture-run.json"
    run = {"schema_version": 1, "status": "running", "passed": False, "started": now(),
           "controller_source": record_file(Path(__file__), root), "root": str(root), "display": os.environ["DISPLAY"],
           "capture_summary": "capture/summary.json", "selected_scene": str(scene),
           "final_media_conversion_performed": False, "public_final_summary_written": False}
    write_document(run_path, run, fresh=True)
    try:
        before = take_snapshot(root, args.python, preview, headless, scene, "before-capture")
        write_document(result / "capture-provenance-before.json", before, fresh=True)
        need(before["renderer_source_sha256"] == EXPECTED_SOURCE_SHA256,
             "renderer inputs differ from the frozen source identity; recording will not start")
        run["capture_before"] = "capture-provenance-before.json"
        command = capture_command(args.python, root, preview, scene, capture)
        run["recorder_command"] = {"argv": command, "cwd": str(root), "shell": False}
        write_document(run_path, run)
        # Even a failed recorder run receives an after-capture provenance proof.
        try:
            recorder_run = run_logged(command, root, result / "recorder-controller.stdout.log",
                                      result / "recorder-controller.stderr.log")
            run["recorder_run"] = recorder_run
        finally:
            after = take_snapshot(root, args.python, preview, headless, scene, "after-capture")
            write_document(result / "capture-provenance-after.json", after, fresh=True)
            run["capture_after"] = "capture-provenance-after.json"
            write_document(run_path, run)
        identical_inputs(before, after)
        need(recorder_run["exit_code"] == 0, "existing recorder returned a failure; see preserved logs")
        saved = select_scheduled_save(capture)
        confirm_capture_inputs(capture, before)
        run["scheduled_S_save"] = saved
        reference_before = take_snapshot(root, args.python, preview, headless, scene, "before-reference")
        identical_inputs(after, reference_before)
        write_document(result / "reference-provenance-before.json", reference_before, fresh=True)
        command = reference_command(headless, scene, saved["samples"], prefix)
        receipt_path = result / "post-space-reference.command.json"
        receipt = {"argv": command, "cwd": str(root), "shell": False, "passed": False, "status": "running",
                   "renderer_source_sha256": reference_before["renderer_source_sha256"],
                   "headless_binary_sha256": reference_before["binary_sha256"]["headless"],
                   "scheduled_S_save": saved, "source_snapshot": "reference-provenance-before.json"}
        write_document(receipt_path, receipt, fresh=True)
        run["reference_command_receipt"] = receipt_path.name
        write_document(run_path, run)
        reference_run = run_logged(command, root, result / "post-space-reference.stdout.log",
                                   result / "post-space-reference.stderr.log")
        receipt.update(reference_run)
        write_document(receipt_path, receipt)
        need(reference_run["exit_code"] == 0, "independent headless reference failed; see preserved logs")
        metadata = confirm_reference(prefix, saved["samples"], reference_before)
        reference_after = take_snapshot(root, args.python, preview, headless, scene, "after-reference")
        identical_inputs(reference_before, reference_after)
        write_document(result / "reference-provenance-after.json", reference_after, fresh=True)
        receipt.update({"passed": True, "status": "succeeded", "build_identifier": metadata["compatibility"]["build_identifier"],
                        "outputs": {kind: record_file(path, root) for kind, path in {
                            "png": prefix.with_suffix(".png"), "metadata": prefix.with_suffix(".json"),
                            "linear_sum": prefix.with_suffix(".sum.bin")}.items()},
                        "source_snapshot_after": "reference-provenance-after.json"})
        write_document(receipt_path, receipt)
        run.update({"passed": True, "status": "succeeded", "completed": now(),
                    "capture_inputs_unchanged": True, "reference_inputs_unchanged": True})
        write_document(run_path, run)
        return run
    except Exception as error:
        run.update({"passed": False, "status": "failed", "completed": now(),
                    "failure": {"type": type(error).__name__, "message": str(error)}})
        write_document(run_path, run)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/ubuntu/cis5650/project3")
    parser.add_argument("--result-dir", default="results/hedge-final-interaction")
    parser.add_argument("--scene", default="scenes/hedge-motion.json")
    parser.add_argument("--preview", default="build-final/bin/cis565_path_tracer")
    parser.add_argument("--headless", default="build-final/bin/cis565_path_tracer_headless")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    try:
        run = execute(args)
        print(json.dumps({"passed": run["passed"], "status": run["status"],
                          "scheduled_S_save": run["scheduled_S_save"],
                          "result_dir": str(inside_root(Path(args.root).resolve(), args.result_dir))}, indent=2))
        return 0
    except (ControllerError, OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        print(f"final interaction controller failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
