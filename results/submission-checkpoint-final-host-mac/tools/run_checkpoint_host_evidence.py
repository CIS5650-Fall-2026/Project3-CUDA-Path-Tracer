#!/usr/bin/env python3
"""Fresh, isolated host builds and test receipts for the checkpoint correction.

This helper is independently authored in Project 3. It calls only the local
compiler and repository tests; it neither fetches code nor touches cloud state.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/submission-checkpoint-final-host-mac"
CORE = ["checkpoint", "gltf_loader", "geometry", "image", "scene", "stb",
        "utilities", "output_path"]
TESTS = ["geometry_tests", "gltf_tests", "checkpoint_tests", "output_path_tests"]
SAN_ENV = {"ASAN_OPTIONS": "detect_leaks=0:halt_on_error=1",
           "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1"}
SAN_FLAGS = ["-O1", "-g", "-fsanitize=address,undefined,float-cast-overflow",
             "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"]
INSTRUCTION_SHA = "bdc2f3475f456e2ea21d08e52f71ce87678a13653cd265ea0b0c44f72a8f1a8f"


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest_bytes(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def file_record(path, relative_to=None):
    record = digest_bytes(path.read_bytes())
    record["path"] = str(path.relative_to(relative_to)) if relative_to else str(path)
    return record


def source_paths():
    paths = [ROOT / "CMakeLists.txt"]
    for directory in ("src", "stream_compaction", "external", "cmake"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file())
    return sorted(paths, key=lambda p: p.relative_to(ROOT).as_posix())


def source_hash():
    value = hashlib.sha256()
    for path in source_paths():
        name = path.relative_to(ROOT).as_posix().encode()
        data = path.read_bytes()
        value.update(len(name).to_bytes(8, "little"))
        value.update(name)
        value.update(len(data).to_bytes(8, "little"))
        value.update(data)
    return value.hexdigest()


def input_snapshot():
    paths = set(source_paths())
    paths.update(p for p in (ROOT / "tests").rglob("*") if p.is_file())
    paths.update(p for p in (ROOT / "scenes").rglob("*") if p.is_file())
    paths.add(ROOT / "scripts/source_hash.py")
    paths.add(ROOT / "INSTRUCTION.md")
    return [file_record(p, ROOT) for p in sorted(paths)]


def command(argv, label, overrides=None):
    started = utc()
    ticks = time.monotonic()
    env = os.environ.copy()
    env.update(overrides or {})
    proc = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True)
    elapsed = time.monotonic() - ticks
    ended = utc()
    stdout_path = OUT / (label + ".stdout")
    stderr_path = OUT / (label + ".stderr")
    stdout_path.write_bytes(proc.stdout)
    stderr_path.write_bytes(proc.stderr)
    return {
        "label": label, "command": argv, "cwd": str(ROOT),
        "environment": {"inherit_parent_environment": True,
                        "disclosed_overrides_only": overrides or {},
                        "note": "Unrelated inherited environment values are not recorded."},
        "started_utc": started, "ended_utc": ended,
        "duration_seconds": elapsed, "returncode": proc.returncode,
        "stdout": digest_bytes(proc.stdout), "stderr": digest_bytes(proc.stderr),
        "stdout_file": stdout_path.name, "stderr_file": stderr_path.name,
    }


def result_text(receipt):
    return (OUT / receipt["stdout_file"]).read_text()


def require_success(receipt):
    if receipt["returncode"] != 0:
        raise RuntimeError(receipt["label"] + " failed; retained raw diagnostics")


def parse_geometry(receipt):
    require_success(receipt)
    match = re.search(r"Executed (\d+) test groups, (\d+) repository scene\(s\), "
                      r"and (\d+) assertions; (\d+) group\(s\) failed\.", result_text(receipt))
    if not match:
        raise RuntimeError("Missing actual geometry result line")
    groups, scenes, assertions, failures = map(int, match.groups())
    if failures or groups == 0 or assertions == 0:
        raise RuntimeError("Geometry result was not a useful passing suite")
    return {"groups": groups, "repository_scenes": scenes,
            "assertions": assertions, "failed_groups": failures, "passed": True}


def parse_gltf(receipt):
    require_success(receipt)
    match = re.search(r"Executed (\d+) glTF test groups and (\d+) assertions; "
                      r"(\d+) group\(s\) failed\.", result_text(receipt))
    if not match:
        raise RuntimeError("Missing actual glTF result line")
    groups, assertions, failures = map(int, match.groups())
    if failures or not groups or not assertions:
        raise RuntimeError("glTF result was not a useful passing suite")
    return {"groups": groups, "assertions": assertions,
            "failed_groups": failures, "passed": True}


def parse_checkpoint(receipt):
    require_success(receipt)
    match = re.search(r"checkpoint_tests: PASS \((\d+) groups, (\d+) checks\)",
                      result_text(receipt))
    if not match:
        raise RuntimeError("Missing actual checkpoint result line with counts")
    groups, checks = map(int, match.groups())
    if not groups or not checks:
        raise RuntimeError("Checkpoint result was not a useful passing suite")
    return {"groups": groups, "checks": checks, "passed": True}


def parse_output_path(receipt):
    require_success(receipt)
    match = re.search(r"output_path_tests: PASS \((\d+) groups, (\d+) checks((?:,[^)\n]*)?)\)",
                      result_text(receipt))
    if not match:
        raise RuntimeError("Missing actual output-path result line")
    groups, checks = map(int, match.groups()[:2])
    if not groups or not checks:
        raise RuntimeError("Output-path result was not a useful passing suite")
    calibration = {}
    for item in match.group(3).strip(", ").split(","):
        if not item.strip():
            continue
        key, separator, value = item.strip().partition("=")
        if not separator:
            raise RuntimeError("Malformed actual filesystem calibration field")
        calibration[key] = value
    return {"groups": groups, "checks": checks, "passed": True,
            "filesystem_calibration": calibration,
            "actual_result_line": match.group(0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_source_sha256):
        raise SystemExit("Expected source SHA must be a full SHA-256")
    if source_hash() != args.expected_source_sha256:
        raise SystemExit("Source hash differs from the authorized ready source")
    if OUT.exists():
        raise SystemExit("Refusing to replace an existing evidence directory")
    OUT.mkdir(parents=True)
    (OUT / "tools").mkdir()
    helper_before = file_record(Path(__file__))
    shutil.copyfile(__file__, OUT / "tools/run_checkpoint_host_evidence.py")
    started = utc()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    build_root = ROOT / ".local" / ("submission-checkpoint-final-host-build-" + stamp)
    build_root.mkdir()
    before = input_snapshot()
    (OUT / "source.sha256").write_text(args.expected_source_sha256 + "\n")
    receipts = []
    try:
        system_commands = [
            (["/usr/bin/clang++", "--version"], "system-compiler"),
            (["/usr/bin/sw_vers"], "system-os"),
            (["/usr/bin/uname", "-m"], "system-architecture"),
            (["/usr/sbin/sysctl", "hw.model", "hw.memsize", "machdep.cpu.brand_string"],
             "system-hardware"),
            (["git", "rev-parse", "HEAD"], "system-base-revision"),
            (["python3", "-B", "scripts/source_hash.py"], "system-source-hash"),
        ]
        system = {}
        for argv, label in system_commands:
            record = command(argv, label)
            receipts.append(record)
            require_success(record)
            system[label] = {"receipt": record, "output": result_text(record)}
        if system["system-source-hash"]["output"].strip() != args.expected_source_sha256:
            raise RuntimeError("Repository source hash script disagrees with helper")
        write_json(OUT / "system.json", system)

        compile_jobs = []
        modes = {"optimized": ["-O3", "-DNDEBUG"], "sanitized": SAN_FLAGS}
        includes = ["-I", "src", "-I", ".", "-isystem", "external/include",
                    "-isystem", "external/tinygltf"]
        objects = {}
        for mode, flags in modes.items():
            directory = build_root / mode
            directory.mkdir()
            objects[mode] = {}
            for source, name in [("src/" + n + ".cpp", n) for n in CORE] + [
                ("tests/" + n + ".cpp", n) for n in TESTS]:
                object_path = directory / (name + ".o")
                objects[mode][name] = object_path
                argv = ["/usr/bin/clang++", "-std=c++17", *flags, *includes,
                        "-c", source, "-o", str(object_path)]
                compile_jobs.append((argv, "build-" + mode + "-" + name))
        with ThreadPoolExecutor(max_workers=4) as pool:
            built = list(pool.map(lambda job: command(*job), compile_jobs))
        receipts.extend(built)
        for record in built:
            require_success(record)
        binaries = {}
        for mode, flags in modes.items():
            binaries[mode] = {}
            for name in TESTS:
                binary = build_root / mode / name
                argv = ["/usr/bin/clang++", *flags, str(objects[mode][name]),
                        *[str(objects[mode][n]) for n in CORE], "-o", str(binary)]
                record = command(argv, "link-" + mode + "-" + name)
                receipts.append(record)
                require_success(record)
                binaries[mode][name] = binary
        binaries_before = [dict(file_record(path), mode=mode, name=name,
                                executable=os.access(path, os.X_OK))
                           for mode, items in binaries.items() for name, path in items.items()]
        build_records = [r for r in receipts if r["label"].startswith(("build-", "link-"))]
        write_json(OUT / "build-provenance.json", {
            "renderer_source_sha256": args.expected_source_sha256,
            "fresh_build": True,
            "build_started_after_source_ready": True,
            "build_directory": str(build_root),
            "core_translation_units": ["src/" + n + ".cpp" for n in CORE],
            "test_translation_units": ["tests/" + n + ".cpp" for n in TESTS],
            "compile_and_link_receipts": build_records,
            "binaries_before_fresh_tests": binaries_before,
            "no_executables_delivered": True,
        })
        scenes = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "scenes").glob("*.json"))
        parsers = {"geometry_tests": parse_geometry, "gltf_tests": parse_gltf,
                   "checkpoint_tests": parse_checkpoint, "output_path_tests": parse_output_path}
        names = {"geometry_tests": "geometry", "gltf_tests": "gltf",
                 "checkpoint_tests": "checkpoint", "output_path_tests": "output-path"}
        invocations = {}
        for mode in modes:
            for name in TESTS:
                args_tail = scenes if mode == "optimized" and name == "geometry_tests" else []
                label = ("geometry-41-scenes" if mode == "optimized" and name == "geometry_tests"
                         else names[name] + ("-sanitized" if mode == "sanitized" else ""))
                argv = [str(binaries[mode][name]), *args_tail]
                record = command(argv, label, SAN_ENV if mode == "sanitized" else None)
                receipts.append(record)
                record["parsed_result"] = parsers[name](record)
                invocations[names[name].replace("-", "_") + "_" + mode] = record

        after = input_snapshot()
        if before != after or source_hash() != args.expected_source_sha256:
            raise RuntimeError("Protected source, test, or scene inputs changed during validation")
        if helper_before != file_record(Path(__file__)):
            raise RuntimeError("Executed helper changed during validation")
        binaries_after = [dict(file_record(path), mode=mode, name=name,
                               executable=os.access(path, os.X_OK))
                          for mode, items in binaries.items() for name, path in items.items()]
        if binaries_before != binaries_after:
            raise RuntimeError("Test executable changed during validation")
        write_json(OUT / "verification-input-hashes.json", {
            "recorded_utc": utc(), "renderer_source_sha256": args.expected_source_sha256,
            "source_hash_coverage_note": "CMakeLists.txt and all files under src, stream_compaction, external and cmake; tests, all scenes/assets, INSTRUCTION.md, and the source-hash helper are also hashed individually here.",
            "test_sources": [r["path"] for r in before if r["path"].startswith("tests/")],
            "helper_inputs": ["scripts/source_hash.py", str(Path(__file__).relative_to(ROOT))],
            "runtime_repository_inputs": [r["path"] for r in before if r["path"].startswith("scenes/")],
            "before": before, "after": after, "unchanged": True,
            "helper_before": helper_before, "helper_after": file_record(Path(__file__)),
        })
        instruction = file_record(ROOT / "INSTRUCTION.md", ROOT)
        original = Path("/Users/didrikwiigandersen/Downloads/INSTRUCTION (3).md")
        original_record = file_record(original)
        if instruction["sha256"] != INSTRUCTION_SHA or instruction["sha256"] != original_record["sha256"]:
            raise RuntimeError("Original instructions restoration check failed")
        write_json(OUT / "instruction-restoration.json", {
            "repository": instruction, "original_download": original_record,
            "exact_original_bytes": True,
        })
        previous = json.loads((ROOT / "results/submission-final-host-mac/verification-input-hashes.json").read_text())
        previous_scene_inputs = [r for r in previous["before"] if r["path"].startswith("scenes/")]
        current_map = {r["path"]: r for r in before}
        comparisons = []
        for record in previous_scene_inputs:
            current = current_map[record["path"]]
            unchanged = record["bytes"] == current["bytes"] and record["sha256"] == current["sha256"]
            comparisons.append({"previous": record, "current": current, "unchanged": unchanged})
        if not comparisons or not all(r["unchanged"] for r in comparisons):
            raise RuntimeError("Previously validated repository scenes/assets changed")
        write_json(OUT / "existing-scenes-assets-comparison.json", {
            "baseline": "results/submission-final-host-mac/verification-input-hashes.json",
            "baseline_renderer_source_sha256": "07233055f7de64aac672f3b5f173066827eac89b27c3fc12f4b7f35982847703",
            "current_renderer_source_sha256": args.expected_source_sha256,
            "all_previous_scene_and_asset_inputs_unchanged": True,
            "count": len(comparisons), "comparisons": comparisons,
        })
        ended = utc()
        execution = {
            "recorded_utc": ended, "started_utc": started, "ended_utc": ended,
            "renderer_source_sha256": args.expected_source_sha256,
            "helper_invocation": {"interpreter_executable": sys.executable,
                "script_argument_vector": list(sys.argv),
                "python_bytecode_writing_disabled": bool(sys.dont_write_bytecode)},
            "scope": "CPU-only host correctness; GPU correctness and performance are outside this evidence directory",
            "repository_scene_argument_policy": "Every sorted top-level scenes/*.json is passed explicitly to optimized geometry; sanitized geometry runs the complete useful host groups with no repository scene arguments.",
            "repository_scene_arguments": scenes,
            "sanitizers": {"build_instrumentation": "address,undefined,float-cast-overflow",
                "sanitizer_recovery_enabled": False, "frame_pointers_omitted": False,
                "environment_overrides": SAN_ENV, "leak_checking_enabled": False,
                "leak_checking_disclosure": "Apple ASan does not support leak detection; detect_leaks=0 disables only leak detection while ASan, UBSan, float-cast-overflow, no recovery and halt-on-error remain enabled."},
            "fresh_test_invocations": invocations,
            "binaries_before": binaries_before, "binaries_after": binaries_after,
            "all_protected_inputs_unchanged": True,
            "no_cloud_actions": True, "no_downloaded_code": True,
        }
        write_json(OUT / "execution.json", execution)
        write_json(OUT / "process-log.json", {"started_utc": started, "ended_utc": ended,
            "processes": receipts, "all_returncodes_zero": all(r["returncode"] == 0 for r in receipts)})
        command_lines = []
        for record in receipts:
            command_lines.extend([record["label"], "```text", shlex.join(record["command"]), "```",
                "Start: " + record["started_utc"], "End: " + record["ended_utc"],
                "Exit: " + str(record["returncode"]),
                "Stdout: " + record["stdout_file"], "Stderr: " + record["stderr_file"], ""])
        (OUT / "executed-commands.md").write_text("\n".join(command_lines) + "\n")
        write_json(OUT / "provenance.json", {
            "renderer_source_sha256": args.expected_source_sha256,
            "helper_authorship": "Independently authored in Project 3; no Internet or other student implementation was used.",
            "helper_snapshot": file_record(OUT / "tools/run_checkpoint_host_evidence.py", OUT),
            "actual_repository_core_and_tests_compiled": True,
            "approved_dependencies": "Existing unchanged course starter dependencies and the previously approved pinned tinygltf header/license only.",
            "no_new_third_party_source": True, "no_gpu_claim": True,
        })
        geo_opt = invocations["geometry_optimized"]["parsed_result"]
        geo_san = invocations["geometry_sanitized"]["parsed_result"]
        gltf_opt = invocations["gltf_optimized"]["parsed_result"]
        gltf_san = invocations["gltf_sanitized"]["parsed_result"]
        checkpoint_opt = invocations["checkpoint_optimized"]["parsed_result"]
        checkpoint_san = invocations["checkpoint_sanitized"]["parsed_result"]
        path_opt = invocations["output_path_optimized"]["parsed_result"]
        path_san = invocations["output_path_sanitized"]["parsed_result"]
        if geo_opt["repository_scenes"] != len(scenes) or geo_san["repository_scenes"] != 0:
            raise RuntimeError("Actual geometry scene counts differ from the explicit arguments")
        if gltf_opt != gltf_san or checkpoint_opt != checkpoint_san or path_opt != path_san:
            raise RuntimeError("Optimized and sanitized test counts disagree")
        summary = {
            "renderer_source_sha256": args.expected_source_sha256,
            "scope": "host correctness only; fresh builds and executions; GPU validation recorded separately",
            "status": "passed", "geometry_optimized": geo_opt, "geometry_asan_ubsan": geo_san,
            "gltf_optimized_and_asan_ubsan": {"groups": gltf_opt["groups"],
                "assertions_each": gltf_opt["assertions"], "passed": True},
            "checkpoint_optimized_and_asan_ubsan": {"groups": checkpoint_opt["groups"],
                "checks_each": checkpoint_opt["checks"], "passed": True},
            "output_path_optimized_and_asan_ubsan": {"groups": path_opt["groups"],
                "checks_each": path_opt["checks"], "passed": True,
                "filesystem_calibration": path_opt["filesystem_calibration"],
                "actual_result_line_each": path_opt["actual_result_line"]},
            "fresh_test_invocations": len(invocations),
            "fresh_compile_and_link_invocations": len(build_records),
            "empty_stderr": all(r["stderr"]["bytes"] == 0 for r in invocations.values()),
            "all_build_stderr_empty": all(r["stderr"]["bytes"] == 0 for r in build_records),
            "artifacts": [file_record(p, OUT) for p in sorted(OUT.rglob("*")) if p.is_file()],
        }
        write_json(OUT / "test-summary.json", summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "artifacts"}))
    except Exception as error:
        write_json(OUT / "failure.json", {"recorded_utc": utc(), "error": str(error),
            "renderer_source_sha256": args.expected_source_sha256, "receipts": receipts})
        raise


if __name__ == "__main__":
    main()
