#!/usr/bin/env python3
"""Retain pure local tooling checks; never invoke a renderer or compiler."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
EXPECTED_SOURCE = "a16c1aba9a7cf17d4d2ffd902e5f78e0967acd06942d399ff3af256029227e86"
TOOLS = (
    "validate_hedge.py", "measure_hedge.py", "measure_gltf.py",
    "plot_hedge.py", "run_hedge_suite.py", "generate_gltf_assets.py",
    "validate_render.py", "source_hash.py",
)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def file_record(path):
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "mtime_ns": path.stat().st_mtime_ns}


def snapshot(paths):
    return {p.relative_to(ROOT).as_posix(): file_record(p) for p in sorted(paths)}


def source_identity():
    paths = [ROOT / "CMakeLists.txt"]
    for directory in ("src", "stream_compaction", "external", "cmake"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file())
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.relative_to(ROOT).as_posix()):
        name = path.relative_to(ROOT).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "little"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()


def save(report):
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    if (OUTPUT / "summary.json").exists() or (OUTPUT / "logs").exists():
        raise SystemExit("Refusing to overwrite an earlier local tooling run")
    (OUTPUT / "logs").mkdir()
    asset_paths = list((ROOT / "scenes" / "assets").rglob("*"))
    asset_paths.extend((ROOT / "scenes").glob("*.json"))
    asset_paths = [p for p in asset_paths if p.is_file()]
    report = {
        "schema_version": 1,
        "kind": "local-tooling-self-tests-and-read-only-asset-verification",
        "synthetic_self_test_outputs_are_not_gpu_evidence": True,
        "renderer_executed": False, "compiler_executed": False,
        "gpu_measurement": False, "started_utc": utc_now(),
        "source_input_sha256_before": source_identity(),
        "script_files_before": snapshot([ROOT / "scripts" / n for n in TOOLS]),
        "scene_and_asset_files_before": snapshot(asset_paths),
        "runner_sha256": file_record(Path(__file__).resolve())["sha256"],
        "commands": [], "status": "running",
    }
    save(report)
    if report["source_input_sha256_before"] != EXPECTED_SOURCE:
        report.update(status="failed", error="Renderer input identity differs from the assigned frozen source")
        save(report)
        return 1
    commands = [
        ("validate-hedge", ["scripts/validate_hedge.py", "--self-test"], True),
        ("measure-hedge", ["scripts/measure_hedge.py", "--self-test", "--output",
                           "results/hedge-final-tools/measure-hedge"], True),
        ("measure-gltf", ["scripts/measure_gltf.py", "--self-test", "--output",
                          "results/hedge-final-tools/measure-gltf"], True),
        ("plot-hedge", ["scripts/plot_hedge.py", "--self-test", "--output",
                        "results/hedge-final-tools/plot-hedge"], True),
        ("run-hedge-suite", ["scripts/run_hedge_suite.py", "--self-test"], False),
        ("generate-gltf-self-test", ["scripts/generate_gltf_assets.py", "--self-test"], False),
        ("generate-gltf-verify", ["scripts/generate_gltf_assets.py", "--verify"], False),
    ]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    failed = False
    for label, arguments, json_output in commands:
        argv = [sys.executable, "-B", *arguments]
        record = {"name": label, "argv": argv, "cwd": str(ROOT),
                  "started_utc": utc_now(), "self_test": label != "generate-gltf-verify",
                  "synthetic_output": label != "generate-gltf-verify"}
        started = time.perf_counter()
        process = subprocess.run(argv, cwd=ROOT, env=environment,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 check=False, timeout=300)
        record.update(returncode=process.returncode,
                      elapsed_wall_seconds=time.perf_counter() - started,
                      finished_utc=utc_now())
        for stream in ("stdout", "stderr"):
            path = OUTPUT / "logs" / f"{label}.{stream}.txt"
            path.write_bytes(getattr(process, stream))
            record[f"{stream}_log"] = path.relative_to(OUTPUT).as_posix()
            record[f"{stream}_sha256"] = file_record(path)["sha256"]
            record[f"{stream}_bytes"] = path.stat().st_size
        if process.returncode == 0 and json_output:
            structured = json.loads(process.stdout)
            record["reported_check_count"] = len(structured["checks"])
            if isinstance(structured["checks"], list) and structured["checks"] and isinstance(structured["checks"][0], dict):
                if any(item.get("passed") is False for item in structured["checks"]):
                    raise RuntimeError(f"{label} reports a failing check with exit code zero")
            destination = OUTPUT / label / "self-test-result.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(structured, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            record["structured_result"] = destination.relative_to(OUTPUT).as_posix()
            if "generated_svg_count" in structured:
                record["synthetic_generated_svg_count"] = structured["generated_svg_count"]
        record["status"] = "passed" if process.returncode == 0 else "failed"
        report["commands"].append(record)
        save(report)
        print(f"{label}: {record['status']} ({record['elapsed_wall_seconds']:.3f}s)", flush=True)
        if process.returncode:
            failed = True
            break
    report.update(
        finished_utc=utc_now(),
        source_input_sha256_after=source_identity(),
        script_files_after=snapshot([ROOT / "scripts" / n for n in TOOLS]),
        scene_and_asset_files_after=snapshot(asset_paths),
    )
    report["renderer_inputs_unchanged"] = report["source_input_sha256_before"] == report["source_input_sha256_after"] == EXPECTED_SOURCE
    report["script_bytes_and_mtimes_unchanged"] = report["script_files_before"] == report["script_files_after"]
    report["scene_and_asset_bytes_and_mtimes_unchanged"] = report["scene_and_asset_files_before"] == report["scene_and_asset_files_after"]
    failed = failed or not all(report[key] for key in (
        "renderer_inputs_unchanged", "script_bytes_and_mtimes_unchanged",
        "scene_and_asset_bytes_and_mtimes_unchanged"))
    report["status"] = "failed" if failed else "passed"
    save(report)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
