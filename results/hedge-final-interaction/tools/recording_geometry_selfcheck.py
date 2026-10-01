#!/usr/bin/env python3
"""Original narrow regression checks for the recorder's X11 geometry handling.

Fixtures and subprocess responses are synthetic. This checker imports the
current recorder's definitions, executes its actual parser/placement/checkpoint
code under mocks, and compares timeline ASTs with the preserved old recorder.
It starts no renderer, capture, X11 command, or remote process.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest import mock


BASELINE_SHA256 = "2dd506d3560bae30bfd61ba212949eec485572e056f5e8231d35404c9d60f333"
SYNTHETIC_XWININFO = '''
xwininfo: Window id: 0x7 "CIS565 Path Tracer | 4 Iterations"
  Absolute upper-left X: 160
  Absolute upper-left Y: 69
  Relative upper-left X: 14
  Relative upper-left Y: 49
  Width: 960
  Height: 640
  Border width: 0
  Map State: IsViewable
'''


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path, root):
    resolved = path.resolve()
    return {"path": resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else str(resolved),
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rejected(function, error_type, *arguments):
    try:
        function(*arguments)
    except error_type as error:
        return str(error)
    raise AssertionError("invalid synthetic geometry was accepted")


def class_methods(text):
    tree = ast.parse(text)
    session = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RecordingSession")
    return {node.name: ast.dump(node, include_attributes=False)
            for node in session.body if isinstance(node, ast.FunctionDef)}


def run_checks(recorder, baseline):
    recorder_before, baseline_before = sha256(recorder), sha256(baseline)
    require(baseline_before == BASELINE_SHA256, "preserved pre-geometry baseline identity changed")
    current_text, baseline_text = recorder.read_text(encoding="utf-8"), baseline.read_text(encoding="utf-8")
    ast.parse(current_text)
    current = runpy.run_path(str(recorder), run_name="private_geometry_current_module")
    old = runpy.run_path(str(baseline), run_name="private_geometry_baseline_module")
    parse, error = current["parse_xwininfo_geometry"], current["RecordingError"]
    checks = []

    geometry = parse(SYNTHETIC_XWININFO)
    require(geometry == {"x": 160, "y": 69, "width": 960, "height": 640, "window": 7},
            "parser did not use the absolute client origin")
    checks.append({"name": "absolute_client_origin_ignores_parent_relative_fields", "passed": True,
                   "fixture_kind": "synthetic xwininfo text matching root-observed coordinate fields",
                   "absolute_origin": [160, 69], "parent_relative_origin": [14, 49], "parsed_geometry": geometry})

    invalids = {
        "missing_absolute_x": SYNTHETIC_XWININFO.replace("  Absolute upper-left X: 160\n", ""),
        "duplicate_absolute_x": SYNTHETIC_XWININFO + "  Absolute upper-left X: 160\n",
        "nonnumeric_width": SYNTHETIC_XWININFO.replace("  Width: 960", "  Width: bad"),
        "negative_height": SYNTHETIC_XWININFO.replace("  Height: 640", "  Height: -640"),
        "duplicate_width": SYNTHETIC_XWININFO + "  Width: 960\n",
        "unmapped_window": SYNTHETIC_XWININFO.replace("Map State: IsViewable", "Map State: IsUnMapped"),
        "duplicate_map_state": SYNTHETIC_XWININFO + "  Map State: IsViewable\n",
        "duplicate_window_id": SYNTHETIC_XWININFO + 'xwininfo: Window id: 0x7 "duplicate"\n',
        "missing_window_id": SYNTHETIC_XWININFO.replace(
            'xwininfo: Window id: 0x7 "CIS565 Path Tracer | 4 Iterations"\n', ""),
    }
    for label, text in invalids.items():
        reason = rejected(parse, error, text)
        checks.append({"name": label, "passed": True, "fixture_kind": "synthetic invalid text",
                       "rejection_message": reason})

    session = object.__new__(current["RecordingSession"])
    session.window_id = "7"
    session.tools = {"xwininfo": Path("/synthetic/mock/xwininfo")}
    session.environment, session.summary, session.actions = {}, {}, {}
    session.flush_evidence = lambda: None
    session._add_relative_time = lambda value: None
    session.focus_window = lambda purpose: None
    position = {"x": 130, "y": 168, "width": 960, "height": 640}
    state = {"response": "normal"}
    movements = []
    inspection_argv = []

    def fake_xdo(arguments, purpose, **kwargs):
        if arguments == ["getdisplaygeometry"]:
            return {"stdout": "1920 1080\n"}
        require(arguments[0] == "windowmove" and "--sync" not in arguments,
                "placement issued an unexpected action or used blocking windowmove --sync")
        require(len(arguments) == 4 and arguments[1] == "7", "placement selected a different window")
        requested = [int(arguments[-2]), int(arguments[-1])]
        movements.append({"requested": requested, "absolute_after_mock_GNOME": [requested[0], max(69, requested[1] + 37)]})
        position.update({"x": requested[0], "y": max(69, requested[1] + 37)})
        return {"stdout": ""}

    session.xdo = fake_xdo

    def fake_xwininfo(argv, **kwargs):
        require(argv == ["/synthetic/mock/xwininfo", "-id", "7", "-stats"],
                "unexpected inspection command; real subprocess execution is prohibited")
        require(kwargs.get("shell") is False, "geometry inspection must use shell=False")
        inspection_argv.append(list(argv))
        if state["response"] == "timeout":
            raise current["subprocess"].TimeoutExpired(argv, 3.0, output=b"synthetic timeout", stderr=b"synthetic stderr")
        if state["response"] == "failure":
            return SimpleNamespace(returncode=1, stdout="", stderr="synthetic tool failure")
        text = SYNTHETIC_XWININFO.replace("Absolute upper-left X: 160", "Absolute upper-left X: " + str(position["x"]))
        text = text.replace("Absolute upper-left Y: 69", "Absolute upper-left Y: " + str(position["y"]))
        text = text.replace("Width: 960", "Width: " + str(position["width"]))
        if state["response"] == "different-window":
            text = text.replace("Window id: 0x7", "Window id: 0x8")
        return SimpleNamespace(returncode=0, stdout=text, stderr="")

    with mock.patch.object(current["subprocess"], "run", fake_xwininfo), \
            mock.patch.object(current["time"], "sleep", lambda seconds: None):
        session.configure_geometry()
        require(movements == [{"requested": [160, 69], "absolute_after_mock_GNOME": [160, 106]},
                              {"requested": [160, 32], "absolute_after_mock_GNOME": [160, 69]}],
                "GNOME frame-offset correction did not settle at the required absolute client origin")
        result = session.summary["capture_geometry"]
        require(result["verified_client"] == {"x": 160, "y": 69, "width": 960, "height": 640, "window": 7},
                "placement verification did not retain native dimensions and absolute origin")
        require(result["client_margins_in_capture"] == {"left": 160, "top": 69, "right": 160, "bottom": 11},
                "native client margins do not fit the preserved capture region")
        require(result["capture_region"] == {"x": 0, "y": 0, "width": 1280, "height": 720, "within_display_bounds": True},
                "capture rectangle changed")
        require(result["native_client_verified"] is True and result["window_was_resized_or_maximized"] is False,
                "placement resized or failed native geometry verification")
        checks.append({"name": "mock_GNOME_placement_correction", "passed": True,
                       "fixture_kind": "synthetic responses from mocked xdotool/xwininfo; no X11 command ran",
                       "movement_sequence": movements, "verified_client": result["verified_client"],
                       "capture_region": result["capture_region"], "client_bottom": 709,
                       "client_margins": result["client_margins_in_capture"]})
        session.verify_geometry_checkpoint("synthetic valid checkpoint")
        checks.append({"name": "required_position_checkpoint_acceptance", "passed": True, "fixture_kind": "synthetic"})
        position["y"] = 70
        reason = rejected(session.verify_geometry_checkpoint, error, "synthetic one-pixel displacement")
        checks.append({"name": "one_pixel_displacement_rejection", "passed": True,
                       "fixture_kind": "synthetic", "rejection_message": reason})
        position.update({"y": 69, "width": 959})
        reason = rejected(session.verify_geometry_checkpoint, error, "synthetic nonnative width")
        checks.append({"name": "nonnative_dimensions_rejection", "passed": True,
                       "fixture_kind": "synthetic", "rejection_message": reason})
        position["width"] = 960
        for response in ("different-window", "failure", "timeout"):
            state["response"] = response
            reason = rejected(session.get_geometry, error, "synthetic " + response)
            checks.append({"name": "inspection_" + response + "_rejection", "passed": True,
                           "fixture_kind": "synthetic", "rejection_message": reason})
        require(session.actions["x11_geometry_commands"][-1]["timed_out"] is True,
                "timed-out geometry inspection was not preserved as failed evidence")

    unchanged_methods = ["begin_event", "finish_event", "xdo", "require_native_geometry", "launch_capture",
                         "await_capture_frame", "establish_interaction_epoch", "wait_until", "convergence",
                         "reset_camera", "drag", "scheduled_save", "send_escape", "stop_capture",
                         "ffprobe_video", "verify_geometry_checkpoint", "run"]
    current_ast, old_ast = class_methods(current_text), class_methods(baseline_text)
    for name in unchanged_methods:
        require(current_ast[name] == old_ast[name], "original timeline/validation method changed: " + name)
    require(current["schedule_description"](48) == old["schedule_description"](48), "interaction schedule changed")
    require(current["renderer_options"]("halton", "on") == old["renderer_options"]("halton", "on"),
            "renderer options changed")
    preserved = ["CAPTURE_WIDTH", "CAPTURE_HEIGHT", "CLIENT_WIDTH", "CLIENT_HEIGHT", "CLIENT_X",
                 "MIN_DURATION_SECONDS", "MAX_FPS", "MAX_EVENT_DRIFT_SECONDS"]
    for name in preserved:
        require(current[name] == old[name], "original capture or timing limit changed: " + name)
    require(current["CLIENT_Y"] == 69 and old["CLIENT_Y"] == 48, "client Y target change is not the intended 48 to 69")
    checks.append({"name": "timeline_capture_and_validation_AST_preservation", "passed": True,
                   "method": "Python AST structural equality excluding source location",
                   "unchanged_methods": unchanged_methods, "preserved_constants": preserved,
                   "old_client_y": 48, "new_client_y": 69,
                   "interaction_schedule_seconds": current["schedule_description"](48)})
    require(sha256(recorder) == recorder_before and sha256(baseline) == baseline_before,
            "checker altered its recorder inputs")
    checks.append({"name": "source_inputs_unchanged_by_checks", "passed": True,
                   "recorder_sha256_before_after": recorder_before, "baseline_sha256_before_after": baseline_before})
    return checks, inspection_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path.cwd()))
    parser.add_argument("--recorder", default="scripts/record_interaction_video.py")
    parser.add_argument("--baseline", default=".local/record_interaction_video.pre-geometry.py")
    parser.add_argument("--output", default=".local/recording_geometry_selfcheck.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    def resolve(name):
        value = Path(name)
        return (value if value.is_absolute() else root / value).resolve()
    recorder, baseline, output = resolve(args.recorder), resolve(args.baseline), resolve(args.output)
    report = {"schema_version": 1, "passed": False, "status": "running",
              "scope": "synthetic geometry regression only",
              "renderer_executed": False, "capture_executed": False, "real_X11_command_executed": False,
              "remote_action_performed": False, "footage_or_interaction_success_claimed": False,
              "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"), "checks": []}
    exit_code = 1
    try:
        report["inputs"] = {"current_recorder": artifact(recorder, root), "preserved_baseline": artifact(baseline, root),
                            "selfcheck_source": artifact(Path(__file__), root)}
        checks, commands = run_checks(recorder, baseline)
        report.update({"passed": True, "status": "passed", "checks": checks, "check_count": len(checks),
                       "mocked_inspection_commands": commands})
        exit_code = 0
    except Exception as error:
        report.update({"status": "failed", "failure": {"type": type(error).__name__, "message": str(error)}})
    report["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    print(json.dumps({"passed": report["passed"], "status": report["status"], "scope": report["scope"],
                      "check_count": report.get("check_count", 0), "report": str(output)}, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
