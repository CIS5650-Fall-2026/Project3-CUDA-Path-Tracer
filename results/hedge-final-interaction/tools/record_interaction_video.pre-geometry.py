#!/usr/bin/env python3
"""Record a timed, genuine X11 interaction with the CUDA preview."""

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
import shutil
import signal
import struct
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import zlib


CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CLIENT_WIDTH = 960
CLIENT_HEIGHT = 640
CLIENT_X = 160
CLIENT_Y = 48
MIN_DURATION_SECONDS = 48.0
MAX_FPS = 240
MAX_EVENT_DRIFT_SECONDS = 0.75
TITLE_SEARCH_PATTERN = r"^CIS ?565 Path Tracer( [|] [0-9]+ Iterations)?$"
LIVE_TITLE_PATTERN = re.compile(r"^CIS565 Path Tracer [|] ([0-9]+) Iterations$")
FIXED_RENDERER_OPTIONS = [
    "--sort", "off",
    "--wavefront", "on",
    "--rr", "on",
    "--rr-start", "3",
]


class RecordingError(RuntimeError):
    """A runtime condition made the recording invalid."""


def duration_argument(text: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a finite number") from error
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be finite")
    if value < MIN_DURATION_SECONDS:
        raise argparse.ArgumentTypeError(
            f"must be at least {MIN_DURATION_SECONDS:g} seconds"
        )
    return value


def fps_argument(text: str) -> int:
    if re.fullmatch(r"[0-9]+", text) is None:
        raise argparse.ArgumentTypeError("must be an integer")
    value = int(text)
    if not 1 <= value <= MAX_FPS:
        raise argparse.ArgumentTypeError(f"must be in [1, {MAX_FPS}]")
    return value


def renderer_options(sampler: str, motion: str) -> List[str]:
    """Build the preview's supported rendering options as discrete argv values."""
    return [
        *FIXED_RENDERER_OPTIONS,
        "--sampler", sampler,
        "--motion", motion,
    ]


def capture_readiness_timeout(fps: int) -> float:
    """Allow enough real input frames for an encoder with frame buffering."""
    return max(20.0, min(90.0, 12.0 + 32.0 / fps))


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def timestamp() -> Dict[str, Any]:
    return {"utc": utc_now(), "monotonic_ns": time.monotonic_ns()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_text(path, text)


def normalize_x11_display(display: str) -> str:
    """Add an X11 screen suffix only when DISPLAY has no screen suffix."""
    match = re.fullmatch(r"(.*:[0-9]+)(?:[.]([0-9]+))?", display)
    if match is None:
        raise RecordingError(f"unsupported DISPLAY syntax: {display!r}")
    if match.group(2) is not None:
        return display
    return display + ".0"


def parse_window_id(text: str) -> int:
    value = text.strip()
    if value.lower().startswith("0x"):
        return int(value, 16)
    return int(value, 10)


def parse_geometry_shell(text: str) -> Dict[str, int]:
    values: Dict[str, int] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        if key in {"WINDOW", "X", "Y", "WIDTH", "HEIGHT", "SCREEN"}:
            try:
                values[key.lower()] = int(raw_value.strip(), 0)
            except ValueError as error:
                raise RecordingError(
                    f"invalid xdotool geometry value {line!r}"
                ) from error
    required = {"x", "y", "width", "height"}
    if not required.issubset(values):
        raise RecordingError(
            "xdotool getwindowgeometry --shell omitted required fields: "
            + text.strip()
        )
    return values


def inspect_png(path: Path) -> Dict[str, Any]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RecordingError(f"{path.name} does not have a PNG signature")
    offset = 8
    width: Optional[int] = None
    height: Optional[int] = None
    chunk_names: List[str] = []
    saw_iend = False
    while offset < len(data):
        if offset + 12 > len(data):
            raise RecordingError(f"{path.name} ends inside a PNG chunk")
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        chunk_type = data[offset + 4:offset + 8]
        payload_start = offset + 8
        payload_end = payload_start + length
        crc_end = payload_end + 4
        if crc_end > len(data):
            raise RecordingError(f"{path.name} has a truncated PNG chunk")
        expected_crc = struct.unpack(">I", data[payload_end:crc_end])[0]
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(data[payload_start:payload_end], actual_crc)
        actual_crc &= 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise RecordingError(f"{path.name} has a PNG CRC mismatch")
        try:
            chunk_name = chunk_type.decode("ascii")
        except UnicodeDecodeError as error:
            raise RecordingError(f"{path.name} has a non-ASCII PNG chunk") from error
        chunk_names.append(chunk_name)
        if chunk_type == b"IHDR":
            if length != 13 or width is not None:
                raise RecordingError(f"{path.name} has an invalid IHDR chunk")
            width, height = struct.unpack(">II", data[payload_start:payload_start + 8])
        if chunk_type == b"IEND":
            if length != 0 or crc_end != len(data):
                raise RecordingError(f"{path.name} has an invalid PNG ending")
            saw_iend = True
            offset = crc_end
            break
        offset = crc_end
    if not saw_iend or offset != len(data):
        raise RecordingError(f"{path.name} is missing a complete IEND chunk")
    if (width, height) != (CLIENT_WIDTH, CLIENT_HEIGHT):
        raise RecordingError(
            f"{path.name} is {width}x{height}, expected "
            f"{CLIENT_WIDTH}x{CLIENT_HEIGHT}"
        )
    return {
        "name": path.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": width,
        "height": height,
        "png_complete": True,
        "chunks": chunk_names,
    }


def schedule_description(duration: float) -> List[Dict[str, Any]]:
    return [
        {"at_s": 0.0, "action": "release buttons 1/3/2; Space reset"},
        {"from_s": 0.0, "to_s": 12.0, "phase": "original-camera convergence"},
        {
            "from_s": 12.0,
            "to_s": 15.0,
            "action": "left drag (700,400) to (810,410), 12 steps",
        },
        {"from_s": 15.0, "to_s": 20.0, "phase": "left-drag convergence"},
        {
            "from_s": 20.0,
            "to_s": 22.0,
            "action": "right drag (700,400) to (700,350), 12 steps",
        },
        {"from_s": 22.0, "to_s": 27.0, "phase": "right-drag convergence"},
        {
            "from_s": 27.0,
            "to_s": 29.0,
            "action": "middle drag (700,400) to (740,425), 12 steps",
        },
        {"from_s": 29.0, "to_s": 34.0, "phase": "middle-drag convergence"},
        {"at_s": 34.0, "action": "release buttons 1/3/2; Space reset"},
        {"from_s": 34.0, "to_s": 44.0, "phase": "restored-camera convergence"},
        {"at_s": 44.0, "action": "S save"},
        {"at_s": 46.0, "action": "release buttons 1/3/2; Escape save and exit"},
        {"at_s": duration, "action": "SIGINT ffmpeg after renderer exit"},
    ]


class RecordingSession:
    def __init__(self, args: argparse.Namespace, output: Path) -> None:
        self.args = args
        self.output = output
        self.environment = os.environ.copy()
        self.renderer_options = renderer_options(args.sampler, args.motion)
        self.actions_path = output / "actions.json"
        self.summary_path = output / "summary.json"
        self.interaction_epoch_ns: Optional[int] = None
        self.window_id: Optional[str] = None
        self.tools: Dict[str, Path] = {}
        self.binary: Optional[Path] = None
        self.scene: Optional[Path] = None
        self.input_hashes_before: Dict[str, str] = {}
        self.renderer: Optional[subprocess.Popen[Any]] = None
        self.capture: Optional[subprocess.Popen[Any]] = None
        self.renderer_stdout: Optional[Any] = None
        self.renderer_stderr: Optional[Any] = None
        self.capture_stdout: Optional[Any] = None
        self.capture_stderr: Optional[Any] = None
        self.capture_stop_requested = False
        self.escape_sent = False
        self.cleaned_up = False

        self.summary: Dict[str, Any] = {
            "schema_version": 1,
            "status": "initializing",
            "passed": False,
            "created": timestamp(),
            "arguments": {
                "binary": str(args.binary),
                "scene": str(args.scene),
                "output": str(output),
                "duration_seconds": args.duration,
                "capture_fps": args.fps,
                "sampler": args.sampler,
                "motion": args.motion,
            },
            "subprocess_policy": {
                "argv_arrays_only": True,
                "shell": False,
                "renderer_cwd": str(output),
            },
            "timing": {
                "clock": "time.monotonic_ns",
                "maximum_allowed_event_start_drift_seconds": MAX_EVENT_DRIFT_SECONDS,
                "interaction_epoch": None,
            },
            "rate_semantics": {
                "capture_clock_fps": args.fps,
                "capture_clock_meaning": (
                    "ffmpeg x11grab screen-sampling rate; it is not renderer FPS"
                ),
                "window_title_iteration_meaning": (
                    "renderer completed cover-sample iteration progress; it is not "
                    "capture FPS and is not the ImGui application FPS"
                ),
                "imgui_application_fps": (
                    "display-loop telemetry only; this script does not read or claim it"
                ),
            },
            "desktop_capture_disclosure": {
                "crop_includes_surrounding_desktop_margins": True,
                "automated_private_information_detection_performed": False,
                "root_visual_verification_required": True,
                "statement": (
                    "xdotool cannot determine whether the surrounding desktop is clean "
                    "or contains private information; root must verify captured desktop "
                    "content visually before publication"
                ),
            },
            "renderer_provenance": {
                "fixed_options": list(FIXED_RENDERER_OPTIONS),
                "selected_sampler": args.sampler,
                "selected_motion": args.motion,
                "full_renderer_options": list(self.renderer_options),
                "options_forwarded_as_discrete_argv": True,
            },
            "cleanup": {"actions": [], "errors": []},
        }
        self.actions: Dict[str, Any] = {
            "schema_version": 1,
            "status": "initializing",
            "interaction_epoch": None,
            "schedule": schedule_description(args.duration),
            "xdotool_commands": [],
            "events": [],
            "title_samples": [],
            "convergence_phases": [],
            "png_checks": [],
        }
        self.flush_evidence()

    def _add_relative_time(self, value: Any) -> None:
        if isinstance(value, dict):
            if "monotonic_ns" in value:
                if self.interaction_epoch_ns is None:
                    value["relative_to_interaction_s"] = None
                else:
                    value["relative_to_interaction_s"] = (
                        value["monotonic_ns"] - self.interaction_epoch_ns
                    ) / 1_000_000_000.0
            for child in list(value.values()):
                self._add_relative_time(child)
        elif isinstance(value, list):
            for child in value:
                self._add_relative_time(child)

    def flush_evidence(self) -> None:
        self._add_relative_time(self.summary)
        self._add_relative_time(self.actions)
        atomic_write_json(self.actions_path, self.actions)
        atomic_write_json(self.summary_path, self.summary)

    def begin_event(
        self,
        label: str,
        scheduled_s: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "label": label,
            "status": "running",
            "scheduled_relative_s": scheduled_s,
            "start": timestamp(),
        }
        if details:
            record["details"] = details
        self._add_relative_time(record)
        if scheduled_s is not None and self.interaction_epoch_ns is not None:
            actual = record["start"]["relative_to_interaction_s"]
            record["start_drift_s"] = actual - scheduled_s
        self.actions["events"].append(record)
        self.flush_evidence()
        return record

    def finish_event(
        self,
        record: Dict[str, Any],
        status: str = "completed",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        record["end"] = timestamp()
        record["status"] = status
        if details:
            record.setdefault("result", {}).update(details)
        self.flush_evidence()

    def xdo(
        self,
        arguments: Sequence[Any],
        purpose: str,
        *,
        check: bool = True,
        timeout: float = 5.0,
        scheduled_s: Optional[float] = None,
        enforce_drift: bool = False,
    ) -> Dict[str, Any]:
        if "xdotool" not in self.tools:
            raise RecordingError("xdotool was not resolved")
        argv = [str(self.tools["xdotool"]), *[str(value) for value in arguments]]
        start = timestamp()
        timed_out = False
        returncode: Optional[int]
        stdout = ""
        stderr = ""
        try:
            result = subprocess.run(
                argv,
                env=self.environment,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            returncode = result.returncode
            stdout = result.stdout
            stderr = result.stderr
        except subprocess.TimeoutExpired as error:
            timed_out = True
            returncode = None
            if isinstance(error.stdout, bytes):
                stdout = error.stdout.decode("utf-8", errors="replace")
            else:
                stdout = error.stdout or ""
            if isinstance(error.stderr, bytes):
                stderr = error.stderr.decode("utf-8", errors="replace")
            else:
                stderr = error.stderr or ""
        end = timestamp()
        record: Dict[str, Any] = {
            "purpose": purpose,
            "argv": argv,
            "start": start,
            "end": end,
            "exit_code": returncode,
            "timed_out": timed_out,
            "stdout": stdout,
            "stderr": stderr,
            "scheduled_relative_s": scheduled_s,
        }
        self._add_relative_time(record)
        if scheduled_s is not None:
            if self.interaction_epoch_ns is None:
                raise RecordingError("scheduled xdotool command ran before interaction epoch")
            drift = record["start"]["relative_to_interaction_s"] - scheduled_s
            record["start_drift_s"] = drift
        self.actions["xdotool_commands"].append(record)
        self.flush_evidence()
        if timed_out:
            raise RecordingError(f"xdotool timed out during {purpose}: {argv!r}")
        if check and returncode != 0:
            raise RecordingError(
                f"xdotool failed during {purpose} with exit code {returncode}: "
                f"{stderr.strip()}"
            )
        if enforce_drift and scheduled_s is not None:
            drift = record["start_drift_s"]
            if abs(drift) > MAX_EVENT_DRIFT_SECONDS:
                raise RecordingError(
                    f"{purpose} timing drift {drift:+.3f}s exceeded "
                    f"{MAX_EVENT_DRIFT_SECONDS:.3f}s"
                )
        return record

    def _resolve_tool(self, name: str) -> Path:
        discovered = shutil.which(name, path=self.environment.get("PATH"))
        if discovered is None:
            raise RecordingError(f"required tool is not on PATH: {name}")
        path = Path(discovered).resolve(strict=True)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RecordingError(f"required tool is not executable: {path}")
        return path

    def _version_command(self, name: str) -> None:
        argv = [str(self.tools[name]), "-version"]
        start = timestamp()
        try:
            result = subprocess.run(
                argv,
                env=self.environment,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            end = timestamp()
            timeout_stdout = error.stdout or ""
            timeout_stderr = error.stderr or ""
            if isinstance(timeout_stdout, bytes):
                timeout_stdout = timeout_stdout.decode("utf-8", errors="replace")
            if isinstance(timeout_stderr, bytes):
                timeout_stderr = timeout_stderr.decode("utf-8", errors="replace")
            self.summary.setdefault("tool_versions", {})[name] = {
                "argv": argv,
                "start": start,
                "end": end,
                "exit_code": None,
                "timed_out": True,
                "stdout": timeout_stdout,
                "stderr": timeout_stderr,
            }
            self.flush_evidence()
            raise RecordingError(f"{name} -version timed out") from error
        end = timestamp()
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        version = lines[0] if lines else ""
        record = {
            "argv": argv,
            "start": start,
            "end": end,
            "exit_code": result.returncode,
            "timed_out": False,
            "version_string": version,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        self.summary.setdefault("tool_versions", {})[name] = record
        self.flush_evidence()
        if result.returncode != 0 or not version:
            raise RecordingError(f"{name} -version failed with {result.returncode}")

    def preflight(self) -> None:
        display = self.environment.get("DISPLAY", "")
        if not display or display.strip() != display:
            raise RecordingError("DISPLAY must be inherited as a nonempty value")
        normalized_display = normalize_x11_display(display)
        self.summary["display"] = {
            "inherited_DISPLAY": display,
            "inherited_XAUTHORITY_path": self.environment.get("XAUTHORITY"),
            "ffmpeg_x11grab_display": normalized_display,
            "environment_inherited_without_overrides": True,
            "xauthority_contents_read": False,
        }

        for name in ("xdotool", "ffmpeg", "ffprobe"):
            self.tools[name] = self._resolve_tool(name)
        self.summary["tools"] = {
            name: str(path) for name, path in self.tools.items()
        }

        try:
            binary = Path(self.args.binary).expanduser().resolve(strict=True)
        except OSError as error:
            raise RecordingError(f"preview binary does not exist: {self.args.binary}") from error
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RecordingError(f"preview binary is not an executable file: {binary}")
        try:
            scene = Path(self.args.scene).expanduser().resolve(strict=True)
        except OSError as error:
            raise RecordingError(f"scene does not exist: {self.args.scene}") from error
        if not scene.is_file():
            raise RecordingError(f"scene is not a regular file: {scene}")
        self.binary = binary
        self.scene = scene
        self.input_hashes_before = {
            "binary_sha256": sha256_file(binary),
            "scene_sha256": sha256_file(scene),
        }
        self.summary["inputs"] = {
            "binary": str(binary),
            "scene": str(scene),
            **self.input_hashes_before,
            "renderer_options": list(self.renderer_options),
        }

        try:
            scene_document = json.loads(scene.read_text(encoding="utf-8"))
            camera = scene_document["Camera"]
            resolution = camera["RES"]
            iterations = camera["ITERATIONS"]
            depth = camera["DEPTH"]
            image_name = camera["FILE"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise RecordingError(f"could not read the required scene camera contract: {error}") from error
        scene_contract = {
            "resolution": resolution,
            "iterations": iterations,
            "depth": depth,
            "image_name": image_name,
        }
        self.summary["inputs"]["scene_camera"] = scene_contract
        if resolution != [CLIENT_WIDTH, CLIENT_HEIGHT]:
            raise RecordingError(
                f"scene Camera.RES must be [{CLIENT_WIDTH}, {CLIENT_HEIGHT}]"
            )
        if type(iterations) is not int or iterations != 2048:
            raise RecordingError("scene Camera.ITERATIONS must be exactly 2048")
        if type(depth) is not int or depth != 10:
            raise RecordingError("scene Camera.DEPTH must be exactly 10")
        if (
            not isinstance(image_name, str)
            or not image_name
            or Path(image_name).name != image_name
            or image_name in {".", ".."}
        ):
            raise RecordingError(
                "scene Camera.FILE must be a nonempty basename so saves remain in output"
            )

        self._version_command("ffmpeg")
        self._version_command("ffprobe")
        self.summary["status"] = "preflight-passed"
        self.actions["status"] = "preflight-passed"
        self.flush_evidence()

    def _record_process_exit(self, name: str, returncode: int) -> None:
        process_record = self.summary[name]
        if process_record.get("exit") is None:
            process_record["exit"] = timestamp()
            process_record["exit_code"] = returncode
            self.flush_evidence()

    def monitor_processes(self, *, allow_renderer_exit: bool = False) -> None:
        if self.capture is not None:
            capture_code = self.capture.poll()
            if capture_code is not None:
                self._record_process_exit("capture", capture_code)
                if not self.capture_stop_requested:
                    raise RecordingError(
                        f"ffmpeg exited early with status {capture_code}"
                    )
        if self.renderer is not None:
            renderer_code = self.renderer.poll()
            if renderer_code is not None:
                self._record_process_exit("renderer", renderer_code)
                if not allow_renderer_exit:
                    raise RecordingError(
                        f"renderer exited before Escape with status {renderer_code}"
                    )

    def launch_renderer(self) -> None:
        assert self.binary is not None and self.scene is not None
        command = [str(self.binary), str(self.scene), *self.renderer_options]
        stdout_path = self.output / "renderer.stdout.log"
        stderr_path = self.output / "renderer.stderr.log"
        self.renderer_stdout = stdout_path.open("w", encoding="utf-8")
        self.renderer_stderr = stderr_path.open("w", encoding="utf-8")
        before = timestamp()
        self.summary["renderer"] = {
            "command": command,
            "cwd": str(self.output),
            "stdout_log": stdout_path.name,
            "stderr_log": stderr_path.name,
            "launch_start": before,
            "exit": None,
            "exit_code": None,
        }
        self.flush_evidence()
        self.renderer = subprocess.Popen(
            command,
            cwd=str(self.output),
            env=self.environment,
            stdin=subprocess.DEVNULL,
            stdout=self.renderer_stdout,
            stderr=self.renderer_stderr,
        )
        self.summary["renderer"]["pid"] = self.renderer.pid
        self.summary["renderer"]["launch_complete"] = timestamp()
        self.flush_evidence()

    def find_window(self) -> None:
        assert self.renderer is not None
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            self.monitor_processes()
            result = self.xdo(
                [
                    "search",
                    "--all",
                    "--onlyvisible",
                    "--pid",
                    str(self.renderer.pid),
                    "--name",
                    TITLE_SEARCH_PATTERN,
                ],
                "find visible preview window by child PID and title",
                check=False,
                timeout=3.0,
            )
            candidates = sorted(
                {line.strip() for line in result["stdout"].splitlines() if line.strip()}
            )
            if len(candidates) > 1:
                raise RecordingError(
                    f"more than one visible preview window matched: {candidates}"
                )
            if len(candidates) == 1:
                try:
                    parse_window_id(candidates[0])
                except ValueError as error:
                    raise RecordingError(
                        f"xdotool returned an invalid window id: {candidates[0]!r}"
                    ) from error
                self.window_id = candidates[0]
                self.summary["window"] = {
                    "id": self.window_id,
                    "search_pattern": TITLE_SEARCH_PATTERN,
                    "search_constrained_to_renderer_pid": self.renderer.pid,
                    "search_requires_all_predicates": True,
                }
                self.flush_evidence()
                return
            time.sleep(0.10)
        raise RecordingError("preview window did not become visible within 30 seconds")

    def focus_window(self, purpose: str) -> None:
        if self.window_id is None:
            raise RecordingError("preview window has not been selected")
        self.xdo(
            ["windowactivate", "--sync", self.window_id],
            purpose + ": activate",
        )
        self.xdo(["windowraise", self.window_id], purpose + ": raise")
        focus = self.xdo(["getwindowfocus"], purpose + ": verify focus")
        try:
            focused_id = parse_window_id(focus["stdout"])
            expected_id = parse_window_id(self.window_id)
        except ValueError as error:
            raise RecordingError("xdotool returned an invalid focused window id") from error
        if focused_id != expected_id:
            raise RecordingError(
                f"focused window {focused_id} is not preview window {expected_id}"
            )

    def get_geometry(self, label: str) -> Dict[str, int]:
        assert self.window_id is not None
        result = self.xdo(
            ["getwindowgeometry", "--shell", self.window_id],
            "read preview client geometry: " + label,
        )
        geometry = parse_geometry_shell(result["stdout"])
        observation = {"label": label, "observed": timestamp(), **geometry}
        self.summary.setdefault("capture_geometry", {}).setdefault(
            "window_geometry_history", []
        ).append(observation)
        self.flush_evidence()
        return geometry

    def require_native_geometry(self, geometry: Dict[str, int], label: str) -> None:
        if (geometry["width"], geometry["height"]) != (
            CLIENT_WIDTH,
            CLIENT_HEIGHT,
        ):
            raise RecordingError(
                f"{label}: preview client is {geometry['width']}x{geometry['height']}, "
                f"expected native {CLIENT_WIDTH}x{CLIENT_HEIGHT}; refusing to resize"
            )

    def configure_geometry(self) -> None:
        assert self.window_id is not None
        screen_result = self.xdo(
            ["getdisplaygeometry"], "read inherited X display geometry"
        )
        fields = screen_result["stdout"].split()
        if len(fields) != 2:
            raise RecordingError(
                "xdotool getdisplaygeometry returned unexpected output: "
                + screen_result["stdout"].strip()
            )
        try:
            screen_width, screen_height = (int(value) for value in fields)
        except ValueError as error:
            raise RecordingError("display geometry was not numeric") from error
        if screen_width < CAPTURE_WIDTH or screen_height < CAPTURE_HEIGHT:
            raise RecordingError(
                f"display is {screen_width}x{screen_height}; capture requires at least "
                f"{CAPTURE_WIDTH}x{CAPTURE_HEIGHT}"
            )
        self.summary["capture_geometry"] = {
            "display": {"width": screen_width, "height": screen_height},
            "capture_region": {
                "x": 0,
                "y": 0,
                "width": CAPTURE_WIDTH,
                "height": CAPTURE_HEIGHT,
                "within_display_bounds": True,
            },
            "requested_client": {
                "x": CLIENT_X,
                "y": CLIENT_Y,
                "width": CLIENT_WIDTH,
                "height": CLIENT_HEIGHT,
            },
            "window_was_resized_or_maximized": False,
            "titlebar": {
                "inclusion_intended": True,
                "method": (
                    "native client placed with 48 pixels of top headroom; xdotool client "
                    "geometry does not prove decoration extents"
                ),
                "visual_verification_required": True,
            },
            "window_geometry_history": [],
        }
        initial = self.get_geometry("before move")
        self.require_native_geometry(initial, "before move")

        requested_x = CLIENT_X
        requested_y = CLIENT_Y
        final_geometry: Optional[Dict[str, int]] = initial
        if initial["x"] != CLIENT_X or initial["y"] != CLIENT_Y:
            for attempt in range(1, 6):
                self.xdo(
                    [
                        "windowmove",
                        "--sync",
                        self.window_id,
                        str(requested_x),
                        str(requested_y),
                    ],
                    f"move preview without resizing (correction attempt {attempt})",
                )
                observed = self.get_geometry(f"after move attempt {attempt}")
                self.require_native_geometry(observed, f"after move attempt {attempt}")
                final_geometry = observed
                if observed["x"] == CLIENT_X and observed["y"] == CLIENT_Y:
                    break
                requested_x += CLIENT_X - observed["x"]
                requested_y += CLIENT_Y - observed["y"]
        assert final_geometry is not None
        if final_geometry["x"] != CLIENT_X or final_geometry["y"] != CLIENT_Y:
            raise RecordingError(
                "window manager prevented exact client placement at "
                f"({CLIENT_X},{CLIENT_Y}); observed "
                f"({final_geometry['x']},{final_geometry['y']})"
            )
        if not (
            0 <= final_geometry["x"]
            and 0 <= final_geometry["y"]
            and final_geometry["x"] + final_geometry["width"] <= CAPTURE_WIDTH
            and final_geometry["y"] + final_geometry["height"] <= CAPTURE_HEIGHT
        ):
            raise RecordingError("native preview client is outside the capture bounds")
        self.summary["capture_geometry"]["verified_client"] = final_geometry
        self.summary["capture_geometry"]["client_margins_in_capture"] = {
            "left": final_geometry["x"],
            "top": final_geometry["y"],
            "right": CAPTURE_WIDTH - final_geometry["x"] - final_geometry["width"],
            "bottom": CAPTURE_HEIGHT - final_geometry["y"] - final_geometry["height"],
        }
        self.summary["capture_geometry"]["native_client_verified"] = True
        self.summary["capture_geometry"]["client_within_capture_bounds"] = True
        self.focus_window("initial preview placement")
        self.flush_evidence()

    def sample_title(
        self,
        label: str,
        *,
        require_live: bool = True,
        allow_failure: bool = False,
    ) -> Optional[int]:
        assert self.window_id is not None
        result = self.xdo(
            ["getwindowname", self.window_id],
            "sample preview title: " + label,
            check=not allow_failure,
        )
        title = result["stdout"].rstrip("\r\n")
        match = LIVE_TITLE_PATTERN.fullmatch(title)
        iteration = int(match.group(1)) if match else None
        record = {
            "label": label,
            "observed": timestamp(),
            "title": title,
            "window_title_iteration": iteration,
            "xdotool_exit_code": result["exit_code"],
        }
        self.actions["title_samples"].append(record)
        self.flush_evidence()
        if require_live and iteration is None:
            raise RecordingError(f"preview title was not a live iteration title: {title!r}")
        return iteration

    def wait_for_live_title(self) -> None:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            self.monitor_processes()
            iteration = self.sample_title(
                "wait-for-live-title", require_live=False, allow_failure=False
            )
            if iteration is not None:
                return
            time.sleep(0.10)
        raise RecordingError("preview never published a live iteration title")

    def launch_capture(self) -> None:
        display = self.summary["display"]["ffmpeg_x11grab_display"]
        video_path = self.output / "interaction.mp4"
        progress_path = self.output / "capture.progress"
        if video_path.exists():
            raise RecordingError(f"capture output unexpectedly exists: {video_path}")
        command = [
            str(self.tools["ffmpeg"]),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "info",
            "-nostats",
            "-progress",
            str(progress_path),
            "-f",
            "x11grab",
            "-draw_mouse",
            "1",
            "-framerate",
            str(self.args.fps),
            "-video_size",
            f"{CAPTURE_WIDTH}x{CAPTURE_HEIGHT}",
            "-i",
            f"{display}+0,0",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ]
        stdout_path = self.output / "capture.stdout.log"
        stderr_path = self.output / "capture.stderr.log"
        self.capture_stdout = stdout_path.open("w", encoding="utf-8")
        self.capture_stderr = stderr_path.open("w", encoding="utf-8")
        before = timestamp()
        self.summary["capture"] = {
            "command": command,
            "cwd": str(self.output),
            "stdout_log": stdout_path.name,
            "stderr_log": stderr_path.name,
            "progress_file": progress_path.name,
            "video_file": video_path.name,
            "launch_start": before,
            "exit": None,
            "exit_code": None,
            "stop_requested": None,
        }
        self.flush_evidence()
        self.capture = subprocess.Popen(
            command,
            cwd=str(self.output),
            env=self.environment,
            stdin=subprocess.DEVNULL,
            stdout=self.capture_stdout,
            stderr=self.capture_stderr,
        )
        self.summary["capture"]["pid"] = self.capture.pid
        self.summary["capture"]["launch_complete"] = timestamp()
        self.flush_evidence()

    def await_capture_frame(self) -> None:
        progress_path = self.output / "capture.progress"
        timeout_seconds = capture_readiness_timeout(self.args.fps)
        deadline = time.monotonic() + timeout_seconds
        self.summary["capture"]["readiness_wait"] = {
            "timeout_seconds": timeout_seconds,
            "rate_aware_formula": "max(20, min(90, 12 + 32 / capture_fps))",
            "buffering_frame_budget": 32,
            "capture_fps": self.args.fps,
        }
        self.flush_evidence()
        last_progress = ""
        while time.monotonic() < deadline:
            self.monitor_processes()
            if progress_path.is_file():
                try:
                    last_progress = progress_path.read_text(
                        encoding="utf-8", errors="replace"
                    )
                except OSError:
                    last_progress = ""
                frames: List[int] = []
                for line in last_progress.splitlines():
                    if line.startswith("frame="):
                        try:
                            frames.append(int(line.split("=", 1)[1].strip()))
                        except ValueError:
                            pass
                if frames and max(frames) > 0:
                    proof = {
                        "observed": timestamp(),
                        "maximum_progress_frame": max(frames),
                        "criterion": "ffmpeg progress frame > 0",
                        "progress_file_retained": progress_path.name,
                        "progress_snapshot": last_progress,
                    }
                    self.summary["capture"]["real_frame_readiness_proof"] = proof
                    self.flush_evidence()
                    return
            time.sleep(0.05)
        self.summary["capture"]["last_progress_before_startup_timeout"] = last_progress
        self.flush_evidence()
        raise RecordingError(
            "ffmpeg did not report a captured frame within "
            f"{timeout_seconds:g} seconds at {self.args.fps} capture FPS"
        )

    def establish_interaction_epoch(self) -> None:
        epoch = timestamp()
        self.interaction_epoch_ns = epoch["monotonic_ns"]
        self._add_relative_time(epoch)
        self.summary["timing"]["interaction_epoch"] = epoch
        self.actions["interaction_epoch"] = dict(epoch)
        launch_ns = self.summary["capture"]["launch_start"]["monotonic_ns"]
        self.summary["timing"]["recorded_startup_lead_in_seconds"] = (
            self.interaction_epoch_ns - launch_ns
        ) / 1_000_000_000.0
        self.summary["timing"]["startup_lead_in_disclosure"] = (
            "The video begins before the interaction epoch; this genuine encoder/X11 "
            "startup lead-in is retained and included in probed duration."
        )
        self.summary["status"] = "recording"
        self.actions["status"] = "recording"
        self.flush_evidence()

    def relative_now(self) -> float:
        if self.interaction_epoch_ns is None:
            raise RecordingError("interaction epoch has not been established")
        return (time.monotonic_ns() - self.interaction_epoch_ns) / 1_000_000_000.0

    def wait_until(
        self,
        relative_s: float,
        *,
        title_label: Optional[str] = None,
        allow_renderer_exit: bool = False,
    ) -> None:
        if self.interaction_epoch_ns is None:
            raise RecordingError("cannot wait on schedule before interaction epoch")
        deadline_ns = self.interaction_epoch_ns + round(relative_s * 1_000_000_000)
        next_title_ns = time.monotonic_ns() if title_label else None
        title_index = 0
        while True:
            self.monitor_processes(allow_renderer_exit=allow_renderer_exit)
            now_ns = time.monotonic_ns()
            if now_ns >= deadline_ns:
                return
            if (
                next_title_ns is not None
                and now_ns >= next_title_ns
                and deadline_ns - now_ns > 150_000_000
            ):
                self.sample_title(f"{title_label}-{title_index:02d}")
                title_index += 1
                next_title_ns += 1_000_000_000
                continue
            wake_ns = deadline_ns
            if next_title_ns is not None:
                wake_ns = min(wake_ns, next_title_ns)
            sleep_seconds = max(0.001, min(0.05, (wake_ns - now_ns) / 1e9))
            time.sleep(sleep_seconds)

    def convergence(self, label: str, start_s: float, end_s: float) -> None:
        phase = {
            "label": label,
            "scheduled_start_s": start_s,
            "scheduled_end_s": end_s,
            "start": timestamp(),
            "title_sample_start_index": len(self.actions["title_samples"]),
            "status": "running",
        }
        self.actions["convergence_phases"].append(phase)
        self.flush_evidence()
        self.sample_title(label + "-start")
        self.wait_until(end_s, title_label=label + "-periodic")
        sample_start = phase["title_sample_start_index"]
        selected = self.actions["title_samples"][sample_start:]
        counts = [
            sample["window_title_iteration"]
            for sample in selected
            if sample["window_title_iteration"] is not None
        ]
        growth = any(later > earlier for earlier, later in zip(counts, counts[1:]))
        phase.update(
            {
                "end": timestamp(),
                "title_sample_end_index": len(self.actions["title_samples"]),
                "observed_iterations": counts,
                "parseable_sample_count": len(counts),
                "observed_sequential_growth": growth,
                "status": "passed" if len(counts) >= 2 and growth else "failed",
            }
        )
        self.flush_evidence()
        if len(counts) < 2 or not growth:
            raise RecordingError(
                f"{label} did not show realistic renderer iteration growth: {counts}"
            )

    def _assert_event_drift(self, event: Dict[str, Any]) -> None:
        drift = event.get("start_drift_s")
        if drift is not None and abs(drift) > MAX_EVENT_DRIFT_SECONDS:
            raise RecordingError(
                f"{event['label']} timing drift {drift:+.3f}s exceeded "
                f"{MAX_EVENT_DRIFT_SECONDS:.3f}s"
            )

    def reset_camera(
        self,
        label: str,
        scheduled_s: float,
        *,
        already_focused: bool = False,
        pre_sample: bool = True,
    ) -> None:
        self.wait_until(scheduled_s)
        if pre_sample:
            self.sample_title(label + "-before")
        if not already_focused:
            self.focus_window(label)
        event = self.begin_event(label, scheduled_s)
        self._assert_event_drift(event)
        for button in (1, 3, 2):
            self.xdo(
                ["mouseup", str(button)],
                f"{label}: release button {button}",
                scheduled_s=scheduled_s,
                enforce_drift=True,
            )
        self.xdo(
            ["key", "--window", self.window_id, "space"],
            f"{label}: Space",
            scheduled_s=scheduled_s,
            enforce_drift=True,
        )
        self.finish_event(event)
        self.sample_title(label + "-after")

    def drag(
        self,
        label: str,
        button: int,
        start_xy: Tuple[int, int],
        end_xy: Tuple[int, int],
        start_s: float,
        end_s: float,
    ) -> None:
        self.wait_until(start_s)
        self.sample_title(label + "-before")
        self.focus_window(label)
        self.xdo(
            [
                "mousemove",
                "--window",
                self.window_id,
                str(start_xy[0]),
                str(start_xy[1]),
            ],
            f"{label}: position pointer at client-relative start",
        )
        event = self.begin_event(
            label,
            start_s,
            {
                "button": button,
                "client_relative_start": list(start_xy),
                "client_relative_end": list(end_xy),
                "steps": 12,
                "scheduled_end_s": end_s,
            },
        )
        self._assert_event_drift(event)
        self.xdo(
            ["mousedown", str(button)],
            f"{label}: button down",
            scheduled_s=start_s,
            enforce_drift=True,
        )
        for step in range(1, 13):
            scheduled = start_s + (end_s - start_s) * step / 12.0
            self.wait_until(scheduled)
            x = round(start_xy[0] + (end_xy[0] - start_xy[0]) * step / 12.0)
            y = round(start_xy[1] + (end_xy[1] - start_xy[1]) * step / 12.0)
            self.xdo(
                [
                    "mousemove",
                    "--window",
                    self.window_id,
                    "--sync",
                    str(x),
                    str(y),
                ],
                f"{label}: incremental step {step}/12",
                scheduled_s=scheduled,
                enforce_drift=True,
            )
        self.xdo(
            ["mouseup", str(button)],
            f"{label}: button up",
            scheduled_s=end_s,
            enforce_drift=True,
        )
        self.finish_event(event)
        self.sample_title(label + "-after")

    def png_paths(self) -> Set[Path]:
        return {path for path in self.output.glob("*.png") if path.is_file()}

    def wait_for_one_png(
        self,
        before: Set[Path],
        label: str,
        deadline_relative_s: float,
        *,
        allow_renderer_exit: bool,
    ) -> Dict[str, Any]:
        previous_signature: Optional[Tuple[int, int]] = None
        stable_observations = 0
        last_error = "no new PNG observed"
        while self.relative_now() < deadline_relative_s:
            self.monitor_processes(allow_renderer_exit=allow_renderer_exit)
            added = self.png_paths() - before
            if len(added) > 1:
                raise RecordingError(
                    f"{label} created more than one PNG: "
                    + ", ".join(sorted(path.name for path in added))
                )
            if len(added) == 1:
                candidate = next(iter(added))
                try:
                    stat = candidate.stat()
                    signature = (stat.st_size, stat.st_mtime_ns)
                    if stat.st_size <= 0:
                        last_error = f"{candidate.name} is empty"
                    elif signature == previous_signature:
                        stable_observations += 1
                    else:
                        previous_signature = signature
                        stable_observations = 1
                    if stable_observations >= 2:
                        details = inspect_png(candidate)
                        details.update(
                            {
                                "created_by": label,
                                "verified": timestamp(),
                                "new_relative_to": sorted(path.name for path in before),
                                "stable_stat_observations": stable_observations,
                            }
                        )
                        self.actions["png_checks"].append(details)
                        self.flush_evidence()
                        return details
                except (OSError, RecordingError) as error:
                    last_error = str(error)
            time.sleep(0.10)
        raise RecordingError(
            f"{label} did not produce exactly one complete unique PNG before "
            f"t={deadline_relative_s:g}s: {last_error}"
        )

    def verify_geometry_checkpoint(self, label: str) -> None:
        geometry = self.get_geometry(label)
        self.require_native_geometry(geometry, label)
        if geometry["x"] != CLIENT_X or geometry["y"] != CLIENT_Y:
            raise RecordingError(
                f"{label}: preview moved from ({CLIENT_X},{CLIENT_Y}) to "
                f"({geometry['x']},{geometry['y']})"
            )
        if (
            geometry["x"] + geometry["width"] > CAPTURE_WIDTH
            or geometry["y"] + geometry["height"] > CAPTURE_HEIGHT
        ):
            raise RecordingError(f"{label}: preview left the capture bounds")

    def scheduled_save(self) -> Tuple[Set[Path], Optional[Dict[str, Any]], List[str]]:
        self.wait_until(44.0)
        self.sample_title("scheduled-save-before")
        self.verify_geometry_checkpoint("before scheduled S save")
        self.focus_window("scheduled S save")
        before = self.png_paths()
        errors: List[str] = []
        if before:
            errors.append(
                "PNG files existed before the first scheduled save: "
                + ", ".join(sorted(path.name for path in before))
            )
        event = self.begin_event("scheduled-S-save", 44.0)
        self._assert_event_drift(event)
        self.xdo(
            ["key", "--window", self.window_id, "s"],
            "scheduled S save",
            scheduled_s=44.0,
            enforce_drift=True,
        )
        self.finish_event(event)
        self.sample_title("scheduled-save-after")
        saved: Optional[Dict[str, Any]] = None
        try:
            saved = self.wait_for_one_png(
                before,
                "scheduled-S-save",
                45.70,
                allow_renderer_exit=False,
            )
        except RecordingError as error:
            errors.append(str(error))
        return self.png_paths(), saved, errors

    def send_escape(
        self, before: Set[Path]
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        self.wait_until(46.0, title_label="post-save-wait")
        self.sample_title("escape-before")
        self.verify_geometry_checkpoint("before Escape")
        self.focus_window("Escape save and exit")
        event = self.begin_event("release-and-Escape", 46.0)
        self._assert_event_drift(event)
        for button in (1, 3, 2):
            self.xdo(
                ["mouseup", str(button)],
                f"Escape: release button {button}",
                scheduled_s=46.0,
                enforce_drift=True,
            )
        self.escape_sent = True
        self.xdo(
            ["key", "--window", self.window_id, "Escape"],
            "Escape save and renderer exit",
            scheduled_s=46.0,
            enforce_drift=True,
        )
        self.finish_event(event)
        self.sample_title(
            "escape-after-best-effort", require_live=False, allow_failure=True
        )
        errors: List[str] = []
        saved: Optional[Dict[str, Any]] = None
        png_deadline = max(46.2, self.args.duration - 0.25)
        try:
            saved = self.wait_for_one_png(
                before,
                "Escape-save",
                png_deadline,
                allow_renderer_exit=True,
            )
        except RecordingError as error:
            errors.append(str(error))
        return saved, errors

    def stop_capture(self, scheduled_s: Optional[float]) -> List[str]:
        errors: List[str] = []
        if self.capture is None:
            return errors
        if self.capture.poll() is not None:
            self._record_process_exit("capture", self.capture.returncode)
            if not self.capture_stop_requested:
                errors.append(
                    f"ffmpeg had already exited with status {self.capture.returncode}"
                )
            return errors
        event = self.begin_event("ffmpeg-SIGINT", scheduled_s)
        if "start_drift_s" in event:
            event["orchestration_event_start_drift_s_not_used_for_validation"] = (
                event.pop("start_drift_s")
            )
        self.capture_stop_requested = True
        stop_stamp = timestamp()
        self.capture.send_signal(signal.SIGINT)
        signal_complete = timestamp()
        self._add_relative_time(stop_stamp)
        self._add_relative_time(signal_complete)
        delivery_drift: Optional[float] = None
        if scheduled_s is not None:
            delivery_drift = (
                stop_stamp["relative_to_interaction_s"] - scheduled_s
            )
        self.summary["capture"]["stop_requested"] = {
            "method": "SIGINT",
            "at": stop_stamp,
            "scheduled_relative_s": scheduled_s,
            "actual_relative_s": stop_stamp["relative_to_interaction_s"],
            "delivery_start_drift_s": delivery_drift,
            "drift_validation_basis": "timestamp immediately before send_signal(SIGINT)",
            "send_signal_returned": signal_complete,
        }
        event["signal_delivery"] = dict(self.summary["capture"]["stop_requested"])
        if (
            delivery_drift is not None
            and abs(delivery_drift) > MAX_EVENT_DRIFT_SECONDS
        ):
            errors.append(
                f"ffmpeg SIGINT delivery timing drift {delivery_drift:+.3f}s exceeded "
                f"{MAX_EVENT_DRIFT_SECONDS:.3f}s"
            )
        method = "SIGINT"
        graceful = True
        try:
            returncode = self.capture.wait(timeout=20)
        except subprocess.TimeoutExpired:
            graceful = False
            errors.append("ffmpeg did not exit within 20 seconds after SIGINT")
            method = "SIGINT-then-terminate"
            self.capture.terminate()
            try:
                returncode = self.capture.wait(timeout=5)
            except subprocess.TimeoutExpired:
                method = "SIGINT-then-terminate-then-kill"
                self.capture.kill()
                try:
                    returncode = self.capture.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    returncode = self.capture.poll()
                    errors.append("ffmpeg remained alive after kill timeout")
        if returncode is not None:
            self._record_process_exit("capture", returncode)
        self.summary["capture"]["termination"] = {
            "method": method,
            "graceful_after_SIGINT": graceful,
            "accepted_SIGINT_exit_codes": [0, 255],
            "exit_code": returncode,
        }
        if graceful and returncode not in (0, 255):
            errors.append(f"ffmpeg returned unexpected status {returncode} after SIGINT")
        self.finish_event(
            event,
            "completed" if not errors else "completed-with-errors",
            {"termination_method": method, "exit_code": returncode},
        )
        self.flush_evidence()
        return errors

    def ffprobe_video(self) -> None:
        video_path = self.output / "interaction.mp4"
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise RecordingError("interaction.mp4 is missing or empty")
        command = [
            str(self.tools["ffprobe"]),
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(video_path),
        ]
        start = timestamp()
        try:
            result = subprocess.run(
                command,
                cwd=str(self.output),
                env=self.environment,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            self.summary["ffprobe"] = {
                "command": command,
                "start": start,
                "end": timestamp(),
                "exit_code": None,
                "timed_out": True,
            }
            self.flush_evidence()
            raise RecordingError("ffprobe timed out") from error
        end = timestamp()
        atomic_write_text(self.output / "ffprobe.json", result.stdout)
        atomic_write_text(self.output / "ffprobe.stderr.log", result.stderr)
        self.summary["ffprobe"] = {
            "command": command,
            "start": start,
            "end": end,
            "exit_code": result.returncode,
            "timed_out": False,
            "complete_json_file": "ffprobe.json",
            "stderr_log": "ffprobe.stderr.log",
            "stderr": result.stderr,
        }
        self.flush_evidence()
        if result.returncode != 0:
            raise RecordingError(f"ffprobe exited with status {result.returncode}")
        try:
            document = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RecordingError("ffprobe did not emit valid JSON") from error
        video_streams = [
            stream
            for stream in document.get("streams", [])
            if stream.get("codec_type") == "video"
        ]
        if len(video_streams) != 1:
            raise RecordingError(
                f"ffprobe found {len(video_streams)} video streams, expected exactly one"
            )
        stream = video_streams[0]
        checks: Dict[str, Any] = {
            "video_bytes": video_path.stat().st_size,
            "codec_name": stream.get("codec_name"),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "pix_fmt": stream.get("pix_fmt"),
            "r_frame_rate": stream.get("r_frame_rate"),
            "avg_frame_rate": stream.get("avg_frame_rate"),
            "nb_read_frames": stream.get("nb_read_frames"),
            "stream_duration": stream.get("duration"),
            "format_duration": document.get("format", {}).get("duration"),
        }
        if stream.get("codec_name") != "h264":
            raise RecordingError(f"video codec is {stream.get('codec_name')!r}, not h264")
        if (stream.get("width"), stream.get("height")) != (
            CAPTURE_WIDTH,
            CAPTURE_HEIGHT,
        ):
            raise RecordingError(
                f"video is {stream.get('width')}x{stream.get('height')}, expected "
                f"{CAPTURE_WIDTH}x{CAPTURE_HEIGHT}"
            )
        if stream.get("pix_fmt") != "yuv420p":
            raise RecordingError(
                f"video pixel format is {stream.get('pix_fmt')!r}, not yuv420p"
            )
        try:
            frame_count = int(stream["nb_read_frames"])
        except (KeyError, TypeError, ValueError) as error:
            raise RecordingError("ffprobe did not report a numeric nb_read_frames") from error
        if frame_count <= 0:
            raise RecordingError("ffprobe reported no decoded video frames")

        for rate_name in ("r_frame_rate", "avg_frame_rate"):
            try:
                rate_value = float(Fraction(stream[rate_name]))
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
                raise RecordingError(f"invalid ffprobe {rate_name}") from error
            checks[rate_name + "_decimal"] = rate_value
            if abs(rate_value - self.args.fps) > max(0.05, self.args.fps * 0.01):
                raise RecordingError(
                    f"{rate_name} {rate_value:g} differs from requested capture rate "
                    f"{self.args.fps}"
                )
        duration_text = document.get("format", {}).get("duration")
        try:
            media_duration = float(duration_text)
        except (TypeError, ValueError) as error:
            raise RecordingError("ffprobe format duration is not numeric") from error
        if not math.isfinite(media_duration) or media_duration <= 0:
            raise RecordingError("ffprobe format duration is not finite and positive")
        launch_ns = self.summary["capture"]["launch_start"]["monotonic_ns"]
        stop_ns = self.summary["capture"]["stop_requested"]["at"]["monotonic_ns"]
        real_interval = (stop_ns - launch_ns) / 1_000_000_000.0
        tolerance = 3.0
        duration_difference = media_duration - real_interval
        checks["decoded_frame_count"] = frame_count
        checks["media_duration_seconds"] = media_duration
        checks["real_capture_launch_to_SIGINT_seconds"] = real_interval
        checks["duration_difference_seconds"] = duration_difference
        checks["duration_tolerance_seconds"] = tolerance
        checks["duration_consistent_with_real_elapsed"] = (
            abs(duration_difference) <= tolerance
        )
        self.summary["ffprobe"]["validated"] = checks
        self.flush_evidence()
        if abs(duration_difference) > tolerance:
            raise RecordingError(
                f"probed duration {media_duration:.3f}s differs from real capture interval "
                f"{real_interval:.3f}s by {duration_difference:+.3f}s"
            )

    def verify_inputs_unchanged(self) -> None:
        assert self.binary is not None and self.scene is not None
        after = {
            "binary_sha256": sha256_file(self.binary),
            "scene_sha256": sha256_file(self.scene),
        }
        unchanged = after == self.input_hashes_before
        self.summary["inputs"]["hashes_after"] = after
        self.summary["inputs"]["unchanged_during_recording"] = unchanged
        self.flush_evidence()
        if not unchanged:
            raise RecordingError("preview binary or scene changed during recording")

    def run(self) -> None:
        self.preflight()
        self.launch_renderer()
        self.find_window()
        self.configure_geometry()
        self.wait_for_live_title()
        self.launch_capture()
        self.await_capture_frame()

        # Prepare the target while the capture is known to be live, then define t=0.
        self.focus_window("pre-epoch deterministic target")
        self.sample_title("initial-reset-before-epoch")
        self.establish_interaction_epoch()

        self.reset_camera(
            "initial-release-and-reset",
            0.0,
            already_focused=True,
            pre_sample=False,
        )
        self.convergence("original-camera-convergence", 0.0, 12.0)

        self.drag("left-orbit-drag", 1, (700, 400), (810, 410), 12.0, 15.0)
        self.convergence("left-drag-convergence", 15.0, 20.0)

        self.drag("right-zoom-drag", 3, (700, 400), (700, 350), 20.0, 22.0)
        self.convergence("right-drag-convergence", 22.0, 27.0)

        self.drag("middle-pan-drag", 2, (700, 400), (740, 425), 27.0, 29.0)
        self.convergence("middle-drag-convergence", 29.0, 34.0)

        self.reset_camera("restore-original-camera", 34.0)
        self.convergence("restored-camera-convergence", 34.0, 44.0)

        pngs_after_s, first_png, post_errors = self.scheduled_save()
        second_png, escape_errors = self.send_escape(pngs_after_s)
        post_errors.extend(escape_errors)

        # Keep genuine capture running to the requested absolute deadline even if a
        # post-Escape verification already found a problem.
        self.wait_until(self.args.duration, allow_renderer_exit=True)
        renderer_code = self.renderer.poll() if self.renderer is not None else None
        if renderer_code is None:
            post_errors.append(
                "renderer had not exited by the requested capture deadline after Escape"
            )
        else:
            self._record_process_exit("renderer", renderer_code)
            if renderer_code != 0:
                post_errors.append(
                    f"renderer exited with status {renderer_code}, expected 0"
                )

        post_errors.extend(self.stop_capture(self.args.duration))

        final_pngs = self.png_paths()
        if len(final_pngs) != 2:
            post_errors.append(
                f"final PNG count is {len(final_pngs)}, expected exactly 2: "
                + ", ".join(sorted(path.name for path in final_pngs))
            )
        if first_png is None:
            post_errors.append("the scheduled S save was not verified")
        if second_png is None:
            post_errors.append("the Escape save was not verified")
        if first_png is not None and second_png is not None:
            if first_png["name"] == second_png["name"]:
                post_errors.append("S and Escape saves reused the same PNG path")

        probe_error: Optional[str] = None
        try:
            self.ffprobe_video()
        except Exception as error:  # Preserve post-Escape failures alongside probe failure.
            probe_error = f"{type(error).__name__}: {error}"
        try:
            self.verify_inputs_unchanged()
        except Exception as error:
            post_errors.append(f"input immutability check failed: {error}")
        if probe_error is not None:
            post_errors.append(probe_error)
        if post_errors:
            raise RecordingError("; ".join(post_errors))

    def collect_png_inventory(self) -> None:
        inventory: List[Dict[str, Any]] = []
        for path in sorted(self.png_paths(), key=lambda item: item.name):
            item: Dict[str, Any] = {"name": path.name}
            try:
                item["bytes"] = path.stat().st_size
                item["sha256"] = sha256_file(path)
                item.update(inspect_png(path))
            except Exception as error:
                item["complete"] = False
                item["inspection_error"] = f"{type(error).__name__}: {error}"
            inventory.append(item)
        self.summary["png_files"] = inventory

    def cleanup(self) -> List[str]:
        if self.cleaned_up:
            return list(self.summary["cleanup"]["errors"])
        self.cleaned_up = True
        errors: List[str] = self.summary["cleanup"]["errors"]
        cleanup_actions: List[Dict[str, Any]] = self.summary["cleanup"]["actions"]

        if self.renderer is not None and self.renderer.poll() is not None:
            self._record_process_exit("renderer", self.renderer.returncode)
        if self.capture is not None and self.capture.poll() is not None:
            self._record_process_exit("capture", self.capture.returncode)

        if self.window_id is not None and "xdotool" in self.tools:
            for button in (1, 3, 2):
                try:
                    result = self.xdo(
                        ["mouseup", str(button)],
                        f"cleanup: release button {button}",
                        check=False,
                        timeout=3.0,
                    )
                    if result["exit_code"] != 0:
                        errors.append(
                            f"cleanup mouseup {button} exited {result['exit_code']}: "
                            f"{result['stderr'].strip()}"
                        )
                except Exception as error:
                    errors.append(f"cleanup mouseup {button} failed: {error}")

        if self.renderer is not None and self.renderer.poll() is None:
            action = {"process": "renderer", "method": "terminate", "start": timestamp()}
            try:
                self.renderer.terminate()
                try:
                    code = self.renderer.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    action["method"] = "terminate-then-kill"
                    self.renderer.kill()
                    code = self.renderer.wait(timeout=5)
                action["exit_code"] = code
                self._record_process_exit("renderer", code)
            except Exception as error:
                action["error"] = f"{type(error).__name__}: {error}"
                errors.append("renderer cleanup failed: " + action["error"])
            action["end"] = timestamp()
            cleanup_actions.append(action)

        if self.capture is not None and self.capture.poll() is None:
            action = {"process": "capture", "method": "SIGINT", "start": timestamp()}
            try:
                self.capture_stop_requested = True
                self.capture.send_signal(signal.SIGINT)
                try:
                    code = self.capture.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    action["method"] = "SIGINT-then-terminate"
                    self.capture.terminate()
                    try:
                        code = self.capture.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        action["method"] = "SIGINT-then-terminate-then-kill"
                        self.capture.kill()
                        code = self.capture.wait(timeout=5)
                action["exit_code"] = code
                self._record_process_exit("capture", code)
            except Exception as error:
                action["error"] = f"{type(error).__name__}: {error}"
                errors.append("capture cleanup failed: " + action["error"])
            action["end"] = timestamp()
            cleanup_actions.append(action)

        for name, stream in (
            ("renderer stdout", self.renderer_stdout),
            ("renderer stderr", self.renderer_stderr),
            ("capture stdout", self.capture_stdout),
            ("capture stderr", self.capture_stderr),
        ):
            if stream is not None:
                try:
                    stream.flush()
                    stream.close()
                except Exception as error:
                    errors.append(f"closing {name} failed: {error}")

        self.collect_png_inventory()
        try:
            if self.binary is not None and self.scene is not None:
                after = {
                    "binary_sha256": sha256_file(self.binary),
                    "scene_sha256": sha256_file(self.scene),
                }
                self.summary.setdefault("inputs", {})["hashes_after_cleanup"] = after
                if self.input_hashes_before:
                    unchanged = after == self.input_hashes_before
                    self.summary["inputs"]["unchanged_after_cleanup"] = unchanged
                    if not unchanged:
                        errors.append("binary or scene hash changed by cleanup time")
        except Exception as error:
            errors.append(f"final input hash check failed: {error}")
        self.flush_evidence()
        return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        required=True,
        type=Path,
        help="existing CUDA preview executable",
    )
    parser.add_argument(
        "--scene",
        required=True,
        type=Path,
        help="960x640, 2048-iteration, depth-10 scene JSON",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new evidence directory (must not already exist)",
    )
    parser.add_argument(
        "--duration",
        type=duration_argument,
        default=48.0,
        help="recording duration after interaction epoch, at least 48 (default: 48)",
    )
    parser.add_argument(
        "--fps",
        type=fps_argument,
        default=30,
        help="x11grab capture frame rate in [1, 240] (default: 30)",
    )
    parser.add_argument(
        "--sampler",
        choices=("counter", "halton"),
        default="counter",
        help="preview sampler (default: counter)",
    )
    parser.add_argument(
        "--motion",
        choices=("on", "off"),
        default="on",
        help="preview motion mode (default: on)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve(strict=False)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"error: output directory already exists: {output}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"error: could not create output directory {output}: {error}", file=sys.stderr)
        return 2

    session = RecordingSession(args, output)
    failure: Optional[BaseException] = None
    failure_traceback: Optional[str] = None
    try:
        session.run()
    except BaseException as error:  # Always retain evidence, including Ctrl-C.
        failure = error
        failure_traceback = traceback.format_exc()

    cleanup_errors: List[str] = []
    try:
        cleanup_errors = session.cleanup()
    except BaseException as error:
        cleanup_errors.append(f"cleanup raised {type(error).__name__}: {error}")
        if failure is None:
            failure = error
            failure_traceback = traceback.format_exc()

    if failure is None and cleanup_errors:
        failure = RecordingError("; ".join(cleanup_errors))
        failure_traceback = "Cleanup completed with errors; see summary cleanup evidence."

    if failure is None:
        session.summary["status"] = "succeeded"
        session.summary["passed"] = True
        session.summary["completed"] = timestamp()
        session.actions["status"] = "succeeded"
        exit_code = 0
    else:
        session.summary["status"] = "failed"
        session.summary["passed"] = False
        session.summary["completed"] = timestamp()
        session.summary["failure"] = {
            "type": type(failure).__name__,
            "message": str(failure),
            "traceback": failure_traceback,
        }
        session.actions["status"] = "failed"
        exit_code = 1
    try:
        session.flush_evidence()
    except Exception as write_error:
        print(f"error: final evidence write failed: {write_error}", file=sys.stderr)
        exit_code = 1

    if failure is not None:
        print(f"error: {type(failure).__name__}: {failure}", file=sys.stderr)
        print(f"failure evidence: {output}", file=sys.stderr)
    else:
        print(json.dumps(session.summary, indent=2, sort_keys=True))
        print(output)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
