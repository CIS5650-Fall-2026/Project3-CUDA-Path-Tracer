#!/usr/bin/env python3
"""Check the actual driver's completion branch with synthetic mocked boundaries.

The caller must supply a brand-new --output directory. Outputs are explicitly
synthetic policy checks, not render measurements. No renderer or network is used.
The driver is imported from scripts/measure_hedge.py without replacing its
run_measurement or final_gate_status functions. Apply the prepared exit-policy
patch separately before expecting all three policy cases to pass.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import datetime as dt
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import socket
import sys
from typing import Any
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "scripts/measure_hedge.py"
CASES = (("qualified", "passed", 0), ("not-qualified", "not-qualified", 2),
         ("inconclusive", "inconclusive", 2))
BANNER = "SYNTHETIC EXIT-POLICY CHECK: no renderer or network invocation.\n"


class MockSession:
    """In-memory session boundary; persistence snapshots never resemble evidence files."""

    def __init__(self, output: Path, binary: Path, phase: str):
        self.output = output
        self.binary = binary
        self.records: list[dict[str, Any]] = []
        self.initial_binary_identity = {"synthetic": True}
        self.state: dict[str, Any] = {
            "mode": "synthetic-exit-policy-check", "synthetic": True,
            "renderer_executed": False, "status": "running", "execution_status": "running",
            "phase_requested": phase, "identity": {}, "design": {},
            "input_scenes": {}, "phases": {}, "errors": [],
        }
        self.events: list[str] = ["construct mock session"]
        self.persisted_states: list[dict[str, Any]] = []

    def persist(self) -> None:
        self.events.append("persist in memory")
        self.persisted_states.append(copy.deepcopy(self.state))

    def initialize_identity(self) -> None:
        self.events.append("initialize synthetic identity")
        self.state["identity"]["mock_identity"] = True


def import_actual_driver() -> Any:
    # Prevent the import from leaving .pyc files beside shared source files.
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    try:
        spec = importlib.util.spec_from_file_location("_synthetic_exit_policy_actual_driver", DRIVER)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot create an import specification for the actual driver")
        module = importlib.util.module_from_spec(spec)
        # dataclasses resolves annotations through sys.modules during import.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)
        sys.dont_write_bytecode = previous_bytecode_setting


def exercise_completion(driver: Any, qualification: str, expected_status: str,
                        expected_returncode: int, case_path: Path) -> dict[str, Any]:
    """Exercise real run_measurement/final_gate_status, mocking only external boundaries."""
    sessions: list[MockSession] = []
    attempted_renderer_calls: list[str] = []
    attempted_network_calls: list[str] = []

    def construct_session(output: Path, binary: Path, phase: str) -> MockSession:
        session = MockSession(output, binary, phase)
        sessions.append(session)
        return session

    def complete_quality(session: MockSession) -> None:
        session.events.append("complete synthetic quality phase")
        session.state["phases"]["quality"] = {
            "synthetic": True, "phase": "quality", "status": "completed",
            "qualification": {"status": qualification},
        }

    def forbid_renderer(*args: Any, **kwargs: Any) -> None:
        attempted_renderer_calls.append("blocked subprocess invocation")
        raise AssertionError("a synthetic exit-policy case attempted to invoke a subprocess")

    def forbid_network(*args: Any, **kwargs: Any) -> None:
        attempted_network_calls.append("blocked network connection")
        raise AssertionError("a synthetic exit-policy case attempted a network connection")

    def unexpected_phase(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a quality-only synthetic case entered another rendering phase")

    stdout, stderr = io.StringIO(), io.StringIO()
    args = argparse.Namespace(binary=ROOT / ".local/SYNTHETIC-NO-RENDERER", phase="quality")
    unexpected_error = None
    result = None
    with ExitStack() as stack:
        session_boundary = stack.enter_context(mock.patch.object(driver, "MeasurementSession", side_effect=construct_session))
        scene_boundary = stack.enter_context(mock.patch.object(driver, "inspect_scene_inputs", return_value={"synthetic": True}))
        quality_boundary = stack.enter_context(mock.patch.object(driver, "run_quality", side_effect=complete_quality))
        identity_boundary = stack.enter_context(mock.patch.object(driver, "observed_identities", return_value={"passed": True, "synthetic": True}))
        stack.enter_context(mock.patch.object(driver, "run_timing", side_effect=unexpected_phase))
        stack.enter_context(mock.patch.object(driver, "run_gallery", side_effect=unexpected_phase))
        for name in ("Popen", "run", "call", "check_call", "check_output"):
            stack.enter_context(mock.patch.object(driver.subprocess, name, side_effect=forbid_renderer))
        stack.enter_context(mock.patch.object(socket, "create_connection", side_effect=forbid_network))
        stack.enter_context(mock.patch.object(socket.socket, "connect", side_effect=forbid_network))
        stack.enter_context(mock.patch.object(socket.socket, "connect_ex", side_effect=forbid_network))
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                result = driver.run_measurement(args, case_path)
            except BaseException as error:
                unexpected_error = f"{type(error).__name__}: {error}"
        counts = {"session_construction": session_boundary.call_count,
                  "scene_inspection": scene_boundary.call_count,
                  "quality_completion": quality_boundary.call_count,
                  "observed_identity_audit": identity_boundary.call_count}
    session = sessions[0] if len(sessions) == 1 else None
    state = copy.deepcopy(session.state) if session is not None else {}
    final_persisted = copy.deepcopy(session.persisted_states[-1]) if session and session.persisted_states else {}
    checks = {
        "actual_driver_returncode_matches_policy": result == expected_returncode,
        "actual_final_gate_status_matches_policy": state.get("status") == expected_status,
        "execution_remains_completed": state.get("execution_status") == "completed",
        "synthetic_quality_phase_completed": state.get("phases", {}).get("quality", {}).get("status") == "completed",
        "qualification_retained": state.get("phases", {}).get("quality", {}).get("qualification", {}).get("status") == qualification,
        "completed_timestamp_present": isinstance(state.get("completed_utc"), str) and bool(state["completed_utc"]),
        "completed_state_persisted_by_actual_branch": final_persisted == state and len(session.persisted_states if session else []) == 3,
        "completion_calls_all_required_boundaries_once": all(count == 1 for count in counts.values()),
        "no_execution_errors": not state.get("errors") and unexpected_error is None,
        "no_subprocess_attempted": not attempted_renderer_calls,
        "no_network_attempted": not attempted_network_calls,
        "actual_completion_stdout_present": str(case_path) in stdout.getvalue(),
        "actual_completion_stderr_empty": not stderr.getvalue(),
    }
    return {
        "mode": "synthetic-exit-policy-check", "synthetic": True,
        "qualification_case": qualification, "status": "passed" if all(checks.values()) else "failed",
        "expected_returncode": expected_returncode, "actual_returncode": result,
        "expected_status": expected_status, "checks": checks, "boundary_call_counts": counts,
        "final_in_memory_state": state, "mock_session_events": session.events if session else [],
        "unexpected_error": unexpected_error, "renderer_executed": False,
        "attempted_subprocess_calls": attempted_renderer_calls, "attempted_network_calls": attempted_network_calls,
        "stdout": BANNER + stdout.getvalue(), "stderr": BANNER + stderr.getvalue(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="brand-new directory for clearly labeled synthetic policy report/logs")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"refusing to overwrite an existing output path: {output}")
    driver_bytes = DRIVER.read_bytes()
    start_hash = hashlib.sha256(driver_bytes).hexdigest()
    driver = import_actual_driver()
    # The actual entrypoint must come from the exact on-disk driver, not a copied expression.
    if Path(driver.run_measurement.__code__.co_filename).resolve() != DRIVER.resolve():
        raise RuntimeError("run_measurement did not originate in the actual driver file")
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for qualification, status, returncode in CASES:
        case = exercise_completion(driver, qualification, status, returncode,
                                   output / (qualification + "-SYNTHETIC-NOT-RENDER-EVIDENCE"))
        stdout = case.pop("stdout")
        stderr = case.pop("stderr")
        case["stdout_log"] = qualification + ".synthetic.stdout.log"
        case["stderr_log"] = qualification + ".synthetic.stderr.log"
        (output / case["stdout_log"]).write_text(stdout, encoding="utf-8")
        (output / case["stderr_log"]).write_text(stderr, encoding="utf-8")
        (output / (qualification + ".synthetic.state.json")).write_text(
            json.dumps(case, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        cases.append(case)
    end_hash = hashlib.sha256(DRIVER.read_bytes()).hexdigest()
    passed = all(case["status"] == "passed" for case in cases) and start_hash == end_hash
    report = {
        "schema_version": 1, "mode": "synthetic-exit-policy-check", "synthetic": True,
        "status": "passed" if passed else "failed", "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "actual_driver_path": str(DRIVER), "actual_driver_sha256_start": start_hash,
        "actual_driver_sha256_end": end_hash, "actual_driver_unchanged": start_hash == end_hash,
        "actual_entrypoints_exercised": ["run_measurement", "final_gate_status"],
        "mocked_boundaries": ["MeasurementSession construction/persistence/identity", "inspect_scene_inputs",
            "run_quality", "observed_identities"],
        "renderer_executed": False, "network_used": False, "numerical_measurements": False,
        "policy": "completed qualified -> 0; completed not-qualified/inconclusive -> 2; all execution_status values remain completed",
        "notice": "Synthetic mocked completion policy integration check. This report contains no render evidence or measured image quality/timing values.",
        "cases": cases,
    }
    report_path = output / "synthetic-exit-policy-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Synthetic exit-policy check {report['status']}: {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
