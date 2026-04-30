"""Fail CI when quality findings exceed the checked-in baseline."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / ".github" / "ci-baselines.json"


class GateFailure(RuntimeError):
    """Raised when a gate cannot produce a reliable finding count."""


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    print(f"::group::{command[0]} gate")
    print("$ " + " ".join(command))
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def print_output(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)


def finish_group() -> None:
    print("::endgroup::")


def load_baseline(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def require_thresholds(
    baseline: dict[str, Any],
    gate: str,
    required_keys: tuple[str, ...],
) -> dict[str, int]:
    values = baseline.get(gate)
    if not isinstance(values, dict):
        raise GateFailure(f"Baseline for '{gate}' is missing")

    thresholds: dict[str, int] = {}
    for key in required_keys:
        value = values.get(key)
        if not isinstance(value, int):
            raise GateFailure(f"Baseline key '{gate}.{key}' must be an integer")
        thresholds[key] = value
    return thresholds


def check_count(gate: str, count: int, maximum: int, label: str = "total") -> bool:
    status = "PASS" if count <= maximum else "FAIL"
    print(f"{status}: {gate} {label} findings = {count}, baseline <= {maximum}")
    return count <= maximum


def flake8_gate(baseline: dict[str, Any]) -> bool:
    thresholds = require_thresholds(baseline, "flake8", ("max_total",))
    result = run_command(
        [
            "flake8",
            "tensoraerospace",
            "--max-complexity=10",
            "--max-line-length=127",
            "--statistics",
        ]
    )
    print_output(result)
    finish_group()

    output = f"{result.stdout}\n{result.stderr}"
    count = len(re.findall(r"^[^:\n]+:\d+:\d+:\s+[A-Z]\d{3}\b", output, re.M))
    if result.returncode not in (0, 1):
        raise GateFailure(f"flake8 exited with unexpected status {result.returncode}")
    return check_count("flake8", count, thresholds["max_total"])


def ruff_gate(baseline: dict[str, Any]) -> bool:
    thresholds = require_thresholds(baseline, "ruff", ("max_total",))
    result = run_command(
        [
            "ruff",
            "check",
            "tensoraerospace",
            "--output-format=json",
        ]
    )
    try:
        findings = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        print_output(result)
        finish_group()
        raise GateFailure("ruff did not emit valid JSON") from exc

    for finding in findings[:50]:
        location = finding.get("location", {})
        print(
            "{filename}:{row}:{column}: {code} {message}".format(
                filename=finding.get("filename", "<unknown>"),
                row=location.get("row", "?"),
                column=location.get("column", "?"),
                code=finding.get("code", "?"),
                message=finding.get("message", ""),
            )
        )
    if len(findings) > 50:
        print(f"... {len(findings) - 50} more ruff findings omitted")
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    finish_group()

    if result.returncode not in (0, 1):
        raise GateFailure(f"ruff exited with unexpected status {result.returncode}")
    return check_count("ruff", len(findings), thresholds["max_total"])


def mypy_gate(baseline: dict[str, Any]) -> bool:
    thresholds = require_thresholds(baseline, "mypy", ("max_total",))
    result = run_command(
        [
            "mypy",
            "tensoraerospace",
            "--ignore-missing-imports",
            "--no-error-summary",
            "--hide-error-context",
            "--show-error-codes",
        ]
    )
    print_output(result)
    finish_group()

    output = f"{result.stdout}\n{result.stderr}"
    count = len(re.findall(r"^.+?:\d+(?::\d+)?: error:", output, re.M))
    if result.returncode not in (0, 1):
        raise GateFailure(f"mypy exited with unexpected status {result.returncode}")
    return check_count("mypy", count, thresholds["max_total"])


def bandit_gate(baseline: dict[str, Any]) -> bool:
    thresholds = require_thresholds(
        baseline,
        "bandit",
        ("max_total", "max_medium", "max_high"),
    )
    result = run_command(
        [
            "bandit",
            "-c",
            "pyproject.toml",
            "-r",
            "tensoraerospace",
            "-f",
            "json",
            "-q",
        ]
    )
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        print_output(result)
        finish_group()
        raise GateFailure("bandit did not emit valid JSON") from exc

    findings = payload.get("results", [])
    severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for finding in findings:
        severity = str(finding.get("issue_severity", "")).upper()
        if severity in severity_counts:
            severity_counts[severity] += 1

    for finding in findings[:50]:
        print(
            "{filename}:{line}: {severity} {test_id} {text}".format(
                filename=finding.get("filename", "<unknown>"),
                line=finding.get("line_number", "?"),
                severity=finding.get("issue_severity", "?"),
                test_id=finding.get("test_id", "?"),
                text=finding.get("issue_text", ""),
            )
        )
    if len(findings) > 50:
        print(f"... {len(findings) - 50} more bandit findings omitted")
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    finish_group()

    if result.returncode not in (0, 1):
        raise GateFailure(f"bandit exited with unexpected status {result.returncode}")

    ok = check_count("bandit", len(findings), thresholds["max_total"])
    ok &= check_count(
        "bandit",
        severity_counts["MEDIUM"],
        thresholds["max_medium"],
        "medium",
    )
    ok &= check_count(
        "bandit",
        severity_counts["HIGH"],
        thresholds["max_high"],
        "high",
    )
    return ok


GATES: dict[str, Callable[[dict[str, Any]], bool]] = {
    "flake8": flake8_gate,
    "ruff": ruff_gate,
    "mypy": mypy_gate,
    "bandit": bandit_gate,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gates",
        nargs="*",
        choices=sorted(GATES),
        default=sorted(GATES),
        help="Gates to run. Defaults to all gates.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Path to the JSON baseline file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = load_baseline(args.baseline)
    ok = True

    for gate in args.gates:
        try:
            ok &= GATES[gate](baseline)
        except GateFailure as exc:
            finish_group()
            print(f"FAIL: {gate}: {exc}", file=sys.stderr)
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
