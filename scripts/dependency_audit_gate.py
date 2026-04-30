"""Fail CI when pip-audit reports vulnerabilities outside the baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / ".github" / "pip-audit-baseline.json"


AuditKey = tuple[str, str, str]


def load_baseline(path: Path) -> set[AuditKey]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    keys = set()
    for item in payload.get("vulnerabilities", []):
        keys.add(
            (
                str(item["name"]).lower(),
                str(item["version"]),
                str(item["id"]),
            )
        )
    return keys


def run_pip_audit() -> subprocess.CompletedProcess[str]:
    command = ["pip-audit", "-f", "json"]
    print("$ " + " ".join(command))
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def parse_audit_json(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    raw = result.stdout.strip()
    if not raw:
        raw = f"{result.stdout}\n{result.stderr}".strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise RuntimeError("pip-audit did not emit JSON output")
    return json.loads(raw[start : end + 1])


def current_vulnerabilities(payload: dict[str, Any]) -> set[AuditKey]:
    keys = set()
    for dependency in payload.get("dependencies", []):
        name = str(dependency.get("name", "")).lower()
        version = str(dependency.get("version", ""))
        for vulnerability in dependency.get("vulns", []):
            keys.add((name, version, str(vulnerability.get("id", ""))))
    return keys


def format_key(key: AuditKey) -> str:
    name, version, vulnerability_id = key
    return f"{name} {version} {vulnerability_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = load_baseline(args.baseline)
    result = run_pip_audit()

    try:
        payload = parse_audit_json(result)
    except (RuntimeError, json.JSONDecodeError) as exc:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    current = current_vulnerabilities(payload)
    new_vulnerabilities = sorted(current - baseline)
    resolved = sorted(baseline - current)

    print(f"Current dependency vulnerabilities: {len(current)}")
    print(f"Baseline dependency vulnerabilities: {len(baseline)}")

    if resolved:
        print("Resolved baseline entries:")
        for key in resolved:
            print(f"- {format_key(key)}")

    if new_vulnerabilities:
        print("New dependency vulnerabilities:")
        for key in new_vulnerabilities:
            print(f"- {format_key(key)}")
        return 1

    print("PASS: dependency vulnerabilities are within baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
