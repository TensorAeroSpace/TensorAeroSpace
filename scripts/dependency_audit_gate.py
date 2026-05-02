"""Fail CI when pip-audit reports vulnerabilities outside the baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from packaging.markers import InvalidMarker, Marker, default_environment

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / ".github" / "pip-audit-baseline.json"
DEFAULT_LOCK_FILE = ROOT / "poetry.lock"
DEFAULT_GROUPS = ("main", "dev", "test")


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


def split_groups(raw: str) -> set[str]:
    groups = {group.strip() for group in raw.split(",") if group.strip()}
    if not groups:
        raise ValueError("--groups must contain at least one Poetry group")
    return groups


def marker_applies(marker: str) -> bool:
    try:
        return Marker(marker).evaluate(default_environment())
    except InvalidMarker as exc:
        raise RuntimeError(f"Invalid Poetry lock marker: {marker!r}") from exc


def package_applies(package: Mapping[str, Any], selected_groups: set[str]) -> bool:
    groups = set(package.get("groups") or ["main"])
    relevant_groups = groups & selected_groups
    if not relevant_groups:
        return False
    if bool(package.get("optional", False)):
        return False

    marker = package.get("markers")
    if marker is None:
        return True
    if isinstance(marker, str):
        return marker_applies(marker)
    if isinstance(marker, dict):
        for group in relevant_groups:
            group_marker = marker.get(group)
            if group_marker is None or marker_applies(str(group_marker)):
                return True
        return False
    raise RuntimeError(f"Unsupported Poetry lock marker type: {type(marker).__name__}")


def build_lock_requirements(lock_file: Path, selected_groups: set[str]) -> list[str]:
    with lock_file.open("rb") as file:
        payload = tomllib.load(file)

    requirements: dict[str, str] = {}
    for package in payload.get("package", []):
        if not isinstance(package, dict):
            continue
        if not package_applies(package, selected_groups):
            continue

        name = str(package["name"]).lower()
        version = str(package["version"])
        previous = requirements.get(name)
        if previous is not None and previous != version:
            raise RuntimeError(
                f"Multiple active locked versions for {name}: {previous}, {version}"
            )
        requirements[name] = version

    return [f"{name}=={version}" for name, version in sorted(requirements.items())]


def run_pip_audit(
    requirements_file: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = ["pip-audit", "-f", "json", "--desc", "off", "--aliases", "off"]
    if requirements_file is not None:
        command = [
            "pip-audit",
            "-r",
            str(requirements_file),
            "--no-deps",
            "--disable-pip",
            "-f",
            "json",
            "--desc",
            "off",
            "--aliases",
            "off",
            "--progress-spinner",
            "off",
        ]
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
    parser.add_argument(
        "--source",
        choices=("poetry-lock", "environment"),
        default="poetry-lock",
        help=(
            "Audit poetry.lock by default. Use 'environment' to audit the current "
            "installed Poetry environment."
        ),
    )
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=DEFAULT_LOCK_FILE,
        help="Path to poetry.lock when --source=poetry-lock.",
    )
    parser.add_argument(
        "--groups",
        default=",".join(DEFAULT_GROUPS),
        help="Comma-separated Poetry groups to audit from poetry.lock.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = load_baseline(args.baseline)

    try:
        selected_groups = split_groups(args.groups)
        if args.source == "poetry-lock":
            requirements = build_lock_requirements(args.lock_file, selected_groups)
            with tempfile.TemporaryDirectory(
                prefix="tensoraerospace-pip-audit-"
            ) as directory:
                requirements_file = Path(directory) / "requirements.txt"
                requirements_file.write_text(
                    "\n".join(requirements) + "\n",
                    encoding="utf-8",
                )
                print(
                    "Auditing poetry.lock groups "
                    f"{', '.join(sorted(selected_groups))}: "
                    f"{len(requirements)} pinned packages"
                )
                result = run_pip_audit(requirements_file)
        else:
            result = run_pip_audit()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

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
