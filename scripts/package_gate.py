"""Validate built distribution artifacts before publishing."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / ".github" / "package-gate.json"
DEFAULT_DIST = ROOT / "dist"


@dataclass(frozen=True)
class ArtifactMember:
    artifact: Path
    name: str
    size: int


def load_config(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    return {key: int(value) for key, value in payload.items()}


def bytes_label(value: int) -> str:
    mib = value / (1024 * 1024)
    return f"{value} bytes ({mib:.2f} MiB)"


def pass_or_fail(ok: bool, message: str) -> bool:
    print(("PASS: " if ok else "FAIL: ") + message)
    return ok


def wheel_members(path: Path) -> list[ArtifactMember]:
    with zipfile.ZipFile(path) as archive:
        return [
            ArtifactMember(path, member.filename, member.file_size)
            for member in archive.infolist()
            if not member.is_dir()
        ]


def sdist_members(path: Path) -> list[ArtifactMember]:
    with tarfile.open(path, "r:gz") as archive:
        return [
            ArtifactMember(path, member.name, member.size)
            for member in archive.getmembers()
            if member.isfile()
        ]


def largest_members(
    members: Iterable[ArtifactMember], limit: int = 10
) -> list[ArtifactMember]:
    return sorted(members, key=lambda item: item.size, reverse=True)[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dist", type=Path, default=DEFAULT_DIST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    dist = args.dist

    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    artifacts = wheels + sdists

    ok = True
    ok &= pass_or_fail(
        len(wheels) == config["expected_wheels"],
        f"wheel count = {len(wheels)}, expected {config['expected_wheels']}",
    )
    ok &= pass_or_fail(
        len(sdists) == config["expected_sdists"],
        f"sdist count = {len(sdists)}, expected {config['expected_sdists']}",
    )

    total_size = sum(path.stat().st_size for path in artifacts)
    ok &= pass_or_fail(
        total_size <= config["max_dist_total_bytes"],
        "dist total size = "
        f"{bytes_label(total_size)}, limit {bytes_label(config['max_dist_total_bytes'])}",
    )

    for artifact in artifacts:
        size = artifact.stat().st_size
        ok &= pass_or_fail(
            size <= config["max_artifact_bytes"],
            f"{artifact.name} size = {bytes_label(size)}, "
            f"limit {bytes_label(config['max_artifact_bytes'])}",
        )

    all_members: list[ArtifactMember] = []
    for wheel in wheels:
        members = wheel_members(wheel)
        all_members.extend(members)
        uncompressed_size = sum(member.size for member in members)
        ok &= pass_or_fail(
            uncompressed_size <= config["max_wheel_uncompressed_bytes"],
            f"{wheel.name} uncompressed size = {bytes_label(uncompressed_size)}, "
            f"limit {bytes_label(config['max_wheel_uncompressed_bytes'])}",
        )

    for sdist in sdists:
        all_members.extend(sdist_members(sdist))

    largest = largest_members(all_members)
    print("Largest packaged files:")
    for member in largest:
        print(f"- {member.artifact.name}: {bytes_label(member.size)} {member.name}")

    for member in all_members:
        ok &= member.size <= config["max_member_bytes"]
    pass_or_fail(
        all(member.size <= config["max_member_bytes"] for member in all_members),
        f"each packaged file <= {bytes_label(config['max_member_bytes'])}",
    )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
