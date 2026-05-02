"""Keep package metadata version aligned with git release tags."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYPROJECT = ROOT / "pyproject.toml"

SECTION_RE = re.compile(r"^\[[^\]]+\]\s*$")
STABLE_TAG_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
TAG_VERSION_RE = re.compile(r"^v?([0-9]+[.][0-9]+[.][0-9]+[A-Za-z0-9.+_-]*)$")
STABLE_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
VERSION_LINE_RE = re.compile(
    r'^(?P<prefix>\s*version\s*=\s*")(?P<version>[^"]+)(?P<suffix>".*)$'
)


@dataclass(frozen=True)
class StableTag:
    tag: str
    version: str
    key: tuple[int, int, int]


def read_pyproject_version(pyproject: Path) -> str:
    in_poetry_section = False

    for line in pyproject.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "[tool.poetry]":
            in_poetry_section = True
            continue
        if in_poetry_section and SECTION_RE.match(stripped):
            break
        if not in_poetry_section:
            continue

        match = VERSION_LINE_RE.match(line)
        if match:
            return match.group("version")

    raise ValueError(f"Cannot find [tool.poetry].version in {pyproject}")


def write_pyproject_version(pyproject: Path, version: str) -> None:
    lines = pyproject.read_text(encoding="utf-8").splitlines(keepends=True)
    in_poetry_section = False
    changed = False

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "[tool.poetry]":
            in_poetry_section = True
            continue
        if in_poetry_section and SECTION_RE.match(stripped):
            break
        if not in_poetry_section:
            continue

        match = VERSION_LINE_RE.match(line.rstrip("\n"))
        if match:
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = (
                f"{match.group('prefix')}{version}{match.group('suffix')}{newline}"
            )
            changed = True
            break

    if not changed:
        raise ValueError(f"Cannot update [tool.poetry].version in {pyproject}")

    pyproject.write_text("".join(lines), encoding="utf-8")


def git_tags() -> list[str]:
    result = subprocess.run(
        ["git", "tag", "--list"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def stable_tags(tags: list[str]) -> list[StableTag]:
    parsed = []
    for tag in tags:
        match = STABLE_TAG_RE.match(tag)
        if not match:
            continue
        major, minor, patch = (int(value) for value in match.groups())
        parsed.append(
            StableTag(
                tag=tag,
                version=f"{major}.{minor}.{patch}",
                key=(major, minor, patch),
            )
        )
    return parsed


def latest_stable_tag() -> StableTag:
    tags = stable_tags(git_tags())
    if not tags:
        raise ValueError(
            "No stable semver tags found. Fetch tags before running this gate."
        )
    return max(tags, key=lambda item: item.key)


def normalize_tag_version(tag_or_version: str) -> str:
    match = TAG_VERSION_RE.match(tag_or_version.strip())
    if not match:
        raise ValueError(f"Tag '{tag_or_version}' is not a valid package version tag.")
    return match.group(1)


def check_version(pyproject_version: str, expected_version: str, source: str) -> bool:
    if pyproject_version == expected_version:
        print(f"PASS: pyproject.toml version {pyproject_version} matches {source}.")
        return True

    print(
        "FAIL: pyproject.toml version "
        f"{pyproject_version} does not match {source} ({expected_version}).",
        file=sys.stderr,
    )
    return False


def parse_stable_version(version: str) -> tuple[int, int, int] | None:
    match = STABLE_VERSION_RE.match(version)
    if not match:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def is_valid_next_increment(
    latest_key: tuple[int, int, int], candidate_key: tuple[int, int, int]
) -> bool:
    """Return True iff ``candidate_key`` is exactly one valid semver increment ahead.

    Acceptable increments relative to a latest tag ``X.Y.Z``:
      * patch:  ``X.Y.(Z+1)``
      * minor:  ``X.(Y+1).0``
      * major:  ``(X+1).0.0``
    """
    lmaj, lmin, lpatch = latest_key
    return candidate_key in {
        (lmaj, lmin, lpatch + 1),
        (lmaj, lmin + 1, 0),
        (lmaj + 1, 0, 0),
    }


def check_release_ready(pyproject_version: str, latest: StableTag) -> bool:
    """Allow ``pyproject_version`` to either match ``latest`` or be one step ahead.

    Used by the pre-merge gate on release-prep PRs so the version bump can land
    on the integration branch before the matching tag exists.
    """
    if pyproject_version == latest.version:
        print(
            f"PASS: pyproject.toml version {pyproject_version} matches "
            f"latest stable tag {latest.tag}."
        )
        return True

    candidate_key = parse_stable_version(pyproject_version)
    if candidate_key is None:
        print(
            f"FAIL: pyproject.toml version '{pyproject_version}' is not a stable "
            "semver string (X.Y.Z); release-prep gate requires a clean release version.",
            file=sys.stderr,
        )
        return False

    if is_valid_next_increment(latest.key, candidate_key):
        print(
            f"PASS: pyproject.toml version {pyproject_version} is a valid next "
            f"increment from latest stable tag {latest.tag} ({latest.version})."
        )
        return True

    lmaj, lmin, lpatch = latest.key
    expected = ", ".join(
        [
            latest.version,
            f"{lmaj}.{lmin}.{lpatch + 1}",
            f"{lmaj}.{lmin + 1}.0",
            f"{lmaj + 1}.0.0",
        ]
    )
    print(
        f"FAIL: pyproject.toml version {pyproject_version} is not aligned with "
        f"latest stable tag {latest.tag} ({latest.version}). "
        f"Expected one of: {expected}.",
        file=sys.stderr,
    )
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=DEFAULT_PYPROJECT)
    parser.add_argument("--print", action="store_true", help="Print detected versions.")
    parser.add_argument(
        "--check-latest-tag",
        action="store_true",
        help="Require pyproject.toml version to match the latest stable git tag.",
    )
    parser.add_argument(
        "--check-release-ready",
        action="store_true",
        help=(
            "Require pyproject.toml version to either match the latest stable git "
            "tag or be a valid one-step increment (patch+1, minor+1.0, or "
            "major+1.0.0). Use this on release-prep PRs so the version bump can "
            "land before the matching tag exists."
        ),
    )
    parser.add_argument(
        "--check-tag",
        metavar="TAG_OR_VERSION",
        help="Require pyproject.toml version to match a specific release tag.",
    )
    parser.add_argument(
        "--sync-latest-tag",
        action="store_true",
        help="Update pyproject.toml version from the latest stable git tag.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    did_work = False

    try:
        pyproject_version = read_pyproject_version(args.pyproject)

        if args.print:
            did_work = True
            print(f"pyproject.toml version: {pyproject_version}")
            latest = latest_stable_tag()
            print(f"latest stable tag: {latest.tag} ({latest.version})")

        if args.check_latest_tag:
            did_work = True
            latest = latest_stable_tag()
            if not check_version(
                pyproject_version, latest.version, f"latest stable tag {latest.tag}"
            ):
                return 1

        if args.check_release_ready:
            did_work = True
            latest = latest_stable_tag()
            if not check_release_ready(pyproject_version, latest):
                return 1

        if args.check_tag:
            did_work = True
            expected_version = normalize_tag_version(args.check_tag)
            if not check_version(
                pyproject_version, expected_version, f"tag {args.check_tag}"
            ):
                return 1

        if args.sync_latest_tag:
            did_work = True
            latest = latest_stable_tag()
            if pyproject_version == latest.version:
                print(f"PASS: pyproject.toml already uses {latest.version}.")
            else:
                write_pyproject_version(args.pyproject, latest.version)
                print(
                    "Updated pyproject.toml version "
                    f"from {pyproject_version} to {latest.version}."
                )

    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if not did_work:
        print("FAIL: no action requested.", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
