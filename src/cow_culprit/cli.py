"""``cow-culprit`` command-line entry point: the static linter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .linter import Finding, check_file


def _iter_python_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.py")))
        elif p.suffix == ".py":
            files.append(p)
        else:
            print(f"cow-culprit: skipping non-Python path: {p}", file=sys.stderr)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cow-culprit",
        description=(
            "Flag in-place mutation of arrays obtained from pandas accessors "
            "(.values, .to_numpy(), np.asarray(df), .array) that can be "
            "read-only under pandas Copy-on-Write."
        ),
    )
    parser.add_argument("paths", nargs="+", help="Python files or directories to check")
    args = parser.parse_args(argv)

    files = _iter_python_files(args.paths)
    if not files:
        print("cow-culprit: no Python files found", file=sys.stderr)
        return 2

    all_findings: list[Finding] = []
    had_error = False
    for path in files:
        try:
            all_findings.extend(check_file(path))
        except SyntaxError as exc:
            print(f"cow-culprit: could not parse {path}: {exc}", file=sys.stderr)
            had_error = True

    for finding in all_findings:
        print(finding.format())

    if had_error:
        return 2
    if all_findings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
