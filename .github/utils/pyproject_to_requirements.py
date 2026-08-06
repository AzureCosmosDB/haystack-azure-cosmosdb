# SPDX-FileCopyrightText: 2024-present Microsoft Corporation
#
# SPDX-License-Identifier: MIT
"""Convert a `pyproject.toml` file to a flat list of requirements.

Used by the license-compliance workflow to feed direct dependencies to the license checker.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import toml


def main(pyproject_path: Path, *, include_optional: bool = False) -> None:
    content = toml.load(pyproject_path)
    deps = set(content["project"].get("dependencies", []))

    if include_optional:
        for dep_list in content["project"].get("optional-dependencies", {}).values():
            deps.update(dep_list)

    sys.stdout.write("\n".join(sorted(deps)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="pyproject_to_requirements.py",
        description="Convert pyproject.toml to a requirements list",
    )
    parser.add_argument("pyproject_path", type=Path, help="Path to the pyproject.toml file")
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also include optional-dependencies (extras)",
    )
    args = parser.parse_args()
    main(args.pyproject_path, include_optional=args.include_optional)
