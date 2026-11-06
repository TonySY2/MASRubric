#!/usr/bin/env python3
"""Check preserved legacy entrypoints by parsing source, without importing it."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path(os.environ["MASRUBRIC_ARCHIVE_ROOT"]) if os.environ.get("MASRUBRIC_ARCHIVE_ROOT") else None)
    parser.add_argument("--ac-root", type=Path)
    parser.add_argument("--fixed-root", type=Path)
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    catalog = json.loads((ROOT / "configs/rebuttal_legacy_programs.json").read_text(encoding="utf-8"))
    if not any((args.archive_root, args.ac_root, args.fixed_root, args.training_root)):
        print(json.dumps(catalog, ensure_ascii=False, indent=2))
        return 0
    roots = {}
    for key, rel in catalog["roots"].items():
        roots[key] = getattr(args, key + "_root") or (args.archive_root / rel if args.archive_root else None)
    report = []
    for entry in catalog["programs"]:
        root = roots[entry["root_key"]]
        path = root / entry["path"] if root else None
        item = {"root_key": entry["root_key"], "path": entry["path"]}
        if path is None or not path.is_file():
            item.update(status="missing", errors=["Root not configured or file missing"])
        else:
            try:
                source = path.read_text(encoding="utf-8-sig")
                tree = ast.parse(source)
                flags = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                        flags.update(value.value for value in node.args if isinstance(value, ast.Constant) and isinstance(value.value, str))
                expected = {flag for option in entry["arguments"] for flag in option["flags"]}
                missing = sorted(expected - flags)
                damaged = sorted(flag for flag in flags if "REMOVED_HISTORICAL" in flag)
                errors = (["Catalogued flags missing: " + ", ".join(missing)] if missing else [])
                errors += (["Damaged historical flags: " + ", ".join(damaged)] if damaged else [])
                item.update(status="invalid" if errors else "source_ready", errors=errors, cli_flags=sorted(flags))
            except (OSError, SyntaxError, UnicodeError) as exc:
                item.update(status="invalid", errors=[str(exc)])
        report.append(item)
    ready = all(item["status"] == "source_ready" for item in report)
    print(json.dumps({"source_ready": ready, "programs": report}, ensure_ascii=False, indent=2))
    return 1 if args.strict and not ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
