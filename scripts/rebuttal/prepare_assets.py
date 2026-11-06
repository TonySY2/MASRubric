#!/usr/bin/env python3
"""Inventory and install the preserved MASRubric assets without model calls."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from urllib.parse import quote
from urllib.request import urlopen
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/paper_main.json"
LIVECODE_IDS = ROOT / "configs/rebuttal_livecode_v1_ids.json"
ID_FIELDS = ("id", "task_id", "question_id", "unique_id", "problem_idx", "name")
EXPECTED_BANK_COUNTS = {"math": 2000, "code": 2545}
DATASET_FIELDS = {
    "gsm8k": ("question", "answer"), "math500": ("problem", "answer"),
    "aqua": ("question", "options", "correct"), "amc23": ("question", "answer"),
    "olympiad": ("question", "final_answer"), "olymmath_easy": ("problem", "answer"),
    "olymmath_hard": ("problem", "answer"), "aime24": ("problem", "answer"),
    "aime25": ("problem", "answer"), "mbpp": ("prompt", "test_list"),
    "humaneval": ("prompt", "test", "entry_point"), "codecontest": ("problem", "tests"),
    "livecode": ("question_content", "public_test_cases", "private_test_cases", "question_id", "platform"),
}


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream JSONL (including the large LCB archive); handle the AMC wrapper."""
    with path.open(encoding="utf-8-sig") as stream:
        if path.suffix.lower() == ".jsonl":
            for line_no, line in enumerate(stream, 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"Record {line_no} is not an object")
                    yield value
            return
        value = json.load(stream)
    if isinstance(value, dict):
        value = next((value[key] for key in ("data", "test", "problems", "questions")
                      if isinstance(value.get(key), list)), list(value.values()))
    if not isinstance(value, list):
        raise ValueError("Expected a JSON array, wrapped array, or JSONL")
    for row in value:
        if not isinstance(row, dict):
            raise ValueError("Dataset record is not an object")
        yield row


def validate_dataset(path: Path, expected: int, livecode_reference: list[dict] | None = None,
                     required_fields: tuple[str, ...] = ()) -> dict[str, Any]:
    ids: set[str] = set()
    count = duplicate_ids = positional_ids = 0
    fields: set[str] = set()
    livecode_ids = []
    invalid_schema = 0
    for count, row in enumerate(iter_records(path), 1):
        fields.update(row)
        invalid_schema += any(key not in row for key in required_fields)
        value = next((row[key] for key in ID_FIELDS if key in row), None)
        if value is None:
            # CodeContest's preserved converter has no original ID. Keep order.
            positional_ids += 1
            sample_id = f"row:{count - 1}"
        else:
            sample_id = str(value)
        duplicate_ids += sample_id in ids
        ids.add(sample_id)
        if livecode_reference is not None:
            livecode_ids.append((row.get("platform"), str(row.get("question_id"))))
    errors = []
    if count != expected:
        errors.append(f"Expected {expected} records, found {count}")
    if duplicate_ids:
        errors.append(f"Found {duplicate_ids} repeated sample IDs")
    if invalid_schema:
        errors.append(f"{invalid_schema} records lack required fields: {', '.join(required_fields)}")
    if livecode_reference is not None:
        expected_ids = [(row["platform"], row["question_id"]) for row in livecode_reference]
        if livecode_ids != expected_ids:
            errors.append("LiveCodeBench population or order differs from the preserved 400-problem export")
    return {"records": count, "expected_records": expected, "duplicate_ids": duplicate_ids,
            "positional_ids": positional_ids, "fields": sorted(fields), "errors": errors}


def validate_bank(path: Path, expected: int | None = None) -> tuple[dict[str, Any], set[str]]:
    rows = list(iter_records(path))
    names = [row.get("name") for row in rows]
    errors = []
    if expected is not None and len(rows) != expected:
        errors.append(f"Expected {expected} criteria, found {len(rows)}")
    if not rows:
        errors.append("Criterion bank is empty")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        errors.append("Every criterion needs a nonempty string name")
    valid_names = {name for name in names if isinstance(name, str) and name.strip()}
    if len(valid_names) != len(names):
        errors.append("Criterion names are missing or duplicated")
    missing_content = sum(not row.get("detailed_definition") or
                          not isinstance(row.get("evaluator_prompt"), dict) or
                          not row.get("evaluator_prompt", {}).get("risk_alert") for row in rows)
    if missing_content:
        errors.append(f"{missing_content} criteria lack diagnostic definition or audit directive")
    profile_count = sum(isinstance(row.get("match_profile"), dict) and
                        all(row["match_profile"].get(key) for key in ("interface_shape", "io_contract"))
                        for row in rows)
    return {"criteria": len(rows), "expected_criteria": expected,
            "complete_match_profiles": profile_count, "errors": errors}, valid_names


def validate_cache(path: Path, names: set[str], expected_dimension: int | None = None) -> dict[str, Any]:
    seen: set[str] = set()
    dims: set[int] = set()
    duplicates = invalid_vectors = 0
    for row in iter_records(path):
        name, vector = row.get("name"), row.get("vector")
        if not isinstance(name, str) or not name:
            raise ValueError("Embedding record has no criterion name")
        duplicates += name in seen
        seen.add(name)
        valid = isinstance(vector, list) and bool(vector) and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in vector)
        if valid:
            valid = any(x != 0 for x in vector)
            dims.add(len(vector))
        invalid_vectors += not valid
    missing, extra = names - seen, seen - names
    errors = []
    if missing or extra:
        errors.append(f"Cache coverage differs: missing={len(missing)}, extra={len(extra)}")
    if duplicates:
        errors.append(f"Duplicate cache names: {duplicates}")
    if invalid_vectors:
        errors.append(f"Empty, nonfinite, zero, or invalid vectors: {invalid_vectors}")
    if len(dims) != 1:
        errors.append(f"Expected one vector dimension, found {sorted(dims)}")
    if expected_dimension is not None and dims != {expected_dimension}:
        errors.append(f"Expected embedding dimension {expected_dimension}")
    return {"records": len(seen), "dimensions": sorted(dims), "missing_names": len(missing),
            "extra_names": len(extra), "duplicate_names": duplicates,
            "invalid_vectors": invalid_vectors, "errors": errors}


def asset_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [{"id": name, "kind": "dataset", "path": value["dataset"], "count": value["total"]}
             for name, value in config["benchmarks"].items()]
    for domain, value in config["pools"].items():
        specs += [{"id": f"{domain}_bank", "kind": "bank", "domain": domain,
                   "path": value["metrics"], "count": EXPECTED_BANK_COUNTS[domain]},
                  {"id": f"{domain}_cache", "kind": "cache", "domain": domain,
                   "path": value["embeddings"]}]
    return specs


def inventory(assets_root: Path, config: dict[str, Any], embedding_dimension: int | None = None) -> dict[str, Any]:
    rows = []
    bank_names: dict[str, set[str]] = {}
    for spec in asset_specs(config):
        path = assets_root / spec["path"]
        item = {"id": spec["id"], "kind": spec["kind"], "relative_path": spec["path"]}
        if not path.is_file():
            item.update(status="missing", errors=["File is missing"])
        else:
            item["bytes"] = path.stat().st_size
            try:
                if spec["kind"] == "dataset":
                    reference = json.loads(LIVECODE_IDS.read_text(encoding="utf-8"))["records"] if spec["id"] == "livecode" else None
                    details = validate_dataset(path, spec["count"], reference, DATASET_FIELDS[spec["id"]])
                    if reference is not None:
                        details["population_identity"] = "preserved platform/question_id sequence"
                elif spec["kind"] == "bank":
                    details, names = validate_bank(path, spec["count"])
                    bank_names[spec["domain"]] = names
                elif spec["domain"] not in bank_names:
                    details = {"errors": ["Cannot validate cache without its criterion bank"]}
                else:
                    details = validate_cache(path, bank_names[spec["domain"]], embedding_dimension)
                item.update(details)
                item["status"] = "invalid" if details["errors"] else "ready"
            except (OSError, ValueError, KeyError, TypeError) as exc:
                item.update(status="invalid", errors=[str(exc)])
        rows.append(item)
    return {"assets_root": str(assets_root), "ready": all(r["status"] == "ready" for r in rows),
            "checks": "record counts, sample IDs, schema, criterion names, finite vectors, cache coverage",
            "assets": rows}


def install_file(source: Path, target: Path, mode: str) -> str:
    if target.exists() or target.is_symlink():
        return "existing"
    if not source.is_file():
        return "source_missing"
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        # Exclusive creation: a concurrent installer cannot overwrite a file.
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        shutil.copystat(source, target)
    return "installed"


def fetch_livecode(target: Path, revision: str = "main") -> dict[str, Any]:
    """Fetch only release_v1's official file; never execute a dataset script."""
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Preserving existing LiveCodeBench file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".download.jsonl")
    url = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/" + quote(revision, safe="") + "/test.jsonl"
    with partial.open("xb") as output:
        with urlopen(url, timeout=60) as response:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    reference = json.loads(LIVECODE_IDS.read_text(encoding="utf-8"))["records"]
    report = validate_dataset(partial, 400, reference, DATASET_FIELDS["livecode"])
    if report["errors"]:
        raise ValueError("Downloaded file rejected; retained .download.jsonl for inspection: " + "; ".join(report["errors"]))
    # Same-directory hard link installs atomically without replacing an existing
    # file or making a second multi-GB copy. Remove only our temporary filename.
    os.link(partial, target)
    partial.unlink()
    return {"id": "livecode", "action": "downloaded", "source_url": url,
            "population": "matches preserved question IDs and order", "validation": report}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "install", "fetch-livecode"))
    parser.add_argument("--assets-root", type=Path,
                        default=Path(os.environ.get("MASRUBRIC_ASSETS_ROOT", str(ROOT / "paper/assets"))))
    parser.add_argument("--source-root", type=Path,
                        default=Path(os.environ["MASRUBRIC_ASSET_SOURCE"]) if os.environ.get("MASRUBRIC_ASSET_SOURCE") else None)
    parser.add_argument("--mode", choices=("copy", "symlink"), default="symlink")
    parser.add_argument("--livecode-file", type=Path)
    parser.add_argument("--math-cache", type=Path)
    parser.add_argument("--code-cache", type=Path)
    parser.add_argument("--source-revision", default="main", help="Upstream Hugging Face revision for optional LCB download.")
    parser.add_argument("--embedding-dimension", type=int)
    parser.add_argument("--report", type=Path, help="New JSON report; existing files are preserved.")
    parser.add_argument("--strict", action="store_true", help="Return nonzero if any required asset is missing or invalid.")
    args = parser.parse_args(argv)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    actions = []
    if args.action == "fetch-livecode":
        target = args.assets_root / config["benchmarks"]["livecode"]["dataset"]
        actions.append(fetch_livecode(target, args.source_revision))
    if args.action == "install":
        if not any((args.source_root, args.livecode_file, args.math_cache, args.code_cache)):
            parser.error("install needs --source-root or explicit external asset files")
        source_root = args.source_root
        if source_root and (source_root / "paper/assets").is_dir():
            source_root = source_root / "paper/assets"
        overrides = {"livecode": args.livecode_file, "math_cache": args.math_cache, "code_cache": args.code_cache}
        for spec in asset_specs(config):
            source = overrides.get(spec["id"]) or (source_root / spec["path"] if source_root else None)
            if source is not None:
                state = install_file(source, args.assets_root / spec["path"], args.mode)
                actions.append({"id": spec["id"], "action": state})
    report = inventory(args.assets_root.resolve(), config, args.embedding_dimension)
    report["install_actions"] = actions
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    print(rendered)
    return 1 if args.strict and not report["ready"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"Asset preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
