#!/usr/bin/env python3
"""Summarize audit decisions and prepare independent human review of revisions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

from summarize import read_records, sample_id


def audit_records(result_file, diagnostics=None):
    diagnostics = diagnostics if diagnostics is not None else Counter()
    seen = set()
    for row in read_records(result_file):
        identity = sample_id(row)
        if identity in seen:
            raise ValueError(f"Duplicate sample id {identity}")
        seen.add(identity)
        diagnostics["source_samples"] += 1
        scores = row.get("scores")
        if scores is None or scores == {}:
            diagnostics["samples_without_score_records"] += 1
            continue
        if not isinstance(scores, dict):
            diagnostics["malformed_score_containers"] += 1
            continue
        for message_id, score in scores.items():
            if not isinstance(score, dict):
                diagnostics["malformed_score_records"] += 1
                continue
            message = score.get("message") or {}
            content = message.get("content", "") if isinstance(message, dict) else str(message)
            judgments = score.get("judgements")
            state = "missing" if judgments is None else ("available" if isinstance(judgments, list) and all(isinstance(j, dict) for j in judgments) else "malformed")
            if state != "available":
                diagnostics[state + "_judgment_messages"] += 1
                judgments = None
            yield {"sample_id": identity, "message_id": str(message_id),
                   "question": row.get("question") or row.get("task_prompt") or row.get("problem") or "",
                   "message": content, "role": score.get("role"),
                   "judgements": judgments, "judgements_state": state,
                   "withheld": score.get("is_pruned") if isinstance(score.get("is_pruned"), bool) else None,
                   "audit_attempts": score.get("audit_attempts", []),
                   "withholding_policy": score.get("withholding_policy", "unknown"),
                   "action": score.get("action", "unknown"),
                   "mechanism_session_id": score.get("mechanism_session_id"),
                   "mechanism_phase": score.get("mechanism_phase"),
                   "sample_identity_verified": True}


def mechanism_records(path, final_records):
    """Prefer complete finalized episodes, including those discarded by fallback."""
    sessions = {}
    for row in final_records:
        if row.get("mechanism_session_id"):
            session = row["mechanism_session_id"]
            if session in sessions and sessions[session]["sample_id"] != row["sample_id"]:
                raise ValueError("One mechanism session is linked to multiple problem ids")
            sessions[session] = row
    rows, seen = [], set()
    for event in read_records(path):
        if event.get("event_type") != "message_finalized":
            continue
        session, message_id = event["session_id"], str(event["message_id"])
        identity = (session, message_id)
        if identity in seen:
            raise ValueError(f"Duplicate finalized mechanism message: {identity}")
        seen.add(identity)
        parent = sessions.get(session, {})
        attempts = event.get("audit_attempts", [])
        rows.append({"sample_id": parent.get("sample_id", "unresolved_session:" + session),
                     "message_id": message_id, "message": event.get("message", ""),
                     "question": parent.get("question") or next((a.get("question") for a in attempts if a.get("question")), ""),
                     "role": next((a.get("role") for a in attempts if a.get("role")), None),
                     "judgements": event.get("judgements"),
                     "judgements_state": "available" if isinstance(event.get("judgements"), list) else "missing",
                     "withheld": event.get("is_pruned"), "audit_attempts": attempts,
                     "withholding_policy": event.get("withholding_policy", "unknown"),
                     "action": event.get("action", "unknown"),
                     "mechanism_session_id": session, "mechanism_phase": event.get("phase"),
                     "sample_identity_verified": bool(parent)})
    return rows


def mechanism_diagnostics(path):
    observed, finalized = set(), set()
    counts = Counter()
    for event in read_records(path):
        kind = event.get("event_type", "unknown")
        counts[kind] += 1
        if kind == "audit_attempt":
            observed.add((event["session_id"], event["audit_message_id"], event["attempt"]))
        elif kind == "message_finalized":
            finalized.update((event["session_id"], a["audit_message_id"], a["attempt"])
                             for a in event.get("audit_attempts", []))
        elif kind == "scoreboard_reset" and event.get("discarded_message_ids"):
            counts["resets_discarding_scored_messages"] += 1
    return {"events": dict(counts), "unfinalized_audit_attempts": len(observed - finalized),
            "finalized_attempts_without_start_event": len(finalized - observed)}


def annotation_sample(rows, count, seed):
    """Uniform episode sample: each annotator independently labels every version."""
    selected = random.Random(seed).sample(rows, min(count, len(rows)))
    blind, key = [], []
    for i, row in enumerate(selected):
        annotation_id = f"audit_{i + 1:04d}"
        attempts = row.get("audit_attempts") or []
        versions = attempts or [{"message": row["message"]}]
        blind.append({"annotation_id": annotation_id, "question": row["question"],
                      "message": row["message"], "role": row["role"], "annotator": "",
                      "requires_additional_context": None,
                      "context_messages": attempts[0].get("input_messages", []) if attempts else [],
                      "versions": [{"version_id": f"v{j+1}", "message": version["message"],
                                    "correctness": None, "contains_material_error": None,
                                    "error_location": "", "independent_reason": ""}
                                   for j, version in enumerate(versions)]})
        key.append({"annotation_id": annotation_id, **row})
    return blind, key


def _label_index(rows):
    index = {}
    for row in rows:
        annotation_id = row["annotation_id"]
        if annotation_id in index:
            raise ValueError(f"Duplicate annotation id: {annotation_id}")
        versions = {}
        for version in row.get("versions", []):
            version_id = version["version_id"]
            if version_id in versions:
                raise ValueError(f"Duplicate version id: {annotation_id}/{version_id}")
            label = version.get("correctness")
            if label not in (None, "correct", "incorrect", "unverifiable"):
                raise ValueError(f"Invalid correctness label: {label}")
            versions[version_id] = label
        index[annotation_id] = {"versions": versions, "annotator": row.get("annotator", "").strip()}
    return index


def review_labels(key, labels_a, labels_b):
    """Use two agreeing human labels; unresolved cases retain their own denominator."""
    left, right = _label_index(labels_a), _label_index(labels_b)
    agreement, transitions, first_last = Counter(), Counter(), Counter()
    correct_audited = false_flags = any_flaw_on_correct = 0
    expected_ids = {row["annotation_id"] for row in key}
    if (set(left) | set(right)) - expected_ids:
        raise ValueError("Annotation labels contain ids outside the private sample key")
    for row in key:
        identity = row["annotation_id"]
        a, b = left.get(identity, {}), right.get(identity, {})
        attempts = row.get("audit_attempts") or []
        count = len(attempts) or 1
        expected_versions = {f"v{i+1}" for i in range(count)}
        if (set(a.get("versions", {})) | set(b.get("versions", {}))) - expected_versions:
            raise ValueError(f"Annotation contains versions outside the sample key: {identity}")
        agreed = []
        for i in range(count):
            version_id = f"v{i+1}"
            va, vb = a.get("versions", {}).get(version_id), b.get("versions", {}).get(version_id)
            if va is None or vb is None or not a.get("annotator") or not b.get("annotator"):
                agreement["pending_versions"] += 1
                agreed.append(None)
                continue
            if a["annotator"] == b["annotator"]:
                raise ValueError("Independent reviews require different annotator identities")
            if va != vb:
                agreement["disagreements"] += 1
                agreed.append(None)
                continue
            agreement["agreed_versions"] += 1
            if va == "unverifiable":
                agreement["agreed_unverifiable"] += 1
                agreed.append(None)
                continue
            agreed.append(va)
            if va == "correct" and attempts and isinstance(attempts[i].get("pass_flag"), bool):
                correct_audited += 1
                false_flags += not attempts[i]["pass_flag"]
                any_flaw_on_correct += any(j.get("verdict", "").lower() != "correct"
                                          for j in attempts[i].get("judgements", []))
        for before, after in zip(agreed, agreed[1:]):
            if before is not None and after is not None:
                transitions[f"{before}_to_{after}"] += 1
        if len(agreed) > 1 and agreed[0] is not None and agreed[-1] is not None:
            first_last[f"{agreed[0]}_to_{agreed[-1]}"] += 1
    agreement.setdefault("pending_versions", 0)
    agreement.setdefault("disagreements", 0)
    agreement.setdefault("agreed_versions", 0)
    return {"annotation_agreement": dict(agreement),
            "adjacent_revision_transitions": dict(transitions),
            "initial_to_final_transitions": dict(first_last),
            "false_positive": {"agreed_correct_audited_versions": correct_audited,
                               "failed_audit_among_correct": false_flags,
                               "any_flawed_criterion_among_correct": any_flaw_on_correct,
                               "rate": false_flags / correct_audited if correct_audited else None},
            "interpretation": "Human-labeled sampled message versions; final task correctness is not an intermediate-message label. Disagreements require separate adjudication."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sample", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261106, help="Diagnostic sampling seed only")
    p.add_argument("--mechanism-trace", type=Path, help="Use all finalized episodes, including before fallback resets")
    p.add_argument("--annotation-key", type=Path)
    p.add_argument("--labels-a", type=Path)
    p.add_argument("--labels-b", type=Path)
    args = p.parse_args()
    if args.sample < 0:
        p.error("--sample cannot be negative")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    diagnostics = Counter()
    rows = list(audit_records(args.result, diagnostics))
    if args.mechanism_trace:
        rows = mechanism_records(args.mechanism_trace, rows)
    size = Counter(len(row["judgements"]) for row in rows if row["judgements_state"] == "available")
    withheld = Counter("unknown" if row["withheld"] is None else str(row["withheld"]).lower() for row in rows)
    report = {"source_result": str(args.result), "final_scored_messages": len(rows),
              "input_diagnostics": dict(diagnostics),
              "empty_final_judgment_messages": size.get(0, 0),
              "final_rubric_size_histogram": dict(sorted(size.items())),
              "withholding": dict(withheld),
              "denominator": "Final message score records, not all audit attempts or LLM calls",
              "empty_cause": "Unknown for legacy scores without audit_attempts; structured attempts distinguish selection and audit failure."}
    attempts = [attempt for row in rows for attempt in row.get("audit_attempts", [])]
    report["attempts"] = {"recorded": len(attempts),
                          "empty_causes": dict(Counter(a.get("empty_cause", "unknown") for a in attempts if not a.get("judgment_count", len(a.get("judgements", []))))),
                          "actions": dict(Counter(a.get("action", "unknown") for a in attempts)),
                          "no_selected_criteria": sum(a.get("selected_count") == 0 for a in attempts),
                          "missing_judgments_for_selected_criteria": sum(a.get("selected_count", 0) > a.get("judgment_count", 0) for a in attempts)}
    report["withholding_policies"] = dict(Counter(row.get("withholding_policy", "unknown") for row in rows))
    if args.mechanism_trace:
        report["denominator"] = "All finalized episodes in the structured trace, including episodes discarded by scoreboard resets"
        report["trace_diagnostics"] = mechanism_diagnostics(args.mechanism_trace)
    if any((args.annotation_key, args.labels_a, args.labels_b)):
        if not all((args.annotation_key, args.labels_a, args.labels_b)):
            p.error("Use --annotation-key, --labels-a and --labels-b together")
        read = lambda path: json.loads(path.read_text(encoding="utf-8"))
        report["human_review"] = review_labels(read(args.annotation_key), read(args.labels_a), read(args.labels_b))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    labels, key = annotation_sample(rows, args.sample, args.seed)
    for name, value in (("annotation_blind.json", labels), ("annotation_reviewer_a.json", labels),
                        ("annotation_reviewer_b.json", labels), ("annotation_key.private.json", key)):
        (args.output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
