import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/rebuttal"))
import audit_traces


class AuditAnnotationTests(unittest.TestCase):
    def test_export_all_versions_blinded_and_independent(self):
        row = dict(sample_id="sample", message_id="message", question="1+1?", message="2", role="solver",
                   judgements=[{"verdict": "Correct"}], judgements_state="available", withheld=False,
                   audit_attempts=[dict(message="3", pass_flag=False), dict(message="2", pass_flag=True)])
        blind, key = audit_traces.annotation_sample([row], 100, 7)
        self.assertEqual([v["message"] for v in blind[0]["versions"]], ["3", "2"])
        self.assertTrue(all(v["correctness"] is None for v in blind[0]["versions"]))
        self.assertNotIn("pass_flag", json.dumps(blind))
        self.assertFalse(key[0]["audit_attempts"][0]["pass_flag"])

    def test_double_labels_count_flips_and_flagging_with_distinct_denominators(self):
        key = [{"annotation_id": "case1", "audit_attempts": [
            {"pass_flag": False, "judgements": [{"verdict": "Flawed"}]},
            {"pass_flag": False, "judgements": [{"verdict": "Flawed"}]},
            {"pass_flag": True, "judgements": [{"verdict": "Correct"}]}]}]
        labels = [{"annotation_id": "case1", "annotator": "A", "versions": [
            {"version_id": "v1", "correctness": "incorrect"},
            {"version_id": "v2", "correctness": "correct"},
            {"version_id": "v3", "correctness": "incorrect"}]}]
        other = json.loads(json.dumps(labels)); other[0]["annotator"] = "B"
        report = audit_traces.review_labels(key, labels, other)
        self.assertEqual(report["adjacent_revision_transitions"], {"incorrect_to_correct": 1, "correct_to_incorrect": 1})
        self.assertEqual(report["false_positive"]["agreed_correct_audited_versions"], 1)
        self.assertEqual(report["false_positive"]["failed_audit_among_correct"], 1)
        self.assertEqual(report["annotation_agreement"]["agreed_versions"], 3)
        other[0]["versions"][1]["correctness"] = "incorrect"
        disputed = audit_traces.review_labels(key, labels, other)
        self.assertEqual(disputed["annotation_agreement"]["disagreements"], 1)
        self.assertEqual(disputed["adjacent_revision_transitions"], {})

    def test_missing_human_labels_stay_pending(self):
        key = [{"annotation_id": "case1", "audit_attempts": [{"pass_flag": False}]}]
        labels = [{"annotation_id": "case1", "annotator": "", "versions": [{"version_id": "v1", "correctness": None}]}]
        report = audit_traces.review_labels(key, labels, labels)
        self.assertEqual(report["annotation_agreement"]["pending_versions"], 1)
        self.assertIsNone(report["false_positive"]["rate"])
        self.assertEqual(report["adjacent_revision_transitions"], {})

    def test_unfinished_audit_is_reported_instead_of_becoming_a_final_message(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(json.dumps(dict(event_type="audit_attempt", session_id="s",
                                            audit_message_id="m", attempt=1)), encoding="utf-8")
            self.assertEqual(audit_traces.mechanism_records(path, []), [])
            self.assertEqual(audit_traces.mechanism_diagnostics(path)["unfinalized_audit_attempts"], 1)

    def test_trace_preserves_finalized_records_before_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mechanism.jsonl"
            events = [dict(event_type="message_finalized", session_id="s", phase=0,
                           message_id="before", source="p", message="3", is_pruned=True,
                           judgements=[{"verdict": "Flawed"}], audit_attempts=[dict(question="1+1?", message="3", pass_flag=False)]),
                      dict(event_type="scoreboard_reset", session_id="s", phase=0),
                      dict(event_type="message_finalized", session_id="s", phase=1,
                           message_id="after", source="p", message="2", is_pruned=False,
                           judgements=[], audit_attempts=[])]
            path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
            records = audit_traces.mechanism_records(path, [])
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["audit_attempts"][0]["message"], "3")
            self.assertFalse(records[0]["sample_identity_verified"])


if __name__ == "__main__":
    unittest.main()
