"""Asset failures that can silently invalidate a benchmark or bank rebuild."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "scripts/rebuttal"


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


assets = load("prepare_assets")
bank = load("build_bank")


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def jsonl(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_complete_count_with_repeated_id_is_rejected(self):
        path = self.jsonl("dataset.jsonl", [{"id": "same"}, {"id": "same"}])
        report = assets.validate_dataset(path, 2)
        self.assertEqual(report["records"], 2)
        self.assertTrue(report["errors"])

    def test_same_size_wrong_livecode_population_is_rejected(self):
        path = self.jsonl("lcb.jsonl", [{"platform": "atcoder", "question_id": "different"}])
        report = assets.validate_dataset(path, 1, [{"platform": "atcoder", "question_id": "original"}])
        self.assertIn("population", report["errors"][0])

    def test_livecode_order_is_part_of_smoke_selection(self):
        reference = [{"platform": "atcoder", "question_id": str(i)} for i in range(2)]
        path = self.jsonl("lcb.jsonl", list(reversed(reference)))
        self.assertTrue(assets.validate_dataset(path, 2, reference)["errors"])

    def test_amc_wrapped_json_and_missing_required_field(self):
        path = self.root / "amc.json"
        path.write_text(json.dumps({"data": [{"id": 1, "question": "q", "answer": 2}]}))
        self.assertFalse(assets.validate_dataset(path, 1, required_fields=("question", "answer"))["errors"])
        self.assertTrue(assets.validate_dataset(path, 1, required_fields=("nonexistent",))["errors"])

    def test_cache_duplicates_and_unknown_names_cannot_mask_missing_criteria(self):
        path = self.jsonl("cache.jsonl", [{"name": "a", "vector": [1, 0]},
                                           {"name": "a", "vector": [1, 0]},
                                           {"name": "extra", "vector": [1, 0]}])
        report = assets.validate_cache(path, {"a", "b"})
        self.assertEqual(report["missing_names"], 1)
        self.assertEqual(report["extra_names"], 1)
        self.assertEqual(report["duplicate_names"], 1)
        self.assertTrue(report["errors"])

    def test_nonfinite_zero_and_mixed_dimension_vectors_fail(self):
        path = self.jsonl("cache.jsonl", [{"name": "a", "vector": [float("nan"), 1]},
                                           {"name": "b", "vector": [0, 0]},
                                           {"name": "c", "vector": [1, 0]},
                                           {"name": "d", "vector": [1, 0, 0]}])
        report = assets.validate_cache(path, {"a", "b", "c", "d"})
        self.assertEqual(report["invalid_vectors"], 2)
        self.assertTrue(report["errors"])

    def test_copy_never_overwrites_an_existing_asset(self):
        source, dest = self.root / "source", self.root / "dest"
        source.write_text("new")
        dest.write_text("preserved")
        self.assertEqual(assets.install_file(source, dest, "copy"), "existing")
        self.assertEqual(dest.read_text(), "preserved")
        fresh = self.root / "sub/fresh"
        self.assertEqual(assets.install_file(source, fresh, "copy"), "installed")
        self.assertEqual(fresh.read_text(), "new")

    def test_bank_build_plan_does_not_emit_keys_or_credentialed_urls(self):
        secret = "private-example-value-9284"
        env = os.environ.copy()
        env.update(SUPERVISOR_KEY=secret, SUPERVISOR_URL=f"https://user:{secret}@example.invalid/v1")
        result = subprocess.run([sys.executable, str(TOOLS / "build_bank.py"), "collect-math",
                                 "--output-dir", str(self.root / "planned")],
                                env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(secret, result.stdout + result.stderr)
        self.assertFalse((self.root / "planned").exists())

    def test_merge_preserves_colliding_task_ids_from_two_training_sources(self):
        inputs = []
        for index in range(2):
            path = self.root / f"source{index}.json"
            path.write_text(json.dumps({"0": {"is_correct": False, "trajectories": {}}}))
            inputs += ["--input-file", str(path)]
        out = self.root / "merged"
        code = bank.main(["merge-trajectories", *inputs, "--output-dir", str(out), "--execute"])
        self.assertEqual(code, 0)
        rows = json.loads((out / "trajectories.json").read_text())
        self.assertEqual(set(rows), {"source0:0", "source1:0"})
        with self.assertRaises(FileExistsError):
            bank.main(["merge-trajectories", *inputs, "--output-dir", str(out), "--execute"])

    def test_deduplication_refuses_unmined_trajectories_before_creating_output(self):
        path = self.root / "raw.json"
        path.write_text(json.dumps({"0": {"is_correct": False, "trajectories": {}}}))
        out = self.root / "bank"
        env = {key: "example" for key in ("SUPERVISOR_URL", "SUPERVISOR_MODEL", "EMBEDDING_URL", "EMBEDDING_MODEL")}
        with patch.dict(os.environ, env):
            with self.assertRaisesRegex(ValueError, "no generated_adversarial_metrics"):
                bank.main(["deduplicate", "--input-file", str(path), "--output-dir", str(out), "--execute"])
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
