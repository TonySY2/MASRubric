"""The legacy readiness gate must not report an unconfigured archive as ready."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LegacyPreflightTests(unittest.TestCase):
    def run_inspector(self, *arguments):
        env = os.environ.copy()
        env.pop("MASRUBRIC_ARCHIVE_ROOT", None)
        env["PYTHONIOENCODING"] = "utf-8"
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/rebuttal/inspect_legacy.py"), *arguments],
            env=env, capture_output=True, text=True, encoding="utf-8", check=False,
        )

    def test_default_without_archive_still_lists_catalog(self):
        process = self.run_inspector()
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertIn("roots", result)
        self.assertTrue(result["programs"])
        self.assertNotIn("source_ready", result)

    def test_strict_without_archive_reports_all_programs_missing(self):
        process = self.run_inspector("--strict")
        self.assertEqual(process.returncode, 1, process.stderr)
        result = json.loads(process.stdout)
        catalog = json.loads((ROOT / "configs/rebuttal_legacy_programs.json").read_text(encoding="utf-8"))
        self.assertFalse(result["source_ready"])
        self.assertEqual(len(result["programs"]), len(catalog["programs"]))
        self.assertTrue(all(item["status"] == "missing" for item in result["programs"]))
        self.assertTrue(all(item["errors"] for item in result["programs"]))


if __name__ == "__main__":
    unittest.main()
