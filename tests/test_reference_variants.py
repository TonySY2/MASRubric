import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rebuttal_reference_launcher", ROOT / "test/run_paper_main.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class ReferenceVariantTests(unittest.TestCase):
    def plan(self, *arguments):
        args = launcher.parser().parse_args(["--suite", "math_8b", "--benchmark", "gsm8k", *arguments])
        config = json.loads(launcher.CONFIG.read_text(encoding="utf-8"))
        return launcher.build_plan(args, config, args.benchmark, Path(tempfile.gettempdir()) / "variant-plan")

    def test_top3_retains_threshold_and_retry_budget(self):
        cmd, _, manifest, _ = self.plan("--variant-name", "top3", "--select-q", "3")
        self.assertEqual(cmd[cmd.index("--select_q") + 1], "3")
        self.assertEqual(cmd[cmd.index("--pass_rate") + 1], "0.6")
        self.assertEqual(cmd[cmd.index("--retries_times") + 1], "3")
        self.assertEqual(manifest["parameter_overrides"], {"select_q": 3})

    def test_zero_retry_remains_zero(self):
        cmd, _, _, _ = self.plan("--variant-name", "withhold_only", "--retry-budget", "0")
        self.assertEqual(cmd[cmd.index("--retries_times") + 1], "0")

    def test_changed_protocol_requires_variant_identity(self):
        with self.assertRaisesRegex(ValueError, "variant-name"):
            self.plan("--select-q", "3")

    def test_invalid_threshold_and_baseline_override_rejected(self):
        for flags in (("--pass-rate", "1.1"), ("--retry-budget", "-1"),
                      ("--method", "baseline", "--select-q", "3")):
            with self.assertRaises(ValueError):
                self.plan("--variant-name", "invalid", *flags)

    def test_humaneval_identity_matches_runner(self):
        self.assertEqual(launcher.input_sample_id({"name": "HumanEval_23_strlen"}, 0, "humaneval"), "HumanEval_23_strlen")

    def test_14b_profile_rejects_8b_services_and_records_override(self):
        environment = {}
        for role, model in {"reasoning": "Qwen3-8B", "supervisor": "Qwen3-8B", "selector": "Qwen3.5-9B", "embedding": "Qwen3-Embedding-8B"}.items():
            environment[role.upper() + "_MODEL"] = model
            environment[role.upper() + "_URL"] = "http://127.0.0.1:1/v1"
        with patch.dict(launcher.os.environ, environment):
            args = launcher.parser().parse_args(["--suite", "math_14b", "--benchmark", "gsm8k"])
            config = json.loads(launcher.CONFIG.read_text(encoding="utf-8"))
            _, _, manifest, missing = launcher.build_plan(args, config, "gsm8k", Path(tempfile.gettempdir()) / "variant-plan")
        errors = launcher.preflight(manifest, missing, args)
        self.assertTrue(any("reasoning: expected Qwen3-14B" in error for error in errors))
        self.assertTrue(any("supervisor: expected Qwen3-14B" in error for error in errors))
        args.allow_model_override = True
        errors = launcher.preflight(manifest, missing, args)
        self.assertFalse(any("model mapping differs" in error for error in errors))
        self.assertEqual(len(manifest["model_name_mismatches"]), 2)


if __name__ == "__main__":
    unittest.main()
