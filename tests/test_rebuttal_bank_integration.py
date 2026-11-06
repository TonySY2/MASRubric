"""Run the offline bank pipeline through real child processes and loopback HTTP."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_paper_runtime_integration import PaperModelEndpoint, metric

ROOT = Path(__file__).resolve().parents[1]


class MiningEndpoint(PaperModelEndpoint):
    def reply(self, path, body):
        response = super().reply(path, body)
        if path.endswith("/embeddings"):
            return response
        prompt = "\n".join(str(item.get("content", "")) for item in body.get("messages", []))
        if "Lead Mathematics Auditor and Logic Specialist" in prompt:
            response["choices"][0]["message"]["content"] = json.dumps([
                dict(metric("criterion_a"), specific_diagnostic_report="Fixture answer conflicts with the reference."),
                dict(metric("criterion_b"), specific_diagnostic_report="Fixture final-answer consistency check.")])
            self.calls[-1]["stage"] = "criterion_mining"
        elif "Check if the NEW metric describes the SAME Logic Error" in prompt:
            response["choices"][0]["message"]["content"] = json.dumps({"is_duplicate": False})
            self.calls[-1]["stage"] = "deduplication"
        return response


class BankPipelineTests(unittest.TestCase):
    def test_failure_collection_deduplication_and_embedding_complete(self):
        with tempfile.TemporaryDirectory() as directory, MiningEndpoint() as endpoint:
            root = Path(directory)
            source = root / "train.jsonl"
            source.write_text(json.dumps({"id": "0", "problem": "What is 1+1?",
                                          "solution": r"\boxed{3}", "answer": "3"}) + "\n", encoding="utf-8")
            env = {key: value for key, value in os.environ.items() if not key.startswith("MASRUBRIC_")}
            env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", NO_PROXY="127.0.0.1,localhost",
                       no_proxy="127.0.0.1,localhost")
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
                env.pop(key, None)
            for role in ("SELECTOR", "REASONING", "SUPERVISOR", "EMBEDDING"):
                env[f"{role}_URL"] = endpoint.url
                env[f"{role}_MODEL"] = "embedding-local" if role == "EMBEDDING" else "gpt-4o"
                env[f"{role}_KEY"] = "local-test"

            def run(stage, input_file, output, extras=()):
                command = [sys.executable, str(ROOT / "scripts/rebuttal/build_bank.py"), stage,
                           "--input-file", str(input_file), "--output-dir", str(output), "--execute", *extras]
                result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                                        encoding="utf-8", errors="replace", timeout=90)
                log = output / "process.log"
                detail = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
                self.assertEqual(result.returncode, 0, (result.stdout + result.stderr + detail)[-8000:])
                report = json.loads((output / "validation.json").read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "complete", report)
                return report

            collected = root / "collected"
            collection = run("collect-math", source, collected, ("--limit", "1", "--max-turns", "3"))
            self.assertEqual(collection["after"]["tasks"], 1)
            self.assertEqual(collection["after"]["failed_tasks"], 1)
            self.assertGreaterEqual(collection["after"]["raw_criterion_occurrences"], 2)
            bank = root / "bank"
            dedup = run("deduplicate", collected / "trajectories.json", bank)
            self.assertEqual(dedup["after"]["bank"]["criteria"], 2)
            self.assertEqual(dedup["after"]["cache"]["records"], 2)
            embedded = run("embed", bank / "bank.json", root / "embeddings")
            self.assertEqual(embedded["after"]["cache"]["missing_names"], 0)
            self.assertEqual(embedded["after"]["cache"]["records"], 2)
            stages = {call["stage"] for call in endpoint.calls}
            self.assertTrue({"criterion_mining", "deduplication", "embedding"}.issubset(stages), stages)


if __name__ == "__main__":
    unittest.main()
