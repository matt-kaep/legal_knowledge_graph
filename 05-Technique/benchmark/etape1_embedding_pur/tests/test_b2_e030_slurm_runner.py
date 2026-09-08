from pathlib import Path
import os
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sbatch_b2_e030_judge_v1.sh"


class E030SlurmRunnerTest(unittest.TestCase):
    def test_refuses_to_run_without_isolated_repository_path(self):
        """A future runner that guesses a repository path must fail this safety gate."""
        environment = os.environ.copy()
        for key in ("LKG_REPO", "LKG_DATA_ROOT", "E030_EXECUTION_MANIFEST", "E030_EXECUTION_MANIFEST_SHA", "E030_SHARD"):
            environment.pop(key, None)

        completed = subprocess.run(
            ["bash", str(SCRIPT)],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Set LKG_REPO", completed.stderr)


if __name__ == "__main__":
    unittest.main()
