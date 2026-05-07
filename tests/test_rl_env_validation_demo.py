from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from marlsim.demos.rl_env_validation import run_validation


class RLEnvValidationDemoTests(unittest.TestCase):
    def test_validation_demo_runs_and_prints_core_cases(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            run_validation()

        text = output.getvalue()
        self.assertIn("RL Environment Validation Demo", text)
        self.assertIn("Reset output", text)
        self.assertIn("Normal movement step", text)
        self.assertIn("Blocked movement step", text)
        self.assertIn("Goal-reaching episode", text)
        self.assertIn("Timeout episode", text)
        self.assertIn("not RL training", text)


if __name__ == "__main__":
    unittest.main()
