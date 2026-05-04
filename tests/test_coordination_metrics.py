from __future__ import annotations

import unittest

from marlsim.experiments import format_comparison_table, run_coordination_comparison


class CoordinationMetricsTests(unittest.TestCase):
    def test_comparison_runs_all_current_scenarios_and_policies(self) -> None:
        results = run_coordination_comparison()

        self.assertEqual(len(results), 12)
        self.assertEqual(
            {
                (result.scenario_name, result.policy_name)
                for result in results
            },
            {
                ("simple", "baseline"),
                ("simple", "waiting"),
                ("simple", "priority"),
                ("simple", "replanning"),
                ("narrow_corridor", "baseline"),
                ("narrow_corridor", "waiting"),
                ("narrow_corridor", "priority"),
                ("narrow_corridor", "replanning"),
                ("crossing_paths", "baseline"),
                ("crossing_paths", "waiting"),
                ("crossing_paths", "priority"),
                ("crossing_paths", "replanning"),
            },
        )

    def test_crossing_paths_comparison_captures_coordination_tradeoffs(self) -> None:
        results = {
            result.policy_name: result.summary
            for result in run_coordination_comparison(scenarios=("crossing_paths",))
        }

        self.assertTrue(results["baseline"].succeeded)
        self.assertEqual(results["baseline"].blocked_moves, 1)
        self.assertFalse(results["waiting"].succeeded)
        self.assertTrue(results["priority"].succeeded)
        self.assertEqual(results["priority"].blocked_moves, 0)
        self.assertTrue(results["replanning"].succeeded)
        self.assertEqual(results["replanning"].blocked_moves, 0)

    def test_format_comparison_table_includes_metrics(self) -> None:
        table = format_comparison_table(
            run_coordination_comparison(scenarios=("crossing_paths",))
        )

        self.assertIn("scenario", table)
        self.assertIn("blocked reasons", table)
        self.assertIn("final positions", table)
        self.assertIn("crossing_paths", table)
        self.assertIn("cell_conflict=1", table)


if __name__ == "__main__":
    unittest.main()
