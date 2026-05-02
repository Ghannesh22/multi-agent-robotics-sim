from __future__ import annotations

import unittest

from marlsim.core.actions import Action
from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.experiments import ExperimentLogger, format_run_summary


class ExperimentLoggerTests(unittest.TestCase):
    def test_summary_tracks_success_steps_positions_and_reached_goals(self) -> None:
        env = GridWorldEnv(
            GridWorldConfig(
                width=3,
                height=1,
                agents=(AgentState("agent_1", Position(0, 0), Position(1, 0)),),
            )
        )
        logger = ExperimentLogger("unit")

        logger.record_step(env.step({"agent_1": Action.RIGHT}))
        summary = logger.summary()

        self.assertEqual(summary.scenario_name, "unit")
        self.assertEqual(summary.total_steps, 1)
        self.assertTrue(summary.succeeded)
        self.assertEqual(summary.final_positions, {"agent_1": Position(1, 0)})
        self.assertEqual(summary.reached_goals, frozenset({"agent_1"}))
        self.assertEqual(summary.blocked_moves, 0)
        self.assertEqual(summary.blocked_reasons, {})

    def test_summary_counts_blocked_moves_by_reason(self) -> None:
        env = GridWorldEnv(
            GridWorldConfig(
                width=3,
                height=1,
                agents=(
                    AgentState("agent_1", Position(0, 0), Position(2, 0)),
                    AgentState("agent_2", Position(2, 0), Position(0, 0)),
                ),
            )
        )
        logger = ExperimentLogger("blocked")

        logger.record_step(env.step({"agent_1": Action.RIGHT, "agent_2": Action.LEFT}))
        summary = logger.summary()

        self.assertFalse(summary.succeeded)
        self.assertEqual(summary.total_steps, 1)
        self.assertEqual(summary.blocked_moves, 2)
        self.assertEqual(summary.blocked_reasons, {"cell_conflict": 2})

    def test_format_run_summary_is_readable_console_text(self) -> None:
        env = GridWorldEnv(
            GridWorldConfig(
                width=2,
                height=1,
                agents=(AgentState("agent_1", Position(0, 0), Position(1, 0)),),
            )
        )
        logger = ExperimentLogger("format")

        logger.record_step(env.step({"agent_1": Action.RIGHT}))

        self.assertEqual(
            format_run_summary(logger.summary()),
            "\n".join(
                (
                    "Run summary",
                    "- scenario: format",
                    "- result: success",
                    "- total steps: 1",
                    "- reached goals: agent_1",
                    "- blocked moves: 0",
                    "- blocked reasons: none",
                    "Final positions:",
                    "- agent_1: (1, 0)",
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
