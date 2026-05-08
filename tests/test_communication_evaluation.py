from __future__ import annotations

import unittest

from marlsim.core.environment import ConflictPolicy, GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import (
    CommunicationEvaluationMetrics,
    compare_communication_modes,
    evaluate_communication_q_learning,
    evaluate_independent_marl,
    evaluate_rule_based_communication,
    format_communication_table,
    summarize_communication_metrics,
)


class CommunicationEvaluationMetricsTests(unittest.TestCase):
    def test_metrics_properties_are_computed(self) -> None:
        metrics = CommunicationEvaluationMetrics(
            mode_name="test",
            total_episodes=4,
            success_count=3,
            timeout_count=1,
            blocked_move_count=2,
            total_reward=12.0,
            total_steps=10,
            communication_usage_count=8,
            predicted_conflict_count=2,
            waiting_count=1,
        )

        self.assertEqual(metrics.failure_count, 1)
        self.assertEqual(metrics.success_rate, 0.75)
        self.assertEqual(metrics.timeout_frequency, 0.25)
        self.assertEqual(metrics.average_reward, 3.0)
        self.assertEqual(metrics.average_episode_length, 2.5)
        self.assertEqual(metrics.predicted_conflict_frequency, 0.25)
        self.assertEqual(metrics.waiting_frequency, 0.125)

    def test_no_communication_frequencies_are_zero(self) -> None:
        metrics = CommunicationEvaluationMetrics(
            mode_name="independent",
            total_episodes=1,
            success_count=1,
            timeout_count=0,
            blocked_move_count=0,
            total_reward=1.0,
            total_steps=1,
        )

        self.assertEqual(metrics.predicted_conflict_frequency, 0.0)
        self.assertEqual(metrics.waiting_frequency, 0.0)


class CommunicationEvaluationTests(unittest.TestCase):
    def test_independent_evaluation_reports_no_communication_counts(self) -> None:
        metrics = evaluate_independent_marl(_parallel_env, episodes=3)

        self.assertEqual(metrics.mode_name, "independent")
        self.assertEqual(metrics.total_episodes, 3)
        self.assertEqual(metrics.communication_usage_count, 0)
        self.assertEqual(metrics.predicted_conflict_count, 0)
        self.assertEqual(metrics.waiting_count, 0)

    def test_rule_based_evaluation_counts_messages(self) -> None:
        metrics = evaluate_rule_based_communication(_same_target_env, episodes=2)

        self.assertEqual(metrics.mode_name, "rule-based communication")
        self.assertEqual(metrics.total_episodes, 2)
        self.assertGreater(metrics.communication_usage_count, 0)
        self.assertGreater(metrics.predicted_conflict_count, 0)
        self.assertGreaterEqual(metrics.waiting_count, 0)
        self.assertEqual(metrics.blocked_move_count, 0)

    def test_communication_q_learning_evaluation_is_stable(self) -> None:
        metrics = evaluate_communication_q_learning(_parallel_env, episodes=3)

        self.assertEqual(metrics.mode_name, "communication q-learning")
        self.assertEqual(metrics.total_episodes, 3)
        self.assertGreater(metrics.communication_usage_count, 0)
        self.assertGreaterEqual(metrics.blocked_move_count, 0)
        self.assertGreaterEqual(metrics.average_episode_length, 1.0)

    def test_compare_communication_modes_returns_all_modes(self) -> None:
        results = compare_communication_modes(_parallel_env, episodes=2)

        self.assertEqual(
            tuple(result.mode_name for result in results),
            ("independent", "rule-based communication", "communication q-learning"),
        )

    def test_format_communication_table_includes_expected_columns(self) -> None:
        results = (
            CommunicationEvaluationMetrics(
                mode_name="independent",
                total_episodes=2,
                success_count=1,
                timeout_count=1,
                blocked_move_count=3,
                total_reward=4.0,
                total_steps=5,
            ),
            CommunicationEvaluationMetrics(
                mode_name="communication",
                total_episodes=2,
                success_count=2,
                timeout_count=0,
                blocked_move_count=0,
                total_reward=8.0,
                total_steps=4,
                communication_usage_count=8,
                predicted_conflict_count=2,
                waiting_count=1,
            ),
        )

        table = format_communication_table(results)

        self.assertIn("mode", table)
        self.assertIn("messages", table)
        self.assertIn("conflict freq", table)
        self.assertIn("independent", table)
        self.assertIn("communication", table)

    def test_summary_formats_metrics(self) -> None:
        metrics = CommunicationEvaluationMetrics(
            mode_name="communication",
            total_episodes=2,
            success_count=2,
            timeout_count=0,
            blocked_move_count=0,
            total_reward=8.0,
            total_steps=4,
            communication_usage_count=8,
            predicted_conflict_count=2,
            waiting_count=1,
        )

        summary = summarize_communication_metrics(metrics)

        self.assertIn("communication Communication Evaluation", summary)
        self.assertIn("Success rate: 1.00", summary)
        self.assertIn("Communication messages: 8", summary)


def _parallel_env() -> GridWorldEnv:
    return GridWorldEnv(
        GridWorldConfig(
            width=3,
            height=2,
            agents=(
                AgentState("agent_1", Position(0, 0), Position(2, 0)),
                AgentState("agent_2", Position(0, 1), Position(2, 1)),
            ),
            max_steps=5,
            conflict_policy=ConflictPolicy.BLOCK_ALL,
        )
    )


def _same_target_env() -> GridWorldEnv:
    return GridWorldEnv(
        GridWorldConfig(
            width=3,
            height=3,
            agents=(
                AgentState("agent_1", Position(0, 1), Position(1, 1)),
                AgentState("agent_2", Position(2, 1), Position(1, 1)),
            ),
            max_steps=4,
            conflict_policy=ConflictPolicy.BLOCK_ALL,
        )
    )


if __name__ == "__main__":
    unittest.main()
