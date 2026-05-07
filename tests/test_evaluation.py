from __future__ import annotations

import unittest

from marlsim.agents import GreedyGoalAgent, ShortestPathAgent
from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import QLearningAgent, SingleAgentRLEnv, evaluate_policy, format_evaluation_table


class EvaluationTests(unittest.TestCase):
    def test_evaluation_loop_runs_and_computes_metrics(self) -> None:
        metrics = evaluate_policy(
            env=self._eval_env(),
            policy=ShortestPathAgent(),
            episodes=3,
            policy_name="shortest_path",
        )

        self.assertEqual(metrics.policy_name, "shortest_path")
        self.assertEqual(metrics.total_episodes, 3)
        self.assertEqual(metrics.success_count, 3)
        self.assertEqual(metrics.failure_count, 0)
        self.assertEqual(metrics.timeout_count, 0)
        self.assertEqual(metrics.success_rate, 1.0)
        self.assertEqual(metrics.timeout_frequency, 0.0)
        self.assertEqual(metrics.average_episode_length, 2.0)
        self.assertAlmostEqual(metrics.average_reward, 9.8)

    def test_evaluation_table_formats_metrics(self) -> None:
        metrics = evaluate_policy(
            env=self._eval_env(),
            policy=GreedyGoalAgent(),
            episodes=1,
            policy_name="greedy",
        )

        table = format_evaluation_table([metrics])

        self.assertIn("policy", table)
        self.assertIn("success", table)
        self.assertIn("greedy", table)
        self.assertIn("1.00", table)

    def test_trained_agent_evaluation_disables_exploration_and_restores_epsilon(self) -> None:
        q_agent = QLearningAgent(action_size=5, epsilon=1.0, seed=1)
        q_agent.update((0, 0, 2, 0), 3, reward=1.0, next_state=(1, 0, 2, 0), done=True)
        q_agent.update((1, 0, 2, 0), 3, reward=1.0, next_state=(2, 0, 2, 0), done=True)

        metrics = evaluate_policy(
            env=self._eval_env(),
            policy=q_agent,
            episodes=3,
            policy_name="q_learning",
        )

        self.assertEqual(q_agent.epsilon, 1.0)
        self.assertEqual(metrics.success_count, 3)
        self.assertEqual(metrics.timeout_count, 0)
        self.assertEqual(metrics.average_episode_length, 2.0)

    def test_evaluation_rejects_non_positive_episode_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "episodes must be positive"):
            evaluate_policy(
                env=self._eval_env(),
                policy=ShortestPathAgent(),
                episodes=0,
                policy_name="shortest_path",
            )

    @staticmethod
    def _eval_env() -> SingleAgentRLEnv:
        env = GridWorldEnv(
            GridWorldConfig(
                width=3,
                height=1,
                agents=(AgentState("agent_1", Position(0, 0), Position(2, 0)),),
                max_steps=5,
            )
        )
        return SingleAgentRLEnv(env)


if __name__ == "__main__":
    unittest.main()
