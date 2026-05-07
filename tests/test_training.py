from __future__ import annotations

import unittest

from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import QLearningAgent, SingleAgentRLEnv, train_q_learning


class QLearningTrainingTests(unittest.TestCase):
    def test_training_loop_runs_and_returns_metrics(self) -> None:
        env = self._training_env()
        agent = QLearningAgent(action_size=env.action_size, epsilon=0.2, seed=1)

        metrics = train_q_learning(env=env, agent=agent, episodes=5)

        self.assertEqual(metrics.total_episodes, 5)
        self.assertEqual(metrics.success_count + metrics.failure_count, 5)
        self.assertGreaterEqual(metrics.timeout_count, 0)
        self.assertEqual(len(metrics.episode_rewards), 5)
        self.assertEqual(len(metrics.episode_lengths), 5)
        self.assertEqual(metrics.total_steps, sum(metrics.episode_lengths))
        self.assertAlmostEqual(metrics.total_reward, sum(metrics.episode_rewards))

    def test_q_table_changes_during_training(self) -> None:
        env = self._training_env()
        agent = QLearningAgent(action_size=env.action_size, epsilon=0.0)

        self.assertEqual(agent.q_table, {})

        train_q_learning(env=env, agent=agent, episodes=3)

        self.assertTrue(agent.q_table)
        self.assertTrue(
            any(value != 0.0 for values in agent.q_table.values() for value in values)
        )

    def test_training_metrics_averages_are_computed(self) -> None:
        env = self._training_env()
        agent = QLearningAgent(action_size=env.action_size, epsilon=0.2, seed=2)

        metrics = train_q_learning(env=env, agent=agent, episodes=4)

        self.assertAlmostEqual(metrics.average_reward, metrics.total_reward / 4)
        self.assertAlmostEqual(metrics.average_episode_length, metrics.total_steps / 4)

    def test_training_rejects_non_positive_episode_count(self) -> None:
        env = self._training_env()
        agent = QLearningAgent(action_size=env.action_size)

        with self.assertRaisesRegex(ValueError, "episodes must be positive"):
            train_q_learning(env=env, agent=agent, episodes=0)

    @staticmethod
    def _training_env() -> SingleAgentRLEnv:
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
