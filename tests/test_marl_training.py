from __future__ import annotations

import unittest

from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import (
    MultiAgentRLEnv,
    QLearningAgent,
    summarize_marl_training,
    train_independent_q_learning,
)


class IndependentMARLTrainingTests(unittest.TestCase):
    def test_training_loop_runs_and_returns_metrics(self) -> None:
        env = self._env()
        agents = self._agents(env)

        metrics = train_independent_q_learning(env=env, agents=agents, episodes=5)

        self.assertEqual(metrics.total_episodes, 5)
        self.assertEqual(metrics.success_count + metrics.failure_count, 5)
        self.assertEqual(len(metrics.episode_rewards), 5)
        self.assertEqual(len(metrics.episode_lengths), 5)
        self.assertEqual(len(metrics.success_history), 5)
        self.assertEqual(len(metrics.timeout_history), 5)
        self.assertEqual(set(metrics.per_agent_rewards), {"agent_1", "agent_2"})
        self.assertEqual(len(metrics.per_agent_rewards["agent_1"]), 5)
        self.assertGreaterEqual(metrics.blocked_move_count, 0)

    def test_q_tables_update_independently(self) -> None:
        env = self._env()
        agents = self._agents(env, epsilon=0.0)

        train_independent_q_learning(env=env, agents=agents, episodes=3)

        self.assertTrue(agents["agent_1"].q_table)
        self.assertTrue(agents["agent_2"].q_table)
        self.assertIsNot(agents["agent_1"].q_table, agents["agent_2"].q_table)
        self.assertTrue(
            any(value != 0.0 for values in agents["agent_1"].q_table.values() for value in values)
        )
        self.assertTrue(
            any(value != 0.0 for values in agents["agent_2"].q_table.values() for value in values)
        )

    def test_simultaneous_training_step_can_succeed(self) -> None:
        env = self._env()
        agents = {
            agent_id: QLearningAgent(action_size=env.action_size, epsilon=0.0)
            for agent_id in env.controlled_agent_ids
        }
        for agent in agents.values():
            agent.update((0, 0, 2, 0), 3, reward=1.0, next_state=(1, 0, 2, 0), done=True)
            agent.update((0, 1, 2, 1), 3, reward=1.0, next_state=(1, 1, 2, 1), done=True)
            agent.update((1, 0, 2, 0), 3, reward=1.0, next_state=(2, 0, 2, 0), done=True)
            agent.update((1, 1, 2, 1), 3, reward=1.0, next_state=(2, 1, 2, 1), done=True)

        metrics = train_independent_q_learning(env=env, agents=agents, episodes=1)

        self.assertEqual(metrics.success_count, 1)
        self.assertEqual(metrics.timeout_count, 0)
        self.assertEqual(metrics.episode_lengths, (2,))

    def test_summary_formats_metrics(self) -> None:
        env = self._env()
        metrics = train_independent_q_learning(env=env, agents=self._agents(env), episodes=2)

        summary = summarize_marl_training(metrics)

        self.assertIn("Independent Q-Learning MARL Summary", summary)
        self.assertIn("Success rate:", summary)
        self.assertIn("Blocked moves:", summary)

    def test_missing_agent_mapping_is_rejected(self) -> None:
        env = self._env()

        with self.assertRaisesRegex(ValueError, "Missing Q-learning agents"):
            train_independent_q_learning(
                env=env,
                agents={"agent_1": QLearningAgent(action_size=env.action_size)},
                episodes=1,
            )

    @staticmethod
    def _env() -> MultiAgentRLEnv:
        env = GridWorldEnv(
            GridWorldConfig(
                width=3,
                height=2,
                agents=(
                    AgentState("agent_1", Position(0, 0), Position(2, 0)),
                    AgentState("agent_2", Position(0, 1), Position(2, 1)),
                ),
                max_steps=5,
            )
        )
        return MultiAgentRLEnv(env, ("agent_1", "agent_2"))

    @staticmethod
    def _agents(env: MultiAgentRLEnv, epsilon: float = 0.2) -> dict[str, QLearningAgent]:
        return {
            agent_id: QLearningAgent(action_size=env.action_size, epsilon=epsilon, seed=index)
            for index, agent_id in enumerate(env.controlled_agent_ids, start=1)
        }


if __name__ == "__main__":
    unittest.main()
