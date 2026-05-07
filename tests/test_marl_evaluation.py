from __future__ import annotations

import unittest

from marlsim.agents import ShortestPathAgent
from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import (
    MultiAgentRLEnv,
    QLearningAgent,
    evaluate_marl_policy,
    format_marl_evaluation_table,
)


class MARLEvaluationTests(unittest.TestCase):
    def test_marl_evaluation_runs_and_computes_metrics(self) -> None:
        env = self._env()

        metrics = evaluate_marl_policy(
            env=env,
            policies={
                "agent_1": ShortestPathAgent(),
                "agent_2": ShortestPathAgent(),
            },
            episodes=3,
            policy_name="shortest-path",
        )

        self.assertEqual(metrics.policy_name, "shortest-path")
        self.assertEqual(metrics.total_episodes, 3)
        self.assertEqual(metrics.success_count, 3)
        self.assertEqual(metrics.timeout_count, 0)
        self.assertEqual(metrics.blocked_move_count, 0)
        self.assertEqual(metrics.episode_lengths, (2, 2, 2))
        self.assertEqual(metrics.success_rate, 1.0)
        self.assertEqual(metrics.timeout_frequency, 0.0)
        self.assertAlmostEqual(metrics.average_reward, 19.6)

    def test_q_tables_are_not_updated_during_evaluation(self) -> None:
        env = self._env()
        agents = self._trained_right_moving_agents(env)
        before = {
            agent_id: {state: values[:] for state, values in agent.q_table.items()}
            for agent_id, agent in agents.items()
        }

        evaluate_marl_policy(
            env=env,
            policies=agents,
            episodes=2,
            policy_name="q-learning",
        )

        after = {
            agent_id: {state: values[:] for state, values in agent.q_table.items()}
            for agent_id, agent in agents.items()
        }
        self.assertEqual(after, before)

    def test_epsilon_is_restored_after_q_learning_evaluation(self) -> None:
        env = self._env()
        agents = self._trained_right_moving_agents(env)
        agents["agent_1"].epsilon = 0.7
        agents["agent_2"].epsilon = 0.4

        evaluate_marl_policy(
            env=env,
            policies=agents,
            episodes=1,
            policy_name="q-learning",
        )

        self.assertEqual(agents["agent_1"].epsilon, 0.7)
        self.assertEqual(agents["agent_2"].epsilon, 0.4)

    def test_metrics_track_timeout_and_blocked_moves(self) -> None:
        env = MultiAgentRLEnv(
            GridWorldEnv(
                GridWorldConfig(
                    width=3,
                    height=2,
                    agents=(
                        AgentState("agent_1", Position(0, 0), Position(2, 0)),
                        AgentState("agent_2", Position(0, 1), Position(2, 1)),
                    ),
                    max_steps=1,
                )
            ),
            ("agent_1", "agent_2"),
        )

        metrics = evaluate_marl_policy(
            env=env,
            policies={
                "agent_1": QLearningAgent(action_size=env.action_size, epsilon=0.0),
                "agent_2": QLearningAgent(action_size=env.action_size, epsilon=0.0),
            },
            episodes=2,
            policy_name="unseen-q",
        )

        self.assertEqual(metrics.success_count, 0)
        self.assertEqual(metrics.timeout_count, 2)
        self.assertEqual(metrics.blocked_move_count, 4)
        self.assertEqual(metrics.episode_lengths, (1, 1))
        self.assertEqual(metrics.final_positions[0]["agent_1"], Position(0, 0))

    def test_format_marl_evaluation_table(self) -> None:
        metrics = evaluate_marl_policy(
            env=self._env(),
            policies={
                "agent_1": ShortestPathAgent(),
                "agent_2": ShortestPathAgent(),
            },
            episodes=1,
            policy_name="shortest-path",
        )

        table = format_marl_evaluation_table([metrics])

        self.assertIn("policy", table)
        self.assertIn("success", table)
        self.assertIn("blocked", table)
        self.assertIn("shortest-path", table)

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
    def _trained_right_moving_agents(env: MultiAgentRLEnv) -> dict[str, QLearningAgent]:
        agents = {
            agent_id: QLearningAgent(action_size=env.action_size, epsilon=0.3)
            for agent_id in env.controlled_agent_ids
        }
        agents["agent_1"].q_table[(0, 0, 2, 0)] = [0.0, 0.0, 0.0, 1.0, 0.0]
        agents["agent_1"].q_table[(1, 0, 2, 0)] = [0.0, 0.0, 0.0, 1.0, 0.0]
        agents["agent_2"].q_table[(0, 1, 2, 1)] = [0.0, 0.0, 0.0, 1.0, 0.0]
        agents["agent_2"].q_table[(1, 1, 2, 1)] = [0.0, 0.0, 0.0, 1.0, 0.0]
        return agents


if __name__ == "__main__":
    unittest.main()
