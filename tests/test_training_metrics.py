from __future__ import annotations

import io
import unittest

from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import (
    QLearningAgent,
    SingleAgentRLEnv,
    export_training_metrics,
    reward_chart,
    success_rate_chart,
    summarize_training,
    train_q_learning,
)


class TrainingMetricsReportingTests(unittest.TestCase):
    def test_metrics_history_lengths_are_correct(self) -> None:
        metrics = self._metrics(episodes=6)

        self.assertEqual(len(metrics.episode_rewards), 6)
        self.assertEqual(len(metrics.episode_lengths), 6)
        self.assertEqual(len(metrics.success_history), 6)
        self.assertEqual(len(metrics.timeout_history), 6)

    def test_summary_computes_rates_and_averages(self) -> None:
        metrics = self._metrics(episodes=5)

        summary = summarize_training(metrics)

        self.assertIn("Q-Learning Training Summary", summary)
        self.assertIn(f"Success rate: {metrics.success_rate:.2f}", summary)
        self.assertIn(f"Timeout frequency: {metrics.timeout_frequency:.2f}", summary)
        self.assertIn(f"Average reward: {metrics.average_reward:.2f}", summary)

    def test_export_training_metrics_writes_csv(self) -> None:
        metrics = self._metrics(episodes=3)
        output = io.StringIO()

        export_training_metrics(metrics, output)

        lines = output.getvalue().splitlines()
        self.assertEqual(lines[0], "episode,reward,length,success,timeout")
        self.assertEqual(len(lines), 4)

    def test_reward_chart_returns_text_chart(self) -> None:
        metrics = self._metrics(episodes=3)

        chart = reward_chart(metrics, width=10)

        self.assertIn("Episode rewards", chart)
        self.assertIn("001 |", chart)

    def test_success_rate_chart_returns_text_chart(self) -> None:
        metrics = self._metrics(episodes=3)

        chart = success_rate_chart(metrics, window=2, width=10)

        self.assertIn("Rolling success rate", chart)
        self.assertIn("001 |", chart)

    @staticmethod
    def _metrics(episodes: int):
        env = GridWorldEnv(
            GridWorldConfig(
                width=3,
                height=1,
                agents=(AgentState("agent_1", Position(0, 0), Position(2, 0)),),
                max_steps=5,
            )
        )
        rl_env = SingleAgentRLEnv(env)
        agent = QLearningAgent(action_size=rl_env.action_size, epsilon=0.2, seed=7)
        return train_q_learning(env=rl_env, agent=agent, episodes=episodes)


if __name__ == "__main__":
    unittest.main()
