from __future__ import annotations

import unittest

from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import (
    MARLTrajectoryStep,
    MultiAgentRLEnv,
    QLearningAgent,
    format_marl_trajectory,
    rollout_marl_policy,
    summarize_marl_outcome,
)
from marlsim.demos.marl_visualization_demo import build_demo_output


class MARLVisualizationTests(unittest.TestCase):
    def test_trajectory_format_includes_step_actions_rewards_and_done(self) -> None:
        text = format_marl_trajectory(
            (
                MARLTrajectoryStep(
                    step_number=1,
                    positions={"agent_1": Position(1, 0)},
                    actions={"agent_1": "right"},
                    rewards={"agent_1": -0.1},
                    blocked={"agent_1": False},
                    reached_goals={"agent_1": False},
                    done=False,
                ),
            )
        )

        self.assertIn("step 1", text)
        self.assertIn("pos=(1,0)", text)
        self.assertIn("action=right", text)
        self.assertIn("reward=-0.10", text)
        self.assertIn("done=False", text)

    def test_rollout_produces_expected_success_trajectory(self) -> None:
        env = self._env()
        agents = self._right_moving_agents(env)

        trajectory = rollout_marl_policy(env=env, policies=agents)

        self.assertEqual(len(trajectory), 2)
        self.assertTrue(trajectory[-1].done)
        self.assertEqual(trajectory[-1].positions["agent_1"], Position(2, 0))
        self.assertEqual(trajectory[-1].positions["agent_2"], Position(2, 1))
        self.assertEqual(trajectory[-1].actions["agent_1"], "right")
        self.assertTrue(trajectory[-1].reached_goals["agent_1"])
        self.assertTrue(trajectory[-1].reached_goals["agent_2"])

    def test_outcome_summary_marks_success_and_failure(self) -> None:
        success = (
            MARLTrajectoryStep(
                step_number=1,
                positions={"agent_1": Position(2, 0)},
                actions={"agent_1": "right"},
                rewards={"agent_1": 9.9},
                blocked={"agent_1": False},
                reached_goals={"agent_1": True},
                done=True,
            ),
        )
        failure = (
            MARLTrajectoryStep(
                step_number=1,
                positions={"agent_1": Position(0, 0)},
                actions={"agent_1": "up"},
                rewards={"agent_1": -5.1},
                blocked={"agent_1": True},
                reached_goals={"agent_1": False},
                done=True,
            ),
        )

        self.assertEqual(summarize_marl_outcome("case", success), "case: success in 1 steps")
        self.assertEqual(summarize_marl_outcome("case", failure), "case: failure in 1 steps")

    def test_demo_output_explains_individual_and_shared_reward_results(self) -> None:
        output = build_demo_output()

        self.assertIn("individual reward", output)
        self.assertIn("shared reward", output)
        self.assertIn("success", output)
        self.assertIn("failure", output)
        self.assertIn("credit-assignment", output)

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
    def _right_moving_agents(env: MultiAgentRLEnv) -> dict[str, QLearningAgent]:
        agents = {
            agent_id: QLearningAgent(action_size=env.action_size, epsilon=0.8)
            for agent_id in env.controlled_agent_ids
        }
        agents["agent_1"].q_table[(0, 0, 2, 0)] = [0.0, 0.0, 0.0, 1.0, 0.0]
        agents["agent_1"].q_table[(1, 0, 2, 0)] = [0.0, 0.0, 0.0, 1.0, 0.0]
        agents["agent_2"].q_table[(0, 1, 2, 1)] = [0.0, 0.0, 0.0, 1.0, 0.0]
        agents["agent_2"].q_table[(1, 1, 2, 1)] = [0.0, 0.0, 0.0, 1.0, 0.0]
        return agents


if __name__ == "__main__":
    unittest.main()
