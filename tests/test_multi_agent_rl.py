from __future__ import annotations

import unittest

from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import MultiAgentRLEnv


class MultiAgentRLEnvTests(unittest.TestCase):
    def test_reset_returns_multi_agent_observations(self) -> None:
        env = MultiAgentRLEnv(self._two_agent_env(), ("agent_1", "agent_2"))

        observations = env.reset()

        self.assertEqual(
            observations,
            {
                "agent_1": (0, 0, 2, 0),
                "agent_2": (0, 1, 2, 1),
            },
        )
        self.assertEqual(env.step_count, 0)
        self.assertFalse(env.done)
        self.assertFalse(env.timeout)

    def test_simultaneous_actions_update_controlled_agents(self) -> None:
        env = MultiAgentRLEnv(self._two_agent_env(), ("agent_1", "agent_2"))
        env.reset()

        observations, rewards, done, info = env.step({"agent_1": 3, "agent_2": 3})

        self.assertEqual(
            observations,
            {
                "agent_1": (1, 0, 2, 0),
                "agent_2": (1, 1, 2, 1),
            },
        )
        self.assertEqual(rewards, {"agent_1": -0.1, "agent_2": -0.1})
        self.assertFalse(done)
        self.assertFalse(info["done"])
        self.assertEqual(info["step_count"], 1)
        self.assertFalse(info["agents"]["agent_1"]["blocked"])
        self.assertFalse(info["agents"]["agent_2"]["blocked"])

    def test_rewards_returned_per_agent_for_blocked_move(self) -> None:
        env = MultiAgentRLEnv(self._two_agent_env(), ("agent_1", "agent_2"))
        env.reset()

        _, rewards, done, info = env.step({"agent_1": 2, "agent_2": 3})

        self.assertEqual(rewards["agent_1"], -1.1)
        self.assertEqual(rewards["agent_2"], -0.1)
        self.assertFalse(done)
        self.assertTrue(info["agents"]["agent_1"]["blocked"])
        self.assertEqual(info["agents"]["agent_1"]["blocked_reason"], "out_of_bounds")

    def test_done_behavior_when_all_controlled_agents_reach_goals(self) -> None:
        env = MultiAgentRLEnv(self._two_agent_env(), ("agent_1", "agent_2"))
        env.reset()
        env.step({"agent_1": 3, "agent_2": 3})

        observations, rewards, done, info = env.step({"agent_1": 3, "agent_2": 3})

        self.assertEqual(observations["agent_1"], (2, 0, 2, 0))
        self.assertEqual(observations["agent_2"], (2, 1, 2, 1))
        self.assertEqual(rewards, {"agent_1": 9.9, "agent_2": 9.9})
        self.assertTrue(done)
        self.assertTrue(info["done"])
        self.assertTrue(info["agents"]["agent_1"]["reached_goal"])
        self.assertTrue(info["agents"]["agent_2"]["reached_goal"])

    def test_timeout_done_behavior(self) -> None:
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
        env.reset()

        _, rewards, done, info = env.step({"agent_1": 4, "agent_2": 4})

        self.assertEqual(rewards, {"agent_1": -5.1, "agent_2": -5.1})
        self.assertTrue(done)
        self.assertTrue(info["timeout"])
        self.assertTrue(info["agents"]["agent_1"]["timeout"])
        self.assertTrue(info["agents"]["agent_2"]["timeout"])

    def test_invalid_actions_are_rejected(self) -> None:
        env = MultiAgentRLEnv(self._two_agent_env(), ("agent_1", "agent_2"))
        env.reset()

        with self.assertRaisesRegex(ValueError, "0 to 4"):
            env.step({"agent_1": 5, "agent_2": 3})

    def test_missing_actions_are_rejected(self) -> None:
        env = MultiAgentRLEnv(self._two_agent_env(), ("agent_1", "agent_2"))
        env.reset()

        with self.assertRaisesRegex(ValueError, "Missing actions"):
            env.step({"agent_1": 3})

    @staticmethod
    def _two_agent_env() -> GridWorldEnv:
        return GridWorldEnv(
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


if __name__ == "__main__":
    unittest.main()
