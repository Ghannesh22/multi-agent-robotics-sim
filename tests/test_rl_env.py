from __future__ import annotations

import unittest

from marlsim.core.actions import Action
from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import SingleAgentRLEnv


class SingleAgentRLEnvTests(unittest.TestCase):
    def test_reset_returns_valid_observation(self) -> None:
        rl_env = SingleAgentRLEnv(self._single_agent_env())

        observation = rl_env.reset()

        self.assertEqual(observation, (0, 0, 2, 0))
        self.assertEqual(rl_env.step_count, 0)
        self.assertFalse(rl_env.done)
        self.assertFalse(rl_env.episode_done)
        self.assertFalse(rl_env.reached_goal)
        self.assertFalse(rl_env.timeout)

    def test_observation_contract_is_named_and_sized(self) -> None:
        rl_env = SingleAgentRLEnv(self._single_agent_env())

        self.assertEqual(
            rl_env.observation_names,
            ("agent_x", "agent_y", "goal_x", "goal_y"),
        )
        self.assertEqual(rl_env.observation_size, 4)
        self.assertEqual(len(rl_env.get_observation()), rl_env.observation_size)

    def test_observation_value_order_is_explicit(self) -> None:
        rl_env = SingleAgentRLEnv(
            GridWorldEnv(
                GridWorldConfig(
                    width=5,
                    height=5,
                    agents=(AgentState("agent_1", Position(1, 2), Position(4, 3)),),
                )
            )
        )

        self.assertEqual(rl_env.get_observation(), (1, 2, 4, 3))

    def test_observation_updates_after_movement(self) -> None:
        rl_env = SingleAgentRLEnv(self._single_agent_env())

        self.assertEqual(rl_env.reset(), (0, 0, 2, 0))
        observation, _, _, _ = rl_env.step(3)

        self.assertEqual(observation, (1, 0, 2, 0))
        self.assertEqual(rl_env.get_observation(), (1, 0, 2, 0))

    def test_reset_restores_initial_observation_after_movement(self) -> None:
        rl_env = SingleAgentRLEnv(self._single_agent_env())

        rl_env.reset()
        rl_env.step(3)

        self.assertEqual(rl_env.reset(), (0, 0, 2, 0))

    def test_step_returns_observation_reward_done_and_info(self) -> None:
        rl_env = SingleAgentRLEnv(self._single_agent_env())
        rl_env.reset()

        observation, reward, done, info = rl_env.step(3)

        self.assertEqual(observation, (1, 0, 2, 0))
        self.assertEqual(reward, -0.1)
        self.assertFalse(done)
        self.assertEqual(info["controlled_agent_id"], "agent_1")
        self.assertEqual(info["action"], "right")
        self.assertEqual(info["step_count"], 1)
        self.assertFalse(info["reached_goal"])
        self.assertFalse(info["blocked"])
        self.assertIsNone(info["blocked_reason"])
        self.assertFalse(info["timeout"])
        self.assertFalse(info["done"])

    def test_info_dictionary_contains_expected_keys(self) -> None:
        rl_env = SingleAgentRLEnv(self._single_agent_env())
        rl_env.reset()

        _, _, _, info = rl_env.step(3)

        self.assertEqual(
            set(info),
            {
                "controlled_agent_id",
                "action",
                "step_count",
                "reached_goal",
                "blocked",
                "blocked_reason",
                "timeout",
                "done",
            },
        )

    def test_action_contract_is_named_and_sized(self) -> None:
        rl_env = SingleAgentRLEnv(self._single_agent_env())

        self.assertEqual(rl_env.action_names, ("up", "down", "left", "right", "stay"))
        self.assertEqual(rl_env.action_size, 5)

    def test_integer_actions_map_to_existing_action_enum(self) -> None:
        self.assertEqual(SingleAgentRLEnv.action_to_env_action(0), Action.UP)
        self.assertEqual(SingleAgentRLEnv.action_to_env_action(1), Action.DOWN)
        self.assertEqual(SingleAgentRLEnv.action_to_env_action(2), Action.LEFT)
        self.assertEqual(SingleAgentRLEnv.action_to_env_action(3), Action.RIGHT)
        self.assertEqual(SingleAgentRLEnv.action_to_env_action(4), Action.STAY)
        self.assertEqual(SingleAgentRLEnv.action_from_int(3), Action.RIGHT)

    def test_invalid_integer_action_below_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "0 to 4"):
            SingleAgentRLEnv.action_to_env_action(-1)

    def test_invalid_integer_action_above_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "0 to 4"):
            SingleAgentRLEnv.action_to_env_action(5)

    def test_step_uses_the_correct_mapped_action(self) -> None:
        rl_env = SingleAgentRLEnv(self._single_agent_env())
        rl_env.reset()

        observation, _, _, info = rl_env.step(1)

        self.assertEqual(info["action"], "down")
        self.assertEqual(observation, (0, 1, 2, 0))

    def test_reward_values_are_stable_and_documented(self) -> None:
        rl_env = SingleAgentRLEnv(self._single_agent_env())

        self.assertEqual(rl_env.step_penalty, -0.1)
        self.assertEqual(rl_env.goal_reward, 10.0)
        self.assertEqual(rl_env.blocked_penalty, -1.0)
        self.assertEqual(rl_env.timeout_penalty, -5.0)
        self.assertEqual(
            rl_env.reward_values,
            {
                "step_penalty": -0.1,
                "goal_reward": 10.0,
                "blocked_penalty": -1.0,
                "timeout_penalty": -5.0,
            },
        )

    def test_normal_step_reward_is_step_penalty_only(self) -> None:
        rl_env = SingleAgentRLEnv(self._single_agent_env())
        rl_env.reset()

        _, reward, done, info = rl_env.step(3)

        self.assertEqual(reward, -0.1)
        self.assertFalse(done)
        self.assertFalse(info["blocked"])
        self.assertFalse(info["reached_goal"])
        self.assertFalse(info["timeout"])

    def test_reaching_goal_ends_episode_with_positive_reward(self) -> None:
        rl_env = SingleAgentRLEnv(self._single_agent_env())
        rl_env.reset()

        first_observation, first_reward, first_done, _ = rl_env.step(3)
        final_observation, final_reward, final_done, info = rl_env.step(3)

        self.assertEqual(first_observation, (1, 0, 2, 0))
        self.assertEqual(first_reward, -0.1)
        self.assertFalse(first_done)
        self.assertEqual(final_observation, (2, 0, 2, 0))
        self.assertEqual(final_reward, 9.9)
        self.assertTrue(final_done)
        self.assertTrue(info["reached_goal"])
        self.assertTrue(info["done"])
        self.assertTrue(rl_env.done)
        self.assertTrue(rl_env.episode_done)
        self.assertTrue(rl_env.reached_goal)
        self.assertFalse(rl_env.timeout)

    def test_blocked_move_gives_penalty(self) -> None:
        rl_env = SingleAgentRLEnv(self._single_agent_env())
        rl_env.reset()

        observation, reward, done, info = rl_env.step(2)

        self.assertEqual(observation, (0, 0, 2, 0))
        self.assertEqual(reward, -1.1)
        self.assertFalse(done)
        self.assertTrue(info["blocked"])
        self.assertEqual(info["blocked_reason"], "out_of_bounds")

    def test_timeout_failure_gives_timeout_penalty(self) -> None:
        rl_env = SingleAgentRLEnv(
            GridWorldEnv(
                GridWorldConfig(
                    width=3,
                    height=1,
                    agents=(AgentState("agent_1", Position(0, 0), Position(2, 0)),),
                    max_steps=1,
                )
            )
        )
        rl_env.reset()

        observation, reward, done, info = rl_env.step(4)

        self.assertEqual(observation, (0, 0, 2, 0))
        self.assertEqual(reward, -5.1)
        self.assertTrue(done)
        self.assertFalse(info["reached_goal"])
        self.assertTrue(info["timeout"])
        self.assertTrue(info["done"])
        self.assertTrue(rl_env.done)
        self.assertTrue(rl_env.episode_done)
        self.assertFalse(rl_env.reached_goal)
        self.assertTrue(rl_env.timeout)

    def test_step_after_done_raises_clear_error(self) -> None:
        rl_env = SingleAgentRLEnv(self._single_agent_env())
        rl_env.reset()
        rl_env.step(3)
        rl_env.step(3)

        with self.assertRaisesRegex(RuntimeError, "Call reset"):
            rl_env.step(4)

    def test_reset_clears_episode_state_after_done(self) -> None:
        rl_env = SingleAgentRLEnv(self._single_agent_env())
        rl_env.reset()
        rl_env.step(3)
        rl_env.step(3)

        observation = rl_env.reset()

        self.assertEqual(observation, (0, 0, 2, 0))
        self.assertEqual(rl_env.step_count, 0)
        self.assertEqual(rl_env.env.step_count, 0)
        self.assertFalse(rl_env.done)
        self.assertFalse(rl_env.episode_done)
        self.assertFalse(rl_env.reached_goal)
        self.assertFalse(rl_env.timeout)

    def test_reset_after_done_allows_new_episode(self) -> None:
        rl_env = SingleAgentRLEnv(self._single_agent_env())
        rl_env.reset()
        rl_env.step(3)
        rl_env.step(3)

        rl_env.reset()
        observation, reward, done, info = rl_env.step(3)

        self.assertEqual(observation, (1, 0, 2, 0))
        self.assertEqual(reward, -0.1)
        self.assertFalse(done)
        self.assertFalse(info["done"])

    def test_other_agents_stay_for_now(self) -> None:
        env = GridWorldEnv(
            GridWorldConfig(
                width=4,
                height=2,
                agents=(
                    AgentState("agent_1", Position(0, 0), Position(2, 0)),
                    AgentState("agent_2", Position(3, 1), Position(0, 1)),
                ),
            )
        )
        rl_env = SingleAgentRLEnv(env)

        rl_env.reset()
        rl_env.step(3)

        self.assertEqual(env.agents["agent_1"].position, Position(1, 0))
        self.assertEqual(env.agents["agent_2"].position, Position(3, 1))

    @staticmethod
    def _single_agent_env() -> GridWorldEnv:
        return GridWorldEnv(
            GridWorldConfig(
                width=3,
                height=2,
                agents=(AgentState("agent_1", Position(0, 0), Position(2, 0)),),
                max_steps=5,
            )
        )


if __name__ == "__main__":
    unittest.main()
