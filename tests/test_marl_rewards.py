from __future__ import annotations

import unittest

from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import (
    MARLRewardMode,
    MultiAgentRLEnv,
    QLearningAgent,
    apply_reward_mode,
    train_independent_q_learning,
)


class CooperativeRewardTests(unittest.TestCase):
    def test_individual_reward_mode_uses_per_agent_rewards(self) -> None:
        rewards = apply_reward_mode(
            {"agent_1": -1.1, "agent_2": -0.1},
            info={"agents": {}},
            controlled_agent_ids=("agent_1", "agent_2"),
            reward_mode=MARLRewardMode.INDIVIDUAL,
        )

        self.assertEqual(rewards, {"agent_1": -1.1, "agent_2": -0.1})

    def test_shared_reward_mode_gives_same_reward_to_all_agents(self) -> None:
        rewards = apply_reward_mode(
            {"agent_1": -1.1, "agent_2": -0.1},
            info={"agents": {}},
            controlled_agent_ids=("agent_1", "agent_2"),
            reward_mode=MARLRewardMode.SHARED,
        )

        self.assertEqual(rewards["agent_1"], rewards["agent_2"])
        self.assertEqual(rewards["agent_1"], -0.6000000000000001)

    def test_shared_reward_adds_team_completion_bonus(self) -> None:
        rewards = apply_reward_mode(
            {"agent_1": 9.9, "agent_2": 9.9},
            info={
                "agents": {
                    "agent_1": {"reached_goal": True},
                    "agent_2": {"reached_goal": True},
                }
            },
            controlled_agent_ids=("agent_1", "agent_2"),
            reward_mode=MARLRewardMode.SHARED,
        )

        self.assertEqual(rewards, {"agent_1": 14.9, "agent_2": 14.9})

    def test_training_with_shared_reward_mode_returns_valid_metrics(self) -> None:
        env = self._env()
        metrics = train_independent_q_learning(
            env=env,
            agents=self._agents(env),
            episodes=5,
            reward_mode=MARLRewardMode.SHARED,
        )

        self.assertEqual(metrics.total_episodes, 5)
        self.assertEqual(metrics.success_count + metrics.failure_count, 5)
        self.assertEqual(len(metrics.episode_rewards), 5)
        self.assertEqual(set(metrics.per_agent_rewards), {"agent_1", "agent_2"})

    def test_training_with_individual_reward_mode_still_works(self) -> None:
        env = self._env()
        metrics = train_independent_q_learning(
            env=env,
            agents=self._agents(env),
            episodes=5,
            reward_mode="individual",
        )

        self.assertEqual(metrics.total_episodes, 5)
        self.assertGreaterEqual(metrics.blocked_move_count, 0)

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
    def _agents(env: MultiAgentRLEnv) -> dict[str, QLearningAgent]:
        return {
            agent_id: QLearningAgent(action_size=env.action_size, epsilon=0.2, seed=index)
            for index, agent_id in enumerate(env.controlled_agent_ids, start=1)
        }


if __name__ == "__main__":
    unittest.main()
