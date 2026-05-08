from __future__ import annotations

import unittest

from marlsim.communication import CommunicationMessage
from marlsim.core.actions import Action
from marlsim.core.environment import ConflictPolicy, GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import (
    CommunicationAwareQLearningAgent,
    CommunicatingMultiAgentRLEnv,
    MultiAgentRLEnv,
    QLearningAgent,
    build_communication_observation,
    extend_observation_with_messages,
    train_communication_q_learning,
    train_independent_q_learning,
)


class CommunicationObservationTests(unittest.TestCase):
    def test_build_communication_observation_adds_message_features(self) -> None:
        agents = {
            "agent_1": AgentState("agent_1", Position(0, 1), Position(1, 1)),
            "agent_2": AgentState("agent_2", Position(2, 1), Position(1, 1)),
        }
        messages = {
            "agent_1": CommunicationMessage(
                sender_id="agent_1",
                intended_action=Action.RIGHT,
                target_position=Position(1, 1),
                priority=0,
            ),
            "agent_2": CommunicationMessage(
                sender_id="agent_2",
                intended_action=Action.LEFT,
                target_position=Position(1, 1),
                priority=1,
            ),
        }

        observation = build_communication_observation(
            agent_id="agent_2",
            observation=(2, 1, 1, 1),
            agents=agents,
            messages=messages,
        )

        self.assertEqual(observation, (2, 1, 1, 1, 0, 0, 1, 1))

    def test_waiting_message_feature_is_encoded(self) -> None:
        agents = {
            "agent_1": AgentState("agent_1", Position(0, 0), Position(2, 0)),
            "agent_2": AgentState("agent_2", Position(0, 2), Position(2, 2)),
        }
        messages = {
            "agent_1": CommunicationMessage(sender_id="agent_1", intended_action=Action.RIGHT),
            "agent_2": CommunicationMessage(
                sender_id="agent_2",
                intended_action=Action.STAY,
                waiting=True,
                priority=2,
            ),
        }

        observation = build_communication_observation(
            agent_id="agent_1",
            observation=(0, 0, 2, 0),
            agents=agents,
            messages=messages,
        )

        self.assertEqual(observation, (0, 0, 2, 0, 1, 2, 0, 0))

    def test_extend_observation_with_messages_handles_all_agents(self) -> None:
        agents = {
            "agent_1": AgentState("agent_1", Position(0, 0), Position(2, 0)),
            "agent_2": AgentState("agent_2", Position(0, 1), Position(2, 1)),
        }
        observations = {
            "agent_1": (0, 0, 2, 0),
            "agent_2": (0, 1, 2, 1),
        }
        messages = {
            "agent_1": CommunicationMessage(sender_id="agent_1", priority=0),
            "agent_2": CommunicationMessage(sender_id="agent_2", priority=1),
        }

        extended = extend_observation_with_messages(
            observations,
            agents=agents,
            messages=messages,
        )

        self.assertEqual(set(extended), {"agent_1", "agent_2"})
        self.assertEqual(len(extended["agent_1"]), 8)
        self.assertEqual(len(extended["agent_2"]), 8)


class CommunicationAwareQLearningTests(unittest.TestCase):
    def test_communication_aware_q_table_updates_with_extended_state(self) -> None:
        agent = CommunicationAwareQLearningAgent(action_size=5, epsilon=0.0)
        state = (0, 1, 1, 1, 0, 1, 1, 1)
        next_state = (1, 1, 1, 1, 0, -1, 0, 0)

        value = agent.update(state, 3, reward=1.0, next_state=next_state, done=False)

        self.assertGreater(value, 0.0)
        self.assertIn(state, agent.q_table)
        self.assertEqual(agent.observation_size, 8)

    def test_training_loop_updates_communication_q_tables(self) -> None:
        env = CommunicatingMultiAgentRLEnv(_two_agent_env(), ("agent_1", "agent_2"))
        agents = _communication_agents(env, epsilon=0.2)

        metrics = train_communication_q_learning(env=env, agents=agents, episodes=5)

        self.assertEqual(metrics.total_episodes, 5)
        self.assertEqual(metrics.success_count + metrics.failure_count, 5)
        self.assertEqual(set(metrics.per_agent_rewards), {"agent_1", "agent_2"})
        self.assertTrue(agents["agent_1"].q_table)
        self.assertTrue(agents["agent_2"].q_table)
        self.assertTrue(all(len(state) == 8 for state in agents["agent_1"].q_table))

    def test_training_rejects_missing_agents(self) -> None:
        env = CommunicatingMultiAgentRLEnv(_two_agent_env(), ("agent_1", "agent_2"))

        with self.assertRaisesRegex(ValueError, "Missing communication-aware"):
            train_communication_q_learning(
                env=env,
                agents={"agent_1": CommunicationAwareQLearningAgent(action_size=env.action_size)},
                episodes=1,
            )

    def test_comparison_metrics_match_independent_training_shape(self) -> None:
        baseline_env = MultiAgentRLEnv(_two_agent_env(), ("agent_1", "agent_2"))
        communication_env = CommunicatingMultiAgentRLEnv(_two_agent_env(), ("agent_1", "agent_2"))

        baseline_metrics = train_independent_q_learning(
            env=baseline_env,
            agents={
                agent_id: QLearningAgent(action_size=baseline_env.action_size, seed=index)
                for index, agent_id in enumerate(baseline_env.controlled_agent_ids, start=1)
            },
            episodes=3,
        )
        communication_metrics = train_communication_q_learning(
            env=communication_env,
            agents=_communication_agents(communication_env),
            episodes=3,
        )

        self.assertEqual(baseline_metrics.total_episodes, 3)
        self.assertEqual(communication_metrics.total_episodes, 3)
        self.assertGreaterEqual(communication_metrics.blocked_move_count, 0)
        self.assertGreaterEqual(communication_metrics.average_episode_length, 1.0)


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
            conflict_policy=ConflictPolicy.BLOCK_ALL,
        )
    )


def _communication_agents(
    env: CommunicatingMultiAgentRLEnv,
    epsilon: float = 0.2,
) -> dict[str, CommunicationAwareQLearningAgent]:
    return {
        agent_id: CommunicationAwareQLearningAgent(
            action_size=env.action_size,
            epsilon=epsilon,
            seed=index,
        )
        for index, agent_id in enumerate(env.controlled_agent_ids, start=1)
    }


if __name__ == "__main__":
    unittest.main()
