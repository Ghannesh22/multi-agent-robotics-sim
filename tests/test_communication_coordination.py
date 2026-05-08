from __future__ import annotations

import unittest
from collections.abc import Mapping

from marlsim.agents import (
    CommunicationAwarePriorityAgent,
    CommunicationAwareWaitingAgent,
    ShortestPathAgent,
)
from marlsim.communication import CommunicationMessage
from marlsim.core.actions import Action
from marlsim.core.environment import ConflictPolicy, GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import CommunicatingMultiAgentRLEnv


class CommunicationAwareWaitingAgentTests(unittest.TestCase):
    def test_build_message_reports_intended_action_and_target(self) -> None:
        agent = AgentState("agent_1", Position(0, 1), Position(2, 1))
        other = AgentState("agent_2", Position(2, 2), Position(0, 2))
        policy = CommunicationAwareWaitingAgent()

        message = policy.build_message(
            agent=agent,
            agents={"agent_1": agent, "agent_2": other},
            obstacles=frozenset(),
            width=3,
            height=3,
            step_count=4,
        )

        self.assertEqual(
            message,
            CommunicationMessage(
                sender_id="agent_1",
                intended_action=Action.RIGHT,
                target_position=Position(2, 1),
                waiting=False,
                priority=0,
                step_count=4,
            ),
        )

    def test_waits_when_message_predicts_same_next_cell(self) -> None:
        agent = AgentState("agent_1", Position(0, 1), Position(2, 1))
        other = AgentState("agent_2", Position(2, 1), Position(0, 1))
        policy = CommunicationAwareWaitingAgent()

        action = policy.choose_action_with_messages(
            agent=agent,
            agents={"agent_1": agent, "agent_2": other},
            obstacles=frozenset(),
            width=3,
            height=3,
            messages={
                "agent_2": CommunicationMessage(
                    sender_id="agent_2",
                    intended_action=Action.LEFT,
                    target_position=Position(0, 1),
                )
            },
        )

        self.assertEqual(action, Action.STAY)

    def test_proceeds_when_other_agent_signals_waiting(self) -> None:
        agent = AgentState("agent_1", Position(0, 0), Position(2, 0))
        other = AgentState("agent_2", Position(0, 2), Position(2, 2))
        policy = CommunicationAwareWaitingAgent()

        action = policy.choose_action_with_messages(
            agent=agent,
            agents={"agent_1": agent, "agent_2": other},
            obstacles=frozenset(),
            width=3,
            height=3,
            messages={
                "agent_2": CommunicationMessage(
                    sender_id="agent_2",
                    intended_action=Action.STAY,
                    target_position=Position(2, 2),
                    waiting=True,
                )
            },
        )

        self.assertEqual(action, Action.RIGHT)


class CommunicationAwarePriorityAgentTests(unittest.TestCase):
    def test_lower_priority_agent_yields_for_same_next_cell(self) -> None:
        agent_1 = AgentState("agent_1", Position(0, 1), Position(1, 1))
        agent_2 = AgentState("agent_2", Position(2, 1), Position(1, 1))
        agents = {"agent_1": agent_1, "agent_2": agent_2}
        policy = CommunicationAwarePriorityAgent()
        messages = _build_messages(policy, agents)

        action_1 = policy.choose_action_with_messages(
            agent=agent_1,
            agents=agents,
            obstacles=frozenset(),
            width=3,
            height=3,
            messages=messages,
        )
        action_2 = policy.choose_action_with_messages(
            agent=agent_2,
            agents=agents,
            obstacles=frozenset(),
            width=3,
            height=3,
            messages=messages,
        )

        self.assertEqual(action_1, Action.RIGHT)
        self.assertEqual(action_2, Action.STAY)

    def test_custom_priority_order_controls_yielding(self) -> None:
        agent_1 = AgentState("agent_1", Position(0, 1), Position(1, 1))
        agent_2 = AgentState("agent_2", Position(2, 1), Position(1, 1))
        agents = {"agent_1": agent_1, "agent_2": agent_2}
        policy = CommunicationAwarePriorityAgent(priority_order=("agent_2", "agent_1"))
        messages = _build_messages(policy, agents)

        action_1 = policy.choose_action_with_messages(
            agent=agent_1,
            agents=agents,
            obstacles=frozenset(),
            width=3,
            height=3,
            messages=messages,
        )
        action_2 = policy.choose_action_with_messages(
            agent=agent_2,
            agents=agents,
            obstacles=frozenset(),
            width=3,
            height=3,
            messages=messages,
        )

        self.assertEqual(action_1, Action.STAY)
        self.assertEqual(action_2, Action.LEFT)

    def test_intended_swap_waits(self) -> None:
        agent_1 = AgentState("agent_1", Position(0, 0), Position(1, 0))
        agent_2 = AgentState("agent_2", Position(1, 0), Position(0, 0))
        agents = {"agent_1": agent_1, "agent_2": agent_2}
        policy = CommunicationAwarePriorityAgent(_FixedPolicy({
            "agent_1": Action.RIGHT,
            "agent_2": Action.LEFT,
        }))
        messages = _build_messages(policy, agents, width=2, height=1)

        action = policy.choose_action_with_messages(
            agent=agent_1,
            agents=agents,
            obstacles=frozenset(),
            width=2,
            height=1,
            messages=messages,
        )

        self.assertEqual(action, Action.STAY)

    def test_communication_reduces_avoidable_same_cell_conflict(self) -> None:
        baseline_env = _same_target_env()
        baseline_actions = {
            agent_id: ShortestPathAgent().choose_action(
                agent=agent,
                agents=baseline_env.agents,
                obstacles=baseline_env.config.obstacles,
                width=baseline_env.config.width,
                height=baseline_env.config.height,
            )
            for agent_id, agent in baseline_env.agents.items()
        }
        baseline_result = baseline_env.step(baseline_actions)

        communication_env = _same_target_env()
        policy = CommunicationAwarePriorityAgent()
        messages = _build_messages(policy, communication_env.agents)
        communication_actions = {
            agent_id: policy.choose_action_with_messages(
                agent=agent,
                agents=communication_env.agents,
                obstacles=communication_env.config.obstacles,
                width=communication_env.config.width,
                height=communication_env.config.height,
                messages=messages,
            )
            for agent_id, agent in communication_env.agents.items()
        }
        communication_result = communication_env.step(communication_actions)

        self.assertEqual(_blocked_count(baseline_result.events), 2)
        self.assertEqual(_blocked_count(communication_result.events), 0)
        self.assertEqual(communication_env.agents["agent_1"].position, Position(1, 1))
        self.assertEqual(communication_env.agents["agent_2"].position, Position(2, 1))

    def test_message_formatting_integrates_with_communicating_wrapper(self) -> None:
        wrapper = CommunicatingMultiAgentRLEnv(
            _same_target_env(),
            ("agent_1", "agent_2"),
        )
        wrapper.reset()
        policy = CommunicationAwarePriorityAgent()
        messages = _build_messages(policy, wrapper.env.agents)

        _, _, _, info = wrapper.step(
            {"agent_1": 3, "agent_2": 4},
            messages,
        )

        self.assertEqual(info["communication"]["communication_count"], 2)
        self.assertIn(
            "step=0 sender=agent_1 action=right target=(1, 1) "
            "blocked=False waiting=False priority=0",
            info["communication"]["formatted_messages"],
        )


class _FixedPolicy:
    def __init__(self, actions: Mapping[str, Action]):
        self.actions = actions

    def choose_action(
        self,
        agent: AgentState,
        agents: Mapping[str, AgentState],
        obstacles: frozenset[Position],
        width: int,
        height: int,
    ) -> Action:
        return self.actions[agent.agent_id]


def _build_messages(
    policy: CommunicationAwarePriorityAgent,
    agents: Mapping[str, AgentState],
    width: int = 3,
    height: int = 3,
) -> dict[str, CommunicationMessage]:
    return {
        agent_id: policy.build_message(
            agent=agent,
            agents=agents,
            obstacles=frozenset(),
            width=width,
            height=height,
            step_count=0,
        )
        for agent_id, agent in agents.items()
    }


def _same_target_env() -> GridWorldEnv:
    return GridWorldEnv(
        GridWorldConfig(
            width=3,
            height=3,
            agents=(
                AgentState("agent_1", Position(0, 1), Position(1, 1)),
                AgentState("agent_2", Position(2, 1), Position(1, 1)),
            ),
            max_steps=4,
            conflict_policy=ConflictPolicy.BLOCK_ALL,
        )
    )


def _blocked_count(events: object) -> int:
    return sum(1 for event in events if event.blocked_reason is not None)


if __name__ == "__main__":
    unittest.main()
