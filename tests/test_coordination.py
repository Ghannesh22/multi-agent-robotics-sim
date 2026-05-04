from __future__ import annotations

import unittest
from typing import Mapping

from marlsim.agents.base import AgentPolicy
from marlsim.agents import (
    ConflictAwareWaitingAgent,
    LocalReplanningAgent,
    PriorityBasedCoordinationAgent,
    ShortestPathAgent,
)
from marlsim.core.actions import Action
from marlsim.core.environment import GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.experiments import ExperimentLogger, RunSummary
from marlsim.scenarios import get_scenario


class ConflictAwareWaitingAgentTests(unittest.TestCase):
    def test_waits_to_avoid_same_cell_conflict(self) -> None:
        agent = AgentState("agent_1", Position(0, 1), Position(2, 1))
        other = AgentState("agent_2", Position(2, 1), Position(0, 1))
        policy = ConflictAwareWaitingAgent()

        action = policy.choose_action(
            agent=agent,
            agents={"agent_1": agent, "agent_2": other},
            obstacles=frozenset(),
            width=3,
            height=3,
        )

        self.assertEqual(action, Action.STAY)

    def test_moves_when_no_conflict_exists(self) -> None:
        agent = AgentState("agent_1", Position(0, 0), Position(2, 0))
        other = AgentState("agent_2", Position(0, 2), Position(2, 2))
        policy = ConflictAwareWaitingAgent()

        action = policy.choose_action(
            agent=agent,
            agents={"agent_1": agent, "agent_2": other},
            obstacles=frozenset(),
            width=3,
            height=3,
        )

        self.assertEqual(action, Action.RIGHT)

    def test_waits_to_avoid_direct_swap_conflict(self) -> None:
        agent = AgentState("agent_1", Position(0, 0), Position(1, 0))
        other = AgentState("agent_2", Position(1, 0), Position(0, 0))
        policy = ConflictAwareWaitingAgent(
            _FixedPolicy({"agent_1": Action.RIGHT, "agent_2": Action.LEFT})
        )

        action = policy.choose_action(
            agent=agent,
            agents={"agent_1": agent, "agent_2": other},
            obstacles=frozenset(),
            width=2,
            height=1,
        )

        self.assertEqual(action, Action.STAY)


class PriorityBasedCoordinationAgentTests(unittest.TestCase):
    def test_lower_priority_agent_yields_in_same_cell_conflict(self) -> None:
        agent = AgentState("agent_2", Position(2, 1), Position(1, 1))
        other = AgentState("agent_1", Position(0, 1), Position(1, 1))
        policy = PriorityBasedCoordinationAgent()

        action = policy.choose_action(
            agent=agent,
            agents={"agent_1": other, "agent_2": agent},
            obstacles=frozenset(),
            width=3,
            height=3,
        )

        self.assertEqual(action, Action.STAY)

    def test_higher_priority_agent_proceeds_in_same_cell_conflict(self) -> None:
        agent = AgentState("agent_1", Position(0, 1), Position(1, 1))
        other = AgentState("agent_2", Position(2, 1), Position(1, 1))
        policy = PriorityBasedCoordinationAgent()

        action = policy.choose_action(
            agent=agent,
            agents={"agent_1": agent, "agent_2": other},
            obstacles=frozenset(),
            width=3,
            height=3,
        )

        self.assertEqual(action, Action.RIGHT)

    def test_custom_priority_order_is_stable(self) -> None:
        agent_1 = AgentState("agent_1", Position(0, 1), Position(1, 1))
        agent_2 = AgentState("agent_2", Position(2, 1), Position(1, 1))
        policy = PriorityBasedCoordinationAgent(priority_order=("agent_2", "agent_1"))

        action_1 = policy.choose_action(
            agent=agent_1,
            agents={"agent_1": agent_1, "agent_2": agent_2},
            obstacles=frozenset(),
            width=3,
            height=3,
        )
        action_2 = policy.choose_action(
            agent=agent_2,
            agents={"agent_1": agent_1, "agent_2": agent_2},
            obstacles=frozenset(),
            width=3,
            height=3,
        )

        self.assertEqual(action_1, Action.STAY)
        self.assertEqual(action_2, Action.LEFT)

    def test_direct_swap_waits_because_target_is_not_safe(self) -> None:
        agent_1 = AgentState("agent_1", Position(0, 0), Position(1, 0))
        agent_2 = AgentState("agent_2", Position(1, 0), Position(0, 0))
        policy = PriorityBasedCoordinationAgent(
            _FixedPolicy({"agent_1": Action.RIGHT, "agent_2": Action.LEFT})
        )

        action_1 = policy.choose_action(
            agent=agent_1,
            agents={"agent_1": agent_1, "agent_2": agent_2},
            obstacles=frozenset(),
            width=2,
            height=1,
        )
        action_2 = policy.choose_action(
            agent=agent_2,
            agents={"agent_1": agent_1, "agent_2": agent_2},
            obstacles=frozenset(),
            width=2,
            height=1,
        )

        self.assertEqual(action_1, Action.STAY)
        self.assertEqual(action_2, Action.STAY)

    def test_crossing_paths_succeeds_better_than_conflict_aware_waiting(self) -> None:
        waiting_summary = _run_scenario("crossing_paths", ConflictAwareWaitingAgent)
        priority_summary = _run_scenario("crossing_paths", PriorityBasedCoordinationAgent)

        self.assertFalse(waiting_summary.succeeded)
        self.assertTrue(priority_summary.succeeded)
        self.assertLess(priority_summary.total_steps, waiting_summary.total_steps)
        self.assertEqual(priority_summary.blocked_moves, 0)


class LocalReplanningAgentTests(unittest.TestCase):
    def test_finds_alternate_route_when_preferred_path_is_blocked(self) -> None:
        agent = AgentState("agent_2", Position(1, 0), Position(1, 2))
        other = AgentState("agent_1", Position(0, 1), Position(1, 1))
        policy = LocalReplanningAgent()

        action = policy.choose_action(
            agent=agent,
            agents={"agent_1": other, "agent_2": agent},
            obstacles=frozenset(),
            width=3,
            height=3,
        )

        self.assertEqual(action, Action.RIGHT)

    def test_waits_if_no_safe_alternate_route_exists(self) -> None:
        agent = AgentState("agent_2", Position(1, 0), Position(1, 2))
        other = AgentState("agent_1", Position(0, 1), Position(1, 1))
        policy = LocalReplanningAgent()

        action = policy.choose_action(
            agent=agent,
            agents={"agent_1": other, "agent_2": agent},
            obstacles=frozenset({Position(0, 0), Position(2, 0)}),
            width=3,
            height=3,
        )

        self.assertEqual(action, Action.STAY)

    def test_existing_scenarios_still_pass(self) -> None:
        for scenario_name in ("simple", "crossing_paths", "narrow_corridor"):
            with self.subTest(scenario_name=scenario_name):
                summary = _run_scenario(scenario_name, LocalReplanningAgent)

                self.assertTrue(summary.succeeded)
                self.assertEqual(summary.blocked_moves, 0)


class _FixedPolicy(AgentPolicy):
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


def _run_scenario(
    scenario_name: str,
    policy_type: type[
        ConflictAwareWaitingAgent
        | LocalReplanningAgent
        | PriorityBasedCoordinationAgent
        | ShortestPathAgent
    ],
) -> RunSummary:
    env = GridWorldEnv(get_scenario(scenario_name))
    logger = ExperimentLogger(scenario_name)
    policies = {agent_id: policy_type() for agent_id in env.agents}

    while True:
        actions = {
            agent_id: policies[agent_id].choose_action(
                agent=agent,
                agents=env.agents,
                obstacles=env.config.obstacles,
                width=env.config.width,
                height=env.config.height,
            )
            for agent_id, agent in env.agents.items()
        }
        result = env.step(actions)
        logger.record_step(result)

        if result.terminated:
            return logger.summary()


if __name__ == "__main__":
    unittest.main()
