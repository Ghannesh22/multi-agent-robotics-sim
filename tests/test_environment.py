from __future__ import annotations

import unittest

from marlsim.core.actions import Action
from marlsim.core.environment import ConflictPolicy, GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position


class GridWorldEnvTests(unittest.TestCase):
    def test_agent_moves_to_empty_cell(self) -> None:
        env = GridWorldEnv(
            GridWorldConfig(
                width=3,
                height=3,
                agents=(AgentState("a", Position(0, 0), Position(2, 0)),),
            )
        )

        result = env.step({"a": Action.RIGHT})

        self.assertEqual(result.agents["a"].position, Position(1, 0))
        self.assertIsNone(result.events[0].blocked_reason)

    def test_out_of_bounds_move_is_blocked(self) -> None:
        env = GridWorldEnv(
            GridWorldConfig(width=3, height=3, agents=(AgentState("a", Position(0, 0)),))
        )

        result = env.step({"a": Action.LEFT})

        self.assertEqual(result.agents["a"].position, Position(0, 0))
        self.assertEqual(result.events[0].blocked_reason, "out_of_bounds")

    def test_obstacle_move_is_blocked(self) -> None:
        env = GridWorldEnv(
            GridWorldConfig(
                width=3,
                height=3,
                obstacles=frozenset({Position(1, 0)}),
                agents=(AgentState("a", Position(0, 0)),),
            )
        )

        result = env.step({"a": Action.RIGHT})

        self.assertEqual(result.agents["a"].position, Position(0, 0))
        self.assertEqual(result.events[0].blocked_reason, "obstacle")

    def test_same_cell_conflict_blocks_both_by_default(self) -> None:
        env = GridWorldEnv(
            GridWorldConfig(
                width=3,
                height=3,
                agents=(
                    AgentState("a", Position(0, 1)),
                    AgentState("b", Position(2, 1)),
                ),
            )
        )

        result = env.step({"a": Action.RIGHT, "b": Action.LEFT})

        self.assertEqual(result.agents["a"].position, Position(0, 1))
        self.assertEqual(result.agents["b"].position, Position(2, 1))
        self.assertEqual(
            {event.blocked_reason for event in result.events},
            {"cell_conflict"},
        )

    def test_priority_policy_allows_one_conflicting_agent_to_move(self) -> None:
        env = GridWorldEnv(
            GridWorldConfig(
                width=3,
                height=3,
                agents=(
                    AgentState("a", Position(0, 1)),
                    AgentState("b", Position(2, 1)),
                ),
                conflict_policy=ConflictPolicy.PRIORITY,
            )
        )

        result = env.step({"a": Action.RIGHT, "b": Action.LEFT})

        self.assertEqual(result.agents["a"].position, Position(1, 1))
        self.assertEqual(result.agents["b"].position, Position(2, 1))

    def test_direct_swap_is_blocked(self) -> None:
        env = GridWorldEnv(
            GridWorldConfig(
                width=3,
                height=3,
                agents=(
                    AgentState("a", Position(0, 1)),
                    AgentState("b", Position(1, 1)),
                ),
            )
        )

        result = env.step({"a": Action.RIGHT, "b": Action.LEFT})

        self.assertEqual(result.agents["a"].position, Position(0, 1))
        self.assertEqual(result.agents["b"].position, Position(1, 1))
        self.assertEqual(
            {event.blocked_reason for event in result.events},
            {"swap_conflict"},
        )

    def test_agent_cannot_move_into_stationary_agent(self) -> None:
        env = GridWorldEnv(
            GridWorldConfig(
                width=3,
                height=3,
                agents=(
                    AgentState("a", Position(0, 1)),
                    AgentState("b", Position(1, 1)),
                ),
            )
        )

        result = env.step({"a": Action.RIGHT, "b": Action.STAY})

        self.assertEqual(result.agents["a"].position, Position(0, 1))
        self.assertEqual(result.agents["b"].position, Position(1, 1))
        self.assertEqual(result.events[0].blocked_reason, "occupied")

    def test_episode_terminates_when_all_agents_reach_goals(self) -> None:
        env = GridWorldEnv(
            GridWorldConfig(
                width=3,
                height=3,
                agents=(
                    AgentState("a", Position(0, 0), Position(1, 0)),
                    AgentState("b", Position(2, 0), Position(2, 1)),
                ),
            )
        )

        result = env.step({"a": Action.RIGHT, "b": Action.DOWN})

        self.assertTrue(result.terminated)
        self.assertEqual(result.reached_goals, frozenset({"a", "b"}))


if __name__ == "__main__":
    unittest.main()
