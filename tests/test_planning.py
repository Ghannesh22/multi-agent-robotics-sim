from __future__ import annotations

import unittest

from marlsim.agents.planning import ShortestPathAgent
from marlsim.core.actions import Action
from marlsim.core.state import AgentState, Position
from marlsim.planning import shortest_path


class PlanningTests(unittest.TestCase):
    def test_bfs_finds_path_around_obstacle(self) -> None:
        path = shortest_path(
            start=Position(0, 1),
            goal=Position(2, 1),
            obstacles=frozenset({Position(1, 1)}),
            width=3,
            height=3,
        )

        self.assertEqual(
            path,
            (
                Position(0, 1),
                Position(0, 0),
                Position(1, 0),
                Position(2, 0),
                Position(2, 1),
            ),
        )

    def test_bfs_returns_none_when_goal_unreachable(self) -> None:
        path = shortest_path(
            start=Position(0, 0),
            goal=Position(2, 0),
            obstacles=frozenset({Position(1, 0), Position(1, 1), Position(1, 2)}),
            width=3,
            height=3,
        )

        self.assertIsNone(path)

    def test_shortest_path_agent_chooses_first_path_step(self) -> None:
        agent = AgentState("a", Position(0, 1), Position(2, 1))
        policy = ShortestPathAgent()

        action = policy.choose_action(
            agent=agent,
            agents={"a": agent},
            obstacles=frozenset({Position(1, 1)}),
            width=3,
            height=3,
        )

        self.assertEqual(action, Action.UP)


if __name__ == "__main__":
    unittest.main()
