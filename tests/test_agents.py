from __future__ import annotations

import unittest

from marlsim.agents.greedy import GreedyGoalAgent
from marlsim.core.actions import Action
from marlsim.core.state import AgentState, Position


class GreedyGoalAgentTests(unittest.TestCase):
    def test_moves_toward_goal_on_open_grid(self) -> None:
        agent = AgentState("a", Position(0, 0), Position(2, 0))
        policy = GreedyGoalAgent()

        action = policy.choose_action(
            agent=agent,
            agents={"a": agent},
            obstacles=frozenset(),
            width=3,
            height=3,
        )

        self.assertEqual(action, Action.RIGHT)

    def test_stays_when_at_goal(self) -> None:
        agent = AgentState("a", Position(1, 1), Position(1, 1))
        policy = GreedyGoalAgent()

        action = policy.choose_action(
            agent=agent,
            agents={"a": agent},
            obstacles=frozenset(),
            width=3,
            height=3,
        )

        self.assertEqual(action, Action.STAY)


if __name__ == "__main__":
    unittest.main()
