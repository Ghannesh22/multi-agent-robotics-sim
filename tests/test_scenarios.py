from __future__ import annotations

import unittest

from marlsim.core.environment import ConflictPolicy, GridWorldEnv
from marlsim.core.state import Position
from marlsim.scenarios import (
    AgentScenario,
    crossing_paths,
    get_scenario,
    make_grid_world_config,
    narrow_corridor,
    scenario_names,
    two_agent_obstacle_course,
)


class ScenarioTests(unittest.TestCase):
    def test_simple_scenario_still_matches_original_shape(self) -> None:
        config = two_agent_obstacle_course()

        self.assertEqual(config.width, 7)
        self.assertEqual(config.height, 5)
        self.assertEqual(
            config.obstacles,
            frozenset({Position(3, 1), Position(3, 2), Position(3, 3)}),
        )
        self.assertEqual(len(config.agents), 2)
        self.assertEqual(config.agents[0].position, Position(0, 0))
        self.assertEqual(config.agents[0].goal, Position(6, 0))
        self.assertEqual(config.agents[1].position, Position(0, 4))
        self.assertEqual(config.agents[1].goal, Position(6, 4))

    def test_custom_config_accepts_dimensions_agents_goals_and_obstacles(self) -> None:
        config = make_grid_world_config(
            width=6,
            height=4,
            obstacles=((2, 1), (2, 2)),
            agents=(
                AgentScenario("agent_1", (0, 0), (5, 0)),
                AgentScenario("agent_2", (0, 3), (5, 3)),
                AgentScenario("agent_3", (5, 1), None),
            ),
            max_steps=15,
            conflict_policy=ConflictPolicy.PRIORITY,
        )

        env = GridWorldEnv(config)

        self.assertEqual(env.config.width, 6)
        self.assertEqual(env.config.height, 4)
        self.assertEqual(env.config.obstacles, frozenset({Position(2, 1), Position(2, 2)}))
        self.assertEqual(tuple(env.agents), ("agent_1", "agent_2", "agent_3"))
        self.assertIsNone(env.agents["agent_3"].goal)
        self.assertEqual(env.config.max_steps, 15)
        self.assertEqual(env.config.conflict_policy, ConflictPolicy.PRIORITY)

    def test_named_scenarios_are_available(self) -> None:
        self.assertEqual(
            scenario_names(),
            ("simple", "narrow_corridor", "crossing_paths"),
        )
        self.assertEqual(get_scenario("simple"), two_agent_obstacle_course())
        self.assertEqual(get_scenario("narrow_corridor"), narrow_corridor())
        self.assertEqual(get_scenario("crossing_paths"), crossing_paths())

    def test_unknown_scenario_name_explains_available_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "Available scenarios"):
            get_scenario("missing")


if __name__ == "__main__":
    unittest.main()
