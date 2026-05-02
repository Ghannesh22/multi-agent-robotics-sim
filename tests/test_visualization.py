from __future__ import annotations

import unittest

from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.visualization.console_renderer import ConsoleRenderer


class ConsoleRendererTests(unittest.TestCase):
    def test_render_includes_grid_symbols_and_metadata(self) -> None:
        env = GridWorldEnv(
            GridWorldConfig(
                width=4,
                height=3,
                obstacles=frozenset({Position(1, 1)}),
                agents=(
                    AgentState("agent_1", Position(0, 0), Position(3, 0)),
                    AgentState("agent_2", Position(3, 2), Position(3, 2)),
                ),
            )
        )

        output = ConsoleRenderer().render(env)

        self.assertIn("Step 0", output)
        self.assertIn("+-----+-----+-----+-----+", output)
        self.assertIn(" A1  ", output)
        self.assertIn(" G1  ", output)
        self.assertIn(" ### ", output)
        self.assertIn(" A2* ", output)
        self.assertIn("Legend: A# = agent", output)
        self.assertIn("- A1 agent_1: position=(0, 0), goal=(3, 0)", output)
        self.assertIn("- A2 agent_2: position=(3, 2), goal=(3, 2)", output)


if __name__ == "__main__":
    unittest.main()
