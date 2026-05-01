from __future__ import annotations

from marlsim.core.environment import GridWorldEnv
from marlsim.core.state import Position


class ConsoleRenderer:
    """Render a grid-world state as plain text."""

    def render(self, env: GridWorldEnv) -> str:
        cells = [["." for _ in range(env.config.width)] for _ in range(env.config.height)]

        for obstacle in env.config.obstacles:
            cells[obstacle.y][obstacle.x] = "#"

        for agent in env.agents.values():
            if agent.goal is not None:
                cells[agent.goal.y][agent.goal.x] = "G"

        for agent_id, agent in sorted(env.agents.items()):
            cells[agent.position.y][agent.position.x] = self._agent_symbol(agent_id)

        return "\n".join(" ".join(row) for row in cells)

    @staticmethod
    def _agent_symbol(agent_id: str) -> str:
        digits = "".join(character for character in agent_id if character.isdigit())
        if digits:
            return digits[-1]
        if agent_id:
            return agent_id[0].upper()
        return "A"
