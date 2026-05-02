from __future__ import annotations

from marlsim.core.environment import GridWorldEnv


class ConsoleRenderer:
    """Render a grid-world state as plain text."""

    cell_width = 3

    def render(self, env: GridWorldEnv) -> str:
        cells = [["." for _ in range(env.config.width)] for _ in range(env.config.height)]

        for obstacle in env.config.obstacles:
            cells[obstacle.y][obstacle.x] = "#"

        for agent in env.agents.values():
            if agent.goal is not None:
                cells[agent.goal.y][agent.goal.x] = "G"

        for agent_id, agent in sorted(env.agents.items()):
            symbol = self._agent_symbol(agent_id)
            if agent.at_goal:
                symbol = f"{symbol}*"
            cells[agent.position.y][agent.position.x] = symbol

        lines = [f"Step: {env.step_count}"]
        lines.extend(self._grid_lines(cells))
        lines.append("Legend: 1,2 = agents; 1* = agent on goal; G = goal; # = obstacle; . = empty")
        lines.append("Agents:")
        for agent_id, agent in sorted(env.agents.items()):
            goal = self._format_position(agent.goal) if agent.goal is not None else "none"
            lines.append(
                f"- {agent_id}: position={self._format_position(agent.position)}, goal={goal}"
            )
        return "\n".join(lines)

    def _grid_lines(self, cells: list[list[str]]) -> list[str]:
        border = "+" + "+".join("-" * self.cell_width for _ in cells[0]) + "+"
        lines = [border]
        for row in cells:
            rendered_cells = "|".join(f"{cell:^{self.cell_width}}" for cell in row)
            lines.append(f"|{rendered_cells}|")
            lines.append(border)
        return lines

    @staticmethod
    def _agent_symbol(agent_id: str) -> str:
        digits = "".join(character for character in agent_id if character.isdigit())
        if digits:
            return digits[-1]
        if agent_id:
            return agent_id[0].upper()
        return "A"

    @staticmethod
    def _format_position(position: object) -> str:
        return f"({position.x}, {position.y})"
