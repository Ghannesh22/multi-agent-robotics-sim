from __future__ import annotations

from collections import defaultdict

from marlsim.core.environment import GridWorldEnv
from marlsim.core.state import Position


class ConsoleRenderer:
    """Render a grid-world state as plain text."""

    cell_width = 5

    def render(self, env: GridWorldEnv) -> str:
        cells = self._empty_cells(env)

        lines = [f"Step {env.step_count}"]
        lines.extend(self._grid_lines(cells))
        lines.append(
            "Legend: A# = agent; A#* = agent on goal; "
            "G# = goal; G+ = shared goal; ### = obstacle; . = empty"
        )
        lines.append("Agents:")
        for agent_id, agent in sorted(env.agents.items()):
            goal = self._format_position(agent.goal) if agent.goal is not None else "none"
            lines.append(
                f"- {self._agent_symbol(agent_id)} {agent_id}: "
                f"position={self._format_position(agent.position)}, goal={goal}"
            )
        return "\n".join(lines)

    def _empty_cells(self, env: GridWorldEnv) -> list[list[str]]:
        cells = [["." for _ in range(env.config.width)] for _ in range(env.config.height)]

        for obstacle in env.config.obstacles:
            cells[obstacle.y][obstacle.x] = "###"

        goals_by_position: dict[Position, list[str]] = defaultdict(list)
        for agent_id, agent in sorted(env.agents.items()):
            if agent.goal is not None:
                goals_by_position[agent.goal].append(self._goal_symbol(agent_id))

        for position, symbols in goals_by_position.items():
            cells[position.y][position.x] = symbols[0] if len(symbols) == 1 else "G+"

        for agent_id, agent in sorted(env.agents.items()):
            symbol = self._agent_symbol(agent_id)
            if agent.at_goal:
                symbol = f"{symbol}*"
            cells[agent.position.y][agent.position.x] = symbol

        return cells

    def _grid_lines(self, cells: list[list[str]]) -> list[str]:
        cell_width = max(self.cell_width, *(len(cell) for row in cells for cell in row))
        border = "+" + "+".join("-" * cell_width for _ in cells[0]) + "+"
        lines = [border]
        for row in cells:
            rendered_cells = "|".join(f"{cell:^{cell_width}}" for cell in row)
            lines.append(f"|{rendered_cells}|")
            lines.append(border)
        return lines

    @staticmethod
    def _agent_symbol(agent_id: str) -> str:
        numeric_suffix = agent_id.rsplit("_", maxsplit=1)[-1]
        if numeric_suffix.isdigit():
            return f"A{numeric_suffix}"

        digits = "".join(character for character in agent_id if character.isdigit())
        if digits:
            return f"A{digits}"
        if agent_id:
            return f"A{agent_id[0].upper()}"
        return "A"

    @classmethod
    def _goal_symbol(cls, agent_id: str) -> str:
        agent_symbol = cls._agent_symbol(agent_id)
        if agent_symbol.startswith("A"):
            return f"G{agent_symbol[1:]}"
        return "G"

    @staticmethod
    def _format_position(position: Position) -> str:
        return f"({position.x}, {position.y})"
