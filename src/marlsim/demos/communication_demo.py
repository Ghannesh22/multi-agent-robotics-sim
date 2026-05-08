from __future__ import annotations

from collections.abc import Mapping

from marlsim.agents import CommunicationAwarePriorityAgent, ShortestPathAgent
from marlsim.communication import CommunicationMessage, format_message
from marlsim.core.actions import Action
from marlsim.core.environment import ConflictPolicy, GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import CommunicatingMultiAgentRLEnv

ACTION_TO_INT = {
    Action.UP: 0,
    Action.DOWN: 1,
    Action.LEFT: 2,
    Action.RIGHT: 3,
    Action.STAY: 4,
}


def main() -> None:
    print("No communication baseline")
    print("-------------------------")
    print(_run_no_communication_step())
    print()
    print("Communication-aware priority")
    print("----------------------------")
    print(_run_communication_step())


def _run_no_communication_step() -> str:
    env = GridWorldEnv(_same_target_config())
    policy = ShortestPathAgent()
    actions = {
        agent_id: policy.choose_action(
            agent=agent,
            agents=env.agents,
            obstacles=env.config.obstacles,
            width=env.config.width,
            height=env.config.height,
        )
        for agent_id, agent in env.agents.items()
    }
    result = env.step(actions)

    lines = [
        f"actions: {_format_actions(actions)}",
        f"blocked moves: {_blocked_count(result.events)}",
        f"positions: {_format_positions(env.agents)}",
    ]
    return "\n".join(lines)


def _run_communication_step() -> str:
    wrapper = CommunicatingMultiAgentRLEnv(
        GridWorldEnv(_same_target_config()),
        ("agent_1", "agent_2"),
    )
    wrapper.reset()
    policy = CommunicationAwarePriorityAgent()
    messages = {
        agent_id: policy.build_message(
            agent=agent,
            agents=wrapper.env.agents,
            obstacles=wrapper.env.config.obstacles,
            width=wrapper.env.config.width,
            height=wrapper.env.config.height,
            step_count=wrapper.step_count,
        )
        for agent_id, agent in wrapper.env.agents.items()
    }
    actions = {
        agent_id: policy.choose_action_with_messages(
            agent=agent,
            agents=wrapper.env.agents,
            obstacles=wrapper.env.config.obstacles,
            width=wrapper.env.config.width,
            height=wrapper.env.config.height,
            messages=messages,
        )
        for agent_id, agent in wrapper.env.agents.items()
    }
    _, _, _, info = wrapper.step(
        {agent_id: ACTION_TO_INT[action] for agent_id, action in actions.items()},
        messages,
    )

    lines = ["messages:"]
    lines.extend(f"- {format_message(message)}" for message in messages.values())
    lines.extend(
        [
            f"actions: {_format_actions(actions)}",
            f"blocked moves: {_blocked_count_from_info(info)}",
            f"positions: {_format_positions(wrapper.env.agents)}",
            "communication log:",
        ]
    )
    lines.extend(f"- {line}" for line in info["communication"]["formatted_messages"])
    return "\n".join(lines)


def _same_target_config() -> GridWorldConfig:
    return GridWorldConfig(
        width=3,
        height=3,
        agents=(
            AgentState("agent_1", Position(0, 1), Position(1, 1)),
            AgentState("agent_2", Position(2, 1), Position(1, 1)),
        ),
        max_steps=4,
        conflict_policy=ConflictPolicy.BLOCK_ALL,
    )


def _format_actions(actions: Mapping[str, Action]) -> str:
    return ", ".join(
        f"{agent_id}={action.value}" for agent_id, action in sorted(actions.items())
    )


def _format_positions(agents: Mapping[str, AgentState]) -> str:
    return ", ".join(
        f"{agent_id}=({agent.position.x},{agent.position.y})"
        for agent_id, agent in sorted(agents.items())
    )


def _blocked_count(events: object) -> int:
    return sum(1 for event in events if event.blocked_reason is not None)


def _blocked_count_from_info(info: dict[str, object]) -> int:
    agents = info["agents"]
    return sum(1 for agent_info in agents.values() if agent_info["blocked"])


if __name__ == "__main__":
    main()
