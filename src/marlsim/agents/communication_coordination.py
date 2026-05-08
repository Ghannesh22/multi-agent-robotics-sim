from __future__ import annotations

from collections.abc import Mapping

from marlsim.agents.base import AgentPolicy
from marlsim.agents.planning import ShortestPathAgent
from marlsim.communication import CommunicationMessage
from marlsim.core.actions import Action
from marlsim.core.state import AgentState, Position


class CommunicationAwareWaitingAgent(AgentPolicy):
    """Yield when received messages predict an avoidable movement conflict."""

    def __init__(self, base_policy: AgentPolicy | None = None):
        self.base_policy = base_policy or ShortestPathAgent()

    def choose_action(
        self,
        agent: AgentState,
        agents: Mapping[str, AgentState],
        obstacles: frozenset[Position],
        width: int,
        height: int,
    ) -> Action:
        return self.choose_action_with_messages(
            agent=agent,
            agents=agents,
            obstacles=obstacles,
            width=width,
            height=height,
            messages={},
        )

    def build_message(
        self,
        agent: AgentState,
        agents: Mapping[str, AgentState],
        obstacles: frozenset[Position],
        width: int,
        height: int,
        step_count: int = 0,
    ) -> CommunicationMessage:
        intended_action = self.base_policy.choose_action(
            agent=agent,
            agents=agents,
            obstacles=obstacles,
            width=width,
            height=height,
        )
        return CommunicationMessage(
            sender_id=agent.agent_id,
            intended_action=intended_action,
            target_position=agent.goal,
            waiting=intended_action == Action.STAY,
            priority=0,
            step_count=step_count,
        )

    def choose_action_with_messages(
        self,
        agent: AgentState,
        agents: Mapping[str, AgentState],
        obstacles: frozenset[Position],
        width: int,
        height: int,
        messages: Mapping[str, CommunicationMessage],
    ) -> Action:
        intended_action = self.base_policy.choose_action(
            agent=agent,
            agents=agents,
            obstacles=obstacles,
            width=width,
            height=height,
        )

        if intended_action == Action.STAY:
            return Action.STAY

        if self.predict_conflict_from_messages(
            agent=agent,
            intended_action=intended_action,
            agents=agents,
            messages=messages,
        ):
            return Action.STAY

        return intended_action

    def predict_conflict_from_messages(
        self,
        agent: AgentState,
        intended_action: Action,
        agents: Mapping[str, AgentState],
        messages: Mapping[str, CommunicationMessage],
    ) -> bool:
        intended_target = _target_position(agent.position, intended_action)

        for other_id, message in messages.items():
            if other_id == agent.agent_id or other_id not in agents:
                continue

            other = agents[other_id]
            other_target = _message_intended_target(other, message)

            if message.waiting and intended_target != other.position:
                continue

            same_next_cell = intended_target == other_target
            direct_swap = intended_target == other.position and other_target == agent.position
            moving_into_waiting_agent = intended_target == other.position and message.waiting

            if same_next_cell or direct_swap or moving_into_waiting_agent:
                return True

        return False


class CommunicationAwarePriorityAgent(CommunicationAwareWaitingAgent):
    """Use message priority to decide which agent yields in predicted conflicts."""

    def __init__(
        self,
        base_policy: AgentPolicy | None = None,
        priority_order: tuple[str, ...] | None = None,
    ):
        super().__init__(base_policy=base_policy)
        self.priority_order = priority_order

    def build_message(
        self,
        agent: AgentState,
        agents: Mapping[str, AgentState],
        obstacles: frozenset[Position],
        width: int,
        height: int,
        step_count: int = 0,
    ) -> CommunicationMessage:
        message = super().build_message(
            agent=agent,
            agents=agents,
            obstacles=obstacles,
            width=width,
            height=height,
            step_count=step_count,
        )
        return CommunicationMessage(
            sender_id=message.sender_id,
            intended_action=message.intended_action,
            target_position=message.target_position,
            blocked=message.blocked,
            waiting=message.waiting,
            priority=self._priority_index(agent.agent_id, agents),
            step_count=message.step_count,
        )

    def predict_conflict_from_messages(
        self,
        agent: AgentState,
        intended_action: Action,
        agents: Mapping[str, AgentState],
        messages: Mapping[str, CommunicationMessage],
    ) -> bool:
        intended_target = _target_position(agent.position, intended_action)
        agent_priority = self._priority_index(agent.agent_id, agents)

        for other_id, message in messages.items():
            if other_id == agent.agent_id or other_id not in agents:
                continue

            other = agents[other_id]
            other_target = _message_intended_target(other, message)

            if message.waiting and intended_target != other.position:
                continue

            direct_swap = intended_target == other.position and other_target == agent.position
            if direct_swap:
                return True

            moving_into_waiting_agent = intended_target == other.position and message.waiting
            if moving_into_waiting_agent:
                return True

            same_next_cell = intended_target == other_target
            same_declared_target = (
                message.target_position is not None
                and message.target_position == agent.goal
                and agent.goal is not None
            )
            if same_next_cell or same_declared_target:
                if _message_has_higher_priority(
                    other_id=other_id,
                    other_priority=message.priority,
                    agent_id=agent.agent_id,
                    agent_priority=agent_priority,
                ):
                    return True

        return False

    def _priority_index(self, agent_id: str, agents: Mapping[str, AgentState]) -> int:
        ordered_agent_ids = self.priority_order or tuple(sorted(agents))
        if agent_id in ordered_agent_ids:
            return ordered_agent_ids.index(agent_id)
        return len(ordered_agent_ids) + tuple(sorted(agents)).index(agent_id)


def _target_position(position: Position, action: Action | None) -> Position:
    if action is None:
        return position
    dx, dy = action.delta
    return position.move(dx, dy)


def _message_intended_target(
    agent: AgentState,
    message: CommunicationMessage,
) -> Position:
    return _target_position(agent.position, message.intended_action)


def _message_has_higher_priority(
    *,
    other_id: str,
    other_priority: int,
    agent_id: str,
    agent_priority: int,
) -> bool:
    if other_priority != agent_priority:
        return other_priority < agent_priority
    return other_id < agent_id
