from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

from marlsim.core.actions import Action
from marlsim.core.state import AgentState, Position


class AgentPolicy(ABC):
    """Base class for policies that choose one action for one agent."""

    @abstractmethod
    def choose_action(
        self,
        agent: AgentState,
        agents: Mapping[str, AgentState],
        obstacles: frozenset[Position],
        width: int,
        height: int,
    ) -> Action:
        raise NotImplementedError
