from __future__ import annotations

import random
from typing import Mapping

from marlsim.agents.base import AgentPolicy
from marlsim.core.actions import Action
from marlsim.core.state import AgentState, Position


class RandomAgent(AgentPolicy):
    """Uniform random policy over the five discrete actions."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def choose_action(
        self,
        agent: AgentState,
        agents: Mapping[str, AgentState],
        obstacles: frozenset[Position],
        width: int,
        height: int,
    ) -> Action:
        return self._rng.choice(list(Action))
