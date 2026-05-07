from __future__ import annotations

import random
from collections.abc import Sequence

from marlsim.rl.single_agent_env import Observation


class QLearningAgent:
    """A small tabular Q-learning agent for discrete observations and actions."""

    def __init__(
        self,
        action_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        seed: int | None = None,
    ):
        if action_size <= 0:
            raise ValueError("action_size must be positive.")
        if learning_rate < 0:
            raise ValueError("learning_rate must be non-negative.")
        if discount_factor < 0:
            raise ValueError("discount_factor must be non-negative.")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1.")

        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table: dict[Observation, list[float]] = {}
        self._rng = random.Random(seed)

    def get_q_value(self, state: Observation, action: int) -> float:
        """Return Q(state, action), defaulting unseen values to 0.0."""

        self._validate_action(action)
        values = self.q_table.get(state)
        if values is None:
            return 0.0
        return values[action]

    def choose_action(self, state: Observation) -> int:
        """Choose an action with epsilon-greedy exploration."""

        if self._rng.random() < self.epsilon:
            return self._rng.randrange(self.action_size)
        return self._best_action(state)

    def update(
        self,
        state: Observation,
        action: int,
        reward: float,
        next_state: Observation,
        done: bool,
    ) -> float:
        """Update Q(state, action) and return the new Q-value."""

        self._validate_action(action)
        old_value = self.get_q_value(state, action)
        future_value = 0.0 if done else max(self._q_values(next_state))
        target = reward + self.discount_factor * future_value
        new_value = old_value + self.learning_rate * (target - old_value)
        self._q_values(state)[action] = new_value
        return new_value

    def _q_values(self, state: Observation) -> list[float]:
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in range(self.action_size)]
        return self.q_table[state]

    def _best_action(self, state: Observation) -> int:
        values = self._q_values(state)
        best_value = max(values)
        return _first_index(values, best_value)

    def _validate_action(self, action: int) -> None:
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError("action must be an integer action index.")
        if action < 0 or action >= self.action_size:
            raise ValueError(f"action must be between 0 and {self.action_size - 1}.")


def _first_index(values: Sequence[float], target: float) -> int:
    for index, value in enumerate(values):
        if value == target:
            return index
    raise ValueError("target value not found.")
