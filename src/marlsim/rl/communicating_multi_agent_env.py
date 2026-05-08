from __future__ import annotations

from collections.abc import Mapping, Sequence

from marlsim.communication import (
    CommunicationMessage,
    format_message,
    serialize_message,
    validate_message,
)
from marlsim.core.environment import GridWorldEnv
from marlsim.rl.multi_agent_env import (
    MultiAgentInfo,
    MultiAgentObservations,
    MultiAgentRLEnv,
    MultiAgentRewards,
)
from marlsim.rl.single_agent_env import RewardConfig


CommunicationMessages = Mapping[str, CommunicationMessage]


class CommunicatingMultiAgentRLEnv(MultiAgentRLEnv):
    """A MARL wrapper that logs optional structured agent communication.

    Communication is currently observational only. Messages are validated,
    stored, formatted, and returned through the `info` dictionary, but they do
    not affect actions, rewards, movement, collisions, or episode termination.
    """

    def __init__(
        self,
        env: GridWorldEnv,
        controlled_agent_ids: Sequence[str],
        rewards: RewardConfig | None = None,
    ):
        super().__init__(env=env, controlled_agent_ids=controlled_agent_ids, rewards=rewards)
        self._last_messages: dict[str, CommunicationMessage] = {}
        self._message_history: list[dict[str, CommunicationMessage]] = []

    def reset(self) -> MultiAgentObservations:
        """Reset the environment and clear communication history."""

        observations = super().reset()
        self._last_messages = {}
        self._message_history = []
        return observations

    def step(
        self,
        actions: Mapping[str, int],
        messages: CommunicationMessages | None = None,
    ) -> tuple[MultiAgentObservations, MultiAgentRewards, bool, MultiAgentInfo]:
        """Apply simultaneous actions and attach optional communication info."""

        self._record_messages(messages)
        observations, rewards, done, info = super().step(actions)
        info["communication"] = self._communication_info()
        return observations, rewards, done, info

    def get_last_messages(self) -> dict[str, CommunicationMessage]:
        """Return the most recent messages keyed by sender id."""

        return dict(self._last_messages)

    def format_last_messages(self) -> tuple[str, ...]:
        """Return readable summaries for the most recent messages."""

        return tuple(format_message(message) for message in self._last_messages.values())

    def _record_messages(self, messages: CommunicationMessages | None) -> None:
        if messages is None:
            self._last_messages = {}
            self._message_history.append({})
            return

        self._validate_message_mapping(messages)
        self._last_messages = dict(messages)
        self._message_history.append(dict(messages))

    def _validate_message_mapping(self, messages: CommunicationMessages) -> None:
        expected = set(self.controlled_agent_ids)
        received = set(messages)
        unexpected = received - expected
        if unexpected:
            raise ValueError(f"Unexpected messages for uncontrolled agents: {sorted(unexpected)}")

        for agent_id, message in messages.items():
            if message.sender_id != agent_id:
                raise ValueError(
                    f"Message key '{agent_id}' must match sender_id '{message.sender_id}'."
                )

            errors = validate_message(message)
            if errors:
                joined_errors = "; ".join(errors)
                raise ValueError(f"Invalid message for '{agent_id}': {joined_errors}")

    def _communication_info(self) -> dict[str, object]:
        return {
            "last_messages": {
                agent_id: serialize_message(message)
                for agent_id, message in self._last_messages.items()
            },
            "formatted_messages": self.format_last_messages(),
            "communication_count": len(self._last_messages),
            "history_length": len(self._message_history),
        }
