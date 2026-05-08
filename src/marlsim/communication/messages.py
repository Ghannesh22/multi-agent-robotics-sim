from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from marlsim.core.actions import Action
from marlsim.core.state import Position


@dataclass(frozen=True)
class CommunicationMessage:
    """Structured symbolic message published by one MARL agent."""

    sender_id: str
    intended_action: Action | None = None
    target_position: Position | None = None
    blocked: bool = False
    waiting: bool = False
    priority: int = 0
    step_count: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.intended_action, str):
            object.__setattr__(
                self,
                "intended_action",
                Action.from_value(self.intended_action),
            )


def serialize_message(message: CommunicationMessage) -> dict[str, Any]:
    """Convert a message into simple built-in Python values."""

    target_position: dict[str, int] | None = None
    if message.target_position is not None:
        target_position = {
            "x": message.target_position.x,
            "y": message.target_position.y,
        }

    intended_action: str | None = None
    if message.intended_action is not None:
        intended_action = message.intended_action.value

    return {
        "sender_id": message.sender_id,
        "intended_action": intended_action,
        "target_position": target_position,
        "blocked": message.blocked,
        "waiting": message.waiting,
        "priority": message.priority,
        "step_count": message.step_count,
    }


def format_message(message: CommunicationMessage) -> str:
    """Return a compact human-readable message summary."""

    action = "none"
    if message.intended_action is not None:
        action = message.intended_action.value

    target = "none"
    if message.target_position is not None:
        target = f"({message.target_position.x}, {message.target_position.y})"

    return (
        f"step={message.step_count} "
        f"sender={message.sender_id} "
        f"action={action} "
        f"target={target} "
        f"blocked={message.blocked} "
        f"waiting={message.waiting} "
        f"priority={message.priority}"
    )


def validate_message(message: CommunicationMessage) -> list[str]:
    """Return validation errors for a communication message."""

    errors: list[str] = []

    if not message.sender_id.strip():
        errors.append("sender_id must be a non-empty string")

    if message.priority < 0:
        errors.append("priority must be greater than or equal to 0")

    if message.step_count < 0:
        errors.append("step_count must be greater than or equal to 0")

    if message.waiting and message.intended_action not in (None, Action.STAY):
        errors.append("waiting messages should use intended_action=None or Action.STAY")

    return errors
