from __future__ import annotations

import unittest

from marlsim.communication import (
    CommunicationMessage,
    format_message,
    serialize_message,
    validate_message,
)
from marlsim.core.actions import Action
from marlsim.core.state import Position


class CommunicationMessageTests(unittest.TestCase):
    def test_message_creation_with_explicit_fields(self) -> None:
        message = CommunicationMessage(
            sender_id="agent_1",
            intended_action=Action.RIGHT,
            target_position=Position(2, 0),
            blocked=False,
            waiting=False,
            priority=1,
            step_count=3,
        )

        self.assertEqual(message.sender_id, "agent_1")
        self.assertEqual(message.intended_action, Action.RIGHT)
        self.assertEqual(message.target_position, Position(2, 0))
        self.assertFalse(message.blocked)
        self.assertFalse(message.waiting)
        self.assertEqual(message.priority, 1)
        self.assertEqual(message.step_count, 3)

    def test_message_creation_accepts_action_strings(self) -> None:
        message = CommunicationMessage(
            sender_id="agent_1",
            intended_action="stay",
            waiting=True,
        )

        self.assertEqual(message.intended_action, Action.STAY)

    def test_serialize_message_uses_builtin_values(self) -> None:
        message = CommunicationMessage(
            sender_id="agent_2",
            intended_action=Action.UP,
            target_position=Position(1, 4),
            blocked=True,
            waiting=False,
            priority=2,
            step_count=7,
        )

        self.assertEqual(
            serialize_message(message),
            {
                "sender_id": "agent_2",
                "intended_action": "up",
                "target_position": {"x": 1, "y": 4},
                "blocked": True,
                "waiting": False,
                "priority": 2,
                "step_count": 7,
            },
        )

    def test_format_message_is_readable(self) -> None:
        message = CommunicationMessage(
            sender_id="agent_1",
            intended_action=Action.LEFT,
            target_position=Position(0, 2),
            blocked=True,
            waiting=False,
            priority=3,
            step_count=5,
        )

        self.assertEqual(
            format_message(message),
            (
                "step=5 sender=agent_1 action=left target=(0, 2) "
                "blocked=True waiting=False priority=3"
            ),
        )

    def test_validation_accepts_valid_message(self) -> None:
        message = CommunicationMessage(
            sender_id="agent_1",
            intended_action=Action.STAY,
            waiting=True,
            priority=0,
            step_count=0,
        )

        self.assertEqual(validate_message(message), [])

    def test_validation_reports_invalid_message_fields(self) -> None:
        message = CommunicationMessage(
            sender_id=" ",
            intended_action=Action.RIGHT,
            waiting=True,
            priority=-1,
            step_count=-2,
        )

        self.assertEqual(
            validate_message(message),
            [
                "sender_id must be a non-empty string",
                "priority must be greater than or equal to 0",
                "step_count must be greater than or equal to 0",
                "waiting messages should use intended_action=None or Action.STAY",
            ],
        )

    def test_messages_compare_by_value(self) -> None:
        first = CommunicationMessage(
            sender_id="agent_1",
            intended_action=Action.DOWN,
            target_position=Position(1, 2),
            step_count=4,
        )
        second = CommunicationMessage(
            sender_id="agent_1",
            intended_action=Action.DOWN,
            target_position=Position(1, 2),
            step_count=4,
        )

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
