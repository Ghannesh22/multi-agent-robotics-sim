from __future__ import annotations

import unittest

from marlsim.communication import CommunicationMessage
from marlsim.core.actions import Action
from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import CommunicatingMultiAgentRLEnv


class CommunicatingMultiAgentRLEnvTests(unittest.TestCase):
    def test_step_accepts_messages_and_preserves_environment_behavior(self) -> None:
        env = CommunicatingMultiAgentRLEnv(self._two_agent_env(), ("agent_1", "agent_2"))
        env.reset()

        observations, rewards, done, info = env.step(
            {"agent_1": 3, "agent_2": 3},
            {
                "agent_1": CommunicationMessage(
                    sender_id="agent_1",
                    intended_action=Action.RIGHT,
                    target_position=Position(2, 0),
                    step_count=0,
                ),
                "agent_2": CommunicationMessage(
                    sender_id="agent_2",
                    intended_action=Action.RIGHT,
                    target_position=Position(2, 1),
                    step_count=0,
                ),
            },
        )

        self.assertEqual(
            observations,
            {
                "agent_1": (1, 0, 2, 0),
                "agent_2": (1, 1, 2, 1),
            },
        )
        self.assertEqual(rewards, {"agent_1": -0.1, "agent_2": -0.1})
        self.assertFalse(done)
        self.assertFalse(info["agents"]["agent_1"]["blocked"])
        self.assertFalse(info["agents"]["agent_2"]["blocked"])

    def test_step_works_without_messages(self) -> None:
        env = CommunicatingMultiAgentRLEnv(self._two_agent_env(), ("agent_1", "agent_2"))
        env.reset()

        observations, rewards, done, info = env.step({"agent_1": 3, "agent_2": 3})

        self.assertEqual(observations["agent_1"], (1, 0, 2, 0))
        self.assertEqual(rewards, {"agent_1": -0.1, "agent_2": -0.1})
        self.assertFalse(done)
        self.assertEqual(info["communication"]["last_messages"], {})
        self.assertEqual(info["communication"]["formatted_messages"], ())
        self.assertEqual(info["communication"]["communication_count"], 0)

    def test_communication_info_is_exposed(self) -> None:
        env = CommunicatingMultiAgentRLEnv(self._two_agent_env(), ("agent_1", "agent_2"))
        env.reset()
        messages = {
            "agent_1": CommunicationMessage(
                sender_id="agent_1",
                intended_action=Action.STAY,
                target_position=Position(2, 0),
                waiting=True,
                priority=1,
                step_count=4,
            )
        }

        _, _, _, info = env.step({"agent_1": 4, "agent_2": 3}, messages)

        self.assertEqual(
            info["communication"]["last_messages"],
            {
                "agent_1": {
                    "sender_id": "agent_1",
                    "intended_action": "stay",
                    "target_position": {"x": 2, "y": 0},
                    "blocked": False,
                    "waiting": True,
                    "priority": 1,
                    "step_count": 4,
                }
            },
        )
        self.assertEqual(
            info["communication"]["formatted_messages"],
            (
                "step=4 sender=agent_1 action=stay target=(2, 0) "
                "blocked=False waiting=True priority=1",
            ),
        )
        self.assertEqual(info["communication"]["communication_count"], 1)
        self.assertEqual(info["communication"]["history_length"], 1)

    def test_message_logging_helpers_return_last_messages(self) -> None:
        env = CommunicatingMultiAgentRLEnv(self._two_agent_env(), ("agent_1", "agent_2"))
        env.reset()
        message = CommunicationMessage(
            sender_id="agent_2",
            intended_action=Action.LEFT,
            blocked=True,
            step_count=2,
        )

        env.step({"agent_1": 3, "agent_2": 4}, {"agent_2": message})

        self.assertEqual(env.get_last_messages(), {"agent_2": message})
        self.assertEqual(
            env.format_last_messages(),
            (
                "step=2 sender=agent_2 action=left target=none "
                "blocked=True waiting=False priority=0",
            ),
        )

    def test_reset_clears_message_history(self) -> None:
        env = CommunicatingMultiAgentRLEnv(self._two_agent_env(), ("agent_1", "agent_2"))
        env.reset()
        env.step(
            {"agent_1": 3, "agent_2": 3},
            {"agent_1": CommunicationMessage(sender_id="agent_1", step_count=0)},
        )

        env.reset()

        self.assertEqual(env.get_last_messages(), {})
        self.assertEqual(env.format_last_messages(), ())

    def test_uncontrolled_agent_messages_are_rejected(self) -> None:
        env = CommunicatingMultiAgentRLEnv(self._two_agent_env(), ("agent_1", "agent_2"))
        env.reset()

        with self.assertRaisesRegex(ValueError, "Unexpected messages"):
            env.step(
                {"agent_1": 3, "agent_2": 3},
                {"agent_3": CommunicationMessage(sender_id="agent_3")},
            )

    def test_message_key_must_match_sender_id(self) -> None:
        env = CommunicatingMultiAgentRLEnv(self._two_agent_env(), ("agent_1", "agent_2"))
        env.reset()

        with self.assertRaisesRegex(ValueError, "must match sender_id"):
            env.step(
                {"agent_1": 3, "agent_2": 3},
                {"agent_1": CommunicationMessage(sender_id="agent_2")},
            )

    def test_invalid_messages_are_rejected(self) -> None:
        env = CommunicatingMultiAgentRLEnv(self._two_agent_env(), ("agent_1", "agent_2"))
        env.reset()

        with self.assertRaisesRegex(ValueError, "Invalid message"):
            env.step(
                {"agent_1": 3, "agent_2": 3},
                {"agent_1": CommunicationMessage(sender_id="agent_1", priority=-1)},
            )

    @staticmethod
    def _two_agent_env() -> GridWorldEnv:
        return GridWorldEnv(
            GridWorldConfig(
                width=3,
                height=2,
                agents=(
                    AgentState("agent_1", Position(0, 0), Position(2, 0)),
                    AgentState("agent_2", Position(0, 1), Position(2, 1)),
                ),
                max_steps=5,
            )
        )


if __name__ == "__main__":
    unittest.main()
