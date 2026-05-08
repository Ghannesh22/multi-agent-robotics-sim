from __future__ import annotations

import unittest

from marlsim.communication import CommunicationMessage
from marlsim.core.actions import Action
from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import (
    CommunicationTrajectoryStep,
    CommunicatingMultiAgentRLEnv,
    format_communication_rollout,
    format_message_timeline,
    rollout_rule_based_communication_trace,
    summarize_communication_episode,
    visualize_conflict_predictions,
    visualize_waiting_behavior,
)
from marlsim.demos.communication_visualization_demo import build_demo_output


class CommunicationVisualizationTests(unittest.TestCase):
    def test_rollout_format_includes_messages_actions_rewards_and_conflicts(self) -> None:
        trajectory = (
            CommunicationTrajectoryStep(
                step_number=2,
                mode_name="case",
                positions={"agent_1": Position(1, 0)},
                actions={"agent_1": "right"},
                rewards={"agent_1": -0.1},
                messages={
                    "agent_1": CommunicationMessage(
                        sender_id="agent_1",
                        intended_action=Action.RIGHT,
                        target_position=Position(2, 0),
                        priority=0,
                        step_count=1,
                    )
                },
                formatted_messages=(
                    "step=1 sender=agent_1 action=right target=(2, 0) "
                    "blocked=False waiting=False priority=0",
                ),
                waiting={"agent_1": False},
                priorities={"agent_1": 0},
                predicted_conflicts={"agent_1": True},
                blocked={"agent_1": False},
                reached_goals={"agent_1": False},
                done=False,
                timeout=False,
            ),
        )

        text = format_communication_rollout(trajectory)

        self.assertIn("Communication Rollout: case", text)
        self.assertIn("STEP 2", text)
        self.assertIn("action=right", text)
        self.assertIn("reward=-0.10", text)
        self.assertIn("predicted_conflict=True", text)
        self.assertIn("sent=\"step=1 sender=agent_1", text)

    def test_message_timeline_formats_empty_and_present_messages(self) -> None:
        empty = CommunicationTrajectoryStep(
            step_number=1,
            mode_name="case",
            positions={"agent_1": Position(0, 0)},
            actions={"agent_1": "stay"},
            rewards={"agent_1": -0.1},
            messages={},
            formatted_messages=(),
            waiting={"agent_1": False},
            priorities={"agent_1": -1},
            predicted_conflicts={"agent_1": False},
            blocked={"agent_1": False},
            reached_goals={"agent_1": False},
            done=False,
            timeout=False,
        )

        text = format_message_timeline((empty,))

        self.assertIn("Message Timeline", text)
        self.assertIn("STEP 1: no messages", text)

    def test_summary_counts_waiting_conflicts_messages_and_blocked_moves(self) -> None:
        trajectory = (
            CommunicationTrajectoryStep(
                step_number=1,
                mode_name="case",
                positions={"agent_1": Position(0, 0), "agent_2": Position(1, 0)},
                actions={"agent_1": "stay", "agent_2": "left"},
                rewards={"agent_1": -0.1, "agent_2": -1.1},
                messages={
                    "agent_1": CommunicationMessage(
                        sender_id="agent_1",
                        intended_action=Action.STAY,
                        waiting=True,
                    )
                },
                formatted_messages=("step=0 sender=agent_1 action=stay target=none blocked=False waiting=True priority=0",),
                waiting={"agent_1": True, "agent_2": False},
                priorities={"agent_1": 0, "agent_2": -1},
                predicted_conflicts={"agent_1": False, "agent_2": True},
                blocked={"agent_1": False, "agent_2": True},
                reached_goals={"agent_1": False, "agent_2": False},
                done=True,
                timeout=True,
            ),
        )

        summary = summarize_communication_episode("case", trajectory)

        self.assertIn("case: failure in 1 steps", summary)
        self.assertIn("blocked=1", summary)
        self.assertIn("messages=1", summary)
        self.assertIn("predicted_conflicts=1", summary)
        self.assertIn("waiting=1", summary)

    def test_waiting_and_conflict_traces_are_readable(self) -> None:
        trajectory = (
            CommunicationTrajectoryStep(
                step_number=1,
                mode_name="case",
                positions={"agent_1": Position(0, 0)},
                actions={"agent_1": "stay"},
                rewards={"agent_1": -0.1},
                messages={},
                formatted_messages=(),
                waiting={"agent_1": True},
                priorities={"agent_1": 0},
                predicted_conflicts={"agent_1": True},
                blocked={"agent_1": False},
                reached_goals={"agent_1": False},
                done=False,
                timeout=False,
            ),
        )

        self.assertIn("STEP 1: waiting=agent_1", visualize_waiting_behavior(trajectory))
        self.assertIn(
            "STEP 1: predicted=agent_1 blocked=none",
            visualize_conflict_predictions(trajectory),
        )

    def test_rule_based_rollout_records_communication_details(self) -> None:
        env = CommunicatingMultiAgentRLEnv(_parallel_env(), ("agent_1", "agent_2"))

        trajectory = rollout_rule_based_communication_trace(env)

        self.assertTrue(trajectory)
        self.assertTrue(trajectory[0].messages)
        self.assertIn("agent_1", trajectory[0].priorities)
        self.assertIn("agent_1", trajectory[0].predicted_conflicts)

    def test_demo_output_contains_all_modes_and_observation_note(self) -> None:
        output = build_demo_output()

        self.assertIn("independent MARL", output)
        self.assertIn("rule-based communication", output)
        self.assertIn("communication-aware Q-learning", output)
        self.assertIn("Observation note", output)


def _parallel_env() -> GridWorldEnv:
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
