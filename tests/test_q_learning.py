from __future__ import annotations

import unittest

from marlsim.rl import QLearningAgent


class QLearningAgentTests(unittest.TestCase):
    def test_defaults_are_simple_phase_4_values(self) -> None:
        agent = QLearningAgent(action_size=5)

        self.assertEqual(agent.action_size, 5)
        self.assertEqual(agent.learning_rate, 0.1)
        self.assertEqual(agent.discount_factor, 0.95)
        self.assertEqual(agent.epsilon, 0.1)

    def test_unseen_q_values_default_to_zero(self) -> None:
        agent = QLearningAgent(action_size=5)

        self.assertEqual(agent.get_q_value((0, 0, 2, 0), 3), 0.0)

    def test_epsilon_greedy_exploitation_returns_best_valid_action(self) -> None:
        agent = QLearningAgent(action_size=5, epsilon=0.0)
        state = (0, 0, 2, 0)
        agent.update(state, 3, reward=2.0, next_state=state, done=True)

        self.assertEqual(agent.choose_action(state), 3)

    def test_epsilon_greedy_exploration_returns_valid_actions(self) -> None:
        agent = QLearningAgent(action_size=5, epsilon=1.0, seed=7)
        state = (0, 0, 2, 0)

        actions = {agent.choose_action(state) for _ in range(20)}

        self.assertTrue(actions)
        self.assertTrue(all(0 <= action < 5 for action in actions))

    def test_unseen_exploitation_breaks_ties_with_lowest_action(self) -> None:
        agent = QLearningAgent(action_size=5, epsilon=0.0)

        self.assertEqual(agent.choose_action((0, 0, 2, 0)), 0)

    def test_update_changes_q_value(self) -> None:
        agent = QLearningAgent(action_size=5, learning_rate=0.1, discount_factor=0.95)

        updated = agent.update(
            state=(0, 0, 2, 0),
            action=3,
            reward=-0.1,
            next_state=(1, 0, 2, 0),
            done=False,
        )

        self.assertAlmostEqual(updated, -0.01)
        self.assertAlmostEqual(agent.get_q_value((0, 0, 2, 0), 3), -0.01)

    def test_update_uses_discounted_future_value_for_non_terminal_step(self) -> None:
        agent = QLearningAgent(action_size=5, learning_rate=1.0, discount_factor=0.5)
        state = (0, 0, 2, 0)
        next_state = (1, 0, 2, 0)
        agent.update(next_state, 3, reward=4.0, next_state=next_state, done=True)

        updated = agent.update(
            state=state,
            action=3,
            reward=2.0,
            next_state=next_state,
            done=False,
        )

        self.assertEqual(updated, 4.0)

    def test_terminal_update_ignores_future_value(self) -> None:
        agent = QLearningAgent(action_size=5, learning_rate=1.0, discount_factor=0.95)
        state = (1, 0, 2, 0)
        next_state = (2, 0, 2, 0)
        agent.update(next_state, 0, reward=100.0, next_state=next_state, done=True)

        updated = agent.update(
            state=state,
            action=3,
            reward=9.9,
            next_state=next_state,
            done=True,
        )

        self.assertEqual(updated, 9.9)

    def test_invalid_action_is_rejected(self) -> None:
        agent = QLearningAgent(action_size=5)

        with self.assertRaisesRegex(ValueError, "between 0 and 4"):
            agent.get_q_value((0, 0, 2, 0), 5)


if __name__ == "__main__":
    unittest.main()
