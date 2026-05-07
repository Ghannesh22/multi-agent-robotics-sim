# Phase 4.1 Q-Learning Agent

Phase 4.1 adds the project's first learning component: a simple tabular Q-learning agent.

This is not a training loop yet. The agent stores Q-values, chooses actions with epsilon-greedy exploration, and updates one state-action value from one transition.

## What Q-Learning Is

Q-learning is a reinforcement learning method that estimates how good each action is in each state.

The learned value is called a Q-value:

```text
Q(state, action)
```

A higher Q-value means the agent currently believes that action is better in that state.

After each transition, the agent updates the value for the action it just took:

```text
Q(s, a) = Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))
```

If the transition ends the episode, the future-value term is `0`.

## Q-Table Structure

`QLearningAgent` uses the Phase 3 observation tuple directly as the state key:

```text
(agent_x, agent_y, goal_x, goal_y)
```

The Q-table is a dictionary:

```text
q_table[state][action] -> float
```

For the current wrapper, actions are integer ids:

| Integer | Action |
| ------- | ------ |
| `0` | up |
| `1` | down |
| `2` | left |
| `3` | right |
| `4` | stay |

Unseen Q-values default to `0.0`.

## Defaults

The Phase 4.1 defaults are:

- `learning_rate = 0.1`
- `discount_factor = 0.95`
- `epsilon = 0.1`

These are intentionally simple starting values. They can be tuned later after the training loop and evaluation metrics exist.

## Epsilon-Greedy Action Selection

`choose_action(state)` uses epsilon-greedy selection:

- With probability `epsilon`, choose a random valid action.
- Otherwise, choose the action with the highest Q-value.

When multiple actions tie for the highest Q-value, the current implementation picks the lowest action index. This deterministic tie-break makes tests and early debugging easier.

## Why Tabular RL Comes First

Tabular Q-learning is used first because:

- The Phase 3 state is a small discrete tuple.
- The Phase 3 action interface is five discrete integers.
- Q-values can be printed and inspected.
- Unit tests can verify the update rule directly.
- It does not require PyTorch, TensorFlow, Stable-Baselines3, or Gymnasium.

This keeps the first learning implementation understandable before adding any deep RL infrastructure.

## Limitations Of Tabular RL

Tabular Q-learning does not scale well to very large state spaces.

It is not ideal for:

- Image observations.
- Continuous states.
- Continuous actions.
- Large multi-agent state spaces.
- Complex physics simulation.
- Learned communication protocols.

Those are intentionally out of scope for Phase 4.1. The immediate goal is to prove that simple learning can update values correctly against the existing RL wrapper.
