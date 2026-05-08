# Phase 6.4: Communication-Aware Q-Learning Agents

Phase 6.4 introduces communication into independent tabular Q-learning observations.

This milestone keeps the learning architecture simple: one tabular Q-learning agent per controlled agent, no centralized critic, no parameter sharing, no replay buffer, no target network, and no neural network.

## Communication-Enhanced State Representation

The Phase 5 MARL observation was:

```text
(agent_x, agent_y, goal_x, goal_y)
```

Phase 6.4 appends four symbolic communication features:

```text
(
    agent_x,
    agent_y,
    goal_x,
    goal_y,
    other_agent_waiting,
    other_agent_priority,
    predicted_conflict,
    intended_same_target,
)
```

The added features are discrete integers:

- `other_agent_waiting`: `1` when any other communicated agent is waiting, otherwise `0`.
- `other_agent_priority`: the lowest priority value announced by another agent, or `-1` when no other message is present.
- `predicted_conflict`: `1` when current messages predict a same-cell conflict, direct swap, or movement into a waiting agent, otherwise `0`.
- `intended_same_target`: `1` when another agent declares the controlled agent's goal as its target, otherwise `0`.

Example:

```text
(2, 1, 1, 1, 0, 0, 1, 1)
```

This means the agent is at `(2, 1)`, its goal is `(1, 1)`, no other agent is waiting, another agent has priority `0`, a conflict is predicted, and another agent declared the same target.

## Why Symbolic Features Are Used

The communication features stay symbolic because the project is still focused on lightweight, inspectable MARL.

Symbolic features are:

- Easy to print and test.
- Small enough for tabular Q-learning.
- Derived from existing structured messages.
- Suitable for deterministic demos.
- Easier to compare against no-communication baselines.

This avoids hidden neural encoders or learned language protocols.

## Why Tabular Learning Is Still Feasible

The state space grows from four integers to eight integers, but the extra values are deliberately small:

- Booleans are encoded as `0` or `1`.
- Priority is a small integer from the known agent ordering.
- No full-grid observation is added.
- No message history sequence is added.
- No raw text is added.

This keeps Q-tables sparse but still understandable for the current tiny scenarios.

## Training Design

`train_communication_q_learning(...)` trains one `CommunicationAwareQLearningAgent` per controlled agent.

For each step:

1. Agents exchange deterministic `CommunicationMessage` objects.
2. The base MARL observations are extended with symbolic communication features.
3. Each agent chooses an action from its own Q-table.
4. `CommunicatingMultiAgentRLEnv.step(actions, messages)` applies simultaneous actions and logs messages.
5. Each Q-table updates normally from `(state, action, reward, next_state, done)`.

Rewards still come from the existing MARL reward logic. Communication does not add new rewards.

## Demo

Run:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.communication_q_learning_demo
```

The demo compares:

- Independent MARL without communication.
- Communication-aware tabular Q-learning.

It prints success rate, timeout frequency, blocked moves, average reward, and average episode length.

## Limits Of This Approach

The communication-aware state is still hand-designed. It may help expose useful coordination signals, but it does not learn what to communicate.

Limitations:

- Messages are deterministic and symbolic.
- Communication features are manually selected.
- The Q-table can still grow as scenarios become larger.
- The approach only considers current-step messages.
- There is no centralized critic or joint action-value function.
- There is no learned language or neural communication channel.

## Relationship To Future Deep MARL

This phase is a stepping stone toward richer MARL systems.

If the symbolic communication features prove useful, later work could compare them against learned communication, graph-based observations, centralized critics, or deep MARL methods. Those remain out of scope for this project phase.

Phase 6.4 intentionally does not add:

- Deep RL.
- Learned language.
- Transformers.
- Neural communication.
- Centralized training.
- Graph neural networks.
- PyTorch or TensorFlow.
- RLlib or PettingZoo.
- PPO, DQN, MAPPO, or QMIX.
- ROS or physics simulation.
