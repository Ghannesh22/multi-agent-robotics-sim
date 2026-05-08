# Phase 6 Summary

Phase 6 completed the communication-between-agents milestone.

The milestone explored whether explicit symbolic communication can improve multi-agent coordination after Phase 5 exposed a shared-reward coordination failure. It kept the project lightweight, deterministic where possible, and fully tabular. It did not change `GridWorldEnv` movement rules, reward logic, or the existing no-communication MARL baseline.

## What Phase 6 Achieved

- Added the Phase 6 communication plan in `docs/phase_6_plan.md`.
- Added structured communication messages with `CommunicationMessage`.
- Added message serialization, formatting, and validation helpers.
- Added `CommunicatingMultiAgentRLEnv`.
- Added optional `step(actions, messages)` support.
- Added communication logging through `info["communication"]`.
- Added rule-based communication policies.
- Added communication-aware tabular Q-learning.
- Added communication evaluation and comparison metrics.
- Added readable communication trajectory visualization.
- Added demos for communication behavior, communication-aware learning, communication evaluation, and communication visualization.
- Added tests for message design, wrapper behavior, rule-based communication, communication-aware Q-learning, evaluation, and visualization.

## How Communication Messages Work

`CommunicationMessage` is a small frozen dataclass.

It contains:

- `sender_id`: the agent that produced the message.
- `intended_action`: the symbolic action the agent expects to take.
- `target_position`: the current target or goal the agent is pursuing.
- `blocked`: whether the agent is reporting a blocked state.
- `waiting`: whether the agent intends to wait.
- `priority`: a simple non-negative priority value.
- `step_count`: the environment step associated with the message.

Messages are deliberately symbolic and structured. They do not use natural language, learned language, neural encoders, networking, or distributed systems.

## How The Communicating MARL Wrapper Works

`CommunicatingMultiAgentRLEnv` extends the Phase 5 `MultiAgentRLEnv`.

It supports:

```python
env.step(actions)
env.step(actions, messages)
```

Communication remains optional. Calling `step(actions)` preserves no-communication behavior. Calling `step(actions, messages)` records messages and returns communication metadata in `info`.

Communication does not alter:

- Movement rules.
- Collision rules.
- Rewards.
- Episode termination.
- Q-learning updates.
- Environment physics.

The wrapped `GridWorldEnv` remains the source of truth for simultaneous movement, collisions, goals, and timeouts.

## Rule-Based Communication Results

Phase 6 added:

- `CommunicationAwareWaitingAgent`
- `CommunicationAwarePriorityAgent`

These policies build messages before action selection and then choose actions after inspecting messages from other agents.

The rule-based priority policy can avoid obvious same-cell conflicts by allowing the higher-priority agent to proceed while the lower-priority agent waits. In the communication demo, the no-communication shortest-path baseline produced blocked moves in a same-target setup, while rule-based communication avoided the immediate conflict.

This demonstrates that explicit communication can improve coordination when the communication signal directly matches the coordination problem.

## Communication-Aware Q-Learning Results

Phase 6 also added `CommunicationAwareQLearningAgent` and `train_communication_q_learning(...)`.

The original Phase 5 MARL state:

```text
(agent_x, agent_y, goal_x, goal_y)
```

was extended with symbolic message-derived features:

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

The learning architecture stayed independent and tabular:

- One Q-table per agent.
- No centralized critic.
- No parameter sharing.
- No replay buffer.
- No target network.
- No neural network.

In the first small comparison, communication-aware Q-learning performed worse than independent MARL:

```text
training      | success | timeouts | blocked | avg reward | avg steps
------------- | ------- | -------- | ------- | ---------- | ---------
independent   | 0.76    | 0.24     | 51      | 14.14      | 3.22
communication | 0.56    | 0.44     | 65      | 10.45      | 3.74
```

This is a useful result. It shows that adding communication information to the state does not automatically improve learning.

## Evaluation Findings

Phase 6.5 compared three modes:

- Independent MARL without communication.
- Rule-based communication coordination.
- Communication-aware Q-learning.

The evaluation demo reported:

```text
mode                     | success | timeouts | blocked | avg reward | avg steps | messages | conflict freq | waiting freq
------------------------ | ------- | -------- | ------- | ---------- | --------- | -------- | ------------- | ------------
independent              | 0.76    | 0.24     | 51      | 14.14      | 3.22      | 0        | 0.00          | 0.00
rule-based communication | 1.00    | 0.00     | 0       | 19.60      | 2.00      | 200      | 0.00          | 0.00
communication q-learning | 0.52    | 0.48     | 81      | 9.83       | 3.74      | 374      | 0.00          | 0.19
```

The strongest result is not that communication always helps. The stronger lesson is that communication must be useful to the policy consuming it.

Rule-based communication performed well in the tiny deterministic scenario because its hand-written logic directly matched the conflict structure. Communication-aware Q-learning had more information, but also had a larger state space and had to learn how to use that information from sparse experience.

## Why Symbolic Communication Sometimes Hurt Learning

Tabular Q-learning depends on repeated visits to the same state-action pairs.

Adding communication features expands the state representation. Even simple boolean and priority fields can split experience across more states. With a small training budget, this can make learning slower or noisier.

Communication also interacts with credit assignment. If an agent receives a bad outcome, it must infer whether the issue came from:

- Its own action.
- Another agent's action.
- The message signal.
- The other agent's response to a message.
- The reward mode.

This is a harder learning problem than the original coordinate-only state.

## What Phase 6 Teaches

Phase 6 shows:

- Explicit communication can prevent some coordination failures.
- Structured symbolic messages are easy to inspect and test.
- Communication should remain optional so clean baselines stay available.
- Rule-based communication can outperform learned communication in tiny deterministic settings.
- Communication-aware Q-learning may underperform when state-space expansion outweighs useful signal.
- Better communication requires careful evaluation, not just more information.

## Visualization

Phase 6.6 added text trajectory visualization for communication behavior.

The visualization shows:

- Positions.
- Actions.
- Rewards.
- Sent messages.
- Waiting flags.
- Priority values.
- Predicted conflicts.
- Blocked moves.
- Done and timeout state.

This makes it easier to debug why a communication policy waited, yielded, moved, or failed.

## Why Phase 7 Physics-Based Simulation Comes Next

The grid-world now has:

- Rule-based coordination.
- Single-agent tabular RL.
- Multi-agent tabular RL.
- Shared and individual reward experiments.
- Explicit symbolic communication.
- Communication-aware learning.
- Evaluation and visualization tools.

That is enough grid-world infrastructure to support the next project stage.

Phase 7 should explore a physics-based simulation upgrade. The purpose is not to abandon the grid-world, but to test whether the coordination concepts still make sense when movement becomes less abstract and agents have more realistic state, motion, and sensing constraints.

## Intentionally Out Of Scope

Phase 6 intentionally did not add:

- Natural language communication.
- LLM agents.
- Learned language.
- Transformers.
- Deep RL.
- Neural communication.
- Centralized critics.
- Graph neural networks.
- Parameter sharing.
- Replay buffers.
- Target networks.
- PPO.
- DQN.
- MAPPO.
- QMIX.
- PyTorch.
- TensorFlow.
- RLlib.
- PettingZoo.
- ROS.
- Physics simulation.
- Networking or distributed simulation.
- Pygame.
- Matplotlib.
- Web dashboards.

Phase 6 stays focused on lightweight, inspectable symbolic communication in the existing grid-world.

## Final Status

Phase 6 is complete.

The next milestone is Phase 7: physics-based simulation upgrade.
