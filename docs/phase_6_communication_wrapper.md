# Phase 6.2: Communication-Enabled MARL Wrapper

Phase 6.2 adds `CommunicatingMultiAgentRLEnv`, a MARL wrapper that accepts optional structured communication messages during each environment step.

This milestone exposes messages for logging and evaluation only. It does not change `GridWorldEnv`, rewards, Q-learning, training loops, or action selection.

## Why Communication Is Optional

Communication must remain optional so Phase 5 no-communication baselines stay valid.

The wrapper supports both forms:

```python
env.step(actions)
env.step(actions, messages)
```

Calling `step(actions)` keeps the same movement and reward behavior as the existing `MultiAgentRLEnv`. Calling `step(actions, messages)` records structured messages and returns them in `info`.

This makes communication an experimental variable instead of a hidden behavior change.

## Why Messages Do Not Alter Environment Dynamics Yet

In Phase 6.2, messages are observational only.

They do not:

- Change movement rules.
- Resolve collisions.
- Modify blocked moves.
- Change rewards.
- Influence Q-learning updates.
- Select or override actions.
- End episodes.

The underlying `GridWorldEnv.step(...)` still receives simultaneous actions and remains the only source of movement, collision, goal, and step-count behavior.

## Wrapper Design

`CommunicatingMultiAgentRLEnv` extends `MultiAgentRLEnv`.

It adds:

- `step(actions, messages=None)`.
- `get_last_messages()`.
- `format_last_messages()`.
- Communication metadata in the returned `info` dictionary.

The `messages` argument is a mapping from controlled agent id to `CommunicationMessage`.

Example:

```python
messages = {
    "agent_1": CommunicationMessage(
        sender_id="agent_1",
        intended_action=Action.RIGHT,
        target_position=Position(2, 0),
        step_count=0,
    )
}
observations, rewards, done, info = env.step(actions, messages)
```

Communication information is returned under `info["communication"]`:

```python
{
    "last_messages": {
        "agent_1": {
            "sender_id": "agent_1",
            "intended_action": "right",
            "target_position": {"x": 2, "y": 0},
            "blocked": False,
            "waiting": False,
            "priority": 0,
            "step_count": 0,
        }
    },
    "formatted_messages": (
        "step=0 sender=agent_1 action=right target=(2, 0) blocked=False waiting=False priority=0",
    ),
    "communication_count": 1,
    "history_length": 1,
}
```

## Why This Preserves Clean Baselines

The existing `MultiAgentRLEnv` remains unchanged.

Existing training and evaluation code can continue to use `MultiAgentRLEnv` for no-communication experiments. New communication experiments can use `CommunicatingMultiAgentRLEnv` while keeping rewards, actions, observations, and underlying environment rules stable.

This separation allows future comparisons such as:

```text
MultiAgentRLEnv
vs
CommunicatingMultiAgentRLEnv with rule-based messages
```

Because Phase 6.2 does not change rewards or action selection, behavior differences should only appear in later phases that explicitly consume messages.

## Validation Rules

The wrapper validates messages before stepping:

- Message keys must refer to controlled agents.
- Message keys must match `CommunicationMessage.sender_id`.
- `validate_message(...)` must return no errors.

Messages are allowed to be partial. A step may include no messages, one message, or one message per controlled agent.

## How Future Phases Will Use Messages

Phase 6.3 can add rule-based communication policies that produce and consume messages before action selection.

Phase 6.4 can add communication-aware tabular Q-learning by extending observations with simple symbolic message fields.

Phase 6.5 can compare no-communication and communication-enabled experiments using the same scenarios and metrics.

Phase 6.6 can visualize messages alongside positions, actions, rewards, and blocked status.

## Current Limitations

Phase 6.2 intentionally does not add:

- Learned language.
- Communication rewards.
- Communication policies.
- Communication-aware action selection.
- Networking or distributed simulation.
- Natural language generation.
- Transformers.
- LLM communication.
- Deep RL.
- Neural networks.
- PyTorch or TensorFlow.
- RLlib or PettingZoo.
- ROS or physics simulation.
