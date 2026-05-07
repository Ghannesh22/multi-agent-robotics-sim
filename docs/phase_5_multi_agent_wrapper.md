# Phase 5.1 Multi-Agent RL Wrapper

Phase 5.1 adds a lightweight multi-agent RL wrapper around the existing `GridWorldEnv`.

The wrapper is intentionally small. It does not replace the simulator and does not change movement, collision, obstacle, goal, or max-step rules.

## Why MARL Wrappers Differ From Single-Agent Wrappers

`SingleAgentRLEnv` controls one learning agent and returns one observation, one reward, one done flag, and one info dictionary.

`MultiAgentRLEnv` controls multiple learning agents at the same time. It must return per-agent observations and rewards while still stepping the shared environment once per timestep.

This matters because each agent's action can affect other agents. A move can create a blocked move, conflict, timeout, or successful coordination outcome for more than one learner.

## Simultaneous Stepping

The wrapper accepts integer actions per controlled agent:

```python
{
    "agent_1": 3,
    "agent_2": 1,
}
```

It maps each integer to the existing `Action` enum and calls `GridWorldEnv.step()` once with all actions.

Uncontrolled agents use `Action.STAY` for now.

## Decentralized Observations

The wrapper returns one simple coordinate observation per controlled agent:

```python
{
    "agent_1": (agent_x, agent_y, goal_x, goal_y),
    "agent_2": (agent_x, agent_y, goal_x, goal_y),
}
```

This intentionally matches the Phase 3 single-agent observation format.

The first MARL wrapper does not include:

- Global maps.
- Image observations.
- Full joint state.
- Communication messages.
- Learned embeddings.

## Per-Agent Rewards

The wrapper returns one reward per controlled agent:

```python
{
    "agent_1": reward,
    "agent_2": reward,
}
```

The initial reward logic reuses Phase 3 reward values:

- Step penalty.
- Blocked-move penalty.
- Goal reward.
- Timeout penalty.

Goal reward is applied when an agent newly reaches its goal. Timeout penalty is applied to controlled agents that have not reached their goal when the shared episode times out.

## Episode Handling

`step(actions)` returns:

```python
(observations, rewards, done, info)
```

The first wrapper uses one shared `done` flag.

The episode ends when:

- All controlled agents have reached their goals.
- The wrapped `GridWorldEnv` reaches `max_steps` first.

## Info Dictionary

The info dictionary includes top-level episode state and per-agent details:

```python
{
    "controlled_agent_ids": (...),
    "step_count": 1,
    "timeout": False,
    "done": False,
    "agents": {
        "agent_1": {
            "action": "right",
            "blocked": False,
            "blocked_reason": None,
            "reached_goal": False,
            "timeout": False,
            "step_count": 1,
        },
    },
}
```

## Current Limitations

Phase 5.1 does not add:

- Multi-agent training.
- Multi-agent Q-learning updates.
- Shared team rewards.
- Communication learning.
- Centralized critics.
- Parameter sharing.
- Deep RL.
- External MARL frameworks.
- Physics simulation.

The first scope is only the multi-agent reset/step interface.
