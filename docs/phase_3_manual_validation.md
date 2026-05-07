# Phase 3.6 Manual RL-Environment Validation

Phase 3.6 adds a small scripted validation demo for the single-agent RL wrapper.

The demo is intentionally not training. It only runs fixed actions through `SingleAgentRLEnv` and prints the resulting observation, reward, done flag, and info dictionary.

## Run The Demo

From the project root:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.rl_env_validation
```

## What The Demo Checks

The validation script shows:

- Reset output.
- Normal movement step.
- Blocked movement step.
- Goal-reaching episode.
- Timeout episode.

It also prints the current observation names, action names, and reward values so the wrapper contract is visible before any learning code exists.

## Expected Behavior

Normal movement should move the controlled agent and return the step penalty.

Blocked movement should keep the controlled agent in place and add the blocked-move penalty.

Goal completion should set `done=True`, `reached_goal=True`, and `timeout=False`.

Timeout should set `done=True`, `reached_goal=False`, and `timeout=True`.

## What This Does Not Do

This validation does not add:

- RL training.
- Gymnasium dependency.
- Stable-Baselines3.
- Neural networks.
- PyBullet.
- ROS.
- Physics simulation.
- Multi-agent RL.

The purpose is only to prove the wrapper interface behaves correctly before Phase 4 training work begins.
