# Phase 5 Plan: Multi-Agent Reinforcement Learning

## 1. Phase 5 Vision

Phase 5 transitions the project from single-agent reinforcement learning to multi-agent reinforcement learning.

Phase 4 proved that one learning-controlled agent can train in the grid-world using tabular Q-learning. Phase 5 should build on that foundation by allowing multiple learning agents to train in the same environment.

MARL is harder than single-agent RL because each agent is no longer learning in a stationary world. Other learning agents are changing their behavior at the same time. That means one agent's policy updates can change the transition and reward patterns seen by another agent.

The main challenges are:

- Non-stationary environments caused by multiple learners.
- Agents affecting each other's learning experience.
- Coordination behavior that may or may not emerge from local updates.
- More complex credit assignment.
- Larger joint state and action spaces.

The goal is not to solve all MARL complexity at once. Phase 5 should create a small, readable, reproducible first MARL baseline.

## 2. Why Stay Tabular First

Phase 5 should stay tabular before adding deep MARL.

The current grid-world state and action spaces are still discrete. The existing `Action` enum, grid positions, goals, obstacles, and step events are all inspectable. That makes tabular learning the right first tool for multi-agent experiments.

Tabular MARL is useful first because it is:

- Easier to debug than neural-network MARL.
- Easier to reproduce with seeded randomness.
- Easier to inspect by printing Q-tables.
- Easier to compare against rule-based coordination policies.
- A correct stepping stone before deep MARL frameworks.

Deep MARL should wait until the project has clear multi-agent observations, rewards, metrics, and failure modes.

## 3. Phase 5 Sub-Phases

### Phase 5.0: MARL Planning

Status: this document.

- Define architecture.
- Define training strategy.
- Define reward options.
- Define evaluation metrics.
- Define risks.
- No implementation.

### Phase 5.1: Multi-Agent RL Wrapper

- Extend the RL environment concept to multiple controlled agents.
- Return multi-agent observations.
- Accept multi-agent actions.
- Return per-agent or team rewards.
- Return multi-agent done and info values.
- Preserve `GridWorldEnv` movement and collision rules.
- Phase 5.1 details are documented in `docs/phase_5_multi_agent_wrapper.md`.

### Phase 5.2: Independent Q-Learning Agents

- Use one Q-table per learning agent.
- Keep learning decentralized.
- Select actions for all learning agents each step.
- Step the environment simultaneously.
- Update each Q-table from its own transition.
- Avoid parameter sharing initially.
- Phase 5.2 details are documented in `docs/phase_5_independent_q_learning.md`.

### Phase 5.3: Cooperative Reward Experiments

- Compare shared team rewards.
- Compare individual rewards.
- Compare hybrid reward designs.
- Keep reward values simple and documented.
- Avoid complex reward shaping at first.
- Phase 5.3 details are documented in `docs/phase_5_cooperative_rewards.md`.

### Phase 5.4: MARL Evaluation And Metrics

- Track coordination success rate.
- Track collision or blocked-move frequency.
- Track deadlock frequency.
- Track average completion steps.
- Compare against Phase 2 coordination baselines.
- Phase 5.4 details are documented in `docs/phase_5_marl_evaluation.md`.

### Phase 5.5: MARL Visualization And Demos

- Add multi-agent learning demonstrations.
- Add scenario comparisons.
- Print learning summaries.
- Keep output lightweight and reproducible.
- Reuse console visualization where useful.
- Phase 5.5 details are documented in `docs/phase_5_marl_demos.md`.

### Phase 5.6: Final Phase 5 Polish

- Update documentation.
- Run reproducibility checks.
- Run all tests.
- Stabilize demos and summaries.
- Prepare final Phase 5 commit and tag.

## 4. Proposed MARL Architecture

Possible future files:

- `src/marlsim/rl/multi_agent_env.py`
- `src/marlsim/rl/marl_training.py`
- `src/marlsim/rl/marl_evaluation.py`
- `src/marlsim/demos/marl_demo.py`
- `tests/test_multi_agent_rl.py`

The architecture should reuse existing pieces where possible:

- `GridWorldEnv` remains the source of movement and collision truth.
- `Action` remains the action vocabulary.
- `QLearningAgent` remains the first tabular learner.
- Phase 4 training and evaluation patterns should guide the MARL loop.
- Phase 2 coordination metrics should guide MARL evaluation.

## 5. Initial MARL Strategy

Start with independent Q-learning.

Recommended first strategy:

- One learning agent equals one Q-table.
- No parameter sharing initially.
- Each agent chooses its own action.
- The wrapper submits all actions to `GridWorldEnv.step()` simultaneously.
- Each learner updates after the joint step.
- Start with simple cooperative tasks.
- Start with two agents only.

This is not the most advanced MARL method, but it is the most understandable first milestone.

## 6. Reward Strategy Options

### Shared Team Reward

All agents receive the same reward.

Pros:

- Encourages team success.
- Easier to reason about cooperative tasks.
- Aligns all agents around the same objective.

Cons:

- Harder credit assignment.
- One agent may receive a reward caused mostly by another agent.
- Learning can be noisy when many agents contribute to one outcome.

### Individual Rewards

Each agent receives reward based on its own goal and blocked moves.

Pros:

- Easier credit assignment.
- Similar to the existing single-agent reward.
- Easier to test per-agent behavior.

Cons:

- Agents may learn selfish behavior.
- Coordination may not emerge.
- Agents may block each other while optimizing their own return.

### Collision Or Blocked-Move Penalties

Agents receive penalties for blocked moves, attempted collisions, direct swaps, or repeated conflicts.

Pros:

- Discourages invalid movement.
- Connects directly to existing `GridWorldEnv` step events.
- Useful for measuring coordination quality.

Cons:

- Too much penalty can make agents overly passive.
- Requires careful interpretation because the environment already prevents collisions.

### Timeout Penalties

Agents receive penalties when the episode times out before goals are reached.

Pros:

- Encourages completion.
- Makes deadlock and stalling visibly bad.
- Easy to detect from episode state.

Cons:

- Does not explain which agent caused the timeout.
- Can be too sparse if used alone.

### Coordination Incentives

Agents receive additional reward for group-level behavior, such as all agents reaching goals or avoiding conflicts.

Pros:

- Encourages cooperative outcomes.
- Can make group behavior easier to learn.

Cons:

- Can become complex quickly.
- May hide whether coordination emerged naturally.
- Should wait until simpler reward options are tested.

## 7. State Representation Risks

The main risk is state explosion.

Single-agent Q-learning uses a compact state:

```text
(agent_x, agent_y, goal_x, goal_y)
```

Multi-agent learning may require each agent to observe other agents, other goals, blocked cells, or joint state. This can grow combinatorially as agents and grid size increase.

Risks:

- More agents create many more possible positions.
- Joint action spaces grow quickly.
- Q-tables can become sparse.
- Training can become unstable or slow.

Mitigation strategies:

- Start with two agents only.
- Start with small scenarios.
- Use simple observations.
- Avoid full-grid or image observations.
- Avoid adding communication state initially.
- Track Q-table size and coverage.
- Add complexity only after basic MARL behavior works.

## 8. Evaluation Plan

MARL evaluation should include:

- Success rate.
- Timeout frequency.
- Collision or blocked-move frequency.
- Average episode length.
- Coordination efficiency.

Coordination efficiency can start simple:

- Did all agents reach goals?
- How many steps did completion take?
- How many blocked moves occurred?
- Did agents deadlock?

Evaluation should compare learned MARL behavior against:

- Random agents.
- Greedy agents.
- Shortest-path agents.
- Phase 2 coordination agents.

## 9. Training Safety Rules

Phase 5 should follow these rules:

- No deep RL yet.
- No neural networks yet.
- No communication learning yet.
- No physics simulation.
- No continuous control yet.
- Start with two agents only.
- Start with tiny deterministic scenarios.
- Keep rewards simple and documented.
- Keep metrics reproducible.

## 10. What NOT To Do In Phase 5

Phase 5 should not add:

- PPO.
- DQN.
- Transformers.
- PyTorch.
- TensorFlow.
- Stable-Baselines3.
- RLlib.
- PettingZoo.
- Communication learning.
- PyBullet.
- ROS.
- Physics simulation.

These can be evaluated later after the tabular MARL foundation is stable.

## 11. Success Criteria

Phase 5 should be considered successful when:

- Multiple learning agents can train together.
- Coordination behavior improves over time.
- Metrics remain reproducible.
- Implementation stays understandable.
- Experiments remain lightweight.
- Existing single-agent RL tests still pass.
- `GridWorldEnv` rules remain unchanged.

## 12. Risk Control

Avoid exploding state spaces by:

- Starting with two agents.
- Keeping scenarios small.
- Keeping observations simple.
- Avoiding full-grid and image observations.
- Avoiding communication state at first.

Avoid unstable rewards by:

- Starting with Phase 4 reward components.
- Testing individual and shared reward variants separately.
- Documenting reward values.
- Comparing behavior before and after reward changes.

Avoid overly complex coordination by:

- Starting with independent Q-learning.
- Avoiding negotiation protocols.
- Avoiding communication learning.
- Comparing against Phase 2 rule-based coordination before adding complexity.

Keep training reproducible by:

- Using seeds.
- Keeping episode counts explicit.
- Reporting metrics consistently.
- Keeping demos deterministic where possible.
- Running all tests before milestone completion.

## 13. Next Implementation Step

The next step after Phase 5.0 is Phase 5.1: Multi-Agent RL Wrapper.

Phase 5.1 should define the multi-agent reset/step interface for learning-controlled agents while preserving `GridWorldEnv` rules. It should not add deep RL, neural networks, external MARL frameworks, communication learning, physics, ROS, or multi-agent training yet.
