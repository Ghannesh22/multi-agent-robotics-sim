# Phase 2 Plan: Rule-Based Coordination

## 1. Phase 2 Vision

Phase 2 improves how multiple rule-based agents share the same grid-world without adding learning or robotics middleware.

In this project, rule-based coordination means agents still follow explicit, understandable rules written by the developer. They do not learn from rewards, train policies, or use neural networks. Instead, they use simple information that already exists in the simulator, such as current positions, goals, blocked cells, proposed moves, and shortest paths, to make less selfish movement decisions.

The goal is to make the agents better at moving together through crossings, corridors, and bottlenecks while keeping the environment rules unchanged and beginner-friendly.

## 2. Current Limitation

Phase 1 has a stable grid-world, collision prevention, scenarios, logging, BFS path planning, `GreedyGoalAgent`, and `ShortestPathAgent`. These pieces are useful baselines, but the agents still mostly choose actions from their own point of view.

That means agents can still:

- Try to enter the same cell at the same time.
- Repeatedly attempt blocked moves.
- Block each other in narrow passages.
- Choose shortest individual paths that create group-level conflicts.
- Fail to yield when another agent has a better claim to a bottleneck.

The environment prevents illegal collisions, but preventing invalid moves is not the same as coordinating. Phase 2 should add simple agent-side and coordination-side rules that reduce avoidable blocking before the environment has to reject moves.

## 3. Phase 2 Sub-Phases

### Phase 2.0: Planning Only

Status: this document.

- Create the coordination roadmap.
- Define coordination strategies.
- Define tests and success criteria.
- Keep this phase documentation-only.

### Phase 2.1: Conflict-Aware Waiting

- Let agents detect possible conflicts before committing to a move.
- Prefer waiting when a proposed move is likely to be blocked.
- Keep the logic simple and local.
- Avoid changing environment collision rules.
- Add focused tests for wait behavior in known conflict scenarios.

### Phase 2.2: Priority-Based Coordination

- Move agents according to a stable priority order, such as agent id order.
- Let lower-priority agents yield when two agents want the same constrained cell.
- Compare the strategy with the current greedy and shortest-path baselines.
- Keep priority deterministic so tests are easy to understand.

### Phase 2.3: Local Replanning

- If a path is blocked, let the agent try an alternate local route.
- Reuse the existing BFS planner where possible.
- Treat other agents or likely next cells as temporary blocked cells when planning.
- Avoid changing environment rules, action definitions, or grid semantics.

### Phase 2.4: Coordination Metrics

- Compare scenarios using total steps, blocked moves, success, and failure.
- Use the existing experiment logger where possible.
- Add a simple comparison table or report for baseline vs coordination strategies.
- Focus first on `crossing_paths`, then check `simple` and `narrow_corridor` for regressions.

### Phase 2.5: Final Phase 2 Polish

- Update README and project documentation.
- Run the full test suite.
- Run scenario comparisons.
- Summarize Phase 2 results.
- Prepare a Phase 2 completion tag.

## 4. Coordination Strategies

### Wait-On-Conflict

An agent checks whether its intended next cell is likely to cause a conflict. If the move appears unsafe, it chooses to stay in place for that step. This is the simplest coordination behavior because it does not require finding a new route.

### Priority Yielding

Agents use a stable priority order to resolve contested moves. For example, agent 1 may have priority over agent 2. If both agents want the same bottleneck cell, the lower-priority agent waits while the higher-priority agent moves. This avoids repeated indecision and keeps behavior deterministic.

### Local Replanning

When an agent's preferred path is blocked, it asks for another short path around the temporary blockage. The existing BFS planner should be reused if possible. This is more flexible than waiting, but should stay local and simple.

### Avoid Occupied Next Cell

Agents avoid choosing a next cell that is currently occupied by another agent unless that other agent is expected to move away safely. This reduces attempts to push into blocked positions.

### Avoid Bottleneck Conflicts

Agents treat narrow corridors, crossings, and one-cell passages as places where yielding may be needed. A beginner-friendly version can start by recognizing contested next cells, direct swaps, or cells with very few exits instead of building a complex traffic system.

## 5. What Should Not Be Done In Phase 2

Phase 2 should not add:

- Reinforcement learning.
- A Gymnasium wrapper.
- PyBullet.
- ROS.
- Physics simulation.
- Complex negotiation protocols.
- LLM agents.
- Neural networks.

These topics belong to later phases, after deterministic coordination is easy to run, test, and compare.

## 6. Success Criteria

Phase 2 should be considered successful when:

- Coordinated agents produce fewer blocked moves than the current baseline in `crossing_paths`.
- The `simple` scenario does not regress.
- All tests pass.
- Coordination logic remains understandable to a beginner reading the code.
- Environment rules are unchanged.
- Metrics are clear enough to compare baseline and coordinated behavior.

## 7. Suggested Files For Future Work

These are possible files for later implementation phases. They should not be created during Phase 2.0 unless the implementation phase needs them.

- `src/marlsim/agents/coordination.py`
- `src/marlsim/coordination/`
- `tests/test_coordination.py`
- `docs/phase_2_results.md`

## 8. Risk Control

The main risk in Phase 2 is overengineering. Coordination can quickly become a large planning, scheduling, or negotiation problem. This project should avoid that during Phase 2.

Risk controls:

- Add one coordination strategy at a time.
- Keep each strategy deterministic.
- Prefer small functions and explicit rules over frameworks.
- Reuse existing BFS before adding new planning algorithms.
- Keep environment mechanics stable.
- Use scenario metrics to justify each added behavior.
- Stop when blocked moves improve and the code remains explainable.

## 9. Next Implementation Step

The next step after Phase 2.0 is Phase 2.1: Conflict-Aware Waiting.

Phase 2.1 should introduce the smallest useful coordination behavior: detect likely move conflicts and wait instead of repeatedly forcing blocked moves. It should include focused tests and scenario comparisons, but it should still avoid reinforcement learning, physics, ROS, and any major environment redesign.
