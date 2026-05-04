# Phase 2 Summary

Phase 2 completed the project's first rule-based multi-agent coordination milestone. It built on the Phase 1 grid-world simulator without changing environment rules or adding learning systems.

## What Phase 2 Achieved

- Created a coordination roadmap in `docs/phase_2_plan.md`.
- Added conflict-aware waiting.
- Added priority-based coordination.
- Added local replanning using the existing BFS planner.
- Added a repeatable comparison runner for current policies and scenarios.
- Added focused tests for coordination behavior and metrics.
- Preserved the existing grid-world environment rules.

## Policies Implemented

### Baseline: `ShortestPathAgent`

The baseline follows the first step of a BFS shortest path to the agent's goal. It is useful for comparison because it plans around static obstacles and current occupied cells, but it does not coordinate around likely simultaneous next-step conflicts.

### Conflict-Aware Waiting: `ConflictAwareWaitingAgent`

This policy checks likely next positions. If the agent's intended move would create a same-cell conflict or direct swap, it waits.

The strategy avoids blocked moves, but it can deadlock when both agents detect the same conflict and both choose `STAY`.

### Priority-Based Coordination: `PriorityBasedCoordinationAgent`

This policy uses a stable priority order. By default, sorted agent ids define priority, so `agent_1` has priority over `agent_2`.

When two agents want the same cell, the higher-priority agent proceeds and the lower-priority agent yields. This fixes the waiting deadlock in `crossing_paths` while keeping the behavior deterministic and easy to test.

For direct swaps, the lower-priority agent yields. The higher-priority agent only proceeds if the target is safe. With unchanged environment rules, moving into a yielding agent's current cell is not safe, so direct swaps may still require waiting or replanning.

### Local Replanning: `LocalReplanningAgent`

This policy tries an alternate BFS path when the preferred next cell is unsafe. It temporarily treats unsafe cells, occupied cells, and other agents' likely next cells as blocked.

Replanning is useful because it can route around a local conflict instead of waiting. It is not always faster: in `crossing_paths`, replanning succeeds without blocked moves but takes one extra step compared with priority coordination.

## Comparison Results

Run the comparison with:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.coordination_comparison
```

Current results:

```text
scenario        | policy     | result  | steps | blocked | blocked reasons | reached goals    | final positions
--------------- | ---------- | ------- | ----- | ------- | --------------- | ---------------- | ----------------------------
simple          | baseline   | success | 6     | 0       | none            | agent_1, agent_2 | agent_1=(6,0), agent_2=(6,4)
simple          | waiting    | success | 6     | 0       | none            | agent_1, agent_2 | agent_1=(6,0), agent_2=(6,4)
simple          | priority   | success | 6     | 0       | none            | agent_1, agent_2 | agent_1=(6,0), agent_2=(6,4)
simple          | replanning | success | 6     | 0       | none            | agent_1, agent_2 | agent_1=(6,0), agent_2=(6,4)
narrow_corridor | baseline   | success | 10    | 0       | none            | agent_1, agent_2 | agent_1=(6,2), agent_2=(0,4)
narrow_corridor | waiting    | success | 10    | 0       | none            | agent_1, agent_2 | agent_1=(6,2), agent_2=(0,4)
narrow_corridor | priority   | success | 10    | 0       | none            | agent_1, agent_2 | agent_1=(6,2), agent_2=(0,4)
narrow_corridor | replanning | success | 10    | 0       | none            | agent_1, agent_2 | agent_1=(6,2), agent_2=(0,4)
crossing_paths  | baseline   | success | 7     | 1       | cell_conflict=1 | agent_1, agent_2 | agent_1=(4,2), agent_2=(2,4)
crossing_paths  | waiting    | failed  | 20    | 0       | none            | none             | agent_1=(1,2), agent_2=(2,1)
crossing_paths  | priority   | success | 7     | 0       | none            | agent_1, agent_2 | agent_1=(4,2), agent_2=(2,4)
crossing_paths  | replanning | success | 8     | 0       | none            | agent_1, agent_2 | agent_1=(4,2), agent_2=(2,4)
```

## Key Lessons

Waiting is a useful first coordination rule, but symmetric waiting can create deadlock. In `crossing_paths`, both agents detect the conflict and both wait forever.

Priority coordination improves results by breaking symmetry. A stable ordering lets one agent proceed while the other yields, removing the deadlock without changing environment rules.

Local replanning is useful because not every conflict should be solved by waiting. When an alternate local path exists, replanning can keep an agent moving while still avoiding unsafe next cells.

## Intentionally Out Of Scope

Phase 2 intentionally did not add:

- Reinforcement learning.
- Gymnasium wrapper.
- PyBullet.
- ROS.
- Physics simulation.
- Complex negotiation protocols.
- LLM agents.
- Neural networks.

These exclusions kept the milestone focused on deterministic, testable coordination.

## Why Phase 3 Comes Next

The project now has stable non-learning baselines, coordination policies, scenarios, tests, and comparison metrics. That makes Phase 3 the right time to introduce a Gymnasium-style reinforcement learning interface.

Phase 3 should not replace the rule-based policies. It should use them as baselines for evaluating future learning-based behavior.
