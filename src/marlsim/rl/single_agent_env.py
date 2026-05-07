from __future__ import annotations

from dataclasses import dataclass

from marlsim.core.actions import Action
from marlsim.core.environment import AgentStepEvent, GridWorldEnv


Observation = tuple[int, int, int, int]
ObservationNames = tuple[str, str, str, str]
ActionNames = tuple[str, str, str, str, str]
RewardValues = dict[str, float]
OBSERVATION_NAMES: ObservationNames = ("agent_x", "agent_y", "goal_x", "goal_y")
ACTION_NAMES: ActionNames = ("up", "down", "left", "right", "stay")


@dataclass(frozen=True)
class RewardConfig:
    """Small documented reward values for the first RL wrapper."""

    step_penalty: float = -0.1
    goal_reward: float = 10.0
    blocked_penalty: float = -1.0
    timeout_penalty: float = -5.0


class SingleAgentRLEnv:
    """A small RL-style wrapper around the existing GridWorldEnv.

    The wrapper controls one agent. All other agents stay in place for now.
    It does not change GridWorldEnv movement, collision, obstacle, or goal rules.

    Observations use a four-value coordinate vector:

    - `agent_x`: controlled agent x coordinate, where x is the grid column.
    - `agent_y`: controlled agent y coordinate, where y is the grid row.
    - `goal_x`: controlled agent goal x coordinate.
    - `goal_y`: controlled agent goal y coordinate.

    This deliberately excludes obstacles, other agents, full-grid encodings,
    images, action history, and reward history. The first RL wrapper keeps the
    observation small so reset/step behavior is easy to inspect and test before
    adding richer observation spaces later.

    Actions use a five-value integer interface:

    - `0`: up
    - `1`: down
    - `2`: left
    - `3`: right
    - `4`: stay

    The integers map directly to the existing `Action` enum used by
    `GridWorldEnv`. Invalid action values raise `ValueError`.

    Rewards use a small additive formula:

    - Step penalty on every step.
    - Additional blocked-move penalty when GridWorldEnv blocks the controlled action.
    - Additional goal reward when the controlled agent reaches its goal.
    - Additional timeout penalty when max steps are reached before the goal.

    Episode state is explicit:

    - `step_count`: current underlying GridWorldEnv step count.
    - `done`: true after the controlled agent reaches its goal or times out.
    - `episode_done`: read-only alias for `done`.
    - `reached_goal`: true after the controlled agent reaches its goal.
    - `timeout`: true when max steps are reached before the controlled agent reaches its goal.
    """

    ACTIONS: tuple[Action, ...] = (
        Action.UP,
        Action.DOWN,
        Action.LEFT,
        Action.RIGHT,
        Action.STAY,
    )
    action_names: ActionNames = ACTION_NAMES
    action_size: int = len(ACTION_NAMES)
    observation_names: ObservationNames = OBSERVATION_NAMES
    observation_size: int = len(OBSERVATION_NAMES)

    def __init__(
        self,
        env: GridWorldEnv,
        controlled_agent_id: str = "agent_1",
        rewards: RewardConfig | None = None,
    ):
        self.env = env
        self.controlled_agent_id = controlled_agent_id
        self.rewards = rewards or RewardConfig()
        self.step_count = 0
        self.done = False
        self.reached_goal = False
        self.timeout = False
        self._validate_controlled_agent()

    @property
    def episode_done(self) -> bool:
        return self.done

    @property
    def max_steps(self) -> int:
        return self.env.config.max_steps

    @property
    def step_penalty(self) -> float:
        return self.rewards.step_penalty

    @property
    def goal_reward(self) -> float:
        return self.rewards.goal_reward

    @property
    def blocked_penalty(self) -> float:
        return self.rewards.blocked_penalty

    @property
    def timeout_penalty(self) -> float:
        return self.rewards.timeout_penalty

    @property
    def reward_values(self) -> RewardValues:
        return {
            "step_penalty": self.step_penalty,
            "goal_reward": self.goal_reward,
            "blocked_penalty": self.blocked_penalty,
            "timeout_penalty": self.timeout_penalty,
        }

    def reset(self) -> Observation:
        """Reset the underlying grid-world and return the initial observation."""

        self.env.reset()
        self.step_count = 0
        self.done = False
        self.reached_goal = False
        self.timeout = False
        self._validate_controlled_agent()
        return self.get_observation()

    def step(self, action: int) -> tuple[Observation, float, bool, dict[str, object]]:
        """Apply one integer action and return observation, reward, done, info."""

        if self.done:
            raise RuntimeError("Cannot call step() after the episode is done. Call reset() first.")

        mapped_action = self.action_to_env_action(action)
        actions = {agent_id: Action.STAY for agent_id in self.env.agents}
        actions[self.controlled_agent_id] = mapped_action

        result = self.env.step(actions)
        self.step_count = result.step_count

        event = self._controlled_event(result.events)
        self.reached_goal = self.env.agents[self.controlled_agent_id].at_goal
        self.timeout = result.step_count >= self.max_steps and not self.reached_goal
        self.done = self.reached_goal or self.timeout

        reward = self.compute_reward(
            reached_goal=self.reached_goal,
            timeout=self.timeout,
            event=event,
        )
        info: dict[str, object] = {
            "controlled_agent_id": self.controlled_agent_id,
            "action": mapped_action.value,
            "step_count": self.step_count,
            "reached_goal": self.reached_goal,
            "blocked": event.blocked_reason is not None,
            "blocked_reason": event.blocked_reason,
            "timeout": self.timeout,
            "done": self.done,
        }

        return self.get_observation(), reward, self.done, info

    @classmethod
    def action_to_env_action(cls, action: int) -> Action:
        """Map a discrete integer action to the simulator's Action enum."""

        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError("Action must be an integer from 0 to 4.")
        if action < 0 or action >= len(cls.ACTIONS):
            raise ValueError("Action must be an integer from 0 to 4.")
        return cls.ACTIONS[action]

    @classmethod
    def action_from_int(cls, action: int) -> Action:
        """Compatibility alias for action_to_env_action()."""

        return cls.action_to_env_action(action)

    def get_observation(self) -> Observation:
        """Return `(agent_x, agent_y, goal_x, goal_y)` for the controlled agent."""

        agent = self.env.agents[self.controlled_agent_id]
        if agent.goal is None:
            raise ValueError(f"Controlled agent '{self.controlled_agent_id}' must have a goal.")
        return (agent.position.x, agent.position.y, agent.goal.x, agent.goal.y)

    def compute_reward(
        self,
        *,
        reached_goal: bool,
        timeout: bool,
        event: AgentStepEvent,
    ) -> float:
        reward = self.rewards.step_penalty
        if event.blocked_reason is not None:
            reward += self.rewards.blocked_penalty
        if reached_goal:
            reward += self.rewards.goal_reward
        elif timeout:
            reward += self.rewards.timeout_penalty
        return reward

    def _validate_controlled_agent(self) -> None:
        if self.controlled_agent_id not in self.env.agents:
            raise ValueError(f"Unknown controlled agent id: {self.controlled_agent_id}")
        if self.env.agents[self.controlled_agent_id].goal is None:
            raise ValueError(f"Controlled agent '{self.controlled_agent_id}' must have a goal.")

    def _controlled_event(self, events: tuple[AgentStepEvent, ...]) -> AgentStepEvent:
        for event in events:
            if event.agent_id == self.controlled_agent_id:
                return event
        raise RuntimeError(f"No step event found for controlled agent: {self.controlled_agent_id}")
