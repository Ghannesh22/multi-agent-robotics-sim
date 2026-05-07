from __future__ import annotations

from collections.abc import Mapping, Sequence

from marlsim.core.actions import Action
from marlsim.core.environment import AgentStepEvent, GridWorldEnv
from marlsim.rl.single_agent_env import Observation, RewardConfig, SingleAgentRLEnv


MultiAgentObservations = dict[str, Observation]
MultiAgentRewards = dict[str, float]
MultiAgentInfo = dict[str, object]


class MultiAgentRLEnv:
    """A small multi-agent RL wrapper around the existing GridWorldEnv.

    The wrapper controls multiple agents at once. It accepts one integer action
    per controlled agent, submits all controlled actions to GridWorldEnv in the
    same environment step, and returns per-agent observations and rewards.
    """

    action_names = SingleAgentRLEnv.action_names
    action_size = SingleAgentRLEnv.action_size
    observation_names = SingleAgentRLEnv.observation_names
    observation_size = SingleAgentRLEnv.observation_size

    def __init__(
        self,
        env: GridWorldEnv,
        controlled_agent_ids: Sequence[str],
        rewards: RewardConfig | None = None,
    ):
        if not controlled_agent_ids:
            raise ValueError("controlled_agent_ids must not be empty.")

        self.env = env
        self.controlled_agent_ids = tuple(controlled_agent_ids)
        self.rewards = rewards or RewardConfig()
        self.step_count = 0
        self.done = False
        self.timeout = False
        self.reached_goals: set[str] = set()
        self._validate_controlled_agents()

    @property
    def episode_done(self) -> bool:
        return self.done

    @property
    def max_steps(self) -> int:
        return self.env.config.max_steps

    def reset(self) -> MultiAgentObservations:
        """Reset the wrapped grid world and return per-agent observations."""

        self.env.reset()
        self.step_count = 0
        self.done = False
        self.timeout = False
        self.reached_goals = set()
        self._validate_controlled_agents()
        return self.get_observations()

    def step(
        self, actions: Mapping[str, int]
    ) -> tuple[MultiAgentObservations, MultiAgentRewards, bool, MultiAgentInfo]:
        """Apply simultaneous integer actions for all controlled agents."""

        if self.done:
            raise RuntimeError("Cannot call step() after the episode is done. Call reset() first.")

        self._validate_action_mapping(actions)
        env_actions = {agent_id: Action.STAY for agent_id in self.env.agents}
        for agent_id, action in actions.items():
            env_actions[agent_id] = self.action_to_env_action(action)

        result = self.env.step(env_actions)
        self.step_count = result.step_count
        events = {event.agent_id: event for event in result.events}

        current_reached = {
            agent_id
            for agent_id in self.controlled_agent_ids
            if self.env.agents[agent_id].at_goal
        }
        newly_reached = current_reached - self.reached_goals
        self.reached_goals.update(current_reached)
        self.timeout = self.step_count >= self.max_steps and len(self.reached_goals) < len(
            self.controlled_agent_ids
        )
        self.done = len(self.reached_goals) == len(self.controlled_agent_ids) or self.timeout

        rewards = {
            agent_id: self.compute_reward(
                event=events[agent_id],
                newly_reached=agent_id in newly_reached,
                reached_goal=agent_id in self.reached_goals,
                timeout=self.timeout,
            )
            for agent_id in self.controlled_agent_ids
        }
        info = self._info(events=events)
        return self.get_observations(), rewards, self.done, info

    @classmethod
    def action_to_env_action(cls, action: int) -> Action:
        """Map a discrete integer action to the simulator's Action enum."""

        return SingleAgentRLEnv.action_to_env_action(action)

    def get_observations(self) -> MultiAgentObservations:
        """Return `(agent_x, agent_y, goal_x, goal_y)` for each controlled agent."""

        return {
            agent_id: self._observation(agent_id)
            for agent_id in self.controlled_agent_ids
        }

    def compute_reward(
        self,
        *,
        event: AgentStepEvent,
        newly_reached: bool,
        reached_goal: bool,
        timeout: bool,
    ) -> float:
        reward = self.rewards.step_penalty
        if event.blocked_reason is not None:
            reward += self.rewards.blocked_penalty
        if newly_reached:
            reward += self.rewards.goal_reward
        elif timeout and not reached_goal:
            reward += self.rewards.timeout_penalty
        return reward

    def _observation(self, agent_id: str) -> Observation:
        agent = self.env.agents[agent_id]
        if agent.goal is None:
            raise ValueError(f"Controlled agent '{agent_id}' must have a goal.")
        return (agent.position.x, agent.position.y, agent.goal.x, agent.goal.y)

    def _info(self, events: Mapping[str, AgentStepEvent]) -> MultiAgentInfo:
        per_agent = {}
        for agent_id in self.controlled_agent_ids:
            event = events[agent_id]
            per_agent[agent_id] = {
                "action": event.action.value,
                "blocked": event.blocked_reason is not None,
                "blocked_reason": event.blocked_reason,
                "reached_goal": agent_id in self.reached_goals,
                "timeout": self.timeout and agent_id not in self.reached_goals,
                "step_count": self.step_count,
            }
        return {
            "controlled_agent_ids": self.controlled_agent_ids,
            "step_count": self.step_count,
            "timeout": self.timeout,
            "done": self.done,
            "agents": per_agent,
        }

    def _validate_controlled_agents(self) -> None:
        seen: set[str] = set()
        for agent_id in self.controlled_agent_ids:
            if agent_id in seen:
                raise ValueError(f"Duplicate controlled agent id: {agent_id}")
            seen.add(agent_id)
            if agent_id not in self.env.agents:
                raise ValueError(f"Unknown controlled agent id: {agent_id}")
            if self.env.agents[agent_id].goal is None:
                raise ValueError(f"Controlled agent '{agent_id}' must have a goal.")

    def _validate_action_mapping(self, actions: Mapping[str, int]) -> None:
        expected = set(self.controlled_agent_ids)
        received = set(actions)
        missing = expected - received
        unexpected = received - expected
        if missing:
            raise ValueError(f"Missing actions for controlled agents: {sorted(missing)}")
        if unexpected:
            raise ValueError(f"Unexpected actions for uncontrolled agents: {sorted(unexpected)}")
        for action in actions.values():
            self.action_to_env_action(action)
