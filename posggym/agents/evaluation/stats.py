"""Classes and functions for recording agent statistics during episodes."""
import abc
import time
from collections import ChainMap
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

import posggym
import posggym.model as M
from posggym.agents.policy import Policy

AgentStatisticsMap = Dict[str, Dict[str, Any]]


def generate_episode_statistics(trackers: Iterable["Tracker"]) -> AgentStatisticsMap:
    """Generate episode statistics from set of trackers."""
    statistics = combine_statistics([t.get_episode() for t in trackers])
    return statistics


def generate_statistics(trackers: Iterable["Tracker"]) -> AgentStatisticsMap:
    """Generate summary statistics from set of trackers."""
    statistics = combine_statistics([t.get() for t in trackers])
    return statistics


def combine_statistics(
    statistic_maps: Sequence[AgentStatisticsMap],
) -> AgentStatisticsMap:
    """Combine multiple Agent statistic maps into a single one."""
    agent_ids = list(statistic_maps[0].keys())
    return {
        i: dict(ChainMap(*(stat_maps[i] for stat_maps in statistic_maps)))
        for i in agent_ids
    }


def get_default_trackers() -> List["Tracker"]:
    """Get the default set of Trackers."""
    return [EpisodeTracker()]


class Tracker(abc.ABC):
    """Generic Tracker Base class."""

    @abc.abstractmethod
    def step(
        self,
        episode_t: int,
        env: posggym.Env,
        timestep: M.JointTimestep,
        action: Dict[str, M.ActType],
        policies: Dict[str, Policy],
        episode_end: bool,
    ):
        """Accumulates statistics for a single step."""

    @abc.abstractmethod
    def reset(self):
        """Reset all gathered statistics."""

    @abc.abstractmethod
    def reset_episode(self):
        """Reset all statistics prior to each episode."""

    @abc.abstractmethod
    def get_episode(self) -> AgentStatisticsMap:
        """Aggregate episode statistics for each agent."""

    @abc.abstractmethod
    def get(self) -> AgentStatisticsMap:
        """Aggregate all episode statistics for each agent."""


class EpisodeTracker(Tracker):
    """Tracks episode return and other statistics."""

    def __init__(self):
        # initialized when step first called
        self._agents = None

        self._num_episodes = 0
        self._current_start_time = time.time()
        # is initialized when step is first called
        self._current_returns = None
        self._current_dones = None
        self._current_steps = None
        self._current_outcomes = None

        self._dones = []
        self._times = []
        self._returns = []
        self._steps = []
        self._outcomes = []

    def step(
        self,
        episode_t: int,
        env: posggym.Env,
        timestep: M.JointTimestep,
        action: Dict[str, M.ActType],
        policies: Dict[str, Policy],
        episode_end: bool,
    ):
        if self._agents is None:
            self._agents = env.possible_agents

        if self._current_returns is None:
            self._current_returns = {i: 0.0 for i in env.possible_agents}

        if self._current_dones is None:
            self._current_dones = {i: False for i in env.possible_agents}

        if self._current_steps is None:
            self._current_steps = {i: 0 for i in env.possible_agents}

        if self._current_outcomes is None:
            self._current_outcomes = {i: M.Outcome.NA for i in env.possible_agents}

        if episode_t == 0:
            return

        for i in env.possible_agents:
            self._current_returns[i] += timestep.rewards.get(i, 0.0)

            if i in timestep.observations and not self._current_dones[i]:
                self._current_steps[i] += 1

            if not self._current_dones[i] and (
                timestep.terminations.get(i, False)
                or timestep.truncations.get(i, False)
            ):
                self._current_dones[i] = True
                self._current_outcomes[i] = timestep.infos.get("outcome", M.Outcome.NA)

        if episode_end:
            self._num_episodes += 1
            self._times.append(time.time() - self._current_start_time)
            self._returns.append(self._current_returns)
            self._dones.append(self._current_dones)
            self._steps.append(self._current_steps)

            outcome = {}
            for i in env.possible_agents:
                if i in timestep.infos:
                    outcome[i] = timestep.infos.get("outcome", M.Outcome.NA)
                else:
                    outcome[i] = M.Outcome.NA

                if outcome[i] == M.Outcome.NA:
                    # in case outcome was received earlier in episode (i.e. agent
                    # finished early)
                    outcome[i] = self._current_outcomes[i]
            self._outcomes.append(outcome)

    def reset(self):
        self.reset_episode()
        self._num_episodes = 0
        self._dones = []
        self._times = []
        self._returns = []
        self._steps = []
        self._outcomes = []

    def reset_episode(self):
        self._current_done = False
        self._current_start_time = time.time()

        if self._current_returns is not None:
            for i in self._current_returns:
                self._current_returns[i] = 0.0

        if self._current_dones is not None:
            for i in self._current_dones:
                self._current_dones[i] = False

        if self._current_steps is not None:
            for i in self._current_steps:
                self._current_steps[i] = 0

        if self._current_outcomes is not None:
            for i in self._current_outcomes:
                self._current_outcomes[i] = M.Outcome.NA

    def get_episode(self) -> AgentStatisticsMap:
        assert self._agents is not None

        stats = {}
        for i in self._agents:
            stats[i] = {
                "episode_number": self._num_episodes,
                "episode_return": self._returns[-1][i],
                "episode_steps": self._steps[-1][i],
                "episode_outcome": self._outcomes[-1][i],
                "episode_done": self._dones[-1][i],
                "episode_time": self._times[-1],
            }

        return stats

    def get(self) -> AgentStatisticsMap:
        assert self._agents is not None

        outcome_counts = {k: {i: 0 for i in self._agents} for k in M.Outcome}
        for outcome in self._outcomes:
            for i in self._agents:
                outcome_counts[outcome[i]][i] += 1

        stats = {}
        for i in self._agents:
            returns_i = [ep_return[i] for ep_return in self._returns]
            steps_i = [ep_steps[i] for ep_steps in self._steps]
            dones_i = [ep_done[i] for ep_done in self._dones]
            stats[i] = {
                "num_episodes": self._num_episodes,
                "episode_return_mean": np.mean(returns_i, axis=0),
                "episode_return_std": np.std(returns_i, axis=0),
                "episode_return_max": np.max(returns_i, axis=0),
                "episode_return_min": np.min(returns_i, axis=0),
                "episode_steps_mean": np.mean(steps_i),
                "episode_steps_std": np.std(steps_i),
                "episode_time_mean": np.mean(self._times),
                "episode_time_std": np.std(self._times),
                "num_episode_done": np.sum(dones_i),
            }

            for outcome, counts in outcome_counts.items():
                stats[i][f"num_{outcome}"] = counts[i]

        return stats
