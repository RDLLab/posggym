"""Heuristic policies for Predator-Prey continuous environment."""
from __future__ import annotations

import abc
import math
from typing import TYPE_CHECKING, Tuple, cast

import numpy as np

from posggym.agents.policy import Policy, PolicyID, PolicyState
from posggym.agents.utils import action_distributions
from posggym.envs.continuous.predator_prey_continuous import (
    PPAction,
    PPObs,
    PredatorPreyContinuousModel,
)
from posggym.utils import seeding


if TYPE_CHECKING:
    from posggym.posggym.model import POSGModel
    from posggym.utils.history import AgentHistory


class PPCHeuristicPolicy(Policy[PPAction, PPObs], abc.ABC):
    """Base class for heuristic policies for Predator-Prey continuous environment."""

    def __init__(self, model: POSGModel, agent_id: str, policy_id: PolicyID):
        super().__init__(model, agent_id, policy_id)
        self.model = cast(PredatorPreyContinuousModel, model)
        self._rng, _ = seeding.np_random()

        self.action_space = self.model.action_spaces[self.agent_id]
        self.action_dtype = self.action_space.dtype
        self.obs_dist = self.model.obs_dist
        self.n_sensors = self.model.n_sensors
        self.sensor_angle = 2 * math.pi / self.n_sensors

    def reset(self, *, seed: int | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng, _ = seeding.np_random(seed=seed)

    def get_initial_state(self) -> PolicyState:
        state = super().get_initial_state()
        state.update(
            {
                "pi": None,
                "last_obs": None,
            }
        )
        return state

    def get_next_state(
        self,
        obs: PPObs,
        state: PolicyState,
    ) -> PolicyState:
        pi = self._get_pi_from_obs(obs)
        return {
            "pi": pi,
            "last_obs": obs,
        }

    def get_state_from_history(self, history: AgentHistory) -> PolicyState:
        _, last_obs = history.get_last_step()
        if last_obs is not None:
            return {
                "pi": self._get_pi_from_obs(last_obs),
                "last_obs": last_obs,
            }
        return self.get_initial_state()

    def sample_action(self, state: PolicyState) -> PPAction:
        if state["pi"] is None:
            raise ValueError(
                "Policy state does not contain a valid action distribution. Make sure"
                "to call `step` or `get_next_state` before calling `sample_action`"
            )
        return np.array(state["pi"].sample(), dtype=self.action_dtype)

    def get_pi(self, state: PolicyState) -> action_distributions.ActionDistribution:
        return state["pi"]

    def get_value(self, state: PolicyState) -> float:
        raise NotImplementedError(
            f"`get_value()` no implemented by {self.__class__.__name__} policy"
        )

    @abc.abstractmethod
    def _get_pi_from_obs(self, obs: PPObs) -> action_distributions.ActionDistribution:
        raise NotImplementedError

    def _get_closest_prey(self, obs: PPObs) -> Tuple[float, float] | None:
        prey_obs = obs[2 * self.n_sensors : 3 * self.n_sensors]
        closest_idx = np.argmin(prey_obs)
        if prey_obs[closest_idx] == self.model.obs_dist:
            return None
        closest_angle = closest_idx * self.sensor_angle
        closest_dist = prey_obs[closest_idx]
        return closest_dist, closest_angle

    def _get_closest_predator(self, obs: PPObs) -> Tuple[float, float] | None:
        pred_obs = obs[self.n_sensors : 2 * self.n_sensors]
        closest_idx = np.argmin(pred_obs)
        if pred_obs[closest_idx] == self.model.obs_dist:
            return None
        closest_angle = closest_idx * self.sensor_angle
        closest_dist = pred_obs[closest_idx]
        return closest_dist, closest_angle

    def _get_closest_prey_to_predator(
        self, obs: PPObs, pred_dist: float, pred_angle: float
    ) -> Tuple[float, float] | None:
        # find prey with minimum distance to predator
        # d^2 = P^2 + p^2 - 2Pp cos(theta)
        # d = distance between predator and prey
        # P = distance to closest predator
        # p = distance to prey
        # theta = angle between predator and prey (from ego's perspective)
        # skip sqrt since we just want to find the closest prey idx, not actual dist
        prey_obs = obs[2 * self.n_sensors : 3 * self.n_sensors]
        prey_dists = (
            pred_dist**2
            + prey_obs**2
            - (
                2
                * pred_dist
                * prey_obs
                * np.cos(pred_angle - np.arange(self.n_sensors) * self.sensor_angle)
            )
        )
        prey_dists = np.where(prey_obs < self.model.obs_dist, prey_dists, np.inf)
        if np.all(prey_dists == np.inf):
            return None
        closest_prey_idx = np.argmin(prey_dists)
        closest_prey_angle = closest_prey_idx * self.sensor_angle
        closest_prey_dist = prey_dists[closest_prey_idx]
        return closest_prey_dist, closest_prey_angle

    def _get_action_to_target(
        self, target_dist: float, target_angle: float
    ) -> PPAction:
        dyaw = self._get_angular_velocity(target_angle)
        dvel = self._get_linear_velocity(target_dist)
        return np.array([dyaw, dvel], dtype=np.float32)

    def _get_angular_velocity(self, target_angle: float) -> float:
        if target_angle > math.pi:
            dyaw = max(target_angle - (2 * math.pi), -self.model.dyaw_limit)
        else:
            dyaw = min(target_angle, self.model.dyaw_limit)
        return dyaw

    def _get_linear_velocity(self, target_dist: float) -> float:
        return min(target_dist, 1.0)


class PPCHeuristic0Policy(PPCHeuristicPolicy):
    """Heuristic 0 policy for Predator-Prey continuous environment.

    Predator greedily moves towards closest observed prey, or moves randomly.
    """

    def _get_pi_from_obs(self, obs: PPObs) -> action_distributions.ActionDistribution:
        target = self._get_closest_prey(obs)
        if target is None:
            # No prey or predator in sight
            pi = action_distributions.ContinousUniformActionDistribution(
                low=self.action_space.low, high=self.action_space.high, rng=self._rng
            )
        else:
            # Move towards closest prey
            target_dist, target_angle = target
            action = self._get_action_to_target(target_dist, target_angle)
            pi = action_distributions.DeterministicActionDistribution(action)

        return pi


class PPCHeuristic1Policy(PPCHeuristicPolicy):
    """Heuristic 1 policy for Predator-Prey continuous environment.

    Predator greedily moves towards closest observed prey, closest observed predator,
    or moves randomly, in that order.
    """

    def _get_pi_from_obs(self, obs: PPObs) -> action_distributions.ActionDistribution:
        target = self._get_closest_prey(obs)
        if target is None:
            target = self._get_closest_predator(obs)

        if target is None:
            # No prey or predator in sight
            pi = action_distributions.ContinousUniformActionDistribution(
                low=self.action_space.low, high=self.action_space.high, rng=self._rng
            )
        else:
            # Move towards closest prey or predator
            target_dist, target_angle = target
            action = self._get_action_to_target(target_dist, target_angle)
            pi = action_distributions.DeterministicActionDistribution(action)

        return pi


class PPCHeuristic2Policy(PPCHeuristicPolicy):
    """Heuristic 2 policy for Predator-Prey continuous environment.

    Predator moves randomly until it observes prey, then it greedily moves towards
    closest observed prey, or if it observes a prey and predator then it moves
    towards the closest observed prey to the closest observed predator.

    """

    def _get_pi_from_obs(self, obs: PPObs) -> action_distributions.ActionDistribution:
        prey = self._get_closest_prey(obs)
        if prey is None:
            # No prey in sight
            return action_distributions.ContinousUniformActionDistribution(
                low=self.action_space.low, high=self.action_space.high, rng=self._rng
            )

        pred = self._get_closest_predator(obs)
        if pred is None:
            # No predator in sight, so move towards closest prey
            action = self._get_action_to_target(prey[0], prey[1])
            return action_distributions.DeterministicActionDistribution(action)
        pred_dist, pred_angle = pred

        # Move towards closest prey to predator
        prey = self._get_closest_prey_to_predator(obs, pred_dist, pred_angle)
        assert prey is not None
        action = self._get_action_to_target(prey[0], prey[1])
        return action_distributions.DeterministicActionDistribution(action)


class PPCHeuristic3Policy(PPCHeuristicPolicy):
    """Heuristic 3 policy for Predator-Prey continuous environment.

    Predator moves randomly until it observes a predator, then it greedily moves towards
    closest observed predator, or if it observes a prey and predator then it moves
    towards the closest observed prey to the closest observed predator.

    """

    def _get_pi_from_obs(self, obs: PPObs) -> action_distributions.ActionDistribution:
        pred = self._get_closest_predator(obs)
        if pred is None:
            # No predator in sight
            return action_distributions.ContinousUniformActionDistribution(
                low=self.action_space.low, high=self.action_space.high, rng=self._rng
            )
        pred_dist, pred_angle = pred

        prey = self._get_closest_prey_to_predator(obs, pred_dist, pred_angle)
        if prey is None:
            # No prey in sight, so move towards predator
            action = self._get_action_to_target(pred_dist, pred_angle)
            return action_distributions.DeterministicActionDistribution(action)

        # Move towards closest prey to predator
        action = self._get_action_to_target(prey[0], prey[1])
        return action_distributions.DeterministicActionDistribution(action)
