"""Heuiristic policies for DroneTeamCapture env."""
from __future__ import annotations

import abc
import math
from typing import cast

import numpy as np

from posggym import logger
from posggym.agents.policy import Policy, PolicyID, PolicyState
from posggym.agents.utils import action_distributions
from posggym.envs.continuous.drone_team_capture import (
    DroneTeamCaptureModel,
    DTCAction,
    DTCObs,
)


class DTCHeuristicPolicy(Policy[DTCAction, DTCObs], abc.ABC):
    """Heuristic Policies for the Drone Team Capture continuous environment.

    This is the abstract Drone Team Capture heuristic policy class. Concrete
    implementations must implement the _get_action method.

    """

    def __init__(
        self,
        model: DroneTeamCaptureModel,
        agent_id: str,
        policy_id: PolicyID,
    ):
        if model.n_com_pursuers < model.n_pursuers - 1 or (
            model.observation_limit is not None
            and model.observation_limit < 2 * model.r_arena
        ):
            logger.warn(
                "The DroneTeamCapture Heuristic policies are designed for the case "
                "where each pursuer can see every other pursuer "
                "(i.e. `n_com_pursuers = n_pursuers - 1` and `observation_limit = None`"
                "). Using it for other cases may result in unexpected behavior. "
                f"Currently `n_com_pursuers = {model.n_com_pursuers}`,"
                f"`n_pursuers = {model.n_pursuers}`, and "
                f"`observation_limit = {model.observation_limit}`."
            )
        self.obs_dim = model.obs_dim
        self.omega_max = model.dyaw_limit
        self.pursuer_vel = model.max_pursuer_vel
        self.pred_target_vel = model.max_pursuer_vel
        super().__init__(model, agent_id, policy_id)

    @abc.abstractmethod
    def _get_action(self, state: PolicyState) -> DTCAction:
        """Get the next action from the state.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_initial_state(self) -> PolicyState:
        state = super().get_initial_state()
        n = (self.obs_dim - 8) // 2
        state["prev_yaw"] = 0.0
        state["yaw"] = 0.0
        state["prev_xy"] = np.zeros((2,), dtype=np.float32)
        state["xy"] = np.zeros((2,), dtype=np.float32)
        state["prev_target_xy"] = np.zeros((2,), dtype=np.float32)
        state["target_xy"] = np.zeros((2,), dtype=np.float32)
        state["prev_other_xy"] = np.zeros((n, 2), dtype=np.float32)
        state["other_xy"] = np.zeros((n, 2), dtype=np.float32)
        return state

    def get_next_state(
        self,
        obs: DTCObs,
        state: PolicyState,
    ) -> PolicyState:
        # Unnormalize obs
        min_obs = self.model.observation_spaces[self.agent_id].low
        # need to change this to be the actual min distance
        # instead of -1.0 used in obs space to handle out-of-limit agents
        min_obs[5] = 0.0
        for i in range(9, len(min_obs), 2):
            min_obs[i] = 0.0

        obs = self.model.world.convert_into_interval(
            obs,
            min_obs,
            self.model.observation_spaces[self.agent_id].high,
            self.model.raw_obs_range[0],
            self.model.raw_obs_range[1],
            clip=False,
        )
        xy = obs[2:4]
        yaw = obs[0]

        target_xy = xy + np.array(
            [
                obs[5] * np.cos(obs[4] + yaw),
                obs[5] * np.sin(obs[4] + yaw),
            ],
            dtype=np.float32,
        )

        other_xy = (
            xy
            + np.array(
                [
                    obs[9::2] * np.cos(obs[8::2] + yaw),
                    obs[9::2] * np.sin(obs[8::2] + yaw),
                ],
                dtype=np.float32,
            ).T
        )

        return {
            "action": None,
            "prev_yaw": state["yaw"],
            "yaw": yaw,
            "prev_xy": state["xy"],
            "xy": xy,
            "prev_target_xy": state["target_xy"],
            "target_xy": target_xy,
            "prev_other_xy": state["other_xy"],
            "other_xy": other_xy,
        }

    def sample_action(self, state: PolicyState) -> DTCAction:
        return self._get_action(state)

    def get_pi(self, state: PolicyState) -> action_distributions.ActionDistribution:
        action = self._get_action(state)
        return action_distributions.DeterministicActionDistribution(action)

    def euclidean_dist(self, coord1: np.ndarray, coord2: np.ndarray) -> float:
        return float(np.linalg.norm(coord1[:2] - coord2[:2]))

    def rot(self, los_angle: float) -> np.ndarray:
        R = np.array(
            [
                [math.cos(los_angle), math.sin(los_angle)],
                [-math.sin(los_angle), math.cos(los_angle)],
            ]
        )
        return R

    def pp(self, alpha: float) -> float:
        Kpp = 100
        omega = Kpp * alpha
        omega = np.clip(omega, -self.omega_max, self.omega_max)
        return omega

    def normalise(self, dx: float, dy: float) -> np.ndarray:
        d = math.sqrt(dx**2 + dy**2)
        d = np.clip(d, 0.000001, 10000000)
        return np.array([dx, dy]) / d

    def normalise_safe(self, arr: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(arr, axis=1)
        d = np.clip(d, 0.000001, 10000000)
        return (arr.T / d).T

    def get_value(self, state: PolicyState) -> float:
        raise NotImplementedError(
            f"`get_value()` not implemented by {self.__class__.__name__} policy"
        )


class DTCJanosovHeuristicPolicy(DTCHeuristicPolicy):
    """Drone Team Capture Janosov Heuristic policy.

    Uses the heuristic from the following paper:

    DTCJanosovHeuristicPolicy is the policy from this work:
    Janosov, M., Virágh, C., Vásárhelyi, G., & Vicsek, T. (2017). Group chasing tactics:
    how to catch a faster prey. New Journal of Physics, 19(5), 053003.
    doi:10.1088/1367-2630/aa69e7

    This implementation is a non-holonomic adaptation of the original work following
    the protocol described in:

    C. de Souza, R. Newbury, A. Cosgun, P. Castillo, B. Vidolov and D. Kulić,
    "Decentralized Multi-Agent Pursuit Using Deep Reinforcement Learning,"
    in IEEE Robotics and Automation Letters, vol. 6, no. 3, pp. 4552-4559,
    July 2021, doi: 10.1109/LRA.2021.3068952.

    """

    def _get_action(self, state: PolicyState) -> DTCAction:
        chase = self.attraction2(
            state["xy"],
            state["prev_xy"],
            state["target_xy"],
            state["prev_target_xy"],
        )

        inter = self.alignment2(
            state["xy"],
            state["prev_xy"],
            state["other_xy"],
            state["prev_other_xy"],
        )

        # Sum up the contributions to the desired velocity
        desired_velocity = chase + inter

        # Rotate the desired velocity according to the agent's heading angle
        R = self.rot(state["yaw"])
        direction = R.dot(desired_velocity)

        # Calculate the angle between the desired direction and the current heading
        alpha = cast(float, np.arctan2(direction[1], direction[0]))

        # Calculate the angular velocity (omega) using the pp method
        omega = self.pp(alpha)

        return np.array([omega], dtype=np.float32)

    def attraction2(
        self,
        xy_i: np.ndarray,
        prev_xy_i: np.ndarray,
        target_xy: np.ndarray,
        prev_target_xy: np.ndarray,
    ) -> np.ndarray:
        # Friction term
        dist = self.euclidean_dist(target_xy, xy_i)
        vel_p = xy_i - prev_xy_i
        vel_t = target_xy - prev_target_xy
        visc = (vel_p - vel_t) / dist**2

        # Atraction term
        target_pred = self.prediction(xy_i, target_xy, vel_t)
        atrac = self.normalise(*(target_pred - xy_i))

        chase = atrac + 1.5 * visc
        chase = self.normalise(chase[0], chase[1])

        return chase * self.pursuer_vel

    def alignment2(
        self,
        xy_i: np.ndarray,
        prev_xy_i: np.ndarray,
        other_xy: np.ndarray,
        prev_other_xy: np.ndarray,
    ):
        rad_inter = 250
        C_inter = 0.5
        C_f = 0.5

        d_ij = xy_i - other_xy
        d = np.linalg.norm(d_ij, axis=1)
        d_ij = self.normalise_safe(d_ij)

        vel_i = xy_i - prev_xy_i
        vel_j = other_xy - prev_other_xy

        rep = ((d_ij.T * np.clip(d - rad_inter, -10000, 0)) / d).T
        fric = ((vel_i - vel_j).T / d**2).T

        inte = self.normalise(*(-rep + C_f * fric).sum(axis=0))

        return C_inter * inte * self.pursuer_vel

    def prediction(
        self, xy_i: np.ndarray, target_xy: np.ndarray, vel_t: np.ndarray
    ) -> np.ndarray:
        dist = self.euclidean_dist(target_xy, xy_i)

        tau = 20
        time_pred = dist / self.pred_target_vel
        time_pred = np.clip(time_pred, 0, tau)

        vel_t = self.normalise(vel_t[0], vel_t[1]) * self.pred_target_vel
        pos_pred = target_xy + vel_t * time_pred

        return pos_pred


class DTCAngelaniHeuristicPolicy(DTCHeuristicPolicy):
    """Drone Team Capture Angelani Heuristic policy.

    Uses the heuristic from the following paper:

    DTCAngelaniHeuristicPolicy is the policy from this work:
    Angelani, L. (09 2012). Collective Predation and Escape Strategies.
    Physical Review Letters, 109. doi:10.1103/PhysRevLett.109.118104

    This implementation is a non-holonomic adaptation of the original work following
    the protocol described in:

    C. de Souza, R. Newbury, A. Cosgun, P. Castillo, B. Vidolov and D. Kulić,
    "Decentralized Multi-Agent Pursuit Using Deep Reinforcement Learning,"
    in IEEE Robotics and Automation Letters, vol. 6, no. 3, pp. 4552-4559,
    July 2021, doi: 10.1109/LRA.2021.3068952.

    """

    def _get_action(self, state: PolicyState) -> DTCAction:
        rep = self.repulsion(state["xy"], state["other_xy"])

        align = self.alignment(
            np.concatenate((state["xy"].reshape(1, -1), state["other_xy"]), axis=0),
            np.concatenate(
                (state["prev_xy"].reshape(1, -1), state["prev_other_xy"]), axis=0
            ),
        )
        atrac = self.attraction(state["xy"], state["target_xy"])

        vel = rep + align + atrac

        direction = self.rot(state["yaw"]).dot(vel)
        alpha = math.atan2(direction[1], direction[0])
        omega_i = self.pp(alpha)
        return np.array([omega_i], dtype=np.float32)

    def repulsion(self, xy_i: np.ndarray, other_xy: np.ndarray):
        r = 300
        # Vectorized calculation of relative positions
        r_iT = other_xy - xy_i
        dist = np.linalg.norm(r_iT, axis=1)  # Compute the distance for all pursuers
        mask = dist < r
        repulsive_agents = r_iT[mask]

        dxy = self.rep_force(repulsive_agents, dist[mask])

        return self.normalise(*dxy)

    def rep_force(self, r_ij: np.ndarray, dist: np.ndarray) -> np.ndarray:
        sigma = 3
        u = -r_ij / dist[:, None]
        # Need to use float64 for the exponential to avoid overflow
        den = 1 + np.exp((dist - 20) / sigma, dtype=np.float64)[:, None]
        rep = u / den
        return rep.sum(axis=0, dtype=np.float32)

    def alignment(
        self, pursuers_xy: np.ndarray, prev_pursuers_xy: np.ndarray
    ) -> np.ndarray:
        pursuers_diff = pursuers_xy - prev_pursuers_xy
        return self.normalise(*np.sum(pursuers_diff, axis=0))

    def attraction(self, xy_i: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
        r_iT = target_xy - xy_i
        return self.normalise(*r_iT)


class DTCDPPHeuristicPolicy(DTCHeuristicPolicy):
    """Drone Team Capture Deviated Pure Pursuit Heuristic policy.

    Uses the heuristic from the following paper:

    Souza, C., Castillo, P., & Vidolov, B. (2022). Local interaction and navigation
    guidance for hunters drones: a chase behavior approach with real-time tests.
    Robotica, 40(8), 2697–2715.

    """

    def _get_action(self, state: PolicyState) -> DTCAction:
        offset = math.pi / 8
        sense = 0.0

        sense = np.sum(self.delta(state["xy"], state["other_xy"], state["target_xy"]))

        alphaiT = self.engagement(state["xy"], state["yaw"], state["target_xy"])
        omega_i = 0.6 * (alphaiT - sense * offset)
        omega_i = np.clip(omega_i, -self.omega_max, self.omega_max)
        return np.array([omega_i], dtype=np.float32)

    def delta(
        self, xy_i: np.ndarray, other_xy: np.ndarray, target_xy: np.ndarray
    ) -> float:
        T_p = target_xy - xy_i
        los_angle = math.atan2(T_p[1], T_p[0])

        R = np.array(
            [
                [math.cos(los_angle), math.sin(los_angle)],
                [-math.sin(los_angle), math.cos(los_angle)],
            ]
        )
        i_j = other_xy - xy_i
        i_j = np.dot(i_j, R.T)

        # Avoid dividing by zero
        sense = np.where(np.abs(i_j[:, 1]) != 0, i_j[:, 1] / np.abs(i_j[:, 1]), 0)
        return np.sum(sense)

    def engagement(
        self, xy_i: np.ndarray, yaw_i: float, target_xy: np.ndarray
    ) -> float:
        # Rotation matrix of yaw
        R = np.array(
            [[math.cos(yaw_i), math.sin(yaw_i)], [-math.sin(yaw_i), math.cos(yaw_i)]]
        )

        T_p = R.dot(target_xy - xy_i)
        alpha = math.atan2(T_p[1], T_p[0])
        return alpha


class DTCGreedyHeuristicPolicy(DTCHeuristicPolicy):
    """Drone Team Capture Greedy Heuristic policy.

    Simple policy which selects the action which minimize the distance to the target
    at each step, ignoring the other pursuers.

    """

    def _get_action(self, state: PolicyState) -> DTCAction:
        target_xy = state["target_xy"]
        xy_i = state["xy"]
        yaw_i = state["yaw"]

        R = self.rot(yaw_i)

        T_p = R.dot(target_xy - xy_i)
        alpha = math.atan2(T_p[1], T_p[0])
        omega_i = self.pp(alpha)
        return np.array([omega_i], dtype=np.float32)
