"""Heuiristic policies for DroneTeamCapture env."""
from __future__ import annotations

import abc
import math
from typing import TYPE_CHECKING, List, Tuple, cast

import numpy as np

from posggym.agents.policy import Policy, PolicyID, PolicyState
from posggym.envs.continuous.drone_team_capture import (
    DroneTeamCaptureModel,
    DTCAction,
    DTCState,
)
from posggym.agents.utils import action_distributions


if TYPE_CHECKING:
    from posggym.model import AgentID


class DTCHeuristicPolicy(Policy[DTCAction, DTCState], abc.ABC):
    """Heuristic Policies for the Drone Team Capture continuous environment.

    This is the abstract Drone Team Capture heuristic policy class. Concrete
    implementations must implement the _get_action method.

    """

    # DTC heuristic policies are expect full state by default
    observes_state = True

    def __init__(
        self,
        model: DroneTeamCaptureModel,
        agent_id: AgentID,
        policy_id: PolicyID,
    ):
        super().__init__(model, agent_id, policy_id)
        self.agent_idx = int(self.agent_id)
        self.omega_max = math.pi / 10
        self.cap_rad = 25
        self.vel_pur = 10
        self.vel_tar = 10

    def step(self, state: DTCState) -> DTCAction:
        return self._get_action(state)

    @abc.abstractmethod
    def _get_action(self, state: DTCState) -> DTCAction:
        """Get the next action from the state.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_initial_state(self) -> PolicyState:
        state = super().get_initial_state()
        state["last_state"] = None
        return state

    def get_next_state(
        self,
        obs: DTCState,
        state: PolicyState,
    ) -> PolicyState:
        return {"last_state": obs}

    def sample_action(self, state: PolicyState) -> DTCAction:
        return self._get_action(state["last_state"])

    def get_pi(self, state: PolicyState) -> action_distributions.ActionDistribution:
        action = self._get_action(state["last_state"])
        return action_distributions.DeterministicActionDistribution(action)

    def euclidean_dist(self, coord1: np.ndarray, coord2: np.ndarray) -> float:
        return float(np.linalg.norm(coord1[:2] - coord2[:2]))

    def rot(self, Los_angle: float) -> np.ndarray:
        R = np.array(
            [
                [math.cos(Los_angle), math.sin(Los_angle)],
                [-math.sin(Los_angle), math.cos(Los_angle)],
            ]
        )
        return R

    def pp(self, alpha: float) -> float:
        Kpp = 100
        omega = Kpp * alpha
        omega = np.clip(omega, -self.omega_max, self.omega_max)
        return omega

    def alignment(self, pursuers: np.ndarray, pursuers_prev: np.ndarray) -> List[float]:
        pursuers_diff = pursuers[:, :2] - pursuers_prev[:, :2]
        dx, dy = np.sum(pursuers_diff, axis=0)
        dx, dy = self.normalise(dx, dy)
        return [dx, dy]

    def attraction(self, pursuer_i: np.ndarray, target: np.ndarray) -> List[float]:
        r_iT = target[:2] - pursuer_i[:2]
        dx, dy = self.normalise(r_iT[0], r_iT[1])
        return [dx, dy]

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

    def _get_action(self, state: DTCState) -> DTCAction:
        agent_idx = int(self.agent_id)

        chase = self.attraction2(
            state.pursuer_states[agent_idx],
            state.prev_pursuer_states[agent_idx],
            state.target_state,
            state.prev_target_state,
        )

        inter = self.alignment2(
            state.pursuer_states, state.prev_pursuer_states, agent_idx
        )

        # Sum up the contributions to the desired velocity
        desired_velocity = chase + inter

        # Rotate the desired velocity according to the agent's heading angle
        R = self.rot(state.pursuer_states[agent_idx][2])
        direction = R.dot(desired_velocity)

        # Calculate the angle between the desired direction and the current heading
        alpha = cast(float, np.arctan2(direction[1], direction[0]))

        # Calculate the angular velocity (omega) using the pp method
        omega = self.pp(alpha)

        return np.array([omega], dtype=np.float32)

    def attraction2(
        self,
        pursuer_i: np.ndarray,
        pursuer_i_prev: np.ndarray,
        target: np.ndarray,
        target_prev: np.ndarray,
    ) -> np.ndarray:
        # Friction term
        dist = self.euclidean_dist(target, pursuer_i)
        vel_p = (np.array(pursuer_i) - np.array(pursuer_i_prev))[:2]
        vel_t = (np.array(target) - np.array(target_prev))[:2]
        visc = (vel_p - vel_t) / dist**2

        # Atraction term
        target_pred = self.prediction(pursuer_i, target, vel_t)
        atrac = self.attraction(pursuer_i, target_pred)

        chase = atrac + 1.5 * visc
        chase = self.normalise(chase[0], chase[1])

        return chase * self.vel_pur

    def alignment2(self, pursuer: np.ndarray, pursuer_prev: np.ndarray, i: int):
        rad_inter = 250
        C_inter = 0.5
        C_f = 0.5

        mask = np.arange(len(pursuer)) != i

        d_ij = pursuer[i, :2] - pursuer[mask, :2]
        d = np.linalg.norm(d_ij, axis=1)
        d_ij = self.normalise_safe(d_ij)

        vel_i = pursuer[i, :2] - pursuer_prev[i, :2]
        vel_j = pursuer[mask, :2] - pursuer_prev[mask, :2]

        rep = ((d_ij.T * np.clip(d - rad_inter, -10000, 0)) / d).T
        fric = ((vel_i - vel_j).T / d**2).T

        inte = self.normalise(*(-rep + C_f * fric).sum(axis=0))

        return C_inter * inte * self.vel_pur

    def prediction(
        self, pursuer_i: np.ndarray, target: np.ndarray, vel_t: List[float]
    ) -> np.ndarray:
        dist = self.euclidean_dist(target, pursuer_i)

        tau = 20
        time_pred = dist / self.vel_tar
        time_pred = np.clip(time_pred, 0, tau)

        vel_t = self.normalise(vel_t[0], vel_t[1]) * self.vel_tar  # type: ignore

        pos_pred = target[:2] + vel_t * time_pred

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

    def _get_action(self, state: DTCState) -> DTCAction:
        agent_idx = int(self.agent_id)
        rep = self.repulsion(state.pursuer_states, agent_idx)
        align = self.alignment(state.pursuer_states, state.prev_pursuer_states)
        atrac = self.attraction(state.pursuer_states[agent_idx], state.target_state)

        vx = rep[0] + align[0] + atrac[0]
        vy = rep[1] + align[1] + atrac[1]

        R = self.rot(state.pursuer_states[agent_idx][2])
        direction = R.dot([vx, vy])
        alpha = math.atan2(direction[1], direction[0])
        omega_i = self.pp(alpha)
        return np.array([omega_i], dtype=np.float32)

    def repulsion(self, pursuer: np.ndarray, idx: int) -> List[float]:
        r = 300
        Dx, Dy = 0.0, 0.0
        for j in range(len(pursuer)):
            if j != idx:
                r_iT = (pursuer[j] - pursuer[idx])[:2]
                dist = float(np.linalg.norm(r_iT))
                if dist < r:
                    dx, dy = self.rep_force(r_iT, dist)
                    Dx += dx
                    Dy += dy
        dx, dy = self.normalise(Dx, Dy)
        return [dx, dy]

    def rep_force(self, r_ij: np.ndarray, dist: float) -> Tuple[float, float]:
        sigma = 3
        u = -r_ij / dist
        den = 1 + math.exp((dist - 20) / sigma)
        rep = u / den
        return rep[0], rep[1]


class DTCDPPHeuristicPolicy(DTCHeuristicPolicy):
    """Drone Team Capture Deviated Pure Pursuit Heuristic policy.

    Uses the heuristic from the following paper:

    Souza, C., Castillo, P., & Vidolov, B. (2022). Local interaction and navigation
    guidance for hunters drones: a chase behavior approach with real-time tests.
    Robotica, 40(8), 2697–2715.

    """

    def _get_action(self, state: DTCState) -> DTCAction:
        offset = math.pi / 8
        sense = 0.0
        n_pursuers = len(state.pursuer_states)
        pursuer_i = state.pursuer_states[self.agent_idx]
        for j in range(n_pursuers):
            if j != self.agent_idx:
                sense += self.delta(
                    pursuer_i, state.pursuer_states[j], state.target_state
                )

        alphaiT, _ = self.engagmment(pursuer_i, state.target_state)
        omega_i = 0.6 * (alphaiT - sense * offset)
        omega_i = np.clip(omega_i, -self.omega_max, self.omega_max)
        return np.array([omega_i], dtype=np.float32)

    def delta(self, pos_i: np.ndarray, pos_j: np.ndarray, target: np.ndarray) -> float:
        sense = 0
        _, Los_angle = self.engagmment(pos_i, target)
        R = np.array(
            [
                [math.cos(Los_angle), math.sin(Los_angle)],
                [-math.sin(Los_angle), math.cos(Los_angle)],
            ]
        )
        i_j = pos_j[:2] - pos_i[:2]
        i_j = R.dot(i_j)
        if abs(i_j[1]) != 0:
            sense = i_j[1] / abs(i_j[1])  # for do not divide by zero
        return sense

    def engagmment(self, pos_i: np.ndarray, pos_j: np.ndarray) -> Tuple[float, float]:
        yaw = pos_i[2]
        # Rotation matrix of yaw
        R = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])

        T_p = pos_j[:2] - pos_i[:2]
        Los_angle = math.atan2(T_p[1], T_p[0])
        T_p = R.dot(T_p)
        alpha = math.atan2(T_p[1], T_p[0])
        return (alpha, Los_angle)
