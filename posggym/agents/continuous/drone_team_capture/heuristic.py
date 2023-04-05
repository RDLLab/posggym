"""Shortest path policy for PursuitEvasion env."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from posggym.agents.policy import Policy, PolicyID, PolicyState
from posggym.envs.continuous.drone_team_capture import DTCAction, DTCModel, DTCState


if TYPE_CHECKING:
    from posggym.model import AgentID


class DTCHeuristicPolicy(Policy[DTCAction, DTCState]):
    """Heuristic Policies for the Drone Team Capture continuous environment.

    This is the abstract Drone Team Capture heuristic policy class. Concrete
    implementations must implement the _get_action method.

    """

    # DTC heuristic policies are expect full state by default
    observes_state = True

    def __init__(
        self,
        model: DTCModel,
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

    def get_pi(self, state: PolicyState) -> Dict[DTCAction, float]:
        return {self._get_action(state["last_state"]): 1.0}

    def euclidean_dist(self, coord1: np.ndarray, coord2: np.ndarray) -> float:
        return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

    def rot(self, Los_angle: float) -> np.ndarray:
        R = np.array(
            [
                [math.cos(Los_angle), math.sin(Los_angle)],
                [-math.sin(Los_angle), math.cos(Los_angle)],
            ]
        )
        return R

    def sat(self, val: float, min: float, max: float) -> float:
        if val >= max:
            val = max
        if val <= min:
            val = min
        return val

    def pp(self, alpha: float) -> float:
        Kpp = 100
        omega = Kpp * alpha
        omega = self.sat(omega, -self.omega_max, self.omega_max)
        return omega

    def alignment(self, pursuers: np.ndarray, pursuers_prev: np.ndarray) -> List[float]:
        dx, dy = 0.0, 0.0
        for i in range(len(pursuers)):
            dx = dx + (pursuers[i][0] - pursuers_prev[i][0])
            dy = dy + (pursuers[i][1] - pursuers_prev[i][1])
        dx, dy = self.normalise(dx, dy)
        return [dx, dy]

    def attraction(self, pursuer_i: np.ndarray, target: np.ndarray) -> List[float]:
        r_iT = target[:2] - pursuer_i[:2]
        dx, dy = self.normalise(r_iT[0], r_iT[1])
        return [dx, dy]

    def normalise(self, dx: float, dy: float) -> Tuple[float, float]:
        d = math.sqrt(dx**2 + dy**2)
        d = self.sat(d, 0.000001, 10000000)
        return dx / d, dy / d

    def get_value(self, state: PolicyState) -> float:
        raise NotImplementedError(
            f"`get_value()` not implemented by {self.__class__.__name__} policy"
        )


class DTCJanosovHeuristicPolicy(DTCHeuristicPolicy):
    """Drone Team Capture Janosov Heuristic policy.

    Uses the heuristic from the following paper:

    janosov2017 is the policy from this work:
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
        arena = self.arena()
        coll = [0, 0]
        chase = self.attraction2(
            state.pursuer_coords[agent_idx],
            state.prev_pursuer_coords[agent_idx],
            state.target_coords,
            state.prev_target_coords,
        )

        inter = self.alignment2(
            state.pursuer_coords, state.prev_pursuer_coords, agent_idx
        )
        vx = arena[0] + coll[0] + chase[0] + inter[0]
        vy = arena[1] + coll[1] + chase[1] + inter[1]

        R = self.rot(state.pursuer_coords[agent_idx][2])
        Direction = R.dot([vx, vy])
        alpha = math.atan2(Direction[1], Direction[0])
        omega = self.pp(alpha)
        return np.array([omega], dtype=np.float32)

    def arena(self) -> List[float]:
        return [0.0, 0.0]

    def attraction2(
        self,
        pursuer_i: np.ndarray,
        pursuer_i_prev: np.ndarray,
        target: np.ndarray,
        target_prev: np.ndarray,
    ):
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

        chase_x = chase[0] * self.vel_pur
        chase_y = chase[1] * self.vel_pur

        return [chase_x, chase_y]

    def alignment2(self, pursuer, pursuer_prev, i):
        inte_x = 0
        inte_y = 0
        rad_inter = 250
        C_inter = 0.5
        C_f = 0.5
        for j in range(len(pursuer)):
            if j != i:
                d_ij = (np.array(pursuer[i]) - np.array(pursuer[j]))[:2]
                d = np.linalg.norm(d_ij)

                d_ij = self.normalise(d_ij[0], d_ij[1])

                vel_i = np.array(pursuer[i]) - np.array(pursuer_prev[i])
                vel_j = np.array(pursuer[j]) - np.array(pursuer_prev[j])

                rep_x = d_ij[0] * self.sat((d - rad_inter), -10000, 0) / d
                rep_y = d_ij[1] * self.sat((d - rad_inter), -10000, 0) / d

                fric_x = (vel_i[0] - vel_j[0]) / d**2
                fric_y = (vel_i[1] - vel_j[1]) / d**2

                inte_x += -rep_x + C_f * fric_x
                inte_y += -rep_y + C_f * fric_y

        inte_x, inte_y = self.normalise(inte_x, inte_y)

        dx = C_inter * inte_x * self.vel_pur
        dy = C_inter * inte_y * self.vel_pur

        return [dx, dy]

    def prediction(self, pursuer_i: np.ndarray, target: np.ndarray, vel_t: List[float]):
        dist = self.euclidean_dist(target, pursuer_i)

        tau = 20
        time_pred = dist / self.vel_tar
        time_pred = self.sat(time_pred, 0, tau)

        vel_t = self.normalise(vel_t[0], vel_t[1])  # type: ignore
        vel_x = vel_t[0] * self.vel_tar
        vel_y = vel_t[1] * self.vel_tar

        pos_pred_x = target[0] + vel_x * time_pred
        pos_pred_y = target[1] + vel_y * time_pred

        return [pos_pred_x, pos_pred_y]


class DTCAngelaniHeuristicPolicy(DTCHeuristicPolicy):
    """Drone Team Capture Angelani Heuristic policy.

    Uses the heuristic from the following paper:

    angelani2012 is the policy from this work:
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
        rep = self.repulsion(state.pursuer_coords, agent_idx)
        align = self.alignment(state.pursuer_coords, state.prev_pursuer_coords)
        atrac = self.attraction(state.pursuer_coords[agent_idx], state.target_coords)

        vx = rep[0] + align[0] + atrac[0]
        vy = rep[1] + align[1] + atrac[1]

        R = self.rot(state.pursuer_coords[agent_idx][2])
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
        n_pursuers = len(state.pursuer_coords)
        pursuer_i = state.pursuer_coords[self.agent_idx]
        for j in range(n_pursuers):
            if j != self.agent_idx:
                sense += self.delta(
                    pursuer_i, state.pursuer_coords[j], state.target_coords
                )

        alphaiT, _ = self.engagmment(pursuer_i, state.target_coords)
        omega_i = 0.6 * (alphaiT - sense * offset)
        omega_i = self.sat(omega_i, -self.omega_max, self.omega_max)
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
