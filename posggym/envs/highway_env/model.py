"""The model for Highway-Env.

Ref: https://github.com/eleurent/highway-env

"""
from typing import Tuple, NamedTuple, Optional

from gym import spaces

import posggym.model as M

from highway_env.utils import Vector
from highway_env.envs.common.abstract import AbstractEnv


class HWVehicleState(NamedTuple):
    """State of a vehicle in the HighwayEnv."""
    pos: Vector
    heading: float
    speed: float
    crashed: bool
    hit: bool
    impact: Optional[Vector]
    on_road: bool


HWState = Tuple[HWVehicleState, ...]


def get_env_state(env: AbstractEnv) -> HWState:
    """Get state of HighwayEnv."""
    state = []
    for v in env.road.vehicles:
        v_state = HWVehicleState(
            pos=v.position,
            heading=v.heading,
            speed=v.speed,
            crashed=v.crashed,
            hit=v.hit,
            impact=v.impact,
            on_road=v.on_road
        )
        state.append(v_state)
    return tuple(state)


def set_env_state(env: AbstractEnv, state: HWState):
    """Set state of HighwayEnv."""
    for v, v_state in zip(env.road.vehicles, state):
        v.position = v_state.pos
        v.heading = v_state.heading
        v.speed = v_state.speed
        v.crashed = v_state.crashed
        v.hit = v_state.hit
        v.impact = v_state.impact
        # don't set on_road since this is dynamically computed from position
        # in the Vehicle class


class HWBelief(M.Belief):
    """The initial belief for HighwayEnv."""

    def __init__(self, env: AbstractEnv):
        self._env = env

    def sample(self) -> M.State:
        self._env.reset()
        return get_env_state(self._env)


class HWModel(M.POSGModel):
    """HighwayEnv model."""

    def __init__(self, n_agents: int, env: AbstractEnv, **kwargs):
        super().__init__(n_agents, **kwargs)
        self._env = env
        if "seed" in kwargs:
            self.set_seed(kwargs["seed"])

    @property
    def observation_first(self) -> bool:
        return True

    @property
    def state_space(self) -> spaces.Space:
        raise NotImplementedError

    @property
    def action_spaces(self) -> Tuple[spaces.Space, ...]:
        return self._env.action_space

    @property
    def observation_spaces(self) -> Tuple[spaces.Space, ...]:
        return self._env.observation_space

    @property
    def reward_ranges(self) -> Tuple[Tuple[M.Reward, M.Reward], ...]:
        return ((0.0, 1.0), ) * self.n_agents

    @property
    def initial_belief(self) -> HWBelief:
        return HWBelief(self._env)

    def step(self,
             state: M.State,
             actions: M.JointAction) -> M.JointTimestep:
        set_env_state(self._env, state)
        obs, rewards, dones, info = self._env.step(actions)
        next_state = get_env_state(self._env)
        all_done = all(dones)

        if all_done:
            outcomes = self._get_outcomes(next_state)
        else:
            outcomes = (M.Outcome.NA,) * self.n_agents

        return M.JointTimestep(
            next_state, obs, rewards, dones, all(dones), outcomes
        )

    def _get_outcomes(self, state: HWState) -> Tuple[M.Outcome]:
        # Assumes function is called from final step of an episode
        # An episode is considered a WIN for an agent if they didn't crash or
        # (depending on scenario) go off the road
        outcomes = []
        offroad_terminal = self._env.config["offroad_terminal"]
        for i in range(self.n_agents):
            if state[i].crashed or (offroad_terminal and not state[i].on_road):
                outcome_i = M.Outcome.LOSS
            else:
                outcome_i = M.Outcome.WIN
            outcomes.append(outcome_i)
        return tuple(outcomes)

    def set_seed(self, seed: Optional[int] = None):
        self._env.seed(seed)

    def sample_initial_obs(self, state: M.State) -> M.JointObservation:
        set_env_state(self._env, state)
        obs = self._env.observation_type.observe()
        return obs

    def get_agent_initial_belief(self,
                                 agent_id: M.AgentID,
                                 obs: M.Observation) -> M.Belief:
        raise NotImplementedError
