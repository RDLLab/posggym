"""The POSG Model for the Rock, Paper, Scissors problem."""
import random
from itertools import product
from typing import Tuple, Sequence, Dict, Optional

from gym import spaces

import posggym.model as M


RPSState = int
STATE0 = 0
STATES = [STATE0]
STATE_STRS = ["0"]

RPSAction = int
ROCK = 0
PAPER = 1
SCISSORS = 2
RPSJointAction = Tuple[RPSAction, ...]
ACTIONS = [ROCK, PAPER, SCISSORS]
ACTION_STR = ["R", "P", "S"]

RPSObs = int
RPSJointObs = Tuple[RPSObs, ...]
OBS_SPACE = ACTIONS
OBS_STR = ACTION_STR


class RPSB0(M.Belief):
    """The initial belief in a Rock, Paper, Scissors problem."""

    def sample(self) -> M.State:
        return STATE0

    def sample_k(self, k: int) -> Sequence[M.State]:
        return [STATE0] * k

    def get_dist(self) -> Dict[M.State, float]:
        return {STATE0: 1.0}


class RockPaperScissorsModel(M.POSGFullModel):
    """Rock, Paper, Scissors Model."""

    NUM_AGENTS = 2

    R_MATRIX = [
        [0, -1.0, 1.0],
        [1.0, 0, -1.0],
        [-1.0, 1.0, 0]
    ]

    def __init__(self, **kwargs):
        super().__init__(self.NUM_AGENTS, **kwargs)

        self._state_space = STATES
        self._action_spaces = tuple([*ACTIONS] for _ in range(self.n_agents))
        self._obs_spaces = tuple(OBS_SPACE for _ in range(self.n_agents))

        self._trans_map = self._construct_trans_func()
        self._rew_map = self._construct_rew_func()
        self._obs_map = self._construct_obs_func()

    @property
    def observation_first(self) -> bool:
        return False

    @property
    def is_symmetric(self) -> bool:
        return True

    @property
    def state_space(self) -> spaces.Space:
        return spaces.Discrete(len(STATES))

    @property
    def action_spaces(self) -> Tuple[spaces.Space, ...]:
        return tuple(
            spaces.Discrete(len(ACTIONS)) for _ in range(self.n_agents)
        )

    @property
    def observation_spaces(self) -> Tuple[spaces.Space, ...]:
        return tuple(
            spaces.Discrete(len(OBS_SPACE)) for _ in range(self.n_agents)
        )

    @property
    def reward_ranges(self) -> Tuple[Tuple[M.Reward, M.Reward], ...]:
        return tuple((-1.0, 1.0) for _ in range(self.n_agents))

    @property
    def initial_belief(self) -> M.Belief:
        return RPSB0()

    def get_agent_initial_belief(self,
                                 agent_id: M.AgentID,
                                 obs: M.Observation) -> M.Belief:
        return self.initial_belief

    def sample_initial_obs(self, state: M.State) -> M.JointObservation:
        return tuple(ROCK for _ in range(self.n_agents))

    def step(self,
             state: M.State,
             actions: M.JointAction
             ) -> M.JointTimestep:
        obs = (actions[1], actions[0])
        rewards = self._get_reward(actions)
        dones = (False,) * self.n_agents
        all_done = False
        outcomes = tuple(M.Outcome.NA for _ in range(self.n_agents))
        return M.JointTimestep(
            STATE0, obs, rewards, dones, all_done, outcomes
        )

    def _get_reward(self, actions: RPSJointAction) -> M.JointReward:
        return (
            self.R_MATRIX[actions[0]][actions[1]],
            self.R_MATRIX[actions[1]][actions[0]]
        )

    def set_seed(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def transition_fn(self,
                      state: M.State,
                      actions: M.JointAction,
                      next_state: M.State) -> float:
        return self._trans_map[(state, actions, next_state)]

    def _construct_trans_func(self) -> Dict:
        trans_map = {}
        for a in product(*self._action_spaces):
            trans_map[(STATE0, a, STATE0)] = 1.0
        return trans_map

    def observation_fn(self,
                       obs: M.JointObservation,
                       next_state: M.State,
                       actions: M.JointAction) -> float:
        return self._obs_map[(next_state, actions, obs)]

    def _construct_obs_func(self) -> Dict:
        obs_func = {}
        for (a, o) in product(
                product(*self._action_spaces),
                product(*self._obs_spaces)
        ):
            obs_func[(STATE0, a, o)] = 1.0 if a == o else 0.0
        return obs_func

    def reward_fn(self,
                  state: M.State,
                  actions: M.JointAction) -> M.JointReward:
        return self._rew_map[(state, actions)]

    def _construct_rew_func(self) -> Dict:
        rew_map = {}
        for a in product(*self._action_spaces):
            rew_map[(STATE0, a)] = self._get_reward(a)
        return rew_map
