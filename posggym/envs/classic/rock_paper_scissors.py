"""The classic Rock Paper Scissors problem.

This is the classic game of rock, paper, scissors (RPS).

This scenario involves two agents. Each step both agents choose an action
out of 'ROCK', 'PAPER' or 'SCISSORS' and are rewarded based on the actions
taken in comparison to their opponent.
"""
import sys
from itertools import product
from typing import Dict, List, Optional, SupportsFloat, Tuple

from gymnasium import spaces

import posggym.model as M
from posggym.core import DefaultEnv
from posggym.utils import seeding


RPSState = int
STATE0 = 0
STATES = [STATE0]
STATE_STRS = ["0"]

RPSAction = int
ROCK = 0
PAPER = 1
SCISSORS = 2
ACTIONS = [ROCK, PAPER, SCISSORS]
ACTION_STR = ["R", "P", "S"]

RPSObs = int
RPSJointObs = Tuple[RPSObs, ...]
OBS_SPACE = ACTIONS
OBS_STR = ACTION_STR


class RockPaperScissorsEnv(DefaultEnv):
    """The Rock Paper Scissors Environment.

    This is the classic game of rock, paper, scissors (RPS).

    This scenario involves two agents. Each step both agents choose an action
    out of 'ROCK', 'PAPER' or 'SCISSORS' and are rewarded based on the actions
    taken in comparison to their opponent.

    State
    -----
    There is only a single state in RPS, the None state.

    Actions
    -------
    A_1 = A_2 = {`ROCK`, `PAPER`, `SCISSORS`}

    Observation
    -----------
    Agents observe the last action played by their opponent.

    O_1 = O_2 = {`ROCK`, `PAPER`, `SCISSORS`}

    Reward
    ------
    Agents are rewarded based on the following pay-off matrix (shows pay-off
    for row agent):

             | ROCK     | PAPER    | SCISSORS |
    -------------------------------------------
    ROCK     | 0        | -1       | 1        |
    PAPER    | 1        | 0        | -1       |
    SCISSORS | -1       | 1        | 0        |

    Transition Dynamics
    -------------------
    There is only a single state so the transition function is the identity
    function.

    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, **kwargs):
        super().__init__(RockPaperScissorsModel(**kwargs), render_mode=render_mode)

    def render(self, mode: str = "human"):
        assert self._last_obs is not None

        obs_str = ", ".join([OBS_STR[o] for o in self._last_obs.values()])
        output = [f"Step: {self._step_num}", f"Obs: <{obs_str}>"]
        if self._last_actions is not None:
            action_str = ", ".join([ACTION_STR[a] for a in self._last_actions.values()])
            output.insert(1, f"Actions: <{action_str}>")
            output.append(f"Rewards: <{self._last_rewards}>")

        output_str = "\n".join(output) + "\n"

        if self.render_mode == "human":
            sys.stdout.write(output_str)
        else:
            # ansi mode
            return output_str


class RockPaperScissorsModel(M.POSGFullModel[RPSState, RPSObs, RPSAction]):
    """Rock, Paper, Scissors Model."""

    NUM_AGENTS = 2

    R_MATRIX = [[0, -1.0, 1.0], [1.0, 0, -1.0], [-1.0, 1.0, 0]]

    def __init__(self, **kwargs):
        self.possible_agents = tuple(range(self.NUM_AGENTS))
        self.state_space = spaces.Discrete(len(STATES))
        self.action_spaces = {
            i: spaces.Discrete(len(ACTIONS)) for i in self.possible_agents
        }
        self.observation_spaces = {
            i: spaces.Discrete(len(OBS_SPACE)) for i in self.possible_agents
        }

        # Spaces used internally
        self._state_space = STATES
        self._action_spaces = tuple(
            [*ACTIONS] for _ in range(len(self.possible_agents))
        )
        self._obs_spaces = tuple(OBS_SPACE for _ in range(len(self.possible_agents)))
        self.observation_first = False
        self.is_symmetric = True

        self._trans_map = self._construct_trans_func()
        self._rew_map = self._construct_rew_func()
        self._obs_map = self._construct_obs_func()

    @property
    def reward_ranges(self) -> Dict[M.AgentID, Tuple[SupportsFloat, SupportsFloat]]:
        return {i: (-1.0, 1.0) for i in self.possible_agents}

    @property
    def rng(self) -> seeding.RNG:
        if self._rng is None:
            self._rng, seed = seeding.std_random()
        return self._rng

    def get_agents(self, state: RPSState) -> List[M.AgentID]:
        return list(self.possible_agents)

    def sample_initial_state(self) -> RPSState:
        return STATE0

    def step(
        self, state: RPSState, actions: Dict[M.AgentID, RPSAction]
    ) -> M.JointTimestep[RPSState, RPSObs]:
        assert all(a_i in ACTIONS for a_i in actions.values())
        obs: Dict[M.AgentID, RPSObs] = {0: actions[1], 1: actions[0]}
        rewards = self._get_reward(actions)
        terminated = {i: False for i in self.possible_agents}
        truncated = {i: False for i in self.possible_agents}
        all_done = False
        info: Dict[M.AgentID, Dict] = {i: {} for i in self.possible_agents}
        return M.JointTimestep(
            STATE0, obs, rewards, terminated, truncated, all_done, info
        )

    def _get_reward(
        self, actions: Dict[M.AgentID, RPSAction]
    ) -> Dict[M.AgentID, SupportsFloat]:
        return {
            0: self.R_MATRIX[actions[0]][actions[1]],
            1: self.R_MATRIX[actions[1]][actions[0]],
        }

    def get_initial_belief(self) -> Dict[RPSState, float]:
        return {STATE0: 1.0}

    def transition_fn(
        self, state: RPSState, actions: Dict[M.AgentID, RPSAction], next_state: RPSState
    ) -> float:
        action_tuple = tuple(actions[i] for i in self.possible_agents)
        return self._trans_map[(state, action_tuple, next_state)]

    def _construct_trans_func(self) -> Dict:
        trans_map = {}
        for a in product(*self._action_spaces):
            trans_map[(STATE0, a, STATE0)] = 1.0
        return trans_map

    def observation_fn(
        self,
        obs: Dict[M.AgentID, RPSObs],
        next_state: RPSState,
        actions: Dict[M.AgentID, RPSAction],
    ) -> float:
        obs_tuple = tuple(obs[i] for i in self.possible_agents)
        action_tuple = tuple(actions[i] for i in self.possible_agents)
        return self._obs_map[(next_state, action_tuple, obs_tuple)]

    def _construct_obs_func(self) -> Dict:
        obs_func = {}
        for (a, o) in product(
            product(*self._action_spaces), product(*self._obs_spaces)
        ):
            obs_func[(STATE0, a, o)] = 1.0 if a == o else 0.0
        return obs_func

    def reward_fn(
        self, state: RPSState, actions: Dict[M.AgentID, RPSAction]
    ) -> Dict[M.AgentID, SupportsFloat]:
        action_tuple = tuple(actions[i] for i in self.possible_agents)
        return self._rew_map[(state, action_tuple)]

    def _construct_rew_func(self) -> Dict:
        rew_map = {}
        for a in product(*self._action_spaces):
            rew_map[(STATE0, a)] = self._get_reward(dict(enumerate(a)))
        return rew_map
