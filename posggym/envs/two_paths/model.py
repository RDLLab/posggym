"""The POSG Model for the Two-Paths Problem """
import random
from typing import List, Tuple, Union

import numpy as np
from gym import spaces

import posggym.model as M

from posggym.envs.two_paths import obs as Z
from posggym.envs.two_paths import state as S
from posggym.envs.two_paths import belief as B
import posggym.envs.two_paths.grid as grid_lib

TPState = Tuple[grid_lib.Loc, grid_lib.Loc]
TPAction = int
TPJointAction = Tuple[TPAction, ...]
TPObs = np.ndarray


# TODO Change Actions to int
# TODO Change from returning TPState to a tuple
# TODO Change from return TPObs to a tuple


class TwoPathsModel(M.POSGModel):
    """Two-Paths Problem Model

    Parameters:
    -----------
    grid_name : str
        determines the environment map (see posggym.envs.two_paths.grid for
        available grids)
    action_probs : float or (float, float)
        the action success probability for each agent. By default the
        environment is deterministic (action_probs=1.0).
    infinite_horizon : bool
        whether problem should terminate once a terminal state is reached
        (default, False) or reset to start position and continue (True).

    """

    NUM_AGENTS = 2

    RUNNER_IDX = 0
    CHASER_IDX = 1

    R_ACTION = -1.0
    R_CAPTURE = -100.0    # Runner reward, Chaser = -R_CAPTURE
    R_SAFE = 100.0        # Runner reward, Chaser = -R_SAFE

    def __init__(self,
                 grid_name: str,
                 action_probs: Union[float, Tuple[float, float]] = 1.0,
                 infinite_horizon: bool = False,
                 **kwargs):
        super().__init__(self.NUM_AGENTS, **kwargs)
        self.grid = grid_lib.load_grid(grid_name)
        self._infinite_horizon = infinite_horizon
        self._action_spaces = tuple(
            [*grid_lib.DIRS] for _ in range(self.NUM_AGENTS)
        )

        if isinstance(action_probs, float):
            action_probs = (action_probs, action_probs)
        self._action_probs = action_probs

    @property
    def state_space(self) -> spaces.Space:
        n_locs = self.grid.n_locs
        return spaces.Tuple(
            tuple(spaces.Discrete(n_locs) for _ in range(self.n_agents))
        )

    @property
    def action_spaces(self) -> Tuple[spaces.Space, ...]:
        return tuple(
            spaces.Discrete(len(grid_lib.DIRS)) for _ in range(self.n_agents)
        )

    @property
    def obs_spaces(self) -> Tuple[spaces.Space, ...]:
        return tuple(
            spaces.Tuple((
                spaces.MultiDiscrete([len(Z.CELL_OBS)]*4),
                spaces.Discrete(2)
            ))
            for _ in range(self.n_agents)
        )

    @property
    def reward_ranges(self) -> Tuple[Tuple[M.Reward, M.Reward], ...]:
        return tuple(
            (self.R_CAPTURE, self.R_SAFE) for _ in range(self.n_agents)
        )

    @property
    def initial_belief(self) -> M.Belief:
        return B.TPB0(self.grid)

    def get_agent_initial_belief(self,
                                 agent_id: int,
                                 obs: M.Observation) -> M.Belief:
        return self.initial_belief

    def sample_initial_obs(self, state: M.State) -> M.JointObservation:
        assert isinstance(state, S.TPState)
        return self._get_obs(state, False)

    def step(self,
             state: M.State,
             actions: M.JointAction
             ) -> M.JointTimestep:
        assert isinstance(state, S.TPState)
        next_state = self._get_next_state(state, actions)
        rewards = self._get_rewards(next_state)
        done = terminal_obs = self._state_is_terminal(next_state)

        outcomes = self.get_outcome(next_state) if done else None

        if self._infinite_horizon and done:
            next_state = self.sample_initial_state()   # type: ignore
            done = False

        obs = self._get_obs(next_state, terminal_obs)

        return M.JointTimestep(next_state, obs, rewards, done, outcomes)

    def _get_next_state(self,
                        state: S.TPState,
                        action: M.JointAction) -> S.TPState:
        runner_loc, chaser_loc = state.runner_loc, state.chaser_loc
        runner_a = action[self.RUNNER_IDX].action_num
        chaser_a = action[self.CHASER_IDX].action_num

        if random.random() < self._action_probs[self.CHASER_IDX]:
            other_as = [a for a in grid_lib.DIRS if a != chaser_a]
            chaser_a = random.choice(other_as)
        chaser_next_loc = self.grid.get_next_loc(chaser_loc, chaser_a)

        if chaser_next_loc == runner_loc:
            # Runner considered to capture Fugitive
            runner_next_loc = runner_loc
        else:
            if random.random() < self._action_probs[self.RUNNER_IDX]:
                other_as = [a for a in grid_lib.DIRS if a != runner_a]
                runner_a = random.choice(other_as)
            runner_next_loc = self.grid.get_next_loc(runner_loc, runner_a)

        return S.TPState(runner_next_loc, chaser_next_loc, state.grid)

    def _get_obs(self,
                 next_s: S.TPState,
                 terminal: bool) -> M.JointObservation:
        runner_loc, chaser_loc = next_s.runner_loc, next_s.chaser_loc
        runner_obs = self._get_agent_obs(runner_loc, chaser_loc, terminal)
        chaser_obs = self._get_agent_obs(chaser_loc, runner_loc, terminal)
        return (runner_obs, chaser_obs)

    def _get_agent_obs(self,
                       agent_loc: grid_lib.Loc,
                       opponent_loc: grid_lib.Loc,
                       terminal: bool) -> Z.TPObs:
        adj_obs = self._get_adj_obs(agent_loc, opponent_loc)
        return Z.TPObs(adj_obs, terminal, self.grid)

    def _get_adj_obs(self,
                     loc: grid_lib.Loc,
                     opponent_loc: grid_lib.Loc) -> Tuple[int, ...]:
        width, height = self.grid.width, self.grid.height
        adj_locs = [
            loc-width if loc >= width else -1,             # N
            loc+width if loc < (height-1)*width else -1,   # S
            loc+1 if loc % width < width-1 else -1,        # E
            loc-1 if loc % width > 0 else -1               # W
        ]

        adj_obs = []
        for adj_loc in adj_locs:
            if adj_loc == opponent_loc:
                adj_obs.append(Z.OPPONENT)
            elif adj_loc == -1 or adj_loc in self.grid.block_locs:
                adj_obs.append(Z.WALL)
            else:
                adj_obs.append(Z.EMPTY)

        return tuple(adj_obs)

    def _get_rewards(self, state: S.TPState) -> Tuple[float, float]:
        chaser_loc, runner_loc = state.chaser_loc, state.runner_loc
        if runner_loc in self.grid.goal_locs:
            return (self.R_SAFE, -self.R_SAFE)
        if (
            runner_loc == chaser_loc
            or runner_loc in self.grid.get_neighbouring_locs(chaser_loc, True)
        ):
            return (self.R_CAPTURE, -self.R_CAPTURE)
        return (self.R_ACTION, self.R_ACTION)

    def is_done(self, state: M.State) -> bool:
        assert isinstance(state, S.TPState)
        if self._infinite_horizon:
            return False
        return self._state_is_terminal(state)

    def get_outcome(self, state: M.State) -> Tuple[M.Outcome, ...]:
        assert isinstance(state, S.TPState)
        if self._infinite_horizon:
            return (M.Outcome.NA, M.Outcome.NA)

        chaser_loc, runner_loc = state.chaser_loc, state.runner_loc
        if runner_loc in self.grid.goal_locs:
            return (M.Outcome.WIN, M.Outcome.LOSS)

        if (
            runner_loc == chaser_loc
            or runner_loc in self.grid.get_neighbouring_locs(chaser_loc, True)
        ):
            return (M.Outcome.LOSS, M.Outcome.WIN)

        return (M.Outcome.DRAW, M.Outcome.DRAW)

    def _state_is_terminal(self, state: S.TPState) -> bool:
        chaser_loc, runner_loc = state.chaser_loc, state.runner_loc
        if chaser_loc == runner_loc or runner_loc in self.grid.goal_locs:
            return True
        return runner_loc in self.grid.get_neighbouring_locs(chaser_loc, True)
