"""The POSG Model for the Two-Paths Problem."""
import random
import itertools
from typing import Tuple, Union, Sequence, Dict

from gym import spaces

import posggym.model as M

import posggym.envs.grid_world.two_paths.grid as grid_lib
from posggym.envs.grid_world.utils import Direction, Coord


TPState = Tuple[Coord, Coord]
TPAction = int
TPJointAction = Tuple[TPAction, ...]
# Obs = (adj_obs, terminal)
TPObs = Tuple[Tuple[int, int, int, int], int]
TPJointObs = Tuple[TPObs, ...]

# Cell obs
OPPONENT = 0
WALL = 1
EMPTY = 2

CELL_OBS = [OPPONENT, WALL, EMPTY]
CELL_OBS_STR = ["X", "#", "0"]


class TPB0(M.Belief):
    """The initial belief in a Two-Paths problem."""

    def __init__(self,
                 grid: grid_lib.TPGrid,
                 dist_res: int = 1000):
        self._grid = grid
        self._dist_res = dist_res

    def sample(self) -> M.State:
        return (self._grid.init_runner_coord, self._grid.init_chaser_coord)

    def sample_k(self, k: int) -> Sequence[M.State]:
        return [self.sample() for _ in range(k)]

    def get_dist(self) -> Dict[M.State, float]:
        s_0 = (self._grid.init_runner_coord, self._grid.init_chaser_coord)
        return {
            s: float(s == s_0)
            for s in itertools.product(self._grid.all_coords, repeat=2)
        }


class TwoPathsModel(M.POSGModel):
    """Two-Paths Problem Model.

    Parameters
    ----------
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

    R_ACTION = -0.01
    R_CAPTURE = -1.0    # Runner reward, Chaser = -R_CAPTURE
    R_SAFE = 1.0        # Runner reward, Chaser = -R_SAFE

    def __init__(self,
                 grid_name: str,
                 action_probs: Union[float, Tuple[float, float]] = 1.0,
                 infinite_horizon: bool = False,
                 **kwargs):
        super().__init__(self.NUM_AGENTS, **kwargs)
        self.grid = grid_lib.load_grid(grid_name)
        self._infinite_horizon = infinite_horizon

        if isinstance(action_probs, float):
            action_probs = (action_probs, action_probs)
        self._action_probs = action_probs

    @property
    def state_space(self) -> spaces.Space:
        return spaces.Tuple(
            tuple(
                spaces.Tuple((
                    spaces.Discrete(self.grid.width),
                    spaces.Discrete(self.grid.height)
                )) for _ in range(self.n_agents)
            )
        )

    @property
    def action_spaces(self) -> Tuple[spaces.Space, ...]:
        return tuple(
            spaces.Discrete(len(Direction)) for _ in range(self.n_agents)
        )

    @property
    def obs_spaces(self) -> Tuple[spaces.Space, ...]:
        return tuple(
            spaces.Tuple((
                spaces.Tuple((
                    spaces.Discrete(len(CELL_OBS)),
                    spaces.Discrete(len(CELL_OBS)),
                    spaces.Discrete(len(CELL_OBS)),
                    spaces.Discrete(len(CELL_OBS))
                )),
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
        return TPB0(self.grid)

    def get_agent_initial_belief(self,
                                 agent_id: int,
                                 obs: M.Observation) -> M.Belief:
        return self.initial_belief

    def sample_initial_obs(self, state: M.State) -> M.JointObservation:
        return self._get_obs(state, False)

    def step(self,
             state: M.State,
             actions: M.JointAction
             ) -> M.JointTimestep:
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
                        state: TPState,
                        actions: TPJointAction) -> TPState:
        runner_coord = state[self.RUNNER_IDX]
        chaser_coord = state[self.CHASER_IDX]
        runner_a = actions[self.RUNNER_IDX]
        chaser_a = actions[self.CHASER_IDX]

        if random.random() > self._action_probs[self.CHASER_IDX]:
            other_as = [a for a in Direction if a != chaser_a]
            chaser_a = random.choice(other_as)
        chaser_next_coord = self.grid.get_next_coord(
            chaser_coord, Direction(chaser_a), ignore_blocks=False
        )

        if chaser_next_coord == runner_coord:
            # Runner considered to capture Fugitive
            runner_next_coord = runner_coord
        else:
            if random.random() > self._action_probs[self.RUNNER_IDX]:
                other_as = [a for a in Direction if a != runner_a]
                runner_a = random.choice(other_as)
            runner_next_coord = self.grid.get_next_coord(
                runner_coord, Direction(runner_a), ignore_blocks=False
            )

        return (runner_next_coord, chaser_next_coord)

    def _get_obs(self,
                 state: TPState,
                 terminal: bool) -> M.JointObservation:
        runner_coord = state[self.RUNNER_IDX]
        chaser_coord = state[self.CHASER_IDX]
        runner_obs = self._get_agent_obs(runner_coord, chaser_coord, terminal)
        chaser_obs = self._get_agent_obs(chaser_coord, runner_coord, terminal)
        return (runner_obs, chaser_obs)

    def _get_agent_obs(self,
                       agent_coord: Coord,
                       opponent_coord: Coord,
                       terminal: bool) -> TPObs:
        adj_obs = self._get_adj_obs(agent_coord, opponent_coord)
        return (adj_obs, int(terminal))

    def _get_adj_obs(self,
                     coord: Coord,
                     opponent_coord: Coord) -> Tuple[int, int, int, int]:
        adj_obs = []
        for d in Direction:
            next_coord = self.grid.get_next_coord(coord, d, False)
            if next_coord == opponent_coord:
                adj_obs.append(OPPONENT)
            elif next_coord == coord:
                adj_obs.append(WALL)
            else:
                adj_obs.append(EMPTY)
        return tuple(adj_obs)   # type: ignore

    def _get_rewards(self, state: TPState) -> M.JointReward:
        runner_coord = state[self.RUNNER_IDX]
        chaser_coord = state[self.CHASER_IDX]
        if runner_coord in self.grid.goal_coords:
            return (self.R_SAFE, -self.R_SAFE)
        if (
            runner_coord == chaser_coord
            or runner_coord in self.grid.get_neighbours(chaser_coord, True)
        ):
            return (self.R_CAPTURE, -self.R_CAPTURE)
        return (self.R_ACTION, self.R_ACTION)

    def is_done(self, state: M.State) -> bool:
        if self._infinite_horizon:
            return False
        return self._state_is_terminal(state)

    def get_outcome(self, state: M.State) -> Tuple[M.Outcome, ...]:
        if self._infinite_horizon:
            return (M.Outcome.NA, M.Outcome.NA)

        runner_coord = state[self.RUNNER_IDX]
        chaser_coord = state[self.CHASER_IDX]
        if runner_coord in self.grid.goal_coords:
            return (M.Outcome.WIN, M.Outcome.LOSS)

        if (
            runner_coord == chaser_coord
            or runner_coord in self.grid.get_neighbours(chaser_coord, True)
        ):
            return (M.Outcome.LOSS, M.Outcome.WIN)
        return (M.Outcome.DRAW, M.Outcome.DRAW)

    def _state_is_terminal(self, state: TPState) -> bool:
        runner_coords = state[self.RUNNER_IDX]
        chaser_coords = state[self.CHASER_IDX]
        if (
            chaser_coords == runner_coords
            or runner_coords in self.grid.goal_coords
        ):
            return True
        neighbour_coords = self.grid.get_neighbours(chaser_coords, True)
        return runner_coords in neighbour_coords
