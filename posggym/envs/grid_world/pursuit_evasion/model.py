"""The POSG Model for the Discrete Pursuit Evasion Problem."""
import random
from typing import Optional, Tuple, Union, Sequence, Dict, NamedTuple

from gym import spaces

import posggym.model as M

from posggym.envs.grid_world.core import Direction, Coord
import posggym.envs.grid_world.pursuit_evasion.grid as grid_lib

# State = (e_coord, e_dir, p_coord, p_dir, e_0_coord, p_0_coord, e_goal_coord)
INITIAL_DIR = Direction.NORTH


class PEState(NamedTuple):
    """Environment state in Pursuit Evastion problem."""
    evader_coord: Coord
    evader_dir: Direction
    pursuer_coord: Coord
    pursuer_dir: Direction
    evader_start_coord: Coord
    pursuer_start_coord: Coord
    evader_goal_coord: Coord
    min_goal_dist: int    # for evader


# Action = Direction of movement
# 0 = Forward, 1 = Backward, 2 = Left, 3 = Right
PEAction = int
PEJointAction = Tuple[PEAction, PEAction]

ACTION_TO_DIR = [
    # Forward
    [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST],
    # Backward
    [Direction.SOUTH, Direction.WEST, Direction.NORTH, Direction.EAST],
    # Left
    [Direction.WEST, Direction.NORTH, Direction.EAST, Direction.SOUTH],
    # Right
    [Direction.EAST, Direction.SOUTH, Direction.WEST, Direction.NORTH]
]

# E Obs = Tuple[WallObs, seen, heard, e_0_coord, p_0_coord, goal_coord]
#       = Tuple[Tuple[int, int, int, int, int, int], Coord, Coord, Coord]
# P Obs = Tuple[WallObs, seen, heard, e_0_coord, p_0_coord, blank_coord]
#       = Tuple[Tuple[int, int, int, int, int, int], Coord, Coord, Coord]
# Note, we use blank_coord for P Obs so Obs spaces are identical between the
# two agents. The blank_coord is always (0, 0).
PEEvaderObs = Tuple[Tuple[int, ...], Coord, Coord, Coord]
PEPursuerObs = Tuple[Tuple[int, ...], Coord, Coord, Coord]
PEJointObs = Tuple[PEEvaderObs, PEPursuerObs]


class PEB0(M.Belief):
    """The initial belief in a Pursuit-Evasion problem."""

    def __init__(self,
                 grid: grid_lib.PEGrid,
                 rng: random.Random,
                 evader_start_coord: Optional[Coord] = None,
                 pursuer_start_coord: Optional[Coord] = None,
                 goal_coord: Optional[Coord] = None,
                 dist_res: int = 1000):
        self._grid = grid
        self._rng = rng
        self._dist_res = dist_res
        self._evader_start_coord = evader_start_coord
        self._pursuer_start_coord = pursuer_start_coord
        self._goal_coord = goal_coord

    def sample(self) -> M.State:
        if self._evader_start_coord is None:
            evader_start_coord = self._rng.choice(
                self._grid.evader_start_coords
            )
        else:
            evader_start_coord = self._evader_start_coord

        if self._pursuer_start_coord is None:
            pursuer_start_coord = self._rng.choice(
                self._grid.pursuer_start_coords
            )
        else:
            pursuer_start_coord = self._pursuer_start_coord

        if self._goal_coord is None:
            goal_coord = self._rng.choice(
                self._grid.get_goal_coords(evader_start_coord)
            )
        else:
            goal_coord = self._goal_coord

        return PEState(
            evader_start_coord,
            INITIAL_DIR,
            pursuer_start_coord,
            INITIAL_DIR,
            evader_start_coord,
            pursuer_start_coord,
            goal_coord,
            self._grid.get_shortest_path_distance(
                evader_start_coord, goal_coord
            )
        )

    def sample_k(self, k: int) -> Sequence[M.State]:
        return [self.sample() for _ in range(k)]

    def get_dist(self) -> Dict[M.State, float]:
        return self.sample_belief_dist(self._dist_res)


class PursuitEvasionModel(M.POSGModel):
    """Discrete Pursuit-Evasion Model."""

    NUM_AGENTS = 2

    EVADER_IDX = 0
    PURSUER_IDX = 1

    # Evader-centric rewards
    R_PROGRESS = 0.01              # Reward making progress toward goal
    R_EVADER_ACTION = -0.01        # Reward each step for evader
    R_PURSUER_ACTION = -0.01       # Reward each step for pursuer
    R_CAPTURE = -1.0               # Reward for being captured
    R_EVASION = 1.0                # Reward for reaching goal

    FOV_EXPANSION_INCR = 3
    HEARING_DIST = 2

    def __init__(self,
                 grid_name: str,
                 action_probs: Union[float, Tuple[float, float]] = 1.0,
                 max_obs_distance: int = 12,
                 normalize_reward: bool = True,
                 use_progress_reward: bool = True,
                 **kwargs):
        super().__init__(self.NUM_AGENTS, **kwargs)
        self._grid_name = grid_name
        self.grid = grid_lib.load_grid(grid_name)

        if isinstance(action_probs, float):
            action_probs = (action_probs, action_probs)
        self._action_probs = action_probs

        self._max_obs_distance = max_obs_distance
        self._normalize_reward = normalize_reward
        self._use_progress_reward = use_progress_reward

        self._max_sp_distance = self.grid.get_max_shortest_path_distance()
        self._max_raw_return = self.R_EVASION
        if self._use_progress_reward:
            self._max_raw_return += (self._max_sp_distance+1) * self.R_PROGRESS
        self._min_raw_return = -self._max_raw_return

        self._rng = random.Random(None)

    @property
    def observation_first(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        return False

    @property
    def state_space(self) -> spaces.Space:
        # s = Tuple[Coord, Direction, Coord, Direction, Coord, Coord, Coord]
        return spaces.Tuple((
            # e_coord
            spaces.Tuple((
                spaces.Discrete(self.grid.width),
                spaces.Discrete(self.grid.height)
            )),
            # e_dir
            spaces.Discrete(len(Direction)),
            # p_coord
            spaces.Tuple((
                spaces.Discrete(self.grid.width),
                spaces.Discrete(self.grid.height)
            )),
            # p_dir
            spaces.Discrete(len(Direction)),
            # e_start_coord
            spaces.Tuple((
                spaces.Discrete(self.grid.width),
                spaces.Discrete(self.grid.height)
            )),
            # p_start_coord
            spaces.Tuple((
                spaces.Discrete(self.grid.width),
                spaces.Discrete(self.grid.height)
            )),
            # e_goal_coord
            spaces.Tuple((
                spaces.Discrete(self.grid.width),
                spaces.Discrete(self.grid.height)
            )),
            # Max shortest path distance
            spaces.Discrete(self._max_sp_distance)
        ))

    @property
    def action_spaces(self) -> Tuple[spaces.Space, ...]:
        return tuple(
            spaces.Discrete(len(Direction)) for _ in range(self.n_agents)
        )

    @property
    def observation_spaces(self) -> Tuple[spaces.Space, ...]:
        # Tuple[WallObs, seen , heard]
        env_obs_space = spaces.Tuple(
            tuple(spaces.Discrete(2) for _ in range(6))
        )
        coord_obs_space = spaces.Tuple((
            spaces.Discrete(self.grid.width),
            spaces.Discrete(self.grid.height)
        ))

        obs_space = spaces.Tuple((
            # Wall obs, seen, heard
            env_obs_space,
            # e_start_coord
            coord_obs_space,
            # p_start_coord
            coord_obs_space,
            # e_goal_coord/blank_coord
            coord_obs_space,
        ))
        return (obs_space, obs_space)

    @property
    def reward_ranges(self) -> Tuple[Tuple[M.Reward, M.Reward], ...]:
        max_reward = self.R_CAPTURE
        if self._use_progress_reward:
            max_reward += self.R_PROGRESS
        if self._normalize_reward:
            max_reward = self._get_normalized_reward(max_reward)
        return tuple((-max_reward, max_reward) for _ in range(self.n_agents))

    @property
    def initial_belief(self) -> M.Belief:
        return PEB0(self.grid, self._rng)

    def get_agent_initial_belief(self,
                                 agent_id: int,
                                 obs: M.Observation) -> M.Belief:
        if agent_id == self.EVADER_IDX:
            return PEB0(
                self.grid,
                self._rng,
                evader_start_coord=obs[1],
                pursuer_start_coord=obs[2],
                goal_coord=obs[3]
            )

        return PEB0(
                self.grid,
                self._rng,
                evader_start_coord=obs[1],
                pursuer_start_coord=obs[2]
            )

    def sample_initial_obs(self, state: M.State) -> M.JointObservation:
        return self._get_obs(state)[0]

    def step(self,
             state: M.State,
             actions: M.JointAction
             ) -> M.JointTimestep:
        next_state = self._get_next_state(state, actions)  # type: ignore
        obs, evader_detected = self._get_obs(next_state)
        rewards = self._get_reward(state, next_state, evader_detected)
        dones = self._is_done(next_state)
        all_done = all(dones)

        if all_done:
            outcomes = self._get_outcome(next_state)
        else:
            outcomes = (M.Outcome.NA, ) * self.n_agents

        return M.JointTimestep(
            next_state, obs, rewards, dones, all_done, outcomes
        )

    def _get_next_state(self,
                        state: PEState,
                        actions: PEJointAction) -> PEState:
        evader_a = actions[self.EVADER_IDX]
        pursuer_a = actions[self.PURSUER_IDX]

        if (
            self._action_probs[self.EVADER_IDX] < 1.0
            and self._rng.random() > self._action_probs[self.EVADER_IDX]
        ):
            other_as = [a for a in range(len(Direction)) if a != evader_a]
            evader_a = self._rng.choice(other_as)

        if (
            self._action_probs[self.PURSUER_IDX]
            and self._rng.random() > self._action_probs[self.PURSUER_IDX]
        ):
            other_as = [a for a in range(len(Direction)) if a != pursuer_a]
            pursuer_a = self._rng.choice(other_as)

        pursuer_next_dir = Direction(
            ACTION_TO_DIR[pursuer_a][state.pursuer_dir]
        )
        pursuer_next_coord = self.grid.get_next_coord(
            state.pursuer_coord, pursuer_next_dir, ignore_blocks=False
        )

        evader_next_coord = state.evader_coord
        evader_next_dir = Direction(
            ACTION_TO_DIR[evader_a][state.evader_dir]
        )
        if pursuer_next_coord != state.evader_coord:
            evader_next_coord = self.grid.get_next_coord(
                state.evader_coord, evader_next_dir, ignore_blocks=False
            )

        min_sp_distance = min(
            state.min_goal_dist,
            self.grid.get_shortest_path_distance(
                evader_next_coord, state.evader_goal_coord
            )
        )

        return PEState(
            evader_next_coord,
            evader_next_dir,
            pursuer_next_coord,
            pursuer_next_dir,
            state.evader_start_coord,
            state.pursuer_start_coord,
            state.evader_goal_coord,
            min_sp_distance
        )

    def _get_obs(self, state: PEState) -> Tuple[PEJointObs, bool]:
        evader_obs = self._get_evader_obs(state)
        pursuer_obs, seen = self._get_pursuer_obs(state)
        return (evader_obs, pursuer_obs), seen

    def _get_evader_obs(self, state: PEState) -> PEEvaderObs:
        walls, seen, heard = self._get_agent_obs(
            state.evader_coord, state.evader_dir, state.pursuer_coord
        )
        return (
            (*walls, seen, heard),
            state.evader_start_coord,
            state.pursuer_start_coord,
            state.evader_goal_coord
        )

    def _get_pursuer_obs(self, state: PEState) -> Tuple[PEPursuerObs, bool]:
        walls, seen, heard = self._get_agent_obs(
            state.pursuer_coord, state.pursuer_dir, state.evader_coord
        )
        obs = (
            (*walls, seen, heard),
            state.evader_start_coord,
            state.pursuer_start_coord,
            # pursuer doesn't observe evader goal coord
            (0, 0)
        )
        return obs, bool(seen)

    def _get_agent_obs(self,
                       agent_coord: Coord,
                       agent_dir: Direction,
                       opp_coord: Coord
                       ) -> Tuple[Tuple[int, ...], int, int]:
        walls = self._get_wall_obs(agent_coord)
        seen = self._get_opponent_seen(agent_coord, agent_dir, opp_coord)
        heard = self._get_opponent_heard(agent_coord, opp_coord)
        return walls, int(seen), int(heard)

    def _get_wall_obs(self, coord: Coord) -> Tuple[int, ...]:
        adj_coords = self.grid.get_neighbours(
            coord, ignore_blocks=True, include_out_of_bounds=True
        )
        return tuple(
            int(
                not self.grid.coord_in_bounds(coord)
                or coord in self.grid.block_coords
            ) for coord in adj_coords
        )

    def _get_opponent_seen(self,
                           ego_coord: Coord,
                           ego_dir: Direction,
                           opp_coord: Coord) -> bool:
        fov = self.grid.get_fov(
            ego_coord, ego_dir, self.FOV_EXPANSION_INCR, self._max_obs_distance
        )
        return opp_coord in fov

    def _get_opponent_heard(self,
                            ego_coord: grid_lib.Coord,
                            opp_coord: grid_lib.Coord) -> bool:
        dist = self.grid.manhattan_dist(ego_coord, opp_coord)
        return dist <= self.HEARING_DIST

    def _get_reward(self,
                    state: PEState,
                    next_state: PEState,
                    evader_seen: bool) -> M.JointReward:
        evader_coord = next_state.evader_coord
        pursuer_coord = next_state.pursuer_coord
        evader_goal_coord = next_state.evader_goal_coord

        evader_reward = 0
        if (
            self._use_progress_reward
            and next_state.min_goal_dist < state.min_goal_dist
        ):
            evader_reward += self.R_PROGRESS

        if evader_coord == pursuer_coord or evader_seen:
            evader_reward += self.R_CAPTURE
        elif evader_coord == evader_goal_coord:
            evader_reward += self.R_EVASION

        if self._normalize_reward:
            evader_reward = self._get_normalized_reward(evader_reward)

        return (evader_reward, -evader_reward)

    def _is_done(self, state: M.State) -> Tuple[bool, ...]:
        evader_coord, pursuer_coord = state.evader_coord, state.pursuer_coord
        evader_goal_coord = state.evader_goal_coord
        pursuer_dir = state.pursuer_dir
        if (
            evader_coord == pursuer_coord
            or evader_coord == evader_goal_coord
            or self._get_opponent_seen(
                pursuer_coord, pursuer_dir, evader_coord
            )
        ):
            return (True, True)
        return (False, False)

    def _get_outcome(self, state: M.State) -> Tuple[M.Outcome, ...]:
        # Assuming this method is called on final timestep
        evader_coord, pursuer_coord = state.evader_coord, state.pursuer_coord
        evader_goal_coord = state.evader_goal_coord
        pursuer_dir = state.pursuer_dir

        # check this first before relatively expensive detection check
        if evader_coord == pursuer_coord:
            return (M.Outcome.LOSS, M.Outcome.WIN)
        if evader_coord == evader_goal_coord:
            return (M.Outcome.WIN, M.Outcome.LOSS)

        if self._get_opponent_seen(pursuer_coord, pursuer_dir, evader_coord):
            return (M.Outcome.LOSS, M.Outcome.WIN)
        return (M.Outcome.DRAW, M.Outcome.DRAW)

    def set_seed(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def _get_normalized_reward(self, reward: float) -> float:
        """Normalize reward in [-1, 1] interval."""
        diff = (self._max_raw_return - self._min_raw_return)
        return 2 * (reward - self._min_raw_return) / diff - 1
