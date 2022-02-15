"""The POSG Model for the Discrete Pursuit Evasion Problem """
import random
from typing import Optional, Tuple, Union, Sequence, Dict

from gym import spaces

import posggym.model as M

from posggym.envs.grid_world.utils import Direction, Coord
import posggym.envs.grid_world.pursuit_evasion.grid as grid_lib

# State = (e_coord, e_dir, p_coord, p_dir, e_0_coord, p_0_coord, e_goal_coord)
INITIAL_DIR = Direction.NORTH
PEState = Tuple[Coord, Direction, Coord, Direction, Coord, Coord, Coord]

# Action = Direction
PEAction = int
PEJointAction = Tuple[PEAction, PEAction]

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
    """The initial belief in a Pursuit-Evasion problem """

    def __init__(self,
                 grid: grid_lib.PEGrid,
                 evader_start_coord: Optional[Coord] = None,
                 pursuer_start_coord: Optional[Coord] = None,
                 goal_coord: Optional[Coord] = None,
                 dist_res: int = 1000):
        self._grid = grid
        self._dist_res = dist_res
        self._evader_start_coord = evader_start_coord
        self._pursuer_start_coord = pursuer_start_coord
        self._goal_coord = goal_coord

    def sample(self) -> M.State:
        if self._evader_start_coord is None:
            evader_start_coord = random.choice(self._grid.evader_start_coords)
        else:
            evader_start_coord = self._evader_start_coord

        if self._pursuer_start_coord is None:
            pursuer_start_coord = random.choice(
                self._grid.pursuer_start_coords
            )
        else:
            pursuer_start_coord = self._pursuer_start_coord

        if self._goal_coord is None:
            goal_coord = random.choice(
                self._grid.get_goal_coords(evader_start_coord)
            )
        else:
            goal_coord = self._goal_coord

        return (
            evader_start_coord,
            INITIAL_DIR,
            pursuer_start_coord,
            INITIAL_DIR,
            evader_start_coord,
            pursuer_start_coord,
            goal_coord,
        )

    def sample_k(self, k: int) -> Sequence[M.State]:
        return [self.sample() for _ in range(k)]

    def get_dist(self) -> Dict[M.State, float]:
        return self.sample_belief_dist(self._dist_res)


class PursuitEvasionModel(M.POSGModel):
    """Discrete Pursuit-Evasion Model """

    NUM_AGENTS = 2

    EVADER_IDX = 0
    PURSUER_IDX = 1

    R_EVADER_ACTION = -1.0        # Reward each step for evader
    R_PURSUER_ACTION = -1.0        # Reward each step for pursuer
    R_CAPTURE = 100.0             # Pursuer reward for capturing evader
    R_EVASION = 100.0             # Evader reward for reaching goal

    FOV_EXPANSION_INCR = 3
    HEARING_DIST = 2

    def __init__(self,
                 grid_name: str,
                 action_probs: Union[float, Tuple[float, float]] = 1.0,
                 **kwargs):
        super().__init__(self.NUM_AGENTS, **kwargs)
        self._grid_name = grid_name
        self.grid = grid_lib.load_grid(grid_name)

        if isinstance(action_probs, float):
            action_probs = (action_probs, action_probs)
        self._action_probs = action_probs

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
        ))

    @property
    def action_spaces(self) -> Tuple[spaces.Space, ...]:
        return tuple(
            spaces.Discrete(len(Direction)) for _ in range(self.n_agents)
        )

    @property
    def obs_spaces(self) -> Tuple[spaces.Space, ...]:
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
        return tuple(
            (-self.R_CAPTURE, self.R_CAPTURE) for _ in range(self.n_agents)
        )

    @property
    def initial_belief(self) -> M.Belief:
        return PEB0(self.grid)

    def get_agent_initial_belief(self,
                                 agent_id: int,
                                 obs: M.Observation) -> M.Belief:
        if agent_id == self.EVADER_IDX:
            return PEB0(
                self.grid,
                evader_start_coord=obs[1],
                pursuer_start_coord=obs[2],
                goal_coord=obs[3]
            )

        return PEB0(
                self.grid,
                evader_start_coord=obs[1],
                pursuer_start_coord=obs[2]
            )

    def sample_initial_obs(self, state: M.State) -> M.JointObservation:
        obs, _ = self._get_obs(state)
        return obs

    def step(self,
             state: M.State,
             actions: M.JointAction
             ) -> M.JointTimestep:
        next_state = self._get_next_state(state, actions)  # type: ignore
        obs, evader_detected = self._get_obs(next_state)
        reward = self._get_reward(next_state, evader_detected)
        done = self.is_done(next_state)
        outcomes = self.get_outcome(next_state) if done else None
        return M.JointTimestep(next_state, obs, reward, done, outcomes)

    def _get_next_state(self,
                        state: PEState,
                        actions: PEJointAction) -> PEState:
        evader_a = actions[self.EVADER_IDX]
        pursuer_a = actions[self.PURSUER_IDX]

        if random.random() > self._action_probs[self.EVADER_IDX]:
            other_as = [a for a in Direction if a != evader_a]
            evader_a = random.choice(other_as)

        if random.random() > self._action_probs[self.PURSUER_IDX]:
            other_as = [a for a in Direction if a != pursuer_a]
            pursuer_a = random.choice(other_as)

        pursuer_next_dir = Direction(pursuer_a)
        pursuer_next_coord = self.grid.get_next_coord(
            state[2], pursuer_next_dir, ignore_blocks=False
        )

        evader_next_coord = state[0]
        evader_next_dir = Direction(evader_a)
        if pursuer_next_coord != state[0]:
            evader_next_coord = self.grid.get_next_coord(
                state[0], evader_next_dir, ignore_blocks=False
            )

        return (
            evader_next_coord,
            evader_next_dir,
            pursuer_next_coord,
            pursuer_next_dir,
            state[4],   # evader start coord
            state[5],   # pursuer start coord
            state[6]    # goal coord
        )

    def _get_obs(self, state: PEState) -> Tuple[PEJointObs, bool]:
        evader_obs = self._get_evader_obs(state)
        pursuer_obs, seen = self._get_pursuer_obs(state)
        return (evader_obs, pursuer_obs), seen

    def _get_evader_obs(self, state: PEState) -> PEEvaderObs:
        walls, seen, heard = self._get_agent_obs(state[0], state[1], state[2])
        return ((*walls, seen, heard), state[4], state[5], state[6])

    def _get_pursuer_obs(self, state: PEState) -> Tuple[PEPursuerObs, bool]:
        walls, seen, heard = self._get_agent_obs(state[2], state[3], state[0])
        return ((*walls, seen, heard), state[4], state[5], (0, 0)), bool(seen)

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
                self.grid.coord_in_bounds(coord)
                or coord in self.grid.block_coords
            ) for coord in adj_coords
        )

    def _get_opponent_seen(self,
                           ego_coord: Coord,
                           ego_dir: Direction,
                           opp_coord: Coord) -> bool:
        fov = self.grid.get_fov(ego_coord, ego_dir, self.FOV_EXPANSION_INCR)
        return opp_coord in fov

    def _get_opponent_heard(self,
                            ego_coord: grid_lib.Coord,
                            opp_coord: grid_lib.Coord) -> bool:
        dist = self.grid.manhattan_dist(ego_coord, opp_coord)
        return dist <= self.HEARING_DIST

    def _get_reward(self, state: PEState, evader_seen: bool) -> M.JointReward:
        evader_coord, pursuer_coord = state[0], state[2]
        evader_goal_coord = state[6]

        if evader_coord == pursuer_coord or evader_seen:
            return (-self.R_CAPTURE, self.R_CAPTURE)
        if evader_coord == evader_goal_coord:
            return (self.R_EVASION, -self.R_EVASION)

        return (self.R_EVADER_ACTION, self.R_PURSUER_ACTION)

    def is_done(self, state: M.State) -> bool:
        evader_coord, pursuer_coord = state[0], state[2]
        evader_goal_coord = state[6]
        if evader_coord == pursuer_coord:
            return True
        if evader_coord == evader_goal_coord:
            return True
        return self._get_opponent_seen(pursuer_coord, state[3], evader_coord)

    def get_outcome(self, state: M.State) -> Tuple[M.Outcome, ...]:
        # Assuming this method is called on final timestep
        evader_coord, pursuer_coord = state[0], state[2]
        evader_goal_coord = state[6]

        # check this first before relatively expensive detection check
        if evader_coord == pursuer_coord:
            return (M.Outcome.LOSS, M.Outcome.WIN)
        if evader_coord == evader_goal_coord:
            return (M.Outcome.WIN, M.Outcome.LOSS)

        if self._get_opponent_seen(pursuer_coord, state[3], evader_coord):
            return (M.Outcome.LOSS, M.Outcome.WIN)
        return (M.Outcome.DRAW, M.Outcome.DRAW)
