"""The POSG Model for the Discrete Pursuit Evasion Problem """
import random
from typing import Optional, Tuple, Union, Sequence, Dict

from gym import spaces

import posggym.model as M

from posggym.envs.grid_world.utils import Direction, Coord
import posggym.envs.grid_world.pursuit_evasion.grid as grid_lib

# State = (r_coord, r_dir, c_coord, c_dir, r_start_coord,
#          c_start_coord, r_goal_coord)
INITIAL_DIR = Direction.NORTH
PEState = Tuple[Coord, Direction, Coord, Direction, Coord, Coord, Coord]

# Action = Direction
PEAction = int
PEJointAction = Tuple[PEAction, PEAction]

# R Obs = Tuple[WallObs, seen, heard, goal_coord, r_start_coord, c_start_coord]
#            = Tuple[Tuple[int, int, int, int, int, int], Coord, Coord, Coord]
# C Obs = Tuple[WallObs, seen, heard, r_start_coord, c_start_coord]
#       = Tuple[Tuple[int, int, int, int, int, int], Coord, Coord]
PERunnerObs = Tuple[Tuple[int, ...], Coord, Coord, Coord]
PEChaserObs = Tuple[Tuple[int, ...], Coord, Coord]
PEJointObs = Tuple[PERunnerObs, PEChaserObs]


class PEB0(M.Belief):
    """The initial belief in a Pursuit-Evasion problem """

    def __init__(self,
                 grid: grid_lib.PEGrid,
                 goal_coord: Optional[Coord] = None,
                 runner_start_coord: Optional[Coord] = None,
                 chaser_start_coord: Optional[Coord] = None,
                 dist_res: int = 1000):
        self._grid = grid
        self._dist_res = dist_res
        self._goal_coord = goal_coord
        self._runner_start_coord = runner_start_coord
        self._chaser_start_coord = chaser_start_coord

    def sample(self) -> M.State:
        if self._runner_start_coord is None:
            runner_start_coord = random.choice(self._grid.runner_start_coords)
        else:
            runner_start_coord = self._runner_start_coord

        if self._chaser_start_coord is None:
            chaser_start_coord = random.choice(self._grid.chaser_start_coords)
        else:
            chaser_start_coord = self._chaser_start_coord

        if self._goal_coord is None:
            goal_coord = random.choice(
                self._grid.get_goal_coords(runner_start_coord)
            )
        else:
            goal_coord = self._goal_coord

        return (
            runner_start_coord,
            INITIAL_DIR,
            chaser_start_coord,
            INITIAL_DIR,
            goal_coord,
            runner_start_coord,
            chaser_start_coord
        )

    def sample_k(self, k: int) -> Sequence[M.State]:
        return [self.sample() for _ in range(k)]

    def get_dist(self) -> Dict[M.State, float]:
        return self.sample_belief_dist(self._dist_res)


class PursuitEvasionModel(M.POSGModel):
    """Discrete Pursuit-Evasion Model """

    NUM_AGENTS = 2

    RUNNER_IDX = 0
    CHASER_IDX = 1

    R_RUNNER_ACTION = -1.0        # Reward each step for runner
    R_CHASER_ACTION = -1.0        # Reward each step for chaser
    R_CAPTURE = 100.0             # Chaser reward for capturing runner
    R_EVASION = 100.0             # Runner reward for reaching goal

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
        # s = (r_coord, r_dir, c_coord, c_dir, r_goal_coord,
        #      r_start_coord, c_start_coord)
        #   = Tuple[Coord, Direction, Coord, Direction, Coord, Coord, Coord]
        return spaces.Tuple((
            # r_coord
            spaces.Tuple((
                spaces.Discrete(self.grid.width),
                spaces.Discrete(self.grid.height)
            )),
            # r_dir
            spaces.Discrete(len(Direction)),
            # c_coord
            spaces.Tuple((
                spaces.Discrete(self.grid.width),
                spaces.Discrete(self.grid.height)
            )),
            # c_dir
            spaces.Discrete(len(Direction)),
            # r_goal_coord
            spaces.Tuple((
                spaces.Discrete(self.grid.width),
                spaces.Discrete(self.grid.height)
            )),
            # r_start_coord
            spaces.Tuple((
                spaces.Discrete(self.grid.width),
                spaces.Discrete(self.grid.height)
            )),
            # r_goal_coord
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

        runner_obs_space = spaces.Tuple((
            # Wall obs, seen, heard
            env_obs_space,
            # r_goal_coord
            coord_obs_space,
            # r_start_coord
            coord_obs_space,
            # c_start_coord
            coord_obs_space
        ))

        chaser_obs_space = spaces.Tuple((
            # Wall obs, seen, heard
            env_obs_space,
            # r_start_coord
            coord_obs_space,
            # c_start_coord
            coord_obs_space
        ))
        return (runner_obs_space, chaser_obs_space)

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
        if agent_id == self.RUNNER_IDX:
            return PEB0(
                self.grid,
                goal_coord=obs[1],
                runner_start_coord=obs[2],
                chaser_start_coord=obs[3]
            )

        return PEB0(
                self.grid,
                runner_start_coord=obs[2],
                chaser_start_coord=obs[3]
            )

    def sample_initial_obs(self, state: M.State) -> M.JointObservation:
        obs, _ = self._get_obs(state)
        return obs

    def step(self,
             state: M.State,
             actions: M.JointAction
             ) -> M.JointTimestep:
        next_state = self._get_next_state(state, actions)  # type: ignore
        obs, runner_detected = self._get_obs(next_state)
        reward = self._get_reward(next_state, runner_detected)
        done = self.is_done(next_state)
        outcomes = self.get_outcome(next_state) if done else None
        return M.JointTimestep(next_state, obs, reward, done, outcomes)

    def _get_next_state(self,
                        state: PEState,
                        actions: PEJointAction) -> PEState:
        runner_a = actions[self.RUNNER_IDX]
        chaser_a = actions[self.CHASER_IDX]

        if random.random() < self._action_probs[self.RUNNER_IDX]:
            other_as = [a for a in Direction if a != runner_a]
            runner_a = random.choice(other_as)

        if random.random() < self._action_probs[self.CHASER_IDX]:
            other_as = [a for a in Direction if a != chaser_a]
            chaser_a = random.choice(other_as)

        chaser_next_dir = Direction(chaser_a)
        chaser_next_coord = self.grid.get_next_coord(
            state[2], chaser_next_dir, ignore_blocks=False
        )

        runner_next_coord = state[0]
        runner_next_dir = Direction(runner_a)
        if chaser_next_coord != state[0]:
            runner_next_coord = self.grid.get_next_coord(
                state[0], runner_next_dir, ignore_blocks=False
            )

        return (
            runner_next_coord,
            runner_next_dir,
            chaser_next_coord,
            chaser_next_dir,
            state[4],   # goal coord
            state[5],   # runner start coord
            state[6]    # chaser start coord
        )

    def _get_obs(self, state: PEState) -> Tuple[PEJointObs, bool]:
        runner_obs = self._get_runner_obs(state)
        chaser_obs, seen = self._get_chaser_obs(state)
        return (runner_obs, chaser_obs), seen

    def _get_runner_obs(self, state: PEState) -> PERunnerObs:
        walls, seen, heard = self._get_agent_obs(state[0], state[1], state[2])
        return ((*walls, seen, heard), state[4], state[5], state[6])

    def _get_chaser_obs(self, state: PEState) -> Tuple[PEChaserObs, bool]:
        walls, seen, heard = self._get_agent_obs(state[2], state[3], state[0])
        return ((*walls, seen, heard), state[5], state[6]), bool(seen)

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

    def _get_reward(self, state: PEState, runner_seen: bool) -> M.JointReward:
        runner_coord, chaser_coord = state[0], state[2]
        runner_goal_coord = state[4]

        if runner_coord == chaser_coord or runner_seen:
            return (-self.R_CAPTURE, self.R_CAPTURE)
        if runner_coord == runner_goal_coord:
            return (self.R_EVASION, -self.R_EVASION)

        return (self.R_RUNNER_ACTION, self.R_CHASER_ACTION)

    def is_done(self, state: M.State) -> bool:
        runner_coord, chaser_coord = state[0], state[2]
        runner_goal_coord = state[4]
        if runner_coord == chaser_coord:
            return True
        if runner_coord == runner_goal_coord:
            return True
        return self._get_opponent_seen(chaser_coord, state[3], runner_coord)

    def get_outcome(self, state: M.State) -> Tuple[M.Outcome, ...]:
        # Assuming this method is called on final timestep
        runner_coord, chaser_coord = state[0], state[2]
        runner_goal_coord = state[4]

        # check this first before relatively expensive detection check
        if runner_coord == chaser_coord:
            return (M.Outcome.LOSS, M.Outcome.WIN)
        if runner_coord == runner_goal_coord:
            return (M.Outcome.WIN, M.Outcome.LOSS)

        if self._get_opponent_seen(chaser_coord, state[3], runner_coord):
            return (M.Outcome.LOSS, M.Outcome.WIN)
        return (M.Outcome.DRAW, M.Outcome.DRAW)
