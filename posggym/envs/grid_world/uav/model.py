"""The POSG Model for the MA Unmaned Aerial Vehicle Problem."""
import random
from typing import Optional, List, Tuple, Sequence, Dict

from gym import spaces

import posggym.model as M

import posggym.envs.grid_world.uav.grid as grid_lib
from posggym.envs.grid_world.core import Direction, Coord


UAVState = Tuple[Coord, Coord]
UAVAction = int
UAVJointAction = Tuple[UAVAction, UAVAction]

# UAV Obs = (uav coord, fug coord)
UAVUAVObs = Tuple[Coord, Coord]
# FUG Obs = house direction
UAVFUGObs = int
UAVJointObs = Tuple[UAVUAVObs, UAVFUGObs]

OBSNORTH = 0
OBSSOUTH = 1
OBSLEVEL = 2
OBSNONE = 3
DIR_OBS = [OBSNORTH, OBSSOUTH, OBSLEVEL, OBSNONE]
DIR_OBS_STR = ["N", "S", "L", "0"]

# Observatio Accuracy for each agent
FUG_OBS_ACC = 0.8
UAV_OBS_ACC = 0.9


class UAVB0(M.Belief):
    """The initial belief in a UAV problem."""

    def __init__(self,
                 grid: grid_lib.UAVGrid,
                 rng: random.Random,
                 uav_obs: Optional[UAVUAVObs] = None,
                 fug_obs: Optional[UAVFUGObs] = None,
                 dist_res: int = 1000):
        # either both are None or one is None
        assert (
            (uav_obs is None and fug_obs is None)
            or not (uav_obs is not None and fug_obs is not None)
        )
        self._grid = grid
        self._rng = rng
        self._dist_res = dist_res
        self._uav_obs = uav_obs
        self._fug_obs = fug_obs

        if self._fug_obs is not None:
            valid_fug_coords_dist = self._get_fug_coord_dist(self._fug_obs)
        else:
            valid_fug_coords_dist = ([], [])
        self._valid_fug_coords_dist = valid_fug_coords_dist

    def sample(self) -> M.State:
        if self._uav_obs is not None:
            return self._sample_uav()
        if self._fug_obs is not None:
            return self._sample_fug()
        return self._sample()

    def _sample(self) -> M.State:
        uav_coord = self._rng.choice(self._grid.init_uav_coords)
        fug_start_coords = list(self._grid.init_fug_coords)
        if uav_coord in fug_start_coords:
            fug_start_coords.remove(uav_coord)
        fug_coord = self._rng.choice(fug_start_coords)
        return uav_coord, fug_coord

    def _sample_uav(self) -> M.State:
        uav_coord, fug_coord = self._uav_obs   # type: ignore

        if self._rng.random() < UAV_OBS_ACC:
            # UAV_OBS_ACC probability obs loc is correct
            true_fug_coord = fug_coord
        else:
            fug_start_coords = list(self._grid.init_fug_coords)
            if uav_coord in fug_start_coords:
                fug_start_coords.remove(uav_coord)
            true_fug_coord = self._rng.choice(fug_start_coords)

        return uav_coord, true_fug_coord

    def _sample_fug(self) -> M.State:
        valid_coords, cum_probs = self._valid_fug_coords_dist
        fug_coord = self._rng.choices(
            valid_coords, cum_weights=cum_probs, k=1
        )[0]

        uav_start_coords = list(self._grid.init_uav_coords)
        if fug_coord in uav_start_coords:
            uav_start_coords.remove(fug_coord)
        uav_coord = self._rng.choice(uav_start_coords)

        return uav_coord, fug_coord

    def _get_fug_coord_dist(self,
                            obs: UAVFUGObs) -> Tuple[List[Coord], List[float]]:
        house_adj_coords = self._grid.get_neighbours(
            self._grid.safe_house_coord, ignore_blocks=False
        )

        if obs == OBSNONE:
            valid_fug_coords = set(self._grid.init_fug_coords)
            valid_fug_coords.difference_update(house_adj_coords)
            p = 1.0 / len(valid_fug_coords)
            dist = []
            for i in range(len(valid_fug_coords)):
                if i == 0:
                    dist.append(p)
                else:
                    dist.append(dist[-1]+p)
            return list(valid_fug_coords), dist

        if obs == OBSNORTH:
            p = (1.0 - FUG_OBS_ACC) / (len(house_adj_coords)-1)
            true_coords = [house_adj_coords[Direction.SOUTH]]
        elif obs == OBSSOUTH:
            p = (1.0 - FUG_OBS_ACC) / (len(house_adj_coords)-1)
            true_coords = [house_adj_coords[Direction.NORTH]]
        elif obs == OBSLEVEL:
            p = (1.0 - FUG_OBS_ACC) / (len(house_adj_coords)-2)
            true_coords = [
                house_adj_coords[Direction.EAST],
                house_adj_coords[Direction.WEST]
            ]

        dist = []
        num_adj, num_true = len(house_adj_coords), len(true_coords)
        for coord in house_adj_coords:
            if coord in true_coords:
                dist.append(FUG_OBS_ACC / num_true)
            else:
                dist.append((1.0-FUG_OBS_ACC) / (num_adj - num_true))
        return house_adj_coords, dist

    def sample_k(self, k: int) -> Sequence[M.State]:
        return [self.sample() for _ in range(k)]

    def get_dist(self) -> Dict[M.State, float]:
        return self.sample_belief_dist(self._dist_res)


class UAVModel(M.POSGModel):
    """Unmaned Aerial Vehicle Problem Model."""

    NUM_AGENTS = 2

    UAV_IDX = 0
    FUG_IDX = 1

    R_ACTION = -0.04
    R_CAPTURE = 1.0    # UAV reward, fugitive = -R_CAPTURE
    R_SAFE = -1.0      # UAV reward, fugitive = -R_SAFE

    def __init__(self,
                 grid_name: str,
                 **kwargs):
        super().__init__(self.NUM_AGENTS, **kwargs)
        self.grid = grid_lib.load_grid(grid_name)
        self._rng = random.Random(None)

    @property
    def observation_first(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        return False

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
    def observation_spaces(self) -> Tuple[spaces.Space, ...]:
        coord_space = spaces.Tuple((
            spaces.Discrete(self.grid.width),
            spaces.Discrete(self.grid.height)
        ))
        return (
            # UAV Obs = (uav coords, fug coords)
            spaces.Tuple((coord_space, coord_space)),
            # FUG ubs = (safe house dir)
            spaces.Discrete(len(DIR_OBS))
        )

    @property
    def reward_ranges(self) -> Tuple[Tuple[M.Reward, M.Reward], ...]:
        return tuple(
            (self.R_SAFE, self.R_CAPTURE) for _ in range(self.n_agents)
        )

    @property
    def initial_belief(self) -> M.Belief:
        return UAVB0(self.grid, self._rng)

    def get_agent_initial_belief(self,
                                 agent_id: int,
                                 obs: M.Observation) -> M.Belief:
        if agent_id == self.UAV_IDX:
            return UAVB0(self.grid, self._rng, uav_obs=obs)

        house_adj_coords = self.grid.get_neighbours(
            self.grid.safe_house_coord, ignore_blocks=False
        )
        if len(house_adj_coords) != 4:
            # Doesn't work for 3x3 grid
            raise NotImplementedError(
                "Fugitive observation conditioned belief is only supported "
                "for maps where safe house can be reached from all directions."
            )
        return UAVB0(self.grid, self._rng, fug_obs=obs)

    def sample_initial_obs(self, state: M.State) -> M.JointObservation:
        return self._sample_obs(state)

    def step(self,
             state: M.State,
             actions: M.JointAction
             ) -> M.JointTimestep:
        next_state = self._sample_next_state(state, actions)  # type: ignore
        rewards = self._get_reward(next_state)

        uav_coord, fug_coord = next_state
        if fug_coord in (uav_coord, self.grid.safe_house_coord):
            # fug caught or fug safe
            fug_coord = self._sample_fug_coord(uav_coord)
            next_state = (uav_coord, fug_coord)

        outcomes = None
        obs = self._sample_obs(next_state)
        dones = (False,) * self.n_agents
        outcomes = (M.Outcome.NA,) * self.n_agents
        return M.JointTimestep(
            next_state, obs, rewards, dones, all(dones), outcomes
        )

    def _sample_next_state(self,
                           state: UAVState,
                           actions: UAVJointAction) -> UAVState:
        uav_a, fug_a = actions
        uav_coord, fug_coord = state

        uav_next_coord = self.grid.get_next_coord(
            uav_coord, Direction(uav_a)
        )

        # fugitive reseting is handled in step function
        # to allow for
        if uav_next_coord == fug_coord:
            # UAV considered to capture Fugitive
            fug_next_coord = fug_coord
        else:
            fug_next_coord = self.grid.get_next_coord(
                fug_coord, Direction(fug_a)
            )

        return uav_next_coord, fug_next_coord

    def _sample_fug_coord(self, uav_coord: Coord) -> Coord:
        fug_start_coords = list(self.grid.init_fug_coords)
        if uav_coord in fug_start_coords:
            fug_start_coords.remove(uav_coord)
        return self._rng.choice(fug_start_coords)

    def _sample_obs(self, next_state: UAVState) -> UAVJointObs:
        uav_obs = self._sample_uav_obs(next_state)
        fug_obs = self._sample_fug_obs(next_state)
        return uav_obs, fug_obs

    def _sample_uav_obs(self, next_state: UAVState) -> UAVUAVObs:
        uav_coord, fug_coord = next_state

        if (
            fug_coord == self.grid.safe_house_coord
            or self._rng.random() < UAV_OBS_ACC
        ):
            fug_coord_obs = fug_coord
        else:
            adj_coords = self.grid.get_neighbours(fug_coord)
            if uav_coord in adj_coords:
                adj_coords.remove(uav_coord)
            fug_coord_obs = self._rng.choice(adj_coords)

        return uav_coord, fug_coord_obs

    def _sample_fug_obs(self, next_state: UAVState) -> UAVFUGObs:
        fug_coord = next_state[1]
        fug_adj_coords = self.grid.get_neighbours(
            fug_coord, ignore_blocks=True, include_out_of_bounds=True
        )

        safe_house_coord = self.grid.safe_house_coord
        if fug_adj_coords[0] == safe_house_coord:
            true_obs = OBSSOUTH
        elif fug_adj_coords[1] == safe_house_coord:
            true_obs = OBSNORTH
        elif safe_house_coord in fug_adj_coords[2:]:
            true_obs = OBSLEVEL
        else:
            true_obs = OBSNONE

        if true_obs == OBSNONE:
            return true_obs

        if self._rng.random() < FUG_OBS_ACC:
            return true_obs
        return self._rng.choice([OBSNORTH, OBSSOUTH, OBSLEVEL])

    def _get_reward(self, next_state: UAVState) -> M.JointReward:
        uav_coord, fug_coord = next_state
        rewards = [self.R_ACTION, self.R_ACTION]
        if fug_coord == self.grid.safe_house_coord:
            rewards[self.UAV_IDX] = self.R_SAFE
            rewards[self.FUG_IDX] = -self.R_SAFE
        elif fug_coord == uav_coord:
            rewards[self.UAV_IDX] = self.R_CAPTURE
            rewards[self.FUG_IDX] = -self.R_CAPTURE
        return tuple(rewards)

    def set_seed(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
