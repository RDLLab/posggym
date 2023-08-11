"""The Unmanned Aerial Vehicle Grid World Environment."""
import random
from os import path
from typing import Dict, List, Optional, Set, Tuple, Union

from gymnasium import spaces

import posggym.model as M
from posggym import logger
from posggym.core import DefaultEnv
from posggym.envs.grid_world.core import Coord, Direction, Grid
from posggym.utils import seeding

UAVState = Tuple[Coord, Coord]
UAVAction = int

# UAV Obs = (uav coord, fug coord)
UAVUAVObs = Tuple[Coord, Coord]
# FUG Obs = house direction
UAVFUGObs = int
UAVObs = Union[UAVUAVObs, UAVFUGObs]

OBSNORTH = 0
OBSSOUTH = 1
OBSLEVEL = 2
OBSNONE = 3
DIR_OBS = [OBSNORTH, OBSSOUTH, OBSLEVEL, OBSNONE]
DIR_OBS_STR = ["N", "S", "L", "0"]


class UAVEnv(DefaultEnv[UAVState, UAVObs, UAVAction]):
    """The Unmanned Aerial Vehicle Grid World Environment.

    An adversarial 2D grid world problem involving two agents, a Unmanned
    Aerial Vehicle (UAV) and a fugitive. The UAV's goal is to capture the
    fugitive, while the fugitive's goal is to reach the safe house located at
    a known fixed location on the grid. The fugitive is considered caught if
    it is co-located with the UAV. The UAV observes it's own location and
    receives a noisy observation of the fugitive's location. The fugitive does
    not know it's location but it receives a noisy observation of its relative
    direction to the safe house when it is adjacent to the safe house.

    Posgible Agents
    ---------------
    - UAV = '0'
    - Fugitive = '1'

    State Space
    -----------
    Each state contains the `(x, y)` (x=column, y=row, with origin at the
    top-left square of the grid) of the UAV and fugitive agents. Specifically,
    a states is `((x_uav, y_uav), (x_fugitive, y_fugitive))`.

    Action Space
    ------------
    Each agent has 4 actions corresponding to moving in the 4 cardinal
    directions: `NORTH=0`, `EAST=1`, `SOUTH=2`, `WEST=3`.

    Observation Space
    -----------------
    The UAV observes it's `(x, y)` coordinates and receives a noisy observation
    of the fugitives `(x, y)` coordinates. The UAV observes the correct fugitive
    coordinates with probability `p=0.9`, and one of the adjacent locations to the true
    fugitive location with `p=1-0.9`.

    The fugitive can sense it's position with respect to the safe house, namely
    whether it is north of it (`OBSNORTH=0`), south of it (`OBSSOUTH=1`), or at the
    same level (`OBSLEVEL=3`). These observations are received with accuracy of `0.8`,
    and only when the fugitive is adjacent to it. If the fugitive is not
    adjacent to the safe house it receives no observation (`OBSNONE=4`).

    Rewards
    -------
    Both agents receive a penalty of `-0.04` for each step.
    If the fugitive reaches the safe house then the fugitive receives a reward of `1`,
    while the UAV receives a penalty of `-1`.
    If the fugitive is caught by the UAV, then the fugitive receives a penalty of `-1`,
    while the UAV receives a reward of `1`.

    Dynamics
    --------
    Actions are deterministic, so agents move into the next cell in the actions
    direction if it is not out-of-bounds. The fugitive's position is reset at random if
    it reaches the safe house or gets caught by the UAV.

    Starting State
    --------------
    Initially, the location of both agents is chosen at random, while the safe house
    is always located in the same location depending on grid layout.

    Episode End
    -----------
    Since the fugitive is reset each time it is caught. Episodes only end when the step
    limit is reached, as specified by  `max_episode_steps` when initializing the
    environment with `posggym.make` (default=`50`).

    Arguments
    ---------

    - `grid` - the grid of the environment. This can be an integer specifying
        the width and height of the grid, in which case an empty grid with the given
        dimensions and default position for the safe house will be used. Alternatively,
        it can be a `UAVGrid` instance. (default = `5`).

    Problem Sizes
    -------------

    For reference, the table below contains the size of state, and observation
    spaces for different grid sizes.

    | Grid size | States | Observations UAV | Observations Fugitive |
    |-----------|--------|------------------|-----------------------|
    | `3x3`     | 81     | 81               | 4                     |
    | `4x4`     | 256    | 256              | 4                     |
    | `5x5`     | 625    | 625              | 4                     |
    | `6x6`     | 1296   | 1296             | 4                     |

    Version History
    ---------------
    - `v0`: Initial version

    Reference
    ---------
    Panella, Alessandro, and Piotr Gmytrasiewicz. 2017. “Interactive POMDPs
    with Finite-State Models of Other Agents.” Autonomous Agents and
    Multi-Agent Systems 31 (4): 861–904.

    """

    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 15}

    def __init__(
        self, grid: Union["UAVGrid", int] = 5, render_mode: Optional[str] = None
    ):
        super().__init__(UAVModel(grid), render_mode=render_mode)
        self.renderer = None
        self.uav_img = None
        self.fug_img = None

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. posggym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        if self.render_mode == "ansi":
            return self._render_ansi()
        return self._render_gui()

    def _render_ansi(self):
        grid: UAVGrid = self.model.grid  # type: ignore
        grid_str = grid.get_ascii_repr(self._state[0], self._state[1])

        output = [
            f"Step: {self._step_num}",
            grid_str,
        ]
        if self._last_actions is not None:
            action_str = ", ".join(
                [str(Direction(a)) for a in self._last_actions.values()]
            )
            output.insert(1, f"Actions: <{action_str}>")
            output.append(f"Rewards: <{self._last_rewards}>")
        return "\n".join(output) + "\n"

    def _render_gui(self):
        grid: UAVGrid = self.model.grid  # type: ignore

        import posggym.envs.grid_world.render as render_lib

        if self.renderer is None:
            self.renderer = render_lib.GWRenderer(
                self.render_mode,
                grid,
                render_fps=self.metadata["render_fps"],
                env_name="UAV",
                bg_color=(255, 255, 255),
                grid_line_color=(0, 0, 0),
            )

            # add house to static objects list
            img_path = path.join(path.dirname(__file__), "img", "house.png")
            img = render_lib.load_img_file(img_path, self.renderer.cell_size)
            house_img = render_lib.GWImage(
                grid.safe_house_coord, self.renderer.cell_size, img
            )
            self.renderer.static_objects.append(house_img)

        if self.uav_img is None:
            img_path = path.join(path.dirname(__file__), "img", "drone.png")
            img = render_lib.load_img_file(img_path, self.renderer.cell_size)
            self.uav_img = render_lib.GWImage((0, 0), self.renderer.cell_size, img)

        if self.fug_img is None:
            img_path = path.join(path.dirname(__file__), "img", "robber.png")
            img = render_lib.load_img_file(img_path, self.renderer.cell_size)
            self.fug_img = render_lib.GWImage((0, 0), self.renderer.cell_size, img)

        self.uav_img.coord = self._state[0]
        self.fug_img.coord = self._state[1]

        return self.renderer.render([self.uav_img, self.fug_img])

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class UAVModel(M.POSGModel[UAVState, UAVObs, UAVAction]):
    """Unmanned Aerial Vehicle Problem Model."""

    NUM_AGENTS = 2

    UAV_ID = "0"
    FUG_ID = "1"

    R_ACTION = -0.04
    R_CAPTURE = 1.0  # UAV reward, fugitive = -R_CAPTURE
    R_SAFE = -1.0  # UAV reward, fugitive = -R_SAFE

    # Observatio Accuracy for each agent
    FUG_OBS_ACC = 0.8
    UAV_OBS_ACC = 0.9

    def __init__(self, grid: Union["UAVGrid", int]):
        if isinstance(grid, int):
            assert grid >= 3, "Grid size must be >= 3."
            # grid specified size of grid,
            grid = UAVGrid(grid, grid, None)
        self.grid = grid
        self._rng = random.Random(None)

        self.possible_agents = tuple(str(i) for i in range(self.NUM_AGENTS))
        self.state_space = spaces.Tuple(
            tuple(
                spaces.Tuple(
                    (
                        spaces.Discrete(self.grid.width),
                        spaces.Discrete(self.grid.height),
                    )
                )
                for _ in range(len(self.possible_agents))
            )
        )
        self.action_spaces = {
            i: spaces.Discrete(len(Direction)) for i in self.possible_agents
        }
        coord_space = spaces.Tuple(
            (spaces.Discrete(self.grid.width), spaces.Discrete(self.grid.height))
        )
        self.observation_spaces = {
            # UAV Obs = (uav coords, fug coords)
            self.UAV_ID: spaces.Tuple((coord_space, coord_space)),
            # FUG ubs = (safe house dir)
            self.FUG_ID: spaces.Discrete(len(DIR_OBS)),
        }
        self.is_symmetric = False

        # cache for sampling obs conditioned init state for fug
        self._cached_init_fug_obs: Optional[UAVFUGObs] = None
        self._valid_fug_coords_dist: Tuple[List[Coord], List[float]] = ([], [])

    @property
    def reward_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {i: (self.R_SAFE, self.R_CAPTURE) for i in self.possible_agents}

    @property
    def rng(self) -> seeding.RNG:
        if self._rng is None:
            self._rng, seed = seeding.std_random()
        return self._rng

    def get_agents(self, state: UAVState) -> List[str]:
        return list(self.possible_agents)

    def sample_initial_state(self) -> UAVState:
        uav_coord = self.rng.choice(self.grid.init_uav_coords)
        fug_start_coords = list(self.grid.init_fug_coords)
        if uav_coord in fug_start_coords:
            fug_start_coords.remove(uav_coord)
        fug_coord = self.rng.choice(fug_start_coords)
        return uav_coord, fug_coord

    def sample_agent_initial_state(self, agent_id: str, obs: UAVObs) -> UAVState:
        if agent_id == self.UAV_ID:
            assert isinstance(obs, tuple)
            return self._sample_uav_obs(obs)  # type: ignore

        assert isinstance(obs, int)
        house_adj_coords = self.grid.get_neighbours(
            self.grid.safe_house_coord, ignore_blocks=False
        )
        if len(house_adj_coords) != 4:
            # Doesn't work for 3x3 grid
            raise NotImplementedError(
                "Sampling observation conditioned initial state for the fugitive is "
                "only supported for maps where safe house can be reached from all "
                "directions."
            )
        return self._sample_fug(obs)

    def _sample_uav_initial_state(self, uav_obs: UAVUAVObs) -> UAVState:
        uav_coord, fug_coord = uav_obs
        if self.rng.random() < self.UAV_OBS_ACC:
            # UAV_OBS_ACC probability obs loc is correct
            true_fug_coord = fug_coord
        else:
            fug_start_coords = list(self.grid.init_fug_coords)
            if uav_coord in fug_start_coords:
                fug_start_coords.remove(uav_coord)
            true_fug_coord = self.rng.choice(fug_start_coords)
        return uav_coord, true_fug_coord

    def _sample_fug(self, fug_obs: UAVFUGObs) -> UAVState:
        if self._cached_init_fug_obs is None or self._cached_init_fug_obs != fug_obs:
            self._valid_fug_coords_dist = self._get_fug_coord_dist(fug_obs)

        valid_coords, cum_probs = self._valid_fug_coords_dist
        fug_coord = self.rng.choices(  # type: ignore
            valid_coords, cum_weights=cum_probs, k=1
        )[0]
        uav_start_coords = list(self.grid.init_uav_coords)
        if fug_coord in uav_start_coords:
            uav_start_coords.remove(fug_coord)
        uav_coord = self.rng.choice(uav_start_coords)
        return uav_coord, fug_coord

    def _get_fug_coord_dist(self, obs: UAVFUGObs) -> Tuple[List[Coord], List[float]]:
        house_adj_coords = self.grid.get_neighbours(
            self.grid.safe_house_coord, ignore_blocks=False
        )

        if obs == OBSNONE:
            valid_fug_coords = set(self.grid.init_fug_coords)
            valid_fug_coords.difference_update(house_adj_coords)
            p = 1.0 / len(valid_fug_coords)
            dist = []
            for i in range(len(valid_fug_coords)):
                if i == 0:
                    dist.append(p)
                else:
                    dist.append(dist[-1] + p)
            return list(valid_fug_coords), dist

        true_coords = []
        if obs == OBSNORTH:
            p = (1.0 - self.FUG_OBS_ACC) / (len(house_adj_coords) - 1)
            true_coords = [house_adj_coords[Direction.SOUTH]]
        elif obs == OBSSOUTH:
            p = (1.0 - self.FUG_OBS_ACC) / (len(house_adj_coords) - 1)
            true_coords = [house_adj_coords[Direction.NORTH]]
        elif obs == OBSLEVEL:
            p = (1.0 - self.FUG_OBS_ACC) / (len(house_adj_coords) - 2)
            true_coords = [
                house_adj_coords[Direction.EAST],
                house_adj_coords[Direction.WEST],
            ]

        dist = []
        num_adj, num_true = len(house_adj_coords), len(true_coords)
        for coord in house_adj_coords:
            if coord in true_coords:
                dist.append(self.FUG_OBS_ACC / num_true)
            else:
                dist.append((1.0 - self.FUG_OBS_ACC) / (num_adj - num_true))
        return house_adj_coords, dist

    def sample_initial_obs(self, state: UAVState) -> Dict[str, UAVObs]:
        return self._sample_obs(state)

    def step(
        self, state: UAVState, actions: Dict[str, UAVAction]
    ) -> M.JointTimestep[UAVState, UAVObs]:
        assert all(0 <= a_i < len(Direction) for a_i in actions.values())
        next_state = self._sample_next_state(state, actions)
        rewards = self._get_reward(next_state)

        uav_coord, fug_coord = next_state
        if fug_coord in (uav_coord, self.grid.safe_house_coord):
            # fug caught or fug safe
            fug_coord = self._sample_fug_coord(uav_coord)
            next_state = (uav_coord, fug_coord)

        obs = self._sample_obs(next_state)
        terminated = {i: False for i in self.possible_agents}
        truncated = {i: False for i in self.possible_agents}
        all_done = False
        info: Dict[str, Dict] = {i: {} for i in self.possible_agents}
        return M.JointTimestep(
            next_state, obs, rewards, terminated, truncated, all_done, info
        )

    def _sample_next_state(
        self, state: UAVState, actions: Dict[str, UAVAction]
    ) -> UAVState:
        uav_a, fug_a = actions[self.UAV_ID], actions[self.FUG_ID]
        uav_coord, fug_coord = state
        uav_next_coord = self.grid.get_next_coord(uav_coord, Direction(uav_a))
        # fugitive resetting is handled in step function
        # to allow for
        if uav_next_coord == fug_coord:
            # UAV considered to capture Fugitive
            fug_next_coord = fug_coord
        else:
            fug_next_coord = self.grid.get_next_coord(fug_coord, Direction(fug_a))
        return uav_next_coord, fug_next_coord

    def _sample_fug_coord(self, uav_coord: Coord) -> Coord:
        fug_start_coords = list(self.grid.init_fug_coords)
        if uav_coord in fug_start_coords:
            fug_start_coords.remove(uav_coord)
        return self.rng.choice(fug_start_coords)

    def _sample_obs(self, state: UAVState) -> Dict[str, UAVObs]:
        return {
            self.UAV_ID: self._sample_uav_obs(state),
            self.FUG_ID: self._sample_fug_obs(state),
        }

    def _sample_uav_obs(self, state: UAVState) -> UAVUAVObs:
        uav_coord, fug_coord = state
        if (
            fug_coord == self.grid.safe_house_coord
            or self.rng.random() < self.UAV_OBS_ACC
        ):
            fug_coord_obs = fug_coord
        else:
            adj_coords = self.grid.get_neighbours(fug_coord)
            if uav_coord in adj_coords:
                adj_coords.remove(uav_coord)
            fug_coord_obs = self.rng.choice(adj_coords)
        return uav_coord, fug_coord_obs

    def _sample_fug_obs(self, state: UAVState) -> UAVFUGObs:
        fug_coord = state[1]
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

        if self.rng.random() < self.FUG_OBS_ACC:
            return true_obs
        return self.rng.choice([OBSNORTH, OBSSOUTH, OBSLEVEL])

    def _get_reward(self, next_state: UAVState) -> Dict[str, float]:
        uav_coord, fug_coord = next_state
        uav_reward, fug_reward = self.R_ACTION, self.R_ACTION
        if fug_coord == self.grid.safe_house_coord:
            uav_reward = self.R_SAFE
            fug_reward = -self.R_SAFE
        elif fug_coord == uav_coord:
            uav_reward = self.R_CAPTURE
            fug_reward = -self.R_CAPTURE
        return {
            self.UAV_ID: uav_reward,
            self.FUG_ID: fug_reward,
        }


class UAVGrid(Grid):
    """A grid for the MA UAV Problem."""

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        block_coords: Optional[Set[Coord]],
        safe_house_coord: Optional[Coord] = None,
        init_fug_coords: Optional[List[Coord]] = None,
        init_uav_coords: Optional[List[Coord]] = None,
    ):
        super().__init__(grid_width, grid_height, block_coords)
        if safe_house_coord is None:
            safe_house_coord = (grid_width // 2, grid_height // 4)
        self.safe_house_coord = safe_house_coord

        if init_fug_coords is None:
            init_fug_coords = self.all_coords
            init_fug_coords.remove(self.safe_house_coord)
        self.init_fug_coords = init_fug_coords

        if init_uav_coords is None:
            init_uav_coords = self.all_coords
            init_uav_coords.remove(self.safe_house_coord)
        self.init_uav_coords = init_uav_coords

        self.valid_coords = set(self.unblocked_coords)
        self.valid_coords.remove(self.safe_house_coord)

    def get_ascii_repr(
        self, fug_coord: Optional[Coord], uav_coord: Optional[Coord]
    ) -> str:
        """Get ascii repr of grid."""
        grid_repr = []
        for row in range(self.height):
            row_repr = []
            for col in range(self.width):
                coord = (col, row)
                if coord == self.safe_house_coord:
                    row_repr.append("S")
                elif coord in self.block_coords:
                    row_repr.append("#")
                else:
                    row_repr.append(".")
            grid_repr.append(row_repr)

        if fug_coord is not None:
            grid_repr[fug_coord[0]][fug_coord[1]] = "F"
        if uav_coord is not None:
            if uav_coord == fug_coord:
                grid_repr[uav_coord[0]][uav_coord[1]] = "X"
            else:
                grid_repr[uav_coord[0]][uav_coord[1]] = "U"

        return str(self) + "\n" + "\n".join([" ".join(r) for r in grid_repr])
