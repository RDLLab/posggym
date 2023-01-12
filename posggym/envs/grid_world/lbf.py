"""The Level-Based Foraging Environment.

This implementation uses and is based on the original implementation of the
Level-Based Foraging environment:

https://github.com/semitable/lb-foraging

References
----------
- Stefano V. Albrecht and Subramanian Ramamoorthy. 2013. A Game-Theoretic Model and
  Best-Response Learning Method for Ad Hoc Coordination in Multia-gent Systems.
  In Proceedings of the 2013 International Conference on Autonomous Agents and
  Multi-Agent Systems. 1155–1156.
- S. V. Albrecht and Peter Stone. 2017. Reasoning about Hypothetical Agent
  Behaviours and Their Parameters. In 16th International Conference on Autonomous
  Agents and Multiagent Systems 2017. International Foundation for Autonomous
  Agents and Multiagent Systems, 547–555
- Filippos Christianos, Lukas Schäfer, and Stefano Albrecht. 2020. Shared Experience
  Actor-Critic for Multi-Agent Reinforcement Learning. Advances in Neural
  Information Processing Systems 33 (2020), 10707–10717
- Georgios Papoudakis, Filippos Christianos, Lukas Schäfer, and Stefano V. Albrecht.
  2021. Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in
  Cooperative Tasks. In Thirty-Fifth Conference on Neural Information Processing
  Systems Datasets and Benchmarks Track (Round 1)

"""
import enum
import math
from collections import defaultdict
from itertools import product
from os import path
from typing import Dict, List, NamedTuple, Optional, SupportsFloat, Tuple, Union

import numpy as np
from gymnasium import spaces

import posggym.model as M
from posggym import logger
from posggym.core import DefaultEnv
from posggym.envs.grid_world.core import Coord, Direction, Grid
from posggym.error import DependencyNotInstalled
from posggym.utils import seeding


class Player(NamedTuple):
    """A player in the Level-Based Foraging environment."""

    idx: int
    coord: Coord
    level: int


class Food(NamedTuple):
    """A food in the Level-Based Foraging environment."""

    coord: Coord
    level: int


class LBFState(NamedTuple):
    """State in Level-Based Foraging environment."""

    players: Tuple[Player, ...]
    food: Tuple[Food, ...]
    food_spawned: int


class LBFAction(enum.IntEnum):
    """Level-Based Foraging Actions."""

    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


ACTION_TO_DIR = {
    LBFAction.NORTH: Direction.NORTH,
    LBFAction.SOUTH: Direction.SOUTH,
    LBFAction.WEST: Direction.WEST,
    LBFAction.EAST: Direction.EAST,
}


class CellEntity(enum.IntEnum):
    """Entity encodings for grid observations."""

    OUT_OF_BOUNDS = 0
    EMPTY = 1
    FOOD = 2
    AGENT = 3


LBFObs = Union[Tuple[int, ...], np.ndarray]


class LBFEntityObs(NamedTuple):
    """Entity (player, food) observation in the LBF environment."""

    coord: Coord
    level: int
    is_self: bool


class LBFEnv(DefaultEnv[LBFState, LBFObs, LBFAction]):
    """The Level-Based Foraging Environment.

    This implementation uses and is based on the original implementation of
    the Level-Based Foraging environment:

    https://github.com/semitable/lb-foraging

    We provide a discription here for convinience.

    Agents
    ------
    Varied number

    State
    -----
    The state of the environment is defined by a (x, y, level) triplet for each
    agent and food object in the environment. The (x, y) components define the
    position, starting from the bottom left square, while the level is the
    level of the agent or food.

    Actions
    -------
    Each agent has six possible discrete actions [0-5] corresponding to:

        Noop, Move North, Move South, Move West, Move East, Pickup

    Observations
    ------------
    Each agent observes the (x, y, level) of food and other agent within their
    field of vision, which is the grid `sight` distance away from the agent in
    all directions.

    There are three observation modes:

    1. grid_observation
       The agent recieves three 2D layers of size (1+2*`sight`, 1+2*`sight`).
       Each cell in each layer corresponds to a specific (x, y) coordinate
       relative to the observing agent.
       The layers are:
         i. agent level
         ii. food level
         iii. whether cell is free or blocked (e.g. out of bounds)
    2. vector observation
       A vector of (x, y, level) triplets for each food and agent in the
       environment. If a given food or agent is not within the observing
       agent's field of vision triplets have a value of [-1, -1, 0].
       The size of the vector is (`num_agents` + `max_food`) * 3, with the
       first `max_food` triplets being for the food, the `max_food`+1 triplet
       being for the observing agent and the remaining `num_agents`-1
       triplets for the other agents.
       The ordering of the triplets for the other agents is consistent, while
       the food obs triplets can change based on how many are visible and their
       relative coordinates to the observing agent.
    3. tuple observation
       This is the same as the vector observation except observations are
       Python tuples of integers instead of numpy arrays of floats.

    References
    ----------
    - Stefano V. Albrecht and Subramanian Ramamoorthy. 2013. A Game-Theoretic Model and
      Best-Response Learning Method for Ad Hoc Coordination in Multia-gent Systems.
      In Proceedings of the 2013 International Conference on Autonomous Agents and
      Multi-Agent Systems. 1155–1156.
    - S. V. Albrecht and Peter Stone. 2017. Reasoning about Hypothetical Agent
      Behaviours and Their Parameters. In 16th International Conference on Autonomous
      Agents and Multiagent Systems 2017. International Foundation for Autonomous
      Agents and Multiagent Systems, 547–555
    - Filippos Christianos, Lukas Schäfer, and Stefano Albrecht. 2020. Shared Experience
      Actor-Critic for Multi-Agent Reinforcement Learning. Advances in Neural
      Information Processing Systems 33 (2020), 10707–10717
    - Georgios Papoudakis, Filippos Christianos, Lukas Schäfer, and Stefano V. Albrecht.
      2021. Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in
      Cooperative Tasks. In Thirty-Fifth Conference on Neural Information Processing
      Systems Datasets and Benchmarks Track (Round 1)

    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        num_agents: int,
        max_agent_level: int,
        field_size: Tuple[int, int],
        max_food: int,
        sight: int,
        force_coop: bool,
        static_layout: bool,
        normalize_reward: bool = True,
        observation_mode: str = "tuple",
        penalty: float = 0.0,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            LBFModel(
                num_agents,
                max_agent_level,
                field_size=field_size,
                max_food=max_food,
                sight=sight,
                force_coop=force_coop,
                static_layout=static_layout,
                normalize_reward=normalize_reward,
                observation_mode=observation_mode,
                penalty=penalty,
                **kwargs,
            ),
            render_mode=render_mode,
        )

        grid: Grid = self.model.grid  # type: ignore
        self.window_size = (min(64 * grid.width, 512), min(64 * grid.height, 512))
        self.cell_size = (
            self.window_size[0] // grid.width,
            self.window_size[1] // grid.height,
        )
        self.window_surface = None
        self.clock = None
        self.food_img = None
        self.agent_img = None
        self.font = None

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. posggym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install posggym[grid-world]`"
            ) from e

        if self.window_surface is None:
            pygame.init()

            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Level-Based Foraging")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif self.render_mode == "rgb_array":
                pygame.font.init()
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.agent_img is None:
            file_name = path.join(path.dirname(__file__), "img/robot.png")
            self.agent_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.food_img is None:
            file_name = path.join(path.dirname(__file__), "img/apple.png")
            self.food_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.font is None:
            self.font = pygame.font.SysFont("Comic Sans MS", self.cell_size[1] // 4)

        self.window_surface.fill((0, 0, 0))
        grid: Grid = self.model.grid  # type: ignore

        # draw grid lines
        for y in range(grid.height):
            for x in range(grid.width):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)
                pygame.draw.rect(self.window_surface, (255, 255, 255), rect, 1)

        for player in self._state.players:
            (x, y) = player.coord
            pos = (x * self.cell_size[0], y * self.cell_size[1])
            self.window_surface.blit(self.agent_img, pos)
            level_txt = self.font.render(
                str(player.level), True, (255, 255, 255), (0, 0, 0)
            )
            text_pos = (pos[0] + 1, pos[1] + 1)
            self.window_surface.blit(level_txt, text_pos)

        for food in self._state.food:
            (x, y) = food.coord
            pos = (x * self.cell_size[0], y * self.cell_size[1])
            self.window_surface.blit(self.food_img, pos)
            level_txt = self.font.render(str(food.level), True, (255, 255, 255))
            text_pos = (pos[0] + 1, pos[1] + 1)
            self.window_surface.blit(level_txt, text_pos)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


class LBFModel(M.POSGModel[LBFState, LBFObs, LBFAction]):
    """Level-Based Foraging model.

    Parameters
    ----------
    num_agents : int
        the number of agents in the environment
    max_agent_level : int
        maximum foraging level of any agent
    field_size : Tuple[int, int]
        width and height of the playing field
    max_food : int
        the max number of food that can be spawned
    sight : int
        observation size of each agent
    force_coop : bool
        whether to force cooperation or not
    normalize_reward : bool, optional
        whether to normalize the reward or not (default=True)
    observation_mode : str, optional
        The observation mode for agent (default='tuple')
          - 'grid' - observations are multiple 2D grids (3D np.ndarray)
          - 'vector' - observations are vector (1D np.ndarray)
          - 'tuple' - observations are a tuple with same format as 'vector'
                      observations but as a hashable Python tuple object
                      containing integers instead of floats

    penalty : float, optional
        the penalty for failing to load food (default=0.0)

    """

    OBSERVATION_MODES = ["grid", "vector", "tuple"]

    # pylint: disable=unused-argument
    def __init__(
        self,
        num_agents: int,
        max_agent_level: int,
        field_size: Tuple[int, int],
        max_food: int,
        sight: int,
        force_coop: bool,
        static_layout: bool,
        normalize_reward: bool = True,
        observation_mode: str = "tuple",
        penalty: float = 0.0,
        **kwargs,
    ):
        assert observation_mode in self.OBSERVATION_MODES
        self._field_size = field_size
        self._rows, self._cols = field_size
        self._max_agent_level = max_agent_level
        self._max_food = max_food
        self._sight = sight
        self._force_coop = force_coop
        self._static_layout = static_layout
        self._normalize_reward = normalize_reward
        self._obs_mode = observation_mode
        self._penalty = penalty

        self.possible_agents = tuple(range(num_agents))
        self.action_spaces = {
            i: spaces.Discrete(len(LBFAction)) for i in self.possible_agents
        }
        self.observation_spaces = {
            i: self.get_agent_observation_space() for i in self.possible_agents
        }
        self.observation_first = True
        self.is_symmetric = True

        self.grid = Grid(
            grid_width=self._cols, grid_height=self._rows, block_coords=None
        )

        self._food_locations: Optional[List[Coord]] = None
        self._player_locations: Optional[List[Coord]] = None
        if static_layout:
            assert (
                self._rows == self._cols
            ), "static layout only supported for square grids"
            self._food_locations = self._generate_static_food_coords()
            self._player_locations = self._generate_static_player_coords()

    def get_agent_observation_space(self) -> spaces.Space:
        """The Observation Space for an agent.

        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count

        """
        n_agents = len(self.possible_agents)
        max_food_level = self._max_agent_level * min(n_agents, 4)
        if self._obs_mode == "tuple":
            max_x, max_y = self._cols, self._rows
            tuple_obs_space = []
            for _ in range(n_agents):
                tuple_obs_space.append(spaces.Discrete(max_x + 1, start=-1))
                tuple_obs_space.append(spaces.Discrete(max_y + 1, start=-1))
                tuple_obs_space.append(
                    spaces.Discrete(self._max_agent_level + 1, start=0)
                )
            for _ in range(self._max_food):
                tuple_obs_space.append(spaces.Discrete(max_x + 2, start=-1))
                tuple_obs_space.append(spaces.Discrete(max_y + 2, start=-1))
                tuple_obs_space.append(spaces.Discrete(max_food_level + 1, start=0))
            return spaces.Tuple(tuple_obs_space)

        if self._obs_mode == "vector":
            max_x, max_y = self._cols - 1, self._rows - 1
            min_obs = [-1, -1, 0] * n_agents + [-1, -1, 0] * self._max_food
            max_obs = [max_x, max_y, self._max_agent_level] * n_agents + [
                max_x,
                max_y,
                max_food_level,
            ] * self._max_food
        else:
            # grid observation space
            grid_shape = (1 + 2 * self._sight, 1 + 2 * self._sight)

            # agents layer: agent levels
            agents_min = np.zeros(grid_shape, dtype=np.float32)
            agents_max = np.ones(grid_shape, dtype=np.float32) * self._max_agent_level

            # foods layer: foods level
            foods_min = np.zeros(grid_shape, dtype=np.float32)
            foods_max = np.ones(grid_shape, dtype=np.float32) * max_food_level

            # access layer: i the cell available
            access_min = np.zeros(grid_shape, dtype=np.float32)
            access_max = np.ones(grid_shape, dtype=np.float32)

            # total layer
            min_obs = np.stack([agents_min, foods_min, access_min])
            max_obs = np.stack([agents_max, foods_max, access_max])

        # dtype = np.int8 if self._obs_mode == "vector" else np.uint8
        dtype = np.float32
        return spaces.Box(
            np.array(min_obs, dtype=dtype),
            np.array(max_obs, dtype=dtype),
            dtype=dtype,
        )

    @property
    def reward_ranges(self) -> Dict[M.AgentID, Tuple[SupportsFloat, SupportsFloat]]:
        if self._normalize_reward:
            reward_range = (self._penalty, 1.0)
        else:
            reward_range = (self._penalty, self._max_agent_level)
        return {i: reward_range for i in self.possible_agents}

    def get_agents(self, state: LBFState) -> List[M.AgentID]:
        return list(self.possible_agents)

    @property
    def rng(self) -> seeding.RNG:
        if self._rng is None:
            self._rng, seed = seeding.std_random()
        return self._rng

    def seed(self, seed: Optional[int] = None):
        super().seed(seed)
        if self._static_layout:
            assert (
                self._rows == self._cols
            ), "static layout only supported for square grids"
            self._food_locations = self._generate_static_food_coords()
            self._player_locations = self._generate_static_player_coords()

    def _generate_static_food_coords(self) -> List[Coord]:
        """Generate food coords for static layout.

        Number and location of food is the same for given pairing of
        (field size, max_food).
        """
        assert self._rows == self._cols
        # must be enough space to surround each food with free cells
        assert self._max_food <= math.floor((self._rows - 1) / 2) ** 2

        food_grid_size = math.floor((self._rows - 1) / 2)
        available_food_locs = sorted_from_middle(
            list(product(range(food_grid_size), repeat=2))
        )

        food_locs = []
        for f in range(self._max_food):
            row = available_food_locs[f][0] * 2 + 1
            col = available_food_locs[f][1] * 2 + 1
            food_locs.append((row, col))
        return food_locs

    def _generate_static_player_coords(self) -> List[Coord]:
        """Generate player start coords for static layout.

        Players always start around edge of field.
        """
        assert self._rows == self._cols

        assert len(self.possible_agents) <= math.ceil(self._rows / 2) * 4 - 4

        idxs = [
            i * 2 if i < self._rows / 4 else i * 2 + ((self._rows + 1) % 2)
            for i in range(math.ceil(self._rows / 2))
        ]
        idxs_reverse = list(reversed(idxs))

        # set it so locations are chosen from corners first then
        # alternating sides
        sides = [
            product([0], idxs[:-1]),
            product([self._rows - 1], idxs_reverse[:-1]),
            product(idxs_reverse[:-1], [0]),
            product(idxs[:-1], [self._rows - 1]),
        ]
        available_locations: List[Coord] = []
        for locs in zip(*sides):
            available_locations.extend(locs)

        return available_locations[: len(self.possible_agents)]

    def sample_initial_state(self) -> LBFState:
        if self._static_layout:
            players = self._spawn_players_static()
        else:
            players = self._spawn_players_generative()

        player_levels = sorted([player.level for player in players])
        max_level = sum(player_levels[:3])

        if self._static_layout:
            food = self._spawn_food_static(max_level)
        else:
            food = self._spawn_food_generative(
                self._max_food, max_level, [p.coord for p in players]
            )
        return LBFState(players, food, sum(f.level for f in food))

    def _spawn_players_static(self) -> Tuple[Player, ...]:
        assert self._player_locations is not None
        players = []
        for i in range(len(self.possible_agents)):
            (x, y) = self._player_locations[i]
            level = self.rng.randint(1, self._max_agent_level - 1)  # type: ignore
            players.append(Player(i, (x, y), level))
        return tuple(players)

    def _spawn_players_generative(self) -> Tuple[Player, ...]:
        players = []
        available_coords = list(product(range(self._cols), range(self._rows)))
        for i in range(len(self.possible_agents)):
            level = self.rng.randint(1, self._max_agent_level - 1)  # type: ignore
            coord = self.rng.choice(available_coords)
            available_coords.remove(coord)
            players.append(Player(i, coord, level))
        return tuple(players)

    def _spawn_food_static(self, max_level: int) -> Tuple[Food, ...]:
        """Spawn food in static layout.

        Number and location of food is the same for given pairing of
        (field size, max_food), only the levels of each food will change.
        """
        assert self._food_locations is not None
        assert max_level >= 1
        min_level = max_level if self._force_coop else 1
        food = []
        for (x, y) in self._food_locations:
            level = min_level
            if min_level != max_level:
                level = self.rng.randint(min_level, max_level)  # type: ignore
            food.append(Food((x, y), level))
        return tuple(food)

    def _spawn_food_generative(
        self,
        max_food: int,
        max_level: int,
        player_coords: List[Coord],
    ) -> Tuple[Food, ...]:
        assert max_level >= 1
        attempts = 0
        min_level = max_level if self._force_coop else 1
        unavailable_coords = set(player_coords)
        food: List[Food] = []
        while len(food) < max_food and attempts < 1000:
            attempts += 1
            x = self.rng.randint(1, self._cols - 2)  # type: ignore
            y = self.rng.randint(1, self._rows - 2)  # type: ignore

            if (x, y) in unavailable_coords:
                continue
            # add coord and adjacent coord to unavailable set so food are always
            # atleast one cell apart
            for x_delta, y_delta in product([-1, 0, 1], repeat=2):
                unavailable_coords.add((x + x_delta, y + y_delta))
            level = min_level
            if min_level != max_level:
                level = self.rng.randint(min_level, max_level)  # type: ignore
            food.append(Food((x, y), level))
        return tuple(food)

    def sample_initial_obs(self, state: LBFState) -> Dict[M.AgentID, LBFObs]:
        return self._get_obs(state)

    def step(
        self, state: LBFState, actions: Dict[M.AgentID, LBFAction]
    ) -> M.JointTimestep[LBFState, LBFObs]:
        assert all(0 <= a < len(LBFAction) for a in actions.values())
        next_state, rewards = self._get_next_state(state, actions)
        obs = self._get_obs(next_state)
        all_done = len(next_state.food) == 0
        terminated = {i: all_done for i in self.possible_agents}
        truncated = {i: False for i in self.possible_agents}
        info: Dict[M.AgentID, Dict] = {i: {} for i in self.possible_agents}
        return M.JointTimestep(
            next_state, obs, rewards, terminated, truncated, all_done, info
        )

    def _get_next_state(
        self, state: LBFState, actions: Dict[M.AgentID, LBFAction]
    ) -> Tuple[LBFState, Dict[M.AgentID, SupportsFloat]]:
        next_food = {f.coord: f for f in state.food}

        # try move agents
        # if two or more agents try to move to the same location they all fail
        collisions = defaultdict(list)
        loading_players = set()
        for player in state.players:
            i = player.idx
            a_i = actions[i]
            if a_i == LBFAction.NONE:
                next_coord = player.coord
            elif a_i == LBFAction.LOAD:
                next_coord = player.coord
                loading_players.add(player)
            else:
                next_coord = self.grid.get_next_coord(player.coord, ACTION_TO_DIR[a_i])
            if next_coord in next_food:
                next_coord = player.coord
            collisions[next_coord].append(player)

        # do movements for non colliding players
        next_players = list(state.players)
        for coord, player_list in collisions.items():
            # make sure no more than an player will arrive at location
            if len(player_list) > 1:
                continue
            player = player_list[0]
            next_players[player.idx] = Player(player.idx, coord, player.level)

        # process the loadings and calculate rewards as necessary
        rewards: Dict[M.AgentID, float] = {i: 0.0 for i in self.possible_agents}
        for player in loading_players:
            adj_coords = self.grid.get_neighbours(player.coord)
            for adj_coord in adj_coords:
                if adj_coord not in next_food:
                    continue
                adj_food = next_food[adj_coord]
                food_adj_coords = self.grid.get_neighbours(adj_coord)
                adj_players = [p for p in loading_players if p.coord in food_adj_coords]
                adj_player_level = sum([p.level for p in adj_players])

                if adj_player_level < adj_food.level:
                    # failed to load
                    for p in adj_players:
                        rewards[p.idx] -= self._penalty
                else:
                    # food was loaded and each player scores points
                    for p in adj_players:
                        p_reward = float(p.level * adj_food.level)
                        if self._normalize_reward:
                            p_reward = p_reward / float(
                                adj_player_level * state.food_spawned
                            )
                        rewards[p.idx] += p_reward
                    # and food is removed
                    next_food.pop(adj_coord)
        next_state = LBFState(
            tuple(next_players), tuple(next_food.values()), state.food_spawned
        )
        return next_state, rewards  # type: ignore

    def _get_obs(self, state: LBFState) -> Dict[M.AgentID, LBFObs]:
        obs: Dict[M.AgentID, LBFObs] = {}
        for i in self.possible_agents:
            player_obs, food_obs = self._get_local_obs(state, int(i))
            if self._obs_mode == "tuple":
                obs[i] = self._get_tuple_obs(player_obs, food_obs)
            elif self._obs_mode == "vector":
                obs[i] = self._get_vector_obs(int(i), player_obs, food_obs)
            else:
                obs[i] = self._get_grid_obs(
                    int(i), state.players[int(i)].coord, player_obs, food_obs
                )
        return obs

    def _get_local_obs(
        self, state: LBFState, agent_id: int
    ) -> Tuple[List[LBFEntityObs], List[LBFEntityObs]]:
        # player is always in center of observable area
        player_obs = []
        ego_player = state.players[agent_id]
        for p in state.players:
            if (
                abs(p.coord[0] - ego_player.coord[0]) <= self._sight
                and abs(p.coord[1] - ego_player.coord[1]) <= self._sight
            ):
                p_obs_coord = (
                    p.coord[0] - ego_player.coord[0] + self._sight,
                    p.coord[1] - ego_player.coord[1] + self._sight,
                )
                player_obs.append(
                    LBFEntityObs(p_obs_coord, p.level, is_self=(ego_player == p))
                )
            else:
                # player p not observed
                player_obs.append(LBFEntityObs((-1, -1), 0, is_self=False))

        food_obs = []
        for f in state.food:
            # note unobserved food are exluded since ordering can change for food obs
            # (unlike for players)
            if (
                abs(f.coord[0] - ego_player.coord[0]) <= self._sight
                and abs(f.coord[1] - ego_player.coord[1]) <= self._sight
            ):
                f_obs_coord = (
                    f.coord[0] - ego_player.coord[0] + self._sight,
                    f.coord[1] - ego_player.coord[1] + self._sight,
                )
                food_obs.append(LBFEntityObs(f_obs_coord, f.level, is_self=False))

        return player_obs, food_obs

    def _get_tuple_obs(
        self, player_obs: List[LBFEntityObs], food_obs: List[LBFEntityObs]
    ) -> LBFObs:
        obs: List[int] = []
        for o in player_obs:
            if o.is_self:
                obs.insert(0, o.level)
                obs.insert(0, o.coord[1])
                obs.insert(0, o.coord[0])
            else:
                obs.extend((*o.coord, o.level))
        for f_idx in range(self._max_food):
            if f_idx < len(food_obs):
                obs.extend((*food_obs[f_idx].coord, food_obs[f_idx].level))
            else:
                obs.extend((-1, -1, 0))
        return tuple(obs)

    def _get_vector_obs(
        self,
        agent_id: int,
        player_obs: List[LBFEntityObs],
        food_obs: List[LBFEntityObs],
    ) -> np.ndarray:
        # initialize obs array to (-1, -1, 0)
        obs = np.full(
            ((len(self.possible_agents) + self._max_food) * 3,), -1.0, dtype=np.float32
        )
        obs[2::3] = 0
        for i, o in enumerate(player_obs):
            if o.level == 0:
                # player unobserved
                continue
            if o.is_self:
                p_idx = 0
            else:
                p_idx = i * 3 if i > agent_id else (i + 1) * 3
            obs[p_idx : p_idx + 3] = (*o.coord, o.level)

        food_start_idx = len(self.possible_agents) * 3
        for i in range(self._max_food):
            if i < len(food_obs):
                f_idx = food_start_idx + (i * 3)
                obs[f_idx : f_idx + 3] = (*food_obs[i].coord, food_obs[i].level)

        return obs

    def _get_grid_obs(
        self,
        agent_id: int,
        agent_coord: Coord,
        player_obs: List[LBFEntityObs],
        food_obs: List[LBFEntityObs],
    ) -> np.ndarray:
        grid_shape_x, grid_shape_y = (2 * self._sight + 1, 2 * self._sight + 1)
        # agent, food, access layers
        grid_obs = np.zeros((3, grid_shape_x, grid_shape_y), dtype=np.float32)

        # add all in-bounds areas as accessible
        # cells containing players and food will be made inaccessible in next steps
        agent_x, agent_y = agent_coord
        for x, y in product(
            range(
                max(agent_x - self._sight, 0), min(self._cols, agent_x + self._sight)
            ),
            range(
                max(agent_y - self._sight, 0), min(self._rows, agent_y + self._sight)
            ),
        ):
            obs_x = x - agent_x + self._sight
            obs_y = y - agent_y + self._sight
            grid_obs[2, obs_x, obs_y] = 1.0

        for p in player_obs:
            if p.level > 0:
                grid_obs[0][p.coord] = p.level
                # agent coords are not accessible
                grid_obs[2][p.coord] = 0.0

        for f in food_obs:
            grid_obs[1][f.coord] = f.level
            # food coords are not accessible
            grid_obs[2][f.coord] = 0.0

        return grid_obs

    def parse_obs(
        self, obs: LBFObs
    ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
        """Parse observation into (x, y, level) agent and food triplets.

        Agent obs are ordered so the observing agent is first, then the
        remaining observations are by agent order.

        On triplet of [-1, -1, 0] means no observation for the given agent or
        food.
        """
        if self._obs_mode == "tuple":
            assert isinstance(obs, tuple)
            return self.parse_tuple_obs(obs)

        assert isinstance(obs, np.ndarray)
        if self._obs_mode == "grid":
            return self.parse_grid_obs(obs)
        # vector obs
        return self.parse_vector_obs(obs)

    def parse_grid_obs(
        self, obs: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
        """Parse grid observation int (x, y, level) agent and food triplets.

        Agent obs are ordered so the observing agent is first, then the
        remaining observations are by agent order.

        On triplet of [-1, -1, 0] means no observation for the given agent or
        food.
        """
        raise NotImplementedError

    def parse_vector_obs(
        self, obs: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
        """Parse vector obs into (x, y, level) agent and food triplets.

        Agent obs are ordered so the observing agent is first, then the
        remaining observations are by agent order.

        On triplet of [-1, -1, 0] means no observation for the given agent or
        food.
        """
        assert obs.shape[0] == 3 * (len(self.possible_agents) + self._max_food)
        agent_obs = []
        food_obs = []
        for i in range(0, obs.shape[0], 3):
            triplet = (int(obs[i]), int(obs[i + 1]), int(obs[i + 2]))
            if i < len(self.possible_agents) * 3:
                agent_obs.append(triplet)
            else:
                food_obs.append(triplet)
        return agent_obs, food_obs

    def parse_tuple_obs(
        self, obs: Tuple[int, ...]
    ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
        """Parse tuple obs into (x, y, level) agent and food triplets.

        Agent obs are ordered so the observing agent is first, then the
        remaining observations are by agent order.

        On triplet of [-1, -1, 0] means no observation for the given agent or
        food.
        """
        assert len(obs) == 3 * (len(self.possible_agents) + self._max_food)
        agent_obs = []
        food_obs = []
        for i in range(0, len(obs), 3):
            triplet = (int(obs[i]), int(obs[i + 1]), int(obs[i + 2]))
            if i < len(self.possible_agents) * 3:
                agent_obs.append(triplet)
            else:
                food_obs.append(triplet)
        return agent_obs, food_obs


def sorted_from_middle(lst):
    """Sorts list from middle out."""
    left = lst[len(lst) // 2 - 1 :: -1]
    right = lst[len(lst) // 2 :]
    output = [right.pop(0)] if len(lst) % 2 else []
    for t in zip(left, right):
        output += sorted(t)
    return output
