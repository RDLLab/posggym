"""The core Level-Based Foraging Environment.

Copied and adapted from
https://github.com/semitable/lb-foraging/blob/master/lbforaging/foraging/environment.py

Mainly to make some small changes to adapt to the POSGGym API.
"""
import math
import random
import logging
from enum import IntEnum
from itertools import product
from collections import namedtuple, defaultdict

import gym
import numpy as np


def sorted_from_middle(lst):
    """Sorts list from middle out."""
    left = lst[len(lst)//2-1::-1]
    right = lst[len(lst)//2:]
    output = [right.pop(0)] if len(lst) % 2 else []
    for t in zip(left, right):
        output += sorted(t)
    return output


class LBFAction(IntEnum):
    """Level-Based Foraging Actions."""
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class CellEntity(IntEnum):
    """Entity encodings for grid observations."""
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    FOOD = 2
    AGENT = 3


class Player:
    """A player in the Level-Based Foraging environment."""

    def __init__(self, idx: int, position=None, level=None, reward=0):
        self.idx = idx
        self.position = position
        self.level = level
        self.reward = reward

    def setup(self, position, level):
        self.position = position
        self.level = level

    @property
    def name(self):
        return "Player"

    def copy(self):
        return Player(self.idx, self.position, self.level, self.reward)

    def __eq__(self, o):
        return (
            self.idx == o.idx
            and self.position == o.position
            and self.level == o.level
        )

    def __hash__(self):
        return hash((self.idx, self.position, self.level))


class ForagingEnv(gym.Env):
    """A class that contains rules/actions for level-based foraging."""

    metadata = {"render.modes": ["human", "rgb_array"]}

    action_set = [
        LBFAction.NORTH,
        LBFAction.SOUTH,
        LBFAction.WEST,
        LBFAction.EAST,
        LBFAction.LOAD
    ]
    Observation = namedtuple(
        "Observation",
        ["field", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation",
        ["position", "level", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        num_players,
        max_player_level,
        field_size,
        max_food,
        sight,
        max_episode_steps,
        force_coop,
        static_layout,
        normalize_reward=True,
        grid_observation=False,
        penalty=0.0,
    ):
        self.logger = logging.getLogger(__name__)
        self._rng = random.Random()
        self.n_agents = num_players
        self.players = [Player(i) for i in range(num_players)]

        self.field = np.zeros(field_size, np.int32)

        self.penalty = penalty

        self.max_food = max_food
        self._food_spawned = 0.0
        self.max_player_level = max_player_level
        self.sight = sight
        self.force_coop = force_coop
        self._game_over = None

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self.normalize_reward = normalize_reward
        self._grid_observation = grid_observation

        self.action_space = gym.spaces.Tuple(
            tuple([gym.spaces.Discrete(6)] * self.n_agents)
        )
        self.observation_space = gym.spaces.Tuple(
            tuple([self._get_observation_space()] * self.n_agents)
        )

        self.viewer = None

        self._static_layout = static_layout
        if static_layout:
            assert self.rows == self.cols, \
                "static layout only supported for square grids"
            self._food_locations = self._generate_food_locations(max_food)
            self._player_locations = self._generate_player_locations(
                num_players
            )
        else:
            self._food_locations = None
            self._player_locations = None

    def _generate_food_locations(self, max_food):
        """Generate food locations for static layout.

        Number and location of food is the same for given pairing of
        (field size, max_food).
        """
        assert self.rows == self.cols
        # must be enough space to surround each food with free cells
        assert max_food <= math.floor((self.rows-1) / 2)**2

        food_grid_size = math.floor((self.rows-1) / 2)
        available_food_locs = sorted_from_middle(
            list(product(range(food_grid_size), repeat=2))
        )

        food_locs = []
        for f in range(max_food):
            row = available_food_locs[f][0]*2 + 1
            col = available_food_locs[f][1]*2 + 1
            food_locs.append((row, col))
        return food_locs

    def _generate_player_locations(self, num_players):
        """Generate player start locations for static layout.

        Players always start around edge of field.
        """
        assert self.rows == self.cols
        assert num_players <= math.ceil(self.rows/2)*4 - 4

        idxs = [
            i*2 if i < self.rows/4 else i*2+((self.rows+1) % 2)
            for i in range(math.ceil(self.rows/2))
        ]
        idxs_reverse = list(reversed(idxs))

        # set it so locations are chosen from corners first then
        # alternating sides
        sides = [
            product([0], idxs[:-1]),
            product([self.rows-1], idxs_reverse[:-1]),
            product(idxs_reverse[:-1], [0]),
            product(idxs[:-1], [self.rows-1])
        ]
        available_locations = []
        for locs in zip(*sides):
            available_locations.extend(locs)

        return available_locations[:num_players]

    def seed(self, seed=None):
        self._rng = random.Random(seed)

    def _get_observation_space(self):
        """The Observation Space for each agent.

        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count

        """
        if not self._grid_observation:
            field_x = self.field.shape[1]
            field_y = self.field.shape[0]
            # field_size = field_x * field_y
            n_players = len(self.players)

            max_food = self.max_food
            max_food_level = self.max_player_level * n_players

            min_obs = [-1, -1, 0] * max_food + [-1, -1, 0] * n_players
            max_obs = (
                [field_x-1, field_y-1, max_food_level] * max_food
                + [field_x-1, field_y-1, self.max_player_level] * n_players
            )
        else:
            # grid observation space
            grid_shape = (1 + 2 * self.sight, 1 + 2 * self.sight)

            # agents layer: agent levels
            agents_min = np.zeros(grid_shape, dtype=np.float32)
            agents_max = np.ones(
                grid_shape, dtype=np.float32
            ) * self.max_player_level

            # foods layer: foods level
            max_food_level = self.max_player_level * len(self.players)
            foods_min = np.zeros(grid_shape, dtype=np.float32)
            foods_max = np.ones(grid_shape, dtype=np.float32) * max_food_level

            # access layer: i the cell available
            access_min = np.zeros(grid_shape, dtype=np.float32)
            access_max = np.ones(grid_shape, dtype=np.float32)

            # total layer
            min_obs = np.stack([agents_min, foods_min, access_min])
            max_obs = np.stack([agents_max, foods_max, access_max])

        return gym.spaces.Box(
            np.array(min_obs), np.array(max_obs), dtype=np.float32
        )

    @classmethod
    def from_obs(cls, obs):
        players = []
        for i, p in enumerate(obs.players):
            player = Player(i, p.position, p.level)
            players.append(player)

        env = cls(players, None, None, None, None)
        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        return env

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in LBFAction
                if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0): min(row + distance + 1, self.rows),
                max(col - distance, 0): min(col + distance + 1, self.cols),
            ]

        return (
            self.field[
                max(row - distance, 0): min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0): min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_food(self, row, col):
        return (
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
        )

    def adjacent_food_location(self, row, col):
        if row > 1 and self.field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        elif col > 1 and self.field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.players
            if (
                abs(player.position[0] - row) == 1
                and player.position[1] == col
                or abs(player.position[1] - col) == 1
                and player.position[0] == row
            )
        ]

    def spawn_food(self, max_food, max_level):
        if self._static_layout:
            self._spawn_food_static(max_level)
        else:
            self._spawn_food_generative(max_food, max_level)

    def _spawn_food_static(self, max_level):
        """Spawn food in static layout.

        Number and location of food is the same for given pairing of
        (field size, max_food), only the levels of each food will change.
        """
        assert max_level >= 1
        min_level = max_level if self.force_coop else 1

        for (row, col) in self._food_locations:
            self.field[row, col] = (
                min_level
                if min_level == max_level
                else self._rng.randint(min_level, max_level-1)
            )
        self._food_spawned = self.field.sum()

    def _spawn_food_generative(self, max_food, max_level):
        assert max_level >= 1
        food_count = 0
        attempts = 0
        min_level = max_level if self.force_coop else 1

        while food_count < max_food and attempts < 1000:
            attempts += 1
            row = self._rng.randint(1, self.rows - 2)
            col = self._rng.randint(1, self.cols - 2)

            # check if it has neighbors:
            if (
                self.neighborhood(row, col).sum() > 0
                or self.neighborhood(
                    row, col, distance=2, ignore_diag=True
                ) > 0
                or not self._is_empty_location(row, col)
            ):
                continue

            self.field[row, col] = (
                min_level
                if min_level == max_level
                else self._rng.randint(min_level, max_level-1)
            )
            food_count += 1
        self._food_spawned = self.field.sum()

    def _is_empty_location(self, row, col):
        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False
        return True

    def spawn_players(self, field, max_player_level):
        if self._static_layout:
            return self._spawn_players_static(field, max_player_level)
        return self._spawn_players_generative(field, max_player_level)

    def _spawn_players_static(self, field, max_player_level):
        players = []
        for i in range(self.n_agents):
            (row, col) = self._player_locations[i]
            player = Player(
                i, (row, col), self._rng.randint(1, max_player_level-1)
            )
            players.append(player)
        return players

    def _spawn_players_generative(self, field, max_player_level):
        players = []
        player_positions = set()

        for i in range(self.n_agents):
            player = Player(i, reward=0)

            attempts = 0
            while attempts < 1000:
                row = self._rng.randint(0, self.rows-1)
                col = self._rng.randint(0, self.cols-1)

                if field[row, col] == 0 and (row, col) not in player_positions:
                    player.position = (row, col)
                    player.level = self._rng.randint(1, max_player_level-1)
                    player_positions.add((row, col))
                    players.append(player)
                    break
                attempts += 1

        return players

    def _is_valid_action(self, player, action):
        if action == LBFAction.NONE:
            return True
        elif action == LBFAction.NORTH:
            return (
                player.position[0] > 0
                and self.field[player.position[0] - 1, player.position[1]] == 0
            )
        elif action == LBFAction.SOUTH:
            return (
                player.position[0] < self.rows - 1
                and self.field[player.position[0] + 1, player.position[1]] == 0
            )
        elif action == LBFAction.WEST:
            return (
                player.position[1] > 0
                and self.field[player.position[0], player.position[1] - 1] == 0
            )
        elif action == LBFAction.EAST:
            return (
                player.position[1] < self.cols - 1
                and self.field[player.position[0], player.position[1] + 1] == 0
            )
        elif action == LBFAction.LOAD:
            return self.adjacent_food(*player.position) > 0

        self.logger.error(f"Undefined action {action} from {player.name}")
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        self._gen_valid_moves()
        return list(
            product(*[self._valid_actions[player] for player in self.players])
        )

    def _make_obs(self, player):
        player_obs_pos = self._transform_to_neighborhood(
            player.position, self.sight, player.position
        )
        # Need to correct for player obs position being off-center due to
        # being too close to 0th row or column
        player_obs = []
        for a in self.players:
            a_pos = self._transform_to_neighborhood(
                player.position, self.sight, a.position
            )
            if (
                abs(a_pos[0] - player_obs_pos[0]) <= self.sight
                and abs(a_pos[1] - player_obs_pos[1]) <= self.sight
            ):
                player_obs.append(self.PlayerObservation(
                    position=a_pos,
                    level=a.level,
                    is_self=a == player,
                    reward=a.reward if a == player else None,
                ))
            else:
                player_obs.append(None)

        return self.Observation(
            players=player_obs,
            # todo also check max?
            field=self.neighborhood(*player.position, self.sight),
            game_over=self._game_over,
            sight=self.sight,
            current_step=self.current_step,
        )

    def _make_obs_array(self, observation):
        obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
        # obs[: observation.field.size] = observation.field.flatten()

        for i in range(self.max_food):
            f_idx = 3 * i
            obs[f_idx] = -1
            obs[f_idx + 1] = -1
            obs[f_idx + 2] = 0

        for i, (y, x) in enumerate(zip(*np.nonzero(observation.field))):
            f_idx = 3 * i
            obs[f_idx] = y
            obs[f_idx + 1] = x
            obs[f_idx + 2] = observation.field[y, x]

        p_seen_num = 1
        p_unseen_num = len(self.players)-1
        p_start_idx = self.max_food * 3
        for p in observation.players:
            if p is None:
                p_idx = p_start_idx + 3 * p_unseen_num
                p_unseen_num -= 1
                pos, level = (-1, -1), 0
            elif p.is_self:
                # self player is always first
                p_idx = p_start_idx
                pos, level = p.position, p.level
            else:
                p_idx = p_start_idx + 3 * p_seen_num
                p_seen_num += 1
                pos, level = p.position, p.level
            obs[p_idx] = pos[0]
            obs[p_idx + 1] = pos[1]
            obs[p_idx + 2] = level

        return obs

    def _make_global_grid_arrays(self):
        """Create global arrays for grid observation space."""
        grid_shape_x, grid_shape_y = self.field_size
        grid_shape_x += 2 * self.sight
        grid_shape_y += 2 * self.sight
        grid_shape = (grid_shape_x, grid_shape_y)

        agents_layer = np.zeros(grid_shape, dtype=np.float32)
        for player in self.players:
            x, y = player.position
            agents_layer[x + self.sight, y + self.sight] = player.level

        foods_layer = np.zeros(grid_shape, dtype=np.float32)
        foods_layer[
            self.sight:-self.sight, self.sight:-self.sight
        ] = self.field.copy()

        access_layer = np.ones(grid_shape, dtype=np.float32)
        # out of bounds not accessible
        access_layer[:self.sight, :] = 0.0
        access_layer[-self.sight:, :] = 0.0
        access_layer[:, :self.sight] = 0.0
        access_layer[:, -self.sight:] = 0.0
        # agent locations are not accessible
        for player in self.players:
            x, y = player.position
            access_layer[x + self.sight, y + self.sight] = 0.0
        # food locations are not accessible
        foods_x, foods_y = self.field.nonzero()
        for x, y in zip(foods_x, foods_y):
            access_layer[x + self.sight, y + self.sight] = 0.0

        return np.stack([agents_layer, foods_layer, access_layer])

    def _get_agent_grid_bounds(self, agent_x, agent_y):
        return (
            agent_x,
            agent_x + 2 * self.sight + 1,
            agent_y,
            agent_y + 2 * self.sight + 1
        )

    def _make_gym_obs(self):
        observations = [self._make_obs(player) for player in self.players]
        if self._grid_observation:
            layers = self._make_global_grid_arrays()
            agents_bounds = [
                self._get_agent_grid_bounds(*player.position)
                for player in self.players
            ]
            nobs = tuple(
                layers[:, start_x:end_x, start_y:end_y]
                for start_x, end_x, start_y, end_y in agents_bounds
            )
        else:
            nobs = tuple(self._make_obs_array(obs) for obs in observations)

        nreward = tuple(
            obs.players[i].reward for i, obs in enumerate(observations)

        )
        ndone = tuple(obs.game_over for obs in observations)
        ninfo = {}

        return nobs, nreward, ndone, ninfo

    def reset(self):
        self.field = np.zeros(self.field_size, np.int32)
        self.players = self.spawn_players(self.field, self.max_player_level)
        player_levels = sorted([player.level for player in self.players])

        self.spawn_food(
            self.max_food, max_level=sum(player_levels[:3])
        )
        self.current_step = 0
        self._game_over = False
        # self._gen_valid_moves()

        nobs, _, _, _ = self._make_gym_obs()
        return nobs

    def step(self, actions):
        self.current_step += 1

        for p in self.players:
            p.reward = 0

        # check if actions are valid
        actions = list(actions)
        for i, (player, action) in enumerate(zip(self.players, actions)):
            try:
                if not self._is_valid_action(player, action):
                    actions[i] = LBFAction.NONE
            except ValueError:
                actions[i] = LBFAction.NONE

        loading_players = set()

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(self.players, actions):
            if action == LBFAction.NONE:
                collisions[player.position].append(player)
            elif action == LBFAction.NORTH:
                collisions[
                    (player.position[0] - 1, player.position[1])
                ].append(player)
            elif action == LBFAction.SOUTH:
                collisions[
                    (player.position[0] + 1, player.position[1])
                ].append(player)
            elif action == LBFAction.WEST:
                collisions[
                    (player.position[0], player.position[1] - 1)
                ].append(player)
            elif action == LBFAction.EAST:
                collisions[
                    (player.position[0], player.position[1] + 1)
                ].append(player)
            elif action == LBFAction.LOAD:
                collisions[player.position].append(player)
                loading_players.add(player)

        # and do movements for non colliding players

        for k, v in collisions.items():
            # make sure no more than an player will arrive at location
            if len(v) > 1:
                continue
            v[0].position = k

        # finally process the loadings:
        while loading_players:
            # find adjacent food
            player = loading_players.pop()
            frow, fcol = self.adjacent_food_location(*player.position)
            food = self.field[frow, fcol]

            adj_players = self.adjacent_players(frow, fcol)
            adj_players = [
                p for p in adj_players if p in loading_players or p is player
            ]

            adj_player_level = sum([a.level for a in adj_players])

            loading_players = loading_players - set(adj_players)

            if adj_player_level < food:
                # failed to load
                for a in adj_players:
                    a.reward -= self.penalty
                continue

            # else the food was loaded and each player scores points
            for a in adj_players:
                a.reward = float(a.level * food)
                if self.normalize_reward:
                    a.reward = a.reward / float(
                        adj_player_level * self._food_spawned
                    )  # normalize reward
            # and the food is removed
            self.field[frow, fcol] = 0

        self._game_over = (
            bool(self.field.sum() == 0)
            or self._max_episode_steps <= self.current_step
        )

        return self._make_gym_obs()

    def _init_render(self):
        from lbforaging.foraging.rendering import Viewer
        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if mode not in self.metadata["render.modes"]:
            super().render(mode)

        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=(mode == "rgb_array"))

    def close(self):
        if self.viewer:
            self.viewer.close()
