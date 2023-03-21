"""The Continuous Predator-Prey Environment.

A co-operative 2D continuous world problem involving multiple predator agents working
together to catch prey agents in the environment.

This intends to be an adaptation of the 2D grid-world to the continuous setting.

Reference
---------
- Ming Tan. 1993. Multi-Agent Reinforcement Learning: Independent vs. Cooperative Agents
  In Proceedings of the Tenth International Conference on Machine Learning. 330–337.
- J. Z. Leibo, V. F. Zambaldi, M. Lanctot, J. Marecki, and T. Graepel. 2017. Multi-Agent
  Reinforcement Learning in Sequential Social Dilemmas. In AAMAS, Vol. 16. ACM, 464–473
- Lowe, Ryan, Yi I. Wu, Aviv Tamar, Jean Harb, OpenAI Pieter Abbeel, and Igor Mordatch.
  2017. “Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.”
  Advances in Neural Information Processing Systems 30.

"""
import math
from itertools import product
from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    Set,
    Callable,
    Iterable,
    cast,
)

from gymnasium import spaces
import numpy as np

import posggym.model as M
from posggym.core import DefaultEnv
from posggym.envs.continuous.core import (
    RectangularContinuousWorld,
    Object,
    Position,
    clip_actions,
)
from posggym.utils import seeding


EMPTY = 0
WALL = 1
PREDATOR = 2
PREY = 3


class PPState(NamedTuple):
    """A state in the Continuous Predator-Prey Environment."""

    predator_coords: np.ndarray
    prey_coords: np.ndarray
    prey_caught: Tuple[int, ...]


# Actions
PPAction = List[float]
PPObs = Tuple[Union[int, np.ndarray], ...]
collision_distance = 1.2

AGENT_TYPE = [PREDATOR, PREY]


class PredatorPreyContinuous(DefaultEnv[PPState, PPObs, PPAction]):
    """The Continuous Predator-Prey Environment.

    A co-operative 2D continuous world problem involving multiple predator agents
    working together to catch prey agent/s in the environment.

    Possible Agents
    ---------------
    Varied number

    State Space
    -----------
    Each state consists of:

    1. tuple of the (x, y) position of all predators
    2. tuple of the (x, y) position of all preys
    3. tuple of whether each prey has been caught or not (0=no, 1=yes)

    For the coordinate x=column, y=row, with the origin (0, 0) at the top-left square
    of the world.

    Action Space
    ------------
    Each agent has 2 actions. In the `holonomic` model there are two actions, which
    are the change in x and change in y position. In the non-holonomic model, there
    are also two actions, which are the angular and linear velocity.

    Observation Space
    -----------------
    Each agent observes the contents of local cells. This is achieved by
    a series of 'n_lines' lines starting at the agent which extend for a distance of
    'obs_dim'. For each line the agent observes whether they intersect with one of four
    objects: `EMPTY=0`, `WALL=1`, `PREDATOR=2`, `PREY=3`. They also observe the distance
    to the object. If the object is `EMPTY=0`, the distance will always be equal to
    'obs_dim'

    Rewards
    -------
    There are two modes of play:

    1. *Fully cooperative*: All predators share a reward and each agent receives
    a reward of `1.0 / num_prey` for each prey capture, independent of which
    predator agent/s were responsible for the capture.

    2. *Mixed cooperative*: Predators only receive a reward if they were part
    of the prey capture, receiving `1.0 / num_prey` per capture.

    In both modes prey can only been captured when at least `prey_strength`
    predators are in adjacent cells, where `1 <= prey_strength <= num_predators`.

    Dynamics
    --------
    Actions of the predator agents are deterministic and consist of moving based on
    the dynamic model. If two or more predators attempt to move into the same location
    then no agent moves.

    Prey move according to the following rules (in order of priority):

    1. if predator is within `obs_dim` distance, moves away from closest predator
    2. if another prey is within `obs_dim` distance, moves away from closest prey
    3. else move randomly

    Prey always move first and predators and prey cannot occupy the same location.
    The only exception being if a prey has been caught their final coord is
    recorded in the state but predator and prey agents will be able to move
    into the final coord.

    Starting State
    --------------
    Predators start from random separate locations along the edge of the world
    (either in a corner, or half-way along a side), while prey start together
    in the middle.

    Episodes End
    ------------
    Episodes ends when all prey have been captured. By default a `max_episode_steps`
    limit of `50` steps is also set. This may need to be adjusted when using larger
    grids (this can be done by manually specifying a value for `max_episode_steps` when
    creating the environment with `posggym.make`).

    Arguments
    ---------

    - `grid` - the grid layout to use. This can either be a string specifying one of
         the supported grids, or a custom :class:`PPWorld` object (default = `"10x10"`).
    - `num_predators` - the number of predator (and thus controlled agents)
        (default = `2`).
    - `num_prey` - the number of prey (default = `3`)
    - `cooperative` - whether agents share all rewards or only get rewards for prey they
        are involved in capturing (default = 'True`)
    - `prey_strength` - how many predators are required to capture each prey, minimum is
        `1` and maximum is `min(4, num_predators)`. If `None` this is set to
        `min(4, num_predators)` (default = 'None`)
    - `obs_dim` - the local observation distance, specifying how far away in each
        direction each predator and prey agent observes (default = `2`).
    - `n_lines` - the number of lines eminating from the agent. The agent will observe
        at `n` equidistance intervals over `[0, 2*pi]` (default = `10`).
    - `use_holonomic_predator` - the movement model to use for the predator. There are
        two modes - holonomic or non holonmic, with a unicycle model (default = 'True`).
    - `use_holonomic_prey` - the movement model to use for the prey. There are two
        modes - holonomic or non holonmic, with a unicycle model (default = 'True`).

    Available variants
    ------------------

    The PredatorPrey environment comes with a number of pre-built grid layouts which can
    be passed as an argument to `posggym.make`, to create different grids. All layouts
    support 2 to 8 agents.

    | Grid name         | Grid size |
    |-------------------|-----------|
    | `5x5`             | 5x5       |
    | `5x5Blocks`       | 5x5       |
    | `10x10`           | 10x10     |
    | `10x10Blocks`     | 10x10     |
    | `15x15`           | 15x15     |
    | `15x15Blocks`     | 15x15     |
    | `20x20`           | 20x20     |
    | `20x20Blocks`     | 20x20     |


    For example to use the Predator Prey environment with the `15x15Blocks` grid, 4
    predators, 4 prey, and episode step limit of 100, and the default values for the
    other parameters (`cooperative`, `obs_dim`, `prey_strength`) you would use:

    ```python
    import posggym
    env = posgggym.make(
        'PredatorPreyContinuous-v0',
        max_episode_steps=100,
        grid="15x15Blocks",
        num_predators=4,
        num_prey=4
    )
    ```

    Version History
    ---------------
    - `v0`: Initial version

    Reference
    ---------
    - Ming Tan. 1993. Multi-Agent Reinforcement Learning: Independent vs. Cooperative
      Agents. In Proceedings of the Tenth International Conference on Machine Learning.
      330–337.
    - J. Z. Leibo, V. F. Zambaldi, M. Lanctot, J. Marecki, and T. Graepel. 2017.
      Multi-Agent Reinforcement Learning in Sequential Social Dilemmas. In AAMAS,
      Vol. 16. ACM, 464–473
    - Lowe, Ryan, Yi I. Wu, Aviv Tamar, Jean Harb, OpenAI Pieter Abbeel, and Igor
      Mordatch. 2017. “Multi-Agent Actor-Critic for Mixed Cooperative-Competitive
      Environments.” Advances in Neural Information Processing Systems 30.

    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 30,
    }

    def __init__(
        self,
        grid: Union[str, "PPWorld"] = "10x10",
        num_predators: int = 2,
        num_prey: int = 3,
        cooperative: bool = True,
        prey_strength: Optional[int] = None,
        obs_dim: float = 2,
        n_lines: int = 10,
        use_holonomic_predator: bool = True,
        use_holonomic_prey: bool = True,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            PPModel(
                grid,
                num_predators,
                num_prey,
                cooperative,
                prey_strength,
                obs_dim,
                n_lines,
                use_holonomic_predator,
                use_holonomic_prey,
            ),
            render_mode=render_mode,
        )
        self._obs_dim = obs_dim
        self._viewer = None
        self._renderer = None
        self.render_mode = render_mode

    def render(self):
        if self.render_mode == "human":
            import posggym.envs.continuous.render as render_lib

            if self._renderer is None:
                self._renderer = render_lib.GWContinuousRender(
                    self.render_mode,
                    env_name="PredatorPreyContinuous",
                    domain_max=self.model.grid.width,
                    render_fps=self.metadata["render_fps"],
                    num_colors=4,
                    arena_size=200,
                )

            pred = array_to_position(self._state.predator_coords)
            prey = array_to_position(self._state.prey_coords)
            colored_pred = tuple(t + (0,) for t in pred)
            colored_prey = tuple(
                t + (1 + caught,) for t, caught in zip(prey, self.state.prey_caught)
            )

            num_agents = len(colored_prey + colored_pred)

            sizes = [self.model.grid.agent_size] * num_agents

            holonomic = [self.model.use_holonomic_prey] * len(colored_prey) + [
                self.model.use_holonomic_predator
            ] * len(colored_pred)

            self._renderer.clear_render()
            self._renderer.draw_arena()
            self._renderer.render_lines(self._last_obs, self._state.predator_coords)
            self._renderer.draw_agents(
                colored_prey + colored_pred,
                sizes=sizes,
                is_holonomic=holonomic,
                alpha=255,
            )
            self._renderer.draw_blocks(self.model.grid.block_coords)
            self._renderer.render()

    def close(self) -> None:
        pass


class PPModel(M.POSGModel[PPState, PPObs, PPAction]):
    """Predator-Prey Problem Model.

    Parameters
    ----------
    grid_size : float
        the size of the grid (height and width)
    num_predators : int
        the number of predator (and thus controlled agents)
    num_prey : int
        the number of prey
    cooperative : bool
        whether environment rewards are fully shared (True) or only awarded to
        capturing predators (i.e. mixed) (False)
    prey_strenth : int
        the minimum number of predators needed to capture a prey
    obs_dims : float
        number of cells in each direction around the agent that the agent can
        observe

    """

    R_MAX = 1.0
    PREY_CAUGHT_COORD = (0, 0)

    def __init__(
        self,
        grid: Union[str, "PPWorld"],
        num_predators: int,
        num_prey: int,
        cooperative: bool,
        prey_strength: Optional[int],
        obs_dim: float,
        n_lines: int,
        use_holonomic_predator: bool,
        use_holonomic_prey: bool,
        **kwargs,
    ):
        assert 1 < num_predators <= 8
        assert num_prey > 0
        assert obs_dim > 0

        if prey_strength is None:
            prey_strength = min(4, num_predators)

        assert 0 < prey_strength <= min(4, num_predators)

        self.use_holonomic_predator = use_holonomic_predator
        self.use_holonomic_prey = use_holonomic_prey

        if isinstance(grid, str):
            assert grid in SUPPORTED_GRIDS, (
                f"Unsupported grid name '{grid}'. Grid name must be one of: "
                f"{list(SUPPORTED_GRIDS)}."
            )
            grid = SUPPORTED_GRIDS[grid][0]()

        # Cannot be a string by this point.
        self.grid = cast(PPWorld, grid)

        # self.grid.set_holonomic_model(use_holonomic)

        self.obs_dim = obs_dim
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.cooperative = cooperative
        self.prey_strength = prey_strength
        self._per_prey_reward = self.R_MAX / self.num_prey

        self.communication_radius = 1

        def _coord_space2():
            return spaces.Box(
                low=np.array([-1, -1, -0.5 * math.pi], dtype=np.float32),
                high=np.array(
                    [self.grid.width, self.grid.height, 0.5 * math.pi], dtype=np.float32
                ),
            )

        self.possible_agents = tuple((str(x) for x in range(self.num_predators)))

        self.state_space = spaces.Tuple(
            (
                # coords of each agent
                spaces.Tuple(tuple(_coord_space2() for _ in range(self.num_predators))),
                # prey coords
                spaces.Tuple(tuple(_coord_space2() for _ in range(self.num_prey))),
                # prey caught/not
                spaces.Tuple(tuple(spaces.Discrete(2) for _ in range(self.num_prey))),
            )
        )

        self.action_spaces = {
            i: spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
            )
            for i in self.possible_agents
        }

        # Observe everyones location
        # Apart from your own
        nested_space = [spaces.Discrete(4), spaces.Box(0, 30)]

        self.n_lines = n_lines

        agent_obs = spaces.Tuple(sum([nested_space for i in range(self.n_lines)], []))

        self.observation_spaces = {i: agent_obs for i in self.possible_agents}
        self.observation_first = True
        self.is_symmetric = True

    @property
    def reward_ranges(self) -> Dict[M.AgentID, Tuple[float, float]]:
        return {i: (0.0, self.R_MAX) for i in self.possible_agents}

    @property
    def rng(self) -> seeding.RNG:
        if self._rng is None:
            self._rng, seed = seeding.std_random()
        return self._rng

    def get_agents(self, state: PPState) -> List[M.AgentID]:
        return list(self.possible_agents)

    def sample_initial_state(self) -> PPState:
        predator_coords: List[Position] = [*self.grid.predator_start_coords]
        self.rng.shuffle(predator_coords)
        predator_coords = predator_coords[: self.num_predators]

        if not self.grid.prey_coords_loaded:
            prey_coords_list = self.grid.get_unblocked_center_coords(
                self.num_prey, self.rng.random, self.use_holonomic_prey, predator_coords
            )
            self.grid.prey_start_coords = prey_coords_list

        prey_coords_list = self.grid.prey_start_coords
        self.rng.shuffle(prey_coords_list)
        prey_coords_list = prey_coords_list[: self.num_prey]

        prey_caught = (0,) * self.num_prey

        return PPState(
            position_to_array(predator_coords),
            position_to_array(prey_coords_list),
            prey_caught,
        )

    def sample_initial_obs(self, state: PPState) -> Dict[M.AgentID, PPObs]:
        return self._get_obs(state, state)

    def step(
        self, state: PPState, actions: Dict[M.AgentID, PPAction]
    ) -> M.JointTimestep[PPState, PPObs]:
        clipped_actions = clip_actions(actions, self.action_spaces)

        next_state = self._get_next_state(state, clipped_actions)
        obs = self._get_obs(state, next_state)
        rewards = self._get_rewards(state, next_state)

        all_done = all(next_state.prey_caught)
        truncated = {i: False for i in self.possible_agents}
        terminated = {i: all_done for i in self.possible_agents}

        info: Dict[M.AgentID, Dict] = {i: {} for i in self.possible_agents}
        if all_done:
            for i in self.possible_agents:
                info[i]["outcome"] = M.Outcome.WIN

        return M.JointTimestep(
            next_state, obs, rewards, terminated, truncated, all_done, info
        )

    def _get_next_state(
        self, state: PPState, actions: Dict[M.AgentID, PPAction]
    ) -> PPState:
        # prey move first
        prey_coords = self._get_next_prey_state(state)
        predator_coords = self._get_next_predator_state(state, actions, prey_coords)
        prey_caught = self._get_next_prey_caught(state, prey_coords, predator_coords)
        return PPState(
            position_to_array(predator_coords),
            position_to_array(prey_coords),
            prey_caught,
        )

    def _get_next_prey_state(self, state: PPState) -> Tuple[Position, ...]:
        next_prey_coords: List[Optional[Position]] = [None] * self.num_prey
        pred_coords = array_to_position(state.predator_coords)
        prey_coords = array_to_position(state.prey_coords)

        occupied_coords = set(
            pred_coords
            + tuple(
                prey_coords[i] for i in range(self.num_prey) if not state.prey_caught[i]
            )
        )

        # handle moving away from predators for all prey
        for i in range(self.num_prey):
            prey_coord = prey_coords[i]
            if state.prey_caught[i]:
                next_prey_coords[i] = prey_coord
                continue

            occupied_coords.remove(prey_coord)

            next_coord = self._move_away_from_predators(
                prey_coord, pred_coords, list(occupied_coords)
            )
            if next_coord:
                next_prey_coords[i] = next_coord
                occupied_coords.add(next_coord)
            else:
                occupied_coords.add(prey_coord)

        if self.num_prey - sum(state.prey_caught) > 1:
            # handle moving away from other prey
            for i in range(self.num_prey):
                if next_prey_coords[i] is not None:
                    # already moved or caught
                    continue

                prey_coord = prey_coords[i]
                occupied_coords.remove(prey_coord)

                next_coord = self._move_away_from_preys(
                    prey_coord, state, list(occupied_coords)
                )
                if next_coord:
                    next_prey_coords[i] = next_coord
                    occupied_coords.add(next_coord)
                else:
                    occupied_coords.add(prey_coord)

        # Handle random moving prey for those that are out of obs range
        # of all predators and other prey
        prey_coords = array_to_position(state.prey_coords)
        for i in range(self.num_prey):
            if next_prey_coords[i] is not None:
                continue
            # No visible prey or predator
            prey_coord = prey_coords[i]
            occupied_coords.remove(prey_coord)

            if self.obs_dim > 1:
                # no chance of moving randomly into an occupied cell
                neighbours = self.grid.get_possible_next_pos(
                    prey_coord, use_holonomic_model=self.use_holonomic_prey
                )
                next_prey_coords[i] = self.rng.choice(neighbours)
            else:
                # possibility for collision between random moving prey
                neighbours = self.available_neighbours(
                    prey_coord, list(occupied_coords)
                )

                if len(neighbours) == 0:
                    next_coord = prey_coord
                else:
                    next_coord = self.rng.choice(neighbours)
                next_prey_coords[i] = next_coord
                occupied_coords.remove(prey_coord)
                occupied_coords.add(next_coord)

        return tuple(next_prey_coords)  # type: ignore

    def _move_away_from_predators(
        self,
        prey_coord: Position,
        predator_coords: Tuple[Position, ...],
        occupied_coords: List[Position],
    ) -> Optional[Position]:
        # get any predators within obs distance
        predator_dists = [
            self.grid.manhattan_dist(prey_coord, c) for c in predator_coords
        ]
        min_predator_dist = min(predator_dists)
        all_closest_predator_coords = [
            predator_coords[i]
            for i in range(self.num_predators)
            if predator_dists[i] == min_predator_dist
        ]
        closest_predator_coord = self.rng.choice(all_closest_predator_coords)

        if all(
            abs(a - b) > self.obs_dim
            for a, b in zip(prey_coord[:2], closest_predator_coord[:2])
        ):
            return None

        # move into furthest away free cell, includes current coord
        neighbours = [
            (self.grid.euclidean_dist(c, closest_predator_coord), c)
            for c in self.grid.get_possible_next_pos(
                prey_coord,
                distance=1.02,
                num_samples=50,
                use_holonomic_model=self.use_holonomic_prey,
            )
            + [prey_coord]
        ]

        neighbours.sort()

        for d, c in reversed(neighbours):
            if c == prey_coord or self._coord_available_for_prey(c, occupied_coords):
                return c

        raise AssertionError("Something has gone wrong, please investigate.")

    def _move_away_from_preys(
        self, prey_coord: Position, state: PPState, occupied_coords: List[Position]
    ) -> Optional[Position]:
        prey_coords = array_to_position(state.prey_coords)
        prey_dists = [
            self.grid.manhattan_dist(prey_coord, c)
            if (c != prey_coord and not state.prey_caught[i])
            else float("inf")
            for i, c in enumerate(prey_coords)
        ]
        min_prey_dist = min(prey_dists)
        all_closest_prey_coords = [
            prey_coords[i]
            for i in range(self.num_prey)
            if prey_dists[i] == min_prey_dist
        ]
        closest_prey_coord = self.rng.choice(all_closest_prey_coords)

        if all(
            abs(a - b) > self.obs_dim for a, b in zip(prey_coord, closest_prey_coord)
        ):
            # closes predator out of obs range
            return None

        # move into furthest away free cell, includes current coord
        neighbours = [
            (self.grid.manhattan_dist(c, closest_prey_coord), c)
            for c in self.grid.get_possible_next_pos(
                prey_coord, use_holonomic_model=self.use_holonomic_prey
            )
            + [prey_coord]
        ]
        neighbours.sort()
        for d, c in reversed(neighbours):
            if c == prey_coord or self._coord_available_for_prey(c, occupied_coords):
                return c

        raise AssertionError("Something has gone wrong, please investigate.")

    def available_neighbours(
        self, coord: Position, occupied_coords: List[Position]
    ) -> List[Position]:
        neighbours = self.grid.get_neighbours(
            coord,
            ignore_blocks=False,
            include_out_of_bounds=False,
            force_non_colliding=True,
        )

        non_collide_neighbours = []
        for c in neighbours:
            flag = True
            for c2 in occupied_coords:
                if self.grid.agents_collide(c, c2):
                    flag = False
                    break

            # No collision
            if flag:
                non_collide_neighbours.append(c)

        return non_collide_neighbours

    def _coord_available_for_prey(
        self, coord: Position, occupied_coords: List[Position]
    ) -> bool:
        if any(self.grid.agents_collide(coord, oc) for oc in occupied_coords):
            return False

        non_collide_neighbours = self.available_neighbours(coord, occupied_coords)
        return len(non_collide_neighbours) >= self.prey_strength

    def _get_next_predator_state(
        self,
        state: PPState,
        actions: Dict[M.AgentID, PPAction],
        next_prey_coords: Tuple[Position, ...],
    ) -> Tuple[Position, ...]:
        potential_next_coords = []
        for i, coord in enumerate(state.predator_coords):
            next_coord, _ = self.grid._get_next_coord(
                coord,
                actions[str(i)],
                ignore_blocks=False,
                use_holonomic_model=self.use_holonomic_predator,
            )

            if self.grid.check_agent_collisions(next_coord, next_prey_coords):
                next_coord = coord

            potential_next_coords.append(next_coord)

        # handle collisions
        next_coords = []
        for i in range(self.num_predators):
            coord_i = potential_next_coords[i]
            collision = False
            for j in range(self.num_predators):
                if i == j:
                    continue
                elif self.grid.agents_collide(coord_i, potential_next_coords[j]):
                    collision = True
                    break
            if collision:
                next_coords.append(state.predator_coords[i])
            else:
                next_coords.append(coord_i)

        return tuple(next_coords)

    def _get_next_prey_caught(
        self,
        state: PPState,
        next_prey_coords: Tuple[Position, ...],
        next_predator_coords: Tuple[Position, ...],
    ) -> Tuple[int, ...]:
        prey_caught = []

        for i in range(self.num_prey):
            if state.prey_caught[i]:
                prey_caught.append(1)
            else:
                predator_dists = [
                    self.grid.manhattan_dist(next_prey_coords[i], c)
                    for c in next_predator_coords
                ]
                num_adj_predators = sum(d <= collision_distance for d in predator_dists)
                prey_caught.append(int(num_adj_predators >= self.prey_strength))
        return tuple(prey_caught)

    def _get_obs(self, state: PPState, next_state: PPState) -> Dict[M.AgentID, PPObs]:
        a = {i: self._get_local_obs(i, state, next_state) for i in self.possible_agents}
        return a

    def _get_local_obs(
        self, agent_id: M.AgentID, state: PPState, next_state: PPState
    ) -> Tuple[Union[int, np.ndarray], ...]:
        self_pos = state.predator_coords[int(agent_id)]

        line_dist = self.obs_dim
        tmp_obs: List[Union[int, np.ndarray]] = []
        for i in range(self.n_lines):
            angle = 2 * math.pi * i / self.n_lines

            closest_agent_index, closest_agent_distance = self.grid.check_collision_ray(
                self_pos,
                line_dist,
                angle,
                other_agents=array_to_position(
                    np.concatenate((state.predator_coords, state.prey_coords), axis=0)
                ),
                skip_id=int(agent_id),
            )
            if closest_agent_index is None:
                agent_collision = EMPTY
            elif closest_agent_index == -1:
                agent_collision = WALL
            elif closest_agent_index < len(state.predator_coords):
                agent_collision = PREDATOR
            else:
                agent_collision = PREY

            tmp_obs.append(agent_collision)
            tmp_obs.append(np.array([closest_agent_distance], dtype=np.float32))

        return tuple(tmp_obs)

    def _get_rewards(
        self, state: PPState, next_state: PPState
    ) -> Dict[M.AgentID, float]:
        new_caught_prey = []
        for i in range(self.num_prey):
            if not state.prey_caught[i] and next_state.prey_caught[i]:
                new_caught_prey.append(next_state.prey_coords[i])

        if len(new_caught_prey) == 0:
            return {i: 0.0 for i in self.possible_agents}

        if self.cooperative:
            reward = len(new_caught_prey) * (self._per_prey_reward)
            return {i: reward for i in self.possible_agents}

        rewards = {i: 0.0 for i in self.possible_agents}
        for prey_coord in new_caught_prey:
            involved_predators = []
            for index, predator in enumerate(next_state.predator_coords):
                if self.grid.manhattan_dist(predator, prey_coord) < collision_distance:
                    involved_predators.append(index)

            predator_reward = self._per_prey_reward / len(involved_predators)
            for i in involved_predators:
                rewards[str(i)] += predator_reward

        return rewards


class PPWorld(RectangularContinuousWorld):
    """A grid for the Predator-Prey Problem."""

    def __init__(
        self,
        grid_size: float,
        block_coords: Optional[List[Object]],
        predator_start_coords: Optional[List[Position]] = None,
        prey_start_coords: Optional[List[Position]] = None,
        predator_angles: Optional[List[float]] = None,
    ):
        assert grid_size >= 3
        super().__init__(grid_size, grid_size, block_coords)
        self.size = grid_size
        # predators start in corners or half-way along a side
        if predator_start_coords is None:
            predator_start_coords_: List[Tuple[float, float]] = [
                (float(c[0] + 0.5), float(c[1] + 0.5))  # type: ignore
                for c in product([0, grid_size // 2, grid_size - 1], repeat=2)
                if c[0] in (0, grid_size - 1) or c[1] in (0, grid_size - 1)
            ]

            predator_angles = [0.0] * len(predator_start_coords_)

            predator_start_coords = [
                x + (y,) for x, y in zip(predator_start_coords_, predator_angles)
            ]
        self.predator_start_coords = predator_start_coords

        self.prey_coords_loaded = prey_start_coords is not None
        self.prey_start_coords = prey_start_coords

    def get_unblocked_center_coords(
        self,
        num: int,
        rng: Callable[[], float],
        use_holonomoic: bool = True,
        predator_coords: List[Position] = [],
    ) -> List[Position]:
        """Get at least num closest coords to the center of grid.

        May return more than num, since can be more than one coord at equal
        distance from the center.
        """
        # assert num < self.n_coords - len(self.block_coords)
        center = (self.width / 2, self.height / 2, 0)
        min_dist_from_center = math.ceil(math.sqrt(num)) - 1
        coords: List[Position] = []

        while len(coords) < num:
            coords_sampled = [
                self.sample_coords_within_dist(
                    center,
                    min_dist_from_center,
                    ignore_blocks=False,
                    rng=rng,
                    use_holonomic_model=use_holonomoic,
                )
                for _ in range(num * 20)
            ]

            for c in coords_sampled:
                flag = True
                for c_f in coords + predator_coords:
                    if self.agents_collide(c, c_f):
                        flag = False

                if flag:
                    coords.append(c)

            min_dist_from_center += 1
        return coords


def parse_grid_str(grid_str: str) -> PPWorld:
    """Parse a str representation of a grid into a grid object.

    Notes on grid str representation:

    . = empty/unallocated cell
    # = a block
    P = starting location for predator agents [optional] (defaults to edges)
    p = starting location for prey agent [optional] (defaults to center)

    Examples (" " quotes and newline chars omitted):

    1. A 10x10 grid with 4 groups of blocks and using the default predator
       and prey start locations.

    ..........
    ..........
    ..##..##..
    ..##..##..
    ..........
    ..........
    ..##..##..
    ..##..##..
    ..........
    ..........

    2. Same as above but with predator and prey start locations defined for
    up to 8 predators and 4 prey. (This would be the default layout for the
    scenario where there are between 2 and 4 prey, i.e. if prey and predator
    start locations were left unspecified as in example 1.)

    P....P...P
    ..........
    ..##..##..
    ..##..##..
    ....pp....
    P...pp...P
    ..##..##..
    ..##..##..
    ..........
    P....P...P

    """
    row_strs = grid_str.splitlines()
    assert len(row_strs) > 1
    assert all(len(row) == len(row_strs[0]) for row in row_strs)
    assert len(row_strs[0]) > 1
    assert len(row_strs) == len(row_strs[0])

    grid_size = len(row_strs)
    block_coords: Set[Object] = set()
    predator_coords = set()
    prey_coords = set()
    for r, c in product(range(grid_size), repeat=2):
        # This is offset to the center of the square
        coord = (c + 0.5, r + 0.5, 0)
        char = row_strs[r][c]

        if char == "#":
            # Radius is 0.5
            block_coords.add((coord, 0.5))
        elif char == "P":
            predator_coords.add(coord)
        elif char == "p":
            prey_coords.add(coord)
        else:
            assert char == "."

    return PPWorld(
        grid_size,
        block_coords=list(block_coords),
        predator_start_coords=None
        if len(predator_coords) == 0
        else list(predator_coords),
        prey_start_coords=None if len(prey_coords) == 0 else list(prey_coords),
    )


def get_5x5_grid() -> PPWorld:
    """Generate 5x5 grid layou`t."""
    return PPWorld(grid_size=5, block_coords=None)


def get_5x5_blocks_grid() -> PPWorld:
    """Generate 5x5 Blocks grid layou`t."""
    grid_str = ".....\n" ".#.#.\n" ".....\n" ".#.#.\n" ".....\n"
    return parse_grid_str(grid_str)


def get_10x10_grid() -> PPWorld:
    """Generate 10x10 grid layou`t."""
    return PPWorld(grid_size=10, block_coords=None)


def get_10x10_blocks_grid() -> PPWorld:
    """Generate 10x10 Blocks grid layou`t."""
    grid_str = (
        "..........\n"
        "..........\n"
        "..##..##..\n"
        "..##..##..\n"
        "..........\n"
        "..........\n"
        "..##..##..\n"
        "..##..##..\n"
        "..........\n"
        "..........\n"
    )
    return parse_grid_str(grid_str)


def get_15x15_grid() -> PPWorld:
    """Generate 15x15 grid layou`t."""
    return PPWorld(grid_size=15, block_coords=None)


def get_15x15_blocks_grid() -> PPWorld:
    """Generate 10x10 Blocks grid layou`t."""
    grid_str = (
        "...............\n"
        "...............\n"
        "...............\n"
        "...###...###...\n"
        "...###...###...\n"
        "...###...###...\n"
        "...............\n"
        "...............\n"
        "...............\n"
        "...###...###...\n"
        "...###...###...\n"
        "...###...###...\n"
        "...............\n"
        "...............\n"
        "...............\n"
    )
    return parse_grid_str(grid_str)


def get_20x20_grid() -> PPWorld:
    """Generate 20x20 grid layou`t."""
    return PPWorld(grid_size=20, block_coords=None)


def get_20x20_blocks_grid() -> PPWorld:
    """Generate 20x20 Blocks grid layou`t."""
    grid_str = (
        "....................\n"
        "....................\n"
        "....................\n"
        "....................\n"
        "....####....####....\n"
        "....####....####....\n"
        "....####....####....\n"
        "....####....####....\n"
        "....................\n"
        "....................\n"
        "....................\n"
        "....................\n"
        "....####....####....\n"
        "....####....####....\n"
        "....####....####....\n"
        "....####....####....\n"
        "....................\n"
        "....................\n"
        "....................\n"
        "....................\n"
    )
    return parse_grid_str(grid_str)


#  (grid_make_fn, step_limit)
SUPPORTED_GRIDS = {
    "5x5": (get_5x5_grid, 25),
    "5x5Blocks": (get_5x5_blocks_grid, 50),
    "10x10": (get_10x10_grid, 50),
    "10x10Blocks": (get_10x10_blocks_grid, 50),
    "15x15": (get_15x15_grid, 100),
    "15x15Blocks": (get_15x15_blocks_grid, 100),
    "20x20": (get_20x20_grid, 200),
    "20x20Blocks": (get_20x20_blocks_grid, 200),
}


def position_to_array(coords: Iterable[Position]) -> np.ndarray:
    return np.array([np.array(x, dtype=np.float32) for x in coords], dtype=np.float32)


def array_to_position(coords: np.ndarray) -> Tuple[Position, ...]:
    if coords.ndim == 2:
        assert coords.shape[1] == 3
        output = []
        for i in range(coords.shape[0]):
            output.append(tuple(coords[i, :]))
        return tuple(output)  # type: ignore

    elif coords.ndim == 1:
        assert coords.shape[0] == 3
        return tuple(coords)
    else:
        raise Exception("Cannot convert")
