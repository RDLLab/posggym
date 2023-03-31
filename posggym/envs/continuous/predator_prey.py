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
import warnings
from itertools import product
from typing import Dict, List, NamedTuple, Optional, Set, Tuple, Union, cast

import numpy as np
import pymunk
from gymnasium import spaces
from pymunk import Vec2d

import posggym.model as M
from posggym.core import DefaultEnv
from posggym.envs.continuous.core import Coord, Object, Position, clip_actions
from posggym.utils import seeding


class PPState(NamedTuple):
    """A state in the Continuous Predator-Prey Environment."""

    predator_states: np.ndarray
    prey_states: np.ndarray
    prey_caught: np.ndarray


NUM_STATE_FEATURES = 6


class _PMBodyState(NamedTuple):
    """State of a Pymunk Body."""

    x: float
    y: float
    angle: float  # in radians
    vx: float
    vy: float
    vangle: float  # in radians/s


# Obs vector order
WALL_OBS_IDX = 0
PREDATOR_OBS_IDX = 1
PREY_OBS_IDX = 2
PPObs = np.ndarray

# Actions
PPAction = np.ndarray


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
    Each agent observes a local circle around themselves as a vector. This is achieved
    by a series of 'n_sensors' lines starting at the agent which extend for a distance
    of 'obs_dist'. For each line the agent observes the closest entity (wall, predator,
    prey) along the line. This table enumerates the observation space:

    |        Index: [start, end)        | Description                       | Values |
    | :-------------------------------: | --------------------------------: | :----: |
    |           0 - n_sensors           | Wall distance for each sensor     | [0, d] |
    |    n_sensors - (2 * n_sensors)    | Predator distance for each sensor | [0, d] |
    | (2 * n_sensors) - (3 * n_sensors) | Prey distance for each sensor     | [0, d] |


    Where `d = obs_dist`.

    If an entity is not observed (i.e. there is none along the sensor's line or it
    isn't the closest entity to the observing agent along the line), The distance will
    be 1.

    The sensor reading ordering is relative to the agent's direction.

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

    1. if predator is within `obs_dist` distance, moves away from closest predator
    2. if another prey is within `obs_dist` distance, moves away from closest prey
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
    worlds (this can be done by manually specifying a value for `max_episode_steps` when
    creating the environment with `posggym.make`).

    Arguments
    ---------

    - `world` - the world layout to use. This can either be a string specifying one of
        the supported worlds, or a custom :class:`PPWorld` object (default = `"10x10"`).
    - `num_predators` - the number of predator (and thus controlled agents)
        (default = `2`).
    - `num_prey` - the number of prey (default = `3`)
    - `cooperative` - whether agents share all rewards or only get rewards for prey they
        are involved in capturing (default = 'True`)
    - `prey_strength` - how many predators are required to capture each prey, minimum is
        `1` and maximum is `min(4, num_predators)`. If `None` this is set to
        `min(4, num_predators)` (default = 'None`)
    - `obs_dist` - the local observation distance, specifying how far away in each
        direction each predator and prey agent observes (default = `2`).
    - `n_sensors` - the number of lines eminating from the agent. The agent will observe
        at `n` equidistance intervals over `[0, 2*pi]` (default = `10`).
    - `use_holonomic_predator` - the movement model to use for the predator. There are
        two modes - holonomic or non holonmic, with a unicycle model (default = 'True`).
    - `use_holonomic_prey` - the movement model to use for the prey. There are two
        modes - holonomic or non holonmic, with a unicycle model (default = 'True`).

    Available variants
    ------------------

    The PredatorPrey environment comes with a number of pre-built world layouts which
    can be passed as an argument to `posggym.make`, to create different worlds. All
    layouts support 2 to 8 agents.

    | World name         | World size |
    |-------------------|-----------|
    | `5x5`             | 5x5       |
    | `5x5Blocks`       | 5x5       |
    | `10x10`           | 10x10     |
    | `10x10Blocks`     | 10x10     |
    | `15x15`           | 15x15     |
    | `15x15Blocks`     | 15x15     |
    | `20x20`           | 20x20     |
    | `20x20Blocks`     | 20x20     |


    For example to use the Predator Prey environment with the `15x15Blocks` world, 4
    predators, 4 prey, and episode step limit of 100, and the default values for the
    other parameters (`cooperative`, `obs_dist`, `prey_strength`) you would use:

    ```python
    import posggym
    env = posggym.make(
        'PredatorPreyContinuous-v0',
        max_episode_steps=100,
        world="15x15Blocks",
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
        "render_fps": 15,
    }

    def __init__(
        self,
        world: Union[str, "PPWorld"] = "10x10",
        num_predators: int = 2,
        num_prey: int = 3,
        cooperative: bool = True,
        prey_strength: Optional[int] = None,
        obs_dist: float = 2,
        n_sensors: int = 10,
        use_holonomic: bool = True,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            PPModel(
                world,
                num_predators,
                num_prey,
                cooperative,
                prey_strength,
                obs_dist,
                n_sensors,
                use_holonomic,
            ),
            render_mode=render_mode,
        )
        self._obs_dist = obs_dist
        self.screen = None
        self.clock = None
        self.screen_size = 600
        self.draw_options = None
        self.pm_space_and_bodies = None

        # TODO use pymunk -> pygame rendering
        # TODO use transform to rescale

    def render(self):
        if self.render_mode == "human":
            # import posggym.envs.continuous.render as render_lib
            import pygame
            from pymunk import pygame_util

            model = cast(PPModel, self.model)
            state = cast(PPState, self.state)

            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_size, self.screen_size)
                )
                # Turn off alpha since we don't use it.
                self.screen.set_alpha(None)
                self.screen.fill(pygame.Color("white"))

            if self.clock is None:
                self.clock = pygame.time.Clock()

            if self.draw_options is None:
                pygame_util.positive_y_is_up = False
                self.draw_options = pygame_util.DrawOptions(self.screen)
                self.draw_options.transform = pymunk.Transform.scaling(
                    self.screen_size // model.world.size
                )
                # make collision lines transparent
                self.draw_options.flags = (
                    pygame_util.DrawOptions.DRAW_SHAPES
                    | pygame_util.DrawOptions.DRAW_CONSTRAINTS
                )

            if self.pm_space_and_bodies is None:
                self.pm_space_and_bodies = model.get_populated_pm_space(
                    model.world.size,
                    model.num_predators,
                    model.num_prey,
                    model.world.blocks,
                    model.world.agent_radius,
                )

            space, predators, preys = self.pm_space_and_bodies

            for (body, _), p_state in zip(predators, state.predator_states):
                model.set_body_state(body, p_state)

            for (body, shape), p_state, p_caught in zip(
                preys, state.prey_states, state.prey_caught
            ):
                model.set_body_state(body, p_state)
                if p_caught:
                    shape.color = (0, 255, 0, 255)

            # Need to do this for space to update with changes
            space.step(0.000001)

            self.screen.fill(pygame.Color("white"))

            # draw sensor lines
            n_sensors = model.n_sensors
            scale_factor = self.screen_size // model.world.size
            for i, obs_i in self._last_obs.items():
                p_state = state.predator_states[int(i)]
                x, y, agent_angle = p_state[:3]
                angle_inc = 2 * math.pi / n_sensors
                for k in range(n_sensors):
                    dist = min(obs_i[k], obs_i[n_sensors + k], obs_i[2 * n_sensors + k])
                    angle = angle_inc * k + agent_angle
                    end_x = x + dist * math.cos(angle)
                    end_y = y + dist * math.sin(angle)
                    scaled_start = (int(x * scale_factor), int(y * scale_factor))
                    scaled_end = int(end_x * scale_factor), (end_y * scale_factor)

                    pygame.draw.line(
                        self.screen, pygame.Color("red"), scaled_start, scaled_end
                    )

            # self.screen.fill(pygame.Color("white"))
            space.debug_draw(self.draw_options)
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def close(self) -> None:
        pass


class PPModel(M.POSGModel[PPState, PPObs, PPAction]):
    """Predator-Prey Problem Model.

    Parameters
    ----------
    world_size : float
        the size of the world (height and width)
    num_predators : int
        the number of predator (and thus controlled agents)
    num_prey : int
        the number of prey
    cooperative : bool
        whether environment rewards are fully shared (True) or only awarded to
        capturing predators (i.e. mixed) (False)
    prey_strenth : int
        the minimum number of predators needed to capture a prey
    obs_dists : float
        number of cells in each direction around the agent that the agent can
        observe
    n_sensors : int
        the number of sensor lines for each predator
    use_holonomic : bool
        whether to use holonomic dynamics (each action specifies dx, dy of agent) or
        non-holonomic dynamics (each action is a dyaw, velocity of agent)

    """

    R_MAX = 1.0
    # max distance a predator or prey agent can move in a single step
    STEP_VEL = 0.5
    # distance predator can be from prey to be considered to be within catching range
    COLLISION_DIST = 1.2

    # Green
    PREDATOR_COLOR = (55, 155, 205, 255)  # Blueish
    PREY_COLOR = (110, 55, 155, 255)  # purpleish
    BLOCK_COLOR = (0, 0, 0, 255)  # Black

    @staticmethod
    def get_body_state(body: pymunk.Body) -> _PMBodyState:
        """Get state of pymunk Body."""
        x, y = body.position
        vx, vy = body.velocity
        return _PMBodyState(x, y, body.angle, vx, vy, body.angular_velocity)

    @staticmethod
    def set_body_state(body: pymunk.Body, state: Union[_PMBodyState, np.ndarray]):
        """Set the state of a pymunk Body."""
        body.position = Vec2d(state[0], state[1])
        body.angle = state[2]
        body.velocity = Vec2d(state[3], state[4])
        body.angular_velocity = state[5]

    @staticmethod
    def add_walls_to_space(space: pymunk.Space, size: int, thickness: float = 0.1):
        """Add walls to pymunk space."""
        for start, end in [
            ((0, thickness), (0, size)),  # bottom
            ((thickness, size), (size, size)),  # top
            ((size - thickness, size), (size - thickness, thickness)),  # right
            ((thickness, thickness), (size, thickness)),  # left
        ]:
            wall = pymunk.Segment(space.static_body, start, end, thickness)
            wall.friction = 1.0
            wall.collision_type = 1
            wall.color = PPModel.BLOCK_COLOR
            space.add(wall)

    @staticmethod
    def add_blocks_to_space(space: pymunk.Space, blocks: List[Object]):
        """Add blocks to pymunk space."""
        for pos, radius in blocks:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            shape = pymunk.Circle(body, radius)
            body.position = Vec2d(pos[0], pos[1])
            shape.elasticity = 0.0  # no bouncing
            shape.color = PPModel.BLOCK_COLOR
            space.add(body, shape)

    @staticmethod
    def get_predators_bodies(
        num: int, radius: float
    ) -> List[Tuple[pymunk.Body, pymunk.Circle]]:
        """Get predator pymunk Body instances."""
        mass = 1.0
        inertia = pymunk.moment_for_circle(mass, 0.0, radius)
        predators = []
        for i in range(num):
            body = pymunk.Body(mass, inertia)
            shape = pymunk.Circle(body, radius)
            shape.elasticity = 0.0  # no bouncing
            shape.color = PPModel.PREDATOR_COLOR  # needed for rendering
            predators.append((body, shape))
        return predators

    @staticmethod
    def get_prey_bodies(
        num: int, radius: float
    ) -> List[Tuple[pymunk.Body, pymunk.Circle]]:
        """Get prey pymunk Body instances."""
        mass = 1.0
        inertia = pymunk.moment_for_circle(mass, 0.0, radius)
        preys = []
        for i in range(num):
            body = pymunk.Body(mass, inertia)
            shape = pymunk.Circle(body, radius)
            shape.elasticity = 0.0  # no bouncing
            shape.color = PPModel.PREY_COLOR  # needed for rendering
            preys.append((body, shape))
        return preys

    @staticmethod
    def get_populated_pm_space(
        size: int,
        num_predators: int,
        num_prey: int,
        blocks: List[Object],
        agent_radius: float,
        border_thickness: float = 0.1,
    ) -> Tuple[
        pymunk.Space,
        List[Tuple[pymunk.Body, pymunk.Circle]],
        List[Tuple[pymunk.Body, pymunk.Circle]],
    ]:
        """Get pymunk Space populated with predators, prey, and obstacles.

        Also return Body instances for predators and prey.
        """
        space = pymunk.Space()
        space.gravity = Vec2d(0.0, 0.0)
        PPModel.add_walls_to_space(space, size, thickness=0.1)
        PPModel.add_blocks_to_space(space, blocks)

        predators = PPModel.get_predators_bodies(num_predators, agent_radius)
        for body, shape in predators:
            space.add(body, shape)

        preys = PPModel.get_prey_bodies(num_prey, agent_radius)
        for body, shape in preys:
            space.add(body, shape)

        return space, predators, preys

    def __init__(
        self,
        world: Union[str, "PPWorld"],
        num_predators: int,
        num_prey: int,
        cooperative: bool,
        prey_strength: Optional[int],
        obs_dist: float,
        n_sensors: int,
        use_holonomic: bool,
        **kwargs,
    ):
        assert 1 < num_predators <= 8
        assert num_prey > 0
        assert obs_dist > 0

        if prey_strength is None:
            prey_strength = min(4, num_predators)

        assert 0 < prey_strength <= min(4, num_predators)

        if isinstance(world, str):
            assert world in SUPPORTED_WORLDS, (
                f"Unsupported world name '{world}'. World name must be one of: "
                f"{list(SUPPORTED_WORLDS)}."
            )
            world = SUPPORTED_WORLDS[world][0]()
        # Cannot be a string by this point.
        self.world = cast(PPWorld, world)

        assert len(self.world.predator_start_positions) >= num_predators, (
            f"World of size ({self.world.size, self.world.size}) cannot support "
            f"{num_predators} predators. Try with less predators, using a larger world,"
            " or using a different world layout."
        )
        assert len(self.world.prey_start_positions) >= num_prey, (
            f"World of size ({self.world.size, self.world.size}) cannot support "
            f"{num_prey} prey. Try with less prey, using a larger world, or using a "
            "different world layout."
        )

        # self.world.set_holonomic_model(use_holonomic)
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.obs_dist = obs_dist
        self.cooperative = cooperative
        self.prey_strength = prey_strength
        self._per_prey_reward = self.R_MAX / self.num_prey
        self.n_sensors = n_sensors
        self.use_holonomic = use_holonomic
        self.communication_radius = 1

        def _pos_space(n_agents: int):
            # x, y, angle, vx, vy, vangle
            # stacked n_agents time
            # shape = (n_agents, 6)
            low = np.array(
                [-1, -1, -2 * math.pi, -1, -1, -2 * math.pi], dtype=np.float32
            )
            size = self.world.size
            high = np.array(
                [
                    size,
                    size,
                    2 * math.pi,
                    1.0,
                    1.0,
                    2 * math.pi,
                ],
                dtype=np.float32,
            )
            return spaces.Box(
                low=np.tile(low, (n_agents, 1)), high=np.tile(high, (n_agents, 1))
            )

        self.possible_agents = tuple((str(x) for x in range(self.num_predators)))
        self.state_space = spaces.Tuple(
            (
                # state of each predator
                _pos_space(self.num_predators),
                # state of each prey
                _pos_space(self.num_prey),
                # prey caught/not
                spaces.MultiBinary(self.num_prey),
            )
        )

        self.dyaw_limit = math.pi / 10
        if self.use_holonomic:
            # dx, dy
            self.action_spaces = {
                i: spaces.Box(
                    low=np.array([-1.0, -1.0], dtype=np.float32),
                    high=np.array([1.0, 1.0], dtype=np.float32),
                )
                for i in self.possible_agents
            }
        else:
            # dyaw, vel
            self.action_spaces = {
                i: spaces.Box(
                    low=np.array([-self.dyaw_limit, 0.0], dtype=np.float32),
                    high=np.array([self.dyaw_limit, 1.0], dtype=np.float32),
                )
                for i in self.possible_agents
            }

        # Observes entity and distance to entity along a n_sensors rays from the agent
        # 0 to n_sensors = wall distance obs
        # n_sensors to (2 * n_sensors) = pred dist
        # (2 * n_sensors) to (3 * n_sensors) = prey dist
        self.obs_dim = self.n_sensors * 3
        self.observation_spaces = {
            i: spaces.Box(
                low=0.0, high=self.obs_dist, shape=(self.obs_dim,), dtype=np.float32
            )
            for i in self.possible_agents
        }

        # All predators are identical so env is symmetric
        self.is_symmetric = True

        # 2D physics setup
        self.space, self.predators, self.preys = self.get_populated_pm_space(
            self.world.size,
            self.num_predators,
            self.num_prey,
            self.world.blocks,
            self.world.agent_radius,
        )

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
        predator_positions = [*self.world.predator_start_positions]
        self.rng.shuffle(predator_positions)
        predator_states = np.zeros(
            (self.num_predators, NUM_STATE_FEATURES), dtype=np.float32
        )
        for i in range(self.num_predators):
            predator_states[i][:3] = predator_positions[i]

        prey_positions = [*self.world.prey_start_positions]
        self.rng.shuffle(prey_positions)
        prey_states = np.zeros((self.num_prey, NUM_STATE_FEATURES), dtype=np.float32)
        for i in range(self.num_prey):
            prey_states[i][:3] = prey_positions[i]

        return PPState(
            predator_states,
            prey_states,
            np.zeros(self.num_prey, dtype=np.int8),
        )

    def sample_initial_obs(self, state: PPState) -> Dict[M.AgentID, PPObs]:
        return self.get_obs(state)

    def step(
        self, state: PPState, actions: Dict[M.AgentID, PPAction]
    ) -> M.JointTimestep[PPState, PPObs]:
        clipped_actions = clip_actions(actions, self.action_spaces)

        next_state = self._get_next_state(state, clipped_actions)
        obs = self.get_obs(next_state)
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
        prey_move_angles = self._get_prey_move_angles(state)

        # apply prey actions
        for i in range(self.num_prey):
            if state.prey_caught[i]:
                # do nothing
                continue
            body, _ = self.preys[i]
            self.set_body_state(body, state.prey_states[i])
            body.angle = prey_move_angles[i]
            body.velocity = self.STEP_VEL * Vec2d(1, 0).rotated(body.angle)

        # apply predator actions
        for i in range(self.num_predators):
            body, _ = self.predators[i]
            action = actions[str(i)]
            self.set_body_state(body, state.predator_states[i])
            if self.use_holonomic:
                body.velocity = Vec2d(action[0], action[1])
                body.angle = math.atan2(action[1], action[0])
            else:
                body.angle += action[0]
                body.velocity = action[1] * Vec2d(1, 0).rotated(body.angle)

        # simulate
        for _ in range(10):
            self.space.step(1.0 / 10)

        # extract next state
        next_pred_states = np.array(
            [self.get_body_state(body) for body, _ in self.predators], dtype=np.float32
        )
        next_prey_states = np.array(
            [self.get_body_state(body) for body, _ in self.preys], dtype=np.float32
        )

        next_prey_caught = state.prey_caught.copy()
        for i in range(self.num_prey):
            if state.prey_caught[i]:
                next_prey_states[i] = [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0]
                continue
            pred_dists = np.linalg.norm(
                next_prey_states[i][:2] - next_pred_states[:, :2], axis=1
            )
            if (
                np.where(pred_dists <= self.COLLISION_DIST, 1, 0).sum()
                >= self.prey_strength
            ):
                next_prey_caught[i] = 1
                next_prey_states[i] = [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0]
        return PPState(next_pred_states, next_prey_states, next_prey_caught)

    def _get_prey_move_angles(self, state: PPState) -> List[float]:
        prey_actions = []
        active_prey = self.num_prey - state.prey_caught.sum()
        for i in range(self.num_prey):
            if state.prey_caught[i]:
                # prey stays in same position
                prey_actions.append(0.0)
                continue

            prey_state = state.prey_states[i]
            # try move away from predators
            pred_states = state.predator_states
            pred_dists = np.linalg.norm(prey_state[:2] - pred_states[:, :2], axis=1)
            min_pred_dist = pred_dists.min()
            if min_pred_dist <= self.obs_dist:
                # get any predators within obs distance
                pred_idx = self.rng.choice(np.where(pred_dists == min_pred_dist)[0])
                pred_state = state.predator_states[pred_idx]
                angle = math.atan2(
                    prey_state[1] - pred_state[1], prey_state[0] - pred_state[0]
                )
                prey_actions.append(angle)
                continue

            if active_prey == 1:
                # no other prey to move away from so just move randomly
                angle = self.rng.uniform(0, 2 * math.pi)
                prey_actions.append(angle)
                continue

            # try move away from prey
            prey_dists = [
                np.linalg.norm(prey_state[:2] - p[:2])
                for j, p in enumerate(state.prey_states)
                if not state.prey_caught[j] and j != i
            ]
            min_prey_dist = min(prey_dists)
            if min_prey_dist <= self.obs_dist:
                other_prey_idx = self.rng.choice(
                    np.where(prey_dists == min_prey_dist)[0]
                )
                other_prey_state = state.prey_states[other_prey_idx]
                angle = math.atan2(
                    prey_state[1] - other_prey_state[1],
                    prey_state[0] - other_prey_state[0],
                )
                prey_actions.append(angle)
                continue

            # move in random direction
            angle = self.rng.uniform(0, 2 * math.pi)
            prey_actions.append(angle)

        return prey_actions

    def get_obs(self, state: PPState) -> Dict[M.AgentID, PPObs]:
        return {i: self._get_local_obs(i, state) for i in self.possible_agents}
        # return {
        #     i: np.full(
        #         shape=(self.obs_dim,), fill_value=self.obs_dist, dtype=np.float32
        #     )
        #     for i in self.possible_agents
        # }

    def _get_local_obs(self, agent_id: M.AgentID, state: PPState) -> np.ndarray:
        state_i = state.predator_states[int(agent_id)]
        pos_i = (state_i[0], state_i[1], state_i[2])

        prey_coords = np.array(
            [
                [s[0], s[1]]
                for i, s in enumerate(state.prey_states)
                if not state.prey_caught[i]
            ]
        )
        prey_obs = self.world.check_collision_ray(
            pos_i,
            self.obs_dist,
            self.n_sensors,
            prey_coords,
            include_blocks=False,
            check_walls=False,
            use_relative_angle=True,
        )

        pred_coords = np.array(
            [
                [s[0], s[1]]
                for i, s in enumerate(state.predator_states)
                if i != int(agent_id)
            ]
        )
        pred_obs = self.world.check_collision_ray(
            pos_i,
            self.obs_dist,
            self.n_sensors,
            pred_coords,
            include_blocks=False,
            check_walls=False,
            use_relative_angle=True,
        )

        obstacle_obs = self.world.check_collision_ray(
            pos_i,
            self.obs_dist,
            self.n_sensors,
            other_agents=None,
            include_blocks=True,
            check_walls=True,
            use_relative_angle=True,
        )

        obs = np.full((self.obs_dim,), self.obs_dist, dtype=np.float32)
        # TODO try and vectorize this
        for k in range(self.n_sensors):
            sensor_readings = [obstacle_obs[k], pred_obs[k], prey_obs[k]]
            min_val = min(sensor_readings)
            min_idx = sensor_readings.index(min_val)
            obs[min_idx * self.n_sensors + k] = min_val

        return obs

    def _get_rewards(
        self, state: PPState, next_state: PPState
    ) -> Dict[M.AgentID, float]:
        new_caught_prey = []
        for i in range(self.num_prey):
            if not state.prey_caught[i] and next_state.prey_caught[i]:
                new_caught_prey.append(next_state.prey_states[i])

        if len(new_caught_prey) == 0:
            return {i: 0.0 for i in self.possible_agents}

        if self.cooperative:
            reward = len(new_caught_prey) * (self._per_prey_reward)
            return {i: reward for i in self.possible_agents}

        rewards = {i: 0.0 for i in self.possible_agents}
        pred_states = next_state.predator_states
        for prey_state in new_caught_prey:
            pred_dists = np.linalg.norm(prey_state[:2] - pred_states[:, :2], axis=1)
            involved_predators = np.where(pred_dists <= self.COLLISION_DIST)[0]
            predator_reward = self._per_prey_reward / len(involved_predators)
            for i in involved_predators:
                rewards[str(i)] += predator_reward

        return rewards


class PPWorld:
    """A continuous 2D world for the Predator-Prey Problem."""

    def __init__(
        self,
        world_size: int,
        blocks: Optional[List[Object]],
        predator_start_positions: Optional[List[Position]] = None,
        prey_start_positions: Optional[List[Position]] = None,
        predator_angles: Optional[List[float]] = None,
    ):
        assert world_size >= 3
        self.size = world_size
        self.agent_radius = 0.5

        # world border lines (start coords, end coords)
        self.walls = (
            np.array(
                [[0, 0], [0, 0], [0, world_size], [world_size, 0]], dtype=np.float32
            ),
            np.array(
                [
                    [world_size, 0],
                    [0, world_size],
                    [world_size, world_size],
                    [world_size, world_size],
                ],
                dtype=np.float32,
            ),
        )

        if blocks is None:
            blocks = []
        self.blocks = blocks

        if predator_start_positions is None:
            predator_start_positions = []
            for col, row in product([0, world_size // 2, world_size - 1], repeat=2):
                if col not in (0, world_size - 1) and row not in (0, world_size - 1):
                    continue
                x, y = col + self.agent_radius, row + self.agent_radius
                invalid_pos = False
                for block_pos, block_size in self.blocks:
                    dist = self.euclidean_dist((x, y, 0.0), block_pos)
                    if dist <= self.agent_radius + block_size:
                        invalid_pos = True
                        break
                if not invalid_pos:
                    predator_start_positions.append((x, y, 0.0))

        self.predator_start_positions = predator_start_positions

        if prey_start_positions is None:
            # prey can start anywhere at least distance 2 * self.agent size away from
            # any predator (i.e. an agent wide gap from any predator)
            # Not very efficient, but only needs to be run once at the start
            prey_start_positions = []
            for col, row in product(range(world_size), range(world_size)):
                if col in (0, world_size - 1) or row in (0, world_size - 1):
                    continue
                x, y = col + self.agent_radius, row + self.agent_radius
                invalid_pos = False
                for pred_pos in self.predator_start_positions:
                    dist = self.euclidean_dist((x, y), pred_pos)
                    if dist < 2 * self.agent_radius:
                        invalid_pos = True
                        break
                if invalid_pos:
                    continue
                for block_pos, block_size in self.blocks:
                    dist = self.euclidean_dist((x, y), block_pos)
                    if dist < self.agent_radius + block_size:
                        invalid_pos = True
                        break
                if not invalid_pos:
                    prey_start_positions.append((x, y, 0.0))

        self.prey_start_positions = prey_start_positions

    @staticmethod
    def euclidean_dist(
        coord1: Union[Coord, Position, np.ndarray],
        coord2: Union[Coord, Position, np.ndarray],
    ) -> float:
        """Get Euclidean distance between two positions on the world."""
        return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

    def check_circle_line_intersection(
        self,
        circle_coord: np.ndarray,
        circle_radius: float,
        lines_start_coords: np.ndarray,
        lines_end_coords: np.ndarray,
    ) -> np.ndarray:
        """Check if lines intersect circle.

        `circle_coords` is the `[x, y]` coords of the center of the circle.
        `circle_radius` is the radius of the circle.
        `line_start_coords` are the `[x, y]` coords of the start of the lines, and
            should have shape `(n_lines, 2)`
        `line_start_coords` are the `[x, y]` coords of the end of the lines, and
            should have shape `(n_lines, 2)`

        Returns an array with an entry for each line. For each line the entry will be
        the distance to intersection for that line, if the line intersects the
        circle, otherwise returns np.nan.

        """
        # https://math.stackexchange.com/questions/275529/check-if-line-intersects-with-circles-perimeter
        # translate into circles reference frame
        starts = lines_start_coords - circle_coord
        ends = lines_end_coords - circle_coord
        v = ends - starts

        a = (v**2).sum(axis=1)
        b = 2 * (starts * v).sum(axis=1)
        c = (starts**2).sum(axis=1) - 0.5**2
        disc = b**2 - 4 * a * c
        with np.errstate(invalid="ignore"):
            # this will outpit np.nan for any negative discriminants, which is what we
            # want, but will throw a runtime warning which we want to ignore
            sqrtdisc = np.sqrt(disc)

        t1 = (-b - sqrtdisc) / (2 * a)
        t2 = (-b + sqrtdisc) / (2 * a)
        t1 = np.where(((t1 >= 0.0) & (t1 <= 1.0)), t1, np.nan)
        t2 = np.where(((t2 >= 0.0) & (t2 <= 1.0)), t2, np.nan)
        t = np.where(t1 <= t2, t1, t2)

        t = np.expand_dims(t, axis=-1)
        return np.sqrt(((t * v) ** 2).sum(axis=1))

    def check_line_line_intersection(
        self,
        l1_start_coords: np.ndarray,
        l1_end_coords: np.ndarray,
        l2_start_coords: np.ndarray,
        l2_end_coords: np.ndarray,
    ) -> np.ndarray:
        """Check if lines intersect.

        Checks for each line in `l1` if it intersects with any line in `l2`.

        Arguments
        ---------
        l1_start_coords: array with shape `(n_lines1, 2)` containing the (x, y) coord
          for the start of each of the first set of lines.
        l1_end_coords: array with shape `(n_lines1, 2)` containing the (x, y) coord
          for the end of each of the first set of lines.
        l2_start_coords: array with shape `(n_lines2, 2)` containing the (x, y) coord
          for the start of each of the second set of lines.
        l2_end_coords: array with shape `(n_lines2, 2)` containing the (x, y) coord
          for the end of each of the second set of lines.

        Returns
        -------
        intersection_coords: array with shape `(n_lines1, n_lines2, 2)` containing the
           (x, y) coords for the point of intersection between each pair of line
           segments from `l1` and `l2`. If a pair of lines does not intersect, the
           x and y coordinate values will be `np.nan`.

        """
        # https://stackoverflow.com/questions/3252194/numpy-and-line-intersections?noredirect=1&lq=1
        # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
        dl1 = l1_end_coords - l1_start_coords  # (n_lines1, 2)
        dl2 = l2_end_coords - l2_start_coords  # (n_lines2, 2)

        # diff between each l1 with each l2 along x and y directions
        # (n_lines1, :, 2) - (:, n_lines2, 2) = (n_lines1, n_lines2, 2)
        dl1l2 = l1_start_coords[:, np.newaxis, :] - l2_start_coords[np.newaxis, :, :]
        # needed for matmul (n_lines1, 2, n_lines2)
        dl1l2_T = dl1l2.transpose(0, 2, 1)
        # needed for matmul (n_lines2, 2, n_lines1)
        dl1l2_T2 = dl1l2.transpose(1, 2, 0)

        # line perpendicular to each l1 line
        dl1p = np.empty_like(dl1)  # (n_lines1, 2)
        dl1p[:, 0] = -dl1[:, 1]
        dl1p[:, 1] = dl1[:, 0]

        # line perpendicular to each l2 line
        dl2p = np.empty_like(dl2)  # (n_lines1, 2)
        dl2p[:, 0] = -dl2[:, 1]
        dl2p[:, 1] = dl2[:, 0]

        # mult (n_lines1, 2) @ (n_lines1, 2, nlines2) = (n_lines1, n_lines2)
        # each i in n_lines1 is multiplied with one of the n_lines1 matrices in dl1l2
        # l1[i] @ (l1[i] - l2[j]) for i in [0, n_lines1], j in [0, n_lines2]
        u_num = np.stack([np.matmul(dl1p[i], dl1l2_T[i]) for i in range(dl1p.shape[0])])

        # mult (n_lines2, 2) @ (n_lines2, 2, nlines1) = (n_lines2, n_lines1)
        # same as above except for l2 lines
        t_num = np.stack(
            [np.matmul(dl2p[j], dl1l2_T2[j]) for j in range(dl2p.shape[0])]
        )

        # mult (n_lines1, 2) @ (2, n_lines2) = (n_lines1, n_lines2)
        # get l1[i] dot l2[j] for i in [0, n_lines1], j in [0, n_lines2]
        # but using perpendicular lines to l1,
        denom = np.matmul(dl1p, dl2.transpose())
        # handle case where lines are parallel/colinear, leading to denom being zero
        denom = np.where(np.isclose(denom, 0.0), np.nan, denom)

        # from wiki, lines only intersect if 0 <= u <= 1
        u = u_num / denom  # (n_lines1, n_lines2)
        t = t_num.transpose() / denom  # (n_lines1, n_lines2)

        # segments intersect when 0 <= u <= 1 and 0 <= t <= 1
        u = np.where(((u >= 0) & (u <= 1) & (t >= 0) & (t <= 1)), u, np.nan)
        u = u[:, :, np.newaxis]  # (n_lines1, n_lines2, 1)

        return u * dl2 + l2_start_coords

    def check_wall_line_intersection(
        self, line_start_coords: np.ndarray, line_end_coords: np.ndarray
    ) -> np.ndarray:
        """Check if lines intersect world boundary wall.

        Arguments
        ---------
        line_start_coords: the `(x, y)` coords of the start of the lines, and
            should have shape `(n_lines, 2)`
        line_end_coords: are the `(x, y)` coords of the end of the lines, and
            should have shape `(n_lines, 2)`

        Returns
        -------
        intersect_coords: an array with an entry for each line. For each line the entry
            will be the `(x, y)` coord of the point of intersection for that line with
            the wall, if the line intersects the boundary, otherwise will be
            `(np.nan, np.nan)`. The array will have shape `(n_lines, 2)`

        """
        # TODO
        # shape = (n_lines, walls, 2)
        wall_intersect_coords = self.check_line_line_intersection(
            line_start_coords, line_end_coords, *self.walls
        )
        # Need to get coords of intersected walls
        return wall_intersect_coords

    def check_collision_ray(
        self,
        origin: Position,
        line_distance: float,
        n_lines: int,
        other_agents: Optional[np.ndarray],
        include_blocks: bool = True,
        check_walls: bool = True,
        use_relative_angle: bool = True,
    ) -> np.ndarray:
        """Check for collision along ray.

        Returns entity index and distance if there is a collision, otherwise if there is
        no collision returns None and `line distance`.

        If collision is with a wall or block, return index of -1.

        If `use_relative_angle=True` then line angle is treated as relative to agents
        yaw angle. Otherwise line angle is treated as absolute (i.e. relative to angle
        of 0).

        """
        x, y, rel_angle = origin
        if not use_relative_angle:
            rel_angle = 0.0

        angles = np.linspace(
            0.0, 2 * math.pi, n_lines, endpoint=False, dtype=np.float32
        )
        ray_end_xs = x + line_distance * np.cos(angles + rel_angle)
        ray_end_ys = y + line_distance * np.sin(angles + rel_angle)

        ray_start_coords = np.tile((x, y), (n_lines, 1))
        ray_end_coords = np.stack([ray_end_xs, ray_end_ys], axis=1)

        closest_distances = np.full_like(angles, line_distance)

        if other_agents is not None:
            for i, coord in enumerate(other_agents):
                dists = self.check_circle_line_intersection(
                    coord, self.agent_radius, ray_start_coords, ray_end_coords
                )
                # use fmin to ignore NaNs
                np.fmin(closest_distances, dists, out=closest_distances)

        if include_blocks:
            for index, (pos, size) in enumerate(self.blocks):
                dists = self.check_circle_line_intersection(
                    np.array([pos[0], pos[1]]), size, ray_start_coords, ray_end_coords
                )
                np.fmin(closest_distances, dists, out=closest_distances)

        if check_walls:
            # shape = (n_lines, walls, 2)
            wall_intersect_coords = self.check_line_line_intersection(
                ray_start_coords, ray_end_coords, *self.walls
            )
            # Need to get coords of intersected walls, each ray can intersect a max of
            # of 1 wall, so we just find the minimum non nan coords
            # shape = (n_lines, 2)
            with warnings.catch_warnings():
                # if no wall intersected, we take min of all NaN which throws a warning
                # but this is acceptable behevaiour, so we suppress the warning
                warnings.simplefilter("ignore")
                wall_intersect_coords = np.nanmin(wall_intersect_coords, axis=1)
            dists = np.sqrt(
                ((wall_intersect_coords - ray_start_coords) ** 2).sum(axis=1)
            )
            np.fmin(closest_distances, dists, out=closest_distances)

        return closest_distances


def parse_world_str(world_str: str) -> PPWorld:
    """Parse a str representation of a world into a world object.

    Notes on world str representation:

    . = empty/unallocated cell
    # = a block
    P = starting location for predator agents [optional] (defaults to edges)
    p = starting location for prey agent [optional] (defaults to center)

    Examples (" " quotes and newline chars omitted):

    1. A 10x10 world with 4 groups of blocks and using the default predator
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
    row_strs = world_str.splitlines()
    assert len(row_strs) > 1
    assert all(len(row) == len(row_strs[0]) for row in row_strs)
    assert len(row_strs[0]) > 1
    assert len(row_strs) == len(row_strs[0])

    world_size = len(row_strs)
    blocks: Set[Object] = set()
    predator_coords = set()
    prey_coords = set()
    for r, c in product(range(world_size), repeat=2):
        # This is offset to the center of the square
        coord = (c + 0.5, r + 0.5, 0)
        char = row_strs[r][c]

        if char == "#":
            # Radius is 0.5
            blocks.add((coord, 0.5))
        elif char == "P":
            predator_coords.add(coord)
        elif char == "p":
            prey_coords.add(coord)
        else:
            assert char == "."

    return PPWorld(
        world_size,
        blocks=list(blocks),
        predator_start_positions=None
        if len(predator_coords) == 0
        else list(predator_coords),
        prey_start_positions=None if len(prey_coords) == 0 else list(prey_coords),
    )


def get_default_world(size: int, include_blocks: bool) -> PPWorld:
    """Get function for generaing default world with given size.

    If `include_blocks=True` then world will contain blocks with the following layout:

    .....
    .#.#.
    .....
    .#.#.
    .....

    Where `#` are the blocks, which will be represented as a single circle.
    """
    r = float(size / 10)
    if include_blocks:
        blocks = [
            ((x + r, y + r, 0.0), r)
            for x, y in product([size / 5, 3 * size / 5], repeat=2)
        ]
    else:
        blocks = []
    return PPWorld(world_size=size, blocks=blocks)


def get_5x5_world() -> PPWorld:
    """Generate 5x5 world layou`t."""
    return get_default_world(5, include_blocks=False)


def get_5x5_blocks_world() -> PPWorld:
    """Generate 5x5 Blocks world layout."""
    return get_default_world(5, include_blocks=True)


def get_10x10_world() -> PPWorld:
    """Generate 10x10 world layou`t."""
    return get_default_world(10, include_blocks=False)


def get_10x10_blocks_world() -> PPWorld:
    """Generate 10x10 Blocks world layout."""
    return get_default_world(10, include_blocks=True)


def get_15x15_world() -> PPWorld:
    """Generate 15x15 world layou`t."""
    return get_default_world(15, include_blocks=False)


def get_15x15_blocks_world() -> PPWorld:
    """Generate 15x15 Blocks world layout."""
    return get_default_world(15, include_blocks=True)


def get_20x20_world() -> PPWorld:
    """Generate 20x20 world layout."""
    return get_default_world(20, include_blocks=False)


def get_20x20_blocks_world() -> PPWorld:
    """Generate 20x20 Blocks world layout."""
    return get_default_world(20, include_blocks=True)


#  (world_make_fn, step_limit)
SUPPORTED_WORLDS = {
    "5x5": (get_5x5_world, 25),
    "5x5Blocks": (get_5x5_blocks_world, 50),
    "10x10": (get_10x10_world, 50),
    "10x10Blocks": (get_10x10_blocks_world, 50),
    "15x15": (get_15x15_world, 100),
    "15x15Blocks": (get_15x15_blocks_world, 100),
    "20x20": (get_20x20_world, 200),
    "20x20Blocks": (get_20x20_blocks_world, 200),
}
