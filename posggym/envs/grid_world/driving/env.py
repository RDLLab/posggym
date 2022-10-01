from typing import Optional, Tuple, Dict, Any

from posggym import core
import posggym.model as M

import posggym.envs.grid_world.core as grid_lib
import posggym.envs.grid_world.render as render_lib

import posggym.envs.grid_world.driving.model as dmodel
from posggym.envs.grid_world.driving.grid import DrivingGrid
from posggym.envs.grid_world.driving.gen import DrivingGridGenerator


class DrivingEnv(core.DefaultEnv):
    """The Driving Grid World Environment.

    A general-sum 2D grid world problem involving multiple agents. Each agent
    controls a vehicle and is tasked with driving the vehicle from it's start
    location to a destination location while avoiding crashing into obstacles
    or other vehicles.

    This environment requires each agent to navigate in the world while also
    taking care to avoid crashing into other players. The dynamics and
    observations of the environment are such that avoiding collisions requires
    some planning in order for the vehicle to brake in time or maintain a good
    speed. Depending on the grid layout, the environment will require agents to
    reason about and possibly coordinate with the other vehicles.


    Agents
    ------
    Varied number

    State
    -----
    Each state is made up of the state of each vehicle, which in turn is
    defined by:

    - the (x, y) (x=column, y=row, with origin at the top-left square of the
      grid) of the vehicle,
    - the direction the vehicle is facing NORTH=0, SOUTH=1, EAST=2, WEST=3),
    - the speed of the vehicle (REVERSE=-1, STOPPED=0, FORWARD_SLOW=1,
      FORWARD_FAST=2),
    - the (x, y) of the vehicles destination
    - whether the vehicle has reached it's destination or not
    - whether the vehicle has crashed or not

    Actions
    -------
    Each agent has 5 actions: DO_NOTHING=0, ACCELERATE=1, DECELERATE=2,
    TURN_RIGHT=3, TURN_LEFT=4

    Observation
    -----------
    Each agent observes their current speed along with the cells in their local
    area. The size of the local area observed is controlled by the `obs_dims`
    parameter. For each cell in the observed are the agent observes whether
    they are one of four things: VEHICLE=0, WALL=1, EMPTY=2, DESTINATION=3.
    Each agent also observes the (x, y) coordinates of their destination,
    whether they have reached the destination, and whether they have crashed.

    Each observation is represented as a tuple:
        ((local obs), speed, destination coord, destination reached, crashed)

    Reward
    ------
    All agents receive a penalty of 0.00 for each step. They also recieve a
    penalty of -0.5 for hitting an obstacle (if ``obstacle_collision=True``),
    and -1.0 for hitting another vehicle. A reward of 1.0 is given if the agent
    reaches it's destination.

    Transition Dynamics
    -------------------
    Actions are deterministic and movement is determined by direction the
    vehicle is facing and it's speed:

    - Speed=-1 (REVERSE) - vehicle moves one cell in the opposite direction
    - Speed=0 (STOPPED) - vehicle remains in same cell
    - Speed=1 (FORWARD_SLOW) - vehicle move one cell in facing direction
    - Speed=1 (FORWARD_FAST) - vehicle moves two cells in facing direction

    Accelerating increases speed by 1, while deceleration decreased speed by 1.
    If the vehicle will hit a wall or an other vehicle when moving from one
    cell to another then it remains in it's current cell and its crashed state
    variable is updated. Once a vehicle reaches it's destination it is stuck.

    Episodes ends when all agents have either reached their destination or
    crashed, or the episode step limit is reached.

    """

    metadata = {"render.modes": ['human', 'ansi', 'rgb_array']}

    def __init__(self,
                 grid: DrivingGrid,
                 num_agents: int,
                 obs_dim: Tuple[int, int, int],
                 obstacle_collisions: bool,
                 **kwargs):
        self._model = dmodel.DrivingModel(
            grid,
            num_agents,
            obs_dim,
            obstacle_collisions,
            **kwargs
        )
        self._obs_dim = obs_dim
        self._viewer = None
        self._renderer: Optional[render_lib.GWRenderer] = None
        super().__init__()

    def render(self, mode: str = "human"):
        if mode == "ansi":
            grid_str = self._model.grid.get_ascii_repr(
                [vs.coord for vs in self._state],
                [vs.facing_dir for vs in self._state],
                [vs.dest_coord for vs in self._state]
            )

            output = [
                f"Step: {self._step_num}",
                grid_str,
            ]
            if self._last_actions is not None:
                action_str = ", ".join(
                    [dmodel.ACTIONS_STR[a] for a in self._last_actions]
                )
                output.insert(1, f"Actions: <{action_str}>")
                output.append(f"Rewards: <{self._last_rewards}>")

            return "\n".join(output) + "\n"
        elif mode in ("human", "rgb_array"):
            grid = self.model.grid
            if mode == "human" and self._viewer is None:
                # pylint: disable=[import-outside-toplevel]
                from posggym.envs.grid_world import viewer
                self._viewer = viewer.GWViewer(   # type: ignore
                    "Driving Env",
                    (min(grid.width, 9), min(grid.height, 9)),
                    num_agent_displays=self.n_agents
                )
                self._viewer.show(block=False)   # type: ignore

            if self._renderer is None:
                self._renderer = render_lib.GWRenderer(
                    self.n_agents, grid, [], render_blocks=True
                )

            agent_obs_coords = tuple(
                self._model.get_obs_coords(vs.coord, vs.facing_dir)
                for vs in self._state
            )
            agent_coords = tuple(vs.coord for vs in self._state)
            agent_dirs = tuple(vs.facing_dir for vs in self._state)

            # Add agent destination locations
            other_objs = [
                render_lib.GWObject(
                    vs.dest_coord,
                    render_lib.get_agent_color(i),
                    render_lib.Shape.RECTANGLE,
                    # make dest squares slightly different to vehicle color
                    alpha=0.2
                )
                for i, vs in enumerate(self._state)
            ]
            # Add visualization for crashed agents
            for i, vs in enumerate(self._state):
                if vs.crashed:
                    other_objs.append(
                        render_lib.GWObject(
                            vs.coord,
                            'yellow',
                            render_lib.Shape.CIRCLE
                        )
                    )

            env_img = self._renderer.render(
                agent_coords,
                agent_obs_coords,
                agent_dirs=agent_dirs,
                other_objs=other_objs,
                agent_colors=None
            )
            agent_obs_imgs = self._renderer.render_all_agent_obs(
                env_img,
                agent_coords,
                agent_dirs,
                agent_obs_dims=self._obs_dim,
                out_of_bounds_obj=render_lib.GWObject(
                    (0, 0), 'grey', render_lib.Shape.RECTANGLE
                )
            )

            if mode == "human":
                self._viewer.display_img(        # type: ignore
                    env_img, agent_idx=None
                )
                for i, obs_img in enumerate(agent_obs_imgs):
                    self._viewer.display_img(    # type: ignore
                        obs_img, agent_idx=i
                    )
            else:
                return (env_img, agent_obs_imgs)
        else:
            super().render(mode)

    @property
    def model(self) -> dmodel.DrivingModel:
        return self._model

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()   # type: ignore
            self._viewer = None


class DrivingGenEnv(DrivingEnv):
    """The Generated Driving Grid World Environment.

    This is the same as the Driving Environment except that a new grid is
    generated at each reset.

    The generated Grids can be:

    1. set to be from some list of grids, in which case the grids in the list
       will be cycled through (in the same order or shuffled)
    2. generated a new each reset. Depending on grid size and generator
       parameters this will lead to possibly unique grids each episode.

    The seed parameter can be used to ensure the same grids are used on
    different runs.

    """

    def __init__(self,
                 num_agents: int,
                 obs_dim: Tuple[int, int, int],
                 obstacle_collisions: bool,
                 n_grids: Optional[int],
                 generator_params: Dict[str, Any],
                 shuffle_grid_order: bool = True,
                 seed: Optional[int] = None,
                 **kwargs):
        if "seed" not in generator_params:
            generator_params["seed"] = seed

        self._n_grids = n_grids
        self._gen_params = generator_params
        self._shuffle_grid_order = shuffle_grid_order
        self._gen = DrivingGridGenerator(**generator_params)

        if n_grids is not None:
            grids = self._gen.generate_n(n_grids)
            self._cycler = grid_lib.GridCycler(
                grids, shuffle_grid_order, seed=seed
            )
            grid = grids[0]
        else:
            self._cycler = None     # type: ignore
            grid = self._gen.generate()

        self._model_kwargs = {
            "num_agents": num_agents,
            "obs_dim": obs_dim,
            "obstacle_collisions": obstacle_collisions,
            **kwargs
        }

        super().__init__(
            grid,                   # type: ignore
            num_agents,
            obs_dim,
            obstacle_collisions=obstacle_collisions,
            seed=seed,
            **kwargs
        )

    def reset(self,
              *,
              seed: Optional[int] = None) -> Optional[M.JointObservation]:
        if seed is not None:
            self._gen_params["seed"] = seed
            self._gen = DrivingGridGenerator(**self._gen_params)

            if self._n_grids is not None:
                grids = self._gen.generate_n(self._n_grids)
                self._cycler = grid_lib.GridCycler(
                    grids, self._shuffle_grid_order, seed=seed
                )

        if self._n_grids:
            grid = self._cycler.next()
        else:
            grid = self._gen.generate()

        self._model = dmodel.DrivingModel(
            grid,                   # type: ignore
            **self._model_kwargs    # type: ignore
        )

        return super().reset(seed=seed)

    def render(self, mode: str = "human"):
        if mode in ("human", "rgb_array"):
            grid = self.model.grid
            if mode == "human" and self._viewer is None:
                # pylint: disable=[import-outside-toplevel]
                from posggym.envs.grid_world import viewer
                self._viewer = viewer.GWViewer(   # type: ignore
                    "Driving Env",
                    (min(grid.width, 9), min(grid.height, 9)),
                    num_agent_displays=self.n_agents
                )
                self._viewer.show(block=False)   # type: ignore

            if self._renderer is None:
                # Need to re-add blocks each time since these change by
                # episode
                self._renderer = render_lib.GWRenderer(
                    self.n_agents, grid, [], render_blocks=False
                )

            agent_obs_coords = tuple(
                self._model.get_obs_coords(vs.coord, vs.facing_dir)
                for vs in self._state
            )

            agent_coords = tuple(vs.coord for vs in self._state)
            agent_dirs = tuple(vs.facing_dir for vs in self._state)

            # Add agent destination locations
            other_objs = [
                render_lib.GWObject(
                    vs.dest_coord,
                    render_lib.get_agent_color(i),
                    render_lib.Shape.RECTANGLE,
                    # make dest squares slightly different to vehicle color
                    alpha=0.2
                )
                for i, vs in enumerate(self._state)
            ]
            # Add blocks
            for block_coord in grid.block_coords:
                other_objs.append(
                    render_lib.GWObject(
                        block_coord, 'grey', render_lib.Shape.RECTANGLE
                    )
                )
            # Add visualization for crashed agents
            for i, vs in enumerate(self._state):
                if vs.crashed:
                    other_objs.append(
                        render_lib.GWObject(
                            vs.coord, 'yellow', render_lib.Shape.CIRCLE
                        )
                    )

            env_img = self._renderer.render(
                agent_coords,
                agent_obs_coords,
                agent_dirs=agent_dirs,
                other_objs=other_objs,
                agent_colors=None
            )
            agent_obs_imgs = self._renderer.render_all_agent_obs(
                env_img,
                agent_coords,
                agent_dirs,
                agent_obs_dims=self._obs_dim,
                out_of_bounds_obj=render_lib.GWObject(
                    (0, 0), 'grey', render_lib.Shape.RECTANGLE
                )
            )

            if mode == "human":
                self._viewer.display_img(             # type: ignore
                    env_img, agent_idx=None
                )
                for i, obs_img in enumerate(agent_obs_imgs):
                    self._viewer.display_img(         # type: ignore
                        obs_img, agent_idx=i
                    )
            else:
                return (env_img, agent_obs_imgs)
        else:
            super().render(mode)
