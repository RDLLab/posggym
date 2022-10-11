from typing import Optional

from posggym import core

import posggym.envs.grid_world.core as grid_lib
import posggym.envs.grid_world.render as render_lib

from posggym.envs.grid_world.predator_prey.grid import PPGrid
import posggym.envs.grid_world.predator_prey.model as ppmodel


class PPEnv(core.DefaultEnv):
    """The Predator-Prey Grid World Environment.

    A co-operative 2D grid world problem involving multiple predator agents
    working together to catch prey agent/s in the environment.

    Agents
    ------
    Varied number

    State
    -----
    Each state consists of:

    1. tuple of the (x, y) position of all predators
    2. tuple of the (x, y) position of all preys
    3. tuple of whether each prey has been caught or not (0=no, 1=yes)

    For the coordinate x=column, y=row, with the origin (0, 0) at the
    top-left square of the grid.

    Actions
    -------
    Each agent has 5 actions: DO_NOTHING=0, UP=1, DOWN=2, LEFT=3, RIGHT=4

    Observation
    -----------
    Each agent observes the contents of local cells. The size of the
    local area observed is controlled by the `obs_dims` parameter. For each
    cell in the observed are the agent observes whether they are one of four
    things: EMPTY=0, WALL=1, PREDATOR=2, PREY=3.

    Reward
    ------
    There are two modes of play:

    1. Fully cooperative: All predators share a reward and each agent recieves
    a reward of 1.0 / `num_prey` for each prey capture, independent of which
    predator agent/s were responsible for the capture.

    2. Mixed cooperative: Predators only recieve a reward if they were part
    of the prey capture, recieving 1.0 / `num_prey`.

    In both modes prey can only been captured when at least `prey_strength`
    predators are in adjacent cells,
    where 1 <= `prey_strength` <= `num_predators`.

    Transition Dynamics
    -------------------
    Actions of the predator agents are deterministic and consist of moving in
    to the adjacent cell in each of the four cardinal directions. If two or
    more predators attempt to move into the same cell then no agent moves.

    Prey move according to the following rules (in order of priority):

    1. if predator is within `obs_dim` cells, moves away from closest predator
    2. if another prey is within `obs_dim` cells, moves away from closest prey
    3. else move randomly

    Prey always move first and predators and prey cannot occupy the same cell.
    The only exception being if a prey has been caught their final coord is
    recorded in the state but predator and prey agents will be able to move
    into the final coord.

    Episodes ends when all prey have been captured or the episode step limit is
    reached.

    Initial Conditions
    ------------------
    Predators start from random seperate locations along the edge of the grid
    (either in a corner, or half-way along a side), while prey start together
    in the middle.

    """

    metadata = {"render.modes": ['human', 'ansi', 'rgb_array']}

    def __init__(self,
                 grid: PPGrid,
                 num_predators: int,
                 num_prey: int,
                 cooperative: bool,
                 prey_strength: int,
                 obs_dim: int,
                 **kwargs):
        self._model = ppmodel.PPModel(
            grid,
            num_predators,
            num_prey,
            cooperative,
            prey_strength,
            obs_dim,
            **kwargs
        )
        self._obs_dim = obs_dim
        self._viewer = None
        self._renderer: Optional[render_lib.GWRenderer] = None
        super().__init__()

    def render(self, mode: str = "human"):
        if mode == "ansi":
            uncaught_prey_coords = [
                self._state.prey_coords[i] for i in range(self._model.num_prey)
                if not self._state.prey_caught[i]
            ]
            grid_str = self._model.grid.get_ascii_repr(
                self._state.predator_coords, uncaught_prey_coords
            )
            output = [
                f"Step: {self._step_num}",
                grid_str,
            ]
            if self._last_actions is not None:
                action_str = ", ".join(
                    [ppmodel.ACTIONS_STR[a] for a in self._last_actions]
                )
                output.insert(1, f"Actions: <{action_str}>")
                output.append(f"Rewards: <{self._last_rewards}>")

            return "\n".join(output) + "\n"
        elif mode in ("human", "rgb_array"):
            grid = self._model.grid
            if mode == "human" and self._viewer is None:
                # pylint: disable=[import-outside-toplevel]
                from posggym.envs.grid_world import viewer
                self._viewer = viewer.GWViewer(   # type: ignore
                    "Predator-Prey Env",
                    (min(grid.width, 9), min(grid.height, 9)),
                    num_agent_displays=self.n_agents
                )
                self._viewer.show(block=False)   # type: ignore

            if self._renderer is None:
                self._renderer = render_lib.GWRenderer(
                    self.n_agents, grid, [], render_blocks=True
                )

            agent_obs_coords = tuple(
                self._model.get_obs_coords(c)
                for c in self._state.predator_coords
            )
            agent_coords = self._state.predator_coords

            prey_objs = [
                render_lib.GWObject(
                    c,
                    'cyan',
                    render_lib.Shape.CIRCLE,
                    # alpha=0.25
                )
                for i, c in enumerate(self._state.prey_coords)
                if not self._state.prey_caught[i]
            ]

            env_img = self._renderer.render(
                agent_coords,
                agent_obs_coords,
                agent_dirs=None,
                other_objs=prey_objs,
                agent_colors=None
            )
            agent_obs_imgs = self._renderer.render_all_agent_obs(
                env_img,
                agent_coords,
                grid_lib.Direction.NORTH,
                agent_obs_dims=(self._obs_dim, self._obs_dim, self._obs_dim),
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
    def model(self) -> ppmodel.PPModel:
        return self._model

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()   # type: ignore
            self._viewer = None
