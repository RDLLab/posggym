from typing import Optional, Tuple, Union

from posggym import core

from posggym.envs.grid_world.core import Direction
import posggym.envs.grid_world.render as render_lib
import posggym.envs.grid_world.two_paths.model as tp_model


class TwoPathsEnv(core.DefaultEnv):
    """The Two-Paths Grid World Environment.

    An adversarial 2D grid world problem involving two agents, a runner and
    a chaser. The runner's goal is to reach one of two goal location, with each
    goal located at the end of a seperate path. The lengths of the two paths
    have different lengths. The goal of the chaser is to intercept the runner
    before it reaches a goal. The runner is considered caught if it is observed
    by the chaser, or occupies the same location. However, the chaser is only
    able to effectively cover one of the two goal locations.

    This environment requires each agent to reason about the which path the
    other agent will choose. It offers an ideal testbed for planning under
    finite-nested reasoning assumptions since it is possible to map reasoning
    level to the expected path choice.

    The two agents start at opposite ends of the maps.

    Agents
    ------
    Runner=0
    Chaser=1

    State
    -----
    Each state contains the (x, y) (x=column, y=row, with origin at the
    top-left square of the grid) of the runner and chaser agent. Specifically,
    a states is ((x_runner, y_runner), (x_chaser, y_chaser))

    Actions
    -------
    Each agent has 4 actions corresponding to moving in the 4 cardinal
    directions (NORTH=0, EAST=1, SOUTH=2, WEST=3).

    Observation
    -----------
    Each agent observes the adjacent cells in the four cardinal directions and
    whether they are one of three things: OPPONENT=0, WALL=1, EMPTY=2.
    Each agent also observes whether a terminal state was reach (0/1). This is
    necessary for the infinite horizon model of the environment.

    Each observation is represented as a tuple:
        ((cell_north, cell_south, cell_east, cell_west), terminal)

    Reward
    ------
    Both agents receive a penalty of -0.01 for each step.
    If the runner reaches the goal then the runner recieves a reward of 1.0,
    while the chaser recieves a penalty of -1.0.
    If the runner is observed by the chaser, then the runner recieves a penalty
    of -1.0, while the chaser recieves a reward of 1.0.

    The rewards make the environment adversarial, but not strictly zero-sum,
    due to the small penalty each step.

    Transition Dynamics
    -------------------
    By default actions are deterministic and an episode ends when either the
    runner is caught, the runner reaches a goal, or the step limit is reached.

    The environment can also be run in stochastic mode by changing the
    action_probs parameter at initialization. This controls the probability
    the agent will move in the desired direction each step, otherwise moving
    randomly in one of the other 3 possible directions.

    Lastly, if using infinite_horizon mode then the environment resets to the
    start state once a terminal state is reached.
    """

    metadata = {"render.modes": ['human', 'ansi', 'rgb_array']}

    def __init__(self,
                 grid_name: str,
                 action_probs: Union[float, Tuple[float, float]] = 1.0,
                 infinite_horizon: bool = False,
                 **kwargs):
        self._model = tp_model.TwoPathsModel(
            grid_name, action_probs, infinite_horizon, **kwargs
        )
        self._viewer = None
        self._renderer: Optional[render_lib.GWRenderer] = None
        super().__init__()

    def render(self, mode: str = "human"):
        if mode == "ansi":
            grid_str = self._model.grid.get_ascii_repr(
                self._state[0], self._state[1]
            )

            output = [
                f"Step: {self._step_num}",
                grid_str,
            ]
            if self._last_actions is not None:
                action_str = ", ".join(
                    [str(Direction(a)) for a in self._last_actions]
                )
                output.insert(1, f"Actions: <{action_str}>")
                output.append(f"Rewards: <{self._last_rewards}>")

            return "\n".join(output) + "\n"
        elif mode in ("human", 'rgb_array'):
            grid = self.model.grid
            if mode == 'human' and self._viewer is None:
                # pylint: disable=[import-outside-toplevel]
                from posggym.envs.grid_world import viewer
                self._viewer = viewer.GWViewer(   # type: ignore
                    "Two-Paths Env",
                    (min(grid.width, 9), min(grid.height, 9)),
                    num_agent_displays=self.n_agents
                )
                self._viewer.show(block=False)   # type: ignore

            if self._renderer is None:
                static_objs = [
                    render_lib.GWObject(
                        coord, 'green', render_lib.Shape.RECTANGLE
                    ) for coord in grid.goal_coords
                ]
                self._renderer = render_lib.GWRenderer(
                    self.n_agents,
                    grid,
                    static_objs,
                    render_blocks=True
                )

            agent_coords = self._state
            agent_obs_coords = tuple(
                grid.get_neighbours(self._state[i], True)
                for i in range(self.n_agents)
            )
            for i, coord in enumerate(agent_coords):
                agent_obs_coords[i].append(coord)
            agent_dirs = tuple(Direction.NORTH for _ in range(self.n_agents))

            env_img = self._renderer.render(
                self._state,
                agent_obs_coords,
                agent_dirs,
                other_objs=None,
                agent_colors=None
            )
            agent_obs_imgs = self._renderer.render_all_agent_obs(
                env_img,
                agent_coords,
                agent_dirs,
                agent_obs_dims=(1, 1, 1),
                out_of_bounds_obj=render_lib.GWObject(
                    (0, 0), 'grey', render_lib.Shape.RECTANGLE
                ),
                agent_obs_coords=agent_obs_coords,
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
    def model(self) -> tp_model.TwoPathsModel:
        return self._model

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()   # type: ignore
            self._viewer = None
