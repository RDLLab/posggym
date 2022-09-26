from typing import Optional, Tuple, Union

from posggym import core
import posggym.model as M

from posggym.envs.grid_world.core import Direction
import posggym.envs.grid_world.render as render_lib
import posggym.envs.grid_world.pursuit_evasion.model as pe_model


class PursuitEvasionEnv(core.DefaultEnv):
    """The Pursuit-Evasion Grid World Environment.

    An adversarial 2D grid world problem involving two agents, a evader and
    a pursuer. The evader's goal is to reach a goal location, on the other side
    of the grid, while the goal of the pursuer is to spot the evader
    before it reaches it's goal. The evader is considered caught if it is
    observed by the pursuer, or occupies the same location. The evader and
    pursuer have knowledge of each others starting locations. However, only the
    evader has knowledge of it's goal location. The pursuer only knowns that
    the goal location is somewhere on the opposite side of the grid to the
    evaders start location.

    This environment requires each agent to reason about the which path the
    other agent will take through the dense grid environment.

    Agents
    ------
    Evader=0
    Pursuer=1

    State
    -----
    Each state is made up of:

    0. the (x, y) coordinate of the evader
    1. the direction the evader is facing
    2. the (x, y) coordinate of the pursuer
    3. the direction the pursuer is facing
    4. the (x, y) coordinate of the evader
    5. the (x, y) coordinate of the evader's start location
    6. the (x, y) coordinate of the pursuer's start location

    Actions
    -------
    Each agent has 4 actions corresponding to moving in the 4 available
    directions w.r.t the direction the agent is currently faction
    (FORWARD=0, BACKWARDS=1, LEFT=2, RIGHT=3).

    Observation
    -----------
    Each agent observes:

    1. whether there is a wall or not in the adjacent cells in the four
       cardinal directions,
    2. whether they see the other agent in a cone in front of them. The cone
       projects forward up to 'max_obs_distance' (default=12) cells in front of
       the agent.
    3. whether they hear the other agent (whether the other agent is within
       distance 2 from the agent in any direction),
    4. the (x, y) coordinate of the evader's start location,
    5. the (x, y) coordinate of the pursuer's start location,
    6. Evader: the (x, y) coordinate of the evader's goal location.
       Pursuer: blank coordinate = (0, 0).

    Note, the goal and start coordinate observations do not change during a
    single episode, but they do change between episodes.

    Reward
    ------
    The environment is zero-sum with the pursuer recieving the negative of the
    evader reward. Additionally, rewards are by default normalized so that
    returns are bounded between -1 and 1 (this can be disabled by the
    `normalize_reward` parameter).

    The evader recieves a reward of 1 for reaching it's goal location and a
    reward of -1 if it gets captured. Additionally, the evader recieves a small
    reward of 0.01 each time it's minumum distance to it's goal along the
    shortest path decreases. This is to make it so the environment is no
    longer sparesely rewarded and to help with exploration and learning (it
    can be disabled by the `use_progress_reward` parameter.)

    Transition Dynamics
    -------------------
    By default actions are deterministic and an episode ends when either the
    evader is caught, the evader reaches a goal, or the step limit is reached.

    The environment can also be run in stochastic mode by changing the
    `action_probs` parameter at initialization. This controls the probability
    the agent will move in the desired direction each step, otherwise moving
    randomly in one of the other 3 possible directions.

    References
    ----------
    This Pursuit-Evasion implementation is directly inspired by the problem
    presented in the paper:
    - Seaman, Iris Rubi, Jan-Willem van de Meent, and David Wingate. 2018.
      “Nested Reasoning About Autonomous Agents Using Probabilistic Programs.”
      ArXiv Preprint ArXiv:1812.01569.

    """

    metadata = {"render.modes": ['human', 'rgb_array', 'ansi']}

    def __init__(self,
                 grid_name: str,
                 action_probs: Union[float, Tuple[float, float]] = 1.0,
                 max_obs_distance: int = 12,
                 normalize_reward: bool = True,
                 use_progress_reward: bool = True,
                 **kwargs):
        self._model = pe_model.PursuitEvasionModel(
            grid_name,
            action_probs=action_probs,
            max_obs_distance=max_obs_distance,
            normalize_reward=normalize_reward,
            use_progress_reward=use_progress_reward,
            **kwargs
        )

        self._max_obs_distance = max_obs_distance
        grid = self._model.grid
        fov_width = grid.get_max_fov_width(
            self._model.FOV_EXPANSION_INCR, max_obs_distance
        )

        self._obs_dims = (
            min(max_obs_distance, max(grid.width, grid.height)),
            0,
            fov_width // 2
        )
        self._viewer = None
        self._renderer: Optional[render_lib.GWRenderer] = None

        super().__init__()

    def reset(self,
              *,
              seed: Optional[int] = None) -> Optional[M.JointObservation]:
        # reset renderer since goal location can change between episodes
        self._renderer = None
        return super().reset(seed=seed)

    def render(self, mode: str = "human"):
        evader_coord = self._state[0]
        pursuer_coord = self._state[2]
        goal_coord = self._state[6]

        if mode == "ansi":
            grid_str = self._model.grid.get_ascii_repr(
                goal_coord, evader_coord, pursuer_coord
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
        elif mode in ("human", "rgb_array"):
            grid = self.model.grid
            if mode == "human" and self._viewer is None:
                # pylint: disable=[import-outside-toplevel]
                from posggym.envs.grid_world import viewer
                self._viewer = viewer.GWViewer(   # type: ignore
                    "Pursuit-Evasion Env",
                    (min(grid.height, 8), min(grid.width, 8)),
                    num_agent_displays=self.n_agents
                )
                self._viewer.show(block=False)   # type: ignore

            if self._renderer is None:
                self._renderer = render_lib.GWRenderer(
                    self.n_agents, grid, [], render_blocks=True
                )

            agent_obs_coords = tuple(
                list(grid.get_fov(
                    self._state[2*i],
                    self._state[2*i + 1],
                    self.model.FOV_EXPANSION_INCR,
                    self._max_obs_distance
                )) for i in range(self.n_agents)
            )
            agent_coords = (evader_coord, pursuer_coord)
            agent_dirs = (self._state[1], self._state[3])

            other_objs = [
                render_lib.GWObject(
                    goal_coord, 'green', render_lib.Shape.RECTANGLE
                )
            ]

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
                agent_obs_dims=self._obs_dims,
                out_of_bounds_obj=render_lib.GWObject(
                    (0, 0), 'grey', render_lib.Shape.RECTANGLE
                ),
                agent_obs_coords=agent_obs_coords
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
            return super().render(mode)

    @property
    def model(self) -> pe_model.PursuitEvasionModel:
        return self._model

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()   # type: ignore
            self._viewer = None
