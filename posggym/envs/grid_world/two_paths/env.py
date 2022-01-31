"""Environment class for the Two-Paths Grid World Problem """
import sys
from typing import Optional, Tuple, Union


from posggym import core
import posggym.model as M

from posggym.envs.grid_world.utils import Direction
import posggym.envs.grid_world.render as render_lib
import posggym.envs.grid_world.two_paths.model as tp_model


class TwoPathsEnv(core.Env):
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
    directions (NORTH=0, SOUTH=1, EAST=2, WEST=3).

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
    Both agents receive a penalty of -1.0 for each step.
    If the runner reaches the goal then the runner recieves a reward of 100,
    while the chaser recieves a penalty of -100.
    If the runner is observed by the chaser, then the runner recieves a penalty
    of -100, while the chaser recieves a penalty of 100.

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

    metadata = {"render.modes": ['human', 'ascii']}

    def __init__(self,
                 grid_name: str,
                 action_probs: Union[float, Tuple[float, float]] = 1.0,
                 infinite_horizon: bool = False,
                 **kwargs):
        self._model = tp_model.TwoPathsModel(
            grid_name, action_probs, infinite_horizon, **kwargs
        )

        init_conds = self._model.sample_initial_state_and_obs()
        self._state, self._last_obs = init_conds
        self._step_num = 0
        self._last_actions: Optional[tp_model.TPJointAction] = None
        self._last_rewards: Optional[M.JointReward] = None

        self._viewer = None
        self._renderer: Optional[render_lib.GWRenderer] = None

    def step(self,
             actions: M.JointAction
             ) -> Tuple[M.JointObservation, M.JointReward, bool, dict]:
        step = self._model.step(self._state, actions)
        self._step_num += 1
        self._state = step.state
        self._last_obs = step.observations
        self._last_actions = actions
        self._last_rewards = step.rewards
        aux = {"outcomes": step.outcomes}
        return (step.observations, step.rewards, step.done, aux)

    def reset(self, seed: Optional[int] = None) -> M.JointObservation:
        init_conds = self._model.sample_initial_state_and_obs()
        self._state, self._last_obs = init_conds
        self._last_actions = None
        self._last_rewards = None
        self._step_num = 0
        return self._last_obs

    def render(self, mode: str = "human") -> None:
        if mode == "ascii":
            outfile = sys.stdout

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

            outfile.write("\n".join(output) + "\n")
        elif mode == "human":
            grid = self.model.grid
            if self._viewer is None:
                # pylint: disable=[import-outside-toplevel]
                from posggym.envs.grid_world import viewer
                self._viewer = viewer.GWViewer(   # type: ignore
                    "Two-Paths Env", (grid.width, grid.height)
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

            agent_obs_coords = tuple(
                grid.get_neighbours(self._state[i], True)
                for i in range(self.n_agents)
            )

            img = self._renderer.render(
                self._state,
                agent_obs_coords,
                agent_dirs=None,
                other_objs=None,
                agent_colors=None
            )
            self._viewer.display_img(img)  # type: ignore

    @property
    def model(self) -> tp_model.TwoPathsModel:
        return self._model