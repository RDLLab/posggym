"""Environment class for the Two-Paths Grid World Problem """
import sys
from typing import Optional, Tuple, Union

from posggym import core
import posggym.model as M

from posggym.envs.grid_world.utils import Direction
import posggym.envs.grid_world.render as render_lib
import posggym.envs.grid_world.pursuit_evasion.model as pe_model


class PursuitEvasionEnv(core.Env):
    """The Pursuit-Evasion Grid World Environment.

    An adversarial 2D grid world problem involving two agents, a runner and
    a chaser. The runner's goal is to reach a goal location, on the other side
    of the grid, while the goal of the chaser is to spot the runner
    before it reaches it's goal. The runner is considered caught if it is
    observed by the chaser, or occupies the same location. The runner and
    chaser have knowledge of each others starting locations. However, only the
    runner has knowledge of it's goal location. The chaser only knowns that the
    goal location is somewhere on the opposite side of the grid to the runners
    start location.

    This environment requires each agent to reason about the which path the
    other agent will take through the dense grid environment.

    Agents
    ------
    Runner=0
    Chaser=1

    State
    -----
    Each state is made up of:

    0. the (x, y) coordinate of the runner
    1. the direction the runner is facing
    2. the (x, y) coordinate of the chaser
    3. the direction the chaser is facing
    4. the (x, y) coordinate of the runner
    5. the (x, y) coordinate of the runner's start location
    6. the (x, y) coordinate of the chaser's start location

    Actions
    -------
    Each agent has 4 actions corresponding to moving in the 4 cardinal
    directions (NORTH=0, SOUTH=1, EAST=2, WEST=3).

    Observation
    -----------
    Each agent observes:

    1. whether there is a wall or not in the adjacent cells in the four
       cardinal directions,
    2. whether they see the other agent in a cone in front of them
    3. whether they hear the other agent (whether the other agent is within
       distance 2 from the agent in any direction)
    4. the (x, y) coordinate of the runner's start location
    5. the (x, y) coordinate of the chaser's start location

    The Runner also observes:

    1. the (x, y) coordinate of the runner's goal location

    Note, the goal and start coordinate observations do not change during a
    single episode, but they do change between episodes.

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

    References
    ----------

    This Pursuit-Evasion implementation is a discrete version of the problem
    presented in the paper:
    - Seaman et al (2018) *Nested Reasoning About Autonomous Agents Using
      Probabilistic Programs*
    """

    metadata = {"render.modes": ['human', 'ascii']}

    def __init__(self,
                 grid_name: str,
                 action_probs: Union[float, Tuple[float, float]] = 1.0,
                 **kwargs):
        self._model = pe_model.PursuitEvasionModel(
            grid_name, action_probs, **kwargs
        )

        init_conds = self._model.sample_initial_state_and_obs()
        self._state, self._last_obs = init_conds
        self._step_num = 0
        self._last_actions: Optional[pe_model.PEJointAction] = None
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
        self._last_actions = actions       # type: ignore
        self._last_rewards = step.rewards
        aux = {"outcomes": step.outcomes}
        return (step.observations, step.rewards, step.done, aux)

    def reset(self, seed: Optional[int] = None) -> M.JointObservation:
        init_conds = self._model.sample_initial_state_and_obs()
        self._state, self._last_obs = init_conds
        self._last_actions = None
        self._last_rewards = None
        # reset renderer since goal location can change between episodes
        self._renderer = None
        self._step_num = 0
        return self._last_obs

    def render(self, mode: str = "human") -> None:
        if mode == "ascii":
            outfile = sys.stdout

            grid_str = self._model.grid.get_ascii_repr(
                goal_coord=self._state[4],
                runner_coord=self._state[0],
                chaser_coord=self._state[2]
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
                    "Pursuit-Evasion Env", (grid.width, grid.height)
                )
                self._viewer.show(block=False)   # type: ignore

            if self._renderer is None:
                static_objs = [
                    render_lib.GWObject(
                        self._state[4], 'green', render_lib.Shape.RECTANGLE
                    )
                ]
                self._renderer = render_lib.GWRenderer(
                    self.n_agents,
                    grid,
                    static_objs,
                    render_blocks=True
                )

            agent_obs_coords = tuple(
                list(grid.get_fov(
                    self._state[2*i],
                    self._state[2*i + 1],
                    self.model.FOV_EXPANSION_INCR
                )) for i in range(self.n_agents)
            )

            img = self._renderer.render(
                (self._state[0], self._state[2]),
                agent_obs_coords,
                agent_dirs=(self._state[1], self._state[3]),
                other_objs=None,
                agent_colors=None
            )
            self._viewer.display_img(img)  # type: ignore

    @property
    def model(self) -> pe_model.PursuitEvasionModel:
        return self._model