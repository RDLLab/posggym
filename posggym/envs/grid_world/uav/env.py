"""Environment class for the Unmanned Aerial Vehicle Grid World Problem."""
import sys
from typing import Optional, Tuple


from posggym import core
import posggym.model as M

from posggym.envs.grid_world.core import Direction
import posggym.envs.grid_world.render as render_lib
import posggym.envs.grid_world.uav.model as uav_model


class UAVEnv(core.Env):
    """The Unmanned Aerial Vehicle Grid World Environment.

    An adversarial 2D grid world problem involving two agents, a Unmanned
    Aerial Vehicle (UAV) and a fugitive. The UAV's goal is to capture the
    fugitive, while the fugitive's goal is to reach the safe house located at
    a known fixed location on the grid. The fugitive is considered caught if
    it is co-located with the UAV. The UAV observes it's own location and
    recieves a noisy observation of the fugitive's location. The fugitive does
    not know it's location but it recieves a noisy observation of its relative
    direction to the safe house when it is adjacent to the safe house.

    Agents
    ------
    UAV = 0
    Fugitive = 1

    State
    -----
    Each state contains the (x, y) (x=column, y=row, with origin at the
    top-left square of the grid) of the UAV and fugitive agent. Specifically,
    a states is ((x_uav, y_uav), (x_fugitive, y_fugitive))

    Initially, the location of both agents is chosen at random.

    Actions
    -------
    Each agent has 4 actions corresponding to moving in the 4 cardinal
    directions (NORTH=0, EAST=1, SOUTH=2, WEST=3).

    Observation
    -----------
    The UAV observes its (x, y) coordinates and recieves a noisy observation
    of the fugitives (x, y) coordinates. The UAV observes the correct fugitive
    coordinates with p=0.9, and one of the adjacent locations to the true
    fugitive location with p=1-0.9.

    The fugitive can sense it's position with respect to the safe house, namely
    whether it is north of it (OBSNORTH=0), south of it (OBSSOUTH=1), or at the
    same level (OBSLEVEL=3). These observations are recieved with accuracy 0.8,
    and only when the fugitive is adjacent to it. If the fugitive is not
    adjacent to the safe house it recieves no observation (OBSNONE=4).

    Reward
    ------
    Both agents receive a penalty of -0.04 for each step.
    If the fugitive reaches the safe house then the fugitive recieves a reward
    of 1, while the UAV recieves a penalty of -1.
    If the fugitive is caught by the UAV, then the fugitive recieves a penalty
    of -1, while the UAV recieves a reward of 1.

    Transition Dynamics
    -------------------
    Actions are deterministic. The fugitive's position is reset at random if
    it reaches the safe house or gets caught by the UAV.

    Reference
    ---------
    Panella, Alessandro, and Piotr Gmytrasiewicz. 2017. “Interactive POMDPs
    with Finite-State Models of Other Agents.” Autonomous Agents and
    Multi-Agent Systems 31 (4): 861–904.
    """

    metadata = {"render.modes": ['human', 'ascii']}

    def __init__(self,
                 grid_name: str,
                 **kwargs):
        self._model = uav_model.UAVModel(grid_name, **kwargs)

        init_conds = self._model.sample_initial_state_and_obs()
        self._state, self._last_obs = init_conds
        self._step_num = 0
        self._last_actions: Optional[uav_model.UAVJointAction] = None
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
        self._last_actions = actions   # type: ignore
        self._last_rewards = step.rewards
        aux = {"outcomes": step.outcomes}
        return (step.observations, step.rewards, step.done, aux)

    def reset(self, *, seed: Optional[int] = None) -> M.JointObservation:
        if seed is not None:
            self._model.set_seed(seed)

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
                    "Unmanned Aerial Vehicle Env", (grid.width, grid.height)
                )
                self._viewer.show(block=False)   # type: ignore

            if self._renderer is None:
                safe_house_obj = render_lib.GWObject(
                    grid.safe_house_coord, 'green', render_lib.Shape.RECTANGLE
                )
                self._renderer = render_lib.GWRenderer(
                    self.n_agents,
                    grid,
                    [safe_house_obj],
                    render_blocks=True
                )

            img = self._renderer.render(
                self._state,
                agent_obs_coords=None,
                agent_dirs=None,
                other_objs=None,
                agent_colors=None
            )
            self._viewer.display_img(img)  # type: ignore

    @property
    def model(self) -> uav_model.UAVModel:
        return self._model

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()   # type: ignore
            self._viewer = None
