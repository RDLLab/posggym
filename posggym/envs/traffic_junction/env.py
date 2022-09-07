import copy
from typing import Optional, Tuple

import numpy as np

from posggym import core
import posggym.model as M

from ma_gym.envs.utils.draw import draw_grid, fill_cell, write_cell_text

from baposgmcp.envs.traffic_junction.model import TrafficJunctionModel


CELL_SIZE = 30

WALL_COLOR = 'black'

# fixed colors for #agents = n_max <= 10
AGENTS_COLORS = [
    "red",
    "blue",
    "yellow",
    "orange",
    "black",
    "green",
    "purple",
    "pink",
    "brown",
    "grey"
]


class TrafficJunctionEnv(core.Env):
    """The ma-gym TrafficJuntion env.

    Ref:
    https://github.com/koulanurag/ma-gym/blob/master/ma_gym/envs/traffic_junction/traffic_junction.py
    """

    metadata = {"render.modes": ['human', 'rgb_array']}

    def __init__(self, **kwargs):
        self._model = TrafficJunctionModel(**kwargs)
        self._grid_shape = self._model.GRID_SHAPE

        init_conds = self._model.sample_initial_state_and_obs()
        self._state, self._last_obs = init_conds
        self._step_num = 0
        self._last_actions: Optional[M.JointAction] = None
        self._last_rewards: Optional[M.JointReward] = None

        # rendering objects
        # only create as required
        self._viewer = None
        self._base_img = None

    def step(self,
             actions: M.JointAction
             ) -> Tuple[M.JointObservation, M.JointReward, bool, dict]:
        step = self._model.step(self._state, actions)
        self._step_num += 1
        self._state = step.state
        self._last_obs = step.observations
        self._last_actions = actions
        self._last_rewards = step.rewards
        aux = {"outcome": step.outcomes}
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

    def render(self, mode: str = "human"):
        if self._base_img is None:
            # cache base img (speeds up future rendering steps a lot)
            self._base_img = self._draw_base_img()

        img = copy.copy(self._base_img)
        for i in range(self.n_agents):
            if not self._state.agent_dones[i] and self._state.on_the_road[i]:
                fill_cell(
                    img,
                    self._state.agent_pos[i],
                    cell_size=CELL_SIZE,
                    fill=AGENTS_COLORS[i]
                )
                write_cell_text(
                    img,
                    str(i+1),
                    self._state.agent_pos[i],
                    cell_size=CELL_SIZE,
                    fill='white',
                    margin=0.3
                )

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self._viewer is None:
                self._viewer = rendering.SimpleImageViewer()
            self._viewer.imshow(img)
            return self._viewer.isopen

        raise NotImplementedError(f"Unsupported rendering mode '{mode}'")

    @property
    def model(self) -> TrafficJunctionModel:
        return self._model

    @property
    def state(self) -> M.State:
        return copy.copy(self._state)

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()   # type: ignore
            self._viewer = None

    def _draw_base_img(self):
        # create grid and make everything black
        img = draw_grid(
            self._grid_shape[0],
            self._grid_shape[1],
            cell_size=CELL_SIZE,
            fill=WALL_COLOR
        )

        middle_rows = [self._grid_shape[0] // 2 - 1, self._grid_shape[0] // 2]
        middle_cols = [self._grid_shape[1] // 2 - 1, self._grid_shape[1] // 2]

        # draw tracks
        for row in range(self._grid_shape[0]):
            for col in range(self._grid_shape[1]):
                if row in middle_rows or col in middle_cols:
                    # empty
                    fill_cell(
                        img,
                        (row, col),
                        cell_size=CELL_SIZE,
                        fill=(143, 141, 136),
                        margin=0.05
                    )
                else:
                    # wall
                    fill_cell(
                        img,
                        (row, col),
                        cell_size=CELL_SIZE,
                        fill=(242, 227, 167),
                        margin=0.02
                    )

        return img
