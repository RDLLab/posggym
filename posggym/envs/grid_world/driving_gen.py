"""The Generated Driving Grid World Environment."""
from typing import Any, Dict, Optional, Set, Tuple, Union

from posggym.envs.grid_world.core import Coord, GridCycler, GridGenerator
from posggym.envs.grid_world.driving import DObs, DrivingEnv, DrivingGrid


class DrivingGenEnv(DrivingEnv):
    """The Generated Driving Grid World Environment.

    This is the same as the [Driving](/environments/grid_world/driving) Environment
    except that a new grid is generated at each reset.

    For environment attributes see [Driving](/environments/grid_world/driving)
    environment class documentation.

    Arguments
    ---------

    - `num_agents` - the number of agents in the environment (default = `2`).
    - `obs_dim` - the local observation dimensions, specifying how many cells in front,
        behind, and to each side the agent observes (default = `(3, 1, 1)`, resulting
        in the agent observing a 5x3 area: 3 in front, 1 behind, 1 to each side.)
    - `obstacle_collisions` -  whether running into a wall results in the agent's
        vehicle crashing and thus the agent reaching a terminal state. This can make
        the problem significantly harder (default = "False").
    - `generator_params` - the parameters to use for generating the grid
        (default = "14x14"). This can either be a string specyfing one of the supported
        sets of params (can be "7x7", "14x14", "28x28") or a dictionary with the
        following keys:
        - `width` - width of the grid
        - `height` - height of the grid
        - `max_obstacle_size` - max size of an obstacle
        - `max_num_obstacles` - maximum number of obstacles in the grid
    - `n_grids` - the number of different grids to generate, if provided then `n_grids`
        will be generated and these will be cycled through for each reset. If `None`
        then a different, possibly unique, grid will be generated each episode
        (default = `None`).
    - `shuffle_grid_order` - whether to shuffle the order in which different grids
        appear. Only has any effect if `n_grids > 1` (default = `True`).

    Available variants
    ------------------

    The DrivingGen environment comes with a number of pre-built sets of grid generator
    parameters which can be passed as an argument to `posggym.make`:

    | Name    | Grid size | max obstacle size | max num obstacles |
    |---------|-----------|-------------------|-------------------|
    | `7x7`   | 7x7       | 2                 | 21                |
    | `14x14` | 14x14     | 3                 | 42                |
    | `28x28` | 28x28     | 4                 | 84                |


    For example to use the DrivingGen environment with the pre-build `7x7` set of
    generation parameters and 2 agents, you would use:

    ```python
    import posggym
    env = posggym.make('DrivingGen-v0', generator_params="7x7", num_agents="2")
    ```

    Version History
    ---------------
    - `v0`: Initial version

    """

    def __init__(
        self,
        num_agents: int = 2,
        obs_dim: Tuple[int, int, int] = (3, 1, 2),
        obstacle_collisions: bool = False,
        generator_params: Union[str, Dict[str, int]] = "14x14",
        n_grids: Optional[int] = None,
        shuffle_grid_order: bool = True,
        render_mode: Optional[str] = None,
    ):
        if isinstance(generator_params, str):
            assert generator_params in SUPPORTED_GEN_PARAMS, (
                f"Unsupported grid generator parameters'{generator_params}'. If "
                "grid `generator_params` argument is a string it must be one of: "
                f"{SUPPORTED_GEN_PARAMS.keys()}."
            )
            generator_params = SUPPORTED_GEN_PARAMS[generator_params][0]

        self._generator_params = generator_params
        self._n_grids = n_grids
        self._shuffle_grid_order = shuffle_grid_order
        self._gen = DrivingGridGenerator(**self._generator_params)

        if n_grids is not None:
            grids = self._gen.generate_n(n_grids)
            self._cycler = GridCycler(grids, shuffle_grid_order)
            grid: "DrivingGrid" = grids[0]  # type: ignore
        else:
            self._cycler = None  # type: ignore
            grid = self._gen.generate()

        super().__init__(
            grid,
            num_agents,
            obs_dim,
            obstacle_collisions=obstacle_collisions,
            render_mode=render_mode,
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, DObs], Dict[str, Dict]]:
        if seed is not None:
            self._model_seed = seed
            self._gen = DrivingGridGenerator(seed=seed, **self._generator_params)

            if self._n_grids is not None:
                grids = self._gen.generate_n(self._n_grids)
                self._cycler = GridCycler(grids, self._shuffle_grid_order, seed=seed)

        grid = self._cycler.next() if self._cycler is not None else self._gen.generate()

        self.model.grid = grid  # type: ignore

        if self.render_mode != "ansi" and self.renderer is not None:
            self.renderer.grid = grid
            self.renderer.reset_blocks()

        return super().reset(seed=seed)


class DrivingGridGenerator(GridGenerator):
    """Class for generating grid layouts for Driving Environment.

    Generates starting and destination coords in an alternating pattern along
    the outside of edge of the grid.
    """

    def __init__(
        self,
        width: int,
        height: int,
        max_obstacle_size: int,
        max_num_obstacles: int,
        seed: Optional[int] = None,
    ):
        super().__init__(
            width,
            height,
            self._generate_mask(width, height),
            max_obstacle_size,
            max_num_obstacles,
            ensure_grid_connected=True,
            seed=seed,
        )

        self._start_coords = [self.mask for _ in range(len(self.mask))]
        self._dest_coords = [self.mask for _ in range(len(self.mask))]

    def _generate_mask(self, width: int, height: int) -> Set[Coord]:
        start = 1
        mask = set()
        for x in range(start, width, 2):
            mask.add((x, 0))
            mask.add((x, height - 1))

        for y in range(start, height, 2):
            mask.add((0, y))
            mask.add((width - 1, y))

        return mask

    def generate(self) -> DrivingGrid:
        """Generate a new Driving grid."""
        base_grid = super().generate()
        driving_grid = DrivingGrid(
            base_grid.width,
            base_grid.height,
            base_grid.block_coords,
            self._start_coords,
            self._dest_coords,
        )
        return driving_grid


# (generator params, finite horizon step limit)
SUPPORTED_GEN_PARAMS = {
    "7x7": (
        {
            "width": 7,
            "height": 7,
            "max_obstacle_size": 2,
            "max_num_obstacles": 21,  # size * 3
        },
        50,
    ),
    "14x14": (
        {
            "width": 14,
            "height": 14,
            "max_obstacle_size": 3,
            "max_num_obstacles": 42,  # size * 3
        },
        50,
    ),
    "28x28": (
        {
            "width": 28,
            "height": 28,
            "max_obstacle_size": 4,
            "max_num_obstacles": 84,  # size * 3
        },
        100,
    ),
}
