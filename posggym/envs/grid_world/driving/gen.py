from typing import Set, Optional

import posggym.envs.grid_world.driving.grid as dgrid
from posggym.envs.grid_world.core import Coord, GridGenerator


class DrivingGridGenerator(GridGenerator):
    """Class for generating grid layouts for Driving Environment.

    Generates starting and destination coords in an alternating pattern along
    the outside of edge of the grid.
    """

    def __init__(self,
                 width: int,
                 height: int,
                 max_obstacle_size: int,
                 max_num_obstacles: int,
                 seed: Optional[int] = None):
        super().__init__(
            width,
            height,
            self._generate_mask(width, height),
            max_obstacle_size,
            max_num_obstacles,
            ensure_grid_connected=True,
            seed=seed
        )

        self._start_coords = [self.mask for _ in range(len(self.mask))]
        self._dest_coords = [self.mask for _ in range(len(self.mask))]

    def _generate_mask(self, width: int, height: int) -> Set[Coord]:
        start = 1
        mask = set()
        for x in range(start, width, 2):
            mask.add((x, 0))
            mask.add((x, height-1))

        for y in range(start, height, 2):
            mask.add((0, y))
            mask.add((width-1, y))

        return mask

    def generate(self) -> dgrid.DrivingGrid:
        """Generate a new Driving grid."""
        base_grid = super().generate()
        driving_grid = dgrid.DrivingGrid(
            base_grid.width,
            base_grid.height,
            base_grid.block_coords,
            self._start_coords,
            self._dest_coords
        )
        return driving_grid


# (generator params, finite horizon step limit, infinite horizon step limit)
SUPPORTED_GEN_PARAMS = {
    "7x7": (
        {
            "width": 7,
            "height": 7,
            "max_obstacle_size": 2,
            "max_num_obstacles": 21    # size * 3
        },
        40,
        100
    ),
    "15x15": (
        {
            "width": 15,
            "height": 15,
            "max_obstacle_size": 3,
            "max_num_obstacles": 45,   # size * 3
        },
        100,
        200
    ),
    "29x29": (
        {
            "width": 29,
            "height": 29,
            "max_obstacle_size": 4,
            "max_num_obstacles": 77,    # size * 3
        },
        200,
        400
    )
}
