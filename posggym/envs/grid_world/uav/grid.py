"""A grid in the Unmanned Aerial Vehicle Problem."""
import itertools
from typing import List, Optional, Set

from posggym.envs.grid_world.core import Grid, Coord


class UAVGrid(Grid):
    """A grid for the MA UAV Problem."""

    def __init__(self,
                 grid_width: int,
                 grid_height: int,
                 block_coords: Optional[Set[Coord]],
                 safe_house_coord: Coord,
                 init_fug_coords: List[Coord],
                 init_uav_coords: List[Coord]):
        super().__init__(grid_width, grid_height, block_coords)
        self.safe_house_coord = safe_house_coord
        self.init_fug_coords = init_fug_coords
        self.init_uav_coords = init_uav_coords

        self.valid_coords = set(self.unblocked_coords)
        self.valid_coords.remove(self.safe_house_coord)

    def get_ascii_repr(self,
                       fug_coord: Optional[Coord],
                       uav_coord: Optional[Coord]) -> str:
        """Get ascii repr of grid."""
        grid_repr = []
        for row in range(self.height):
            row_repr = []
            for col in range(self.width):
                coord = (col, row)
                if coord == self.safe_house_coord:
                    row_repr.append("S")
                elif coord in self.block_coords:
                    row_repr.append("#")
                else:
                    row_repr.append(".")
            grid_repr.append(row_repr)

        if fug_coord is not None:
            grid_repr[fug_coord[0]][fug_coord[1]] = "F"
        if uav_coord is not None:
            if uav_coord == fug_coord:
                grid_repr[uav_coord[0]][uav_coord[1]] = "X"
            else:
                grid_repr[uav_coord[0]][uav_coord[1]] = "U"

        return (
            str(self)
            + "\n" + "\n".join(list(list((" ".join(r) for r in grid_repr))))
        )


def _empty_uav_grid(width, height, safe_house_coord) -> UAVGrid:
    all_coords = set(itertools.product(range(width,), range(height)))
    all_coords.remove(safe_house_coord)

    init_fug_coords = list(all_coords)
    init_uav_coords = [*all_coords]

    return UAVGrid(
        grid_height=width,
        grid_width=height,
        block_coords=None,
        safe_house_coord=safe_house_coord,
        init_fug_coords=init_fug_coords,
        init_uav_coords=init_uav_coords
    )


def get_3x3_grid() -> UAVGrid:
    """Generate UAV 3x3 grid layout.

    .S.
    ...
    ...

    FUG and UAV can start at any point on grid that is not the safe house.

    |S| = 81
    |A_uav| = |A_fug| = 4
    |O_uav| = 81
    |O_fug| = 4
    """
    return _empty_uav_grid(3, 3, (1, 0))


def get_4x4_grid() -> UAVGrid:
    """Generate UAV 4x4 grid layout.

    ....
    ..S.
    ....
    ....

    FUG and UAV can start at any point on grid that is not the safe house

    |S| = 256
    |A_uav| = |A_fug| = 4
    |O_uav| = 256
    |O_fug| = 4
    """
    return _empty_uav_grid(4, 4, (2, 1))


def get_5x5_grid() -> UAVGrid:
    """Generate UAV 5x5 grid layout.

    .....
    ..S..
    .....
    .....
    .....

    FUG and UAV can start at any point on grid that is not the safe house

    |S| = 625
    |A_uav| = |A_fug| = 4
    |O_uav| = 625
    |O_fug| = 4
    """
    return _empty_uav_grid(5, 5, (2, 1))


def get_6x6_grid() -> UAVGrid:
    """Generate UAV 6x6 grid layout.

    ......
    ......
    ..S...
    ......
    ......
    ......

    FUG and UAV can start at any point on grid that is not the safe house

    |S| = 1296
    |A_uav| = |A_fug| = 4
    |O_uav| = 1296
    |O_fug| = 4

    """
    return _empty_uav_grid(6, 6, (2, 2))


SUPPORTED_GRIDS = {
    "3x3": get_3x3_grid,
    "4x4": get_4x4_grid,
    "5x5": get_5x5_grid,
    "6x6": get_6x6_grid
}


def load_grid(grid_name: str) -> UAVGrid:
    """Load grid with given name."""
    grid_name = grid_name.lower()
    assert grid_name in SUPPORTED_GRIDS, (
        f"Unsupported grid name '{grid_name}'. Grid name must be one of: "
        f"{SUPPORTED_GRIDS.keys()}."
    )
    return SUPPORTED_GRIDS[grid_name]()
