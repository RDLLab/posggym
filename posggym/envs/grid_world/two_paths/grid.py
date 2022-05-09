"""A grid in the Two-Paths Problem."""
from typing import Set, Optional

from posggym.envs.grid_world.core import Grid, Coord


class TPGrid(Grid):
    """A grid for the Two-Paths Problem."""

    def __init__(self,
                 grid_width: int,
                 grid_height: int,
                 block_coords: Set[Coord],
                 goal_coords: Set[Coord],
                 init_runner_coord: Coord,
                 init_chaser_coord: Coord):
        super().__init__(grid_width, grid_height, block_coords)
        self.goal_coords = goal_coords
        self.init_runner_coord = init_runner_coord
        self.init_chaser_coord = init_chaser_coord

    def get_ascii_repr(self,
                       runner_coord: Optional[Coord],
                       chaser_coord: Optional[Coord]) -> str:
        """Get ascii repr of grid."""
        grid_repr = []
        for row in range(self.height):
            row_repr = []
            for col in range(self.width):
                coord = (col, row)
                if coord in self.goal_coords:
                    row_repr.append("G")
                elif coord in self.block_coords:
                    row_repr.append("#")
                else:
                    row_repr.append(".")
            grid_repr.append(row_repr)

        if runner_coord is not None:
            grid_repr[runner_coord[1]][runner_coord[0]] = "R"
        if chaser_coord is not None:
            if chaser_coord == runner_coord:
                grid_repr[chaser_coord[1]][chaser_coord[0]] = "X"
            else:
                grid_repr[chaser_coord[1]][chaser_coord[0]] = "C"

        return "\n".join(list(list((" ".join(r) for r in grid_repr))))

    def get_init_ascii_repr(self) -> str:
        """Get ascii repr of initial grid."""
        return self.get_ascii_repr(
            self.init_runner_coord, self.init_chaser_coord
        )


def get_3x3_grid() -> TPGrid:
    """Generate the Two-Paths 3-by-3 grid layout.

       012
      #####
    0 #GC #
    1 # #G#
    2 # R #
      #####

    """
    return TPGrid(
        grid_height=3,
        grid_width=3,
        block_coords=set([(1, 1)]),
        goal_coords=set([(0, 0), (2, 1)]),
        init_runner_coord=(1, 2),
        init_chaser_coord=(1, 0),
    )


def get_4x4_grid() -> TPGrid:
    """Generate the Choose a Path 4-by-4 grid layout.

       0123
      ######
    0 #G C #
    1 # ##G#
    2 # #  #
    3 #  R##
      ######

    """
    block_coords = set([
        #
        (1, 1), (2, 1),
        (1, 2),
        (3, 3)
    ])

    return TPGrid(
        grid_height=4,
        grid_width=4,
        block_coords=block_coords,
        goal_coords=set([(0, 0), (3, 1)]),
        init_runner_coord=(2, 3),
        init_chaser_coord=(2, 0),
    )


def get_7x7_grid() -> TPGrid:
    """Generate the Two-Paths 7-by-7 grid layout.

       0123456
      #########
    0 #G   C  #
    1 # ##### #
    2 # #### G#
    3 #  ### ##
    4 ## ### ##
    5 ##  #  ##
    6 ### R ###
      #########

    """
    block_coords = set([
        #
        (1, 1), (2, 1), (3, 1), (4, 1), (5, 1),
        (1, 2), (2, 2), (3, 2), (4, 2),
        (2, 3), (3, 3), (4, 3), (6, 3),
        (0, 4), (2, 4), (3, 4), (4, 4), (6, 4),
        (0, 5), (3, 5), (6, 5),
        (0, 6), (1, 6), (5, 6), (6, 6)
    ])

    return TPGrid(
        grid_height=7,
        grid_width=7,
        block_coords=block_coords,
        goal_coords=set([(0, 0), (6, 2)]),
        init_runner_coord=(3, 6),
        init_chaser_coord=(4, 0),
    )


# grid_name:
#  (grid_make_fn, finite horizon step_limit, infinite horizon step_limit)
SUPPORTED_GRIDS = {
    '3x3': (get_3x3_grid, 20, 100),
    '4x4': (get_4x4_grid, 20, 100),
    '7x7': (get_7x7_grid, 20, 100)
}


def load_grid(grid_name: str) -> TPGrid:
    """Load grid with given name."""
    grid_name = grid_name.lower()
    assert grid_name in SUPPORTED_GRIDS, (
        f"Unsupported grid name '{grid_name}'. Grid name must be one of: "
        f"{SUPPORTED_GRIDS.keys()}."
    )
    return SUPPORTED_GRIDS[grid_name][0]()
