import math
from itertools import product
from typing import Optional, Set, Sequence, List

from posggym.envs.grid_world.core import Grid, Coord


class PPGrid(Grid):
    """A grid for the Predator-Prey Problem."""

    def __init__(self,
                 grid_size: int,
                 block_coords: Optional[Set[Coord]],
                 predator_start_coords: Optional[List[Coord]] = None,
                 prey_start_coords: Optional[List[Coord]] = None):
        assert grid_size >= 3
        super().__init__(grid_size, grid_size, block_coords)
        self.size = grid_size
        # predators start in corners or half-way along a side
        if predator_start_coords is None:
            predator_start_coords = list(
                c for c in product([0, grid_size//2, grid_size-1], repeat=2)
                if c[0] in (0, grid_size-1) or c[1] in (0, grid_size-1)
            )
        self.predator_start_coords = predator_start_coords
        self.prey_start_coords = prey_start_coords

    def get_ascii_repr(self,
                       predator_coords: Optional[Sequence[Coord]],
                       prey_coords: Optional[Sequence[Coord]]) -> str:
        """Get ascii repr of grid."""
        grid_repr = []
        for row in range(self.height):
            row_repr = []
            for col in range(self.width):
                coord = (col, row)
                if coord in self.block_coords:
                    row_repr.append("#")
                else:
                    row_repr.append(".")
            grid_repr.append(row_repr)

        if predator_coords is not None:
            for c in predator_coords:
                grid_repr[c[0]][c[1]] = "P"
        if prey_coords is not None:
            for c in predator_coords:
                grid_repr[c[0]][c[1]] = "p"

        return (
            str(self)
            + "\n" + "\n".join(list(list((" ".join(r) for r in grid_repr))))
        )

    def get_unblocked_center_coords(self, num: int) -> List[Coord]:
        """Get at least num closest coords to the center of grid.

        May return more than num, since can be more than one coord at equal
        distance from the center.
        """
        assert num < self.n_coords - len(self.block_coords)
        center = (self.width // 2, self.height // 2)
        min_dist_from_center = math.ceil(math.sqrt(num)) - 1
        coords = self.get_coords_within_dist(
            center,
            min_dist_from_center,
            ignore_blocks=False,
            include_origin=True
        )

        while len(coords) < num:
            # not the most efficient as it repeats work,
            # but function should only be called once when model is initialized
            # and for small num
            for c in coords:
                coords.update(self.get_neighbours(
                    c, ignore_blocks=False, include_out_of_bounds=False
                ))

        return list(coords)

    def num_unblocked_neighbours(self, coord: Coord) -> int:
        """Get number of neighbouring coords that are unblocked."""
        return len(self.get_neighbours(
            coord, ignore_blocks=False, include_out_of_bounds=False
        ))


def parse_grid_str(grid_str: str) -> PPGrid:
    """Parse a str representation of a grid into a grid object.

    Notes on grid str representation:

    . = empty/unallocated cell
    # = a block
    P = starting location for predator agents [optional] (defaults to edges)
    p = starting location for prey agent [optional] (defaults to center)

    Examples (" " quotes and newline chars ommited):

    1. A 10x10 grid with 4 groups of blocks and using the default predator
       and prey start locations.

    ..........
    ..........
    ..##..##..
    ..##..##..
    ..........
    ..........
    ..##..##..
    ..##..##..
    ..........
    ..........

    2. Same as above but with predator and prey start locations defined for
    up to 8 predators and 4 prey. (This would be the default layout for the
    scenario where there are between 2 and 4 prey, i.e. if prey and predator
    start locations were left unspecified as in example 1.)

    P....P...P
    ..........
    ..##..##..
    ..##..##..
    ....pp....
    P...pp...P
    ..##..##..
    ..##..##..
    ..........
    P....P...P

    """
    row_strs = grid_str.splitlines()
    assert len(row_strs) > 1
    assert all(len(row) == len(row_strs[0]) for row in row_strs)
    assert len(row_strs[0]) > 1
    assert len(row_strs) == len(row_strs[0])

    grid_size = len(row_strs)
    block_coords = set()
    predator_start_coords = set()
    prey_start_coords = set()
    for r, c in product(range(grid_size), repeat=2):
        coord = (c, r)
        char = row_strs[r][c]

        if char == "#":
            block_coords.add(coord)
        elif char == "P":
            predator_start_coords.add(coord)
        elif char == "p":
            prey_start_coords.add(coord)
        else:
            assert char == "."

    if len(predator_start_coords) == 0:
        predator_start_coords = None
    else:
        predator_start_coords = list(predator_start_coords)

    if len(prey_start_coords) == 0:
        prey_start_coords = None
    else:
        prey_start_coords = list(prey_start_coords)

    return PPGrid(
        grid_size,
        block_coords,
        predator_start_coords,
        prey_start_coords
    )


def get_5x5_grid() -> PPGrid:
    """Generate 5x5 grid layout."""
    return PPGrid(grid_size=5, block_coords=None)


def get_5x5_blocks_grid() -> PPGrid:
    """Generate 5x5 Blocks grid layout."""
    grid_str = (
        ".....\n"
        ".#.#.\n"
        ".....\n"
        ".#.#.\n"
        ".....\n"
    )
    return parse_grid_str(grid_str)


def get_10x10_grid() -> PPGrid:
    """Generate 10x10 grid layout."""
    return PPGrid(grid_size=10, block_coords=None)


def get_10x10_blocks_grid() -> PPGrid:
    """Generate 10x10 Blocks grid layout."""
    grid_str = (
        "..........\n"
        "..........\n"
        "..##..##..\n"
        "..##..##..\n"
        "..........\n"
        "..........\n"
        "..##..##..\n"
        "..##..##..\n"
        "..........\n"
        "..........\n"
    )
    return parse_grid_str(grid_str)


def get_15x15_grid() -> PPGrid:
    """Generate 15x15 grid layout."""
    return PPGrid(grid_size=15, block_coords=None)


def get_15x15_blocks_grid() -> PPGrid:
    """Generate 10x10 Blocks grid layout."""
    grid_str = (
        "...............\n"
        "...............\n"
        "...............\n"
        "...###...###...\n"
        "...###...###...\n"
        "...###...###...\n"
        "...............\n"
        "...............\n"
        "...............\n"
        "...###...###...\n"
        "...###...###...\n"
        "...###...###...\n"
        "...............\n"
        "...............\n"
        "...............\n"
    )
    return parse_grid_str(grid_str)


def get_20x20_grid() -> PPGrid:
    """Generate 20x20 grid layout."""
    return PPGrid(grid_size=20, block_coords=None)


def get_20x20_blocks_grid() -> PPGrid:
    """Generate 20x20 Blocks grid layout."""
    grid_str = (
        "....................\n"
        "....................\n"
        "....................\n"
        "....................\n"
        "....####....####....\n"
        "....####....####....\n"
        "....####....####....\n"
        "....####....####....\n"
        "....................\n"
        "....................\n"
        "....................\n"
        "....................\n"
        "....####....####....\n"
        "....####....####....\n"
        "....####....####....\n"
        "....####....####....\n"
        "....................\n"
        "....................\n"
        "....................\n"
        "....................\n"
    )
    return parse_grid_str(grid_str)


#  (grid_make_fn, step_limit)
SUPPORTED_GRIDS = {
    "5x5": (get_5x5_grid, 25),
    "5x5Blocks": (get_5x5_blocks_grid, 50),
    "10x10": (get_10x10_grid, 50),
    "10x10Blocks": (get_10x10_blocks_grid, 50),
    "15x15": (get_15x15_grid, 100),
    "15x15Blocks": (get_15x15_blocks_grid, 100),
    "20x20": (get_20x20_grid, 200),
    "20x20Blocks": (get_20x20_blocks_grid, 200)
}


def load_grid(grid_name: str) -> PPGrid:
    """Load grid with given name."""
    grid_name = grid_name.lower()
    assert grid_name in SUPPORTED_GRIDS, (
        f"Unsupported grid name '{grid_name}'. Grid name must be one of: "
        f"{SUPPORTED_GRIDS.keys()}."
    )
    return SUPPORTED_GRIDS[grid_name][0]()
