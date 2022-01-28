"""General grid world problem utility functions and classes """
import enum
import itertools
from typing import Tuple, List, Set, Optional

Coord = Tuple[int, int]


class Direction(enum.IntEnum):
    """A direction in a grid """
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class Grid:
    """A base grid class, with general utility functions and attributes """

    def __init__(self,
                 grid_width: int,
                 grid_height: int,
                 block_coords: Optional[Set[Coord]] = None):
        self.width = grid_width
        self.height = grid_height

        if block_coords is None:
            block_coords = set()
        self.block_coords = block_coords

    @property
    def all_coords(self) -> List[Coord]:
        """The list of all locations on grid, including blocks """
        return list(itertools.product(range(self.width), range(self.width)))

    @property
    def n_coords(self) -> int:
        """The total number of coordinates on grid, including blocks """
        return self.height * self.width

    @property
    def unblocked_coords(self) -> List[Coord]:
        """The list of all coordinates on the grid excluding blocks """
        return [
            coord for coord in self.all_coords
            if coord not in self.block_coords
        ]

    @staticmethod
    def manhattan_dist(coord1: Coord, coord2: Coord) -> int:
        """Get manhattan distance between two coordinates on the grid """
        return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

    def get_neighbours(self,
                       coord: Coord,
                       ignore_blocks: bool,
                       include_out_of_bounds: bool = False
                       ) -> List[Coord]:
        """Get set of adjacent non-blocked coordinates """
        neighbours = []
        if coord[1] > 0 or include_out_of_bounds:
            neighbours.append((coord[0], coord[1]-1))    # N
        if coord[1] < self.height - 1 or include_out_of_bounds:
            neighbours.append((coord[0], coord[1]+1))    # S
        if coord[0] < self.width - 1 or include_out_of_bounds:
            neighbours.append((coord[0]+1, coord[1]))    # E
        if coord[0] > 0 or include_out_of_bounds:
            neighbours.append((coord[0]-1, coord[1]))    # W

        if ignore_blocks:
            return neighbours

        for i in range(len(neighbours), 0, -1):
            if neighbours[i-1] in self.block_coords:
                neighbours.pop(i-1)

        return neighbours

    def get_next_coord(self,
                       coord: Coord,
                       move_dir: Direction,
                       ignore_blocks: bool) -> Coord:
        """Get next coordinate given loc and movement direction

        If new coordinate is outside of the grid boundary then returns the
        original coordinate.
        """
        new_coord_list = list(coord)
        if move_dir == Direction.NORTH:
            new_coord_list[1] = max(0, coord[1]-1)
        elif move_dir == Direction.SOUTH:
            new_coord_list[1] = min(self.height-1, coord[1]+1)
        elif move_dir == Direction.EAST:
            new_coord_list[0] = min(self.width-1, coord[0]+1)
        elif move_dir == Direction.WEST:
            new_coord_list[0] = max(0, coord[0]-1)
        new_coord = (new_coord_list[0], new_coord_list[1])

        if not ignore_blocks and new_coord in self.block_coords:
            return coord
        return new_coord
