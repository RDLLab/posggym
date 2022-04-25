"""General grid world problem utility functions and classes."""
import enum
import itertools
from queue import PriorityQueue
from typing import Tuple, List, Set, Optional, Iterable, Dict

Coord = Tuple[int, int]


class Direction(enum.IntEnum):
    """A direction in a grid."""
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


DIRECTION_ASCII_REPR = ["^", "v", ">", "<"]


class Grid:
    """A base grid class, with general utility functions and attributes."""

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
        """The list of all locations on grid, including blocks."""
        return list(itertools.product(range(self.width), range(self.width)))

    @property
    def n_coords(self) -> int:
        """The total number of coordinates on grid, including blocks."""
        return self.height * self.width

    @property
    def unblocked_coords(self) -> List[Coord]:
        """The list of all coordinates on the grid excluding blocks."""
        return [
            coord for coord in self.all_coords
            if coord not in self.block_coords
        ]

    @staticmethod
    def manhattan_dist(coord1: Coord, coord2: Coord) -> int:
        """Get manhattan distance between two coordinates on the grid."""
        return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

    def coord_in_bounds(self, coord: Coord) -> bool:
        """Return whether a coordinate is inside the grid or not."""
        return 0 <= coord[0] < self.width and 0 <= coord[1] < self.height

    def get_neighbours(self,
                       coord: Coord,
                       ignore_blocks: bool = False,
                       include_out_of_bounds: bool = False
                       ) -> List[Coord]:
        """Get set of adjacent non-blocked coordinates."""
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
                       ignore_blocks: bool = False) -> Coord:
        """Get next coordinate given loc and movement direction.

        If new coordinate is outside of the grid boundary, or (ignore_blocks is
        False and new coordinate is a block) then returns the original
        coordinate.
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

    def get_coords_within_dist(self,
                               origin: Coord,
                               dist: int,
                               ignore_blocks: bool,
                               include_origin: bool) -> Set[Coord]:
        """Get set of coords within given distance from origin."""
        if dist == 0:
            return {origin} if include_origin else set()

        adj_coords = self.get_neighbours(origin, ignore_blocks)
        in_dist_coords = set(adj_coords)

        if (
            include_origin
            and (ignore_blocks or origin not in self.block_coords)
        ):
            in_dist_coords.add(origin)

        if dist == 1:
            return in_dist_coords

        for coord in adj_coords:
            in_dist_coords.update(
                self.get_coords_within_dist(
                    coord, dist-1, ignore_blocks, False
                )
            )

        if not include_origin and origin in in_dist_coords:
            # must remove since it will get added again during recursive call
            in_dist_coords.remove(origin)

        return in_dist_coords

    def get_coords_at_dist(self,
                           origin: Coord,
                           dist: int,
                           ignore_blocks: bool) -> Set[Coord]:
        """Get set of coords at given distance from origin."""
        if dist == 0:
            return {origin}

        in_dist_coords = self.get_coords_within_dist(
            origin, dist, ignore_blocks, False
        )

        at_dist_coords: Set[Coord] = set()
        for coord in in_dist_coords:
            if self.manhattan_dist(origin, coord) == dist:
                at_dist_coords.add(coord)

        return at_dist_coords

    def get_min_dist_coords(self,
                            origin: Coord,
                            coords: Iterable[Coord]) -> List[Coord]:
        """Get list of coord in coords closest to origin."""
        dists = self.get_coords_by_distance(origin, coords)
        if len(dists) == 0:
            return []
        return dists[min(dists)]

    def get_coords_by_distance(self,
                               origin: Coord,
                               coords: Iterable[Coord]
                               ) -> Dict[int, List[Coord]]:
        """Get mapping from distance to coords at that distance from origin."""
        dists: Dict[int, List[Coord]] = {}
        for coord in coords:
            d = self.manhattan_dist(origin, coord)
            if d not in dists:
                dists[d] = []
            dists[d].append(coord)
        return dists

    def get_all_shortest_paths(self,
                               origins: List[Coord]
                               ) -> Dict[Coord, Dict[Coord, float]]:
        """Get shortest path distance from every origin to all other coords."""
        src_dists = {}
        for origin in origins:
            src_dists[origin] = self.dijkstra(origin)
        return src_dists

    def dijkstra(self, origin: Coord) -> Dict[Coord, float]:
        """Get shortest path distance between origin and all other coords."""
        dist = {origin: 0.0}
        pq = PriorityQueue()   # type: ignore
        pq.put((dist[origin], origin))

        visited = set([origin])

        while not pq.empty():
            _, coord = pq.get()

            for adj_coord in self.get_neighbours(coord, False):
                if dist[coord] + 1 < dist.get(adj_coord, float('inf')):
                    dist[adj_coord] = dist[coord] + 1

                    if adj_coord not in visited:
                        pq.put((dist[adj_coord], adj_coord))
                        visited.add(adj_coord)
        return dist
