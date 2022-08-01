"""General grid world problem utility functions and classes."""
import enum
import random
import itertools
from queue import PriorityQueue, Queue
from typing import Tuple, List, Set, Optional, Iterable, Dict

# (x, y) coord = (col, row) coord
Coord = Tuple[int, int]


class Direction(enum.IntEnum):
    """A direction in a grid."""
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


DIRECTION_ASCII_REPR = ["^", ">", "v", "<"]


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
        if coord[0] < self.width - 1 or include_out_of_bounds:
            neighbours.append((coord[0]+1, coord[1]))    # E
        if coord[1] < self.height - 1 or include_out_of_bounds:
            neighbours.append((coord[0], coord[1]+1))    # S
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
        elif move_dir == Direction.EAST:
            new_coord_list[0] = min(self.width-1, coord[0]+1)
        elif move_dir == Direction.SOUTH:
            new_coord_list[1] = min(self.height-1, coord[1]+1)
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
                               origins: Iterable[Coord]
                               ) -> Dict[Coord, Dict[Coord, int]]:
        """Get shortest path distance from every origin to all other coords."""
        src_dists = {}
        for origin in origins:
            src_dists[origin] = self.dijkstra(origin)
        return src_dists

    def dijkstra(self, origin: Coord) -> Dict[Coord, int]:
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

    def get_connected_components(self) -> List[Set[Coord]]:
        """Get list of connected components.

        A connected component is the set of all coords that are connected to
        each other - i.e. can be reached from each other.
        """
        components = []
        to_explore = set(self.unblocked_coords)

        while len(to_explore) > 0:
            coord = to_explore.pop()
            current_component = set([coord])
            to_expand: Queue[Coord] = Queue()
            to_expand.put(coord)

            while not to_expand.empty():
                coord = to_expand.get()
                for adj_coord in self.get_neighbours(coord):
                    if adj_coord in to_explore:
                        to_explore.remove(adj_coord)
                        to_expand.put(adj_coord)
                        current_component.add(adj_coord)

            components.append(current_component)
        return components


class GridGenerator:
    """Class for generating grid layouts.

    Note, a block in the grid is an obstacle occupying a single cell, while
    here and obstacle refers to a collection of one or more connected blocks.
    """

    def __init__(self,
                 width: int,
                 height: int,
                 mask: Set[Coord],
                 max_obstacle_size: int,
                 max_num_obstacles: int,
                 ensure_grid_connected: bool,
                 seed: Optional[int] = None):
        assert max_obstacle_size > 0
        self.width = width
        self.height = height
        self.mask = mask
        self.max_obstacle_size = max_obstacle_size
        self.max_num_obstacles = max_num_obstacles
        self.ensure_grid_connected = ensure_grid_connected
        self._rng = random.Random(seed)

    def generate(self) -> Grid:
        """Generate a new grid."""
        block_coords: Set[Coord] = set()
        for _ in range(self.max_num_obstacles):
            obstacle = self._get_random_obstacle()
            if not self.mask.intersection(obstacle):
                block_coords.update(obstacle)

        grid = Grid(self.width, self.height, block_coords)
        if self.ensure_grid_connected:
            grid = self.connect_grid_components(grid)

        return grid

    def generate_n(self, n: int) -> List[Grid]:
        """Generate N new grids."""
        grids = [self.generate() for _ in range(n)]
        return grids

    def _get_random_obstacle(self) -> Set[Coord]:
        obstacle_height = self._rng.randint(1, self.max_obstacle_size)
        obstacle_width = self._rng.randint(1, self.max_obstacle_size)
        obstacle_x = self._rng.randint(0, self.width-1)
        obstacle_y = self._rng.randint(0, self.height-1)
        obstacle = set()
        for x in range(obstacle_x, obstacle_x + obstacle_width):
            if x >= self.width:
                break
            for y in range(obstacle_y, obstacle_y + obstacle_height):
                if y >= self.height:
                    break
                obstacle.add((x, y))
        return obstacle

    def connect_grid_components(self, grid: Grid) -> Grid:
        """Get grid with only a single conencted component.

        Removes blocks along shortest path between components until there is
        only a single connected component.
        """
        # make a copy of original grid
        grid = Grid(grid.width, grid.height, set(grid.block_coords))

        components = grid.get_connected_components()
        if len(components) == 1:
            return grid

        # connect components in order of distance from (0, 0)
        component_dists = [
            (self._component_distance(grid, (0, 0), c), i)
            for i, c in enumerate(components)
        ]
        component_dists.sort()

        single_component = components[component_dists[0][1]]
        for j in range(1, len(components)):
            component_j = components[component_dists[j][1]]
            closest_pair = self._get_closest_pair(
                grid, single_component, component_j
            )

            path = self._get_shortest_direct_path(grid, *closest_pair)
            for coord in path:
                if coord in grid.block_coords:
                    grid.block_coords.remove(coord)

            single_component.update(component_j)
            single_component.update(path)

        return grid

    def _component_distance(self,
                            grid: Grid,
                            origin: Coord,
                            component: Set[Coord]) -> int:
        return min([grid.manhattan_dist(origin, coord) for coord in component])

    def _get_closest_pair(self,
                          grid: Grid,
                          component_0: Set[Coord],
                          component_1: Set[Coord]) -> Tuple[Coord, Coord]:
        min_dist = grid.width * grid.height
        min_pair = ((0, 0), (0, 0))
        for c0, c1 in itertools.product(component_0, component_1):
            d = grid.manhattan_dist(c0, c1)
            if d < min_dist:
                min_dist = d
                min_pair = (c0, c1)
        return min_pair

    def _get_shortest_direct_path(self,
                                  grid: Grid,
                                  start_coord: Coord,
                                  goal_coord: Coord) -> List[Coord]:
        """Get shortest direct path between two coords.

        This will possibly include blocks in the path, but will greedily chose
        paths that do not contain a block. There's no guarantee it'll be the
        direct path with the least number of blocks.
        """
        path = [start_coord]

        x_dir = 0
        if start_coord[0] != goal_coord[0]:
            x_dir = 1 if start_coord[0] < goal_coord[0] else -1

        y_dir = 0
        if start_coord[1] != goal_coord[1]:
            y_dir = 1 if start_coord[1] < goal_coord[1] else -1

        coord = start_coord
        while coord != goal_coord:
            next_coords = []
            if coord[0] != goal_coord[0]:
                next_coords.append((coord[0] + x_dir, coord[1]))
            if coord[1] != goal_coord[1]:
                next_coords.append((coord[0], coord[1] + y_dir))

            if len(next_coords) == 1:
                coord = next_coords[0]
                path.append(coord)
                continue

            blocked = [c in grid.block_coords for c in next_coords]
            if all(blocked) or not any(blocked):
                coord = self._rng.choice(next_coords)
            else:
                # if [0] is blocked, choose [1], and vice versa
                coord = next_coords[blocked[0]]
            path.append(coord)

        return path

    def get_grid_str(self, grid: Grid) -> str:
        """Get a str representation of a grid with mask applied."""
        grid_repr = []
        for row in range(grid.height):
            row_repr = []
            for col in range(grid.width):
                coord = (col, row)
                if coord in grid.block_coords:
                    row_repr.append("#")
                elif coord in self.mask:
                    row_repr.append("M")
                else:
                    row_repr.append(".")
            grid_repr.append(row_repr)

        return "\n".join(list(list((" ".join(r) for r in grid_repr))))


class GridCycler:
    """Class for handling cycling through a set of generated grids."""

    def __init__(self,
                 grids: List[Grid],
                 shuffle_each_cycle: bool,
                 seed: Optional[int] = None):
        self.grids = grids
        self.shuffle = shuffle_each_cycle
        self._rng = random.Random(seed)

        self._next_idx = 0

    def next(self) -> Grid:
        if self._next_idx >= len(self.grids):
            self._next_idx = 0

            if self.shuffle:
                self._rng.shuffle(self.grids)

        grid = self.grids[self._next_idx]
        self._next_idx += 1
        return grid
