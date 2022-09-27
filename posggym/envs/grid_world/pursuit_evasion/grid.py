"""A grid in the Runner Chaser Problem."""
from collections import deque
from typing import List, Set, Optional, Dict, Union

from posggym.envs.grid_world.core import Grid, Coord, Direction


class PEGrid(Grid):
    """A grid for the Pursuit Evasion Problem."""

    def __init__(self,
                 grid_width: int,
                 grid_height: int,
                 block_coords: Set[Coord],
                 goal_coords_map: Dict[Coord, List[Coord]],
                 evader_start_coords: List[Coord],
                 pursuer_start_coords: List[Coord]):
        super().__init__(grid_width, grid_height, block_coords)
        self._goal_coords_map = goal_coords_map
        self.evader_start_coords = evader_start_coords
        self.pursuer_start_coords = pursuer_start_coords

        self.shortest_paths = self.get_all_shortest_paths(
            set(evader_start_coords)
        )

    @property
    def all_goal_coords(self) -> List[Coord]:
        """The list of all evader goal locations."""
        all_locs = set()
        for v in self._goal_coords_map.values():
            all_locs.update(v)
        return list(all_locs)

    def get_goal_coords(self, evader_start_coord: Coord) -> List[Coord]:
        """Get list of possible evader goal coords for given start coords."""
        return self._goal_coords_map[evader_start_coord]

    def get_shortest_path_distance(self, coord: Coord, dest: Coord) -> int:
        """Get the shortest path distance from coord to destination."""
        return int(self.shortest_paths[dest][coord])

    def get_max_shortest_path_distance(self) -> int:
        """Get max shortest path distance between any start and goal coords."""
        max_dist = 0
        for start_coord, dest_coords in self._goal_coords_map.items():
            max_dist = max(
                max_dist,
                int(max([
                    self.get_shortest_path_distance(start_coord, dest)
                    for dest in dest_coords
                ]))
            )
        return max_dist

    def get_ascii_repr(self,
                       goal_coord: Union[None, Coord, List[Coord]],
                       evader_coord: Union[None, Coord, List[Coord]],
                       pursuer_coord: Union[None, Coord, List[Coord]]) -> str:
        """Get ascii repr of grid."""
        if goal_coord is None:
            goal_coords = set()
        elif isinstance(goal_coord, list):
            goal_coords = set(goal_coord)
        else:
            goal_coords = set([goal_coord])

        grid_repr = []
        for row in range(self.height):
            row_repr = []
            for col in range(self.width):
                coord = (col, row)
                if coord in goal_coords:
                    row_repr.append("G")
                elif coord in self.block_coords:
                    row_repr.append("#")
                else:
                    row_repr.append(".")
            grid_repr.append(row_repr)

        if evader_coord is None:
            evader_coord = []
        elif not isinstance(evader_coord, List):
            evader_coord = [evader_coord]

        for coord in evader_coord:
            grid_repr[coord[1]][coord[0]] = "R"

        if pursuer_coord is None:
            pursuer_coord = []
        elif not isinstance(pursuer_coord, List):
            pursuer_coord = [pursuer_coord]

        for coord in pursuer_coord:
            if coord in evader_coord:
                grid_repr[coord[1]][coord[0]] = "X"
            else:
                grid_repr[coord[1]][coord[0]] = "C"

        return "\n".join(list(list((" ".join(r) for r in grid_repr))))

    def get_init_ascii_repr(self) -> str:
        """Get ascii repr of initial grid."""
        return self.get_ascii_repr(
            self.all_goal_coords,
            self.evader_start_coords,
            self.pursuer_start_coords
        )

    def get_fov(self,
                origin: Coord,
                direction: Direction,
                widening_increment: int,
                max_depth: int) -> Set[Coord]:
        """Get the Field of vision from origin looking in given direction.

        Uses BFS starting from origin and expanding in the direction, while
        accounting for obstacles blocking the field of view,
        to get the field of vision.
        """
        assert widening_increment > 0
        assert max_depth > 0
        fov = set([origin])

        frontier_queue: deque[Coord] = deque([origin])
        visited = set([origin])

        while len(frontier_queue):
            coord = frontier_queue.pop()

            for next_coord in self._get_fov_successors(
                    origin, direction, coord, widening_increment, max_depth
            ):
                if next_coord not in visited:
                    visited.add(next_coord)
                    frontier_queue.append(next_coord)
                    fov.add(next_coord)
        return fov

    def _get_fov_successors(self,
                            origin: Coord,
                            direction: Direction,
                            coord: Coord,
                            widening_increment: int,
                            max_depth: int) -> List[Coord]:
        if direction in [Direction.NORTH, Direction.SOUTH]:
            depth = abs(origin[1] - coord[1])
        else:
            depth = abs(origin[0] - coord[0])

        if depth >= max_depth:
            return []

        successors = []
        forward_successor = self._get_fov_successor(coord, direction)
        if forward_successor is None:
            return []
        else:
            successors.append(forward_successor)

        # Expands FOV at depth 1 and then every widening_increment depth
        if depth != 1 and depth % widening_increment != 0:
            # Don't expand sideways
            return successors

        side_coords_list: List[Coord] = []

        if direction in [Direction.NORTH, Direction.SOUTH]:
            if 0 < coord[0] <= origin[0]:
                side_coords_list.append((coord[0]-1, coord[1]))
            if origin[0] <= coord[0] < self.width-1:
                side_coords_list.append((coord[0]+1, coord[1]))
        else:
            if 0 < coord[1] <= origin[1]:
                side_coords_list.append((coord[0], coord[1]-1))
            elif origin[1] <= coord[1] < self.height-1:
                side_coords_list.append((coord[0], coord[1]+1))

        side_successor: Optional[Coord] = None
        for side_coord in side_coords_list:
            if side_coord in self.block_coords:
                continue

            side_successor = self._get_fov_successor(side_coord, direction)
            if side_successor is not None:
                successors.append(side_successor)

        return successors

    def _get_fov_successor(self,
                           coord: Coord,
                           direction: Direction) -> Optional[Coord]:
        new_coord = self.get_next_coord(coord, direction, ignore_blocks=False)
        if new_coord == coord:
            # move in given direction is blocked or out-of-bounds
            return None
        return new_coord

    def get_max_fov_width(self,
                          widening_increment: int,
                          max_depth: int) -> int:
        """Get the maximum width of field of vision."""
        max_width = 1
        for d in range(max_depth):
            if d == 1 or d % widening_increment == 0:
                max_width += 2
        return min(max_depth, min(self.width, self.height))


def get_5x5_grid() -> PEGrid:
    """Generate the 5-by-5 PE grid layout.

    - 0, 1, 2, 3 are possible evader start and goal locations
    - 5 are possible pursuer start locations

    The evader start and goal locations are always on opposite sides of the
    map.

    """
    ascii_map = (
        " 9  8"
        " # # "
        " #5  "
        " # # "
        "0  1 "
    )

    return _convert_map_to_grid(
        ascii_map,
        5,
        5,
        pursuer_start_symbols=set(['5']),
        evader_start_symbols=set(['0', '1', '8', '9']),
        evader_goal_symbol_map={
            '0': ['8', '9'],
            '1': ['8', '9'],
            '8': ['0', '1'],
            '9': ['0', '1'],
        }
    )


def get_8x8_grid() -> PEGrid:
    """Generate the 8-by-8 PE grid layout.

    This is an approximate discrete version of the map layout from:
    Seaman et al 2018, 'Nested Reasoning About Autonomous Agents Using
    Probabilistic Programs'

    - 0, 1, 2, 7, 8, 9 are possible evader start and goal locations
    - 5, 6 are possible pursuer start locations

    The evader start and goal locations are always on opposite sides of the
    map.
    """
    ascii_map = (
        "  9 #8 #"
        "# #    #"
        "# ## # 7"
        "  6   # "
        "# #5# # "
        "   #    "
        "#    #2 "
        "0 #1### "
    )

    return _convert_map_to_grid(ascii_map, 8, 8)


def get_16x16_grid() -> PEGrid:
    """Generate the 16-by-16 PE grid layout.

    This is an approximate discrete version of the map layout from:
    Seaman et al 2018, 'Nested Reasoning About Autonomous Agents Using
    Probabilistic Programs'

    - 0, 1, 2, 7, 8, 9 are possible evader start and goal locations
    - 5, 6 are possible pursuer start locations

    The evader start and goal locations are always on opposite sides of the
    map.

    """
    ascii_map = (
        "  ## ####### ###"
        "    9##    8 ###"
        "##       # # ###"
        "## ## ##      ##"
        "## ## ##   ## ##"
        "#  #  ##   ## 7#"
        "# ## ###   #   #"
        "  #  6       # #"
        "#   ## ##  ##  #"
        "##   #5##  #   #"
        "## #   #     # #"
        "   # #   ##    #"
        " # # ##  ##   ##"
        "0#       #     #"
        "   ### #   ##2  "
        "###    1 #####  "
    )
    return _convert_map_to_grid(ascii_map, 16, 16)


def get_32x32_grid() -> PEGrid:
    """Generate the 32-by-32 PE grid layout.

    This is an approximate discrete version of the map layout from:
    Seaman et al 2018, 'Nested Reasoning About Autonomous Agents Using
    Probabilistic Programs'

    - 0, 1, 2, 7, 8, 9 are possible evader start and goal locations
    - 5, 6 are possible pursuer start locations

    The evader start and goal locations are always on opposite sides of the
    map.

    """
    ascii_map = (
        "       #  ########### #         "
        "   #####  ########### #  #######"
        "      #   ######      8  #######"
        " #       9#               ######"
        " ##              #### ##  ######"
        " ##   #####  ##  #### ##   #####"
        " ##   #####  ##  ##        #####"
        "      ##### ####      ###  #####"
        " ### ###### ####      ###   ####"
        " ### #####  ####      ####  ####"
        " ##  ##### #####      ##### 7###"
        " ##  ####  #####       ####  ###"
        " #  #####  #####       ####  ###"
        " #  ##### ######       #     ###"
        " #       6               ##  ###"
        "    #       #   ##      ####  ##"
        "     #   ####  ###     ####   ##"
        "#### #   ####  ###    ####    ##"
        "#### #   ### 5####   ####     ##"
        "####  #      ####     ##  #   ##"
        "##### #      ####        ##   ##"
        "#                          ##  #"
        "         ###        ##    2 #  #"
        "  ###### ###      #### ## #    #"
        "########  ##      ####    #  #  "
        "########  ##       ####     ####"
        "###          ##   1##        ###"
        "          #  ####       #       "
        "   0  ###### ######   ####      "
        "  ########## #        #####     "
        "##########      ############    "
        "#                               "
    )
    return _convert_map_to_grid(ascii_map, 32, 32)


def _loc_to_coord(loc: int, grid_width: int) -> Coord:
    return (loc % grid_width, loc // grid_width)


def _convert_map_to_grid(ascii_map: str,
                         height: int,
                         width: int,
                         block_symbol: str = "#",
                         pursuer_start_symbols: Optional[Set[str]] = None,
                         evader_start_symbols: Optional[Set[str]] = None,
                         evader_goal_symbol_map: Optional[Dict] = None
                         ) -> PEGrid:
    assert len(ascii_map) == height * width

    if pursuer_start_symbols is None:
        pursuer_start_symbols = set(['3', '4', '5', '6'])
    if evader_start_symbols is None:
        evader_start_symbols = set(['0', '1', '2', '7', '8', '9'])
    if evader_goal_symbol_map is None:
        evader_goal_symbol_map = {
            '0': ['7', '8', '9'],
            '1': ['7', '8', '9'],
            '2': ['8', '9'],
            '7': ['0', '1'],
            '8': ['0', '1', '2'],
            '9': ['0', '1', '2'],
        }

    block_coords = set()
    evader_start_coords = []
    pursuer_start_coords = []
    evader_symbol_coord_map = {}

    for loc, symbol in enumerate(ascii_map):
        coord = _loc_to_coord(loc, width)
        if symbol == block_symbol:
            block_coords.add(coord)
        elif symbol in pursuer_start_symbols:
            pursuer_start_coords.append(coord)
        elif symbol in evader_start_symbols:
            evader_start_coords.append(coord)
            evader_symbol_coord_map[symbol] = coord

    evader_goal_coords_map = {}
    for start_symbol, goal_symbols in evader_goal_symbol_map.items():
        start_coord = evader_symbol_coord_map[start_symbol]
        evader_goal_coords_map[start_coord] = [
            evader_symbol_coord_map[symbol] for symbol in goal_symbols
        ]

    return PEGrid(
        grid_width=width,
        grid_height=height,
        block_coords=block_coords,
        goal_coords_map=evader_goal_coords_map,
        evader_start_coords=evader_start_coords,
        pursuer_start_coords=pursuer_start_coords
    )


# grid_name: (grid_make_fn, step_limit)
SUPPORTED_GRIDS = {
    '5x5': (get_5x5_grid, 20),
    '8x8': (get_8x8_grid, 50),
    '16x16': (get_16x16_grid, 100),
    '32x32': (get_32x32_grid, 200)
}


def load_grid(grid_name: str) -> PEGrid:
    """Load grid with given name."""
    grid_name = grid_name.lower()
    assert grid_name in SUPPORTED_GRIDS, (
        f"Unsupported grid name '{grid_name}'. Grid name must be one of: "
        f"{SUPPORTED_GRIDS.keys()}."
    )
    return SUPPORTED_GRIDS[grid_name][0]()
