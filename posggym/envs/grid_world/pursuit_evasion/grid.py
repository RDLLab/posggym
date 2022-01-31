"""A grid in the Runner Chaser Problem """
from queue import Queue
from typing import List, Set, Optional, Dict, Union

from posggym.envs.grid_world.utils import Grid, Coord, Direction


class PEGrid(Grid):
    """A grid for the Pursuit Evasion Problem """

    def __init__(self,
                 grid_width: int,
                 grid_height: int,
                 block_coords: Set[Coord],
                 goal_coords_map: Dict[Coord, List[Coord]],
                 runner_start_coords: List[Coord],
                 chaser_start_coords: List[Coord]):
        super().__init__(grid_width, grid_height, block_coords)
        self._goal_coords_map = goal_coords_map
        self.runner_start_coords = runner_start_coords
        self.chaser_start_coords = chaser_start_coords

    @property
    def all_goal_coords(self) -> List[Coord]:
        """The list of all runner goal locations """
        all_locs = set()
        for v in self._goal_coords_map.values():
            all_locs.update(v)
        return list(all_locs)

    def get_goal_coords(self, runner_start_coord: Coord) -> List[Coord]:
        """Get list of possible runner goal coords for given start coords """
        return self._goal_coords_map[runner_start_coord]

    def get_ascii_repr(self,
                       goal_coord: Union[None, Coord, List[Coord]],
                       runner_coord: Union[None, Coord, List[Coord]],
                       chaser_coord: Union[None, Coord, List[Coord]]) -> str:
        """Get ascii repr of grid """
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

        if runner_coord is None:
            runner_coord = []
        elif not isinstance(runner_coord, List):
            runner_coord = [runner_coord]

        for coord in runner_coord:
            grid_repr[coord[0]][coord[1]] = "R"

        if chaser_coord is None:
            chaser_coord = []
        elif not isinstance(chaser_coord, List):
            chaser_coord = [chaser_coord]

        for coord in chaser_coord:
            if coord in runner_coord:
                grid_repr[coord[0]][coord[1]] = "X"
            else:
                grid_repr[coord[0]][coord[1]] = "C"

        return (
            str(self)
            + "\n" + "\n".join(list(list((" ".join(r) for r in grid_repr))))
        )

    def get_init_ascii_repr(self) -> str:
        """Get ascii repr of initial grid """
        return self.get_ascii_repr(
            self.all_goal_coords,
            self.runner_start_coords,
            self.chaser_start_coords
        )

    def get_fov(self,
                origin: Coord,
                direction: Direction,
                widening_increment: int) -> Set[Coord]:
        """Get the Field of vision from origin looking in given direction

        Uses BFS starting from origin and expanding in the direction, while
        accounting for obstacles blocking the field of view,
        to get the field of vision.
        """
        assert widening_increment > 0
        fov = set([origin])

        frontier_queue: Queue[Coord] = Queue()
        frontier_queue.put(origin)
        visited = set([origin])

        while not frontier_queue.empty():
            coords = frontier_queue.get()

            for next_coords in self._get_fov_successors(
                    origin, direction, coords, widening_increment
            ):
                if next_coords not in visited:
                    visited.add(next_coords)
                    frontier_queue.put(next_coords)
                    fov.add(next_coords)
        return fov

    def _get_fov_successors(self,
                            origin: Coord,
                            direction: Direction,
                            coords: Coord,
                            widening_increment: int) -> List[Coord]:
        successors = []

        forward_successor = self._get_fov_successor(coords, direction)
        if forward_successor is not None:
            successors.append(forward_successor)
        else:
            return successors

        if not self._check_expand_fov(origin, coords, widening_increment):
            return successors

        side_successor: Optional[Coord] = None
        side_coords_list: List[Coord] = []
        if direction in [Direction.NORTH, Direction.SOUTH]:
            if 0 < coords[1] <= origin[1]:
                side_coords_list.append((coords[0], coords[1]-1))
            if origin[1] <= coords[1] < self.width-1:
                side_coords_list.append((coords[0], coords[1]+1))
        if direction in [Direction.EAST, Direction.WEST]:
            if 0 < coords[0] <= origin[0]:
                side_coords_list.append((coords[0]-1, coords[1]))
            elif origin[0] <= coords[0] < self.height-1:
                side_coords_list.append((coords[0]+1, coords[1]))

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

    @staticmethod
    def _check_expand_fov(origin: Coord,
                          coord: Coord,
                          widening_increment: int) -> bool:
        # Expands field of vision at depth 1 and
        # then every widening_increment depth
        d = max(abs(origin[0] - coord[0]), abs(origin[1] - coord[1]))
        return d == 1 or (d > 1 and d % widening_increment == 0)


def get_8x8_grid() -> PEGrid:
    """Generate the 8-by-8 PE grid layout.

    This is an approximate discrete version of the map layout from:
    Seaman et al 2018, 'Nested Reasoning About Autonomous Agents Using
    Probabilistic Programs'

    - 0, 1, 2, 7, 8, 9 are possible runner start and goal locations
    - 5, 6 are possible chaser start locations

    The runner start and goal locations are always on opposite sides of the
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

    - 0, 1, 2, 7, 8, 9 are possible runner start and goal locations
    - 5, 6 are possible chaser start locations

    The runner start and goal locations are always on opposite sides of the
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
        "  # 6        # #"
        "#   ## ##  ##  #"
        "##   #5##  #   #"
        "## #   #     # #"
        "   # #   ##  2 #"
        "## # #   ##   ##"
        "#0     # 1     #"
        "  #### ##  ##   "
        "###     ######  "
    )
    return _convert_map_to_grid(ascii_map, 16, 16)


def get_32x32_grid() -> PEGrid:
    """Generate the 32-by-32 PE grid layout.

    This is an approximate discrete version of the map layout from:
    Seaman et al 2018, 'Nested Reasoning About Autonomous Agents Using
    Probabilistic Programs'

    - 0, 1, 2, 7, 8, 9 are possible runner start and goal locations
    - 5, 6 are possible chaser start locations

    The runner start and goal locations are always on opposite sides of the
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
                         chaser_start_symbols: Optional[Set[str]] = None,
                         runner_start_symbols: Optional[Set[str]] = None,
                         runner_goal_symbol_map: Optional[Dict] = None
                         ) -> PEGrid:
    assert len(ascii_map) == height * width

    if chaser_start_symbols is None:
        chaser_start_symbols = set(['3', '4', '5', '6'])
    if runner_start_symbols is None:
        runner_start_symbols = set(['0', '1', '2', '7', '8', '9'])
    if runner_goal_symbol_map is None:
        runner_goal_symbol_map = {
            '0': ['7', '8', '9'],
            '1': ['7', '8', '9'],
            '2': ['8', '9'],
            '7': ['0', '1'],
            '8': ['0', '1', '2'],
            '9': ['0', '1', '2'],
        }

    block_coords = set()
    runner_start_coords = []
    chaser_start_coords = []
    runner_symbol_coord_map = {}

    for loc, symbol in enumerate(ascii_map):
        coord = _loc_to_coord(loc, width)
        if symbol == block_symbol:
            block_coords.add(coord)
        elif symbol in chaser_start_symbols:
            chaser_start_coords.append(coord)
        elif symbol in runner_start_symbols:
            runner_start_coords.append(coord)
            runner_symbol_coord_map[symbol] = coord

    runner_goal_coords_map = {}
    for start_symbol, goal_symbols in runner_goal_symbol_map.items():
        start_coord = runner_symbol_coord_map[start_symbol]
        runner_goal_coords_map[start_coord] = [
            runner_symbol_coord_map[symbol] for symbol in goal_symbols
        ]

    return PEGrid(
        grid_width=width,
        grid_height=height,
        block_coords=block_coords,
        goal_coords_map=runner_goal_coords_map,
        runner_start_coords=runner_start_coords,
        chaser_start_coords=chaser_start_coords
    )


SUPPORTED_GRIDS = {
    '8x8': get_8x8_grid,
    '16x16': get_16x16_grid,
    '32x32': get_32x32_grid
}


def load_grid(grid_name: str) -> PEGrid:
    """Load grid with given name """
    grid_name = grid_name.lower()
    assert grid_name in SUPPORTED_GRIDS, (
        f"Unsupported grid name '{grid_name}'. Grid name must be one of: "
        f"{SUPPORTED_GRIDS.keys()}."
    )
    return SUPPORTED_GRIDS[grid_name]()
