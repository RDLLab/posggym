"""A grid in the Driving Problem."""
from typing import Set, List, Dict
from itertools import product

from posggym.envs.grid_world.utils import (
    Grid, Coord, Direction, DIRECTION_ASCII_REPR
)

# TODO
# 1. function for generating a grid
# 2. function which returns a collection of grids or a grid python generator
#    which is deterministic given rng_key


class DrivingGrid(Grid):
    """A grid for the Driving Problem."""

    def __init__(self,
                 grid_width: int,
                 grid_height: int,
                 block_coords: Set[Coord],
                 start_coords: List[Set[Coord]],
                 dest_coords: List[Set[Coord]]):
        super().__init__(grid_width, grid_height, block_coords)
        assert len(start_coords) == len(dest_coords)
        self.start_coords = start_coords
        self.dest_coords = dest_coords

    @property
    def supported_num_agents(self) -> int:
        """Get the number of agents supported by this grid."""
        return len(self.start_coords)

    def get_ascii_repr(self,
                       vehicle_coords: List[Coord],
                       vehicle_dirs: List[Direction],
                       vehicle_dests: List[Coord]) -> str:
        """Get ascii repr of grid."""
        grid_repr = []
        for row in range(self.height):
            row_repr = []
            for col in range(self.width):
                coord = (row, col)
                if coord in self.block_coords:
                    row_repr.append("#")
                elif coord in vehicle_dests:
                    row_repr.append("D")
                else:
                    row_repr.append(".")
            grid_repr.append(row_repr)

        for coord, direction in zip(vehicle_coords, vehicle_dirs):
            grid_repr[coord[0]][coord[1]] = DIRECTION_ASCII_REPR[direction]

        return "\n".join(list(list((" ".join(r) for r in grid_repr))))


def parse_grid_str(grid_str: str, supported_num_agents: int) -> DrivingGrid:
    """Parse a str representation of a grid.

    Notes on grid str representation:

    . = empty/unallocated cell
    # = a block
    0, 1, ..., 9 = starting location for agent with given index
    + = starting point for any agent
    a, b, ..., j = destination location for agent with given index
                   (a=0, b=1, ..., j=9)
    - = destination location for any agent

    Examples (" " quotes and newline chars ommited):

    1. A 3x3 grid with two agents, one block, and where each agent has a single
    starting location and a single destination location.

    a1.
    .#.
    .0.

    2. A 6x6 grid with 4 common start and destination locations and many
       blocks. This grid can support up to 4 agents.

    +.##+#
    ..+#.+
    #.###.
    #.....
    -..##.
    #-.-.-

    """
    row_strs = grid_str.splitlines()
    assert len(row_strs) > 1
    assert all(len(row) == len(row_strs[0]) for row in row_strs)
    assert len(row_strs[0]) > 1

    grid_height = len(row_strs)
    grid_width = len(row_strs[0])

    agent_start_chars = set(["+"] + [str(i) for i in range(10)])
    agent_dest_chars = set(["-"] + [c for c in "abcdefghij"])

    block_coords: Set[Coord] = set()
    shared_start_coords: Set[Coord] = set()
    agent_start_coords_map: Dict[int, Set[Coord]] = {}
    shared_dest_coords: Set[Coord] = set()
    agent_dest_coords_map: Dict[int, Set[Coord]] = {}
    for r, c in product(range(grid_height), range(grid_width)):
        coord = (r, c)
        char = row_strs[r][c]

        if char == "#":
            block_coords.add(coord)
        elif char in agent_start_chars:
            if char != "+":
                agent_id = int(char)
                if agent_id not in agent_start_coords_map:
                    agent_start_coords_map[agent_id] = set()
                agent_start_coords_map[agent_id].add(coord)
            else:
                shared_start_coords.add(coord)
        elif char in agent_dest_chars:
            if char != "-":
                agent_id = ord(char) - ord('a')
                if agent_id not in agent_dest_coords_map:
                    agent_dest_coords_map[agent_id] = set()
                agent_dest_coords_map[agent_id].add(coord)
            else:
                shared_dest_coords.add(coord)

    assert (
        len(shared_start_coords) + len(agent_start_coords_map)
        >= supported_num_agents
    )
    assert (
        len(shared_dest_coords) + len(agent_dest_coords_map)
        >= supported_num_agents
    )

    included_agent_ids = list(set(
        [*agent_start_coords_map, *agent_dest_coords_map]
    ))
    if len(included_agent_ids) > 0:
        assert max(included_agent_ids) < supported_num_agents

    start_coords: List[Set[Coord]] = []
    dest_coords: List[Set[Coord]] = []
    for i in range(supported_num_agents):
        agent_start_coords = set(shared_start_coords)
        agent_start_coords.update(agent_start_coords_map.get(i, {}))
        start_coords.append(agent_start_coords)

        agent_dest_coords = set(shared_dest_coords)
        agent_dest_coords.update(agent_dest_coords_map.get(i, {}))
        dest_coords.append(agent_dest_coords)

    return DrivingGrid(
        grid_width=grid_width,
        grid_height=grid_height,
        block_coords=block_coords,
        start_coords=start_coords,
        dest_coords=dest_coords
    )


def get_3x3_static_grid() -> DrivingGrid:
    """Generate a simple Driving 3-by-3 grid layout."""
    grid_str = (
        "a1.\n"
        ".#b\n"
        ".0.\n"
    )
    return parse_grid_str(grid_str, 2)


def get_6x6_static_grid() -> DrivingGrid:
    """Generate a static Driving 6-by-6 grid layout."""
    grid_str = (
        "+.##+#\n"
        "..+#.+\n"
        "#.###.\n"
        "#.....\n"
        "-..##.\n"
        "#-.-.-\n"
    )
    return parse_grid_str(grid_str, 4)


def get_7x7_criss_cross_grid() -> DrivingGrid:
    """Generate a 7-by-7 Criss-Cross grid layout."""
    grid_str = (
        "#a#0#a#\n"
        "b.....1\n"
        "#.#.#.#\n"
        "1.....b\n"
        "#.#.#.#\n"
        "b.....1\n"
        "#0#a#0#\n"
    )
    return parse_grid_str(grid_str, 2)


def get_7x7_criss_cross_grid_paired() -> DrivingGrid:
    """Generate a 7-by-7 Criss-Cross grid layout."""
    grid_str = (
        "#c#0#e#\n"
        "d.....3\n"
        "#.#.#.#\n"
        "1.....b\n"
        "#.#.#.#\n"
        "f.....5\n"
        "#2#a#4#\n"
    )
    return parse_grid_str(grid_str, 6)


def get_7x7_criss_cross_grid_shared() -> DrivingGrid:
    """Generate a 7-by-7 Criss-Cross grid layout."""
    grid_str = (
        "#-#+#-#\n"
        "-.....+\n"
        "#.#.#.#\n"
        "+.....-\n"
        "#.#.#.#\n"
        "-.....+\n"
        "#+#-#+#\n"
    )
    return parse_grid_str(grid_str, 6)


#  (grid_make_fn, finite horizon step_limit, infinite horizon step_limit)
SUPPORTED_GRIDS = {
    '3x3Static': (get_3x3_static_grid, 30, 100),
    '6x6Static': (get_6x6_static_grid, 60, 100),
    '7x7CrissCross': (get_7x7_criss_cross_grid, 70, 200),
    '7x7CrissCrossPaired': (get_7x7_criss_cross_grid_paired, 70, 200),
    '7x7CrissCrossShared': (get_7x7_criss_cross_grid_shared, 70, 200),
}


def load_grid(grid_name: str) -> DrivingGrid:
    """Load grid with given name."""
    grid_name = grid_name.lower()
    assert grid_name in SUPPORTED_GRIDS, (
        f"Unsupported grid name '{grid_name}'. Grid name must be one of: "
        f"{SUPPORTED_GRIDS.keys()}."
    )
    return SUPPORTED_GRIDS[grid_name][0]()
