"""A grid in the Driving Problem."""
from typing import Set, List, Dict
from itertools import product

from posggym.envs.grid_world.core import (
    Grid, Coord, Direction, DIRECTION_ASCII_REPR
)


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

        self.shortest_paths = self.get_all_shortest_paths(
            set.union(*dest_coords)
        )

    @property
    def supported_num_agents(self) -> int:
        """Get the number of agents supported by this grid."""
        return len(self.start_coords)

    def get_shortest_path_distance(self, coord: Coord, dest: Coord) -> int:
        """Get the shortest path distance from coord to destination."""
        return int(self.shortest_paths[dest][coord])

    def get_max_shortest_path_distance(self) -> int:
        """Get the longest shortest path distance to any destination."""
        return int(
            max([max(d.values()) for d in self.shortest_paths.values()])
        )

    def get_ascii_repr(self,
                       vehicle_coords: List[Coord],
                       vehicle_dirs: List[Direction],
                       vehicle_dests: List[Coord]) -> str:
        """Get ascii repr of grid."""
        grid_repr = []
        for row in range(self.height):
            row_repr = []
            for col in range(self.width):
                coord = (col, row)
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
        coord = (c, r)
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


def get_3x3_grid() -> DrivingGrid:
    """Generate a simple Driving 3-by-3 grid layout."""
    grid_str = (
        "a1.\n"
        ".#.\n"
        ".0b\n"
    )
    return parse_grid_str(grid_str, 2)


def get_4x4_intersection_grid() -> DrivingGrid:
    """Generate a 4-by-4 intersection grid layout."""
    grid_str = (
        "#0b#\n"
        "d..3\n"
        "2..c\n"
        "#a1#\n"
    )
    return parse_grid_str(grid_str, 4)


def get_6x6_intersection_grid() -> DrivingGrid:
    """Generate a 6-by-6 intersection grid layout."""
    grid_str = (
        "##0b##\n"
        "##..##\n"
        "d....3\n"
        "2....c\n"
        "##..##\n"
        "##a1##\n"
    )
    return parse_grid_str(grid_str, 4)


def get_7x7_crisscross_grid() -> DrivingGrid:
    """Generate a 7-by-7 Criss-Cross grid layout."""
    grid_str = (
        "#-#-#-#\n"
        "-.....+\n"
        "#.#.#.#\n"
        "-.....+\n"
        "#.#.#.#\n"
        "-.....+\n"
        "#+#+#+#\n"
    )
    return parse_grid_str(grid_str, 6)


def get_7x7_crisscross2_grid() -> DrivingGrid:
    """Generate a 7-by-7 Criss-Cross v2 grid layout."""
    grid_str = (
        "#-#-#-#\n"
        "-.....+\n"
        "#.###.#\n"
        "-.....+\n"
        "#.###.#\n"
        "-.....+\n"
        "#+#+#+#\n"
    )
    return parse_grid_str(grid_str, 6)


def get_7x7_crisscross3_grid() -> DrivingGrid:
    """Generate a 7-by-7 Criss-Cross v3 grid layout."""
    grid_str = (
        "#-#-#-#\n"
        "-.....+\n"
        "#.#.#.#\n"
        "-.#.#.+\n"
        "#.#.#.#\n"
        "-.....+\n"
        "#+#+#+#\n"
    )
    return parse_grid_str(grid_str, 6)


def get_7x7_crisscross4_grid() -> DrivingGrid:
    """Generate a 7-by-7 Criss-Cross v4 grid layout."""
    grid_str = (
        "#-#-#-#\n"
        "-.#.#.+\n"
        "#.#.#.#\n"
        "-.....+\n"
        "#.#.#.#\n"
        "-.#.#.+\n"
        "#+#+#+#\n"
    )
    return parse_grid_str(grid_str, 6)


def get_7x7_crisscross5_grid() -> DrivingGrid:
    """Generate a 7-by-7 Criss-Cross v5 grid layout."""
    grid_str = (
        "#-#-#-#\n"
        "-.....+\n"
        "###.###\n"
        "-.....+\n"
        "###.###\n"
        "-.....+\n"
        "#+#+#+#\n"
    )
    return parse_grid_str(grid_str, 6)


def get_7x7_blocks_grid() -> DrivingGrid:
    """Generate a 7-by-7 blocks grid layout."""
    grid_str = (
        "#-...-#\n"
        "-##.##+\n"
        ".##.##.\n"
        ".......\n"
        ".##.##.\n"
        "-##.##+\n"
        "#+...+#\n"
    )
    return parse_grid_str(grid_str, 4)


def get_7x7_roundabout_grid() -> DrivingGrid:
    """Generate a 7-by-7 round-about grid layout."""
    grid_str = (
        "#-...-#\n"
        "-##.##+\n"
        ".#...#.\n"
        "...#...\n"
        ".#...#.\n"
        "-##.##+\n"
        "#+...+#\n"
    )
    return parse_grid_str(grid_str, 4)


def get_14x14_crisscross_grid() -> DrivingGrid:
    """Generate a 14-by-14 Criss-Cross grid layout."""
    grid_str = (
        "##-##-##-##-##\n"
        "##.##.##.##.##\n"
        "-............+\n"
        "##.##.##.##.##\n"
        "##.##.##.##.##\n"
        "-............+\n"
        "##.##.##.##.##\n"
        "##.##.##.##.##\n"
        "-............+\n"
        "##.##.##.##.##\n"
        "##.##.##.##.##\n"
        "-............+\n"
        "##.##.##.##.##\n"
        "##+##+##+##+##\n"
    )
    return parse_grid_str(grid_str, 4)


def get_14x14_blocks_grid() -> DrivingGrid:
    """Generate a 14-by-14 Blocks grid layout."""
    grid_str = (
        "#-..........-#\n"
        "-###.####.###+\n"
        ".###.####.###.\n"
        ".###.####.###.\n"
        "..............\n"
        ".###.####.###.\n"
        ".###.####.###.\n"
        ".###.####.###.\n"
        ".###.####.###.\n"
        "..............\n"
        ".###.####.###.\n"
        ".###.####.###.\n"
        "-###.####.###+\n"
        "#+..........+#\n"
    )
    return parse_grid_str(grid_str, 4)


def get_14x14_roundabout_grid() -> DrivingGrid:
    """Generate a 14-by-14 Round About grid layout."""
    grid_str = (
        "#-..........-#\n"
        "-######.#####+\n"
        ".######.#####.\n"
        ".######.#####.\n"
        ".######.#####.\n"
        ".####.....###.\n"
        ".####.### ###.\n"
        "......###.....\n"
        ".####.###.###.\n"
        ".####.....###.\n"
        ".######.#####.\n"
        ".###### #####.\n"
        "-######.#####+\n"
        "#+..........+#\n"
    )
    return parse_grid_str(grid_str, 4)


def get_14x14_roundabout_wide_grid() -> DrivingGrid:
    """Generate a 14-by-14 Round About grid layout with wide roads."""
    grid_str = (
        "#-..........-#\n"
        "-#####..#####+\n"
        ".#####..#####.\n"
        ".#####..#####.\n"
        ".###......###.\n"
        ".###......###.\n"
        "......##......\n"
        "......##......\n"
        ".###......###.\n"
        ".###......###.\n"
        ".#####..#####.\n"
        ".#####..#####.\n"
        "-#####..#####+\n"
        "#+..........+#\n"
    )
    return parse_grid_str(grid_str, 4)


#  (grid_make_fn, finite horizon step_limit, infinite horizon step_limit)
# inifinite horizon = cars reset when goal reached rather than epsidoe ends
SUPPORTED_GRIDS = {
    '3x3': (get_3x3_grid, 30, 100),
    '4x4Intersection': (get_4x4_intersection_grid, 20, 100),
    '6x6Intersection': (get_6x6_intersection_grid, 20, 100),
    '7x7CrissCross1': (get_7x7_crisscross_grid, 50, 200),
    '7x7CrissCross2': (get_7x7_crisscross2_grid, 50, 200),
    '7x7CrissCross3': (get_7x7_crisscross3_grid, 50, 200),
    '7x7CrissCross4': (get_7x7_crisscross4_grid, 50, 200),
    '7x7CrissCross5': (get_7x7_crisscross5_grid, 50, 200),
    '7x7Blocks': (get_7x7_blocks_grid, 50, 200),
    '7x7RoundAbout': (get_7x7_roundabout_grid, 50, 200),
    '14x14Blocks': (get_14x14_blocks_grid, 100, 200),
    '14x14CrissCross': (get_14x14_crisscross_grid, 100, 200),
    '14x14RoundAbout': (get_14x14_roundabout_grid, 100, 200),
    '14x14WideRoundAbout': (get_14x14_roundabout_wide_grid, 50, 200),
}


def load_grid(grid_name: str) -> DrivingGrid:
    """Load grid with given name."""
    grid_name = grid_name.lower()
    assert grid_name in SUPPORTED_GRIDS, (
        f"Unsupported grid name '{grid_name}'. Grid name must be one of: "
        f"{SUPPORTED_GRIDS.keys()}."
    )
    return SUPPORTED_GRIDS[grid_name][0]()
