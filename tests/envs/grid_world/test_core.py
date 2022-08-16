from posggym.envs.grid_world.core import Grid, Direction


class TestGrid:
    """Tests for Grid class."""

    def _get_empty_grid(self, width: int, height: int) -> Grid:
        return Grid(width, height, set())

    def test_empty_grid(self):
        """Run basic grid tests.

        ...
        ...
        ...

        """
        width, height = (3, 3)
        grid = self._get_empty_grid(width, height)

        assert grid.n_coords == 9
        assert len(grid.all_coords) == 9
        assert len(grid.unblocked_coords) == 9

        assert grid.manhattan_dist((0, 0), (2, 2)) == 4
        assert all(
            grid.coord_in_bounds(c) for c in [(0, 0), (0, 2), (2, 0), (2, 2)]
        )
        assert all(
            not grid.coord_in_bounds(c)
            for c in [(-1, 1), (1, -1), (3, 1), (1, 3)]
        )

    def test_get_neighbours(self):
        """Test Grid.get_neighbours function."""
        width, height = (3, 3)
        block_coords = set([(1, 1)])
        grid = Grid(width, height, block_coords)

        # test with including out of bounds and ignoring blocks
        expected_neighbours_map = {
            (0, 0): [(-1, 0), (0, -1), (0, 1), (1, 0)],
            (1, 1): [(0, 1), (1, 0), (1, 2), (2, 1)],
            (2, 2): [(1, 2), (2, 1), (3, 2), (2, 3)]
        }
        for origin, expected_neighbours in expected_neighbours_map.items():
            neighbours = grid.get_neighbours(
                origin, ignore_blocks=True, include_out_of_bounds=True
            )
            assert not set(neighbours).difference(expected_neighbours)

        # test with excluding out of bounds but still ignoring blocks
        expected_neighbours_map = {
            (0, 0): [(0, 1), (1, 0)],
            (1, 1): [(0, 1), (1, 0), (1, 2), (2, 1)],
            (2, 2): [(1, 2), (2, 1)]
        }
        for origin, expected_neighbours in expected_neighbours_map.items():
            neighbours = grid.get_neighbours(
                origin, ignore_blocks=True, include_out_of_bounds=False
            )
            assert not set(neighbours).difference(expected_neighbours)

        # test with excluding out of bounds and not ignoring blocks
        expected_neighbours_map = {
            (0, 0): [(0, 1), (1, 0)],
            (1, 1): [(0, 1), (1, 0), (1, 2), (2, 1)],
            (2, 2): [(1, 2), (2, 1)],
            (1, 2): [(0, 2), (2, 2)]
        }
        for origin, expected_neighbours in expected_neighbours_map.items():
            neighbours = grid.get_neighbours(
                origin, ignore_blocks=False, include_out_of_bounds=False
            )
            assert not set(neighbours).difference(expected_neighbours)

    def test_get_next_coord(self):
        width, height = (3, 3)
        block_coords = set([(1, 1)])
        grid = Grid(width, height, block_coords)

        expected_map = {
            ((1, 1), Direction.NORTH, True): (1, 0),
            ((1, 1), Direction.SOUTH, True): (1, 2),
            ((1, 1), Direction.EAST, True): (2, 1),
            ((1, 1), Direction.WEST, True): (0, 1),
            ((0, 0), Direction.NORTH, True): (0, 0),
            ((0, 0), Direction.WEST, True): (0, 0),
            ((2, 2), Direction.SOUTH, True): (2, 2),
            ((2, 2), Direction.EAST, True): (2, 2),
            ((1, 0), Direction.SOUTH, True): (1, 1),
            ((1, 0), Direction.SOUTH, False): (1, 0),
            ((1, 2), Direction.NORTH, True): (1, 1),
            ((1, 2), Direction.NORTH, False): (1, 2),
            ((0, 1), Direction.EAST, True): (1, 1),
            ((0, 1), Direction.EAST, False): (0, 1),
            ((2, 1), Direction.WEST, True): (1, 1),
            ((2, 1), Direction.WEST, False): (2, 1),
        }
        for (origin, move_dir, ignore_blocks), exp in expected_map.items():
            actual_coord = grid.get_next_coord(origin, move_dir, ignore_blocks)
            assert actual_coord == exp

    def test_get_connected_components(self):
        """Test Grid.get_connected_components function.

        Test grids:

        #...    #...
        .#..    .#..
        ....    ..#.
        ...#    ...#

        """
        single_component_grid = Grid(4, 4, set([(0, 0), (1, 1), (3, 3)]))
        two_component_grid = Grid(4, 4, set([(0, 0), (1, 1), (2, 2), (3, 3)]))

        components = single_component_grid.get_connected_components()
        unblocked_coords = single_component_grid.unblocked_coords
        assert len(components) == 1
        assert not components[0].difference(unblocked_coords)

        components = two_component_grid.get_connected_components()
        unblocked_coords = two_component_grid.unblocked_coords
        exp_component_0 = set([(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)])
        exp_component_1 = set(unblocked_coords).difference(exp_component_0)
        assert len(components) == 2
        assert len(components[0].difference(exp_component_0)) == 0
        assert len(components[1].difference(exp_component_1)) == 0
