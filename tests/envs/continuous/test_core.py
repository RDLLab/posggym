"""Tests for the core functionality of continuous envs."""

import pytest

from posggym.envs.continuous import core


@pytest.mark.parametrize(
    "test_values",
    [
        (
            "...\n.#.\n...",
            [
                ((1, 1), (2, 1)),
                ((1, 1), (1, 2)),
                ((2, 1), (2, 2)),
                ((1, 2), (2, 2)),
            ],
        ),
        (
            "..##..\n" "...#..\n" "......\n" "......\n" "...###\n" "...###\n",
            [
                ((2, 0), (2, 1)),
                ((2, 1), (3, 1)),
                ((3, 1), (3, 2)),
                ((3, 2), (4, 2)),
                ((4, 0), (4, 2)),
                ((3, 4), (6, 4)),
                ((3, 4), (3, 6)),
            ],
        ),
    ],
)
def test_parse_world_str_interior_walls(test_values):
    world_str, expected = test_values
    actual = core.parse_world_str_interior_walls(world_str)

    expected_map = {}
    for line in expected:
        if line[0] not in expected_map:
            expected_map[line[0]] = set()
        expected_map[line[0]].add(line[1])
        # add line in both directions since direction doesn't matter
        if line[1] not in expected_map:
            expected_map[line[1]] = set()
        expected_map[line[1]].add(line[0])

    for line in actual:
        l_start, l_end = (
            (int(line[0][0]), int(line[0][1])),
            ((int(line[1][0]), int(line[1][1]))),
        )
        assert (l_start in expected_map and l_end in expected_map[l_start]) or (
            l_end in expected_map and l_start in expected_map[l_end]
        )


if __name__ == "__main__":
    tvals = (
        # "..##..\n" "...#..\n" "......\n" "......\n" "...###\n" "...###\n",
        "..##..\n" "...#..\n" "......\n" "......\n" "......\n" "......\n",
        [
            ((2, 0), (2, 1)),
            ((2, 1), (3, 1)),
            ((3, 1), (3, 2)),
            ((3, 2), (4, 2)),
            ((4, 0), (4, 2)),
            # ((3, 4), (6, 4)),
            # ((3, 4), (3, 6)),
        ],
    )
    test_parse_world_str_interior_walls(tvals)
