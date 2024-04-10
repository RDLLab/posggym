"""Script for manually running and visualizing DrivingGridGenerator."""

import argparse
import sys
from typing import Optional

from posggym.envs.grid_world.driving_gen import DrivingGridGenerator


def main(
    width: int,
    height: int,
    max_obstacle_size: Optional[int] = None,
    seed: int = 0,
):
    """Run."""
    if max_obstacle_size is None:
        max_obstacle_size = max(1, min(width, height) // 4)

    while True:
        n = -1
        while n < 0:
            try:
                n = int(input("Select max_num_obstacles (Ctrl-C to exit): "))
            except ValueError:
                pass
            except KeyboardInterrupt:
                print()
                sys.exit(1)

        grid_gen = DrivingGridGenerator(
            width, height, max_obstacle_size, max_num_obstacles=n, seed=seed
        )
        seed += 1
        grid = grid_gen.generate()
        print(grid_gen.get_grid_str(grid))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("width", type=int, help="Width of grid")
    parser.add_argument("height", type=int, help="Height of grid")
    parser.add_argument(
        "--max_obstacle_size",
        type=int,
        default=None,
        help="Max size of obstacle. If None then uses min(width, height) // 4",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random Seed")
    main(**vars(parser.parse_args()))
