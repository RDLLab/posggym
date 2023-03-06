"""Script for manually running and visualizing DrivingGridGenerator."""
import argparse
import sys

from posggym.envs.grid_world.driving.gen import DrivingGridGenerator


def main(args):
    """Run."""
    if args.max_obstacle_size is None:
        max_obstacle_size = max(1, min(args.width, args.height) // 4)
    else:
        max_obstacle_size = args.max_obstacle_size

    seed = args.seed
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
            args.width,
            args.height,
            max_obstacle_size,
            max_num_obstacles=n,
            seed=seed
        )
        seed += 1
        grid = grid_gen.generate()
        print(grid_gen.get_grid_str(grid))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "width", type=int,
        help="Width of grid"
    )
    parser.add_argument(
        "height", type=int,
        help="Heigh of grid"
    )
    parser.add_argument(
        "--max_obstacle_size", type=int, default=None,
        help="Max size of obstacle. If None then uses min(width, height) // 4"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random Seed"
    )
    main(parser.parse_args())
