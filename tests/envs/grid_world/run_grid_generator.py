"""Script for manually running and visualizing GridGenerator functionality."""
import random
import argparse

from posggym.envs.grid_world.core import GridGenerator


def _generate_mask(args):
    if args.seed is not None:
        rng = random.Random(args.seed + 1)
    else:
        rng = random.Random(None)

    mask = set()
    for _ in range(rng.randint(1, 2 * args.width + args.height)):
        x = rng.randint(0, args.width-1)
        y = rng.randint(0, args.height-1)
        mask.add((x, y))

    return mask


def main(args):
    """Run."""
    if args.use_random_mask:
        mask = _generate_mask(args)
    else:
        mask = set()

    if args.max_obstacle_size is None:
        max_obstacle_size = max(1, min(args.width, args.height) // 4)
    else:
        max_obstacle_size = args.max_obstacle_size

    grid_gen = GridGenerator(
        args.width, args.height, mask, max_obstacle_size, args.seed
    )

    while True:
        n = -1
        while n < 0:
            try:
                n = int(input("Select max_num_obstacles (Ctrl-C to exit): "))
            except ValueError:
                pass

        grid = grid_gen.generate(n)
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
        "--use_random_mask", action='store_true',
        help="Use a random mask for generated grids"
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
