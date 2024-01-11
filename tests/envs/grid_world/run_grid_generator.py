"""Script for manually running and visualizing GridGenerator functionality."""
import random
import sys
from typing import Optional
from typing_extensions import Annotated

import typer
from posggym.envs.grid_world.core import GridGenerator

app = typer.Typer()


def _generate_mask(width: int, height: int, seed: Optional[int]):
    rng = random.Random(None) if seed is None else random.Random(seed + 1)

    mask = set()
    for _ in range(rng.randint(1, 2 * width + height)):
        x = rng.randint(0, width - 1)
        y = rng.randint(0, height - 1)
        mask.add((x, y))

    return mask


@app.command()
def main(
    width: Annotated[int, typer.Option(help="Width of grid")],
    height: Annotated[int, typer.Option(help="Height of grid")],
    use_random_mask: Annotated[
        bool, typer.Option(help="Use a random mask for generated grids")
    ],
    max_obstacle_size: Annotated[
        Optional[int],
        typer.Option(
            help="Max size of obstacle. If None then uses min(width, height) // 4"
        ),
    ] = None,
    seed: Annotated[Optional[int], typer.Option(help="Random Seed")] = None,
    check_grid_connectedness: Annotated[
        Optional[int], typer.Option(help="Also check for grid connectedness")
    ] = False,
):
    """Run."""
    mask = _generate_mask(width, height, seed) if use_random_mask else set()

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

        grid_gen = GridGenerator(
            width,
            height,
            mask,
            max_obstacle_size,
            max_num_obstacles=n,
            ensure_grid_connected=False,
            seed=seed,
        )
        seed += 1
        grid = grid_gen.generate()
        print(grid_gen.get_grid_str(grid))

        if check_grid_connectedness:
            components = grid.get_connected_components()
            if len(components) == 1:
                print("Grid fully connected")
            else:
                print(f"Grid divided into {len(components)} separate parts.")
                print("Connecting grid")
                grid = grid_gen.connect_grid_components(grid)
                print("Et voila")
                print(grid_gen.get_grid_str(grid))


if __name__ == "__main__":
    app()
