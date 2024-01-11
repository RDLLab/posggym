"""Script for manually running and visualizing DrivingGridGenerator."""
import sys
from typing import Optional
from typing_extensions import Annotated

import typer

from posggym.envs.grid_world.driving.gen import DrivingGridGenerator

app = typer.Typer()


@app.command()
def main(
    width: Annotated[int, typer.Option(help="Width of grid")],
    height: Annotated[int, typer.Option(help="Height of grid")],
    max_obstacle_size: Annotated[
        Optional[int],
        typer.Option(
            help="Max size of obstacle. If None then uses min(width, height) // 4"
        ),
    ] = None,
    seed: Annotated[Optional[int], typer.Option(help="Random Seed")] = None,
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
    app()
