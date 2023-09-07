"""A script for timing the step rate of an environment.

To see options and usage:

    python test_env.py --help

"""
import time
from typing import Optional
from typing_extensions import Annotated

import typer
import posggym

app = typer.Typer()


def time_env_step_rate(
    env_id: str, num_steps: int, seed: Optional[int], render_mode: Optional[str]
) -> float:
    """Calculate the step rate of environment.

    Arguments
    ---------
    env_id: ID of environment to test
    num_steps: The number of steps to test for
    seed: the random seed to use
    render_mode: render mode for environment

    Returns
    -------
    step_rate: the average steps per second executed in the environment

    """
    env = posggym.make(env_id, render_mode=render_mode)
    env.reset(seed=seed)

    start_time = time.time()

    for _ in range(num_steps):
        actions = {i: env.action_spaces[i].sample() for i in env.agents}
        _, _, _, _, done, _ = env.step(actions)

        if render_mode is not None:
            env.render()

        if done:
            env.reset()

    time_elapsed = time.time() - start_time
    return num_steps / time_elapsed


@app.command()
def main(
    env_id: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "ID of environment to test. If none will run all registered"
                "environments."
            )
        ),
    ] = None,
    num_steps: Annotated[
        int, typer.Option(help="The number of steps to test for.")
    ] = 1000,
    seed: Annotated[Optional[int], typer.Option(help="Random Seed.")] = None,
    render_mode: Annotated[
        Optional[str],
        typer.Option(
            help="Mode to use for rendering. If None then doesn't render environment."
        ),
    ] = None,
):
    env_ids = list(posggym.registry) if env_id is None else [env_id]

    results = {}
    for env_id in env_ids:
        print(f"Timing {env_id} environment for {num_steps} steps.")
        step_rate = time_env_step_rate(
            env_id=env_id,
            num_steps=num_steps,
            seed=seed,
            render_mode=render_mode,
        )
        print(f"Step rate for '{env_id}' = {step_rate:.4f} steps per second")
        results[env_id] = step_rate

    max_env_id_len = max(len(env_id) for env_id in results)
    second_col_header = "Steps per second"
    col2_width = len(second_col_header)
    print(f"| {'Env':<{max_env_id_len}} | Steps per second |")
    print(f"| {'-' * max_env_id_len} | {'-' * col2_width} |")
    for env_id, step_rate in results.items():
        print(f"| {env_id:<{max_env_id_len}} | {step_rate:<{col2_width}.2f} |")


if __name__ == "__main__":
    app
