"""A script for timing the step rate of an environment.

To see options and usage:

    python test_env.py --help

"""

import argparse
import time
from typing import Optional

import posggym


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


def main(
    env_id: Optional[str] = None,
    num_steps: int = 1000,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default=None,
        help="ID of environment to test. If none will run all registered environments.",
    )
    parser.add_argument(
        "--num_steps", type=int, default=10000, help="The number of steps to test for."
    )
    parser.add_argument("--seed", type=int, default=None, help="Random Seed.")
    parser.add_argument(
        "--render_mode",
        type=str,
        default=None,
        help="Mode to use for rendering. If None then doesn't render environment.",
    )
    args = parser.parse_args()
    main(**vars(args))
