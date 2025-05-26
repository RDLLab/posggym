"""Run random agents on an environment.

This script runs an environment using random agents.

The script takes a number of arguments (number of episodes, environment id, render
mode, etc.). To see all available arguments, run:

    python run_random_agents.py --help

Example, to run 10 episodes of the `Driving-v1` environment with `human` rendering mode,

    python run_random_agents.py \
        --env_id Driving-v1 \
        --num_episodes 10 \
        --render_mode human
"""

import argparse

from posggym.utils.run_random_agents import run_random_agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--env_id", type=str, required=True, help="ID of environment to run"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="The number of episodes to run.",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=None,
        help="Max number of steps to run each episode for.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random Seed.")
    parser.add_argument(
        "--render_mode",
        type=str,
        default=None,
        help="Mode to use for rendering.",
    )
    args = parser.parse_args()
    run_random_agent(**vars(args))
