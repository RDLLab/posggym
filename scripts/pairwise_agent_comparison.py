"""Script for running pairwise evaluation of posggym.agents policies.

The script takes an environment ID and optional environment args ID and runs a pairwise
evaluation for each possible pairing of policies that are registered to the environment
and arguments.
"""

import argparse

from posggym.agents.evaluation import pairwise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default=None,
        help=(
            "ID of the environment to run for. If None we will run for all "
            "environments that have a registered policy attached."
        ),
    )
    parser.add_argument(
        "--env_args_id",
        type=str,
        default=None,
        help=(
            "ID of the environment arguments. If None will run a pairwise comparison "
            "for all arguments which have a registered policy attached. Only used if "
            "--env_id is not None."
        ),
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1000,
        help="Number of episodes to run per trial.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed to use.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results files too.",
    )
    parser.add_argument(
        "--n_procs",
        type=int,
        default=None,
        help=(
            "Number of runs to do in parallel. If None then will use all available "
            "CPUS on machine."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Run in verbose mode.")
    parser.add_argument(
        "--show", action="store_true", help="Show pairwise comparison result plots."
    )
    args = parser.parse_args()

    output_dir = pairwise.run_pairwise_comparisons(
        env_id=args.env_id,
        env_args_id=args.env_args_id,
        num_episodes=args.num_episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        n_procs=args.n_procs,
        verbose=args.verbose,
    )

    pairwise.plot_pairwise_comparison_results(
        output_dir,
        env_id=args.env_id,
        env_args_id=args.env_args_id,
        show=args.show,
        save=True,
        mean_only=True,
    )
