"""Script for measuring the diversity of posggym.agents policies.

The script takes an environment ID and optional environment args ID and runs a diversity
evaluation for each possible pairing of policies that are registered to the environment
and arguments.

"""

import argparse

from posggym.agents.evaluation import diversity, pairwise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "action",
        type=str,
        choices=["run", "plot", "run_and_plot"],
        help=(
            "Action to perform. "
            "`run` - runs pairwise comparisons to generate data. "
            "`plot` - plots results (expects `--output_dir`). "
            "`run_and_plot` - run pairwise comparisons then plots results. "
        ),
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
    args = parser.parse_args()

    if args.action == "run":
        pairwise.run_pairwise_comparisons(
            args.env_id,
            args.env_args_id,
            args.num_episodes,
            args.output_dir,
            args.seed,
            args.n_procs,
            args.verbose,
        )
    elif args.action == "plot":
        assert args.output_dir is not None
        diversity.run_return_diversity_analysis(
            args.output_dir,
            args.env_id,
            args.env_args_id,
            args.verbose,
        )
    else:
        output_dir = pairwise.run_pairwise_comparisons(
            args.env_id,
            args.env_args_id,
            args.num_episodes,
            args.output_dir,
            args.seed,
            args.n_procs,
            args.verbose,
        )
        diversity.run_return_diversity_analysis(
            output_dir,
            args.env_id,
            args.env_args_id,
            args.verbose,
        )
