"""Script for continuing training of Rllib policies."""
import argparse

from posggym.agents.rllib.train.train import continue_training


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "rllib_alg_dir",
        type=str,
        help="Path to directory where rllib Algorithm checkpoint is stored.",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=2500,
        help="Number of iterations to train.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--remote", action="store_true", help="Run each Algorithm on a remote actor."
    )
    parser.add_argument(
        "--save_policy", action="store_true", help="Save policies to file."
    )
    args = parser.parse_args()

    continue_training(
        args.rllib_alg_dir,
        remote=args.remote,
        num_iterations=args.num_iterations,
        seed=args.seed,
        algorithm_class=None,
        save_policies=args.save_policy,
        verbose=True,
    )
