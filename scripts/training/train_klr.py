"""Script for training K-Level reasoning policies using RLlib.

Note, this script will train policies on the given posggym environment using the
environment's default arguments. To use custom environment arguments add them to
the Algorithm configuration:

```
config = config = get_default_ppo_training_config(env_id, seed, log_level)
config.env_config["env_arg_name"] = env_arg_value
```

This will have to be done in a custom script.

"""
import argparse

from posggym.agents.rllib.train.algorithm_config import get_default_ppo_training_config
from posggym.agents.rllib.train.klr import train_klr_policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("env_id", type=str, help="Name of the environment to train on.")
    parser.add_argument(
        "-k", "--k", type=int, default=3, help="Number of reasoning levels"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=2500,
        help="Number of iterations to train.",
    )
    parser.add_argument("--log_level", type=str, default="WARN", help="Log level")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--num_gpus",
        type=float,
        default=1.0,
        help="Number of GPUs to use (can be a proportion).",
    )
    parser.add_argument(
        "-br",
        "--train_best_response",
        action="store_true",
        help="Train a best response on top of KLR policies.",
    )
    parser.add_argument(
        "--save_policies",
        action="store_true",
        help="Save policies to file at end of training.",
    )
    parser.add_argument(
        "--run_serially", action="store_true", help="Run training serially."
    )
    args = parser.parse_args()

    config = get_default_ppo_training_config(args.env_id, args.seed, args.log_level)
    train_klr_policy(
        args.env_id,
        k=args.k,
        best_response=args.train_best_response,
        seed=args.seed,
        algorithm_config=config,
        num_gpus=args.num_gpus,
        num_iterations=args.num_iterations,
        run_serially=args.run_serially,
        save_policies=args.save_policies,
        verbose=True,
    )
