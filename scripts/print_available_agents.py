"""Script for printing available agents of posggym.agents."""
import argparse

import posggym.agents as pga


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default=None,
        help="ID of the environment to print agents for.",
    )
    args = parser.parse_args()
    if args.env_id is None:
        pga.pprint_registry()
    else:
        pga.pprint_registry(include_env_ids=[args.env_id])
