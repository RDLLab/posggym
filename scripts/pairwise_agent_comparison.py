"""Script for running pairwise evaluation of posggym.agents policies.

The script takes an environment ID and optional environment args ID and runs a pairwise
evaluation for each possible pairing of policies that are registered to the environment
and arguments.

"""
from pprint import pprint

import posggym.agents.evaluation as eval_lib
from posggym.agents.registration import get_all_envs


def main(args):  # noqa
    print(f"\n== Running Experiments for {args.env_id} ==")
    pprint(vars(args))

    env_args_map = get_all_envs()[args.env_id]
    for env_args_id, env_args in env_args_map.items():
        print(f"\n== Running for {env_args_id=} ==")
        eval_lib.run_pairwise_experiments(
            args.env_id,
            env_args,
            policy_ids=None,
            init_seed=args.init_seed,
            num_seeds=args.num_seeds,
            num_episodes=args.num_episodes,
            time_limit=args.time_limit,
            exp_id_init=0,
            render_mode=args.render_mode,
            record_env=args.record_env,
            exp_log_level=args.log_level,
            n_procs=args.n_procs,
            root_save_dir=args.root_save_dir,
        )

    print("== All All done ==")


if __name__ == "__main__":
    parser = eval_lib.get_exp_parser()
    parser.add_argument(
        "--env_id", type=str, help="Name of the environment to run experiment in."
    )
    parser.add_argument(
        "--env_args_id",
        type=str,
        default=None,
        help=(
            "ID of the environment arguments. If None will run a pairwise comparison "
            "for all arguments which have a registered policy attached."
        ),
    )
    parser.add_argument(
        "--init_seed", type=int, default=0, help="Experiment start seed."
    )
    parser.add_argument(
        "--num_seeds", type=int, default=1, help="Number of seeds to use."
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default=None,
        help=(
            "Render mode for experiment episodes (set to 'human' to render env to "
            "human display)."
        ),
    )
    parser.add_argument(
        "--record_env",
        action="store_true",
        help=(
            "Record renderings of experiment episodes (note rendering and recording at "
            "the same time are not currently supported)."
        ),
    )
    main(parser.parse_args())
