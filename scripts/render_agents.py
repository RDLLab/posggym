"""Script for rendering episodes of posggym.agents policies.

The script takes an environment ID and a list of policy ids as arguments.
It then runs and renders episodes.

"""
import argparse
from pprint import pprint

import posggym

import posggym.agents as pga
import posggym.agents.evaluation as eval_lib


def main(args):  # noqa
    print("\n== Rendering Episodes ==")
    pprint(vars(args))

    policy_specs = []
    env_args, env_args_id = None, None
    for i, policy_id in enumerate(args.policy_ids):
        try:
            pi_spec = pga.spec(policy_id)
        except posggym.error.NameNotFound as e:
            if "/" not in policy_id:
                # try prepending env id
                policy_id = f"{args.env_id}/{policy_id}"
                pi_spec = pga.spec(policy_id)
            else:
                raise e

        if env_args is None and pi_spec.env_args is not None:
            env_args, env_args_id = pi_spec.env_args, pi_spec.env_args_id
        elif pi_spec.env_args is not None:
            assert pi_spec.env_args_id == env_args_id
        policy_specs.append(pi_spec)

    if env_args:
        env = posggym.make(args.env_id, render_mode=args.render_mode, **env_args)
    else:
        env = posggym.make(args.env_id, render_mode=args.render_mode)

    assert len(policy_specs) == len(env.possible_agents)

    policies = {}
    for idx, spec in enumerate(policy_specs):
        agent_id = env.possible_agents[idx]
        policies[agent_id] = pga.make(spec.id, env.model, agent_id)

    if args.seed is not None:
        env.reset(seed=args.seed)
        for i, policy in enumerate(policies.values()):
            policy.reset(seed=args.seed + i)

    eval_lib.run_episode(
        env,
        policies,
        args.num_episodes,
        trackers=eval_lib.get_default_trackers(),
        renderers=[eval_lib.EpisodeRenderer()],
        time_limit=None,
        logger=None,
        writer=None,
    )

    env.close()
    for policy in policies.values():
        policy.close()

    print("== All done ==")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--env_id", type=str, help="ID of the environment to run experiment in."
    )
    parser.add_argument(
        "-pids",
        "--policy_ids",
        type=str,
        nargs="+",
        help=(
            "List of IDs of policies to compare, one for each agent. Policy IDs should "
            "be provided in order of env.possible_agents (i.e. the first policy ID "
            "will be assigned to the 0-index policy in env.possible_agent, etc.)."
        ),
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1000,
        help="Number of episodes per experiment.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
    parser.add_argument(
        "--render_mode", type=str, default="human", help="The render mode to use."
    )
    main(parser.parse_args())
