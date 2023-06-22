"""Script for running pairwise evaluation of posggym.agents policies.

The script takes an environment ID and optional environment args ID and runs a pairwise
evaluation for each possible pairing of policies that are registered to the environment
and arguments.

Note, if running trials in parallel you may want to limit the number of threads used by
pytorch by setting the environment variable OMP_NUM_THREADS=1:

    $ export OMP_NUM_THREADS=1

This may be different for other backends, depending on your OS, e.g. if using MKL
instead of OMP.

"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple

import numpy as np

import posggym
import posggym.agents as pga
import posggym.model as M
from posggym import logger
from posggym.config import BASE_RESULTS_DIR

POLICIES_TO_SKIP = [
    "DiscreteFixedDistributionPolicy-v0",  # this is the same as Random-v0 by default
]

global lock
lock = None


def _init_pool_processes(write_lock: mp.Lock):
    # ref:
    # https://stackoverflow.com/questions/69907453/lock-objects-should-only-be-shared-between-processes-through-inheritance
    global lock
    lock = write_lock


def run_episodes(
    env_id: str,
    env_args_id: str | None,
    env_args: Dict[str, str],
    policy_ids: Dict[str, str],
    output_file: str,
    num_episodes: int,
    seed: int | None,
) -> Dict[str, Dict[str, float]]:
    """Run episodes and return the average reward for each policy."""
    policy_names = {
        i: pga.spec(policy_id).policy_name for i, policy_id in policy_ids.items()
    }
    print(f"- Running for {env_id}/{env_args_id} with policies={policy_names}")
    env = posggym.make(env_id, **env_args)
    policies = {
        i: pga.make(policy_id, env.model, i) for i, policy_id in policy_ids.items()
    }

    obs, _ = env.reset(seed=seed)
    for idx, policy in enumerate(policies.values()):
        policy.reset(seed=seed + idx if seed is not None else None)

    episode_rewards = {i: np.zeros(num_episodes) for i in policies}
    episode_lens = {i: np.zeros(num_episodes) for i in policies}
    episode_outcomes = {i: {outcome: 0 for outcome in M.Outcome} for i in policies}
    for ep_num in range(num_episodes):
        obs, _ = env.reset()
        for policy in policies.values():
            policy.reset()

        all_done = False
        while not all_done:
            actions = {}
            for i in env.agents:
                if policies[i].observes_state:
                    actions[i] = policies[i].step(env.state)
                else:
                    actions[i] = policies[i].step(obs[i])
            obs, rewards, _, _, all_done, info = env.step(actions)
            for i, r in rewards.items():
                episode_rewards[i][ep_num] += r
                episode_lens[i][ep_num] += 1

            if all_done:
                for i, info_i in info.items():
                    if "outcome" in info_i:
                        episode_outcomes[i][info_i["outcome"]] += 1

    agent_ids = sorted(policy_ids.keys())
    rows = []
    for i in agent_ids:
        policy_id = policy_ids[i]
        policy_spec = pga.spec(policy_id)

        co_player_names = [policy_names[j] for j in agent_ids if j != i]
        co_team_id = "(" + ",".join(co_player_names) + ")"

        row = {
            "env_id": env_id,
            "env_args_id": env_args_id,
            "symmetric": env.is_symmetric,
            "seed": seed,
            "num_agents": len(policy_ids),
            "num_episodes": num_episodes,
            "policy_id": policy_id,
            "policy_name": policy_spec.policy_name,
            "policy_version": policy_spec.version,
            "agent_id": i,
            "co_team_id": co_team_id,
            "episode_reward_mean": np.mean(episode_rewards[i]),
            "episode_reward_std": np.std(episode_rewards[i]),
        }
        for outcome in M.Outcome:
            row[f"num_{outcome.name}"] = episode_outcomes[i][outcome]
        rows.append(row)

    if lock is not None:
        lock.acquire()

    with open(output_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
        for row in rows:
            writer.writerow(row)

    if lock is not None:
        lock.release()

    return row


def get_env_pairwise_comparisons(
    env_id: str,
    env_args_id: str | None = None,
    num_episodes: int = 1000,
    seed: int | None = None,
    output_dir: str | None = None,
) -> List[Tuple]:
    """Get list of pairwise comparisons for all policies registered to environment."""
    print(f"\n=== Running Experiments for {env_id} ===")
    # attempt to make env to check if it is registered (displays nicer error msg)
    posggym.make(env_id)

    # setup output directory
    if output_dir is None:
        output_dir = os.path.join(
            BASE_RESULTS_DIR,
            "pairwise_agent_comparison" + datetime.now().strftime("%d-%m-%y_%H-%M-%S"),
        )
    os.makedirs(output_dir, exist_ok=True)
    env_output_dir = os.path.join(output_dir, env_id)
    os.makedirs(env_output_dir, exist_ok=True)

    # construct list of all possible policy pairings for the environment
    env_args_map = pga.get_all_envs()[env_id]
    all_pairwise_comparisons = []
    for args_id, env_args in env_args_map.items():
        if env_args_id is not None and args_id != env_args_id:
            continue
        print(f"- Running for {args_id=}")
        env_args = env_args or {}
        env = posggym.make(env_id, **env_args)

        all_agent_policies = pga.get_env_agent_policies(
            env_id, env_args, include_generic_policies=True
        )
        agent_ids = list(all_agent_policies.keys())
        print(f"  - {len(agent_ids)} agents")

        # setup output file
        output_file = os.path.join(env_output_dir, f"{args_id}.csv")
        if os.path.exists(output_file):
            logger.warn(f"{output_file} exists. Rewriting.")

        headers = [
            "env_id",
            "env_args_id",
            "symmetric",
            "seed",
            "num_agents",
            "num_episodes",
            "policy_id",
            "policy_name",
            "policy_version",
            "agent_id",
            "co_team_id",
            "episode_reward_mean",
            "episode_reward_std",
        ]
        for outcome in M.Outcome:
            headers.append(f"num_{outcome.name}")

        with open(output_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()

        pairwise_policy_ids = []
        if env.is_symmetric:
            # run one policy against multiple versions of the same policy for all other
            # agents (otherwise the # of policies can be too large for N > 2)
            all_policy_specs = []
            for policy_spec in all_agent_policies[agent_ids[0]]:
                if policy_spec.id not in POLICIES_TO_SKIP:
                    all_policy_specs.append(policy_spec)
                else:
                    print(f"  - Skipping {policy_spec.id}")

            print(f"  - {len(all_policy_specs)} policies to compare")

            for i in agent_ids:
                for policy_spec_i, policy_spec_j in product(all_policy_specs, repeat=2):
                    if policy_spec_i.id == policy_spec_j.id and i != agent_ids[0]:
                        # only add same vs same policy once
                        continue
                    policy_ids = {}
                    for j in agent_ids:
                        if i == j:
                            policy_ids[j] = policy_spec_i.id
                        else:
                            policy_ids[j] = policy_spec_j.id
                    pairwise_policy_ids.append(policy_ids)
        else:
            all_policy_specs = []
            num_policies = {}
            for i in agent_ids:
                agent_policy_specs = []
                for policy_spec in all_agent_policies[i]:
                    if policy_spec.id not in POLICIES_TO_SKIP:
                        agent_policy_specs.append(policy_spec)
                    else:
                        print(f"  - Skipping {policy_spec.id}")
                all_policy_specs.append(agent_policy_specs)
                num_policies[i] = len(agent_policy_specs)

            print(f"  - policies to compare {num_policies}")

            # run all pairwise combinations of policies
            for policy_specs in product(*all_policy_specs):
                policy_ids = {i: spec.id for i, spec in zip(agent_ids, policy_specs)}
                # print(f"  {policy_ids=}")
                pairwise_policy_ids.append(policy_ids)

        print(f"  - {len(pairwise_policy_ids)} comparisons to run")
        for policy_ids in pairwise_policy_ids:
            all_pairwise_comparisons.append(
                (
                    env_id,
                    args_id,
                    env_args,
                    policy_ids,
                    output_file,
                    num_episodes,
                    seed,
                )
            )

    print(f"Running total of {len(all_pairwise_comparisons)} comparisons for {env_id}.")

    return all_pairwise_comparisons


def main(args):  # noqa
    """Run the script."""
    if args.env_id is None:
        env_ids = list(pga.get_all_envs())
        env_args_id = None
    else:
        env_ids = [args.env_id]
        env_args_id = args.env_args_id

    all_pairwise_comparisons = []
    for env_id in env_ids:
        all_pairwise_comparisons.extend(
            get_env_pairwise_comparisons(
                env_id,
                env_args_id,
                output_dir=args.output_dir,
                num_episodes=args.num_episodes,
                seed=args.seed,
            )
        )
    n_procs = args.n_procs
    print(
        f"Running total of {len(all_pairwise_comparisons)} comparisons using {n_procs} "
        "processes."
    )
    write_lock = mp.Lock()
    with mp.Pool(
        n_procs, initializer=_init_pool_processes, initargs=(write_lock,)
    ) as pool:
        pool.starmap(run_episodes, all_pairwise_comparisons)

    print("== All done ==")


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
    main(parser.parse_args())
