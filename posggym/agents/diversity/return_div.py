"""Functionality for measuring return based diversity of posggym.agents policies."""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
from datetime import datetime
from itertools import product
from typing import Dict, List, NamedTuple, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

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


class PWCParams(NamedTuple):
    """Pairwise Comparison Parameters."""

    env_id: str
    env_args_id: str | None
    env_args: Dict[str, str]
    policy_ids: Dict[str, str]
    output_file: str
    num_episodes: int
    seed: int | None
    run_num: int
    verbose: bool = False


def run_episodes(args) -> Dict[str, Dict[str, float]]:
    """Run episodes and return the average reward for each policy."""
    params, total_runs = args
    policy_names = {
        i: pga.spec(policy_id).policy_name for i, policy_id in params.policy_ids.items()
    }
    if params.verbose:
        print(
            f"- Running episodes for {params.env_id}/{params.env_args_id} with "
            f"policies={policy_names}"
        )

    env = posggym.make(params.env_id, **params.env_args)
    policies = {
        i: pga.make(policy_id, env.model, i)
        for i, policy_id in params.policy_ids.items()
    }

    obs, _ = env.reset(seed=params.seed)
    for idx, policy in enumerate(policies.values()):
        policy.reset(seed=params.seed + idx if params.seed is not None else None)

    episode_rewards = {i: np.zeros(params.num_episodes) for i in policies}
    episode_lens = {i: np.zeros(params.num_episodes) for i in policies}
    episode_outcomes = {i: {outcome: 0 for outcome in M.Outcome} for i in policies}
    for ep_num in range(params.num_episodes):
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

    agent_ids = sorted(params.policy_ids.keys())
    rows = []
    for i in agent_ids:
        policy_id = params.policy_ids[i]
        policy_spec = pga.spec(policy_id)

        co_player_names = [policy_names[j] for j in agent_ids if j != i]
        co_team_id = "(" + ",".join(co_player_names) + ")"

        row = {
            "env_id": params.env_id,
            "env_args_id": params.env_args_id,
            "symmetric": env.is_symmetric,
            "seed": params.seed,
            "num_agents": len(params.policy_ids),
            "num_episodes": params.num_episodes,
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

    with open(params.output_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
        for row in rows:
            writer.writerow(row)

    if lock is not None:
        lock.release()

    if params.run_num % max(10, total_runs // 10) == 0:
        print(f"Run {params.run_num}/{total_runs} complete.")

    return row


def get_pairwise_comparison_params(
    env_id: str,
    output_dir: str,
    env_args_id: str | None = None,
    num_episodes: int = 1000,
    seed: int | None = None,
    verbose: bool = False,
) -> List[PWCParams]:
    """Get parameters for pairwise comparisons of all of an environment's policies."""
    # attempt to make env to check if it is registered (displays nicer error msg)
    posggym.make(env_id)

    env_output_dir = os.path.join(output_dir, env_id)
    os.makedirs(env_output_dir, exist_ok=True)

    # construct list of all possible policy pairings for the environment
    env_args_map = pga.get_all_envs()[env_id]
    all_pairwise_comparisons = []
    run_num = 1
    for args_id, env_args in env_args_map.items():
        if env_args_id is not None and args_id != env_args_id:
            continue

        if verbose:
            print(f"- Running for {args_id=}")

        env_args = env_args or {}
        env = posggym.make(env_id, **env_args)

        all_agent_policies = pga.get_env_agent_policies(
            env_id, env_args, include_generic_policies=True
        )
        agent_ids = list(all_agent_policies.keys())
        if verbose:
            print(f"  - {len(agent_ids)} agents")

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
                elif verbose:
                    print(f"  - Skipping {policy_spec.id}")

            if verbose:
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

            if verbose:
                print(f"  - policies to compare {num_policies}")

            # run all pairwise combinations of policies
            for policy_specs in product(*all_policy_specs):
                policy_ids = {i: spec.id for i, spec in zip(agent_ids, policy_specs)}
                # print(f"  {policy_ids=}")
                pairwise_policy_ids.append(policy_ids)

        if verbose:
            print(f"  - {len(pairwise_policy_ids)} comparisons to run")

        for policy_ids in pairwise_policy_ids:
            all_pairwise_comparisons.append(
                PWCParams(
                    env_id=env_id,
                    env_args_id=args_id,
                    env_args=env_args,
                    policy_ids=policy_ids,
                    output_file=output_file,
                    num_episodes=num_episodes,
                    seed=seed,
                    verbose=verbose,
                    run_num=run_num,
                )
            )
            run_num += 1

    if verbose:
        print(f"Running {len(all_pairwise_comparisons)} comparisons for {env_id=}.")

    return all_pairwise_comparisons


def run_pairwise_comparisons(
    env_id: str | None,
    env_args_id: str | None = None,
    num_episodes: int = 1000,
    output_dir: str | None = None,
    seed: int | None = None,
    n_procs: int | None = None,
    verbose: bool = False,
) -> List[PWCParams]:
    """Run pairwise comparisons for all policies in an environment."""
    if env_id is None:
        env_ids = list(pga.get_all_envs())
        env_args_id = None
    else:
        env_ids = [env_id]
        env_args_id = env_args_id

    if output_dir is None:
        output_dir = os.path.join(
            BASE_RESULTS_DIR,
            "return_div" + datetime.now().strftime("%d-%m-%y_%H-%M-%S"),
        )
    os.makedirs(output_dir, exist_ok=True)

    all_pairwise_comparisons = []
    for env_id in env_ids:
        all_pairwise_comparisons.extend(
            get_pairwise_comparison_params(
                env_id=env_id,
                env_args_id=env_args_id,
                output_dir=output_dir,
                num_episodes=num_episodes,
                seed=seed,
                verbose=verbose,
            )
        )

    print(f"Running {len(all_pairwise_comparisons)} comparisons using {n_procs=}.")
    print(f"Writing results to {output_dir=}")

    pw_args = [
        (params, len(all_pairwise_comparisons)) for params in all_pairwise_comparisons
    ]

    if n_procs is None or n_procs > 1:
        # limit number of cpus used by torch and numpy to 1 per process
        # note: setting this for torch should be enough but we set it for numpy too
        # since it changes the variable for OPENMP, MKL, etc. which numpy uses
        torch.set_num_threads(1)

    write_lock = mp.Lock()
    with mp.Pool(
        n_procs, initializer=_init_pool_processes, initargs=(write_lock,)
    ) as pool:
        pool.map(run_episodes, pw_args)

    print("All pairwise comparisons complete.")
    return all_pairwise_comparisons


def load_pairwise_comparison_results(
    env_id: str,
    output_dir: str,
    env_args_id: str | None = None,
) -> pd.DataFrame:
    """Load pairwise comparison results for an environment."""
    env_output_dir = os.path.join(output_dir, env_id)
    output_file = os.path.join(env_output_dir, f"{env_args_id}.csv")
    if not os.path.exists(output_file):
        raise ValueError(
            f"Results do not exist for {env_id=}, {env_args_id=} in {output_dir=}."
        )
    return pd.read_csv(output_file)


def get_pairwise_returns_matrix(
    df: pd.DataFrame,
) -> Dict[str, Tuple[np.ndarray, List[str], List[str]]]:
    """Get pairwise returns matrix for each agent in environment."""
    num_episodes = df["num_episodes"].unique().tolist()[0]
    agent_ids = df["agent_id"].unique().tolist()
    agent_ids.sort()
    policies_per_agent = {
        i: df[df["agent_id"] == i]["policy_name"].unique().tolist() for i in agent_ids
    }
    co_teams_per_agent = {
        i: df[df["agent_id"] == i]["co_team_id"].unique().tolist() for i in agent_ids
    }
    co_teams_per_agent = {i: [] for i in agent_ids}
    for i in agent_ids:
        co_teams_i = df[df["agent_id"] == i]["co_team_id"].unique().tolist()
        # exclude non-homogenous teams
        for co_team in co_teams_i:
            pi_ids = co_team.replace("(", "").replace(")", "").split(",")
            if all(pi_j == pi_ids[0] for pi_j in pi_ids):
                co_teams_per_agent[i].append(co_team)

    for i in agent_ids:
        policies_per_agent[i].sort()
        co_teams_per_agent[i].sort()

    pw_returns_per_agent = {
        i: (
            np.zeros((3, len(policies_per_agent[i]), len(co_teams_per_agent[i]))),
            policies_per_agent[i],
            # store co-team labels for plotting, keep only first policy name for each
            # co-team
            [
                team_id.replace("(", "").replace(")", "").split(",")[0]
                for team_id in co_teams_per_agent[i]
            ],
        )
        for i in agent_ids
    }
    for i in agent_ids:
        pw_returns_i = pw_returns_per_agent[i][0]
        for (policy_idx, policy_name), (co_team_idx, co_team_id) in product(
            enumerate(policies_per_agent[i]), enumerate(co_teams_per_agent[i])
        ):
            if (
                len(
                    df[
                        (df["agent_id"] == i)
                        & (df["policy_name"] == policy_name)
                        & (df["co_team_id"] == co_team_id)
                    ]
                )
                == 0
            ):
                # no results for this policy/co-team matchup
                continue

            pw_returns_i[0, policy_idx, co_team_idx] = df[
                (df["agent_id"] == i)
                & (df["policy_name"] == policy_name)
                & (df["co_team_id"] == co_team_id)
            ]["episode_reward_mean"].values[0]
            pw_returns_i[1, policy_idx, co_team_idx] = df[
                (df["agent_id"] == i)
                & (df["policy_name"] == policy_name)
                & (df["co_team_id"] == co_team_id)
            ]["episode_reward_std"].values[0]
            # compute 95% CI for mean: 1.96*std/sqrt(N)
            pw_returns_i[2, policy_idx, co_team_idx] = (
                1.96 * pw_returns_i[1, policy_idx, co_team_idx] / np.sqrt(num_episodes)
            )
    return pw_returns_per_agent


def generate_pairwise_returns_plot(
    env_id: str,
    output_dir: str,
    env_args_id: str | None,
    pw_returns_per_agent: Dict[str, Tuple[np.ndarray, List[str], List[str]]],
    show: bool = True,
    save: bool = True,
) -> None:
    """Generate a plot of pairwise returns for an environment."""
    num_agents = len(pw_returns_per_agent)

    for stat in ["mean", "std", "CI95"]:
        fig, axs = plt.subplots(
            nrows=1,
            ncols=num_agents,
            figsize=(6 * num_agents, 6),
            layout="constrained",
            squeeze=False,
        )
        for idx, i in enumerate(pw_returns_per_agent):
            pw_returns, policy_ids, co_teams_ids = pw_returns_per_agent[i]
            if stat == "mean":
                values = pw_returns[0]
            elif stat == "std":
                values = pw_returns[1]
            else:
                values = pw_returns[2]

            # only show first policy name for each co-team
            co_team_labels = [
                team_id.replace("(", "").replace(")", "").split(",")[0]
                for team_id in co_teams_ids
            ]

            sns.heatmap(
                values,
                ax=axs[0][idx],
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                xticklabels=co_team_labels,
                yticklabels=policy_ids,
                square=True,
                annot_kws={"fontsize": min(8, 24 / np.sqrt(max(values.shape)))},
            )
            axs[0][idx].set_title(f"agent `{i}`")

        if save:
            if env_id in output_dir:
                output_file = os.path.join(
                    output_dir, f"{env_args_id}_pairwise_returns_{stat}.svg"
                )
            else:
                output_file = os.path.join(
                    output_dir, f"{env_id}_{env_args_id}_pairwise_returns_{stat}.svg"
                )
            fig.savefig(output_file, bbox_inches="tight")

    if show:
        fig.tight_layout()
        plt.show()


def measure_return_ed_diversity(
    pw_returns: np.ndarray,
    policies: List[str],
    co_teams: List[str],
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """Measure return Euclidean distance (ED) diversity for a set of pairwise returns.

    Returns
    -------
    pw_ed : np.ndarray
        The pairwise ED between return distributions of each policy.
    policy_ed : np.ndarray
        The total ED for each policy (the larger the more diverse).
    clusters : np.ndarray
        The cluster each policy belongs to. Cluster ordering arbitrary, but all policies
        in the same cluster are similar w.r.t ED between their return distributions.

    """
    assert (len(policies), len(co_teams)) == pw_returns.shape

    # get max and min returns, excluding random policy if it exists
    random_co_team_idxs = []
    random_idx = None
    if len(policies) > 2 and "Random" in policies:
        random_idx = policies.index("Random")
        pw_returns_excl_random = np.delete(pw_returns, random_idx, axis=0)
        random_co_team_idxs = [
            idx for idx, co_team in enumerate(co_teams) if "Random" in co_team
        ]
        pw_returns_excl_random = np.delete(
            pw_returns_excl_random, random_co_team_idxs, axis=1
        )
        max_return = np.max(pw_returns_excl_random)
        min_return = np.min(pw_returns_excl_random)
    else:
        max_return, min_return = np.max(pw_returns), np.min(pw_returns)

    if verbose:
        with np.printoptions(precision=2, suppress=True):
            # print(f"{pw_returns=}")
            print(f"{max_return=:.2f}, {min_return=:.2f}")

    # normalize pairwise distributions into [0.0, 1.0]
    pw_returns_norm = (pw_returns - min_return) / (max_return - min_return)
    if verbose:
        with np.printoptions(precision=2, suppress=True):
            for idx, policy_name in enumerate(policies):
                print(f"{policy_name}:\n{pw_returns_norm[idx]}")

    # compute ED between normalized return distributions of each policy
    pw_ed = np.zeros((len(policies), len(policies)))
    for pi_idx, pj_idx in product(range(len(policies)), repeat=2):
        pw_ed[pi_idx, pj_idx] = np.sqrt(
            np.sum((pw_returns[pi_idx] - pw_returns[pj_idx]) ** 2)
        )

    if verbose:
        print("Pairwise ED:")
        with np.printoptions(precision=2, suppress=True):
            for pi_idx, row in enumerate(pw_ed):
                print(policies[pi_idx], row)

    # Compute total CE for each policy (the larger the more diverse)
    policy_ed = np.zeros(len(policies), dtype=float)
    for pi_idx in range(len(policies)):
        policy_ed[pi_idx] = np.sum(pw_ed[pi_idx])

    if verbose:
        with np.printoptions(precision=2, suppress=True):
            print(f"{policy_ed=}")

    # Group similar policies by relative MSE
    if len(policies) > 2 and "Random" in policies:
        # exclude random policy from calculating bin sizes for grouping
        pw_ed_excl_random = np.delete(pw_ed, random_idx, axis=0)
        pw_ed_excl_random = np.delete(pw_ed_excl_random, random_idx, axis=1)
        min_ed, max_ed = np.min(pw_ed_excl_random), np.max(pw_ed_excl_random)
    else:
        min_ed, max_ed = np.min(pw_ed), np.max([pw_ed])
    threshold = min_ed + (max_ed - min_ed) * 0.1

    similar_policy_groups = [[policies[0]]]
    for policy_name in policies[1:]:
        pi_idx = policies.index(policy_name)
        min_div = float("inf")
        min_div_group_idx = 0
        for group_idx, group in enumerate(similar_policy_groups):
            pj_idx = policies.index(group[0])
            if pw_ed[pi_idx, pj_idx] < min_div:
                min_div = pw_ed[pi_idx, pj_idx]
                min_div_group_idx = group_idx

        if min_div < threshold:
            similar_policy_groups[min_div_group_idx].append(policy_name)
        else:
            similar_policy_groups.append([policy_name])

    if verbose:
        print("Similar policies:")
        print(f"{min_ed=} {max_ed=} {threshold=}")
        for group in similar_policy_groups:
            print(group)

    # Compute clusters
    clusters = np.zeros(len(policies), dtype=int)
    for cluster_idx, group in enumerate(similar_policy_groups):
        for policy_name in group:
            clusters[policies.index(policy_name)] = cluster_idx

    if verbose:
        with np.printoptions(precision=2, suppress=True):
            print(f"{clusters=}")

    return {
        "pairwise_return_ed": pw_ed,
        "policy_ed": policy_ed.reshape(-1, 1),
        "return_ed_clusters": clusters.reshape(-1, 1),
    }


def run_return_diversity_analysis(
    output_dir: str,
    env_id: str | None,
    env_args_id: str | None = None,
    verbose: bool = False,
):
    """Run return diversity analysis from pairwise comparison results.

    Results of analysis will be output to the results directory for each env.
    """
    all_envs = pga.get_all_envs()
    if env_id is None:
        env_ids = list(all_envs)
        env_args_id = None
    else:
        env_ids = [env_id]
        env_args_id = env_args_id

    for env_id in os.listdir(output_dir):
        if env_id not in env_ids:
            continue
        print(f"Running return diversity analysis for {env_id=}")
        env_results_dir = os.path.join(output_dir, env_id)
        print(f"  results saved to {env_results_dir=}")

        env_result_files = [
            fname for fname in os.listdir(env_results_dir) if fname.endswith(".csv")
        ]

        env_args_map = pga.get_all_envs()[env_id]
        for args_id in env_args_map:
            if (env_args_id is not None and args_id != env_args_id) or (
                env_args_id is None and str(args_id) + ".csv" not in env_result_files
            ):
                continue

            print(f"  {args_id=}")
            df = load_pairwise_comparison_results(env_id, output_dir, args_id)

            if df["symmetric"].unique().tolist()[0]:
                # only keep results for one agent
                agent_ids = df["agent_id"].unique().tolist()
                df = df[df["agent_id"] == agent_ids[0]]

            pw_returns_per_agent = get_pairwise_returns_matrix(df)

            generate_pairwise_returns_plot(
                env_id=env_id,
                output_dir=env_results_dir,
                env_args_id=args_id,
                pw_returns_per_agent=pw_returns_per_agent,
                show=False,
                save=True,
            )

            results = {}
            for i in pw_returns_per_agent:
                pw_returns, policy_ids, co_teams_ids = pw_returns_per_agent[i]
                ce_results = measure_return_ed_diversity(
                    pw_returns[0],
                    policy_ids,
                    co_teams_ids,
                    verbose=verbose,
                )
                results[i] = (ce_results, policy_ids, co_teams_ids)

            for k in list(results.values())[0][0]:
                fig, axs = plt.subplots(
                    nrows=1,
                    ncols=len(results),
                    figsize=(6 * len(results), 6),
                    squeeze=False,
                    layout="constrained",
                )
                for idx, i in enumerate(results):
                    div_results, policy_ids, co_teams_ids = results[i]

                    if len(results) > 2:
                        # only show first policy name for each co-team
                        co_team_labels = [
                            team_id.replace("(", "").replace(")", "").split(",")[0]
                            for team_id in co_teams_ids
                        ]
                    else:
                        co_team_labels = co_teams_ids

                    if div_results[k].shape[1] == 1:
                        xticklabels = [""]
                        annot_fontsize = 8
                    elif k in ["pairwise_return_ed"]:
                        xticklabels = policy_ids
                        annot_fontsize = min(8, 24 / np.sqrt(max(div_results[k].shape)))
                    else:
                        xticklabels = co_team_labels
                        annot_fontsize = min(8, 24 / np.sqrt(max(div_results[k].shape)))

                    sns.heatmap(
                        div_results[k],
                        ax=axs[0][idx],
                        annot=True,
                        fmt=".0f" if div_results[k].dtype == int else ".2f",
                        cmap="YlGnBu",
                        xticklabels=xticklabels,
                        yticklabels=policy_ids,
                        square=div_results[k].shape[1] != 1,
                        annot_kws={"fontsize": annot_fontsize},
                    )
                    axs[0][idx].set_title(f"agent `{i}`")
                fig.savefig(
                    os.path.join(env_results_dir, f"{args_id}_{k}.svg"),
                    bbox_inches="tight",
                )


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
        run_pairwise_comparisons(
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
        run_return_diversity_analysis(
            args.output_dir,
            args.env_id,
            args.env_args_id,
            args.verbose,
        )
    else:
        if args.output_dir is None:
            output_dir = os.path.join(
                BASE_RESULTS_DIR,
                "return_div" + datetime.now().strftime("%d-%m-%y_%H-%M-%S"),
            )
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = args.output_dir
        run_pairwise_comparisons(
            args.env_id,
            args.env_args_id,
            args.num_episodes,
            output_dir,
            args.seed,
            args.n_procs,
            args.verbose,
        )
        run_return_diversity_analysis(
            output_dir,
            args.env_id,
            args.env_args_id,
            args.verbose,
        )
