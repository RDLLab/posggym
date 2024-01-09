"""Functionality for measuring return based diversity of posggym.agents policies."""
from __future__ import annotations

import os
from itertools import product
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import posggym.agents as pga
from posggym.agents.evaluation import pairwise


def measure_return_diversity(
    pw_returns: np.ndarray,
    policies: List[str],
    co_teams: List[str],
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """Measure return diversity for a set of pairwise returns.

    Diversity is measured in terms of the Euclidean distance between the pairwise
    return distributions of each policy.

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
            df = pairwise.load_pairwise_comparison_results(env_id, output_dir, args_id)

            if df["symmetric"].unique().tolist()[0]:
                # only keep results for one agent
                agent_ids = df["agent_id"].unique().tolist()
                df = df[df["agent_id"] == agent_ids[0]]

            pw_returns_per_agent = pairwise.get_pairwise_returns_matrix(df)

            pairwise.generate_pairwise_returns_plot(
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
                results_i = measure_return_diversity(
                    pw_returns[0],
                    policy_ids,
                    co_teams_ids,
                    verbose=verbose,
                )
                results[i] = (results_i, policy_ids, co_teams_ids)

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
