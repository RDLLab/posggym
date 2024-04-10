"""Script for generating pairwise payoff figures from results."""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from posggym.config import BASE_RESULTS_DIR, REPO_DIR

sys.path.append(str(REPO_DIR / "notebooks"))
import plot_utils  # noqa: E402

results_dir = REPO_DIR / "notebooks" / "results" / "pairwise_agent_comparison"


def generate_fig(
    df,
    env_output_dir: Path,
    result_file: Path,
    policy_key: str,
    coplayer_policy_key: str,
):
    """Generate pairwise figure by policy Type."""
    env_symmetric = df["symmetric"].unique().tolist()[0]
    if env_symmetric:
        # can do a single plot
        fig_width = len(df[coplayer_policy_key].unique()) // 1.5
        fig_height = len(df[policy_key].unique()) // 1.5

        fig, _ = plot_utils.plot_pairwise_comparison(
            df,
            y_key="episode_reward_mean",
            policy_key=policy_key,
            coplayer_policy_key=coplayer_policy_key,
            vrange=None,
            figsize=(fig_width, fig_height),
            valfmt="{x:.2f}",
        )
        fig.savefig(
            (env_output_dir / result_file.with_suffix("")) + f"_{policy_key}.pdf",
            format="pdf",
            bbox_inches="tight",
        )
    else:
        # for asymmetric envs we do one plot per agent
        agent_ids = df["agent_id"].unique().tolist()
        agent_ids.sort()
        for i in agent_ids:
            df_i = df[df["agent_id"] == i]
            fig_width = len(df_i[coplayer_policy_key].unique()) // 1.5
            fig_height = len(df_i[policy_key].unique()) // 1.5

            fig, axs = plot_utils.plot_pairwise_comparison(
                df_i,
                y_key="episode_reward_mean",
                policy_key=policy_key,
                coplayer_policy_key=coplayer_policy_key,
                vrange=None,
                figsize=(fig_width, fig_height),
                valfmt="{x:.2f}",
            )
            axs[0][0].set_xlabel(f"AgentID={i}")

            fig.savefig(
                env_output_dir
                / (result_file.with_suffix("") + f"_agent{i}_{policy_key}.pdf"),
                format="pdf",
                bbox_inches="tight",
            )


def main(env_id: Optional[str], output_dir: Optional[str] = None):
    available_env_result_dirs = [x.name for x in results_dir.glob("*")]
    available_env_result_dirs.sort()

    if output_dir is None:
        output_dir = BASE_RESULTS_DIR / (
            "parwise_agent_payoff_figs" + datetime.now().strftime("%d-%m-%y_%H-%M-%S")
        )

    print(f"Saving figures to {output_dir}")
    output_dir.mkdir(exist_ok=True)

    env_ids = available_env_result_dirs if env_id is None else [env_id]

    for env_id in env_ids:
        print(f"Generating figures for {env_id}")
        env_output_dir = output_dir / env_id
        env_output_dir.mkdir(exist_ok=True)
        print(f"Saving {env_id} figures to {env_output_dir}")

        env_result_dir = results_dir / env_id
        env_result_files = env_result_dir.glob("*")
        env_result_files.sort(key=lambda x: x.name)

        for result_path in env_result_files:
            result_file = result_path.name
            print(f"Generating figures for {result_file}")

            df = plot_utils.import_results(result_path)

            generate_fig(
                df,
                env_output_dir,
                result_file,
                policy_key="policy_name",
                coplayer_policy_key="co_team_name",
            )

            generate_fig(
                df,
                env_output_dir,
                result_file,
                policy_key="policy_type",
                coplayer_policy_key="co_team_type",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate pairwise payoff figures from results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default=None,
        help="ID of the environment.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to directory to save figures.",
    )
    main(**vars(parser.parse_args()))
