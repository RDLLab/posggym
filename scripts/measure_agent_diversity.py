"""Script for measuring the diversity of posggym.agents policies.

The script takes an environment ID and optional environment args ID and runs a diversity
evaluation for each possible pairing of policies that are registered to the environment
and arguments.

"""

import typer
from posggym.agents.evaluation import diversity, pairwise

app = typer.Typer()


@app.command()
def run(
    env_id: str = None,
    env_args_id: str = None,
    num_episodes: int = 1000,
    seed: int = 0,
    output_dir: str = None,
    n_procs: int = None,
    verbose: bool = False,
):
    """
    Run pairwise comparisons to generate data.

    :param env_id: ID of the environment to run for.
    :param env_args_id: ID of the environment arguments.
    :param num_episodes: Number of episodes to run per trial.
    :param seed: Seed to use.
    :param output_dir: Directory to save results files to.
    :param n_procs: Number of runs to do in parallel.
    :param verbose: Run in verbose mode.
    """
    pairwise.run_pairwise_comparisons(
        env_id=env_id,
        env_args_id=env_args_id,
        num_episodes=num_episodes,
        output_dir=output_dir,
        seed=seed,
        n_procs=n_procs,
        verbose=verbose,
    )


@app.command()
def plot(
    env_id: str = None,
    env_args_id: str = None,
    output_dir: str = None,
    verbose: bool = False,
):
    """
    Plot results (expects `--output_dir`).

    :param env_id: ID of the environment to run for.
    :param env_args_id: ID of the environment arguments.
    :param output_dir: Directory containing results files.
    :param verbose: Run in verbose mode.
    """
    assert output_dir is not None
    diversity.run_return_diversity_analysis(
        output_dir=output_dir,
        env_id=env_id,
        env_args_id=env_args_id,
        verbose=verbose,
    )


@app.command()
def run_and_plot(
    env_id: str = None,
    env_args_id: str = None,
    num_episodes: int = 1000,
    seed: int = 0,
    output_dir: str = None,
    n_procs: int = None,
    verbose: bool = False,
):
    """
    Run pairwise comparisons and then plot results.

    :param env_id: ID of the environment to run for.
    :param env_args_id: ID of the environment arguments.
    :param num_episodes: Number of episodes to run per trial.
    :param seed: Seed to use.
    :param output_dir: Directory to save results files to.
    :param n_procs: Number of runs to do in parallel.
    :param verbose: Run in verbose mode.
    """
    output_dir = pairwise.run_pairwise_comparisons(
        env_id=env_id,
        env_args_id=env_args_id,
        num_episodes=num_episodes,
        output_dir=output_dir,
        seed=seed,
        n_procs=n_procs,
        verbose=verbose,
    )
    diversity.run_return_diversity_analysis(
        output_dir=output_dir,
        env_id=env_id,
        env_args_id=env_args_id,
        verbose=verbose,
    )


if __name__ == "__main__":
    app()
