"""Script for measuring the diversity of posggym.agents policies.

The script takes an environment ID and optional environment args ID and runs a diversity
evaluation for each possible pairing of policies that are registered to the environment
and arguments.

"""
from pathlib import Path
import typer
from posggym.agents.evaluation import diversity, pairwise
from typing_extensions import Annotated
from typing import Optional

app = typer.Typer()


@app.command()
def run(
    env_id: Annotated[
        Optional[str], typer.Option(help="ID of the environment to run for.")
    ] = None,
    env_args_id: Annotated[
        Optional[str], typer.Option(help="ID of the environment arguments.")
    ] = None,
    num_episodes: Annotated[
        int, typer.Option(help="Number of episodes to run per trial.")
    ] = 1000,
    seed: Annotated[int, typer.Option(help="Seed to use.")] = 0,
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Directory to save results files to.")
    ] = None,
    n_procs: Annotated[
        Optional[int], typer.Option(help="Number of runs to do in parallel.")
    ] = None,
    verbose: Annotated[bool, typer.Option(help="Run in verbose mode.")] = False,
):
    """
    Run pairwise comparisons to generate data.

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
    env_id: Annotated[
        Optional[str], typer.Option(help="ID of the environment to run for.")
    ] = None,
    env_args_id: Annotated[
        Optional[str], typer.Option(help="ID of the environment arguments.")
    ] = None,
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Directory containing results files.")
    ] = None,
    verbose: Annotated[bool, typer.Option(help="Run in verbose mode.")] = False,
):
    """
    Plot results (expects `--output_dir`).
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
    env_id: Annotated[
        Optional[str], typer.Option(help="ID of the environment to run for.")
    ] = None,
    env_args_id: Annotated[
        Optional[str], typer.Option(help="ID of the environment arguments.")
    ] = None,
    num_episodes: Annotated[
        int, typer.Option(help="Number of episodes to run per trial.")
    ] = 1000,
    seed: Annotated[int, typer.Option(help="Seed to use.")] = 0,
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Directory to save results files to.")
    ] = None,
    n_procs: Annotated[
        Optional[int], typer.Option(help="Number of runs to do in parallel.")
    ] = None,
    verbose: Annotated[bool, typer.Option(help="Run in verbose mode.")] = False,
):
    """
    Run pairwise comparisons and then plot results.

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
