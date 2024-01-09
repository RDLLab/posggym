import typer
from posggym.agents.evaluation import pairwise
from typing import Annotated

app = typer.Typer()


@app.command()
def run_pairwise(
    env_id: Annotated[
        str, typer.Option(help="ID of the environment to run for.")
    ] = None,
    env_args_id: Annotated[
        str, typer.Option(help="ID of the environment arguments.")
    ] = None,
    num_episodes: Annotated[
        int, typer.Option(help="Number of episodes to run per trial.")
    ] = 1000,
    seed: Annotated[int, typer.Option(help="Seed to use.")] = 0,
    output_dir: Annotated[
        str, typer.Option(help="Directory to save results files to.")
    ] = None,
    n_procs: Annotated[
        int, typer.Option(help="Number of runs to do in parallel.")
    ] = None,
    verbose: bool = False,
    show: bool = False,
):
    """
    Run pairwise evaluation for each possible pairing of policies.

    :param env_id: ID of the environment to run for.
    :param env_args_id: ID of the environment arguments.
    :param num_episodes: Number of episodes to run per trial.
    :param seed: Seed to use.
    :param output_dir: Directory to save results files to.
    :param n_procs: Number of runs to do in parallel.
    :param verbose: Run in verbose mode.
    :param show: Show pairwise comparison result plots.
    """
    output_dir = pairwise.run_pairwise_comparisons(
        env_id=env_id,
        env_args_id=env_args_id,
        num_episodes=num_episodes,
        seed=seed,
        output_dir=output_dir,
        n_procs=n_procs,
        verbose=verbose,
    )

    pairwise.plot_pairwise_comparison_results(
        output_dir,
        env_id=env_id,
        env_args_id=env_args_id,
        show=show,
        save=True,
        mean_only=True,
    )


if __name__ == "__main__":
    app()
