"""Run random agents on an environment.

This script runs an environment using random agents.

The script takes a number of arguments (number of episodes, environment id, render
mode, etc.). To see all available arguments, run:

    python run_random_agents.py --help

Example, to run 10 episodes of the `Driving-v1` environment with `human` rendering mode,

    python run_random_agents.py \
        --env_id Driving-v1 \
        --num_episodes 10 \
        --render_mode human
"""
from typing import Dict, List, Optional
from typing_extensions import Annotated

import posggym
import typer

app = typer.Typer()


@app.command()
def run_random_agent(
    env_id: Annotated[str, typer.Option(help="ID of environment to run")],
    num_episodes: Annotated[int, typer.Option(help="The number of episodes to run.")],
    max_episode_steps: Annotated[
        Optional[int], typer.Option(help="Max number of steps to run each episode for.")
    ] = None,
    seed: Annotated[Optional[int], typer.Option(help="Random Seed.")] = None,
    render_mode: Annotated[
        Optional[str], typer.Option(help="Mode to use for rendering.")
    ] = None,
):
    """Run random agents."""
    if max_episode_steps is not None:
        env = posggym.make(
            env_id, render_mode=render_mode, max_episode_steps=max_episode_steps
        )
    else:
        env = posggym.make(env_id, render_mode=render_mode)

    env.reset(seed=seed)

    dones = 0
    episode_steps = []
    episode_rewards: Dict[str, List[float]] = {i: [] for i in env.possible_agents}
    for ep_num in range(num_episodes):
        env.render()

        t = 0
        done = False
        rewards = {i: 0.0 for i in env.possible_agents}
        while not done and (max_episode_steps is None or t < max_episode_steps):
            a = {i: env.action_spaces[i].sample() for i in env.agents}
            _, r, _, _, done, _ = env.step(a)
            t += 1

            env.render()

            for i, r_i in r.items():
                rewards[i] += r_i

        print(f"End episode {ep_num}")
        dones += int(done)
        episode_steps.append(t)

        env.reset()

        for i, r_i in rewards.items():
            episode_rewards[i].append(r_i)

        if done:
            print(t, rewards)

    env.close()

    print("All episodes finished")
    print(
        f"Completed episodes (i.e. where 'done=True') = {dones} out of {num_episodes}"
    )
    mean_steps = sum(episode_steps) / len(episode_steps)
    print(f"Mean episode steps = {mean_steps:.2f}")
    mean_returns = {i: sum(r) / len(r) for i, r in episode_rewards.items()}
    print(f"Mean Episode returns {mean_returns}")
    return mean_steps, mean_returns


if __name__ == "__main__":
    app()
