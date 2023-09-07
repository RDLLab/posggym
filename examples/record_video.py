"""Record a video of an environment.

This script runs an environment using random agents and records a video of the
interaction. Videos will be saved into the `~/posggym_video` directory. The video
recording is done using the `RecordVideo` wrapper.

The script takes a number of arguments (number of episodes, environment id, seed, etc.).
To see all available arguments, run:

    python record_video.py --help

Example, to record 10 episodes of the `Driving-v0` environment run,

    python run_random_agents.py \
        --env_id Driving-v0 \
        --num_episodes 10
"""
import os.path as osp
from typing import Dict, List, Optional
from typing_extensions import Annotated

import typer
import posggym

app = typer.Typer()


@app.command()
def record_env(
    env_id: Annotated[
        str,
        typer.Option(
            help="ID of environment to run, if None then runs all registered envs."
        ),
    ],
    num_episodes: Annotated[
        int, typer.Option(help="The number of episodes to run.")
    ] = 1,
    max_episode_steps: Annotated[
        Optional[int], typer.Option(help="Max number of steps to run each episode for.")
    ] = None,
    seed: Annotated[Optional[int], typer.Option(help="Random Seed")] = None,
):
    """Run random agents."""
    if max_episode_steps is not None:
        env = posggym.make(
            env_id, render_mode="rgb_array", max_episode_steps=max_episode_steps
        )
    else:
        env = posggym.make(env_id, render_mode="rgb_array")

    env_id = env.spec.id if env.spec is not None else str(env)

    video_save_dir = osp.join(osp.expanduser("~"), "posggym_video")
    print(f"Saving video to {video_save_dir}")
    name_prefix = f"{env_id}"
    env = posggym.wrappers.RecordVideo(env, video_save_dir, name_prefix=name_prefix)

    env.reset(seed=seed)

    dones = 0
    episode_steps = []
    episode_rewards: Dict[str, List[float]] = {i: [] for i in env.possible_agents}
    for ep_num in range(num_episodes):
        t = 0
        done = False
        rewards = {i: 0.0 for i in env.possible_agents}
        while max_episode_steps is None or t < max_episode_steps:
            a = {i: env.action_spaces[i].sample() for i in env.agents}
            _, r, _, _, done, _ = env.step(a)
            t += 1

            if done:
                print(f"End episode {ep_num}")
                break

            for i, r_i in r.items():
                rewards[i] += r_i  # type: ignore

        dones += int(done)
        episode_steps.append(t)

        env.reset()

        for i, r_i in rewards.items():
            episode_rewards[i].append(r_i)

    env.close()

    print("All episodes finished")
    print(f"Episodes ending with 'done=True' = {dones} out of {num_episodes}")
    mean_steps = sum(episode_steps) / len(episode_steps)
    print(f"Mean episode steps = {mean_steps:.2f}")
    mean_returns = {i: sum(r) / len(r) for i, r in episode_rewards.items()}
    print(f"Mean Episode returns {mean_returns}")
    return mean_steps, mean_returns


if __name__ == "__main__":
    app()
