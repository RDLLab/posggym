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

import argparse
from typing import Dict, List, Optional

import posggym


def run_random_agent(
    env_id: str,
    num_episodes: int,
    max_episode_steps: Optional[int] = None,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--env_id", type=str, required=True, help="ID of environment to run"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="The number of episodes to run.",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=None,
        help="Max number of steps to run each episode for.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random Seed.")
    parser.add_argument(
        "--render_mode",
        type=str,
        default=None,
        help="Mode to use for rendering.",
    )
    args = parser.parse_args()
    run_random_agent(**vars(args))
