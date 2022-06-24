"""Run a random agent on an environment."""
import os
import argparse
from typing import Optional

import posggym
import posggym.model as M
from posggym import wrappers


def _get_outcome_counts(episode_outcomes, n_agents):
    outcome_counts = {k: [0 for _ in range(n_agents)] for k in M.Outcome}
    for outcome in episode_outcomes:
        if outcome is None:
            outcome = tuple(M.Outcome.NA for _ in range(n_agents))
        for i in range(n_agents):
            outcome_counts[outcome[i]][i] += 1
    return outcome_counts


def main(env_name: str,
         num_episodes: int,
         episode_step_limit: int,
         seed: Optional[int],
         render: bool,
         render_mode: str,
         pause_each_step: bool,
         record_env: bool):
    """Run random agents."""
    env = posggym.make(env_name)

    if record_env:
        video_save_dir = os.path.join(os.path.expanduser('~'), "posggym_video")
        print(f"Saving video to {video_save_dir}")
        env = wrappers.RecordVideo(env, video_save_dir)

    action_spaces = env.action_spaces

    # set random seeds
    if seed is not None:
        env.reset(seed=seed)
        for i in range(len(action_spaces)):
            action_spaces[i].seed(seed+1+i)

    dones = 0
    episode_steps = []
    episode_rewards = [[] for _ in range(env.n_agents)]    # type: ignore
    episode_outcomes = []   # type: ignore
    for i in range(num_episodes):

        env.reset()

        if render:
            env.render(render_mode)

        if pause_each_step:
            input("Press any key")

        t = 0
        done = False
        rewards = [0.0] * env.n_agents
        while episode_step_limit is None or t < episode_step_limit:
            a = tuple(a.sample() for a in action_spaces)
            _, r, done, info = env.step(a)
            t += 1

            if render:
                env.render(render_mode)

            if pause_each_step:
                input("Press any key")

            if done:
                print(f"End episode {i}")
                break

            for j in range(env.n_agents):
                rewards[j] += r[j]

        dones += int(done)
        episode_steps.append(t)
        episode_outcomes.append(info.get("outcome", None))

        for j in range(env.n_agents):
            episode_rewards[j].append(rewards[j])

    env.close()

    print("All episodes finished")
    print(f"Episodes ending with 'done=True' = {dones} out of {num_episodes}")
    mean_steps = sum(episode_steps) / len(episode_steps)
    print(f"Mean episode steps = {mean_steps:.2f}")
    mean_returns = [sum(r) / len(r) for r in episode_rewards]
    print(f"Mean Episode returns {mean_returns}")

    outcome_counts = _get_outcome_counts(episode_outcomes, env.n_agents)
    print("Outcomes")
    for k, v in outcome_counts.items():
        print(f"{k} = {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "env_name", type=str,
        help="Name of environment to run"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=1,
        help="The number of episodes to run (default=1)"
    )
    parser.add_argument(
        "--episode_step_limit", type=int, default=None,
        help="Max number of steps to run each epsiode for (default=None)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random Seed (default=None)"
    )
    parser.add_argument(
        "--render", action='store_true',
        help="Render environment steps"
    )
    parser.add_argument(
        "--render_mode", type=str, default='human',
        help="Mode to use for renderering, if rendering (default='human')"
    )
    parser.add_argument(
        "--pause_each_step", action='store_true',
        help="Pause execution after each step"
    )
    parser.add_argument(
        "--record_env", action='store_true',
        help="Record video of environment saved to ~/posggym_videos."
    )
    args = parser.parse_args()
    main(**vars(args))
