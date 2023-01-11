"""Run a random agent on an environment."""
import argparse
from typing import Optional, Dict, List

import posggym
import posggym.model as M


# def _get_outcome_counts(episode_outcomes, n_agents):
#     outcome_counts = {k: [0 for _ in range(n_agents)] for k in M.Outcome}
#     for outcome in episode_outcomes:
#         if outcome is None:
#             outcome = tuple(M.Outcome.NA for _ in range(n_agents))
#         for i in range(n_agents):
#             outcome_counts[outcome[i]][i] += 1
#     return outcome_counts


def run_random_agent(
    env_id: str,
    num_episodes: int,
    episode_step_limit: Optional[int] = None,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    pause_each_step: bool = False,
    record_env: bool = False,
):
    """Run random agents."""
    if episode_step_limit is not None:
        env = posggym.make(
            env_id, render_mode=render_mode, max_episode_steps=episode_step_limit
        )
    else:
        env = posggym.make(env_id, render_mode=render_mode)

    # if record_env:
    #     video_save_dir = os.path.join(os.path.expanduser("~"), "posggym_video")
    #     print(f"Saving video to {video_save_dir}")
    #     env = wrappers.RecordVideo(env, video_save_dir)

    env.reset(seed=seed)

    dones = 0
    episode_steps = []
    episode_rewards: Dict[M.AgentID, List[float]] = {i: [] for i in env.possible_agents}
    # episode_outcomes = []  # type: ignore
    for ep_num in range(num_episodes):
        env.reset()

        if render_mode:
            env.render()

        if pause_each_step:
            input("Press any key")

        t = 0
        done = False
        rewards = {i: 0.0 for i in env.possible_agents}
        while episode_step_limit is None or t < episode_step_limit:
            a = {i: env.action_spaces[i].sample() for i in env.agents}
            _, r, _, _, done, _ = env.step(a)
            t += 1

            if render_mode:
                env.render()

            if pause_each_step:
                input("Press any key")

            if done:
                print(f"End episode {ep_num}")
                break

            for i, r_i in r.items():
                rewards[i] += r_i   # type: ignore

        dones += int(done)
        episode_steps.append(t)
        # episode_outcomes.append(info.get("outcome", None))

        for i, r_i in rewards.items():
            episode_rewards[i].append(r_i)

    env.close()

    print("All episodes finished")
    print(f"Episodes ending with 'done=True' = {dones} out of {num_episodes}")
    mean_steps = sum(episode_steps) / len(episode_steps)
    print(f"Mean episode steps = {mean_steps:.2f}")
    mean_returns = {i: sum(r) / len(r) for i, r in episode_rewards.items()}
    print(f"Mean Episode returns {mean_returns}")

    # outcome_counts = _get_outcome_counts(episode_outcomes, env.n_agents)
    # print("Outcomes")
    # for k, v in outcome_counts.items():
    #     print(f"{k} = {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("env_id", type=str, help="ID of environment to run")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="The number of episodes to run.",
    )
    parser.add_argument(
        "--episode_step_limit",
        type=int,
        default=None,
        help="Max number of steps to run each epsiode for.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random Seed."
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default=None,
        help="Mode to use for renderering.",
    )
    parser.add_argument(
        "--pause_each_step", action="store_true", help="Pause execution after each step"
    )
    parser.add_argument(
        "--record_env",
        action="store_true",
        help="Record video of environment saved to ~/posggym_videos.",
    )
    args = parser.parse_args()
    run_random_agent(**vars(args))
