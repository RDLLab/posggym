"""Run a keyboard agent on an environment."""
from argparse import ArgumentParser
import os.path as osp
from typing import Dict, List, Optional

import numpy as np
from gymnasium import spaces

import posggym
import posggym.model as M


def get_discrete_action(env: posggym.Env, keyboard_agent_ids: List[M.AgentID]):
    """Get discrete action from user."""
    actions = {}
    for i in env.agents:
        action_space: spaces.Discrete = env.action_spaces[i]    # type: ignore
        if i in keyboard_agent_ids:
            while True:
                try:
                    a = input(f"Select action for agent {i} (0, {action_space.n-1}): ")
                    return int(a)
                except ValueError:
                    print("Invalid selection. Try again.")
        else:
            agent_a = action_space.sample()
        actions[i] = agent_a
    return actions


def get_continuous_action(env: posggym.Env, keyboard_agent_ids: List[M.AgentID]):
    """Get continuous action from user."""
    actions = {}
    for i in env.agents:
        action_space: spaces.Box = env.action_spaces[i]   # type: ignore
        if i in keyboard_agent_ids:
            action = []
            for dim in range(action_space.shape[0]):
                low, high = action_space.low[dim], action_space.high[dim]
                while True:
                    try:
                        a = float(
                            input(f"Select action for agent {i} ({low}, {high}): ")
                        )
                        assert low <= a <= high
                        action.append(a)
                        break
                    except (ValueError, AssertionError):
                        print("Invalid selection. Try again.")
            actions[i] = np.array(action)
        else:
            actions[i] = action_space.sample()
    return actions


def run_keyboard_agent(
    env_id: str,
    keyboard_agent_ids: List[M.AgentID],
    num_episodes: int,
    episode_step_limit: Optional[int] = None,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    pause_each_step: bool = False,
    record_env: bool = False,
):
    """Run keyboard agents."""
    if episode_step_limit is not None:
        env = posggym.make(
            env_id, render_mode=render_mode, max_episode_steps=episode_step_limit
        )
    else:
        env = posggym.make(env_id, render_mode=render_mode)

    # get agent ID's in correct format
    if isinstance(env.possible_agents[0], int):
        keyboard_agent_ids = [int(i) for i in keyboard_agent_ids]

    action_spaces = env.action_spaces
    if all(isinstance(action_spaces[i], spaces.Discrete) for i in keyboard_agent_ids):
        get_action_fn = get_discrete_action
    elif all(isinstance(action_spaces[i], spaces.Box) for i in keyboard_agent_ids):
        assert all(
            len(action_spaces[i].shape) == 1   # type: ignore
            for i in keyboard_agent_ids
        ), (
            "Only 1D continous actions supported."
        )
        get_action_fn = get_continuous_action
    else:
        raise AssertionError(
            "Only discrete and 1D continous action spaces supported for keyboard "
            "agents."
        )

    if record_env:
        video_save_dir = osp.join(osp.expanduser("~"), "posggym_video")
        print(f"Saving video to {video_save_dir}")
        name_prefix = f"keyboard-{env_id}"
        env = posggym.wrappers.RecordVideo(env, video_save_dir, name_prefix=name_prefix)

    env.reset(seed=seed)

    dones = 0
    episode_steps = []
    episode_rewards: Dict[M.AgentID, List[float]] = {i: [] for i in env.possible_agents}
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
            a = get_action_fn(env, keyboard_agent_ids)
            _, r, _, _, done, _ = env.step(a)
            t += 1

            for i, r_i in r.items():
                rewards[i] += r_i    # type: ignore

            if render_mode:
                env.render()

            if pause_each_step:
                input("Press any key")

            if done:
                print(f"End episode {ep_num}")
                print(f"Returns: {rewards}")
                break

        dones += int(done)
        episode_steps.append(t)

        for j in env.possible_agents:
            episode_rewards[j].append(rewards[j])

    env.close()

    print("All episodes finished")
    print(f"Episodes ending with 'done=True' = {dones} out of {num_episodes}")
    mean_steps = sum(episode_steps) / len(episode_steps)
    print(f"Mean episode steps = {mean_steps:.2f}")
    mean_returns = {i: sum(r) / len(r) for i, r in episode_rewards.items()}
    print(f"Mean Episode returns {mean_returns}")


if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler="resolve")
    parser.add_argument("env_id", type=str, help="Name of environment to run")
    parser.add_argument(
        "keyboard_agent_ids",
        type=str,
        nargs="+",
        help="IDs of agents to run as keyboard agents.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="The number of episodes to run (default=1)",
    )
    parser.add_argument(
        "--episode_step_limit",
        type=int,
        default=None,
        help="Max number of steps to run each epsiode for (default=None)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random Seed.")
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
    run_keyboard_agent(**vars(args))
