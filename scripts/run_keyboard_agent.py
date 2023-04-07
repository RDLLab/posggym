"""Run a keyboard agent on an environment."""
import math
import os.path as osp
import sys
from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame
from gymnasium import spaces

import posggym
import posggym.model as M


grid_world_key_action_map = {
    "Driving-v0": {
        None: 0,
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_RIGHT: 3,
        pygame.K_LEFT: 4,
    },
    "LevelBasedForaging-v2": {
        None: 0,
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
        pygame.K_SPACE: 5,
    },
    "PredatorPrey-v0": {
        None: 0,
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    },
    "PursuitEvasion-v0": {
        None: 0,
        pygame.K_UP: 0,
        pygame.K_DOWN: 1,
        pygame.K_LEFT: 2,
        pygame.K_RIGHT: 3,
    },
    "TwoPaths-v0": {
        None: 1,
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 2,
    },
    "UAV-v0": {
        None: 1,
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 2,
    },
}
grid_world_key_action_map["DrivingGen-v0"] = grid_world_key_action_map["Driving-v0"]


def display_key_action_map(key_action_map):
    """Prints the key action map."""
    print("Key-Action Bindings")
    for k, a in key_action_map.items():
        if k is None:
            print(f"None: {a}")
        else:
            print(f"{pygame.key.name(k)}: {a}")


def run_grid_world_env_keyboard_agent(
    env: posggym.Env,
    keyboard_agent_id: M.AgentID,
) -> Tuple[Dict[str, float], int]:
    """Run keyboard agent in continuous environment.

    Assumes environment actions are angular and linear velocity.
    """
    assert env.spec is not None
    key_action_map = grid_world_key_action_map[env.spec.id]
    env.metadata["render_fps"] = 4

    o, _ = env.reset()
    env.render()

    t = 0
    done = False
    rewards = {i: 0.0 for i in env.possible_agents}
    while not done:
        action_i = key_action_map[None]
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key in key_action_map:
                    action_i = key_action_map[event.key]
                elif (
                    event.key == pygame.K_c and pygame.key.get_mods() & pygame.KMOD_CTRL
                ):
                    # exit on control-c
                    env.close()
                    sys.exit()

        actions = {}
        for i in env.agents:
            if i == keyboard_agent_id:
                actions[i] = action_i
            else:
                actions[i] = env.action_spaces[i].sample()

        _, r, _, _, done, _ = env.step(actions)
        t += 1

        for i, r_i in r.items():
            rewards[i] += r_i  # type: ignore

        env.render()

    return rewards, t


def run_continuous_env_keyboard_agent(
    env: posggym.Env,
    keyboard_agent_id: M.AgentID,
) -> Tuple[Dict[str, float], int]:
    """Run keyboard agent in continuous environment.

    Assumes environment actions are angular and linear velocity.
    """
    import pygame

    angular_vel_inc = math.pi / 10
    linear_vel_inc = 0.25

    o, _ = env.reset()
    env.render()

    t = 0
    done = False
    rewards = {i: 0.0 for i in env.possible_agents}
    action_i = np.array([0.0, 0.0], dtype=np.float32)
    while not done:
        # reset angular velocity, but maintain linear velocity
        action_i[0] = 0.0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action_i[0] = -angular_vel_inc
        elif keys[pygame.K_RIGHT]:
            action_i[0] = +angular_vel_inc

        if keys[pygame.K_UP]:
            action_i[1] = linear_vel_inc
        elif keys[pygame.K_DOWN]:
            action_i[1] = -linear_vel_inc

        if keys[pygame.K_c] and pygame.key.get_mods() & pygame.KMOD_CTRL:
            # exit on control-c
            env.close()
            sys.exit()

        actions = {}
        for i in env.agents:
            if i == keyboard_agent_id:
                actions[i] = action_i
            else:
                actions[i] = env.action_spaces[i].sample()

        _, r, _, _, done, _ = env.step(actions)
        t += 1

        for i, r_i in r.items():
            rewards[i] += r_i  # type: ignore

        env.render()

    return rewards, t


def run_keyboard_agent(
    env_id: str,
    keyboard_agent_ids: List[M.AgentID],
    num_episodes: int,
    max_episode_steps: Optional[int] = None,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    pause_each_step: bool = False,
    record_env: bool = False,
):
    """Run keyboard agents."""
    if max_episode_steps is not None:
        env = posggym.make(
            env_id, render_mode=render_mode, max_episode_steps=max_episode_steps
        )
    else:
        env = posggym.make(env_id, render_mode=render_mode)

    # get agent ID's in correct format
    if isinstance(env.possible_agents[0], int):
        keyboard_agent_ids = [int(i) for i in keyboard_agent_ids]

    action_spaces = env.action_spaces

    if env_id in grid_world_key_action_map:
        key_action_map = grid_world_key_action_map[env_id]
        display_key_action_map(key_action_map)
        run_env_episode_fn = run_grid_world_env_keyboard_agent
    elif all(isinstance(action_spaces[i], spaces.Box) for i in keyboard_agent_ids):
        run_env_episode_fn = run_continuous_env_keyboard_agent
    else:
        raise AssertionError

    if record_env:
        video_save_dir = osp.join(osp.expanduser("~"), "posggym_video")
        print(f"Saving video to {video_save_dir}")
        name_prefix = f"keyboard-{env_id}"
        env = posggym.wrappers.RecordVideo(env, video_save_dir, name_prefix=name_prefix)

    env.reset(seed=seed)

    episode_steps = []
    episode_rewards: Dict[M.AgentID, List[float]] = {i: [] for i in env.possible_agents}
    for ep_num in range(num_episodes):
        rewards, steps = run_env_episode_fn(
            env, keyboard_agent_id=keyboard_agent_ids[0]
        )
        episode_steps.append(steps)

        for j in env.possible_agents:
            episode_rewards[j].append(rewards[j])

    env.close()

    print("All episodes finished")
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
        "--max_episode_steps",
        type=int,
        default=None,
        help="Max number of steps to run each episode for (default=None)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random Seed.")
    parser.add_argument(
        "--render_mode",
        type=str,
        default=None,
        help="Mode to use for rendering.",
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
