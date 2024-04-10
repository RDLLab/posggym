"""Run a keyboard agent in an environment.

Able to run keyboard agents for both grid-world and continuous environments.

To see all available arguments, run:

    python run_keyboard_agent.py --help

Example, to run a keyboard agent in the `Driving-v1` environment while controlling
agent '0' for 10 episodes, run:

    python run_keyboard_agent.py \
        --env_id Driving-v1 \
        --keyboard_agent_ids 0 \
        --num_episodes 10

"""

import argparse
import math
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame
from gymnasium import spaces

import posggym


grid_world_key_action_map = {
    "Driving-v1": {
        None: 0,
        pygame.K_SPACE: 0,
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_RIGHT: 3,
        pygame.K_LEFT: 4,
    },
    "LevelBasedForaging-v2": {
        None: 0,
        pygame.K_SPACE: 0,
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
        pygame.K_SPACE: 5,
    },
    "PredatorPrey-v0": {
        None: 0,
        pygame.K_SPACE: 0,
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    },
    "PursuitEvasion-v0": {
        None: 0,
        pygame.K_SPACE: 0,
        pygame.K_UP: 0,
        pygame.K_DOWN: 1,
        pygame.K_LEFT: 2,
        pygame.K_RIGHT: 3,
    },
    "TwoPaths-v0": {
        None: 1,
        pygame.K_SPACE: 0,
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 2,
    },
    "UAV-v0": {
        None: 1,
        pygame.K_SPACE: 0,
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 2,
    },
}
grid_world_key_action_map["DrivingGen-v1"] = grid_world_key_action_map["Driving-v1"]


def display_key_action_map(key_action_map):
    """Prints the key action map."""
    print("Key-Action Bindings")
    for k, a in key_action_map.items():
        if k is None:
            print(f"None: {a}")
        else:
            print(f"{pygame.key.name(k)}: {a}")


def display_vector_obs(obs: np.ndarray, width: int):
    """Prints vector obs. Useful for debugging."""
    with np.printoptions(precision=2, suppress=True):
        print()
        if width >= obs.shape[0]:
            print(obs)
        else:
            for end in range(width, obs.shape[0], width):
                print(obs[end - width : end])
            if width % obs.shape[0]:
                print(obs[end:])


def run_discrete_env_manual_keyboard_agent(
    env: posggym.Env, keyboard_agent_id: List[str], pause_each_step: bool = False
) -> Tuple[Dict[str, float], int]:
    """Run manual keyboard agent in discrete environment.

    Assumes environment actions are discrete. So user will be prompted to input an
    integer action for each controlled agent.
    """
    assert env.spec is not None
    env.metadata["render_fps"] = 4

    env.reset()
    env.render()

    t = 0
    done = False
    rewards = {i: 0.0 for i in env.possible_agents}
    while not done:
        actions = {}
        for i in env.agents:
            if i in keyboard_agent_id:
                n_actions = env.action_spaces[i].n
                action_entered = False
                while not action_entered:
                    action_str = input(
                        f"Enter action for agent '{i}' [0, {n_actions-1}] (q to quit): "
                    )
                    if action_str == "q":
                        env.close()
                        sys.exit()
                    try:
                        action_i = int(action_str)
                        if 0 <= action_i < n_actions:
                            action_entered = True
                        else:
                            print("Invalid action.")
                    except ValueError:
                        print("Invalid action.")

                actions[i] = action_i
            else:
                actions[i] = env.action_spaces[i].sample()

        _, r, _, _, done, _ = env.step(actions)
        t += 1

        for i, r_i in r.items():
            rewards[i] += r_i  # type: ignore

        env.render()

    return rewards, t


def run_continuous_env_manual_keyboard_agent(
    env: posggym.Env, keyboard_agent_id: str, pause_each_step: bool = False
) -> Tuple[Dict[str, float], int]:
    """Run manual keyboard agent in continuous environment.

    Assumes environment actions are continuous (i.e. space.Box). So user will be
    prompted to input an floating point actions for each controlled agent (possibly
    multiple space separate value if action space is multi-dimensional).
    """
    assert env.spec is not None

    env.reset()
    env.render()

    t = 0
    done = False
    rewards = {i: 0.0 for i in env.possible_agents}
    while not done:
        actions = {}
        for i in env.agents:
            if i in keyboard_agent_id:
                with np.printoptions(precision=4, suppress=True):
                    low_str = str(env.action_spaces[i].low)
                    high_str = str(env.action_spaces[i].high)

                action_entered = False
                while not action_entered:
                    action_str = input(
                        f"Enter action for agent '{i}' low={low_str}, high={high_str} "
                        "(q to quit): "
                    )
                    print(action_str)
                    if action_str == "q":
                        env.close()
                        sys.exit()
                    try:
                        action_i = np.array(
                            [float(x) for x in action_str.split()],
                            dtype=env.action_spaces[i].dtype,
                        )
                        print(action_str, action_str.split(), action_i)
                        if env.action_spaces[i].contains(action_i):
                            action_entered = True
                        else:
                            print("Invalid action.")
                    except ValueError:
                        print("Invalid action.")

                actions[i] = action_i
            else:
                actions[i] = env.action_spaces[i].sample()

        _, r, _, _, done, _ = env.step(actions)
        t += 1

        for i, r_i in r.items():
            rewards[i] += r_i  # type: ignore

        env.render()

    return rewards, t


def run_grid_world_env_keyboard_agent(
    env: posggym.Env, keyboard_agent_id: str, pause_each_step: bool = False
) -> Tuple[Dict[str, float], int]:
    """Run keyboard agent in grid-world environment.

    Assumes environment actions are angular and linear velocity.
    """
    assert env.spec is not None
    key_action_map = grid_world_key_action_map[env.spec.id]
    env.metadata["render_fps"] = 4

    env.reset()
    env.render()

    t = 0
    done = False
    rewards = {i: 0.0 for i in env.possible_agents}
    while not done:
        action_entered = False
        while not action_entered:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key in key_action_map:
                        action_entered = True
                        action_i = key_action_map[event.key]
                    elif (
                        event.key == pygame.K_c
                        and pygame.key.get_mods() & pygame.KMOD_CTRL
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
    env: posggym.Env, keyboard_agent_id: str, pause_each_step: bool = False
) -> Tuple[Dict[str, float], int]:
    """Run keyboard agent in continuous environment.

    Assumes environment actions are angular and linear velocity.
    """
    import pygame

    use_linear_acc = env.spec is not None and env.spec.id in ("DrivingContinuous-v0",)

    angle_inc = math.pi / 4
    vel_inc = 0.1

    o, _ = env.reset()
    env.render()

    t = 0
    done = False
    rewards = {i: 0.0 for i in env.possible_agents}
    action_i = np.array([0.0, 0.0], dtype=np.float32)

    while not done:
        if pause_each_step:
            # wait till key pressed
            action_entered = False
            while not action_entered:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        action_entered = True
                        break

        action_i[0] = 0.0
        action_i[1] = 0.0 if use_linear_acc else action_i[1]
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action_i[0] = -angle_inc
        elif keys[pygame.K_RIGHT]:
            action_i[0] = +angle_inc

        if keys[pygame.K_UP]:
            action_i[1] = vel_inc if use_linear_acc else action_i[1] + vel_inc
        elif keys[pygame.K_DOWN]:
            action_i[1] = -vel_inc if use_linear_acc else action_i[1] - vel_inc

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

        o, r, _, _, done, _ = env.step(actions)
        t += 1

        for i, r_i in r.items():
            rewards[i] += r_i  # type: ignore

        env.render()

    return rewards, t


def run_keyboard_agent(
    env_id: str,
    keyboard_agent_ids: List[str],
    num_episodes: int,
    max_episode_steps: Optional[int] = None,
    seed: Optional[int] = None,
    pause_each_step: bool = False,
    manual_input: bool = False,
):
    """Run keyboard agents."""
    if max_episode_steps is not None:
        env = posggym.make(
            env_id, render_mode="human", max_episode_steps=max_episode_steps
        )
    else:
        env = posggym.make(env_id, render_mode="human")

    # get agent ID's in correct format
    if isinstance(env.possible_agents[0], int):
        keyboard_agent_ids = [int(i) for i in keyboard_agent_ids]

    action_spaces = env.action_spaces

    if manual_input and isinstance(
        env.action_spaces[keyboard_agent_ids[0]], spaces.Discrete
    ):
        run_env_episode_fn = run_discrete_env_manual_keyboard_agent
    elif manual_input and isinstance(
        env.action_spaces[keyboard_agent_ids[0]], spaces.Box
    ):
        run_env_episode_fn = run_continuous_env_manual_keyboard_agent
    elif env_id in grid_world_key_action_map:
        key_action_map = grid_world_key_action_map[env_id]
        display_key_action_map(key_action_map)
        run_env_episode_fn = run_grid_world_env_keyboard_agent
    elif all(isinstance(action_spaces[i], spaces.Box) for i in keyboard_agent_ids):
        run_env_episode_fn = run_continuous_env_keyboard_agent
    else:
        raise AssertionError

    env.reset(seed=seed)

    episode_steps = []
    episode_rewards: Dict[str, List[float]] = {i: [] for i in env.possible_agents}
    for _ in range(num_episodes):
        if manual_input:
            rewards, steps = run_env_episode_fn(
                env,
                keyboard_agent_id=keyboard_agent_ids,
                pause_each_step=pause_each_step,
            )
        else:
            rewards, steps = run_env_episode_fn(
                env,
                keyboard_agent_id=keyboard_agent_ids[0],
                pause_each_step=pause_each_step,
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
    parser = argparse.ArgumentParser(
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env_id", type=str, help="Name of environment to run")
    parser.add_argument(
        "-kids",
        "--keyboard_agent_ids",
        type=str,
        default=["0"],
        nargs="+",
        help=(
            "IDs of agents to run as keyboard agents. Controlling multiple agents only "
            "supported when running with `--manual_input`"
        ),
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
        "--pause_each_step", action="store_true", help="Pause execution after each step"
    )
    parser.add_argument(
        "--manual_input",
        action="store_true",
        help=(
            "Manually input action values rather than using keyboard arrows (useful "
            "for executing very specific sequences of actions for testing). Supports "
            "controliing multiple keyboard agents."
        ),
    )
    args = parser.parse_args()
    run_keyboard_agent(**vars(args))
