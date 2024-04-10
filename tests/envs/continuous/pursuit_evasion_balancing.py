"""Script for doing balance testing on the pursuit evasion environment."""

import argparse
import sys
from itertools import product
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import posggym
import pygame
from posggym.envs.continuous.pursuit_evasion_continuous import PEWorld

key_action_map = {
    None: 0,
    pygame.K_UP: np.array([0.0, 1.0], dtype=np.float32),
    pygame.K_DOWN: np.array([0.0, 0.0], dtype=np.float32),
    pygame.K_RIGHT: np.array([np.pi / 4, 0.0], dtype=np.float32),
    pygame.K_LEFT: np.array([-np.pi / 4, 0.0], dtype=np.float32),
    pygame.K_w: np.array([0.0, 1.0], dtype=np.float32),  # forward
    pygame.K_s: np.array([0.0, 0.0], dtype=np.float32),  # do nothing
    pygame.K_d: np.array([np.pi / 4, 0.0], dtype=np.float32),  # right
    pygame.K_a: np.array([-np.pi / 4, 0.0], dtype=np.float32),  # left
    pygame.K_e: np.array([np.pi / 4, 1.0], dtype=np.float32),  # right forward
    pygame.K_q: np.array([-np.pi / 4, 1.0], dtype=np.float32),  # left forward
}


def run_keyboard_agent(
    env: posggym.Env, keyboard_agent_id: List[str]
) -> Optional[Tuple[Dict[str, float], int]]:
    """Run manual keyboard agent in continuous environment.

    Assumes environment actions are continuous (i.e. space.Box). So user will be
    prompted to input an floating point actions for each controlled agent (possibly
    multiple space separate value if action space is multi-dimensional).
    """
    env.reset()
    env.render()
    t = 0
    done = False
    rewards = {i: 0.0 for i in env.possible_agents}
    while not done:
        actions = {}
        for i in env.agents:
            if i in keyboard_agent_id:
                action_i = key_action_map[None]
                action_entered = False
                while not action_entered:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key in key_action_map:
                                print(f"Agent {i} action: {key_action_map[event.key]}")
                                action_i = key_action_map[event.key]
                                action_entered = True
                            elif (
                                event.key == pygame.K_c
                                and pygame.key.get_mods() & pygame.KMOD_CTRL
                            ):
                                # exit on control-c
                                env.close()
                                sys.exit()
                            elif event.key == pygame.K_n:
                                # skip episode
                                return None
                actions[i] = action_i
            else:
                actions[i] = env.action_spaces[i].sample()

        _, r, _, _, done, _ = env.step(actions)
        t += 1

        for i, r_i in r.items():
            rewards[i] += r_i  # type: ignore

        env.render()

    print(f"Episode length: {t}")
    print(f"Episode rewards: {rewards}")
    return rewards, t


def run_world_balancing(world_name: str, num_episodes_per_world: int):
    """Run world balancing test on pursuit evasion environment."""
    original_env = posggym.make(
        "PursuitEvasionContinuous-v0",
        world=world_name,
        fov=np.pi / 3,
    )
    original_world = cast(PEWorld, original_env.model.world)
    max_obs_distance = original_world.size / 3

    for evader_coord, pursue_coord in product(
        original_world.evader_start_coords, original_world.pursuer_start_coords
    ):
        for goal_coord in original_world.get_goal_coords(evader_coord):
            print(f"Running world {world_name} with:")
            print(f"fov: {np.pi / 3}")
            print(f"max_obs_distance: {max_obs_distance}")
            print(f"Evader start coord: {evader_coord}")
            print(f"Pursuer start coord: {pursue_coord}")
            print(f"Goal coord: {goal_coord}")
            world = PEWorld(
                size=original_world.size,
                blocked_coords=original_world.blocked_coords,
                evader_start_coords=[evader_coord],
                pursuer_start_coords=[pursue_coord],
                goal_coords_map={evader_coord: [goal_coord]},
            )
            env = posggym.make(
                "PursuitEvasionContinuous-v0",
                world=world,
                fov=np.pi / 3,
                max_obs_distance=max_obs_distance,
                render_mode="human",
            )
            env.reset(seed=5)

            rewards = {i: 0.0 for i in env.possible_agents}
            episode_lengths = []
            for ep_num in range(num_episodes_per_world):
                print(f"Running episode {ep_num + 1} of {num_episodes_per_world}")
                result = run_keyboard_agent(env, env.possible_agents)
                if result is not None:
                    episode_rewards, episode_length = result
                    for i, r_i in episode_rewards.items():
                        rewards[i] += r_i  # type: ignore
                    episode_lengths.append(episode_length)

            print(
                f"Average episode length: {np.mean(episode_lengths):.2f} +/- "
                f"{np.std(episode_lengths):.2f}"
            )
            for i in env.possible_agents:
                print(
                    f"Agent {i} average reward: "
                    f"{rewards[i] / num_episodes_per_world:.2f} +/- "
                    f"{np.std(rewards[i]) / num_episodes_per_world:.2f}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world",
        type=str,
        help="World to run balancing test on.",
    )
    parser.add_argument(
        "--num_episodes_per_world",
        type=int,
        default=4,
        help="Number of episodes to run per world.",
    )
    args = parser.parse_args()
    run_world_balancing(args.world, args.num_episodes_per_world)
