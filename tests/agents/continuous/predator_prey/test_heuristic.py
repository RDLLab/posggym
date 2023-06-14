"""Tests for the heuristic agent in the predator prey continuous environment."""

import numpy as np
import pytest

import posggym
import posggym.agents as pga


RENDER_MODE = None


def _run_episodes(env, policies, num_episodes: int = 1, close: bool = True):
    ep_returns = {i: np.zeros(num_episodes) for i in env.possible_agents}
    for ep_num in range(num_episodes):
        obs, _ = env.reset()
        for i in env.possible_agents:
            policies[i].reset(seed=42)

        done = False
        while not done:
            actions = {}
            for i in env.agents:
                actions[i] = policies[i].step(obs[i])
            obs, rewards, _, _, done, _ = env.step(actions)
            if RENDER_MODE is not None:
                env.render()
            for i in env.possible_agents:
                ep_returns[i][ep_num] += rewards[i]

    if close:
        env.close()
        for i in env.possible_agents:
            policies[i].close()

    return {i: ep_returns[i].mean() for i in env.possible_agents}


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_heuristic_P2(n):
    """Test the heuristic agent in the predator prey continuous environment."""
    env = posggym.make(
        "PredatorPreyContinuous-v0",
        world="10x10",
        num_predators=2,
        num_prey=3,
        render_mode=RENDER_MODE,
    )
    pi_id = f"PredatorPreyContinuous-v0/heuristic{n}-v0"
    policies = {i: pga.make(pi_id, env.model, agent_id=i) for i in env.possible_agents}

    env.reset(seed=42)
    for i in env.possible_agents:
        policies[i].reset(seed=42)

    mean_returns = _run_episodes(env, policies, num_episodes=1, close=True)
    print(f"Mean returns = {mean_returns[env.possible_agents[0]]}")


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_heuristic_P4(n):
    """Test the heuristic agent in the predator prey continuous environment."""
    env = posggym.make(
        "PredatorPreyContinuous-v0",
        world="10x10",
        num_predators=4,
        num_prey=3,
        prey_strength=3,
        render_mode=RENDER_MODE,
    )
    pi_id = f"PredatorPreyContinuous-v0/heuristic{n}-v0"
    policies = {i: pga.make(pi_id, env.model, agent_id=i) for i in env.possible_agents}

    env.reset(seed=42)
    for i in env.possible_agents:
        policies[i].reset(seed=42)

    mean_returns = _run_episodes(env, policies, num_episodes=1, close=True)
    print(f"Mean returns = {mean_returns[env.possible_agents[0]]}")


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_heuristic_P4_blocks(n):
    """Test the heuristic agent in the predator prey continuous environment."""
    env = posggym.make(
        "PredatorPreyContinuous-v0",
        world="10x10Blocks",
        num_predators=4,
        num_prey=3,
        prey_strength=3,
        render_mode=RENDER_MODE,
    )
    pi_id = f"PredatorPreyContinuous-v0/heuristic{n}-v0"
    policies = {i: pga.make(pi_id, env.model, agent_id=i) for i in env.possible_agents}

    env.reset(seed=42)
    for i in env.possible_agents:
        policies[i].reset(seed=42)

    mean_returns = _run_episodes(env, policies, num_episodes=1, close=True)
    print(f"Mean returns = {mean_returns[env.possible_agents[0]]}")


if __name__ == "__main__":
    # For manual debugging
    RENDER_MODE = "human"
    for fn in [
        # test_heuristic_P2,
        test_heuristic_P4,
        test_heuristic_P4_blocks,
    ]:
        print(f"Testing {fn.__name__}")
        for n in range(4):
            print(f"testing heuristic policy {n}")
            fn(n)
