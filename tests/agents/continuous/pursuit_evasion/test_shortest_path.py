"""Tests for the shortest path policy in pursuit-evasion continuous environment."""

import numpy as np
import pytest

import posggym
import posggym.agents as pga
from posggym.agents.utils.action_distributions import DeterministicActionDistribution


RENDER_MODE = None

np.set_printoptions(precision=2, suppress=True)


class ConstantPolicy(pga.Policy):
    """A policy that always returns the same action."""

    def __init__(self, model, agent_id, policy_id, action):
        super().__init__(model, agent_id, policy_id)
        self.action = action

    def step(self, obs):
        return self.action

    def get_next_state(self, obs, state):
        return {"action": self.action}

    def sample_action(self, state):
        return self.action

    def get_pi(self, state):
        return DeterministicActionDistribution(self.action)

    def get_value(self, state):
        return 0.0


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


@pytest.mark.parametrize("agent_id", ["0", "1"])
@pytest.mark.parametrize("world", ["8x8", "16x16", "32x32"])
def test_shortest_path(agent_id, world):
    """Test the shortest path policy in the pursuit evasion continuous environment."""
    env = posggym.make(
        "PursuitEvasionContinuous-v0",
        world=world,
        render_mode=RENDER_MODE,
    )
    pi_id = "PursuitEvasionContinuous-v0/shortest_path-v0"

    policies = {}
    for i in env.possible_agents:
        if i == agent_id:
            policies[i] = pga.make(pi_id, env.model, agent_id=i)
        else:
            policies[i] = ConstantPolicy(env.model, i, pi_id, np.zeros(2))

    env.reset(seed=42)
    for i in env.possible_agents:
        policies[i].reset(seed=42)

    mean_returns = _run_episodes(env, policies, num_episodes=1, close=True)
    print(f"Mean returns = {mean_returns[agent_id]}")


if __name__ == "__main__":
    # For manual debugging
    # RENDER_MODE = "human"
    for fn in [
        test_shortest_path,
    ]:
        print(f"Testing {fn.__name__}")
        for world in [
            # worlds
            "8x8",
            "16x16",
            "32x32",
        ]:
            print("testing world", world)
            for agent_id in [
                # agents
                "0",
                "1",
            ]:
                print("testing agent", agent_id)
                fn(agent_id, world)
