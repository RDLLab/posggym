"""Specific tests for the PredatorPreyDifferentiable-v0 environment."""

import numpy as np
import posggym
import pytest
import torch


@pytest.mark.parametrize("num_predators", [2, 3, 4, 8])
@torch.no_grad()
def test_obs_steps(num_predators: int):
    """Check observations are as expected after reset."""
    BATCH_SIZE = 10

    env = posggym.make(
        "PredatorPreyDifferentiable-v0",
        max_episode_steps=2,
        batch_size=BATCH_SIZE,
        num_predators=num_predators,
        disable_env_checker=True,
    )
    env.reset(seed=35)

    def batch_samples(a_s):
        return np.array([a_s.sample() for _ in range(BATCH_SIZE)])

    for _ in range(10):
        a = {i: torch.Tensor(batch_samples(env.action_spaces[i])) for i in env.agents}

        obs, _, _, _, all_done, _ = env.step(a)

        for i, o_i in obs.items():
            for b in range(BATCH_SIZE):
                assert env.observation_spaces[i].contains(
                    o_i[b].detach().cpu().numpy().squeeze()
                ), f"Agent {i} observation {o_i[b]} is not in its observation space."

    env.close()


def run_grad_step(num_predators: int):
    """Check observations are as expected after reset."""
    BATCH_SIZE = 10

    env = posggym.make(
        "PredatorPreyDifferentiable-v0",
        max_episode_steps=40,
        batch_size=BATCH_SIZE,
        num_predators=num_predators,
        disable_env_checker=True,
    )
    env.reset(seed=35)

    def batch_samples(a_s):
        return np.array([a_s.sample() for _ in range(BATCH_SIZE)])

    for t in range(5):
        a = {i: torch.Tensor(batch_samples(env.action_spaces[i])) for i in env.agents}

        for action in a.values():
            action.requires_grad_(True)

        if t == 0:
            first_action = a

        obs, rews, _, _, all_done, _ = env.step(a)

        assert not all_done.all()

    loss = obs["agent_0"].mean() + rews["agent_0"].mean()
    grad = torch.autograd.grad(loss, first_action["agent_0"], allow_unused=True)

    assert grad is not None
    assert abs(grad[0]).sum() > 0

    env.close()

    return grad[0]


def run_model_step(num_predators: int):
    """Check observations are as expected after reset."""
    BATCH_SIZE = 10

    env = posggym.make(
        "PredatorPreyDifferentiable-v0",
        max_episode_steps=40,
        batch_size=BATCH_SIZE,
        num_predators=num_predators,
        disable_env_checker=True,
    )
    env.reset(seed=35)

    def batch_samples(a_s):
        return np.array([a_s.sample() for _ in range(BATCH_SIZE)])

    for t in range(5):
        a = {i: torch.Tensor(batch_samples(env.action_spaces[i])) for i in env.agents}

        for action in a.values():
            action.requires_grad_(True)

        if t == 0:
            first_action = a

        state = env.model.sample_initial_state()
        state = env.model.step(state, a).state
        state = env.model.step(state, a).state
        state = env.model.step(state, a).state
        state = env.model.step(state, a).state

        obs, rews, _, _, all_done, _ = env.step(a)

        assert not all_done.all()

    loss = obs["agent_0"].mean() + rews["agent_0"].mean()
    grad = torch.autograd.grad(loss, first_action["agent_0"], allow_unused=True)

    assert grad is not None
    assert abs(grad[0]).sum() > 0

    env.close()

    return grad[0]


@pytest.mark.parametrize("num_predators", [3])
def test_grad(num_predators: int):
    run_grad_step(num_predators)


@pytest.mark.parametrize("num_predators", [3])
def test_model_step(num_predators: int):
    run_model_step(num_predators)


@pytest.mark.parametrize("num_predators", [3])
def test_compare_gradients(num_predators: int):
    """Compare gradients from test_grad and test_model_step."""
    g1 = run_grad_step(num_predators)
    g2 = run_model_step(num_predators)

    assert torch.allclose(
        g1, g2
    ), "Gradients from test_grad and test_model_step do not match."
