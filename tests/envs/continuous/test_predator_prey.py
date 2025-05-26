"""Specific tests for the PredatorPreyContinuous environment."""
from typing import cast

import numpy as np
import posggym
import pytest
from posggym.envs.continuous.predator_prey_continuous import (
    SUPPORTED_WORLDS,
    PPState,
    PredatorPreyContinuousModel,
)


def test_obs():
    """Check observations are as expected."""
    obs_dist, n_sensors = 3, 4
    env = posggym.make(
        "PredatorPreyContinuous-v0",
        world="5x5",
        num_predators=2,
        num_prey=1,
        prey_strength=2,
        obs_dist=obs_dist,
        n_sensors=n_sensors,
    )
    env.reset(seed=26)

    model = cast(PredatorPreyContinuousModel, env.model)
    state = cast(PPState, env.state)

    state.predator_states[0][:3] = (1, 1, 0)
    state.predator_states[1][:3] = (2, 1, 0)
    state.prey_states[0][:3] = (1, 2, 0)

    obs = model._get_local_obs("0", state)  # pylint: disable=protected-access

    # each predator has four sensors going right, up, left, down
    expected_obstacle_obs = [obs_dist, obs_dist, 1.0, 1.0]
    expected_predator_obs = [
        1.0 - model.world.agent_radius,
        obs_dist,
        obs_dist,
        obs_dist,
    ]
    expected_prey_obs = [obs_dist, 1.0 - model.world.agent_radius, obs_dist, obs_dist]
    assert np.isclose(
        obs,
        np.concatenate(
            (expected_obstacle_obs, expected_predator_obs, expected_prey_obs)
        ),
    ).all()


@pytest.mark.parametrize("world", list(SUPPORTED_WORLDS))
def test_collisions(world):
    """Check no collisions between predators, prey, blocks and world border."""
    env = posggym.make(
        "PredatorPreyContinuous-v0",
        world=world,
        num_predators=4,
        num_prey=5,
        prey_strength=2,
        obs_dist=3,
        n_sensors=1,  # don't need for test, making it smaller speeds tests up
    )
    env.reset(seed=35)

    model = cast(PredatorPreyContinuousModel, env.model)
    size = model.world.agent_radius
    # pypunk munk allows overlaps, but only typically ~10-15%, so if overlap is as big
    # as agent radius then something is wrong
    min_dist = 2 * size * 0.80

    for _ in range(100):
        state = cast(PPState, env.state)

        for i in range(model.num_predators + model.num_prey):
            if i < model.num_predators:
                pos_i = state.predator_states[i]
            elif not state.prey_caught[i - model.num_predators]:
                pos_i = state.prey_states[i - model.num_predators]
            else:
                continue

            assert size / 2 <= pos_i[0] <= model.world.size - size / 2
            assert size / 2 <= pos_i[1] <= model.world.size - size / 2

            for j in range(model.num_predators + model.num_prey):
                if i == j:
                    continue
                if j < model.num_predators:
                    pos_j = state.predator_states[j]
                elif not state.prey_caught[j - model.num_predators]:
                    pos_j = state.prey_states[j - model.num_predators]
                else:
                    continue
                assert model.world.euclidean_dist(pos_i, pos_j) >= min_dist

            for pos_b, size_b in model.world.blocks:
                assert (
                    model.world.euclidean_dist(pos_i, pos_b) >= (size + size_b) * 0.85
                )

        a = {i: env.action_spaces[i].sample() for i in env.agents}
        _, _, _, _, all_done, _ = env.step(a)

        if all_done:
            env.reset()

    env.close()


if __name__ == "__main__":
    test_obs()
