"""Specific tests for the PursuitEvasionContinuous environment."""
from typing import cast

import numpy as np

import posggym
from posggym.envs.continuous.pursuit_evasion_continuous import (
    PEState,
    PEWorld,
    PursuitEvasionContinuousModel,
)


def test_obs():
    """Check observations are as expected."""
    size, max_obs_dist, n_sensors = 8, 4.0, 16

    evader_start_coord = (0.5, 0.5)
    goal_coord = ((size // 2) + 0.5, (size // 2) + 0.5)
    pursuer_start_coord = (size - 0.5, size - 0.5)

    # Empty world with evader in top left and pursuer in bottom right
    # And goal in center
    pe_world = PEWorld(
        size,
        blocked_coords=set(),
        goal_coords_map={evader_start_coord: [goal_coord]},
        evader_start_coords=[evader_start_coord],
        pursuer_start_coords=[pursuer_start_coord],
    )

    env = posggym.make(
        "PursuitEvasionContinuous-v0",
        world=pe_world,
        max_obs_distance=max_obs_dist,
        fov=1.57,
        n_sensors=n_sensors,
        normalize_reward=True,
        use_progress_reward=True,
        render_mode="human",
    )
    env.reset(seed=26)

    state = cast(PEState, env.state)

    # Check state is as expected
    assert np.allclose(state.evader_state[:3], evader_start_coord + (0,))
    assert np.allclose(state.pursuer_state[:3], pursuer_start_coord + (0,))
    assert np.allclose(state.evader_start_coord, evader_start_coord)
    assert np.allclose(pursuer_start_coord, pursuer_start_coord)
    assert np.allclose(state.evader_goal_coord, goal_coord)

    # 1. make agents face towards middle
    state.evader_state[2] = 1 / 4 * np.pi
    state.pursuer_state[2] = 5 / 4 * np.pi
    obs, _, _, _, _, _ = env.step(
        {"0": np.array([0.0, 0.0]), "1": np.array([0.0, 0.0])}
    )
    # both agents should have max obs distance for all rays
    n2 = n_sensors * 2
    for i in ["0", "1"]:
        obs_i = obs[i]
        # wall obs
        assert np.allclose(obs_i[:n_sensors], max_obs_dist * np.ones(n_sensors))
        # other agent obs
        assert np.allclose(obs_i[n_sensors:n2], max_obs_dist * np.ones(n_sensors))
        # heard
        assert np.allclose(obs_i[n2], np.zeros(1))
        assert np.allclose(obs_i[n2 + 1 : n2 + 3], evader_start_coord)
        assert np.allclose(obs_i[n2 + 3 : n2 + 5], pursuer_start_coord)
        if i == "0":
            assert np.allclose(obs_i[n2 + 5 : n2 + 7], goal_coord)
        else:
            assert np.allclose(obs_i[n2 + 5 : n2 + 7], np.zeros(2))

    # 2. make agents face towards their corners
    state = cast(PEState, env.state)
    state.evader_state[2] = 5 / 4 * np.pi
    state.pursuer_state[2] = 1 / 4 * np.pi
    obs, _, _, _, _, _ = env.step(
        {"0": np.array([0.0, 0.0]), "1": np.array([0.0, 0.0])}
    )
    # all wall obs should be < 1.0
    for i in ["0", "1"]:
        obs_i = obs[i]
        assert np.all(obs_i[:n_sensors] < 1.0)
        assert np.allclose(obs_i[n_sensors:n2], max_obs_dist * np.ones(n_sensors))
        assert np.allclose(obs_i[n2], np.zeros(1))
        assert np.allclose(obs_i[n2 + 1 : n2 + 3], evader_start_coord)
        assert np.allclose(obs_i[n2 + 3 : n2 + 5], pursuer_start_coord)
        if i == "0":
            assert np.allclose(obs_i[n2 + 5 : n2 + 7], goal_coord)
        else:
            assert np.allclose(obs_i[n2 + 5 : n2 + 7], np.zeros(2))

    # 3. make agents face towards each other and within distance
    state = cast(PEState, env.state)
    state.evader_state[:3] = [3.5, 3.5, 1 / 4 * np.pi]
    state.pursuer_state[:3] = [5.5, 5.5, 5 / 4 * np.pi]
    obs, _, _, _, all_done, _ = env.step(
        {"0": np.array([0.0, 0.0]), "1": np.array([0.0, 0.0])}
    )

    assert all_done  # should end since pursuer observed evader

    # all wall obs should be max obs distance
    # middle agent obs ray should be sqrt(2**2 + 2**2) < 3.0
    # also agent should hear other agent
    n_half = n_sensors // 2
    for i in ["0", "1"]:
        obs_i = obs[i]
        assert np.allclose(obs_i[:n_sensors], max_obs_dist * np.ones(n_sensors))
        assert np.all(obs_i[n_sensors + n_half - 1 : n_sensors + n_half + 1] < 3.0)
        assert np.allclose(obs_i[n2], np.ones(1))
        assert np.allclose(obs_i[n2 + 1 : n2 + 3], evader_start_coord)
        assert np.allclose(obs_i[n2 + 3 : n2 + 5], pursuer_start_coord)
        if i == "0":
            assert np.allclose(obs_i[n2 + 5 : n2 + 7], goal_coord)
        else:
            assert np.allclose(obs_i[n2 + 5 : n2 + 7], np.zeros(2))


def test_shortest_path():
    """Check shortest path is as expected."""
    size, max_obs_dist, n_sensors = 8, 4.0, 16

    evader_start_coord = (0.5, 0.5)
    goal_coord = (size - 0.5, 0.5)
    pursuer_start_coord = (size - 0.5, size - 0.5)

    # Empty world with evader in top left and pursuer in bottom right
    # And goal top right corner
    pe_world = PEWorld(
        size,
        blocked_coords=set(),
        goal_coords_map={evader_start_coord: [goal_coord]},
        evader_start_coords=[evader_start_coord],
        pursuer_start_coords=[pursuer_start_coord],
    )

    env = posggym.make(
        "PursuitEvasionContinuous-v0",
        world=pe_world,
        max_obs_distance=max_obs_dist,
        fov=1.57,
        n_sensors=n_sensors,
        normalize_reward=False,  # don't normalize reward for testing
        use_progress_reward=True,
        render_mode="human",
    )
    env.reset(seed=26)

    model = cast(PursuitEvasionContinuousModel, env.model)
    state = cast(PEState, env.state)

    # Check state is as expected
    assert np.allclose(state.evader_state[:3], evader_start_coord + (0,))
    assert np.allclose(state.pursuer_state[:3], pursuer_start_coord + (0,))
    assert np.allclose(state.evader_start_coord, evader_start_coord)
    assert np.allclose(pursuer_start_coord, pursuer_start_coord)
    assert np.allclose(state.evader_goal_coord, goal_coord)
    assert state.min_goal_dist == size - 1, state.min_goal_dist

    # agents start facing right (evader is face goal)
    # pursuer does nothing, evader moves towards goal
    for _ in range(size - 2):
        _, rewards, _, _, _, _ = env.step(
            {"0": np.array([0.0, 1.0]), "1": np.array([0.0, 0.0])}
        )
        assert rewards["0"] == model.R_PROGRESS

    _, rewards, _, _, all_done, _ = env.step(
        {"0": np.array([0.0, 1.0]), "1": np.array([0.0, 0.0])}
    )
    assert all_done
    assert rewards["0"] == model.R_EVASION + model.R_PROGRESS


def test_shortest_path_not_double_reward():
    """Check agent doesn't receive shortest path rewards multiple times."""
    size, max_obs_dist, n_sensors = 8, 4.0, 16

    evader_start_coord = (0.5, 0.5)
    goal_coord = (size - 0.5, 0.5)
    pursuer_start_coord = (size - 0.5, size - 0.5)

    # Empty world with evader in top left and pursuer in bottom right
    # And goal top right corner
    pe_world = PEWorld(
        size,
        blocked_coords=set(),
        goal_coords_map={evader_start_coord: [goal_coord]},
        evader_start_coords=[evader_start_coord],
        pursuer_start_coords=[pursuer_start_coord],
    )

    env = posggym.make(
        "PursuitEvasionContinuous-v0",
        world=pe_world,
        max_obs_distance=max_obs_dist,
        fov=1.57,
        n_sensors=n_sensors,
        normalize_reward=False,  # don't normalize reward for testing
        use_progress_reward=True,
        render_mode="human",
    )
    env.reset(seed=26)

    model = cast(PursuitEvasionContinuousModel, env.model)
    state = cast(PEState, env.state)

    # Check state is as expected
    assert np.allclose(state.evader_state[:3], evader_start_coord + (0,))
    assert np.allclose(state.pursuer_state[:3], pursuer_start_coord + (0,))
    assert np.allclose(state.evader_start_coord, evader_start_coord)
    assert np.allclose(pursuer_start_coord, pursuer_start_coord)
    assert np.allclose(state.evader_goal_coord, goal_coord)
    assert state.min_goal_dist == size - 1, state.min_goal_dist

    # agents start facing right (evader is face goal)
    # pursuer does nothing, evader moves towards goal
    for _ in range(4):
        # evader moves towards goal for 4 steps
        _, rewards, _, _, _, _ = env.step(
            {"0": np.array([0.0, 1.0]), "1": np.array([0.0, 0.0])}
        )
        assert rewards["0"] == model.R_PROGRESS

    while True:
        # evader turns around away from goal
        dyaw = min(model.dyaw_limit, abs(state.evader_state[2] - np.pi))
        _, rewards, _, _, _, _ = env.step(
            {"0": np.array([dyaw, 0.0]), "1": np.array([0.0, 0.0])}
        )
        assert rewards["0"] == 0.0
        state = cast(PEState, env.state)
        if np.isclose(state.evader_state[2], np.pi, rtol=0.0, atol=0.01).all():
            break

    for _ in range(2):
        # evader moves away from goal for 2 steps
        _, rewards, _, _, _, _ = env.step(
            {"0": np.array([0.0, 1.0]), "1": np.array([0.0, 0.0])}
        )
        assert rewards["0"] == 0.0

    while True:
        # evader turns around towards goal
        dyaw = min(model.dyaw_limit, abs(state.evader_state[2]))
        _, rewards, _, _, _, _ = env.step(
            {"0": np.array([dyaw, 0.0]), "1": np.array([0.0, 0.0])}
        )
        assert rewards["0"] == 0.0
        state = cast(PEState, env.state)
        if np.isclose(state.evader_state[2], 0.0, rtol=0.0, atol=0.01).all():
            break

    for _ in range(2):
        # evader moves towards goal for 2 steps
        # but should not receive shortest path reward
        _, rewards, _, _, _, _ = env.step(
            {"0": np.array([0.0, 1.0]), "1": np.array([0.0, 0.0])}
        )
        assert rewards["0"] == 0.0

    for _ in range(2):
        # evader moves towards goal for another 2 steps
        # should receive shortest path reward
        _, rewards, _, _, _, _ = env.step(
            {"0": np.array([0.0, 1.0]), "1": np.array([0.0, 0.0])}
        )
        assert rewards["0"] == model.R_PROGRESS

    # evader reaches goal
    _, rewards, _, _, all_done, _ = env.step(
        {"0": np.array([0.0, 1.0]), "1": np.array([0.0, 0.0])}
    )
    assert all_done
    assert rewards["0"] == model.R_EVASION + model.R_PROGRESS


if __name__ == "__main__":
    test_obs()
    test_shortest_path()
    test_shortest_path_not_double_reward()
