"""Specific tests for the DroneTeamCapture-v0 environment."""

import pytest

import posggym


@pytest.mark.parametrize("num_pursuers", [2, 3, 4, 8])
def test_init_steps(num_pursuers: int):
    """Check observations are as expected after reset."""
    env = posggym.make(
        "DroneTeamCapture-v0",
        num_agents=num_pursuers,
        n_communicating_pursuers=None,
        arena_radius=430,
        observation_limit=None,
        velocity_control=False,
        capture_radius=30,
        use_q_reward=False,
        max_episode_steps=2,
    )
    env.reset(seed=35)

    for t in range(100):
        a = {i: env.action_spaces[i].sample() for i in env.agents}
        obs, _, _, _, all_done, _ = env.step(a)

        for i, o_i in obs.items():
            assert env.observation_spaces[i].contains(
                o_i
            ), f"Agent {i} observation {o_i} is not in its observation space."

        if all_done:
            env.reset()

    env.close()


if __name__ == "__main__":
    test_init_steps(3)
