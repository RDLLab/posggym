"""Tests that `posggym.Env.render` function works as expected.

Ref:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/test_rendering.py
"""
import numpy as np
import pytest
from posggym.envs.registration import EnvSpec
from posggym.logger import warn

from tests.envs.utils import all_testing_env_specs


def check_rendered(rendered_frame, mode: str):
    """Check that the rendered frame is as expected."""
    if mode == "rgb_array_dict":
        assert isinstance(rendered_frame, dict)
        for i, agent_frame in rendered_frame.items():
            assert isinstance(i, str)
            check_rendered(agent_frame, "rgb_array")
    elif mode == "rgb_array":
        assert isinstance(rendered_frame, np.ndarray)
        assert len(rendered_frame.shape) == 3
        assert rendered_frame.shape[2] == 3
        assert np.all(rendered_frame >= 0) and np.all(rendered_frame <= 255)
    elif mode == "ansi_dict":
        assert isinstance(rendered_frame, dict)
        for i, agent_frame in rendered_frame.items():
            assert isinstance(i, str)
            check_rendered(agent_frame, "ansi")
    elif mode == "ansi":
        assert isinstance(rendered_frame, str)
        assert len(rendered_frame) > 0
    else:
        warn(
            f"Unknown render mode: {mode}, cannot check that the rendered data is "
            "correct. Add case to `check_rendered`"
        )


@pytest.mark.parametrize(
    "spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_render_modes(spec: EnvSpec):
    env = spec.make(disable_env_checker=True)

    # assert "rgb_array" in env.metadata["render_modes"]

    for mode in env.metadata["render_modes"]:
        if mode != "human":
            new_env = spec.make(render_mode=mode, disable_env_checker=True)

            new_env.reset()
            rendered = new_env.render()  # type: ignore
            check_rendered(rendered, mode)

            new_env.step(
                {
                    i: act_space.sample()
                    for i, act_space in new_env.action_spaces.items()
                }
            )
            rendered = new_env.render()
            check_rendered(rendered, mode)

            new_env.close()
    env.close()
