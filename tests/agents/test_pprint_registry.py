"""Tests that `posggym_agents.pprint_registry` works as expected.

Ref:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/test_pprint_registry.py
"""
import posggym.agents as pga

# To ignore the trailing whitespaces, will need flake to ignore this file.
# flake8: noqa


def test_pprint_custom_registry():
    """Testing a registry different from default."""
    driving_env_and_args_id = (
        "Driving-v0/"
        "grid=14x14RoundAbout-num_agents=2-obs_dim=(3,1,1)-obstacle_collisions=False"
    )
    a = {
        "Random-v0": pga.registry["Random-v0"],
        "LevelBasedForaging-v2/Heuristic1-v0": pga.registry[
            "LevelBasedForaging-v2/Heuristic1-v0"
        ],
        f"{driving_env_and_args_id}/klr_k0_seed0-v0": pga.registry[
            f"{driving_env_and_args_id}/klr_k0_seed0-v0"
        ],
    }
    out = pga.pprint_registry(a, disable_print=True)

    correct_out = """===== Generic =====
Random-v0


===== LevelBasedForaging-v2 =====
Heuristic1-v0


===== Driving-v0 =====
----- Driving-v0/grid=14x14RoundAbout-num_agents=2-obs_dim=(3,1,1)-obstacle_collisions=False -----
klr_k0_seed0-v0


"""
    assert out == correct_out
