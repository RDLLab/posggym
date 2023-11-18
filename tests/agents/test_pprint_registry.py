"""Tests that `posggym_agents.pprint_registry` works as expected.

Ref:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/test_pprint_registry.py
"""
import posggym.agents as pga

# To ignore the trailing whitespaces, will need flake to ignore this file.
# flake8: noqa


def test_pprint_custom_registry():
    """Testing a registry different from default."""
    pp_env_args_id = "grid=10x10-num_predators=2-num_prey=3-cooperative=True"
    a = {
        "Random-v0": pga.registry["Random-v0"],
        "LevelBasedForaging-v3/H1-v0": pga.registry["LevelBasedForaging-v3/H1-v0"],
        f"PredatorPrey-v0/{pp_env_args_id}/RL1-v0": pga.registry[
            f"PredatorPrey-v0/{pp_env_args_id}/RL1-v0"
        ],
    }
    out = pga.pprint_registry(a, disable_print=True)

    correct_out = f"""===== Generic =====
Random-v0


===== LevelBasedForaging-v3 =====
H1-v0


===== PredatorPrey-v0 =====
----- PredatorPrey-v0/{pp_env_args_id} -----
RL1-v0


"""
    assert out == correct_out
