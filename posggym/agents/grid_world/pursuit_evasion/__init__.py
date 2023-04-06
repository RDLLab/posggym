"""Policies for the PursuitEvasion-v0 environment."""
import os.path as osp

from posggym.agents.grid_world.pursuit_evasion.shortest_path import PEShortestPathPolicy
from posggym.agents.registration import PolicySpec
from posggym.agents.rllib import load_rllib_policy_specs_from_files
from posggym.config import AGENT_MODEL_DIR


ENV_ID = "PursuitEvasion-v0"
agent_model_dir = osp.join(AGENT_MODEL_DIR, "grid_world", "pursuit_evasion")
policy_specs = {}

_shortest_path_spec = PolicySpec(
    policy_name="shortestpath",
    entry_point=PEShortestPathPolicy,
    version=0,
    env_id="PursuitEvasion-v0",
    env_args=None,
    valid_agent_ids=None,
    nondeterministic=False,
)
policy_specs[_shortest_path_spec.id] = _shortest_path_spec

# PursuitEvasion 16x16
# Evader (agent=0)
policy_specs.update(
    load_rllib_policy_specs_from_files(
        env_id=ENV_ID,
        env_args={"grid": "16x16"},
        policy_file_dir_path=osp.join(agent_model_dir, "pursuitevasion_16x16_v0"),
        policy_file_names=[
            "klr_k0_seed0_i0.pkl",
            "klr_k0_seed1_i0.pkl",
            "klr_k0_seed2_i0.pkl",
            "klr_k0_seed3_i0.pkl",
            "klr_k0_seed4_i0.pkl",
            "klr_k1_seed0_i0.pkl",
            "klr_k1_seed1_i0.pkl",
            "klr_k1_seed2_i0.pkl",
            "klr_k1_seed3_i0.pkl",
            "klr_k1_seed4_i0.pkl",
            "klr_k2_seed0_i0.pkl",
            "klr_k2_seed1_i0.pkl",
            "klr_k2_seed2_i0.pkl",
            "klr_k2_seed3_i0.pkl",
            "klr_k2_seed4_i0.pkl",
            "klr_k3_seed0_i0.pkl",
            "klr_k3_seed1_i0.pkl",
            "klr_k3_seed2_i0.pkl",
            "klr_k3_seed3_i0.pkl",
            "klr_k3_seed4_i0.pkl",
            "klr_k4_seed0_i0.pkl",
            "klr_k4_seed1_i0.pkl",
            "klr_k4_seed2_i0.pkl",
            "klr_k4_seed3_i0.pkl",
            "klr_k4_seed4_i0.pkl",
            "klrbr_k4_seed0_i0.pkl",
            "klrbr_k4_seed1_i0.pkl",
            "klrbr_k4_seed2_i0.pkl",
            "klrbr_k4_seed3_i0.pkl",
            "klrbr_k4_seed4_i0.pkl",
            "sp_seed0_i0.pkl",
            "sp_seed1_i0.pkl",
            "sp_seed2_i0.pkl",
            "sp_seed3_i0.pkl",
            "sp_seed4_i0.pkl",
        ],
        version=0,
        valid_agent_ids=["0"],
        nondeterministic=True,
    )
)
# Pursuer (agent=1)
policy_specs.update(
    load_rllib_policy_specs_from_files(
        env_id=ENV_ID,
        env_args={"grid": "16x16"},
        policy_file_dir_path=osp.join(agent_model_dir, "pursuitevasion_16x16_v0"),
        policy_file_names=[
            "klr_k0_seed0_i1.pkl",
            "klr_k0_seed1_i1.pkl",
            "klr_k0_seed2_i1.pkl",
            "klr_k0_seed3_i1.pkl",
            "klr_k0_seed4_i1.pkl",
            "klr_k1_seed0_i1.pkl",
            "klr_k1_seed1_i1.pkl",
            "klr_k1_seed2_i1.pkl",
            "klr_k1_seed3_i1.pkl",
            "klr_k1_seed4_i1.pkl",
            "klr_k2_seed0_i1.pkl",
            "klr_k2_seed1_i1.pkl",
            "klr_k2_seed2_i1.pkl",
            "klr_k2_seed3_i1.pkl",
            "klr_k2_seed4_i1.pkl",
            "klr_k3_seed0_i1.pkl",
            "klr_k3_seed1_i1.pkl",
            "klr_k3_seed2_i1.pkl",
            "klr_k3_seed3_i1.pkl",
            "klr_k3_seed4_i1.pkl",
            "klr_k4_seed0_i1.pkl",
            "klr_k4_seed1_i1.pkl",
            "klr_k4_seed2_i1.pkl",
            "klr_k4_seed3_i1.pkl",
            "klr_k4_seed4_i1.pkl",
            "klrbr_k4_seed0_i1.pkl",
            "klrbr_k4_seed1_i1.pkl",
            "klrbr_k4_seed2_i1.pkl",
            "klrbr_k4_seed3_i1.pkl",
            "klrbr_k4_seed4_i1.pkl",
            "sp_seed0_i1.pkl",
            "sp_seed1_i1.pkl",
            "sp_seed2_i1.pkl",
            "sp_seed3_i1.pkl",
            "sp_seed4_i1.pkl",
        ],
        version=0,
        valid_agent_ids=["1"],
        nondeterministic=True,
    )
)

# PursuitEvasion 8x8
# Evader (agent=0)
policy_specs.update(
    load_rllib_policy_specs_from_files(
        env_id=ENV_ID,
        env_args={"grid": "8x8"},
        policy_file_dir_path=osp.join(agent_model_dir, "pursuitevasion_8x8_v0"),
        policy_file_names=[
            "klr_k0_seed0_i0.pkl",
            "klr_k0_seed1_i0.pkl",
            "klr_k0_seed2_i0.pkl",
            "klr_k0_seed3_i0.pkl",
            "klr_k0_seed4_i0.pkl",
            "klr_k1_seed0_i0.pkl",
            "klr_k1_seed1_i0.pkl",
            "klr_k1_seed2_i0.pkl",
            "klr_k1_seed3_i0.pkl",
            "klr_k1_seed4_i0.pkl",
            "klr_k2_seed0_i0.pkl",
            "klr_k2_seed1_i0.pkl",
            "klr_k2_seed2_i0.pkl",
            "klr_k2_seed3_i0.pkl",
            "klr_k2_seed4_i0.pkl",
            "klr_k3_seed0_i0.pkl",
            "klr_k3_seed1_i0.pkl",
            "klr_k3_seed2_i0.pkl",
            "klr_k3_seed3_i0.pkl",
            "klr_k3_seed4_i0.pkl",
            "klr_k4_seed0_i0.pkl",
            "klr_k4_seed1_i0.pkl",
            "klr_k4_seed2_i0.pkl",
            "klr_k4_seed3_i0.pkl",
            "klr_k4_seed4_i0.pkl",
            "klrbr_k4_seed0_i0.pkl",
            "klrbr_k4_seed1_i0.pkl",
            "klrbr_k4_seed2_i0.pkl",
            "klrbr_k4_seed3_i0.pkl",
            "klrbr_k4_seed4_i0.pkl",
            "sp_seed0_i0.pkl",
            "sp_seed1_i0.pkl",
            "sp_seed2_i0.pkl",
            "sp_seed3_i0.pkl",
            "sp_seed4_i0.pkl",
        ],
        version=0,
        valid_agent_ids=["0"],
        nondeterministic=True,
    )
)
# Pursuer (agent=1)
policy_specs.update(
    load_rllib_policy_specs_from_files(
        env_id=ENV_ID,
        env_args={"grid": "8x8"},
        policy_file_dir_path=osp.join(agent_model_dir, "pursuitevasion_8x8_v0"),
        policy_file_names=[
            "klr_k0_seed0_i1.pkl",
            "klr_k0_seed1_i1.pkl",
            "klr_k0_seed2_i1.pkl",
            "klr_k0_seed3_i1.pkl",
            "klr_k0_seed4_i1.pkl",
            "klr_k1_seed0_i1.pkl",
            "klr_k1_seed1_i1.pkl",
            "klr_k1_seed2_i1.pkl",
            "klr_k1_seed3_i1.pkl",
            "klr_k1_seed4_i1.pkl",
            "klr_k2_seed0_i1.pkl",
            "klr_k2_seed1_i1.pkl",
            "klr_k2_seed2_i1.pkl",
            "klr_k2_seed3_i1.pkl",
            "klr_k2_seed4_i1.pkl",
            "klr_k3_seed0_i1.pkl",
            "klr_k3_seed1_i1.pkl",
            "klr_k3_seed2_i1.pkl",
            "klr_k3_seed3_i1.pkl",
            "klr_k3_seed4_i1.pkl",
            "klr_k4_seed0_i1.pkl",
            "klr_k4_seed1_i1.pkl",
            "klr_k4_seed2_i1.pkl",
            "klr_k4_seed3_i1.pkl",
            "klr_k4_seed4_i1.pkl",
            "klrbr_k4_seed0_i1.pkl",
            "klrbr_k4_seed1_i1.pkl",
            "klrbr_k4_seed2_i1.pkl",
            "klrbr_k4_seed3_i1.pkl",
            "klrbr_k4_seed4_i1.pkl",
            "sp_seed0_i1.pkl",
            "sp_seed1_i1.pkl",
            "sp_seed2_i1.pkl",
            "sp_seed3_i1.pkl",
            "sp_seed4_i1.pkl",
        ],
        version=0,
        valid_agent_ids=["1"],
        nondeterministic=True,
    )
)