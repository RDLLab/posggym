"""Policies for the PredatorPrey-v0 environment."""
import os.path as osp

from posggym.config import AGENT_MODEL_DIR
from posggym.agents.rllib_policy import load_rllib_policy_specs_from_files


ENV_ID = "PredatorPrey-v0"
agent_model_dir = osp.join(AGENT_MODEL_DIR, "grid_world", "predator_prey")
policy_specs = {}

# PredatorPrey-10x10-P2-p3-s2-coop-v0
policy_specs.update(
    load_rllib_policy_specs_from_files(
        env_id=ENV_ID,
        env_args={
            "grid": "10x10",
            "num_predators": 2,
            "num_prey": 3,
            "cooperative": True,
            "prey_strength": 2,
            "obs_dim": 2,
        },
        policy_file_dir_path=osp.join(
            agent_model_dir, "predatorprey_10x10_P2_p3_s2_coop_v0"
        ),
        policy_file_names=[
            "sp_seed0.pkl",
            "sp_seed1.pkl",
            "sp_seed2.pkl",
            "sp_seed3.pkl",
            "sp_seed4.pkl",
        ],
        version=0,
        valid_agent_ids=None,
        nondeterministic=True,
    )
)

# PredatorPrey-10x10-P3-p3-s2-coop-v0
policy_specs.update(
    load_rllib_policy_specs_from_files(
        env_id=ENV_ID,
        env_args={
            "grid": "10x10",
            "num_predators": 3,
            "num_prey": 3,
            "cooperative": True,
            "prey_strength": 2,
            "obs_dim": 2,
        },
        policy_file_dir_path=osp.join(
            agent_model_dir, "predatorprey_10x10_P3_p3_s2_coop_v0"
        ),
        policy_file_names=[
            "sp_seed0.pkl",
            "sp_seed1.pkl",
            "sp_seed2.pkl",
        ],
        version=0,
        valid_agent_ids=None,
        nondeterministic=True,
    )
)


# PredatorPrey-10x10-P4-p3-s3-coop-v0
policy_specs.update(
    load_rllib_policy_specs_from_files(
        env_id=ENV_ID,
        env_args={
            "grid": "10x10",
            "num_predators": 4,
            "num_prey": 3,
            "cooperative": True,
            "prey_strength": 3,
            "obs_dim": 2,
        },
        policy_file_dir_path=osp.join(
            agent_model_dir, "predatorprey_10x10_P4_p3_s3_coop_v0"
        ),
        policy_file_names=[
            "sp_seed0.pkl",
            "sp_seed1.pkl",
            "sp_seed2.pkl",
            "sp_seed3.pkl",
            "sp_seed4.pkl",
        ],
        version=0,
        valid_agent_ids=None,
        nondeterministic=True,
    )
)
