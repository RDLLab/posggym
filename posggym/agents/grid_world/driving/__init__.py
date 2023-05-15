"""Policies for the Driving-v0 environment."""
import os.path as osp

from posggym.agents.torch_policy import PPOPolicy
from posggym.config import AGENT_MODEL_DIR


ENV_ID = "Driving-v0"
agent_model_dir = osp.join(AGENT_MODEL_DIR, "grid_world", "driving")
policy_specs = {}

# 7x7RoundAbout-n2-v0
for policy_file_name in [
    "klr_k0_seed0.pkl",
    "klr_k0_seed1.pkl",
    "klr_k0_seed2.pkl",
    "klr_k0_seed3.pkl",
    "klr_k0_seed4.pkl",
    "klr_k1_seed0.pkl",
    "klr_k1_seed1.pkl",
    "klr_k1_seed2.pkl",
    "klr_k1_seed3.pkl",
    "klr_k1_seed4.pkl",
    "klr_k2_seed0.pkl",
    "klr_k2_seed1.pkl",
    "klr_k2_seed2.pkl",
    "klr_k2_seed3.pkl",
    "klr_k2_seed4.pkl",
    "klr_k3_seed0.pkl",
    "klr_k3_seed1.pkl",
    "klr_k3_seed2.pkl",
    "klr_k3_seed3.pkl",
    "klr_k3_seed4.pkl",
    "klr_k4_seed0.pkl",
    "klr_k4_seed1.pkl",
    "klr_k4_seed2.pkl",
    "klr_k4_seed3.pkl",
    "klr_k4_seed4.pkl",
    "klrbr_k4_seed0.pkl",
    "klrbr_k4_seed1.pkl",
    "klrbr_k4_seed2.pkl",
    "klrbr_k4_seed3.pkl",
    "klrbr_k4_seed4.pkl",
]:
    spec = PPOPolicy.get_spec_from_path(
        env_id=ENV_ID,
        env_args={
            "grid": "7x7RoundAbout",
            "num_agents": 2,
            "obs_dim": (3, 1, 1),
            "obstacle_collisions": False,
        },
        policy_file_path=osp.join(
            agent_model_dir, "driving_7x7roundabout_n2_v0", policy_file_name
        ),
        version=0,
        valid_agent_ids=None,
        # policy is deterministic given random seed
        nondeterministic=False,
        # actions sampled, rather always taking most probable action
        deterministic=False,
    )
    policy_specs[spec.id] = spec


# 14x14RoundAbout-n2-v0
for policy_file_name in [
    "klr_k0_seed0.pkl",
    "klr_k0_seed1.pkl",
    "klr_k0_seed2.pkl",
    "klr_k0_seed3.pkl",
    "klr_k0_seed4.pkl",
    "klr_k1_seed0.pkl",
    "klr_k1_seed1.pkl",
    "klr_k1_seed2.pkl",
    "klr_k1_seed3.pkl",
    "klr_k1_seed4.pkl",
    "klr_k2_seed0.pkl",
    "klr_k2_seed1.pkl",
    "klr_k2_seed2.pkl",
    "klr_k2_seed3.pkl",
    "klr_k2_seed4.pkl",
    "klr_k3_seed0.pkl",
    "klr_k3_seed1.pkl",
    "klr_k3_seed2.pkl",
    "klr_k3_seed3.pkl",
    "klr_k3_seed4.pkl",
    "klr_k4_seed0.pkl",
    "klr_k4_seed1.pkl",
    "klr_k4_seed2.pkl",
    "klr_k4_seed3.pkl",
    "klr_k4_seed4.pkl",
    "klrbr_k4_seed0.pkl",
    "klrbr_k4_seed1.pkl",
    "klrbr_k4_seed2.pkl",
    "klrbr_k4_seed3.pkl",
    "klrbr_k4_seed4.pkl",
]:
    spec = PPOPolicy.get_spec_from_path(
        env_id=ENV_ID,
        env_args={
            "grid": "14x14RoundAbout",
            "num_agents": 2,
            "obs_dim": (3, 1, 1),
            "obstacle_collisions": False,
        },
        policy_file_path=osp.join(
            agent_model_dir, "driving_14x14roundabout_n2_v0", policy_file_name
        ),
        version=0,
        valid_agent_ids=None,
        # policy is deterministic given random seed
        nondeterministic=False,
        # actions sampled, rather always taking most probable action
        deterministic=False,
    )
    policy_specs[spec.id] = spec
