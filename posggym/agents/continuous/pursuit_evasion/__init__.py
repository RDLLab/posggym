"""Policies for the PursuitEvasionContinuous-v0 environment."""
import os.path as osp

from posggym.agents.torch_policy import PPOPolicy
from posggym.config import AGENT_MODEL_DIR
from posggym.agents.utils import processors


ENV_ID = "PursuitEvasionContinuous-v0"
agent_model_dir = osp.join(AGENT_MODEL_DIR, "continuous", "pursuit")
policy_specs = {}

for policy_file_name in [
    "sp_seed0_i0.pkl",
    "sp_seed0_i1.pkl",
]:
    spec = PPOPolicy.get_spec_from_path(
        env_id=ENV_ID,
        env_args={
            "world": "8x8",
            "fov": 1.57,
            "max_obs_distance": 12.0,
            "n_sensors": 16,
        },
        policy_file_path=osp.join(
            agent_model_dir, "pursuit_8x8", policy_file_name
        ),
        version=0,
        valid_agent_ids=None,
        # policy is deterministic given random seed
        nondeterministic=False,
        # actions sampled, rather always taking most probable action
        deterministic=False,
        obs_processor_cls=processors.RescaleProcessor,
        obs_processor_config={"min_val": -1, "max_val": 1},
        action_processor_cls=processors.RescaleProcessor,
        action_processor_config={"min_val": -1, "max_val": 1},
    )
    policy_specs[spec.id] = spec


for policy_file_name in [
    "sp_seed0_i0.pkl",
    "sp_seed0_i1.pkl",
]:
    spec = PPOPolicy.get_spec_from_path(
        env_id=ENV_ID,
        env_args={
            "world": "16x16",
            "fov": 1.57,
            "max_obs_distance": 12.0,
            "n_sensors": 16,
        },
        policy_file_path=osp.join(
            agent_model_dir, "pursuit_16x16", policy_file_name
        ),
        version=0,
        valid_agent_ids=None,
        # policy is deterministic given random seed
        nondeterministic=False,
        # actions sampled, rather always taking most probable action
        deterministic=False,
        obs_processor_cls=processors.RescaleProcessor,
        obs_processor_config={"min_val": -1, "max_val": 1},
        action_processor_cls=processors.RescaleProcessor,
        action_processor_config={"min_val": -1, "max_val": 1},
    )
    policy_specs[spec.id] = spec



for policy_file_name in [
    "sp_seed0_i0.pkl",
    "sp_seed0_i1.pkl",
]:
    spec = PPOPolicy.get_spec_from_path(
        env_id=ENV_ID,
        env_args={
            "world": "32x32",
            "fov": 1.57,
            "max_obs_distance": 12.0,
            "n_sensors": 16,
        },
        policy_file_path=osp.join(
            agent_model_dir, "pursuit_32x32", policy_file_name
        ),
        version=0,
        valid_agent_ids=None,
        # policy is deterministic given random seed
        nondeterministic=False,
        # actions sampled, rather always taking most probable action
        deterministic=False,
        obs_processor_cls=processors.RescaleProcessor,
        obs_processor_config={"min_val": -1, "max_val": 1},
        action_processor_cls=processors.RescaleProcessor,
        action_processor_config={"min_val": -1, "max_val": 1},
    )
    policy_specs[spec.id] = spec
