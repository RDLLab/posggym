"""Policies for the DrivingContinuous-v0 environment."""
import os.path as osp

from posggym.agents.torch_policy import PPOPolicy
from posggym.config import AGENT_MODEL_DIR
from posggym.agents.utils import processors


ENV_ID = "DrivingContinuous-v0"
agent_model_dir = osp.join(AGENT_MODEL_DIR, "continuous", "driving")
policy_specs = {}

# 14x14RoundAbout-n2-v0
for policy_file_name in [
    "sp_seed2.pkl",
    "sp_seed3.pkl",
    "sp_seed4.pkl",
    "sp_seed5.pkl",
    "sp_seedNone.pkl",
]:
    spec = PPOPolicy.get_spec_from_path(
        env_id=ENV_ID,
        env_args={
            "world": "14x14RoundAbout",
            "num_agents": 2,
            "obs_dist": 5.0,
            "n_sensors": 16,
        },
        policy_file_path=osp.join(
            agent_model_dir, "driving_14x14roundabout", policy_file_name
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


# 7x7RoundAbout-n2-v0
for policy_file_name in [
    "sp_seed0.pkl",
    "sp_seed1.pkl",
    "sp_seed2.pkl",
    "sp_seed3.pkl",
    "sp_seed4.pkl",
]:
    spec = PPOPolicy.get_spec_from_path(
        env_id=ENV_ID,
        env_args={
            "world": "7x7RoundAbout",
            "num_agents": 2,
            "obs_dist": 5.0,
            "n_sensors": 16,
        },
        policy_file_path=osp.join(
            agent_model_dir, "driving_7x7roundabout", policy_file_name
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


# 6x6Intersection-n2-v0
for policy_file_name in [
    "sp_seed0.pkl",
    "sp_seed1.pkl",
    "sp_seed2.pkl",
    "sp_seed3.pkl",
    "sp_seed4.pkl",
]:
    spec = PPOPolicy.get_spec_from_path(
        env_id=ENV_ID,
        env_args={
            "world": "6x6Intersection",
            "num_agents": 2,
            "obs_dist": 5.0,
            "n_sensors": 16,
        },
        policy_file_path=osp.join(
            agent_model_dir, "driving_6x6intersection", policy_file_name
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
