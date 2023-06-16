"""Policies for the PredatorPrey-v0 environment."""
import os.path as osp

from posggym.agents.torch_policy import PPOPolicy
from posggym.agents.utils import processors
from posggym.config import AGENT_MODEL_DIR

ENV_ID = "PredatorPrey-v0"
agent_model_dir = osp.join(AGENT_MODEL_DIR, "grid_world", "predator_prey")
policy_specs = {}

# PredatorPrey-10x10-P2-p3-s2-coop-v0
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
            "grid": "10x10",
            "num_predators": 2,
            "num_prey": 3,
            "cooperative": True,
            "prey_strength": 2,
            "obs_dim": 2,
        },
        policy_file_path=osp.join(
            agent_model_dir,
            "grid=10x10-num_predators=2-num_prey=3-cooperative=True-prey_strength=2-obs_dim=2",
            policy_file_name,
        ),
        version=0,
        valid_agent_ids=None,
        # policy is deterministic given random seed
        nondeterministic=False,
        # actions sampled, rather always taking most probable action
        deterministic=False,
        obs_processor_cls=processors.FlattenProcessor,
        obs_processor_config=None,
        action_processor_cls=None,
        action_processor_config=None,
    )
    policy_specs[spec.id] = spec


# PredatorPrey-10x10-P3-p3-s2-coop-v0
for policy_file_name in [
    "sp_seed0.pkl",
    "sp_seed1.pkl",
    "sp_seed2.pkl",
]:
    spec = PPOPolicy.get_spec_from_path(
        env_id=ENV_ID,
        env_args={
            "grid": "10x10",
            "num_predators": 3,
            "num_prey": 3,
            "cooperative": True,
            "prey_strength": 2,
            "obs_dim": 2,
        },
        policy_file_path=osp.join(
            agent_model_dir,
            "grid=10x10-num_predators=3-num_prey=3-cooperative=True-prey_strength=2-obs_dim=2",
            policy_file_name,
        ),
        version=0,
        valid_agent_ids=None,
        # policy is deterministic given random seed
        nondeterministic=False,
        # actions sampled, rather always taking most probable action
        deterministic=False,
        obs_processor_cls=processors.FlattenProcessor,
        obs_processor_config=None,
        action_processor_cls=None,
        action_processor_config=None,
    )
    policy_specs[spec.id] = spec


# PredatorPrey-10x10-P4-p3-s3-coop-v0
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
            "grid": "10x10",
            "num_predators": 4,
            "num_prey": 3,
            "cooperative": True,
            "prey_strength": 3,
            "obs_dim": 2,
        },
        policy_file_path=osp.join(
            agent_model_dir,
            "grid=10x10-num_predators=4-num_prey=3-cooperative=True-prey_strength=3-obs_dim=2",
            policy_file_name,
        ),
        version=0,
        valid_agent_ids=None,
        # policy is deterministic given random seed
        nondeterministic=False,
        # actions sampled, rather always taking most probable action
        deterministic=False,
        obs_processor_cls=processors.FlattenProcessor,
        obs_processor_config=None,
        action_processor_cls=None,
        action_processor_config=None,
    )
    policy_specs[spec.id] = spec
