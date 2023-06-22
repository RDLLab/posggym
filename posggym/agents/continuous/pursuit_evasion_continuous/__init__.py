"""Policies for the PursuitEvasionContinuous-v0 environment."""
import math
import os.path as osp

from posggym.agents.continuous.pursuit_evasion_continuous import shortest_path
from posggym.agents.registration import PolicySpec
from posggym.agents.torch_policy import PPOPolicy
from posggym.agents.utils import processors
from posggym.config import AGENT_MODEL_DIR

ENV_ID = "PursuitEvasionContinuous-v0"
agent_model_dir = osp.join(AGENT_MODEL_DIR, "continuous", "pursuit_evasion_continuous")
policy_specs = {}

_sp_spec = PolicySpec(
    policy_name="shortest_path",
    entry_point=shortest_path.PECShortestPathPolicy,
    version=0,
    env_id=ENV_ID,
    env_args=None,
    valid_agent_ids=None,
    nondeterministic=False,
)
policy_specs[_sp_spec.id] = _sp_spec

# 8x8 self-play trained policies with default env args
for policy_file_name in [
    "sp_seed0_i0.pkl",
    "sp_seed0_i1.pkl",
    "sp_seed1_i0.pkl",
    "sp_seed1_i1.pkl",
    "sp_seed2_i0.pkl",
    "sp_seed2_i1.pkl",
    "sp_seed3_i0.pkl",
    "sp_seed3_i1.pkl",
    "sp_seed4_i0.pkl",
    "sp_seed4_i1.pkl",
]:
    spec = PPOPolicy.get_spec_from_path(
        policy_file_path=osp.join(
            agent_model_dir,
            "world=8x8",
            policy_file_name,
        ),
        env_id=ENV_ID,
        env_args={
            "world": "8x8",
            "fov": math.pi / 3,
            "max_obs_distance": 8.0 / 3,
            "n_sensors": 16,
        },
        env_args_id="world=8x8",
        version=0,
        valid_agent_ids=["0" if "_i0" in policy_file_name else "1"],
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

# 16x16 self-play trained policies with default env args
for policy_file_name in [
    "sp_seed0_i0.pkl",
    "sp_seed0_i1.pkl",
    "sp_seed1_i0.pkl",
    "sp_seed1_i1.pkl",
    "sp_seed2_i0.pkl",
    "sp_seed2_i1.pkl",
    "sp_seed3_i0.pkl",
    "sp_seed3_i1.pkl",
    "sp_seed4_i0.pkl",
    "sp_seed4_i1.pkl",
]:
    spec = PPOPolicy.get_spec_from_path(
        policy_file_path=osp.join(
            agent_model_dir,
            "world=16x16",
            policy_file_name,
        ),
        env_id=ENV_ID,
        env_args={
            "world": "16x16",
            "fov": math.pi / 3,
            "max_obs_distance": 16.0 / 3,
            "n_sensors": 16,
        },
        env_args_id="world=16x16",
        version=0,
        valid_agent_ids=["0" if "_i0" in policy_file_name else "1"],
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
