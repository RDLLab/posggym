"""Policies for the PredatorPreyContinuous-v0 environment."""
import os.path as osp

from posggym.agents.registration import PolicySpec
from posggym.agents.continuous.predator_prey_continuous import heuristic
from posggym.agents.torch_policy import PPOPolicy
from posggym.config import AGENT_MODEL_DIR
from posggym.agents.utils import processors


ENV_ID = "PredatorPreyContinuous-v0"
agent_model_dir = osp.join(AGENT_MODEL_DIR, "continuous", "predator_prey_continuous")
policy_specs = {}


for n, policy_class in enumerate(
    [
        heuristic.PPCHeuristic0Policy,
        heuristic.PPCHeuristic1Policy,
        heuristic.PPCHeuristic2Policy,
        heuristic.PPCHeuristic3Policy,
    ]
):
    _heuristic_spec = PolicySpec(
        policy_name=f"heuristic{n}",
        entry_point=policy_class,
        version=0,
        env_id=ENV_ID,
        env_args=None,
        valid_agent_ids=None,
        nondeterministic=False,
    )
    policy_specs[_heuristic_spec.id] = _heuristic_spec

# 10x10, 2 predators, 3 prey, prey_strength=2, cooperative=True
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
            "world": "10x10",
            "num_predators": 2,
            "num_prey": 3,
            "prey_strength": 2,
            "cooperative": True,
            "obs_dist": 4.0,
            "n_sensors": 16,
        },
        policy_file_path=osp.join(
            agent_model_dir,
            "world=10x10-num_predators=2-num_prey=3-prey_strength=2_cooperative=True-obs_dist=4-n_sensors=16",
            policy_file_name,
        ),
        version=0,
        valid_agent_ids=None,
        # policy is deterministic given random seed
        nondeterministic=False,
        # actions sampled, rather than always taking most probable action
        deterministic=False,
        obs_processor_cls=processors.RescaleProcessor,
        obs_processor_config={"min_val": -1, "max_val": 1},
        action_processor_cls=processors.RescaleProcessor,
        action_processor_config={"min_val": -1, "max_val": 1, "clip": True},
    )
    policy_specs[spec.id] = spec


# 10x10, 4 predators, 3 prey, prey_strength=3, cooperative=True
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
            "world": "10x10",
            "num_predators": 4,
            "num_prey": 3,
            "prey_strength": 3,
            "cooperative": True,
            "obs_dist": 4.0,
            "n_sensors": 16,
        },
        policy_file_path=osp.join(
            agent_model_dir,
            "world=10x10-num_predators=4-num_prey=3-prey_strength=3_cooperative=True-obs_dist=4-n_sensors=16",
            policy_file_name,
        ),
        version=0,
        valid_agent_ids=None,
        # policy is deterministic given random seed
        nondeterministic=False,
        # actions sampled, rather than always taking most probable action
        deterministic=False,
        obs_processor_cls=processors.RescaleProcessor,
        obs_processor_config={"min_val": -1, "max_val": 1},
        action_processor_cls=processors.RescaleProcessor,
        action_processor_config={"min_val": -1, "max_val": 1, "clip": True},
    )
    policy_specs[spec.id] = spec
