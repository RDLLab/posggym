"""Policies for the Driving environments."""
import os.path as osp

from posggym.agents.grid_world.driving.shortest_path import DrivingShortestPathPolicy
from posggym.agents.registration import PolicySpec
from posggym.agents.torch_policy import PPOPolicy
from posggym.agents.utils import processors
from posggym.config import AGENT_MODEL_DIR

agent_model_dir = osp.join(AGENT_MODEL_DIR, "grid_world", "driving")
policy_specs = {}


for aggressiveness in [0.0, 0.4, 0.6, 0.8, 1.0]:
    name = f"A{aggressiveness*100:.0f}"
    spec = PolicySpec(
        policy_name=f"{name}Shortestpath",
        entry_point=DrivingShortestPathPolicy,
        version=0,
        env_id="Driving-v1",
        env_args=None,
        valid_agent_ids=None,
        nondeterministic=False,
        kwargs={"aggressiveness": aggressiveness},
    )
    policy_specs[spec.id] = spec


for policy_file_name in [
    "RL0.pkl",
    "RL1.pkl",
    "RL2.pkl",
    "RL3.pkl",
    "RL4.pkl",
]:
    spec = PPOPolicy.get_spec_from_path(
        env_id="Driving-v1",
        env_args={
            "grid": "14x14RoundAbout",
            "num_agents": 2,
            "obs_dim": (3, 1, 1),
        },
        env_args_id="grid=14x14RoundAbout-num_agents=2",
        policy_file_path=osp.join(
            agent_model_dir,
            "grid=14x14RoundAbout-num_agents=2",
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
