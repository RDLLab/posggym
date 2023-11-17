"""Policies for the LevelBasedForaging-v2 environment."""
import os.path as osp

from posggym.agents.grid_world.level_based_foraging import heuristic
from posggym.agents.registration import PolicySpec
from posggym.agents.torch_policy import PPOPolicy
from posggym.agents.utils import processors
from posggym.config import AGENT_MODEL_DIR


agent_model_dir = osp.join(AGENT_MODEL_DIR, "grid_world", "level_based_foraging")

policy_specs = {}
for policy_class in [
    heuristic.LBFHeuristic1,
    heuristic.LBFHeuristic2,
    heuristic.LBFHeuristic3,
    heuristic.LBFHeuristic4,
    heuristic.LBFHeuristic5,
]:
    num = policy_class.__name__.replace("LBFHeuristic", "")
    policy_spec = PolicySpec(
        policy_name=f"H{num}",
        entry_point=policy_class,
        env_id="LevelBasedForaging-v3",
        env_args=None,
        version=0,
        valid_agent_ids=None,
        nondeterministic=False,
        description=policy_class.__doc__.replace("\n", " "),
    )
    policy_specs[policy_spec.id] = policy_spec


for policy_file_name in [
    "RL1.pkl",
    "RL2.pkl",
    "RL3.pkl",
    "RL4.pkl",
    "RL5.pkl",
]:
    spec = PPOPolicy.get_spec_from_path(
        env_id="LevelBasedForaging-v3",
        env_args={
            "num_agents": 2,
            "max_agent_level": 3,
            "size": 10,
            "max_food": 8,
            "sight": 2,
            "force_coop": False,
            "static_layout": False,
            "observation_mode": "tuple",
        },
        env_args_id="num_agents=2-size=10-static_layout=False",
        policy_file_path=osp.join(
            agent_model_dir,
            "num_agents=2-size=10-static_layout=False",
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
        description="Deep RL policy trained using PPO and self-play.",
    )
    policy_specs[spec.id] = spec


for policy_file_name in [
    "RL1.pkl",
    "RL2.pkl",
    "RL3.pkl",
    "RL4.pkl",
    "RL5.pkl",
    "RL6.pkl",
    "RL7.pkl",
]:
    spec = PPOPolicy.get_spec_from_path(
        env_id="LevelBasedForaging-v3",
        env_args={
            "num_agents": 2,
            "max_agent_level": 3,
            "size": 10,
            "max_food": 8,
            "sight": 2,
            "force_coop": False,
            "static_layout": True,
            "observation_mode": "tuple",
        },
        env_args_id="num_agents=2-size=10-static_layout=True",
        policy_file_path=osp.join(
            agent_model_dir,
            "num_agents=2-size=10-static_layout=True",
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
        description="Deep RL policy trained using PPO and self-play.",
    )
    policy_specs[spec.id] = spec
