"""Policies for the PredatorPrey-v0 environment."""
import os.path as osp

from posggym.agents.grid_world.predator_prey import heuristic
from posggym.agents.registration import PolicySpec
from posggym.agents.torch_policy import PPOPolicy
from posggym.agents.utils import processors
from posggym.config import AGENT_MODEL_DIR

agent_model_dir = osp.join(AGENT_MODEL_DIR, "grid_world", "predator_prey")
policy_specs = {}

for policy_class in [
    heuristic.PPHeuristic1,
    heuristic.PPHeuristic2,
    heuristic.PPHeuristic3,
]:
    num = policy_class.__name__.replace("PPHeuristic", "")
    spec = PolicySpec(
        policy_name=f"H{num}",
        entry_point=policy_class,
        env_id="PredatorPrey-v0",
        env_args=None,
        version=0,
        valid_agent_ids=None,
        nondeterministic=False,
        description=policy_class.__doc__.replace("\n", " "),
    )
    policy_specs[spec.id] = spec

# PredatorPrey-10x10-P2-p3-s2-coop-v0
for policy_file_name in [
    "RL1.pkl",
    "RL2.pkl",
    "RL3.pkl",
    "RL4.pkl",
    "RL5.pkl",
    "RL6.pkl",
    "RL7.pkl",
    "RL8.pkl",
]:
    spec = PPOPolicy.get_spec_from_path(
        env_id="PredatorPrey-v0",
        env_args={
            "grid": "10x10",
            "num_predators": 2,
            "num_prey": 3,
            "cooperative": True,
            "prey_strength": 2,
            "obs_dim": 2,
        },
        env_args_id="grid=10x10-num_predators=2-num_prey=3-cooperative=True",
        policy_file_path=osp.join(
            agent_model_dir,
            "grid=10x10-num_predators=2-num_prey=3-cooperative=True",
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
