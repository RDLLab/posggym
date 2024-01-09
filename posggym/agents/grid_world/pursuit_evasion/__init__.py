"""Policies for the PursuitEvasion environment."""
import os.path as osp

from posggym.agents.grid_world.pursuit_evasion.shortest_path import PEShortestPathPolicy
from posggym.agents.registration import PolicySpec
from posggym.agents.torch_policy import PPOPolicy
from posggym.agents.utils import processors
from posggym.config import AGENT_MODEL_DIR

agent_model_dir = osp.join(AGENT_MODEL_DIR, "grid_world", "pursuit_evasion")
policy_specs = {}

spec = PolicySpec(
    policy_name="ShortestPath",
    entry_point=PEShortestPathPolicy,
    version=0,
    env_id="PursuitEvasion-v1",
    env_args=None,
    valid_agent_ids=None,
    nondeterministic=False,
    description=(
        "Takes the shortest path to the evader's goal (evader) or the evader's start "
        "location then the other possible evader start and goal locations (pursuer)."
    ),
)
policy_specs[spec.id] = spec


def _get_policy_description(policy_file_name):
    if policy_file_name.startswith("RL"):
        return "Deep RL policy trained using PPO and self-play."
    k = policy_file_name.split("_")[0].replace("KLR", "")
    if k == "BR":
        return (
            "Best-response to K-Level Reasoning policies. This is a deep RL policy "
            "training"
            " using PPO and the Synchronous KLR algorithm."
        )
    return (
        f"Level {k} K-Level Reasoning deep RL policy training using PPO and the "
        "Synchronous KLR algorithm."
    )


# PursuitEvasion 16x16
# Evader (agent=0)
for policy_file_name in [
    "KLR0_i0.pkl",
    "KLR1_i0.pkl",
    "KLR2_i0.pkl",
    "KLR3_i0.pkl",
    "KLR4_i0.pkl",
    "KLRBR_i0.pkl",
    "RL1_i0.pkl",
    "RL2_i0.pkl",
    "RL3_i0.pkl",
    "RL4_i0.pkl",
    "RL5_i0.pkl",
    "RL6_i0.pkl",
    "RL7_i0.pkl",
    "RL8_i0.pkl",
    "RL9_i0.pkl",
    "RL10_i0.pkl",
    "RL11_i0.pkl",
]:
    spec = PPOPolicy.get_spec_from_path(
        env_id="PursuitEvasion-v1",
        env_args={
            "grid": "16x16",
            "max_obs_distance": 12,
            "use_progress_reward": True,
        },
        env_args_id="grid=16x16",
        policy_file_path=osp.join(agent_model_dir, "grid=16x16", policy_file_name),
        version=0,
        valid_agent_ids=["0"],
        # policy is deterministic given random seed
        nondeterministic=False,
        # actions sampled, rather always taking most probable action
        deterministic=False,
        obs_processor_cls=processors.FlattenProcessor,
        obs_processor_config=None,
        action_processor_cls=None,
        action_processor_config=None,
        description=_get_policy_description(policy_file_name),
    )
    policy_specs[spec.id] = spec


# Pursuer (agent=1)
for policy_file_name in [
    "KLR0_i1.pkl",
    "KLR1_i1.pkl",
    "KLR2_i1.pkl",
    "KLR3_i1.pkl",
    "KLR4_i1.pkl",
    "KLRBR_i1.pkl",
    "RL1_i1.pkl",
    "RL2_i1.pkl",
    "RL3_i1.pkl",
    "RL4_i1.pkl",
    "RL5_i1.pkl",
    "RL6_i1.pkl",
    "RL7_i1.pkl",
    "RL8_i1.pkl",
    "RL9_i1.pkl",
    "RL10_i1.pkl",
    "RL11_i1.pkl",
]:
    spec = PPOPolicy.get_spec_from_path(
        env_id="PursuitEvasion-v1",
        env_args={
            "grid": "16x16",
            "max_obs_distance": 12,
            "use_progress_reward": True,
        },
        env_args_id="grid=16x16",
        policy_file_path=osp.join(agent_model_dir, "grid=16x16", policy_file_name),
        version=0,
        valid_agent_ids=["1"],
        # policy is deterministic given random seed
        nondeterministic=False,
        # actions sampled, rather than always taking most probable action
        deterministic=False,
        obs_processor_cls=processors.FlattenProcessor,
        obs_processor_config=None,
        action_processor_cls=None,
        action_processor_config=None,
        description=_get_policy_description(policy_file_name),
    )
    policy_specs[spec.id] = spec

# PursuitEvasion 8x8
# Evader (agent=0)
for policy_file_name in [
    "KLR0_i0.pkl",
    "KLR1_i0.pkl",
    "KLR2_i0.pkl",
    "KLR3_i0.pkl",
    "KLR4_i0.pkl",
    "KLRBR_i0.pkl",
    "RL1_i0.pkl",
    "RL2_i0.pkl",
    "RL3_i0.pkl",
    "RL4_i0.pkl",
    "RL5_i0.pkl",
    "RL6_i0.pkl",
    "RL7_i0.pkl",
]:
    spec = PPOPolicy.get_spec_from_path(
        env_id="PursuitEvasion-v1",
        env_args={
            "grid": "8x8",
            "max_obs_distance": 12,
            "use_progress_reward": True,
        },
        env_args_id="grid=8x8",
        policy_file_path=osp.join(agent_model_dir, "grid=8x8", policy_file_name),
        version=0,
        valid_agent_ids=["0"],
        # policy is deterministic given random seed
        nondeterministic=False,
        # actions sampled, rather always taking most probable action
        deterministic=False,
        obs_processor_cls=processors.FlattenProcessor,
        obs_processor_config=None,
        action_processor_cls=None,
        action_processor_config=None,
        description=_get_policy_description(policy_file_name),
    )
    policy_specs[spec.id] = spec


# Pursuer (agent=1)
for policy_file_name in [
    "KLR0_i1.pkl",
    "KLR1_i1.pkl",
    "KLR2_i1.pkl",
    "KLR3_i1.pkl",
    "KLR4_i1.pkl",
    "KLRBR_i1.pkl",
    "RL1_i1.pkl",
    "RL2_i1.pkl",
    "RL3_i1.pkl",
    "RL4_i1.pkl",
    "RL5_i1.pkl",
    "RL6_i1.pkl",
    "RL7_i1.pkl",
]:
    spec = PPOPolicy.get_spec_from_path(
        env_id="PursuitEvasion-v1",
        env_args={
            "grid": "8x8",
            "max_obs_distance": 12,
            "use_progress_reward": True,
        },
        env_args_id="grid=8x8",
        policy_file_path=osp.join(agent_model_dir, "grid=8x8", policy_file_name),
        version=0,
        valid_agent_ids=["1"],
        # policy is deterministic given random seed
        nondeterministic=False,
        # actions sampled, rather always taking most probable action
        deterministic=False,
        obs_processor_cls=processors.FlattenProcessor,
        obs_processor_config=None,
        action_processor_cls=None,
        action_processor_config=None,
        description=_get_policy_description(policy_file_name),
    )
    policy_specs[spec.id] = spec
