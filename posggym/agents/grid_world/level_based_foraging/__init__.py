"""Policies for the LevelBasedForaging-v2 environment."""
import posggym.agents.grid_world.level_based_foraging.heuristic as heuristic_agent
from posggym.agents.registration import PolicySpec
from posggym.envs.registration import registry

policy_specs = {}
for i, policy_class in enumerate(
    [
        heuristic_agent.LBFHeuristicPolicy1,
        heuristic_agent.LBFHeuristicPolicy2,
        heuristic_agent.LBFHeuristicPolicy3,
        heuristic_agent.LBFHeuristicPolicy4,
    ]
):
    policy_spec = PolicySpec(
        policy_name=f"Heuristic{i+1}",
        entry_point=policy_class,
        env_id="LevelBasedForaging-v2",
        env_args=None,
        version=0,
        valid_agent_ids=None,
        nondeterministic=False,
    )
    policy_specs[policy_spec.id] = policy_spec
