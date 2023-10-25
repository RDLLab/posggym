"""Policies for the LevelBasedForaging-v2 environment."""
from posggym.agents.grid_world.level_based_foraging import heuristic
from posggym.agents.registration import PolicySpec

policy_specs = {}
for i, policy_class in enumerate(
    [
        heuristic.LBFHeuristicPolicy1,
        heuristic.LBFHeuristicPolicy2,
        heuristic.LBFHeuristicPolicy3,
        heuristic.LBFHeuristicPolicy4,
        heuristic.LBFHeuristicPolicy5,
        heuristic.LBFHeuristicPolicy6,
        heuristic.LBFHeuristicPolicy7,
        heuristic.LBFHeuristicPolicy8,
        heuristic.LBFHeuristicPolicy9,
        heuristic.LBFHeuristicPolicy10,
        heuristic.LBFHeuristicPolicy11,
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
