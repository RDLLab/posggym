"""Policies for the CooperativeReaching-v0 environment."""
from posggym.agents.grid_world.cooperative_reaching import heuristic
from posggym.agents.registration import PolicySpec

policy_specs = {}
for i, policy_class in enumerate(
    [
        heuristic.CRHeuristicPolicy1,
        heuristic.CRHeuristicPolicy2,
        heuristic.CRHeuristicPolicy3,
        heuristic.CRHeuristicPolicy4,
        heuristic.CRHeuristicPolicy5,
        heuristic.CRHeuristicPolicy6,
        heuristic.CRHeuristicPolicy7,
        heuristic.CRHeuristicPolicy8,
        heuristic.CRHeuristicPolicy9,
        heuristic.CRHeuristicPolicy10,
    ]
):
    policy_spec = PolicySpec(
        policy_name=f"Heuristic{i+1}",
        entry_point=policy_class,
        env_id="CooperativeReaching-v0",
        env_args=None,
        version=0,
        valid_agent_ids=None,
        nondeterministic=False,
    )
    policy_specs[policy_spec.id] = policy_spec
