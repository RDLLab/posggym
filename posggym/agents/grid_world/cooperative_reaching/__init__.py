"""Policies for the CooperativeReaching-v0 environment."""
from posggym.agents.grid_world.cooperative_reaching import heuristic
from posggym.agents.registration import PolicySpec

policy_specs = {}
for i, policy_class in enumerate(
    [
        heuristic.CRHeuristic1,
        heuristic.CRHeuristic2,
        heuristic.CRHeuristic3,
        heuristic.CRHeuristic4,
        heuristic.CRHeuristic5,
        heuristic.CRHeuristic6,
        heuristic.CRHeuristic7,
        heuristic.CRHeuristic8,
        heuristic.CRHeuristic9,
        heuristic.CRHeuristic10,
        heuristic.CRHeuristic11,
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
