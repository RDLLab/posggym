"""Policies for the CooperativeReaching-v0 environment."""
from posggym.agents.grid_world.cooperative_reaching import heuristic
from posggym.agents.registration import PolicySpec

policy_specs = {}
for policy_class in [
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
]:
    num = policy_class.__name__.replace("CRHeuristic", "")
    policy_spec = PolicySpec(
        policy_name=f"H{num}",
        entry_point=policy_class,
        env_id="CooperativeReaching-v0",
        env_args=None,
        version=0,
        valid_agent_ids=None,
        nondeterministic=False,
        description=policy_class.__doc__.replace("\n", " "),
    )
    policy_specs[policy_spec.id] = policy_spec
