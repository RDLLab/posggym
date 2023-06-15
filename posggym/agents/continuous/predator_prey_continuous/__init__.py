"""Policies for the PredatorPreyContinuous-v0 environment."""

from posggym.agents.registration import PolicySpec
from posggym.agents.continuous.predator_prey_continuous import heuristic

ENV_ID = "PursuitEvasion-v0"
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
        env_id="PredatorPreyContinuous-v0",
        env_args=None,
        valid_agent_ids=None,
        nondeterministic=False,
    )
    policy_specs[_heuristic_spec.id] = _heuristic_spec
