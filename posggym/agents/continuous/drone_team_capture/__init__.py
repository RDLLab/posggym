"""Policies for DroneTeamCapture-v0 environment."""
from posggym.agents.continuous.drone_team_capture import heuristic
from posggym.agents.registration import PolicySpec


policy_specs = {}
for policy_name, policy_class in [
    ("DTCAngelaniHeuristic", heuristic.DTCAngelaniHeuristicPolicy),
    ("DTCDPPHeuristic", heuristic.DTCDPPHeuristicPolicy),
    ("DTCJanosovHeuristic", heuristic.DTCJanosovHeuristicPolicy),
    ("DTCGreedyHeuristicPolicy", heuristic.DTCGreedyHeuristicPolicy),
]:
    policy_spec = PolicySpec(
        policy_name=policy_name,
        entry_point=policy_class,
        env_id="DroneTeamCapture-v0",
        env_args=None,
        version=0,
        valid_agent_ids=None,
        nondeterministic=False,
    )
    policy_specs[policy_spec.id] = policy_spec
