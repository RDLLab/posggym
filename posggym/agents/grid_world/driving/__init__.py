"""Policies for the Driving environments."""
from posggym.agents.grid_world.driving.shortest_path import DrivingShortestPathPolicy
from posggym.agents.registration import PolicySpec


ENV_ID = "Driving-v1"
policy_specs = {}

for name, aggressiveness in [
    ("aggressive", 1),
    ("normal", 0.5),
    ("conservative", 0),
]:
    spec = PolicySpec(
        policy_name=f"{name}_shortestpath",
        entry_point=DrivingShortestPathPolicy,
        version=0,
        env_id="Driving-v1",
        env_args=None,
        valid_agent_ids=None,
        nondeterministic=False,
        kwargs={"aggressiveness": aggressiveness},
    )
    policy_specs[spec.id] = spec
