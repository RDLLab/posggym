"""Policies for the DrivingGen environments."""
from posggym.agents.grid_world.driving_gen.shortest_path import (
    DrivingGenShortestPathPolicy,
)
from posggym.agents.registration import PolicySpec

policy_specs = {}

for name, aggressiveness in [
    ("aggressive", 1),
    ("normal", 0.5),
    ("conservative", 0),
]:
    spec = PolicySpec(
        policy_name=f"{name}_shortestpath",
        entry_point=DrivingGenShortestPathPolicy,
        version=0,
        env_id="DrivingGen-v1",
        env_args=None,
        valid_agent_ids=None,
        nondeterministic=False,
        kwargs={"aggressiveness": aggressiveness},
    )
    policy_specs[spec.id] = spec
