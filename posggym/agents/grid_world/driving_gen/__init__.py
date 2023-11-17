"""Policies for the DrivingGen environments."""
from posggym.agents.grid_world.driving_gen.shortest_path import (
    DrivingGenShortestPathPolicy,
)
from posggym.agents.registration import PolicySpec

policy_specs = {}
for a, description in [
    (
        0.0,
        (
            "Follows shortest path to destination, doesn't go full speed, and stops if "
            "it observes another a car at any distance (aggressiveness=0.0)"
        ),
    ),
    (
        0.4,
        (
            "Follows shortest path to destination, doesn't go full speed, and stops if "
            "it observes a car near edge of observation range (aggressiveness=0.40)"
        ),
    ),
    (
        0.6,
        (
            "Follows shortest path to destination, goes up to full speed, and stops if "
            "it observes a car a medium distance away (aggressiveness=0.60)"
        ),
    ),
    (
        0.8,
        (
            "Follows shortest path to destination, goes up to full speed, and stops if "
            "it observes a car a very short distance away (aggressiveness=0.80)"
        ),
    ),
    (
        1.0,
        (
            "Follows shortest path to destination, goes up to full speed, and ignores "
            "other vehicles (aggressiveness=1.0)"
        ),
    ),
]:
    name = f"A{a*100:.0f}"
    spec = PolicySpec(
        policy_name=f"{name}Shortestpath",
        entry_point=DrivingGenShortestPathPolicy,
        version=0,
        env_id="DrivingGen-v1",
        env_args=None,
        valid_agent_ids=None,
        nondeterministic=False,
        kwargs={"aggressiveness": a},
        description=description,
    )
    policy_specs[spec.id] = spec
