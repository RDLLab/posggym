"""Root '__init__' of the posggym package."""
# isort: skip_file
from posggym.agents.registration import (
    make,
    pprint_registry,
    register,
    register_spec,
    registry,
    spec,
)
from posggym.agents.policy import Policy
from posggym.agents import random_policies
from posggym.agents.continuous import drone_team_capture, driving as DrivingContinuous, pursuit_evasion as PEContinuous
from posggym.agents.grid_world import driving, lbf, predator_prey, pursuit_evasion


__all__ = [
    # core classes
    "Policy",
    # registration
    "make",
    "pprint_registry",
    "register",
    "register_spec",
    "registry",
    "spec",
]


# Generic Random Policies
# ------------------------------

register(
    policy_name="Random",
    entry_point=random_policies.RandomPolicy,
    version=0,
    env_id=None,
    env_args=None,
    valid_agent_ids=None,
    nondeterministic=False,
)

register(
    policy_name="DiscreteFixedDistributionPolicy",
    entry_point=random_policies.DiscreteFixedDistributionPolicy,
    version=0,
    env_id=None,
    env_args=None,
    valid_agent_ids=None,
    nondeterministic=False,
)

# Continuous
# ----------

# Drone Team Capture
for policy_spec in drone_team_capture.policy_specs.values():
    register_spec(policy_spec)

for policy_spec in DrivingContinuous.policy_specs.values():
    register_spec(policy_spec)

for policy_spec in PEContinuous.policy_specs.values():
    register_spec(policy_spec)

# Grid World
# ----------

# Driving
for policy_spec in driving.policy_specs.values():
    register_spec(policy_spec)


# Level Based Foraging
for policy_spec in lbf.policy_specs.values():
    register_spec(policy_spec)


# Predator Prey
for policy_spec in predator_prey.policy_specs.values():
    register_spec(policy_spec)


# Pursuit Evasion
for policy_spec in pursuit_evasion.policy_specs.values():
    register_spec(policy_spec)
