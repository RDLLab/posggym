"""Root '__init__' of the posggym.agents package."""
# isort: skip_file
from posggym.agents.registration import (
    make,
    pprint_registry,
    register,
    register_spec,
    registry,
    spec,
    get_all_envs,
    get_all_env_policies,
    get_env_agent_policies,
)
from posggym.agents.policy import Policy
from posggym.agents import random_policies
from posggym.agents.continuous import (
    drone_team_capture,
    driving_continuous,
    pursuit_evasion_continuous,
    predator_prey_continuous,
)
from posggym.agents.grid_world import (
    cooperative_reaching,
    driving,
    driving_gen,
    level_based_foraging,
    predator_prey,
    pursuit_evasion,
)


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
    "get_all_envs",
    "get_all_env_policies",
    "get_env_agent_policies",
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
    description="The uniform random policy",
)

register(
    policy_name="DiscreteFixedDistributionPolicy",
    entry_point=random_policies.DiscreteFixedDistributionPolicy,
    version=0,
    env_id=None,
    env_args=None,
    valid_agent_ids=None,
    nondeterministic=False,
    description=(
        "Random policy that follows a fixed discrete distribution (default is uniform)"
    ),
)

# Continuous
# ----------

# Drone Team Capture
for policy_spec in drone_team_capture.policy_specs.values():
    register_spec(policy_spec)

for policy_spec in driving_continuous.policy_specs.values():
    register_spec(policy_spec)

for policy_spec in predator_prey_continuous.policy_specs.values():
    register_spec(policy_spec)

for policy_spec in pursuit_evasion_continuous.policy_specs.values():
    register_spec(policy_spec)


# Grid World
# ----------
# Cooperative Reaching
for policy_spec in cooperative_reaching.policy_specs.values():
    register_spec(policy_spec)

# Driving
for policy_spec in driving.policy_specs.values():
    register_spec(policy_spec)

# Driving Gen
for policy_spec in driving_gen.policy_specs.values():
    register_spec(policy_spec)


# Level Based Foraging
for policy_spec in level_based_foraging.policy_specs.values():
    register_spec(policy_spec)


# Predator Prey
for policy_spec in predator_prey.policy_specs.values():
    register_spec(policy_spec)


# Pursuit Evasion
for policy_spec in pursuit_evasion.policy_specs.values():
    register_spec(policy_spec)
