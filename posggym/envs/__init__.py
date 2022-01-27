"""Initializes posg implemented environments.

Utilizes the OpenAI Gym env registration functionality.
"""
from posggym.envs.registration import register
from posggym.envs.registration import make
from posggym.envs.registration import spec
from posggym.envs.registration import registry

# Full Model
# -------------------------------------------

register(
    env_id="MABC-v0",
    entry_point="posggym.envs.full_model.mabc:MABCEnv"
)
