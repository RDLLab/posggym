"""Module of wrapper classes.

Ref:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/wrappers/__init__.py
"""
from posggym.wrappers.env_checker import PassiveEnvChecker
# from posggym.wrappers.flatten_observation import FlattenObservation
from posggym.wrappers.order_enforcing import OrderEnforcing
# from posggym.wrappers.record_env import RecordVideo
from posggym.wrappers.time_limit import TimeLimit
