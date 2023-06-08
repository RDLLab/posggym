"""Module of wrapper classes.

Ref:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/wrappers/__init__.py
"""
from posggym.wrappers.discretize_actions import DiscretizeActions
from posggym.wrappers.env_checker import PassiveEnvChecker
from posggym.wrappers.flatten_observations import FlattenObservations
from posggym.wrappers.order_enforcing import OrderEnforcing
from posggym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from posggym.wrappers.record_video import RecordVideo
from posggym.wrappers.rescale_actions import RescaleActions
from posggym.wrappers.rescale_observations import RescaleObservations
from posggym.wrappers.time_limit import TimeLimit
