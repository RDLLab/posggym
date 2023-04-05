"""Custom Algorithm classes and functions for loading Rllib Algorithms."""
import os
import tempfile
from datetime import datetime
from typing import Callable, Optional

import ray
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo.ppo import PPO
from ray.tune.logger import NoopLogger, UnifiedLogger
from ray.tune.result import DEFAULT_RESULTS_DIR


class CustomPPOAlgorithm(PPO):
    """Custom Rllib algorithm class for the Rllib PPOPolicy.

    Adds functions custom features for experiments, training, etc.
    """

    def sync_weights(self):
        """Sync weights between all workers.

        This is only implemented so that it's easier to sync weights when
        running with Algorithms as ray remote Actors (i.e. when training in
        parallel).
        """
        self.workers.sync_weights()


def noop_logger_creator(config):
    """Create a NoopLogger for an rllib algorithm."""
    return NoopLogger(config, "")


def custom_log_creator(
    custom_path: str, custom_str: str, within_default_base_dir: bool = True
) -> Callable:
    """Get custom log creator that can be passed to an Algorithm.

    In particular `custom_path` specifies the path where results will be
    written by the algorithm. If `within_default_base_dir=True` then this will
    be within the RLLIB Default results dir `~/ray_results`.

    `custom_str` is used as the prefix for the logdir.
    """
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    if within_default_base_dir:
        custom_path = os.path.join(DEFAULT_RESULTS_DIR, custom_path)

    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        # use mkdtemp to handle race conditions if two processes try create same dir
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def standard_logger_creator(
    env_id: str, parent_dir: str, seed: Optional[int], suffix: Optional[str]
) -> Callable:
    """Get standard logger creator for training.

    Logs results to ~/ray_results/{env_id}/{parent_dir}/seed{seed}{_suffix}
    """
    custom_path = os.path.join(env_id, parent_dir)

    custom_str = f"seed{seed}"
    if suffix is not None:
        custom_str += f"_{suffix}"

    return custom_log_creator(custom_path, custom_str, True)


def get_algorithm(
    algorithm_class,
    config: AlgorithmConfig,
    remote: bool,
    logger_creator: Optional[Callable] = None,
):
    """Get algorithm."""
    if remote:
        num_cpus = (
            config.num_rollout_workers * config.num_cpus_per_worker
            + config.num_trainer_workers * config.num_cpus_per_trainer_worker
        )

        algorithm_remote = ray.remote(
            num_cpus=num_cpus,
            num_gpus=config.num_gpus,
        )(algorithm_class)
        algorithm = algorithm_remote.remote(
            config=config, logger_creator=logger_creator
        )
    else:
        algorithm = algorithm_class(config=config, logger_creator=logger_creator)
    return algorithm
