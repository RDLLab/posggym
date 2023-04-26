"""Functions for importing rllib algorithms from a file."""
import os
import os.path as osp
import pickle
from typing import Callable, Optional, Tuple, Type

import ray
from ray.rllib.algorithms.algorithm import Algorithm

import posggym.model as M
from posggym.agents.policy import PolicyID
from posggym.agents.train import pbt
from posggym.agents.rllib_policy import PPORllibPolicy, RllibPolicy
from posggym.agents.utils.preprocessors import ObsPreprocessor, identity_preprocessor
from posggym.agents.train.rllib.algorithm import (
    CustomPPOAlgorithm,
    get_algorithm,
    noop_logger_creator,
)
from posggym.agents.train.rllib.export_lib import ALGORITHM_CONFIG_FILE
from posggym.agents.train.rllib.utils import RllibAlgorithmMap, register_posggym_env


def import_algorithm(
    algorithm_dir: str,
    algorithm_class: Type[Algorithm],
    remote: bool,
    logger_creator: Optional[Callable] = None,
) -> Optional[Algorithm]:
    """Import algorithm from a directory."""
    checkpoints = [f for f in os.listdir(algorithm_dir) if f.startswith("checkpoint")]
    if len(checkpoints) == 0:
        # untrained policy, e.g. a random policy
        return None

    # In case multiple checkpoints are stored, take the latest one
    # Checkpoints are named as 'checkpoint_{iteration}'
    checkpoints.sort()
    checkpoint_dir_path = osp.join(algorithm_dir, checkpoints[-1])

    # import algorithm config
    with open(osp.join(algorithm_dir, ALGORITHM_CONFIG_FILE), "rb") as fin:
        config = pickle.load(fin)

    # need to make env_id is registered properly
    register_posggym_env(config["env_config"]["env_id"])

    algorithm = get_algorithm(
        algorithm_class=algorithm_class,
        config=config,
        remote=remote,
        logger_creator=logger_creator,
    )

    if remote:
        ray.get(algorithm.restore.remote(checkpoint_dir_path))  # type: ignore
        return algorithm

    algorithm.restore(checkpoint_dir_path)
    return algorithm


def import_policy_from_dir(
    model: M.POSGModel,
    agent_id: M.AgentID,
    policy_id: PolicyID,
    policy_dir: str,
    policy_cls: Optional[Type[RllibPolicy]] = None,
    algorithm_cls: Optional[Type[Algorithm]] = None,
    preprocessor: Optional[ObsPreprocessor] = None,
    **kwargs,
) -> Optional[RllibPolicy]:
    """Import Rllib Policy from a directory containing saved checkpoint.

    This imports the underlying rllib.Policy object and then handles wrapping
    it within a compatible policy so it's compatible with posggym-agents Policy
    API.

    Note, this policy imports the function assuming the policy will be used
    as is without any further training.

    """
    if algorithm_cls is None:
        algorithm_cls = CustomPPOAlgorithm

    if policy_cls is None:
        policy_cls = PPORllibPolicy

    if preprocessor is None:
        preprocessor = identity_preprocessor

    algorithm = import_algorithm(
        policy_dir,
        algorithm_class=algorithm_cls,
        remote=False,
        logger_creator=noop_logger_creator,
    )
    if algorithm is None:
        # non-trainable policy (e.g. random)
        return None

    # be default this is the name of the dir
    algorithm_policy_id = osp.basename(osp.normpath(policy_dir))
    # release algorithm resources to avoid accumulation of background processes
    rllib_policy = algorithm.get_policy(algorithm_policy_id)
    algorithm.stop()

    return policy_cls(
        model=model,
        agent_id=agent_id,
        policy_id=policy_id,
        policy=rllib_policy,
        preprocessor=preprocessor,
    )


def import_igraph_algorithms(
    igraph_dir: str,
    algorithm_class: Type[Algorithm],
    remote: bool,
    seed: Optional[int] = None,
    logger_creator: Optional[Callable] = None,
) -> Tuple[pbt.InteractionGraph, RllibAlgorithmMap]:
    """Import Rllib algorithms from InteractionGraph directory.

    If policy_mapping_fn is None then will use function from
    baposgmcp.rllib.utils.get_igraph_policy_mapping_function.
    """
    algorithm_map: RllibAlgorithmMap = {}

    def import_fn(
        agent_id: M.AgentID, policy_id: PolicyID, import_dir: str
    ) -> pbt.PolicyState:
        """Imports trained policy weights from local directory.

        The function also returns a reference to a algorithm map object which is
        populated with Algorithm objects as the algorithm import function is called.

        The import function:
        1. Creates a new Algorithm object
        2. Restores the algorithms state from the file in the import dir
        3. Adds algorithm to the algorithm map
        4. Returns the weights of the policy with given ID
        """
        algorithm = import_algorithm(
            algorithm_dir=import_dir,
            algorithm_class=algorithm_class,
            remote=remote,
            logger_creator=logger_creator,
        )
        if algorithm is None:
            # handle save dirs that contain no exported algorithm
            # e.g. save dirs for random policy
            return {}

        if agent_id not in algorithm_map:
            algorithm_map[agent_id] = {}

        if remote:
            weights = algorithm.get_weights.remote(policy_id)  # type: ignore
        else:
            weights = algorithm.get_weights(policy_id)

        algorithm_map[agent_id][policy_id] = algorithm
        return weights

    igraph = pbt.InteractionGraph.import_graph(igraph_dir, import_fn, seed)
    return igraph, algorithm_map
