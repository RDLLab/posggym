"""Code for training self-play agents using rllib."""
import os
from typing import Callable, Optional

import ray
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

from posggym.agents.config import BASE_RESULTS_DIR
from posggym.agents.rllib import pbt
from posggym.agents.rllib.train.algorithm import (
    CustomPPOAlgorithm,
    get_algorithm,
    standard_logger_creator,
)
from posggym.agents.rllib.train.export_lib import export_igraph_algorithms
from posggym.agents.rllib.train.policy_mapping import IGraphPolicyMappingFn
from posggym.agents.rllib.train.train import run_training, sync_policies
from posggym.agents.rllib.train.utils import (
    RllibAlgorithmMap,
    posggym_registered_env_creator,
)
from posggym.wrappers.rllib_multi_agent_env import RllibMultiAgentEnv


def get_symmetric_sp_algorithm(
    env_id: str,
    env: RllibMultiAgentEnv,
    igraph: pbt.InteractionGraph,
    seed: Optional[int],
    config: AlgorithmConfig,
    num_gpus: float,
    logger_creator: Optional[Callable] = None,
) -> RllibAlgorithmMap:
    """Get Rllib algorithm for self-play trained agents in symmetric env."""
    assert igraph.is_symmetric

    # single algorithm so assign all of available gpu to it
    config.num_gpus = num_gpus
    config.num_gpus_per_trainer_worker = num_gpus

    # obs and action spaces are the same for all agents for symmetric envs
    agent_id = list(env.get_agent_ids())[0]
    obs_space = env.observation_space[agent_id]
    act_space = env.action_space[agent_id]

    train_policy_id = igraph.get_agent_policy_ids(igraph.SYMMETRIC_ID)[0]

    config.multi_agent(
        policies={
            train_policy_id: PolicySpec(PPOTorchPolicy, obs_space, act_space, {})
        },
        policy_mapping_fn=IGraphPolicyMappingFn(igraph),  # type: ignore
        policies_to_train=[train_policy_id],
        policy_states_are_swappable=True,
    )

    if logger_creator is None:
        logger_creator = standard_logger_creator(env_id, "sp", seed, train_policy_id)

    algorithm = get_algorithm(
        algorithm_class=CustomPPOAlgorithm,
        config=config,
        remote=True,
        logger_creator=logger_creator,
    )

    algorithms = {train_policy_id: algorithm}
    igraph.update_policy(
        igraph.SYMMETRIC_ID,
        train_policy_id,
        algorithm.get_weights.remote(train_policy_id),
    )

    # need to map from agent_id to algorithm map
    algorithm_map = {pbt.InteractionGraph.SYMMETRIC_ID: algorithms}
    return algorithm_map


def get_asymmetric_sp_algorithm(
    env_id: str,
    env: RllibMultiAgentEnv,
    igraph: pbt.InteractionGraph,
    seed: Optional[int],
    config: AlgorithmConfig,
    num_gpus: float,
    logger_creator: Optional[Callable] = None,
) -> RllibAlgorithmMap:
    """Get Rllib algorithm for self-play trained agents in asymmetric env."""
    assert not igraph.is_symmetric
    # assumes a single policy for each agent
    assert all(len(igraph.get_agent_policy_ids(i)) == 1 for i in igraph._agent_ids)

    policy_spec_map = {}
    for i in igraph.get_agent_ids():
        policy_id = igraph.get_agent_policy_ids(i)[0]
        policy_spec = PolicySpec(
            PPOTorchPolicy, env.observation_space[str(i)], env.action_space[str(i)], {}
        )
        policy_spec_map[policy_id] = policy_spec

    config.multi_agent(
        policies=policy_spec_map,
        policy_mapping_fn=IGraphPolicyMappingFn(igraph),  # type: ignore
        # may have difference act or obs spaces
        policy_states_are_swappable=False,
    )

    num_gpus_per_algorithm = num_gpus / len(env.get_agent_ids())
    config.num_gpus = num_gpus_per_algorithm
    config.num_gpus_per_trainer_worker = num_gpus_per_algorithm

    # map from agent_id to agent algorithm map
    algorithm_map = {}
    for agent_id in igraph.get_agent_ids():
        train_policy_id = igraph.get_agent_policy_ids(agent_id)[0]

        agent_config = config.copy(copy_frozen=False)
        agent_config.policies_to_train = [train_policy_id]

        if logger_creator is None:
            logger_creator = standard_logger_creator(
                env_id, "sp", seed, train_policy_id
            )

        algorithm = get_algorithm(
            algorithm_class=CustomPPOAlgorithm,
            config=config,
            remote=True,
            logger_creator=logger_creator,
        )

        algorithm_map[agent_id] = {train_policy_id: algorithm}
        igraph.update_policy(
            agent_id, train_policy_id, algorithm.get_weights.remote(train_policy_id)
        )

    sync_policies(algorithm_map, igraph)
    return algorithm_map


def train_sp_policy(
    env_id: str,
    seed: Optional[int],
    algorithm_config: AlgorithmConfig,
    num_gpus: float,
    num_iterations: int,
    save_policy: bool = True,
    verbose: bool = True,
):
    """Run training of self-play policy."""
    assert "env_config" in algorithm_config
    if "env_id" not in algorithm_config.env_config:
        algorithm_config.env_config["env_id"] = env_id

    ray.init()
    register_env(env_id, posggym_registered_env_creator)
    env = posggym_registered_env_creator(algorithm_config.env_config)

    agent_ids = list(env.get_agent_ids())
    agent_ids.sort()

    igraph = pbt.construct_sp_interaction_graph(
        agent_ids, is_symmetric=env.env.model.is_symmetric, seed=seed
    )
    igraph.display()

    algorithm_kwargs = {
        "env_id": env_id,
        "env": env,
        "igraph": igraph,
        "seed": seed,
        "config": algorithm_config,
        "num_gpus": num_gpus,
        "logger_creator": None,
    }
    if igraph.is_symmetric:
        algorithm_map = get_symmetric_sp_algorithm(**algorithm_kwargs)
    else:
        algorithm_map = get_asymmetric_sp_algorithm(**algorithm_kwargs)

    run_training(algorithm_map, igraph, num_iterations, verbose=verbose)

    if save_policy:
        print("== Exporting Graph ==")
        parent_dir = os.path.join(BASE_RESULTS_DIR, env_id, "policies")
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        save_dir = f"train_sp_{env_id}_seed{seed}"
        export_dir = export_igraph_algorithms(
            parent_dir,
            igraph,
            algorithm_map,
            remote=True,
            save_dir_name=save_dir,
        )
        print(f"{export_dir=}")
