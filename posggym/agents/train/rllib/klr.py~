"""Code for training K-Level Reasoning agents using rllib."""
import os
from typing import Callable, Optional

import ray
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy
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


def get_symmetric_klr_algorithm(
    env_id: str,
    env: RllibMultiAgentEnv,
    igraph: pbt.InteractionGraph,
    seed: Optional[int],
    config: AlgorithmConfig,
    num_gpus: float,
    logger_creator: Optional[Callable] = None,
    run_serially: bool = False,
) -> RllibAlgorithmMap:
    """Get Rllib algorithm for K-Level Reasoning agents in symmetric env."""
    assert igraph.is_symmetric, "Currently only symmetric envs supported."

    # use ray remote actors to train policies simoultaneously if not running serially
    remote = not run_serially

    # -1 for random policy
    num_algorithms = len(igraph.get_agent_policy_ids(igraph.SYMMETRIC_ID)) - 1
    num_gpus_per_algorithm = num_gpus / num_algorithms
    config.num_gpus = num_gpus_per_algorithm
    config.num_gpus_per_trainer_worker = num_gpus_per_algorithm

    # obs and action spaces are the same for all agents for symmetric envs
    agent_id = list(env.get_agent_ids())[0]
    obs_space = env.observation_space[agent_id]
    act_space = env.action_space[agent_id]
    random_policy_spec = PolicySpec(RandomPolicy, obs_space, act_space, {})
    ppo_policy_spec = PolicySpec(PPOTorchPolicy, obs_space, act_space, {})

    algorithms = {}
    for train_policy_id in igraph.get_agent_policy_ids(igraph.SYMMETRIC_ID):
        connected_policies = igraph.get_all_policies(
            igraph.SYMMETRIC_ID, train_policy_id, igraph.SYMMETRIC_ID
        )
        if len(connected_policies) == 0:
            # k = -1
            continue

        train_policy_spec = ppo_policy_spec
        policy_spec_map = {train_policy_id: train_policy_spec}
        for policy_j_id, _ in connected_policies:
            _, k = pbt.parse_klr_policy_id(policy_j_id)
            policy_spec_j = random_policy_spec if k == -1 else ppo_policy_spec
            policy_spec_map[policy_j_id] = policy_spec_j

        policy_config = config.copy(copy_frozen=False)
        policy_config.multi_agent(
            policies=policy_spec_map,
            policy_mapping_fn=IGraphPolicyMappingFn(igraph),  # type: ignore
            policies_to_train=[train_policy_id],
            policy_states_are_swappable=True,
        )

        if logger_creator is None:
            policy_logger_creator = standard_logger_creator(
                env_id, "klr", seed, train_policy_id
            )

        algorithm_k = get_algorithm(
            algorithm_class=CustomPPOAlgorithm,
            config=policy_config,
            remote=remote,
            logger_creator=policy_logger_creator,
        )

        if remote:
            algorithm_k_weights = algorithm_k.get_weights.remote([train_policy_id])
        else:
            algorithm_k_weights = algorithm_k.get_weights([train_policy_id])

        algorithms[train_policy_id] = algorithm_k
        igraph.update_policy(igraph.SYMMETRIC_ID, train_policy_id, algorithm_k_weights)

    # need to map from agent_id to algorithms
    algorithm_map = {pbt.InteractionGraph.SYMMETRIC_ID: algorithms}
    return algorithm_map


def get_asymmetric_klr_algorithm(
    env_id: str,
    env: RllibMultiAgentEnv,
    igraph: pbt.InteractionGraph,
    seed: Optional[int],
    config: AlgorithmConfig,
    num_gpus: float,
    logger_creator: Optional[Callable] = None,
    run_serially: bool = False,
) -> RllibAlgorithmMap:
    """Get Rllib algorithm for K-Level Reasoning agents in asymmetric env."""
    assert not igraph.is_symmetric

    # use ray remote actors to train policies simoultaneously if not running serially
    remote = not run_serially

    # one algorithm per K per agent, -1 for random policy
    num_algorithms = sum(
        len(agent_policies) - 1 for agent_policies in igraph.policies.values()
    )
    num_gpus_per_algorithm = num_gpus / num_algorithms
    config.num_gpus = num_gpus_per_algorithm
    config.num_gpus_per_trainer_worker = num_gpus_per_algorithm

    agent_ppo_policy_specs = {
        i: PolicySpec(
            PPOTorchPolicy, env.observation_space[str(i)], env.action_space[str(i)], {}
        )
        for i in igraph.get_agent_ids()
    }
    agent_random_policy_specs = {
        i: PolicySpec(
            RandomPolicy, env.observation_space[str(i)], env.action_space[str(i)], {}
        )
        for i in igraph.get_agent_ids()
    }

    # map from agent_id to agent algorithm map
    algorithm_map: RllibAlgorithmMap = {i: {} for i in igraph.get_agent_ids()}
    for agent_id in igraph.get_agent_ids():
        for train_policy_id in igraph.get_agent_policy_ids(agent_id):
            connected_policy_ids = []
            for j in igraph.get_agent_ids():
                connected_policies = igraph.get_all_policies(
                    agent_id, train_policy_id, j
                )
                connected_policy_ids.extend([c[0] for c in connected_policies])

            if len(connected_policy_ids) == 0:
                # k = -1
                continue

            train_policy_spec = agent_ppo_policy_specs[agent_id]
            policy_spec_map = {train_policy_id: train_policy_spec}
            for policy_j_id in connected_policy_ids:
                agent_id_j, k = pbt.parse_klr_policy_id(policy_j_id)
                if k == -1:
                    policy_spec_j = agent_random_policy_specs[str(agent_id_j)]
                else:
                    policy_spec_j = agent_ppo_policy_specs[str(agent_id_j)]
                policy_spec_map[policy_j_id] = policy_spec_j

            policy_config = config.copy(copy_frozen=False)
            policy_config.multi_agent(
                policies=policy_spec_map,
                policy_mapping_fn=IGraphPolicyMappingFn(igraph),  # type: ignore
                policies_to_train=[train_policy_id],
                # agents can have different action and obs spaces
                policy_states_are_swappable=False,
            )

            if logger_creator is None:
                policy_logger_creator = standard_logger_creator(
                    env_id, "klr", seed, train_policy_id
                )

            algorithm_k = get_algorithm(
                algorithm_class=CustomPPOAlgorithm,
                config=policy_config,
                remote=remote,
                logger_creator=policy_logger_creator,
            )

            if remote:
                algorithm_k_weights = algorithm_k.get_weights.remote([train_policy_id])
            else:
                algorithm_k_weights = algorithm_k.get_weights([train_policy_id])

            algorithm_map[agent_id][train_policy_id] = algorithm_k
            igraph.update_policy(agent_id, train_policy_id, algorithm_k_weights)

    sync_policies(algorithm_map, igraph)
    return algorithm_map


def train_klr_policy(
    env_id: str,
    k: int,
    best_response: bool,
    seed: Optional[int],
    algorithm_config: AlgorithmConfig,
    num_gpus: float,
    num_iterations: int,
    run_serially: bool = False,
    save_policies: bool = True,
    verbose: bool = True,
):
    """Run training of KLR policy."""
    assert "env_config" in algorithm_config
    if "env_id" not in algorithm_config.env_config:
        algorithm_config.env_config["env_id"] = env_id

    ray.init()
    register_env(env_id, posggym_registered_env_creator)
    env = posggym_registered_env_creator(algorithm_config.env_config)

    agent_ids = list(env.get_agent_ids())
    agent_ids.sort()

    if best_response:
        igraph = pbt.construct_klrbr_interaction_graph(
            agent_ids,
            k,
            is_symmetric=env.env.model.is_symmetric,
            dist=None,  # uses poisson with lambda=1.0
            seed=seed,
        )
    else:
        igraph = pbt.construct_klr_interaction_graph(
            agent_ids, k, is_symmetric=env.env.model.is_symmetric, seed=seed
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
        "run_serially": run_serially,
    }
    if igraph.is_symmetric:
        algorithm_map = get_symmetric_klr_algorithm(**algorithm_kwargs)
    else:
        algorithm_map = get_asymmetric_klr_algorithm(**algorithm_kwargs)

    run_training(algorithm_map, igraph, num_iterations, verbose=verbose)

    if save_policies:
        print("== Exporting Graph ==")
        parent_dir = os.path.join(BASE_RESULTS_DIR, env_id, "policies")
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        save_dir = f"train_klr_{env_id}_seed{seed}_k{k}"
        if best_response:
            save_dir += "_br"
        export_dir = export_igraph_algorithms(
            parent_dir,
            igraph,
            algorithm_map,
            remote=not run_serially,
            save_dir_name=save_dir,
        )
        print(f"{export_dir=}")
