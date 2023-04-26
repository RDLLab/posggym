"""Code for running training using rllib."""
import os.path as osp
from typing import Any, Dict, Optional, Set, Type

import ray
from ray.air.integrations.wandb import setup_wandb
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.logger import pretty_print

from posggym.agents.train import pbt
from posggym.agents.train.rllib.algorithm import CustomPPOAlgorithm
from posggym.agents.train.rllib.export_lib import export_igraph_algorithms
from posggym.agents.train.rllib.import_lib import import_igraph_algorithms
from posggym.agents.train.rllib.utils import RllibAlgorithmMap


def sync_policies(algorithms: RllibAlgorithmMap, igraph: pbt.InteractionGraph):
    """Sync the policies between algorithms based on interactiong graph."""
    agent_ids = list(algorithms)
    agent_ids.sort()

    for i, policy_map in algorithms.items():
        for policy_k_id, algorithm_k in policy_map.items():
            if isinstance(algorithm_k, ray.actor.ActorHandle):
                weights = algorithm_k.get_weights.remote([policy_k_id])
            else:
                weights = algorithm_k.get_weights([policy_k_id])
            igraph.update_policy(i, policy_k_id, weights)

    # swap weights of other agent policies
    for i, policy_map in algorithms.items():
        for policy_k_id, algorithm_k in policy_map.items():
            for j in agent_ids:
                other_agent_policies = igraph.get_all_policies(i, policy_k_id, j)
                # Notes weights here is a dict from policy id to weights
                # ref:
                # https://docs.ray.io/en/releases-2.3.0/_modules/ray/rllib/algorithms/algorithm.html#Algorithm.get_weights
                for policy_j_id, weights in other_agent_policies:
                    if isinstance(algorithm_k, ray.actor.ActorHandle):
                        algorithm_k.set_weights.remote(weights)
                        algorithm_k.sync_weights.remote()
                    else:
                        algorithm_k.set_weights(weights)
                        algorithm_k.sync_weights()  # type: ignore


def run_training(
    algorithms: RllibAlgorithmMap,
    igraph: pbt.InteractionGraph,
    num_iterations: int,
    verbose: bool = True,
    wandb_config: Optional[Dict[str, Any]] = None,
):
    """Train Rllib training iterations.

    algorithms is Dict[AgentID, Dict[PolicyID, Algorithm]]
    """
    agent_ids = list(algorithms)
    agent_ids.sort()

    wandb = setup_wandb(**wandb_config) if wandb_config else None

    for iteration in range(num_iterations):
        if verbose:
            print(f"== Iteration {iteration} ==")

        sync_policies(algorithms, igraph)

        result_futures: Dict = {i: {} for i in agent_ids}
        for i, policy_map in algorithms.items():
            for policy_k_id, algorithm_k in policy_map.items():
                if isinstance(algorithm_k, ray.actor.ActorHandle):
                    result_future = algorithm_k.train.remote()  # type: ignore
                else:
                    result_future = algorithm_k.train()
                result_futures[i][policy_k_id] = result_future

        results: Dict = {i: {} for i in agent_ids}
        for i, policy_map in result_futures.items():
            for policy_k_id, future in result_futures[i].items():
                if isinstance(future, ray.ObjectRef):
                    result = ray.get(future)
                else:
                    result = future
                results[i][policy_k_id] = result

        for i, policy_map in results.items():
            for policy_k_id, result_k in policy_map.items():
                if verbose:
                    print(f"-- Agent ID {i}, Policy {policy_k_id} --")
                    print(pretty_print(result_k))
                if wandb:
                    log_to_wandb(wandb, result_k)


def log_to_wandb(wandb, result, exclude: Optional[Set[str]] = None):
    result = result.copy()
    result.update(config=None)  # drop config from pretty print
    result.update(hist_stats=None)  # drop hist_stats from pretty print
    out = {}
    for k, v in result.items():
        if v is not None and (exclude is None or k not in exclude):
            out[k] = v
    wandb.log(out)


def run_evaluation(algorithms: RllibAlgorithmMap, verbose: bool = True):
    """Run evaluation for policy algorithms."""
    if verbose:
        print("== Running Evaluation ==")
    results: Dict = {i: {} for i in algorithms}
    for i, policy_map in algorithms.items():
        results[i] = {}
        for policy_k_id, algorithm in policy_map.items():
            if verbose:
                print(f"-- Running Agent ID {i}, Policy {policy_k_id} --")
            results[i][policy_k_id] = algorithm.evaluate()

    if verbose:
        print("== Evaluation results ==")
        for i, policy_map in results.items():
            for policy_k_id, result in policy_map.items():
                print(f"-- Agent ID {i}, Policy {policy_k_id} --")
                print(pretty_print(result))

    return results


def continue_training(
    policy_dir,
    remote: bool,
    num_iterations: int,
    seed: Optional[int],
    algorithm_class: Optional[Type[Algorithm]] = None,
    save_policies: bool = True,
    verbose: bool = True,
):
    """Continue training of saved policies.

    Assumes:
    1. ray has been initialized
    2. training environment has been registered with ray
    """
    if algorithm_class is None:
        algorithm_class = CustomPPOAlgorithm

    igraph, algorithms = import_igraph_algorithms(
        igraph_dir=policy_dir,
        algorithm_class=algorithm_class,
        remote=remote,
        seed=seed,
        logger_creator=None,
    )
    igraph.display()

    run_training(algorithms, igraph, num_iterations, verbose=verbose)

    if save_policies:
        print("== Exporting Graph ==")
        # use same save dir name but with new checkpoint number
        policy_dir_name = osp.basename(osp.normpath(policy_dir))
        name_tokens = policy_dir_name.split("_")[-1]
        if "checkpoint" in name_tokens[-1]:
            try:
                checkpoint_num = int(name_tokens[-1].replace("checkpoint", ""))
                checkpoint = f"checkpoint{checkpoint_num+1}"
            except ValueError:
                checkpoint = name_tokens[-1] + "1"
            name_tokens = name_tokens[:-1]
        else:
            checkpoint = "checkpoint1"
        name_tokens.append(checkpoint)
        save_dir = "_".join(name_tokens)

        export_dir = export_igraph_algorithms(
            osp.dirname(osp.normpath(policy_dir)),
            igraph,
            algorithms,
            remote=remote,
            save_dir_name=save_dir,
        )
        print(f"{export_dir=}")
