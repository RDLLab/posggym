"""Functions for exporting a rllib algorithm to a file."""
import os
import pickle
import tempfile
from datetime import datetime

import ray

from posggym.agents.train import pbt
from posggym.agents.train.rllib.utils import RllibAlgorithmMap


ALGORITHM_CONFIG_FILE = "algorithm_config.pkl"


def get_algorithm_export_fn(
    algorithm_map: RllibAlgorithmMap, remote: bool
) -> pbt.PolicyExportFn:
    """Get function for exporting trained policies to local directory."""

    def export_fn(
        agent_id: str, policy_id: str, policy: pbt.PolicyState, export_dir: str
    ):
        # use str types to prevent importing issues wrt AgentID and PolicyID
        if policy_id not in algorithm_map[agent_id]:
            # untrained policy, e.g. a random policy
            return

        algorithm = algorithm_map[agent_id][policy_id]

        if not remote:
            algorithm.set_weights(policy)
            algorithm.save(export_dir)
            config = algorithm.config  # type: ignore
        else:
            algorithm.set_weights.remote(policy)
            ray.get(algorithm.save.remote(export_dir))  # type: ignore
            config = ray.get(algorithm.get_config.remote())  # type: ignore

        # export algorithm config
        config_path = os.path.join(export_dir, ALGORITHM_CONFIG_FILE)
        with open(config_path, "wb") as fout:
            pickle.dump(config, fout)

    return export_fn


def export_igraph_algorithms(
    parent_dir: str,
    igraph: pbt.InteractionGraph,
    algorithms: RllibAlgorithmMap,
    remote: bool,
    save_dir_name: str = "",
) -> str:
    """Export Rllib algorithm objects to file.

    Handles creation of directory to store
    """
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    export_dir_name = f"{save_dir_name}_{timestr}"
    export_dir = tempfile.mkdtemp(prefix=export_dir_name, dir=parent_dir)
    pbt.InteractionGraph.export_graph(
        igraph,
        export_dir,
        get_algorithm_export_fn(algorithms, remote),
    )
    return export_dir
