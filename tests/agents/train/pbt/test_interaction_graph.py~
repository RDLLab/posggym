import json
import os
import tempfile
from unittest import TestCase

import pytest

from posggym.agents.rllib import pbt


@pytest.mark.parametrize("symmetric", [True, False])
def test_export_import(symmetric):
    policy_file_name = "pi.json"
    k = 2
    agent_ids = ["0", "1"]
    igraph = pbt.construct_klr_interaction_graph(agent_ids, k, symmetric)

    def export_fn(agent_id, policy_id, policy, policy_export_dir):
        file_name = os.path.join(policy_export_dir, policy_file_name)
        with open(file_name, "w", encoding="utf-8") as fout:
            json.dump(policy, fout)

    def import_fn(agent_id, policy_id, policy_import_dir):
        file_name = os.path.join(policy_import_dir, policy_file_name)
        with open(file_name, "r", encoding="utf-8") as fin:
            policy = json.load(fin)
        return policy

    export_dir = tempfile.mkdtemp()
    pbt.InteractionGraph.export_graph(igraph, export_dir, export_fn)

    new_igraph = pbt.InteractionGraph.import_graph(export_dir, import_fn, seed=None)

    TestCase().assertDictEqual(igraph.policies, new_igraph.policies)
    TestCase().assertDictEqual(igraph.graph, new_igraph.graph)
