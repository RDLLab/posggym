"""Interaction Graph for Population Based Training."""
import itertools
import json
import os
import random
from pprint import pprint
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from posggym.agents.policy import Policy, PolicyID
from posggym.model import AgentID


PolicyState = Union[Policy, Dict]

# A function which takes the AgentID, PolicyID, PolicyState Object, and export
# directory and exports a policy to a local directory
PolicyExportFn = Callable[[AgentID, PolicyID, PolicyState, str], None]

# A function which takes the AgentID, PolicyID and import file path and imports
# a policy from the file
PolicyImportFn = Callable[[AgentID, PolicyID, str], PolicyState]


AgentPolicyDist = Dict[AgentID, Dict[PolicyID, float]]
AgentPolicyMap = Dict[AgentID, Dict[PolicyID, PolicyState]]


IGRAPH_FILE_NAME = "igraph.json"
IGRAPH_AGENT_ID_FILE_NAME = "igraph_agents.json"


class InteractionGraph:
    """Interaction Graph for Population Based Training.

    If symmetric policies with a given ID are shared between agents, e.g. the
    policy with 'pi_0' will be the same policy for every agent in the
    environment.

    Otherwise, each agent has a separate policies, e.g. the policy with ID
    'pi_0' will correspond to a different policy for each agent.

    The graph stores policy state, which could be an actual policy or some other part
    of a policy (e.g. policy weights). It's up to the code using the graph to handle
    each case.

    """

    # Agent ID used in case of symmetric interaction graph
    # Need to use str for correct import/export to/from json
    SYMMETRIC_ID = str(None)

    def __init__(self, symmetric: bool, seed: Optional[int]):
        self._symmetric = symmetric
        self._rng = random.Random(seed)
        self._agent_ids: Set[AgentID] = set()
        # maps (agent_id, policy_id, other_agent_id) -> Delta(policy_id))
        self._graph: Dict[AgentID, Dict[PolicyID, AgentPolicyDist]] = {}
        # maps (agent_id, policy_id) -> PolicyState
        self._policies: AgentPolicyMap = {}

    @property
    def policies(self) -> AgentPolicyMap:
        """Get the set of policies for each agent in the graph."""
        return self._policies

    @property
    def graph(self) -> Dict[AgentID, Dict[PolicyID, AgentPolicyDist]]:
        """Get the graph."""
        return self._graph

    @property
    def is_symmetric(self) -> bool:
        """Get if interactive graph is symmetric or not."""
        return self._symmetric

    def get_agent_ids(self) -> List[AgentID]:
        """Get list of agent ids stored in graph."""
        return list(self._agent_ids)

    def get_agent_policy_ids(self, agent_id: AgentID) -> List[PolicyID]:
        """Get list of all policy ids for a given agent."""
        if self._symmetric:
            agent_id = self.SYMMETRIC_ID
        return list(self._policies[agent_id])

    def add_policy(
        self, agent_id: AgentID, policy_id: PolicyID, policy: PolicyState
    ) -> None:
        """Add a policy to the interaction graph.

        Note, for symmetric environments agent id is treated as None.
        """
        self._agent_ids.add(agent_id)
        if self._symmetric:
            agent_id = self.SYMMETRIC_ID

        if agent_id not in self._policies:
            self._policies[agent_id] = {}
        self._policies[agent_id][policy_id] = policy

        if agent_id not in self._graph:
            self._graph[agent_id] = {}
        self._graph[agent_id][policy_id] = {}

    def add_edge(
        self,
        src_agent_id: AgentID,
        src_policy_id: PolicyID,
        dest_agent_id: AgentID,
        dest_policy_id: PolicyID,
        weight: float,
    ) -> None:
        """Add a directed edge between policies on the graph.

        Updates edge weight if an edge already exists between src and dest
        policies.
        """
        if self._symmetric:
            src_agent_id = self.SYMMETRIC_ID
            dest_agent_id = self.SYMMETRIC_ID

        assert (
            src_agent_id in self._policies
        ), f"Source agent with ID={src_agent_id} not in graph."
        assert (
            src_policy_id in self._policies[src_agent_id]
        ), f"Source policy with ID={src_policy_id} not in graph."
        assert (
            dest_agent_id in self._policies
        ), f"Destination agent with ID={dest_agent_id} not in graph."
        assert (
            dest_policy_id in self._policies[dest_agent_id]
        ), f"Destination policy with ID={dest_policy_id} not in graph."
        assert weight >= 0, f"Edge weight={weight} must be non-negative."

        if dest_agent_id not in self._graph[src_agent_id][src_policy_id]:
            self._graph[src_agent_id][src_policy_id][dest_agent_id] = {}

        dest_dist = self._graph[src_agent_id][src_policy_id][dest_agent_id]
        dest_dist[dest_policy_id] = weight

    def has_edge(
        self,
        src_agent_id: AgentID,
        src_policy_id: PolicyID,
        dest_agent_id: AgentID,
        dest_policy_id: PolicyID,
    ) -> bool:
        """Check if an edge exists."""
        try:
            src_policy_map = self._graph[src_agent_id][src_policy_id]
            src_policy_map[dest_agent_id][dest_policy_id]
        except KeyError:
            return False
        return True

    def fully_connect_graph(self):
        """Add an edge between all pairs of different agent policies.

        This is a simple helper function for graph construction. Does not add
        a new edge if an edge already exists.
        """
        if self._symmetric:
            agent_pairings = [(self.SYMMETRIC_ID, self.SYMMETRIC_ID)]
        else:
            agent_pairings = itertools.permutations(self._policies, r=2)

        for src_agent_id, dest_agent_id in agent_pairings:
            for src_policy_id in self._policies[src_agent_id]:
                for dest_policy_id in self._policies[dest_agent_id]:
                    if self.has_edge(
                        src_agent_id, src_policy_id, dest_agent_id, dest_policy_id
                    ):
                        continue
                    self.add_edge(
                        src_agent_id, src_policy_id, dest_agent_id, dest_policy_id, 1
                    )

    def get_agent_outgoing_policies(self, agent_id: AgentID) -> List[PolicyID]:
        """Get list of IDs of outgoing policies for an agent.

        Outgoing policies have at least one edge going to another policy,
        including themselves. These are the policies whose behaviour/evolution
        is dependent will change as policies interact.

        Policies with no outgoing edges are assumed to be fixed.
        """
        if self._symmetric:
            agent_id = self.SYMMETRIC_ID

        outgoing_policy_ids = []
        for policy_id in self._graph[agent_id]:
            if len(self._graph[agent_id][policy_id]) > 0:
                outgoing_policy_ids.append(policy_id)

        return outgoing_policy_ids

    def get_outgoing_policies(self) -> Dict[AgentID, List[PolicyID]]:
        """Get list of IDs of outgoing policies for each agent.

        Outgoing policies have at least one edge going to another policy,
        including themselves. These are the policies whose behaviour/evolution
        is dependent will change as policies interact.

        Policies with no outgoing edges are assumed to be fixed.
        """
        if self._symmetric:
            policies = self.get_agent_outgoing_policies(self.SYMMETRIC_ID)
            return {self.SYMMETRIC_ID: policies}

        return {i: self.get_agent_outgoing_policies(i) for i in self.get_agent_ids()}

    def update_policy(
        self, agent_id: AgentID, policy_id: PolicyID, new_policy: PolicyState
    ) -> None:
        """Update stored policy."""
        if self._symmetric:
            agent_id = self.SYMMETRIC_ID

        assert agent_id in self._policies, (
            f"Agent with ID={agent_id} not in graph. Make sure to add the "
            "agent using the add_policy function, before updating."
        )
        assert policy_id in self._policies[agent_id], (
            f"Policy with ID={policy_id} not in graph. Make sure to add the "
            "policy using the add_policy() function, before updating."
        )
        self._policies[agent_id][policy_id] = new_policy

    def sample_policy(
        self, agent_id: AgentID, policy_id: PolicyID, other_agent_id: AgentID
    ) -> Tuple[PolicyID, PolicyState]:
        """Sample an other agent policy from the graph for given policy_id.

        Returns the sampled policy id and the sampled policy.
        """
        if self._symmetric:
            agent_id = self.SYMMETRIC_ID
            other_agent_id = self.SYMMETRIC_ID

        assert agent_id in self._policies, (
            f"Agent with ID={agent_id} not in graph. Make sure to add the "
            "agent using the add_policy function, before sampling."
        )
        assert other_agent_id in self._policies, (
            f"Other agent with ID={agent_id} not in graph. Make sure to add "
            "the agent using the add_policy function, before sampling."
        )
        assert policy_id in self._policies[agent_id], (
            f"Policy with ID={policy_id} not in graph. Make sure to add the "
            "policy using the add_policy() function, before sampling."
        )
        assert len(self._graph[agent_id][policy_id][other_agent_id]) > 0, (
            f"No edges added from policy with ID={policy_id}. Make sure to "
            "add edges from the policy using the add_edge() function, before "
            "sampling."
        )
        other_policy_dist = self._graph[agent_id][policy_id][other_agent_id]
        other_policy_ids = list(other_policy_dist)
        other_policy_weights = list(other_policy_dist.values())

        sampled_policy_id = self._rng.choices(
            other_policy_ids, weights=other_policy_weights, k=1
        )[0]
        sampled_policy = self._policies[other_agent_id][sampled_policy_id]
        return sampled_policy_id, sampled_policy

    def sample_policies(
        self, agent_id: AgentID, policy_id: PolicyID
    ) -> Dict[AgentID, Tuple[PolicyID, PolicyState]]:
        """Sample a policy for each other agent from the graph.

        Policies are sampled from policies connected to the given
        (agent_id, policy_id).
        """
        other_policies: Dict[AgentID, Tuple[PolicyID, PolicyState]] = {}
        for other_agent_id in self._graph[agent_id][policy_id]:
            other_policies[other_agent_id] = self.sample_policy(
                agent_id, policy_id, other_agent_id
            )
        return other_policies

    def get_all_policies(
        self,
        agent_id: AgentID,
        policy_id: PolicyID,
        other_agent_id: AgentID,
    ) -> List[Tuple[PolicyID, PolicyState]]:
        """Get all connected policies for other agent from the graph."""
        if self._symmetric:
            agent_id = self.SYMMETRIC_ID
            other_agent_id = self.SYMMETRIC_ID

        assert agent_id in self._policies, (
            f"Agent with ID={agent_id} not in graph. Make sure to add the "
            "agent using the add_policy function."
        )
        assert other_agent_id in self._policies, (
            f"Other agent with ID={agent_id} not in graph. Make sure to add "
            "the agent using the add_policy function."
        )
        assert policy_id in self._policies[agent_id], (
            f"Policy with ID={policy_id} not in graph. Make sure to add the "
            "policy using the add_policy() function."
        )
        if len(self._graph[agent_id][policy_id]) == 0:
            return []

        if not self._symmetric and agent_id == other_agent_id:
            return []

        other_policy_dist = self._graph[agent_id][policy_id][other_agent_id]
        other_policy_ids = list(other_policy_dist)
        other_policies = []
        for other_policy_id in other_policy_ids:
            other_policies.append(
                (other_policy_id, self._policies[other_agent_id][other_policy_id])
            )
        return other_policies

    @staticmethod
    def export_graph(
        igraph: "InteractionGraph", export_dir: str, policy_export_fn: PolicyExportFn
    ):
        """Export Interaction Graph to a local directory."""
        igraph_file = os.path.join(export_dir, IGRAPH_FILE_NAME)
        igraph_output = {
            "graph": igraph.graph,
            "agent_ids": igraph.get_agent_ids(),
            "symmetric": igraph.is_symmetric,
        }
        with open(igraph_file, "w", encoding="utf-8") as fout:
            json.dump(igraph_output, fout)

        for agent_id, policy_map in igraph.policies.items():
            agent_dir = os.path.join(export_dir, str(agent_id))
            os.mkdir(agent_dir)

            for policy_id, policy in policy_map.items():
                policy_dir = os.path.join(agent_dir, str(policy_id))
                os.mkdir(policy_dir)
                policy_export_fn(agent_id, policy_id, policy, policy_dir)

    @staticmethod
    def import_graph(
        import_dir: str, policy_import_fn: PolicyImportFn, seed: Optional[int] = None
    ) -> "InteractionGraph":
        """Import interaction graph from a local directory.

        Note, this assumes import directory was generated using the
        InteractionGraph.export_graph function.
        """
        igraph_file = os.path.join(import_dir, IGRAPH_FILE_NAME)
        with open(igraph_file, "r", encoding="utf-8") as fin:
            igraph_data = json.load(fin)

        igraph = InteractionGraph(igraph_data["symmetric"], seed)
        igraph._graph = igraph_data["graph"]
        igraph._agent_ids = set(igraph_data["agent_ids"])

        policies: Dict[AgentID, Dict[PolicyID, PolicyState]] = {}
        for agent_id, policy_map in igraph.graph.items():
            agent_dir = os.path.join(import_dir, str(agent_id))
            policies[agent_id] = {}

            for policy_id in policy_map:
                policy_dir = os.path.join(agent_dir, str(policy_id))
                policies[agent_id][policy_id] = policy_import_fn(
                    agent_id, policy_id, policy_dir
                )
        igraph._policies = policies
        return igraph

    def display(self):
        """Display the graph in human readable format."""
        print("Interaction Graph:")
        print(f"Symmetric={self._symmetric}")
        pprint(self._graph)
