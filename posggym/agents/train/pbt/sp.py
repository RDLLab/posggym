"""Self-Play Population Based Training."""
from itertools import product
from typing import List, Optional

from posggym.agents.train.pbt.interaction_graph import InteractionGraph
from posggym.agents.train.pbt.utils import get_policy_id
from posggym.model import AgentID


def get_sp_policy_id(agent_id: Optional[AgentID], is_symmetric: bool) -> str:
    """Get the policy ID string for a self-play policy."""
    return get_policy_id(agent_id, "SP", is_symmetric)


def construct_sp_interaction_graph(
    agent_ids: List[AgentID], is_symmetric: bool, seed: Optional[int] = None
) -> InteractionGraph:
    """Construct a Self-Play Interaction Graph.

    Note that this function constructs the graph and edges between policy IDs,
    but the actual policies still need to be added.
    """
    igraph = InteractionGraph(is_symmetric, seed)

    for agent_id in agent_ids:
        policy_id = get_sp_policy_id(agent_id, is_symmetric)
        igraph.add_policy(agent_id, policy_id, {})

    for src_agent_id, dest_agent_id in product(agent_ids, agent_ids):
        if not is_symmetric and src_agent_id == dest_agent_id:
            continue

        src_policy_id = get_sp_policy_id(src_agent_id, is_symmetric)
        dest_policy_id = get_sp_policy_id(dest_agent_id, is_symmetric)
        igraph.add_edge(src_agent_id, src_policy_id, dest_agent_id, dest_policy_id, 1.0)

    return igraph
