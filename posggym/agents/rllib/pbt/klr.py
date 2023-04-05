import math
from itertools import product
from typing import Callable, List, Optional, Tuple

from posggym.agents.rllib.pbt.interaction_graph import InteractionGraph
from posggym.agents.rllib.pbt.utils import get_policy_id
from posggym.model import AgentID


def get_klr_policy_id(agent_id: Optional[AgentID], k: int, is_symmetric: bool) -> str:
    """Get the policy ID string for a K-level reasoning policy."""
    return get_policy_id(agent_id, str(k), is_symmetric)


def get_br_policy_id(agent_id: Optional[AgentID], is_symmetric: bool) -> str:
    """Get the policy ID string for a Best Response policy."""
    return get_policy_id(agent_id, "BR", is_symmetric)


def get_klr_poisson_prob(k: int, num_levels: int, lmbda: float = 1.0) -> float:
    """Get Poisson probability for level k agent in hierarchy with num_levels.

    lmbda is the Possion Distribution parameter and is the mean of the
    distribution.
    In this function higher values shift the probability mass towards deeper
    nesting levels.
    """
    i = num_levels - k
    return (lmbda**i * math.exp(-lmbda)) / math.factorial(i)


def get_klr_poisson_prob_fn(
    num_levels: int, lmbda: float = 1.0
) -> Callable[[int], float]:
    """Get Poisson probability function mapping level k to probability."""

    def poisson_fn(k: int):
        return get_klr_poisson_prob(k, num_levels, lmbda=lmbda)

    return poisson_fn


def parse_klr_policy_id(policy_id: str) -> Tuple[Optional[AgentID], int]:
    """Parse KLR policy ID string to get reasoning level.

    Also get optional agent ID (for non-symmetric environments)
    """
    tokens = policy_id.split("_")
    if len(tokens) == 2:
        return None, int(tokens[1])

    if len(tokens) == 3:
        return tokens[2], int(tokens[1])

    raise ValueError(f"Invalid KLR Policy ID str '{policy_id}'")


def construct_klr_interaction_graph(
    agent_ids: List[AgentID],
    k_levels: int,
    is_symmetric: bool,
    seed: Optional[int] = None,
) -> InteractionGraph:
    """Construct a K-Level Reasoning Interaction Graph.

    Note that this function constructs the graph and edges between policy IDs,
    but the actual policies still need to be added.
    """
    igraph = InteractionGraph(is_symmetric, seed)

    for agent_id, k in product(agent_ids, range(-1, k_levels + 1)):
        policy_id = get_klr_policy_id(agent_id, k, is_symmetric)
        igraph.add_policy(agent_id, policy_id, {})

    for src_agent_id, k, dest_agent_id in product(
        agent_ids, range(0, k_levels + 1), agent_ids
    ):
        if not is_symmetric and src_agent_id == dest_agent_id:
            continue

        src_policy_id = get_klr_policy_id(src_agent_id, k, is_symmetric)
        dest_policy_id = get_klr_policy_id(dest_agent_id, k - 1, is_symmetric)
        igraph.add_edge(src_agent_id, src_policy_id, dest_agent_id, dest_policy_id, 1.0)

    return igraph


def construct_klrbr_interaction_graph(
    agent_ids: List[AgentID],
    k_levels: int,
    is_symmetric: bool,
    dist: Optional[Callable[[int], float]],
    seed: Optional[int] = None,
) -> InteractionGraph:
    """Construct a K-Level Reasoning Interaction Graph with Best Response.

    Note that this function constructs the graph and edges between policy IDs,
    but the actual policies still need to be added.
    """
    if dist is None:
        dist = get_klr_poisson_prob_fn(k_levels)

    igraph = construct_klr_interaction_graph(agent_ids, k_levels, is_symmetric, seed)

    for agent_id in agent_ids:
        policy_br_id = get_br_policy_id(agent_id, is_symmetric)
        igraph.add_policy(agent_id, policy_br_id, {})

    for agent_br_id, agent_k_id in product(agent_ids, agent_ids):
        if not is_symmetric and agent_br_id == agent_k_id:
            continue

        policy_br_id = get_br_policy_id(agent_br_id, is_symmetric)

        policies_k_dist = []
        policies_k_ids = []
        for k in range(-1, k_levels + 1):
            policy_k_id = get_klr_policy_id(agent_k_id, k, is_symmetric)
            policies_k_dist.append(dist(k))
            policies_k_ids.append(policy_k_id)

        unnormalized_prob_sum = sum(policies_k_dist)
        for i in range(len(policies_k_dist)):
            policies_k_dist[i] /= unnormalized_prob_sum

        for policy_id, policy_prob in zip(policies_k_ids, policies_k_dist):
            igraph.add_edge(
                agent_br_id, policy_br_id, agent_k_id, policy_id, policy_prob
            )

    return igraph
