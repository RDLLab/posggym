"""Shortest path policy for Driving Gen envs."""
from __future__ import annotations

from typing import TYPE_CHECKING

from posggym.agents.grid_world.driving.shortest_path import DrivingShortestPathPolicy


if TYPE_CHECKING:
    from posggym.agents.policy import PolicyID
    from posggym.model import POSGModel


class DrivingGenShortestPathPolicy(DrivingShortestPathPolicy):
    """Shortest Path Policy for the Driving Gen environment.

    This policy sets the preferred action as the one which is on the shortest
    path to the agent's goal and which doesn't leave agent in same position. Note,
    the policy is stochastic and will sample from a distribution over the preferred
    actions, since there may be multiple actions which are on the shortest path.

    This policy is the same as the shortest path policy for the
    [Driving](/environments/grid_world/driving) environment, except that it recomputes
    the shortest paths when reset (at the start of each episode), since the environment
    grid layout is regenerated at the start of each episode. Additionally, it only
    computes the shortest path for the agent's current destination, rather than for
    all destinations.

    Arguments
    ---------
    aggressiveness : float
        The aggressiveness of the policy towards other vehicles. A value of 0.0 means
        the policy will always stop when it sees another vehicle, while a value of 1.0
        means the policy will ignore other vehicles and always take the shortest path
        action. Values in between will change how far away another vehicle needs to
        be before the policy will stop. Default is 1.0.

    """

    def __init__(
        self,
        model: POSGModel,
        agent_id: str,
        policy_id: PolicyID,
        aggressiveness: float = 1.0,
    ):
        super().__init__(
            model,
            agent_id,
            policy_id,
            aggressiveness=aggressiveness,
            precompute_shortest_paths=False,
        )

    def reset(self, *, seed: int | None = None):
        super().reset(seed=seed)
        self._grid = self.model.grid
        self.shortest_paths = {}
