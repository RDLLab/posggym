"""Heuristic policies for the PredatorPrey grid world environment."""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple, cast

import posggym.envs.grid_world.predator_prey as pp
from posggym.agents.policy import Policy, PolicyID, PolicyState
from posggym.agents.utils import action_distributions
from posggym.envs.grid_world.core import Direction
from posggym.utils import seeding

if TYPE_CHECKING:
    from posggym.envs.grid_world.core import Coord
    from posggym.model import POSGModel
    from posggym.utils.history import AgentHistory


DIR_TO_ACTION = [pp.UP, pp.RIGHT, pp.DOWN, pp.LEFT]


class PPHeuristicPolicy(Policy[pp.PPAction, pp.PPObs]):
    """Base class for PredatorPrey environment heuristic policies."""

    VALID_EXPLORE_STRATEGIES = [
        "uniform_random",
        "spiral",
    ]

    def __init__(
        self,
        model: POSGModel,
        agent_id: str,
        policy_id: PolicyID,
        explore_strategy: str = "uniform_random",
        explore_epsilon: float = 0.05,
    ):
        super().__init__(model, agent_id, policy_id)
        assert explore_strategy in self.VALID_EXPLORE_STRATEGIES
        assert 0 <= explore_epsilon <= 1
        self.explore_strategy = explore_strategy
        self.explore_epsilon = explore_epsilon
        self.model = cast(pp.PredatorPreyModel, model)
        self._grid = self.model.grid
        self._action_space = list(range(self.model.action_spaces[agent_id].n))
        self._rng, _ = seeding.std_random()

        # obs dim is # cells in each direction from agent
        self.obs_dim = self.model.obs_dim
        self.obs_width = (2 * self.obs_dim) + 1
        self.agent_obs_idx = (self.obs_dim * self.obs_width) + self.obs_dim
        self.agent_obs_coord = (self.obs_dim, self.obs_dim)

    def reset(self, *, seed: int | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng, _ = seeding.std_random(seed=seed)

    def get_initial_state(self) -> PolicyState:
        state = super().get_initial_state()
        state.update({"pi": None, "last_obs": None, "last_explore_dir": None})
        return state

    def get_next_state(
        self,
        action: pp.PPAction | None,
        obs: pp.PPObs,
        state: PolicyState,
    ) -> PolicyState:
        pred_coords, prey_coords, wall_obs = self.parse_obs(obs)
        actions = self.get_actions_from_obs(pred_coords, prey_coords)

        explore_dir = state["last_explore_dir"]
        if len(actions) == 0:
            actions, explore_dir = self.get_explore_actions_from_obs(
                wall_obs, explore_dir
            )

        action_probs = {}
        for a in self._action_space:
            if a in actions:
                prob = (1 - self.explore_epsilon) / len(actions)
            else:
                prob = self.explore_epsilon / (len(self._action_space) - len(actions))
            action_probs[a] = prob
        pi = action_distributions.DiscreteActionDistribution(action_probs, self._rng)

        return {
            "pi": pi,
            "last_obs": obs,
            "last_explore_dir": explore_dir,
        }

    def get_state_from_history(self, history: AgentHistory) -> PolicyState:
        _, last_obs = history.get_last_step()
        if last_obs is not None:
            return self.get_next_state(None, last_obs, self.get_initial_state())
        return self.get_initial_state()

    def sample_action(self, state: PolicyState) -> pp.PPAction:
        return self.get_pi(state).sample()

    def get_pi(self, state: PolicyState) -> action_distributions.ActionDistribution:
        if state["pi"] is None:
            raise ValueError(
                "Policy state does not contain a valid action distribution. Make sure"
                "to call `step` or `get_next_state` before calling `get_pi`"
            )
        return state["pi"]

    def get_value(self, state: PolicyState) -> float:
        raise NotImplementedError(
            f"`get_value()` no implemented by {self.__class__.__name__} policy"
        )

    def get_actions_from_obs(
        self, pred_coords: List[Coord], prey_coords: List[Coord]
    ) -> List[pp.PPAction]:
        raise NotImplementedError(
            f"`get_pi_from_obs()` not implemented by {self.__class__.__name__} policy"
        )

    def get_explore_actions_from_obs(
        self, wall_obs: List[bool], explore_dir: Direction | None
    ) -> Tuple[List[pp.PPAction], Direction | None]:
        # using list slice for quick copy of list of primitives
        if self.explore_strategy == "uniform_random":
            # random explore
            return pp.ACTIONS[:], None

        if self.explore_strategy == "spiral":
            # N->E E->S S->W  W->N (i.e clockwise, +1 mod 4)
            if explore_dir is None:
                explore_dir = (
                    Direction.EAST if wall_obs[Direction.SOUTH] else Direction.WEST
                )
            for _ in range(4):
                if not wall_obs[explore_dir]:
                    return [DIR_TO_ACTION[explore_dir]], explore_dir
                explore_dir = (1 + explore_dir) % 4

            # walls observed in all directions, just move in any direction
            return pp.ACTIONS[:], None

        raise NotImplementedError(
            f"Explore strategy `{self.explore_strategy}` not implemented by "
            f"{self.__class__.__name__} policy"
        )

    def parse_obs(self, obs: pp.PPObs) -> Tuple[List[Coord], List[Coord], List[bool]]:
        """Parse obs into list of predator coords, prey coords, and wall directions."""
        pred_coords = []
        prey_coords = []
        walls_obs = [False] * 4
        for idx, cell_obs in enumerate(obs):
            if idx == self.agent_obs_idx:
                continue
            col, row = idx % self.obs_width, idx // self.obs_width
            if cell_obs == pp.PREDATOR:
                pred_coords.append((col, row))
            elif cell_obs == pp.PREY:
                prey_coords.append((col, row))
            elif cell_obs == pp.WALL and row == self.agent_obs_coord[1]:
                if col < self.agent_obs_coord[0]:
                    walls_obs[Direction.WEST] = True
                elif col > self.agent_obs_coord[0]:
                    walls_obs[Direction.EAST] = True
            elif cell_obs == pp.WALL and col == self.agent_obs_coord[0]:
                if row < self.agent_obs_coord[1]:
                    walls_obs[Direction.NORTH] = True
                elif row > self.agent_obs_coord[1]:
                    walls_obs[Direction.SOUTH] = True
        return pred_coords, prey_coords, walls_obs

    def get_closest_coord(self, origin: Coord, coords: List[Coord]) -> Coord | None:
        """Get coord of from list that is closest to the origin coord."""
        return min(
            coords,
            key=lambda coord: self._grid.manhattan_dist(origin, coord),
        )

    def get_actions_towards_target(self, target_coord: Coord) -> List[pp.PPAction]:
        """Get action towards target coord."""
        agent_coord = self.agent_obs_coord

        # Note positioning is relative to observing agents observation grid, not the
        # global grid. So relative directions are different.
        actions = []
        if agent_coord[0] < target_coord[0]:
            actions.append(pp.RIGHT)
        elif agent_coord[0] > target_coord[0]:
            actions.append(pp.LEFT)

        if agent_coord[1] < target_coord[1]:
            actions.append(pp.DOWN)
        elif agent_coord[1] > target_coord[1]:
            actions.append(pp.UP)

        return actions


class PPHeuristic1(PPHeuristicPolicy):
    """H1 moves towards closest observed prey, closest observed predator, or explores
    randomly, in that order.
    """

    def __init__(self, model: POSGModel, agent_id: str, policy_id: PolicyID):
        super().__init__(model, agent_id, policy_id, "uniform_random")

    def get_actions_from_obs(
        self, pred_coords: List[Coord], prey_coords: List[Coord]
    ) -> List[pp.PPAction]:
        if len(prey_coords) != 0:
            closest_prey_coord = self.get_closest_coord(
                self.agent_obs_coord, prey_coords
            )
            return self.get_actions_towards_target(closest_prey_coord)

        if len(pred_coords) != 0:
            closest_pred_coord = self.get_closest_coord(
                self.agent_obs_coord, pred_coords
            )
            return self.get_actions_towards_target(closest_pred_coord)

        return []


class PPHeuristic2(PPHeuristicPolicy):
    """H2 moves towards closest observed prey, closest observed predator, or explores in
    a clockwise spiral around arena, in that order.
    """

    def __init__(self, model: POSGModel, agent_id: str, policy_id: PolicyID):
        super().__init__(model, agent_id, policy_id, "spiral")

    def get_actions_from_obs(
        self, pred_coords: List[Coord], prey_coords: List[Coord]
    ) -> List[pp.PPAction]:
        if len(prey_coords) != 0:
            closest_prey_coord = self.get_closest_coord(
                self.agent_obs_coord, prey_coords
            )
            return self.get_actions_towards_target(closest_prey_coord)

        if len(pred_coords) != 0:
            closest_pred_coord = self.get_closest_coord(
                self.agent_obs_coord, pred_coords
            )
            return self.get_actions_towards_target(closest_pred_coord)

        return []


class PPHeuristic3(PPHeuristicPolicy):
    """H3 moves towards closest observed prey to the closest observed predator or
    explores in a clockwise spiral around arena, in that order.
    """

    def __init__(self, model: POSGModel, agent_id: str, policy_id: PolicyID):
        super().__init__(model, agent_id, policy_id, "spiral")

    def get_actions_from_obs(
        self, pred_coords: List[Coord], prey_coords: List[Coord]
    ) -> List[pp.PPAction]:
        if len(prey_coords) == 0:
            return []

        if len(pred_coords) == 0:
            # no predators observed, just move towards closest prey
            closest_prey_coord = self.get_closest_coord(
                self.agent_obs_coord, prey_coords
            )
            return self.get_actions_towards_target(closest_prey_coord)

        # Move towards prey closest to predator
        closest_pred_coord = self.get_closest_coord(self.agent_obs_coord, pred_coords)
        closest_prey_coord = self.get_closest_coord(closest_pred_coord, prey_coords)
        return self.get_actions_towards_target(closest_prey_coord)
