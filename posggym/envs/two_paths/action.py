"""Actions for Two-Paths Problem """
from typing import List, Tuple

from posggym.model import DiscreteAction

import posggym.envs.two_paths.grid as grid_lib


class TPAction(DiscreteAction):
    """An action in the Runner Chaser Problem """

    def __str__(self):
        return grid_lib.DIR_STRS[self.action_num]


def get_action_spaces(num_agents: int) -> Tuple[List[TPAction], ...]:
    """Get action space for each agent in the Two-Paths problem """
    agent_action_spaces = []
    for _ in range(num_agents):
        agent_actions = []
        for action_num in grid_lib.DIRS:
            agent_actions.append(TPAction(action_num))
        agent_action_spaces.append(agent_actions)

    return tuple(agent_action_spaces)
