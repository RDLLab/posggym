import sys
from typing import Tuple, Set

from gym import spaces

from posggym import Env

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    from ray.rllib.utils.typing import MultiAgentDict, AgentID
except ImportError:
    print(
        "The posggym.wrapper.rllib_multi_agent_env Wrapper depends on the "
        "ray RLlib library. For installation instructions visit "
        "https://docs.ray.io/en/latest/rllib/index.html"
    )
    sys.exit(1)


class RllibMultiAgentEnv(MultiAgentEnv):
    """An interface between a POSGGym env and a Rllib MultiAgentEnv.

    This involves:
    - treating agent IDs as strings (instead of ints)
    - converting action, observations, and rewards from an ordered tuple to a
      dictionary mapping agent ID to the agent's action/observation/reward
    """

    def __init__(self, env: Env):
        # must assign this first before calling super().__init__() so that
        # property functions are initialized before super().__init__() is
        # called
        self.env = env
        self._agent_ids = set(str(i) for i in range(self.env.n_agents))
        super().__init__()

    @property
    def observation_space(self):
        """Get the environment observation space.

        This is a dictionary mapping agent IDs to the agent's observation space
        """
        return spaces.Dict({
            str(i): self.env.observation_spaces[i]
            for i in range(self.env.n_agents)
        })

    @property
    def action_space(self):
        """Get the environment action space.

        This is a dictionary mapping agent IDs to the agent's action space
        """
        return spaces.Dict(
            {
                str(i): self.env.action_spaces[i]
                for i in range(self.env.n_agents)
            }
        )

    @property
    def n_agents(self) -> int:
        return self.env.n_agents

    def get_agent_ids(self) -> Set[AgentID]:
        """Return a set of agent ids in the environment."""
        return self._agent_ids

    def reset(self) -> MultiAgentDict:
        obs = self.env.reset()
        return {str(i): o for i, o in enumerate(obs)}

    def step(self,
             action_dict: MultiAgentDict
             ) -> Tuple[
                 MultiAgentDict,
                 MultiAgentDict,
                 MultiAgentDict,
                 MultiAgentDict
             ]:
        actions = tuple(action_dict[str(i)] for i in range(self.env.n_agents))

        obs, rewards, done, info = self.env.step(actions)

        obs_dict = {str(i): o for i, o in enumerate(obs)}
        reward_dict = {str(i): r for i, r in enumerate(rewards)}
        done_dict = {str(i): done for i in range(self.env.n_agents)}
        info_dict = {str(i): info for i in range(self.env.n_agents)}

        # need to add RLlib specific special key to signify episode termination
        done_dict["__all__"] = done

        return obs_dict, reward_dict, done_dict, info_dict

    def render(self, mode=None):
        return_val = self.env.render(mode)
        if return_val is None:
            return True
        return return_val

    def close(self):
        self.env.close()

    @property
    def unwrapped(self):
        return self
