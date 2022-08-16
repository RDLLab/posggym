import sys
from typing import Optional, Tuple

from posggym import core
import posggym.model as M

import posggym.envs.classic.mabc.model as mabc_model


class MABCEnv(core.DefaultEnv):
    """The Multi-Access Broadcast Channel Environment.

    A cooperative game involving control of a multi-access broadcast channel.
    In this problem, each agent controls a node. Each node need to broadcast
    messages to each other over a shared channel, with only one node able to
    broadcast at a time. If more than one node broadcasts at the same time then
    there is a collision and no message is broadcast. The nodes share the
    common goal of maximizing the throughput of the channel.

    State
    -----
    Each node has a message buffer that can store up to one message at a time.
    That is it can be either `EMPTY` or `FULL`.

    Actions
    -------
    At each timestep each agent can either `SEND` a message or not (`NOSEND`).

    Observation
    -----------
    At the end of each time step, each node recieves a noisy observation of
    whether there was a `COLLISION` or `NOCOLLISION`.

    Each agent observes the true outcome with probability `obs_prob`, which is
    0.9 in the default version.

    Reward
    ------
    Each agent recieves a reward of `1` when a message is successfully
    broadcast and a reward of `0` otherwise.

    Transition Dynamics
    -------------------
    If a node's buffer is EMPTY then at each step it will become full with
    probability `fill_probs[i]`, otherwise it will remain empty (independent
    of the action performed).

    If a node's buffer is `FULL` then if the node does not send a message
    - i.e. uses the `NOSEND` action - then the buffer remains `FULL`.
    Otherwise - i.e. the agent chooses the `SEND` action - if no other nodes
    sends a message (there is no COLLISION) the buffer will be FULL with
    probability `fill_probs[i]`, otherwise it will be empty. If another node
    did sent a message as well then there was a COLLISION and the node's buffer
    remains `FULL`.

    References
    ----------
    - Ooi, J. M., and Wornell, G. W. 1996. Decentralized control of a multiple
      access broadcast channel: Performance bounds. In Proceedings of the 35th
      Conference on Decision and Control, 293–298.
    - Hansen, Eric A., Daniel S. Bernstein, and Shlomo Zilberstein. “Dynamic
      Programming for Partially Observable Stochastic Games.” In Proceedings of
      the 19th National Conference on Artifical Intelligence, 709–715. AAAI’04.
      San Jose, California: AAAI Press, 2004.

    """

    metadata = {"render.modes": ['human', 'ansi']}

    def __init__(self,
                 num_nodes: int = 2,
                 fill_probs: Optional[Tuple[float, ...]] = None,
                 observation_prob: float = 0.9,
                 init_buffer_dist: Optional[Tuple[float, ...]] = None,
                 **kwargs):
        self._model = mabc_model.MABCModel(
            num_nodes, fill_probs, observation_prob, init_buffer_dist, **kwargs
        )
        super().__init__()

    def render(self, mode: str = "human"):
        if mode not in self.metadata["render.modes"]:
            # raise exception
            super().render(mode)

        state_str = ", ".join(
            [mabc_model.NODE_STATE_STR[s] for s in self._state]
        )
        obs_str = ", ".join([mabc_model.OBS_STR[o] for o in self._last_obs])
        output = [
            f"Step: {self._step_num}",
            f"State: <{state_str}>",
            f"Obs: <{obs_str}>"
        ]
        if self._last_actions is not None:
            action_str = ", ".join(
                [mabc_model.ACTION_STR[a] for a in self._last_actions]
            )
            output.insert(1, f"Actions: <{action_str}>")
            output.append(f"Rewards: <{self._last_rewards}>")

        output_str = "\n".join(output) + "\n"

        if mode == "human":
            sys.stdout.write(output_str)
        else:
            # ansi mode
            return output_str

    @property
    def model(self) -> M.POSGModel:
        return self._model
