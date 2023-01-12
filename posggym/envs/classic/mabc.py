"""The Multi-Access Broadcast problem.

A cooperative game involving control of a multi-access broadcast channel. In this
problem, each agent controls a node. Each node need to broadcast messages to each other
over a shared channel, with only one node able to broadcast at a time. If more than one
node broadcasts at the same time then there is a collision and no message is broadcast.
The nodes share the common goal of maximizing the throughput of the channel.

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
import sys
from itertools import product
from typing import Dict, List, Optional, SupportsFloat, Tuple, Union

from gymnasium import spaces

import posggym.model as M
from posggym.core import DefaultEnv
from posggym.utils import seeding


MABCState = Tuple[int, ...]
EMPTY = 0
FULL = 1
NODE_STATES = [EMPTY, FULL]
NODE_STATE_STR = ["E", "F"]

MABCAction = int
SEND = 0
NOSEND = 1
ACTIONS = [SEND, NOSEND]
ACTION_STR = ["S", "NS"]

MABCObs = int
COLLISION = 0
NOCOLLISION = 1
OBS = [COLLISION, NOCOLLISION]
OBS_STR = ["C", "NC"]


class MABCEnv(DefaultEnv[MABCState, MABCObs, MABCAction]):
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

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        num_nodes: int = 2,
        fill_probs: Optional[Tuple[float, ...]] = None,
        observation_prob: float = 0.9,
        init_buffer_dist: Optional[Tuple[float, ...]] = None,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            MABCModel(
                num_nodes, fill_probs, observation_prob, init_buffer_dist, **kwargs
            ),
            render_mode=render_mode,
        )

    def render(self):
        assert self._last_obs is not None

        state_str = ", ".join([NODE_STATE_STR[s] for s in self._state])
        obs_str = ", ".join([OBS_STR[o] for o in self._last_obs.values()])
        output = [
            f"Step: {self._step_num}",
            f"State: <{state_str}>",
            f"Obs: <{obs_str}>",
        ]
        if self._last_actions is not None:
            action_str = ", ".join([ACTION_STR[a] for a in self._last_actions.values()])
            output.insert(1, f"Actions: <{action_str}>")
            output.append(f"Rewards: <{tuple(self._last_rewards.values())}>")

        output_str = "\n".join(output) + "\n"

        if self.render_mode == "human":
            sys.stdout.write(output_str)
        else:
            # ansi mode
            return output_str


class MABCModel(M.POSGFullModel[MABCState, MABCObs, MABCAction]):
    """POSG Model for the Multi-Access Broadcast Channel problem."""

    DEFAULT_FILL_PROBS = (0.9, 0.1)
    DEFAULT_OBS_PROB = 0.9
    DEFAULT_INIT_BUFFER_DIST = (1.0, 1.0)

    R_SEND = 1.0
    R_NO_SEND = 0.0

    def __init__(
        self,
        num_nodes: int = 2,
        fill_probs: Optional[Tuple[float, ...]] = None,
        observation_prob: float = 0.9,
        init_buffer_dist: Optional[Tuple[float, ...]] = None,
        **kwargs,
    ):
        assert num_nodes >= 2

        if fill_probs is None:
            fill_probs = self.DEFAULT_FILL_PROBS
        if init_buffer_dist is None:
            init_buffer_dist = self.DEFAULT_INIT_BUFFER_DIST

        assert len(fill_probs) == num_nodes
        assert all(0 <= x <= 1 for x in fill_probs)
        assert 0 <= observation_prob <= 1
        assert len(init_buffer_dist) == num_nodes
        assert all(0 <= x <= 1 for x in init_buffer_dist)

        self._fill_probs = fill_probs
        self._obs_prob = observation_prob
        self._init_buffer_dist = init_buffer_dist

        self.possible_agents = tuple(range(num_nodes))
        self.state_space = spaces.Tuple(
            tuple(spaces.Discrete(len(NODE_STATES)) for i in self.possible_agents)
        )
        self.action_spaces = {
            i: spaces.Discrete(len(ACTIONS)) for i in self.possible_agents
        }
        self.observation_spaces = {
            i: spaces.Discrete(len(OBS)) for i in self.possible_agents
        }
        self.observation_first = False
        self.is_symmetric = True

        # Spaces used internally
        self._state_space = list(
            product(*[list(NODE_STATES) for i in self.possible_agents])
        )
        self._action_spaces = tuple([*ACTIONS] for i in self.possible_agents)
        self._observation_spaces = tuple([*OBS] for i in self.possible_agents)

        self._trans_map = self._construct_trans_func()
        self._rew_map = self._construct_rew_func()
        self._obs_map = self._construct_obs_func()

    @property
    def reward_ranges(self) -> Dict[M.AgentID, Tuple[SupportsFloat, SupportsFloat]]:
        return {i: (self.R_NO_SEND, self.R_SEND) for i in self.possible_agents}

    @property
    def rng(self) -> seeding.RNG:
        if self._rng is None:
            self._rng, seed = seeding.std_random()
        return self._rng

    def get_agents(self, state: MABCState) -> List[M.AgentID]:
        return list(self.possible_agents)

    def sample_initial_state(self) -> MABCState:
        node_states = []
        for i in range(len(self.possible_agents)):
            if self.rng.random() <= self._init_buffer_dist[i]:
                node_states.append(FULL)
            else:
                node_states.append(EMPTY)
        return tuple(node_states)

    def step(
        self, state: MABCState, actions: Dict[M.AgentID, MABCAction]
    ) -> M.JointTimestep[MABCState, MABCObs]:
        assert all(a_i in ACTIONS for a_i in actions.values())
        next_state = self._sample_next_state(state, actions)
        obs = self._sample_obs(actions)
        agent_reward = float(self._message_sent(state, actions)) * self.R_SEND
        rewards: Dict[M.AgentID, SupportsFloat] = {
            i: agent_reward for i in self.possible_agents
        }
        terminated = {i: False for i in self.possible_agents}
        truncated = {i: False for i in self.possible_agents}
        all_done = False
        info: Dict[M.AgentID, Dict] = {i: {} for i in self.possible_agents}

        return M.JointTimestep(
            next_state, obs, rewards, terminated, truncated, all_done, info
        )

    def _sample_next_state(
        self, state: MABCState, actions: Dict[M.AgentID, MABCAction]
    ) -> MABCState:
        next_node_states = list(state)
        for i, a_i in enumerate(actions):
            if a_i == SEND:
                # buffer emptied even if there is a collision
                next_node_states[i] = EMPTY
            if self.rng.random() <= self._fill_probs[i]:
                next_node_states[i] = FULL
        return tuple(next_node_states)

    def _sample_obs(
        self, actions: Dict[M.AgentID, MABCAction]
    ) -> Dict[M.AgentID, MABCObs]:
        senders = sum(int(a_i == SEND) for a_i in actions)
        if senders > 1:
            correct_obs = COLLISION
            wrong_obs = NOCOLLISION
        else:
            correct_obs = NOCOLLISION
            wrong_obs = COLLISION

        obs = {}
        for i in self.possible_agents:
            if self.rng.random() <= self._obs_prob:
                obs[i] = correct_obs
            else:
                obs[i] = wrong_obs
        return obs

    def get_initial_belief(self) -> Dict[MABCState, float]:
        b_map: Dict[MABCState, float] = {}
        s_prob_sum = 0.0
        for s in self._state_space:
            s_prob = 1.0
            for i in range(len(self.possible_agents)):
                if s[i] == FULL:
                    s_prob *= self._init_buffer_dist[i]
                else:
                    s_prob *= 1 - self._init_buffer_dist[i]
            b_map[s] = s_prob
            s_prob_sum += s_prob

        for s in self._state_space:
            b_map[s] /= s_prob_sum

        return b_map

    def transition_fn(
        self,
        state: MABCState,
        actions: Dict[M.AgentID, MABCAction],
        next_state: MABCState,
    ) -> float:
        action_tuple = tuple(actions[i] for i in self.possible_agents)
        return self._trans_map[(state, action_tuple, next_state)]

    def _construct_trans_func(self) -> Dict:
        trans_map = {}
        agent_ids = [int(i) for i in self.possible_agents]
        for (s, a, s_next) in product(
            self._state_space, product(*self._action_spaces), self._state_space
        ):
            trans_prob = 1.0
            for i in agent_ids:
                if a[i] == NOSEND and s[i] == FULL:
                    if not s_next[i] == FULL:
                        trans_prob *= 0.0
                        break

                if s_next[i] == FULL:
                    trans_prob *= self._fill_probs[i]
                else:
                    trans_prob *= 1 - self._fill_probs[i]

            trans_map[(s, a, s_next)] = trans_prob
        return trans_map

    def observation_fn(
        self,
        obs: Dict[M.AgentID, MABCObs],
        next_state: MABCState,
        actions: Dict[M.AgentID, MABCAction],
    ) -> float:
        obs_tuple = tuple(obs[i] for i in self.possible_agents)
        action_tuple = tuple(actions[i] for i in self.possible_agents)
        return self._obs_map[(next_state, action_tuple, obs_tuple)]

    def _construct_obs_func(self) -> Dict:
        obs_map = {}
        agent_ids = [int(i) for i in self.possible_agents]
        for (s_next, a, o) in product(
            self._state_space,
            product(*self._action_spaces),
            product(*self._observation_spaces),
        ):
            senders = sum(int(a_i == SEND) for a_i in a)
            if senders > 1:
                correct_obs = COLLISION
            else:
                correct_obs = NOCOLLISION

            o_prob = 1.0
            for i in agent_ids:
                if o[i] == correct_obs:
                    o_prob *= self._obs_prob
                else:
                    o_prob *= 1 - self._obs_prob
            obs_map[(s_next, a, o)] = o_prob
        return obs_map

    def reward_fn(
        self, state: MABCState, actions: Dict[M.AgentID, MABCAction]
    ) -> Dict[M.AgentID, SupportsFloat]:
        action_tuple = tuple(actions[i] for i in self.possible_agents)
        return self._rew_map[(state, action_tuple)]

    def _construct_rew_func(self) -> Dict:
        rew_map = {}
        joint_actions_space = product(*self._action_spaces)
        for (s, a) in product(self._state_space, joint_actions_space):
            reward = float(self._message_sent(s, a)) * self.R_SEND
            rew_map[(s, a)] = tuple(reward for _ in self.possible_agents)
        return rew_map

    def _message_sent(
        self,
        state: MABCState,
        actions: Union[Dict[M.AgentID, MABCAction], Tuple[MABCAction, ...]],
    ) -> bool:
        senders = sum(int(actions[int(i)] == SEND) for i in self.possible_agents)
        if senders != 1:
            return False

        message_sent = False
        for i in self.possible_agents:
            if actions[int(i)] == NOSEND:
                continue
            if state[int(i)] == FULL:
                message_sent = True
            break

        return message_sent
