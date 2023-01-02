"""POSG Model for the Multi-Access Broadcast problem."""
from __future__ import annotations

import random
from itertools import product
from typing import Dict, List, Optional, SupportsFloat, Tuple, Union

from gymnasium import spaces

import posggym.model as M


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


class MABCB0(M.Belief[MABCState]):
    """The initial belief in a MABC problem."""

    def __init__(
        self,
        n_agents: int,
        init_buffer_dist: Tuple[float, ...],
        state_space: List[MABCState],
        rng: random.Random,
    ):
        self._n_agents = n_agents
        self._init_buffer_dist = init_buffer_dist
        self._state_space = state_space
        self._rng = rng

    def sample(self) -> MABCState:
        node_states = []
        for i in range(self._n_agents):
            if self._rng.random() <= self._init_buffer_dist[i]:
                node_states.append(FULL)
            else:
                node_states.append(EMPTY)
        return tuple(node_states)

    def get_dist(self) -> Dict[MABCState, float]:
        b_map: Dict[MABCState, float] = {}
        s_prob_sum = 0.0
        for s in self._state_space:
            s_prob = 1.0
            for i in range(self._n_agents):
                if s[i] == FULL:
                    s_prob *= self._init_buffer_dist[i]
                else:
                    s_prob *= 1 - self._init_buffer_dist[i]
            b_map[s] = s_prob
            s_prob_sum += s_prob

        for s in self._state_space:
            b_map[s] /= s_prob_sum

        return b_map


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
        fill_probs: Tuple[float, ...] | None = None,
        observation_prob: float = 0.9,
        init_buffer_dist: Tuple[float, ...] | None = None,
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

        self.possible_agents = tuple(range(num_nodes))

        self._state_space = list(
            product(*[list(NODE_STATES) for i in self.possible_agents])
        )
        self._action_spaces = tuple([*ACTIONS] for i in self.possible_agents)
        self._observation_spaces = tuple([*OBS] for i in self.possible_agents)

        self._fill_probs = fill_probs
        self._obs_prob = observation_prob
        self._init_buffer_dist = init_buffer_dist

        self._rng = random.Random(None)

        self._trans_map = self._construct_trans_func()
        self._rew_map = self._construct_rew_func()
        self._obs_map = self._construct_obs_func()

    @property
    def observation_first(self) -> bool:
        return False

    @property
    def is_symmetric(self) -> bool:
        return True

    @property
    def state_space(self) -> spaces.Space:
        return spaces.Tuple(
            tuple(spaces.Discrete(len(NODE_STATES)) for i in self.possible_agents)
        )

    @property
    def action_spaces(self) -> Dict[M.AgentID, spaces.Space]:
        return {i: spaces.Discrete(len(ACTIONS)) for i in self.possible_agents}

    @property
    def observation_spaces(self) -> Dict[M.AgentID, spaces.Space]:
        return {i: spaces.Discrete(len(OBS)) for i in self.possible_agents}

    @property
    def reward_ranges(self) -> Dict[M.AgentID, Tuple[SupportsFloat, SupportsFloat]]:
        return {i: (self.R_NO_SEND, self.R_SEND) for i in self.possible_agents}

    @property
    def initial_belief(self) -> MABCB0:
        return MABCB0(
            len(self.possible_agents),
            self._init_buffer_dist,
            self._state_space,
            self._rng,
        )

    def get_agents(self, state: MABCState) -> List[M.AgentID]:
        return list(self.possible_agents)

    def step(
        self, state: MABCState, actions: Dict[M.AgentID, MABCAction]
    ) -> M.JointTimestep:
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
        outcomes = {i: M.Outcome.NA for i in self.possible_agents}
        info: Dict[M.AgentID, Dict] = {i: {} for i in self.possible_agents}
        return M.JointTimestep(
            next_state,
            obs,
            rewards,
            terminated,
            truncated,
            all_done,
            outcomes,
            info
        )

    def _sample_next_state(
        self, state: MABCState, actions: Dict[M.AgentID, MABCAction]
    ) -> MABCState:
        next_node_states = list(state)
        for i, a_i in enumerate(actions):
            if a_i == SEND:
                # buffer emptied even if there is a collision
                next_node_states[i] = EMPTY
            if self._rng.random() <= self._fill_probs[i]:
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
            if self._rng.random() <= self._obs_prob:
                obs[i] = correct_obs
            else:
                obs[i] = wrong_obs
        return obs

    def set_seed(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

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
