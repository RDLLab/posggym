"""The POSG Model for the Multi-Access Broadcast problem."""
import random
from itertools import product
from typing import Tuple, Sequence, Dict, Optional

from gym import spaces

import posggym.model as M


MATState = int
TLEFT = 0
TRIGHT = 1
STATES = [TLEFT, TRIGHT]
STATE_STRS = ["TL", "TR"]

MATAction = int
OPENLEFT = 0
OPENRIGHT = 1
LISTEN = 2
MATJointAction = Tuple[MATAction, ...]
ACTIONS = [OPENLEFT, OPENRIGHT, LISTEN]
ACTION_STR = ["OL", "OR", "L"]

MATObs = Tuple[int, int]
GROWLLEFT = 0
GROWLRIGHT = 1
CREAKLEFT = 0
CREAKRIGHT = 1
SILENCE = 2
MATJointObs = Tuple[MATObs, ...]
OBS_PARTS = [(GROWLLEFT, GROWLRIGHT), (CREAKLEFT, CREAKRIGHT, SILENCE)]
OBS_SPACE = [
    (GROWLLEFT, CREAKLEFT),
    (GROWLLEFT, CREAKRIGHT),
    (GROWLLEFT, SILENCE),
    (GROWLRIGHT, CREAKLEFT),
    (GROWLRIGHT, CREAKRIGHT),
    (GROWLRIGHT, SILENCE)
]
OBS_STR = [("GL", "GR"), ("CL", "CR", "S")]


class MATB0(M.Belief):
    """The initial belief in a Multi-Agent Tiger problem."""

    def __init__(self, rng: random.Random):
        self._rng = rng

    def sample(self) -> M.State:
        return self._rng.choice(STATES)

    def sample_k(self, k: int) -> Sequence[M.State]:
        return [self.sample() for _ in range(k)]

    def get_dist(self) -> Dict[M.State, float]:
        b_map: Dict[MATState, float] = {}
        for s in STATES:
            s_prob = 1.0 / len(STATES)
            b_map[s] = s_prob
        return b_map


class MultiAgentTigerModel(M.POSGFullModel):
    """Multi-Agent Tiger problem Model."""

    NUM_AGENTS = 2

    OPEN_GOOD_R = 10.0
    OPEN_BAD_R = -100.0
    LISTEN_R = -1.0

    def __init__(self,
                 observation_prob: float = 0.85,
                 creak_observation_prob: float = 0.9,
                 **kwargs):
        super().__init__(self.NUM_AGENTS, **kwargs)
        self._obs_prob = observation_prob
        self._creak_obs_prob = creak_observation_prob

        self._state_space = STATES
        self._action_spaces = tuple([*ACTIONS] for _ in range(self.n_agents))
        self._obs_spaces = tuple(OBS_SPACE for _ in range(self.n_agents))

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
        return spaces.Discrete(len(STATES))

    @property
    def action_spaces(self) -> Tuple[spaces.Space, ...]:
        return tuple(
            spaces.Discrete(len(ACTIONS)) for _ in range(self.n_agents)
        )

    @property
    def observation_spaces(self) -> Tuple[spaces.Space, ...]:
        return tuple(
            spaces.Tuple((
                spaces.Discrete(len(OBS_PARTS[0])),
                spaces.Discrete(len(OBS_PARTS[1]))
            )) for _ in range(self.n_agents)
        )

    @property
    def reward_ranges(self) -> Tuple[Tuple[M.Reward, M.Reward], ...]:
        return tuple(
            (self.OPEN_BAD_R, self.OPEN_GOOD_R) for _ in range(self.n_agents)
        )

    @property
    def initial_belief(self) -> M.Belief:
        return MATB0(self._rng)

    def step(self,
             state: M.State,
             actions: M.JointAction
             ) -> M.JointTimestep:
        next_state = self._sample_next_state(state, actions)
        obs = self._sample_obs(state, actions)
        rewards = self._get_reward(state, actions)
        dones = (False,) * self.n_agents
        all_done = False
        outcomes = tuple(M.Outcome.NA for _ in range(self.n_agents))
        return M.JointTimestep(
            next_state, obs, rewards, dones, all_done, outcomes
        )

    def _sample_next_state(self,
                           state: MATState,
                           actions: MATJointAction) -> MATState:
        next_state = state
        if any(a != LISTEN for a in actions):
            next_state = self._rng.choice(STATES)
        return next_state

    def _sample_obs(self,
                    state: MATState,
                    actions: MATJointAction) -> MATJointObs:
        obs_list = []
        for agent_id, a in enumerate(actions):
            if a != LISTEN:
                obs_list.append(self._rng.choice(OBS_SPACE))
                continue

            other_agent_id = (agent_id + 1) % self.n_agents
            other_agent_action = actions[other_agent_id]

            tiger_obs = self._sample_tiger_obs(state)
            creak_obs = self._sample_creak_obs(other_agent_action)
            obs_list.append((tiger_obs, creak_obs))

        return tuple(obs_list)

    def _sample_tiger_obs(self, state: MATState) -> int:
        if self._rng.random() < self._obs_prob:
            # correct obs
            return GROWLLEFT if state == TLEFT else GROWLRIGHT
        return GROWLRIGHT if state == TLEFT else GROWLLEFT

    def _sample_creak_obs(self, a_j: MATAction) -> int:
        if self._rng.random() < self._creak_obs_prob:
            # correct obs
            if a_j == LISTEN:
                return SILENCE
            return CREAKLEFT if a_j == OPENLEFT else CREAKRIGHT

        if a_j == LISTEN:
            return self._rng.choice([CREAKLEFT, CREAKRIGHT])
        if a_j == OPENLEFT:
            return self._rng.choice([CREAKRIGHT, SILENCE])
        return self._rng.choice([CREAKRIGHT, SILENCE])

    def _get_reward(self,
                    state: MATState,
                    actions: MATJointAction) -> M.JointReward:
        rewards = []
        for a in actions:
            if a == LISTEN:
                rewards.append(self.LISTEN_R)
            elif a == state:
                rewards.append(self.OPEN_BAD_R)
            else:
                rewards.append(self.OPEN_GOOD_R)
        return tuple(rewards)

    def set_seed(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def transition_fn(self,
                      state: M.State,
                      actions: M.JointAction,
                      next_state: M.State) -> float:
        return self._trans_map[(state, actions, next_state)]

    def _construct_trans_func(self) -> Dict:
        trans_map = {}
        uniform_prob = 1.0 / len(STATES)
        for (s, a, s_next) in product(
                self._state_space,
                product(*self._action_spaces),
                self._state_space
        ):
            if any(a_i != LISTEN for a_i in a):
                p = uniform_prob
            else:
                p = float(s_next == s)
            trans_map[(s, a, s_next)] = p
        return trans_map

    def observation_fn(self,
                       obs: M.JointObservation,
                       next_state: M.State,
                       actions: M.JointAction) -> float:
        return self._obs_map[(next_state, actions, obs)]

    def _construct_obs_func(self) -> Dict:
        obs_func = {}
        uniform_o_prob = 1.0 / len(OBS_SPACE)
        for (s_next, a, o) in product(
                self._state_space,
                product(*self._action_spaces),
                product(*self._obs_spaces)
        ):
            o_prob = 1.0

            a_i, a_j = a
            o_i, o_j = o

            correct_pos = GROWLLEFT if s_next == TLEFT else GROWLRIGHT

            if (a_i, a_j) == (LISTEN, LISTEN):
                for o_k in (o_i, o_j):
                    if o_k[0] == correct_pos:
                        o_prob *= self._obs_prob
                    else:
                        o_prob *= (1.0 - self._obs_prob)

                    if o_k[1] == SILENCE:
                        o_prob *= self._creak_obs_prob
                    else:
                        o_prob *= (1.0 - self._creak_obs_prob) / 2

            elif LISTEN in (a_i, a_j):
                if a_i == LISTEN:
                    a_k, o_k = a_j, o_i
                else:
                    a_k, o_k = a_i, o_j

                o_prob *= uniform_o_prob
                correct_creak = CREAKLEFT if a_k == OPENLEFT else CREAKRIGHT

                if o_k[0] == correct_pos:
                    o_prob *= self._obs_prob
                else:
                    o_prob *= (1.0 - self._obs_prob)

                if o_k[1] == correct_creak:
                    o_prob *= self._creak_obs_prob
                else:
                    o_prob *= (1.0 - self._creak_obs_prob) / 2

            else:
                o_prob *= uniform_o_prob * uniform_o_prob

            obs_func[(s_next, a, o)] = o_prob

        return obs_func

    def reward_fn(self,
                  state: M.State,
                  actions: M.JointAction) -> M.JointReward:
        return self._rew_map[(state, actions)]

    def _construct_rew_func(self) -> Dict:
        rew_map = {}
        joint_actions_space = product(*self._action_spaces)
        for (s, a) in product(self._state_space, joint_actions_space):
            rew_map[(s, a)] = self._get_reward(s, a)
        return rew_map
