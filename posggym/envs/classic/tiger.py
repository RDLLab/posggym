"""Model for the classic Multi-Agent Tiger problem.

This is a general-sum multi-agent version of the classic Tiger problem. It involves two
agents that are in a corridor facing two doors: `left` and `right`. Behind one door lies
a hungry tiger and behind the other lies treasure, but the agents do not know the
position of either the tiger or the treasure.

References
----------
- Gmytrasiewicz, Piotr J., and Prashant Doshi. “A Framework for Sequential
Planning in Multi-Agent Settings.” Journal of Artificial Intelligence
Research 24 (2005): 49–79.

"""
import sys
from itertools import product
from typing import Dict, List, Optional, SupportsFloat, Tuple

from gymnasium import spaces

import posggym.model as M
from posggym.core import DefaultEnv
from posggym.utils import seeding


MATState = int
TLEFT = 0
TRIGHT = 1
STATES = [TLEFT, TRIGHT]
STATE_STRS = ["TL", "TR"]

MATAction = int
OPENLEFT = 0
OPENRIGHT = 1
LISTEN = 2
ACTIONS = [OPENLEFT, OPENRIGHT, LISTEN]
ACTION_STR = ["OL", "OR", "L"]

MATObs = Tuple[int, int]
GROWLLEFT = 0
GROWLRIGHT = 1
CREAKLEFT = 0
CREAKRIGHT = 1
SILENCE = 2
OBS_PARTS = [(GROWLLEFT, GROWLRIGHT), (CREAKLEFT, CREAKRIGHT, SILENCE)]
OBS_SPACE = [
    (GROWLLEFT, CREAKLEFT),
    (GROWLLEFT, CREAKRIGHT),
    (GROWLLEFT, SILENCE),
    (GROWLRIGHT, CREAKLEFT),
    (GROWLRIGHT, CREAKRIGHT),
    (GROWLRIGHT, SILENCE),
]
OBS_STR = [("GL", "GR"), ("CL", "CR", "S")]


class MultiAgentTigerEnv(DefaultEnv):
    """The Multi-Agent Tiger Environment.

    This is a general-sum multi-agent version of the classic Tiger problem.

    This scenario involves two agents that are in a corridor facing two doors:
    `left` and `right`. Behind one door lies a hungry tiger and behind the
    other lies treasure, but the agents do not know the position of either the
    tiger or the treasure.

    State
    -----
    The state is defined by which door the tiger is behind. `TLEFT` for tiger
    is behind the `left` door, and `TRIGHT` for the tiger is behind the `right`
    door.

    S = {`TLEFT`, `TRIGHT`}

    The initial state is uniformly distributed between the possible states.

    Actions
    -------
    Each agent can either open the left-hand door `OPENLEFT`, open the
    right-hand door `OPENRIGHT`, or listen for the presence of the tiger
    `LISTEN`.

    A_1 = A_2 = {`OPENLEFT`, `OPENRIGHT`, `LISTEN`}

    Observation
    -----------
    Agents recieve observations of the position tiger: `GROWLEFT` for tiger
    left, and `GROWLRIGHT` for tiger right. Additionally, they also observe
    if a door has creaked: `CREAKLEFT` for left door, `CREAKRIGHT` for right
    door, and `SILENCE` for silence.

    O_1 = O_2 = {
        (`GROWLEFT`, `CREAKLEFT`),
        (`GROWLEFT`, `CREAKRIGHT`),
        (`GROWLEFT`, `SILENCE`),
        (`GROWLRIGHT`, `CREAKLEFT`),
        (`GROWLRIGHT`, `CREAKRIGHT`),
        (`GROWLRIGHT`, `SILENCE`)
    }

    If an agent uses the `LISTEN` action they will perceive the correct
    current position of the tiger with probability observation_prob=0.85,
    independent of if the other agent opens a door or listens. Furthermore,
    they will perceive the correct door opening or not with probability
    creak_observation_prob=0.9.

    If an agent opens either door they will recieve an observation uniformly at
    random.

    Reward
    ------
    Each agent recieves rewards independent of the other agent.

    An agent recieves a reward of +10 for opening the door without a tiger
    behind it, -100 for opening the door with the tiger behind it, and -1 for
    performing the listening action.

    Although the game is general-sum, and not zero-sum, agents influence each
    other by the way they influence the state.

    Transition Dynamics
    -------------------
    The state is reset to `TLEFT` or `TRIGHT` with equal probability whenever
    either agent opens one of the doors. Otherwise - when both agents perform
    the `LISTEN` action - the state is unchanged.

    References
    ----------
    - Gmytrasiewicz, Piotr J., and Prashant Doshi. “A Framework for Sequential
    Planning in Multi-Agent Settings.” Journal of Artificial Intelligence
    Research 24 (2005): 49–79.

    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        observation_prob: float = 0.85,
        creak_observation_prob: float = 0.9,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            MultiAgentTigerModel(observation_prob, creak_observation_prob, **kwargs),
            render_mode=render_mode,
        )

    def render(self):
        assert self._last_obs is not None

        state_str = STATE_STRS[self._state]
        obs_str = ", ".join(
            [str((OBS_STR[0][o[0]], OBS_STR[1][o[1]])) for o in self._last_obs.values()]
        )
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


class MultiAgentTigerModel(M.POSGFullModel[MATState, MATObs, MATAction]):
    """Multi-Agent Tiger problem Model."""

    NUM_AGENTS = 2

    OPEN_GOOD_R = 10.0
    OPEN_BAD_R = -100.0
    LISTEN_R = -1.0

    def __init__(
        self,
        observation_prob: float = 0.85,
        creak_observation_prob: float = 0.9,
        **kwargs,
    ):
        assert 0 <= observation_prob <= 1.0
        assert 0 <= creak_observation_prob <= 1.0
        self._obs_prob = observation_prob
        self._creak_obs_prob = creak_observation_prob

        self.possible_agents = tuple(range(self.NUM_AGENTS))
        self.state_space = spaces.Discrete(len(STATES))
        self.action_spaces = {
            i: spaces.Discrete(len(ACTIONS)) for i in self.possible_agents
        }
        self.observation_spaces = {
            i: spaces.Tuple(
                (spaces.Discrete(len(OBS_PARTS[0])), spaces.Discrete(len(OBS_PARTS[1])))
            )
            for i in self.possible_agents
        }
        self.observation_first = False
        self.is_symmetric = True

        # Spaces used internally
        self._state_space = STATES
        self._action_spaces = tuple(
            [*ACTIONS] for _ in range(len(self.possible_agents))
        )
        self._obs_spaces = tuple(OBS_SPACE for _ in range(len(self.possible_agents)))

        self._trans_map = self._construct_trans_func()
        self._rew_map = self._construct_rew_func()
        self._obs_map = self._construct_obs_func()

    @property
    def reward_ranges(self) -> Dict[M.AgentID, Tuple[SupportsFloat, SupportsFloat]]:
        return {i: (self.OPEN_BAD_R, self.OPEN_BAD_R) for i in self.possible_agents}

    @property
    def rng(self) -> seeding.RNG:
        if self._rng is None:
            self._rng, seed = seeding.std_random()
        return self._rng

    def get_agents(self, state: MATState) -> List[M.AgentID]:
        return list(self.possible_agents)

    def sample_initial_state(self) -> MATState:
        return self.rng.choice(STATES)

    def step(
        self, state: MATState, actions: Dict[M.AgentID, MATAction]
    ) -> M.JointTimestep[MATState, MATObs]:
        assert all(a_i in ACTIONS for a_i in actions.values())
        next_state = self._sample_next_state(state, actions)
        obs = self._sample_obs(state, actions)
        rewards = self._get_reward(state, actions)
        terminated = {i: False for i in self.possible_agents}
        truncated = {i: False for i in self.possible_agents}
        all_done = False
        info: Dict[M.AgentID, Dict] = {i: {} for i in self.possible_agents}
        return M.JointTimestep(
            next_state, obs, rewards, terminated, truncated, all_done, info
        )

    def _sample_next_state(
        self, state: MATState, actions: Dict[M.AgentID, MATAction]
    ) -> MATState:
        next_state = state
        if any(a != LISTEN for a in actions.values()):
            next_state = self.rng.choice(STATES)
        return next_state

    def _sample_obs(
        self, state: MATState, actions: Dict[M.AgentID, MATAction]
    ) -> Dict[M.AgentID, MATObs]:
        obs: Dict[M.AgentID, MATObs] = {}
        for i, a in actions.items():
            if a != LISTEN:
                obs[i] = self.rng.choice(OBS_SPACE)
                continue

            other_agent_id = (int(i) + 1) % len(self.possible_agents)
            other_agent_action = actions[other_agent_id]

            tiger_obs = self._sample_tiger_obs(state)
            creak_obs = self._sample_creak_obs(other_agent_action)
            obs[i] = (tiger_obs, creak_obs)
        return obs

    def _sample_tiger_obs(self, state: MATState) -> int:
        if self.rng.random() < self._obs_prob:
            # correct obs
            return GROWLLEFT if state == TLEFT else GROWLRIGHT
        return GROWLRIGHT if state == TLEFT else GROWLLEFT

    def _sample_creak_obs(self, a_j: MATAction) -> int:
        if self.rng.random() < self._creak_obs_prob:
            # correct obs
            if a_j == LISTEN:
                return SILENCE
            return CREAKLEFT if a_j == OPENLEFT else CREAKRIGHT

        if a_j == LISTEN:
            return self.rng.choice([CREAKLEFT, CREAKRIGHT])
        if a_j == OPENLEFT:
            return self.rng.choice([CREAKRIGHT, SILENCE])
        return self.rng.choice([CREAKRIGHT, SILENCE])

    def _get_reward(
        self, state: MATState, actions: Dict[M.AgentID, MATAction]
    ) -> Dict[M.AgentID, SupportsFloat]:
        rewards: Dict[M.AgentID, SupportsFloat] = {}
        for i, a in actions.items():
            if a == LISTEN:
                rewards[i] = self.LISTEN_R
            elif a == state:
                rewards[i] = self.OPEN_BAD_R
            else:
                rewards[i] = self.OPEN_GOOD_R
        return rewards

    def get_initial_belief(self) -> Dict[MATState, float]:
        b_map: Dict[MATState, float] = {}
        for s in STATES:
            s_prob = 1.0 / len(STATES)
            b_map[s] = s_prob
        return b_map

    def transition_fn(
        self, state: MATState, actions: Dict[M.AgentID, MATAction], next_state: MATState
    ) -> float:
        action_tuple = tuple(actions[i] for i in self.possible_agents)
        return self._trans_map[(state, action_tuple, next_state)]

    def _construct_trans_func(self) -> Dict:
        trans_map = {}
        uniform_prob = 1.0 / len(STATES)
        for (s, a, s_next) in product(
            self._state_space, product(*self._action_spaces), self._state_space
        ):
            if any(a_i != LISTEN for a_i in a):
                p = uniform_prob
            else:
                p = float(s_next == s)
            trans_map[(s, a, s_next)] = p
        return trans_map

    def observation_fn(
        self,
        obs: Dict[M.AgentID, MATObs],
        next_state: MATState,
        actions: Dict[M.AgentID, MATAction],
    ) -> float:
        obs_tuple = tuple(obs[i] for i in self.possible_agents)
        action_tuple = tuple(actions[i] for i in self.possible_agents)
        return self._obs_map[(next_state, action_tuple, obs_tuple)]

    def _construct_obs_func(self) -> Dict:
        obs_func = {}
        uniform_o_prob = 1.0 / len(OBS_SPACE)
        for (s_next, a, o) in product(
            self._state_space, product(*self._action_spaces), product(*self._obs_spaces)
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
                        o_prob *= 1.0 - self._obs_prob

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
                    o_prob *= 1.0 - self._obs_prob

                if o_k[1] == correct_creak:
                    o_prob *= self._creak_obs_prob
                else:
                    o_prob *= (1.0 - self._creak_obs_prob) / 2

            else:
                o_prob *= uniform_o_prob * uniform_o_prob

            obs_func[(s_next, a, o)] = o_prob

        return obs_func

    def reward_fn(
        self, state: MATState, actions: Dict[M.AgentID, MATAction]
    ) -> Dict[M.AgentID, SupportsFloat]:
        action_tuple = tuple(actions[i] for i in self.possible_agents)
        return self._rew_map[(state, action_tuple)]

    def _construct_rew_func(self) -> Dict:
        rew_map = {}
        joint_actions_space = product(*self._action_spaces)
        for (s, a) in product(self._state_space, joint_actions_space):
            rew_map[(s, a)] = self._get_reward(s, dict(enumerate(a)))
        return rew_map
