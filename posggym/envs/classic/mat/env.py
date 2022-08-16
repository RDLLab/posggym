"""The environment class for the Multi-Agent Tiger Problem."""
import sys

from posggym import core
import posggym.model as M

import posggym.envs.classic.mat.model as mat_model


class MultiAgentTigerEnv(core.DefaultEnv):
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

    metadata = {"render.modes": ['human', 'ansi']}

    def __init__(self,
                 observation_prob: float = 0.85,
                 creak_observation_prob: float = 0.9,
                 **kwargs):
        self._model = mat_model.MultiAgentTigerModel(
            observation_prob, creak_observation_prob, **kwargs
        )
        super().__init__()

    def render(self, mode: str = "human"):
        if mode not in self.metadata["render.modes"]:
            # raise exception
            super().render(mode)

        state_str = mat_model.STATE_STRS[self._state]
        obs_str = ", ".join([
            str((mat_model.OBS_STR[0][o[0]], mat_model.OBS_STR[1][o[1]]))
            for o in self._last_obs
        ])
        output = [
            f"Step: {self._step_num}",
            f"State: <{state_str}>",
            f"Obs: <{obs_str}>"
        ]
        if self._last_actions is not None:
            action_str = ", ".join(
                [mat_model.ACTION_STR[a] for a in self._last_actions]
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
