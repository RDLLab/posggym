"""The environment class for the Multi-Agent Tiger Problem."""
import sys
import copy
from typing import Optional, Tuple

from posggym import core
import posggym.model as M

import posggym.envs.classic.rps.model as rps_model


class RockPaperScissorsEnv(core.Env):
    """The Rock Paper Scissors Environment.

    This is the classic game of rock, paper, scissors (RPS).

    This scenario involves two agents. Each step both agents choose an action
    out of 'ROCK', 'PAPER' or 'SCISSORS' and are rewarded based on the actions
    taken in comparison to their opponent.

    State
    -----
    There is only a single state in RPS, the None state.

    Actions
    -------
    A_1 = A_2 = {`ROCK`, `PAPER`, `SCISSORS`}

    Observation
    -----------
    Agents observe the last action played by their opponent.

    O_1 = O_2 = {`ROCK`, `PAPER`, `SCISSORS`}

    Reward
    ------
    Agents are rewarded based on the following pay-off matrix (shows pay-off
    for row agent):

             | ROCK     | PAPER    | SCISSORS |
    -------------------------------------------
    ROCK     | 0        | -1       | 1        |
    PAPER    | 1        | 0        | -1       |
    SCISSORS | -1       | 1        | 0        |

    Transition Dynamics
    -------------------
    There is only a single state so the transition function is the identity
    function.

    """

    metadata = {"render.modes": ['human']}

    def __init__(self, **kwargs):
        self._model = rps_model.RockPaperScissorsModel(**kwargs)

        init_conds = self._model.sample_initial_state_and_obs()
        self._state, self._last_obs = init_conds
        self._step_num = 0
        self._last_actions: Optional[rps_model.RPSJointAction] = None
        self._last_rewards: Optional[M.JointReward] = None

    def step(self,
             actions: M.JointAction
             ) -> Tuple[M.JointObservation, M.JointReward, bool, dict]:
        step = self._model.step(self._state, actions)
        self._step_num += 1
        self._state = step.state
        self._last_obs = step.observations
        self._last_actions = actions
        self._last_rewards = step.rewards
        aux = {"outcomes": step.outcomes}
        return (step.observations, step.rewards, step.done, aux)

    def reset(self, *, seed: Optional[int] = None) -> M.JointObservation:
        if seed is not None:
            self._model.set_seed(seed)

        init_conds = self._model.sample_initial_state_and_obs()
        self._state, self._last_obs = init_conds
        self._last_actions = None
        self._last_rewards = None
        self._step_num = 0
        return self._last_obs

    def render(self, mode: str = "human") -> None:
        outfile = sys.stdout

        obs_str = ", ".join([
            rps_model.OBS_STR[o] for o in self._last_obs
        ])
        output = [
            f"Step: {self._step_num}",
            f"Obs: <{obs_str}>"
        ]
        if self._last_actions is not None:
            action_str = ", ".join([
                rps_model.ACTION_STR[a] for a in self._last_actions
            ])
            output.insert(1, f"Actions: <{action_str}>")
            output.append(f"Rewards: <{self._last_rewards}>")

        outfile.write("\n".join(output) + "\n")

    @property
    def model(self) -> M.POSGModel:
        return self._model

    @property
    def state(self) -> M.State:
        return copy.copy(self._state)
