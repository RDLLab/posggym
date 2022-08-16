import sys

from posggym import core
import posggym.model as M

import posggym.envs.classic.rps.model as rps_model


class RockPaperScissorsEnv(core.DefaultEnv):
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

    metadata = {"render.modes": ['human', 'ansi']}

    def __init__(self, **kwargs):
        self._model = rps_model.RockPaperScissorsModel(**kwargs)
        super().__init__()

    def render(self, mode: str = "human"):
        if mode not in self.metadata["render.modes"]:
            # raise exception
            super().render(mode)

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

        output_str = "\n".join(output) + "\n"

        if mode == "human":
            sys.stdout.write(output_str)
        else:
            # ansi mode
            return output_str

    @property
    def model(self) -> M.POSGModel:
        return self._model
