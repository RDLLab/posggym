"""A passive environment checker wrapper for an environment."""
from typing import Dict

import posggym
import posggym.model as M
from posggym.utils.passive_env_checker import (
    check_agent_action_spaces,
    check_agent_observation_spaces,
    env_render_passive_checker,
    env_reset_passive_checker,
    env_step_passive_checker,
)


class PassiveEnvChecker(posggym.Wrapper):
    """A passive environment checker wrapper.

    Surrounds the step, reset and render functions to check that they follow the
    posggym environment and model APIs.

    Arguments
    ---------
    env : posggym.Env
        The environment to apply the wrapper

    Note
    ----
    This implementation is based on the similar Gymnasium wrapper:
    https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/wrappers/env_checker.py

    """

    def __init__(self, env: posggym.Env):
        super().__init__(env)

        assert hasattr(env, "model"), "The environment must specify a model."

        assert hasattr(
            env, "action_spaces"
        ), "The environment must specify agent action spaces."
        check_agent_action_spaces(env.action_spaces)
        assert hasattr(
            env, "observation_spaces"
        ), "The environment must specify agent observation spaces."
        check_agent_observation_spaces(env.observation_spaces)

        self.checked_reset = False
        self.checked_step = False
        self.checked_render = False

    def step(self, actions: Dict[str, M.ActType]):
        """Steps through the environment.

        On the first call will run the `passive_env_step_check`.
        """
        if self.checked_step is False:
            self.checked_step = True
            return env_step_passive_checker(self.env, actions)
        else:
            return self.env.step(actions)  # type: ignore

    def reset(self, **kwargs):
        """Resets the environment.

        On the first call will run the `passive_env_reset_check`.

        """
        if self.checked_reset is False:
            self.checked_reset = True
            return env_reset_passive_checker(self.env, **kwargs)
        else:
            return self.env.reset(**kwargs)

    def render(self, *args, **kwargs):
        """Renders the environment.

        On the first call will run the `passive_env_render_check`.
        """
        if self.checked_render is False:
            self.checked_render = True
            return env_render_passive_checker(self.env, *args, **kwargs)
        else:
            return self.env.render(*args, **kwargs)
