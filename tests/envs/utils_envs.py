"""posggym test environments.

Ref:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/utils_envs.py

"""
import posggym

from tests.envs.utils_models import TestModel


class DummyEnv(posggym.DefaultEnv):
    """Dummy env for use in environment registration and make tests ."""

    def __init__(self):
        super().__init__(TestModel())


class RegisterDuringMakeEnv(posggym.DefaultEnv):
    """For `test_registration.py` to check `env.make` can import and register env."""

    def __init__(self):
        super().__init__(TestModel())


class ArgumentEnv(posggym.DefaultEnv):
    """For `test_registration.py` to check `env.make` can import and register env."""

    def __init__(self, arg1, arg2, arg3):
        super().__init__(TestModel())
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3


# Environments to test render_mode
class NoHuman(posggym.DefaultEnv):
    """Environment that does not have human-rendering."""

    metadata = {"render_modes": ["rgb_array_list"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__(TestModel())
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


class NoHumanNoRGB(posggym.DefaultEnv):
    """Environment that has neither human- nor rgb-rendering."""

    metadata = {"render_modes": ["ascii"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__(TestModel())
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
