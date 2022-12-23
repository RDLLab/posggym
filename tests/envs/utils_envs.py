"""posggym test environments.

Ref:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/envs/utils_envs.py

"""
# from gymnasium import spaces

import posggym


class RegisterDuringMakeEnv(posggym.Env):
    """For `test_registration.py` to check `env.make` can import and register env."""

    def step(self, actions):
        return {}, {}, {}, {}, True, {}

    def reset(self, *, seed=None, options=None):
        return None, {}

    @property
    def model(self):
        return None

    @property
    def state(self):
        return None


class ArgumentEnv(posggym.Env):
    """For `test_registration.py` to check `env.make` can import and register env."""

    def __init__(self, arg1, arg2, arg3):
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3

    def step(self, actions):
        return {}, {}, {}, {}, True, {}

    def reset(self, *, seed=None, options=None):
        return None, {}

    @property
    def model(self):
        return None

    @property
    def state(self):
        return None


# Environments to test render_mode
class NoHuman(posggym.Env):
    """Environment that does not have human-rendering."""

    metadata = {"render_modes": ["rgb_array_list"], "render_fps": 4}

    def __init__(self, render_mode=None):
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def step(self, actions):
        return {}, {}, {}, {}, True, {}

    def reset(self, *, seed=None, options=None):
        return None, {}

    @property
    def model(self):
        return None

    @property
    def state(self):
        return None


class NoHumanNoRGB(posggym.Env):
    """Environment that has neither human- nor rgb-rendering."""

    metadata = {"render_modes": ["ascii"], "render_fps": 4}

    def __init__(self, render_mode=None):
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def step(self, actions):
        return {}, {}, {}, {}, True, {}

    def reset(self, *, seed=None, options=None):
        return None, {}

    @property
    def model(self):
        return None

    @property
    def state(self):
        return None
