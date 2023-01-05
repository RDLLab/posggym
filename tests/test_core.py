"""Checks that the core posggym environment API is implemented as expected."""

from posggym.core import DefaultEnv

from tests.test_model import ExampleModel


class ExampleEnv(DefaultEnv[int, int, int]):
    """Example testing environment."""

    def __init__(self):
        """Constructor for example environment."""
        self.model = ExampleModel()


def test_posggym_env():
    """Tests general posggym environment API."""
    env = ExampleEnv()

    assert env.metadata == {"render_modes": []}
    assert env.render_mode is None
    assert env.spec is None
