"""Wrapper to enforce the proper ordering of environment operations."""
import posggym
from posggym.error import ResetNeeded


class OrderEnforcing(posggym.Wrapper):
    """Wraps environment to enforce environment is reset before stepped.

    Will produce an error if :meth:`step` is called before an initial :meth:`reset`.

    Ref:
    https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/wrappers/order_enforcing.py
    """

    def __init__(self, env: posggym.Env, disable_render_order_enforcing: bool = False):
        super().__init__(env)
        self._has_reset = False
        self._disable_render_order_enforcing = disable_render_order_enforcing

    def step(self, actions):
        if not self._has_reset:
            raise ResetNeeded("Cannot call env.step() before calling env.reset()")
        return self.env.step(actions)

    def reset(self, **kwargs):
        self._has_reset = True
        return self.env.reset(**kwargs)

    def render(self):
        if not self._disable_render_order_enforcing and not self._has_reset:
            raise ResetNeeded(
                "Cannot call `env.render()` before calling `env.reset()`, if this is an"
                " intended action, set `disable_render_order_enforcing=True` on the "
                "OrderEnforcer wrapper."
            )
        return self.env.render()

    @property
    def has_reset(self) -> bool:
        """Returns if the environment has been reset before."""
        return self._has_reset
