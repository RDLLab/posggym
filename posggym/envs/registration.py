"""Functions and classes for registering and loading implemented environments.

Is based on the Open AI Gym registration functionality from v0.21.0, copied
here to avoid dependency issues with different version of gym.
"""
import re
import copy
import importlib
import contextlib

from posggym.core import Env
from posggym import error, logger


ENV_ID_RE: re.Pattern = re.compile(r"^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$")


def load(name):
    """Load module."""
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class EnvSpec:
    """A specification for a particular instance of the environment.

    Used to register the parameters for official evaluations.

    Arguments
    ---------
    id (str):
        The official environment ID
    entry_point (Optional[str]):
        The Python entrypoint of the environment class (e.g. module.name:Class)
    reward_threshold (Optional[int]):
        The reward threshold before the task is considered solved
    nondeterministic (bool):
        Whether this environment is non-deterministic even after seeding
    max_episode_steps (Optional[int]):
        The maximum number of steps that an episode can consist of
    order_enforce (Optional[int]):
        Whether to wrap the environment in an orderEnforcing wrapper
    kwargs (dict):
        The kwargs to pass to the environment class

    """

    def __init__(
        self,
        id,
        entry_point=None,
        reward_threshold=None,
        nondeterministic=False,
        max_episode_steps=None,
        order_enforce=True,
        kwargs=None,
    ):
        self.id = id
        self.entry_point = entry_point
        self.reward_threshold = reward_threshold
        self.nondeterministic = nondeterministic
        self.max_episode_steps = max_episode_steps
        self.order_enforce = order_enforce
        self._kwargs = {} if kwargs is None else kwargs

        match = ENV_ID_RE.search(id)
        if not match:
            raise error.Error(
                f"Attempted to register malformed environment ID: {id}. "
                f"(Currently all IDs must be of the form {ENV_ID_RE.pattern}.)"
            )
        self._env_name = match.group(1)

    def make(self, **kwargs) -> Env:
        """Instantiates an instance of environment with appropriate kwargs."""
        if self.entry_point is None:
            raise error.Error(
                f"Attempting to make deprecated env {self.id}. (HINT: is "
                "there a newer registered version of this env?)"
            )

        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)

        if callable(self.entry_point):
            env = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            env = cls(**_kwargs)

        # Make the environment and model aware of which spec they came from.
        spec = copy.deepcopy(self)
        spec._kwargs = _kwargs
        env.unwrapped.spec = spec
        env.unwrapped.model.spec = spec
        if env.spec.max_episode_steps is not None:
            from posggym.wrappers.time_limit import TimeLimit

            env = TimeLimit(env, max_episode_steps=env.spec.max_episode_steps)
        else:
            if self.order_enforce:
                from posggym.wrappers.order_enforcing import OrderEnforcing

                env = OrderEnforcing(env)
        return env

    def __repr__(self):
        return "EnvSpec({})".format(self.id)


class EnvRegistry:
    """Register an env by ID.

    IDs remain stable over time and are guaranteed to resolve to the same
    environment dynamics (or be desupported). The goal is that results on a
    particular environment should always be comparable, and not depend on the
    version of the code that was running.

    """

    def __init__(self):
        self.env_specs = {}
        self._ns = None

    def make(self, path: str, **kwargs) -> Env:
        if len(kwargs) > 0:
            logger.info("Making new env: %s (%s)", path, kwargs)
        else:
            logger.info("Making new env: %s", path)
        spec = self.spec(path)
        env = spec.make(**kwargs)
        return env

    def all(self):
        return self.env_specs.values()

    def spec(self, path: str) -> EnvSpec:
        if ":" in path:
            mod_name, _, id = path.partition(":")
            try:
                importlib.import_module(mod_name)
            except ModuleNotFoundError:
                raise error.Error(
                    f"A module ({mod_name}) was specified for the environment "
                    "but was not found, make sure the package is installed "
                    "with `pip install` before calling `posggym.make()`"
                )
        else:
            id = path

        match = ENV_ID_RE.search(id)
        if not match:
            raise error.Error(
                "Attempted to look up malformed environment ID: "
                f"{id.encode('utf-8')}. (Currently all IDs must be of the form"
                f" {ENV_ID_RE.pattern}.)"
            )

        try:
            return self.env_specs[id]
        except KeyError:
            raise error.UnregisteredEnv(f"No registered env with id: {id}")

    def register(self, id, **kwargs):
        if self._ns is not None:
            if "/" in id:
                namespace, id = id.split("/")
                logger.warn(
                    f"Custom namespace '{namespace}' is being overrode by "
                    "namespace '{self._ns}'. If you are developing a plugin "
                    "you shouldn't specify a namespace in `register` calls. "
                    "The namespace is specified through the entry point key."
                )
            id = f"{self._ns}/{id}"
        if id in self.env_specs:
            logger.warn(f"Overriding environment {id}")
        self.env_specs[id] = EnvSpec(id, **kwargs)

    @contextlib.contextmanager
    def namespace(self, ns):
        self._ns = ns
        yield
        self._ns = None


# Global registry that all implemented environments are added too
# Environments are registered when posggym library is loaded
registry = EnvRegistry()


def register(id, **kwargs):
    """Register an environment with posggym."""
    return registry.register(id, **kwargs)


def make(id, **kwargs) -> Env:
    """Create an environment according to the given ID."""
    return registry.make(id, **kwargs)


def spec(id) -> EnvSpec:
    """Get the specification of the environment with given ID."""
    return registry.spec(id)


@contextlib.contextmanager
def namespace(ns):
    """Namespace context manager.

    Not currently used, but put here for consistency.
    """
    with registry.namespace(ns):
        yield
