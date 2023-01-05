"""Functions and classes for registering and loading implemented environments.

Based on Farama Foundation Gymnasium, copied and adapted here to avoid issues with
gymnasiums global registry and difference in naming format.
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/envs/registration.py

"""
from __future__ import annotations

import contextlib
import copy
import difflib
import importlib
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Tuple

from posggym import error, logger
from posggym.core import Env
from posggym.wrappers import OrderEnforcing, TimeLimit, PassiveEnvChecker


if sys.version_info < (3, 10):
    import importlib_metadata as metadata  # type: ignore
else:
    import importlib.metadata as metadata


# [namespace/](env-name)-v(version)
ENV_ID_RE = re.compile(
    r"^(?:(?P<namespace>[\w:-]+)\/)?(?:(?P<name>[\w:.-]+?))(?:-v(?P<version>\d+))?$"
)


def load(name: str) -> Callable:
    """Loads environment with name and returns an environment creation function.

    Arguments
    ---------
    name: The environment name

    Returns
    -------
    entry_point: Environment creation function.

    """
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def parse_env_id(env_id: str) -> Tuple[str | None, str, int | None]:
    """Parse environment ID string format.

    [namespace/](env-name)-v(version)    env-name is group 1, version is group 2

    Arguments
    ---------
    env_id: The environment id to parse

    Returns
    -------
    ns: The environment namespace
    name: The environment name
    version: The environment version

    Raises
    ------
    Error: If the environment id does not a valid environment regex

    """
    match = ENV_ID_RE.fullmatch(env_id)
    if not match:
        raise error.Error(
            f"Malformed environment ID: {env_id}. (Currently all IDs must be of the "
            "form [namespace/](env-name)-v(version) (namespace is optional))."
        )
    namespace, name, version = match.group("namespace", "name", "version")
    if version is not None:
        version = int(version)

    return namespace, name, version


def get_env_id(ns: str | None, name: str, version: int | None) -> str:
    """Get the full env ID given a name and (optional) version and namespace.

    Inverse of :meth:`parse_env_id`.

    Arguments
    ---------
    ns: The environment namespace
    name: The environment name
    version: The environment version

    Returns
    -------
    env_id: The environment id

    """
    full_name = name
    if version is not None:
        full_name += f"-v{version}"
    if ns is not None:
        full_name = ns + "/" + full_name
    return full_name


@dataclass
class EnvSpec:
    """A specification for creating environments with posggym.make.

    Attributes
    ----------
    id: The official environment ID.
    entry_point: Python entrypoint of the environment class (e.g. module.name:Class).
    reward_threshold: The reward threshold before the task is considered solved.
    nondeterministic: Whether this environment is non-deterministic even after seeding.
    max_episode_steps: The maximum number of steps that an episode can take before.
        truncation.
    order_enforce: Whether to wrap the environment in an orderEnforcing wrapper, that
        enforces the order of of `reset` before `step` and `render` functions.
    disable_env_checker: Whether to disable the environment checker wrapper in
        `posggym.make`, by default False (runs the environment checker)
    kwargs: Additional kwargs to pass to the environment class

    """

    id: str
    entry_point: Callable | str

    # Environment attributes
    reward_threshold: float | None = field(default=None)
    nondeterministic: bool = field(default=False)

    # Wrappers
    max_episode_steps: int | None = field(default=None)
    order_enforce: bool = field(default=True)
    disable_env_checker: bool = field(default=False)

    # Environment Arguments
    kwargs: Dict = field(default_factory=dict)

    # post-init attributes
    namespace: str | None = field(init=False)
    name: str = field(init=False)
    version: int | None = field(init=False)

    def __post_init__(self):
        """Extract the namespace, name and version from id.

        Is called after spec is created.
        """
        # Initialize namespace, name, version
        self.namespace, self.name, self.version = parse_env_id(self.id)

    def make(self, **kwargs: Any) -> Env:
        """Calls ``make`` using the environment spec and any keyword arguments."""
        # For compatibility purposes
        return make(self, **kwargs)


def _check_namespace_exists(ns: str | None):
    """Check if a namespace exists. If it doesn't, print a helpful error message."""
    if ns is None:
        return
    namespaces = {
        spec_.namespace for spec_ in registry.values() if spec_.namespace is not None
    }
    if ns in namespaces:
        return

    suggestion = (
        difflib.get_close_matches(ns, namespaces, n=1) if len(namespaces) > 0 else None
    )
    suggestion_msg = (
        f"Did you mean: `{suggestion[0]}`?"
        if suggestion
        else f"Have you installed the proper package for {ns}?"
    )

    raise error.NamespaceNotFound(f"Namespace {ns} not found. {suggestion_msg}")


def _check_name_exists(ns: str | None, name: str):
    """Check if env exists in namespace. If it doesn't, print helpful error message."""
    _check_namespace_exists(ns)
    names = {
        spec_.name.lower(): spec_.name
        for spec_ in registry.values()
        if spec_.namespace == ns
    }

    if name in names.values():
        return

    suggestion = difflib.get_close_matches(name.lower(), names, n=1)
    namespace_msg = f" in namespace {ns}" if ns else ""
    suggestion_msg = f"Did you mean: `{names[suggestion[0]]}`?" if suggestion else ""

    raise error.NameNotFound(
        f"Environment {name} doesn't exist{namespace_msg}. {suggestion_msg}"
    )


def _check_version_exists(ns: str | None, name: str, version: int | None):
    """Check if env version exists in namespace. Print helpful error message if not.

    This is a complete test whether an environment identifier is valid, and will
    provide the best available hints.

    Arguments
    ---------
    ns: The environment namespace
    name: The environment space
    version: The environment version

    Raises
    ------
    DeprecatedEnv: The environment doesn't exist but a default version does or the
        environment version is deprecated
    VersionNotFound: The ``version`` used doesn't exist

    """
    if get_env_id(ns, name, version) in registry:
        return

    _check_name_exists(ns, name)
    if version is None:
        return

    message = (
        f"Environment version `v{version}` for environment "
        f"`{get_env_id(ns, name, None)}` doesn't exist."
    )

    env_specs = [
        spec_
        for spec_ in registry.values()
        if spec_.namespace == ns and spec_.name == name
    ]
    env_specs = sorted(env_specs, key=lambda spec_: int(spec_.version or -1))

    default_spec = [spec_ for spec_ in env_specs if spec_.version is None]

    if default_spec:
        message += f" It provides the default version {default_spec[0].id}`."
        if len(env_specs) == 1:
            raise error.DeprecatedEnv(message)

    # Process possible versioned environments
    versioned_specs = [spec_ for spec_ in env_specs if spec_.version is not None]

    latest_spec = max(
        versioned_specs, key=lambda spec: spec.version, default=None  # type: ignore
    )
    if latest_spec is None or latest_spec.version is None:
        return

    if version > latest_spec.version:
        version_list_msg = ", ".join(f"`v{spec_.version}`" for spec_ in env_specs)
        message += f" It provides versioned environments: [ {version_list_msg} ]."
        raise error.VersionNotFound(message)

    if version < latest_spec.version:
        raise error.DeprecatedEnv(
            f"Environment version v{version} for `{get_env_id(ns, name, None)}` "
            f"is deprecated. Please use `{latest_spec.id}` instead."
        )


def find_highest_version(ns: str | None, name: str) -> int | None:
    """Finds the highest registered version of the environment in the registry."""
    version: list[int] = [
        spec_.version
        for spec_ in registry.values()
        if spec_.namespace == ns and spec_.name == name and spec_.version is not None
    ]
    return max(version, default=None)


# Global registry of environments. Meant to be accessed through `register` and `make`
registry: dict[str, EnvSpec] = {}
current_namespace: str | None = None


def _check_spec_register(spec: EnvSpec):
    """Checks whether spec is valid to be registered.

    Helper function for `register`.
    """
    global registry
    latest_versioned_spec = max(
        (
            spec_
            for spec_ in registry.values()
            if spec_.namespace == spec.namespace
            and spec_.name == spec.name
            and spec_.version is not None
        ),
        key=lambda spec_: int(spec_.version),  # type: ignore
        default=None,
    )

    unversioned_spec = next(
        (
            spec_
            for spec_ in registry.values()
            if spec_.namespace == spec.namespace
            and spec_.name == spec.name
            and spec_.version is None
        ),
        None,
    )

    if unversioned_spec is not None and spec.version is not None:
        raise error.RegistrationError(
            "Can't register the versioned environment "
            f"`{spec.id}` when the unversioned environment "
            f"`{unversioned_spec.id}` of the same name already exists."
        )
    elif latest_versioned_spec is not None and spec.version is None:
        raise error.RegistrationError(
            "Can't register the unversioned environment "
            f"`{spec.id}` when the versioned environment "
            f"`{latest_versioned_spec.id}` of the same name "
            f"already exists. Note: the default behavior is "
            f"that `gym.make` with the unversioned environment "
            f"will return the latest versioned environment"
        )


def _check_metadata(metadata_: Dict):
    """Checks validity of metadata. Printing warnings if it's invalid."""
    if not isinstance(metadata_, dict):
        raise error.InvalidMetadata(
            f"Expect the environment metadata to be dict, actual type: {type(metadata)}"
        )

    render_modes = metadata_.get("render_modes")
    if render_modes is None:
        logger.warn(
            "The environment creator metadata doesn't include `render_modes`, "
            f"contains: {list(metadata_.keys())}"
        )
    elif not isinstance(render_modes, Iterable):
        logger.warn(
            "Expects the environment metadata render_modes to be a Iterable, actual "
            f"type: {type(render_modes)}"
        )


# Public API


@contextlib.contextmanager
def namespace(ns: str):
    """Context manager for modifying the current namespace."""
    global current_namespace
    old_namespace = current_namespace
    current_namespace = ns
    yield
    current_namespace = old_namespace


def register(
    id: str,
    entry_point: Callable | str,
    reward_threshold: float | None = None,
    nondeterministic: bool = False,
    max_episode_steps: int | None = None,
    order_enforce: bool = True,
    disable_env_checker: bool = False,
    **kwargs,
):
    """Register an environment with posggym.

    The `id` parameter corresponds to the name of the environment, with the syntax as
    follows:

        `(namespace)/(env_name)-v(version)` where `namespace` is optional.

    It takes arbitrary keyword arguments, which are passed to the `EnvSpec` constructor.

    Arguments
    ---------
    id: The environment id
    entry_point: The entry point for creating the environment
    reward_threshold: The reward threshold considered to have learnt an environment
    nondeterministic: If the environment is nondeterministic (even with knowledge of
        the initial seed and all actions)
    max_episode_steps: The maximum number of episodes steps before truncation. Used
        by the Time Limit wrapper.
    order_enforce: If to enable the order enforcer wrapper to ensure users run
        functions in the correct order
    disable_env_checker: Whether to disable the environment checker for the environment.
        Recommended to False.
    **kwargs: arbitrary keyword arguments which are passed to the environment
        constructor

    """
    global registry, current_namespace
    ns, name, version = parse_env_id(id)

    ns_id = ns
    if current_namespace is not None:
        if (
            kwargs.get("namespace") is not None
            and kwargs.get("namespace") != current_namespace
        ):
            logger.warn(
                f"Custom namespace `{kwargs.get('namespace')}` is being overridden by "
                f"namespace `{current_namespace}`. If you are developing a plugin you "
                "shouldn't specify a namespace in `register` calls. "
                "The namespace is specified through the entry point package metadata."
            )
        ns_id = current_namespace

    full_id = get_env_id(ns_id, name, version)

    new_spec = EnvSpec(
        id=full_id,
        entry_point=entry_point,
        reward_threshold=reward_threshold,
        nondeterministic=nondeterministic,
        max_episode_steps=max_episode_steps,
        order_enforce=order_enforce,
        disable_env_checker=disable_env_checker,
        **kwargs,
    )
    _check_spec_register(new_spec)
    if new_spec.id in registry:
        logger.warn(f"Overriding environment {new_spec.id} already in registry.")
    registry[new_spec.id] = new_spec


def make(
    id: str | EnvSpec,
    max_episode_steps: int | None = None,
    disable_env_checker: bool | None = None,
    **kwargs,
) -> Env:
    """Create an environment according to the given ID.

    To find all available environments use `posggym.envs.registry.keys()` for all valid
    ids.

    Arguments
    ---------
    id: Name of the environment. Optionally, a module to import can be included,
        eg. 'module:Env-v0'
    max_episode_steps: Maximum length of an episode (TimeLimit wrapper).
    disable_env_checker: Whether to run the env checker, None will default to the
            environment specification `disable_env_checker` (which is by default False,
            running the environment checker), otherwise will run according to this
            parameter (`True` = not run, `False` = run)
    kwargs: Additional arguments to pass to the environment constructor.

    Returns
    -------
    env: An instance of the environment.

    Raises
    ------
    Error: If the ``id`` doesn't exist then an error is raised

    """
    if isinstance(id, EnvSpec):
        spec_ = id
    else:
        module, id = (None, id) if ":" not in id else id.split(":")  # type: ignore
        if module is not None:
            try:
                importlib.import_module(module)
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    f"{e}. Environment registration via importing a module failed. "
                    f"Check whether '{module}' contains env registration and can be "
                    "imported."
                ) from e

        spec_ = registry.get(id)  # type: ignore

        ns, name, version = parse_env_id(id)
        latest_version = find_highest_version(ns, name)
        if (
            version is not None
            and latest_version is not None
            and latest_version > version
        ):
            logger.warn(
                f"The environment {id} is out of date. You should consider "
                f"upgrading to version `v{latest_version}`."
            )

        if version is None and latest_version is not None:
            version = latest_version
            new_env_id = get_env_id(ns, name, version)
            spec_ = registry.get(new_env_id)  # type: ignore
            logger.warn(
                f"Using the latest versioned environment `{new_env_id}` "
                f"instead of the unversioned environment `{id}`."
            )

        if spec_ is None:
            _check_version_exists(ns, name, version)
            raise error.Error(f"No registered env with id: {id}")

    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    if spec_.entry_point is None:
        raise error.Error(f"{spec_.id} registered but entry_point is not specified")
    elif callable(spec_.entry_point):
        env_creator = spec_.entry_point
    else:
        # Assume it's a string
        env_creator = load(spec_.entry_point)

    render_modes = None
    if hasattr(env_creator, "metadata"):
        _check_metadata(env_creator.metadata)
        render_modes = env_creator.metadata.get("render_modes")
    mode = _kwargs.get("render_mode")

    # TODO Johnny
    # Add support for attempting to apply applying HumanRendering/RenderCollection
    # wrappers (see gymnasium.envs.registration:make function)
    if mode is not None and render_modes is not None and mode not in render_modes:
        raise error.UnsupportedMode(
            f"The environment is being initialised with render_mode={mode} "
            f"that is not in the possible render_modes ({render_modes})."
        )

    try:
        env = env_creator(**_kwargs)
    except TypeError as e:
        raise e

    # Copies the environment creation specification and kwargs to add to the
    # environment's specification details
    spec_ = copy.deepcopy(spec_)
    spec_.kwargs = _kwargs
    env.unwrapped.spec = spec_
    env.unwrapped.model.spec = spec_

    # Run the environment checker as the lowest level wrapper
    if disable_env_checker is False or (
        disable_env_checker is None and spec_.disable_env_checker is False
    ):
        env = PassiveEnvChecker(env)

    # Add the order enforcing wrapper
    if spec_.order_enforce:
        env = OrderEnforcing(env)

    # Add the time limit wrapper
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps)
    elif spec_.max_episode_steps is not None:
        env = TimeLimit(env, spec_.max_episode_steps)

    return env


def spec(env_id: str) -> EnvSpec:
    """Retrieve the spec for the given environment from the global registry.

    Arguments
    ---------
    env_id: the environment id.

    Returns
    -------
    spec: the environment spec from the global registry.

    Raises
    ------
    Error: if environment with given ``env_id`` doesn't exist in global registry.

    """
    spec_ = registry.get(env_id)
    if spec_ is None:
        ns, name, version = parse_env_id(env_id)
        _check_version_exists(ns, name, version)
        raise error.Error(f"No registered env with id: {env_id}")
    else:
        assert isinstance(spec_, EnvSpec)
        return spec_


def pprint_registry(
    _registry: dict = registry,
    num_cols: int = 3,
    exclude_namespaces: list[str] | None = None,
    disable_print: bool = False,
) -> str | None:
    """Pretty print the environments in the registry.

    Arguments
    ---------
    _registry: Environment registry to be printed.
    num_cols: Number of columns to arrange environments in, for display.
    exclude_namespaces: Exclude any namespaces from being printed.
    disable_print: Whether to return a string of all the namespaces and environment IDs
        instead of printing it to console.

    Returns
    -------
    return_str: formatted str representation of registry, if ``disable_print=True``,
        otherwise returns ``None``.

    """
    # Defaultdict to store environment names according to namespace.
    namespace_envs = defaultdict(lambda: [])
    max_justify = float("-inf")
    for env in _registry.values():
        namespace, _, _ = parse_env_id(env.id)
        if namespace is None:
            # Since namespace is currently none, use regex to obtain namespace from
            # entrypoints.
            env_entry_point = re.sub(r":\w+", "", env.entry_point)
            e_ep_split = env_entry_point.split(".")
            if len(e_ep_split) >= 3:
                # If namespace is of the format - posggym.envs.env_group.env_name:env_id
                # or posggym.envs.env_group:env_id
                idx = 2
                namespace = e_ep_split[idx]
            elif len(e_ep_split) > 1:
                # If namespace is of the format - posggym.env_group
                idx = 1
                namespace = e_ep_split[idx]
            else:
                # If namespace cannot be found, default to env id.
                namespace = env.id
        namespace_envs[namespace].append(env.id)
        max_justify = max(max_justify, len(env.id))

    # Iterate through each namespace and print environment alphabetically.
    return_str = ""
    for namespace, envs in namespace_envs.items():
        # Ignore namespaces to exclude.
        if exclude_namespaces is not None and namespace in exclude_namespaces:
            continue
        return_str += f"{'=' * 5} {namespace} {'=' * 5}\n"  # Print namespace.
        # Reference: https://stackoverflow.com/a/33464001
        for count, item in enumerate(sorted(envs), 1):
            return_str += (
                item.ljust(max_justify) + " "
            )  # Print column with justification.
            # Once all rows printed, switch to new column.
            if count % num_cols == 0 or count == len(envs):
                return_str = return_str.rstrip(" ") + "\n"
        return_str += "\n"

    if disable_print:
        return return_str

    print(return_str, end="")
    return None
