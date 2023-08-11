"""Functions and classes for registering and loading implemented agents.

Based on Farama Foundation Gymnasium,
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/envs/registration.py

"""
from __future__ import annotations

import copy
import difflib
import importlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Protocol, Tuple

import posggym
from posggym import error, logger

if TYPE_CHECKING:
    import posggym.model as M
    from posggym.agents.policy import Policy


# [env_id/][env_args_id/](policy_id)-v(version)
# env_id is group 1, env_args_id is group 2. policy_id is group 2, version is group 3
POLICY_ID_RE: re.Pattern = re.compile(
    r"^((?:(?P<env_id>[\w:-]+)\/)?(?:(?P<env_args_id>[\S]+)\/)?)?"
    r"(?:(?P<pi_name>[\w:.-]+?))(?:-v(?P<version>\d+))?$"
)


class PolicyEntryPoint(Protocol):
    """Entry point function for instantiating a new policy instance."""

    def __call__(
        self, model: M.POSGModel, agent_id: str, policy_id: str, **kwargs
    ) -> Policy:
        ...


def load(name: str) -> PolicyEntryPoint:
    """Loads policy with name and returns a policy entry point.

    Arguments
    ---------
    name: The policy name

    Returns
    -------
    entry_point: Policy creation function.

    """
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def parse_policy_id(policy_id: str) -> Tuple[str | None, str | None, str, int | None]:
    """Parse policy ID string format.

    env_id is group 1, env_args_id is group 2. policy_id is group 2, version is group 3

    [env_id/][env_args_id/](policy_id)-v(version)

    Arguments
    ---------
    policy_id: The policy id to parse

    Returns
    -------
    env_id: The environment ID
    env_args_id: ID string of environment arguments
    pi_name: The policy name
    version: The policy version

    Raises
    ------
    Error: If the policy id does not a valid environment regex

    """
    match = POLICY_ID_RE.fullmatch(policy_id)
    if not match:
        raise error.Error(
            f"Malformed policy ID: {policy_id}. Currently all IDs must be of the "
            "form [env_id/][env_args_id/](policy_name)-v(version) (env_id and "
            "env_args_id may be optional, depending on the policy)."
        )
    env_id, env_args_id, pi_name, version = match.group(
        "env_id", "env_args_id", "pi_name", "version"
    )
    if env_args_id is not None and env_id is None:
        raise error.Error(
            f"Malformed policy ID: {policy_id}. env_args_id is only valid if a valid "
            "if a valid env_id if included in the policy ID. Did not find valid env_id "
            f"and got env_args_id: {env_args_id}. Check env_id is correct."
        )

    if version is not None:
        version = int(version)

    return env_id, env_args_id, pi_name, version


def get_policy_id(
    env_id: str | None, env_args_id: str | None, policy_name: str, version: int | None
) -> str:
    """Get the full policy ID given a name and (optional) version and env-name.

    Inverse of :meth:`parse_policy_id`.

    Arguments
    ---------
    env_id: The environment ID
    env_args_id: ID string of environment arguments
    policy_name: The policy name
    version: The policy version

    Returns
    -------
    policy_id: The policy id

    """
    if env_args_id is not None and env_id is None:
        raise error.Error(
            "Cannot create policy ID. Must include env_id if env_args_id is part of "
            "the policy ID."
        )

    full_name = policy_name
    if version is not None:
        full_name += f"-v{version}"
    if env_args_id is not None:
        full_name = env_args_id + "/" + full_name
    if env_id is not None:
        full_name = env_id + "/" + full_name
    return full_name


def get_env_args_id(env_args: Dict[str, Any]) -> str:
    """Get string representation of environment keyword arguments.

    Converts keyword dictionary {k1: v1, k2: v2, k3: v3} into a string:

        k1=v1-k2=v2-k3=v3

    Note we assume keywords are valid python variable names and so do not contain
    any hyphen '-' characters.

    Arguments
    ---------
    env_args: environment keyword arguments

    Returns
    -------
    env_args_id: string representation of the envrinment keyword arguments.

    """
    return ("-".join(f"{k}={v}" for k, v in env_args.items())).replace(" ", "")


@dataclass
class PolicySpec:
    """A specification for a particular agent policy.

    Used to register agent policies that can then be dynamically loaded using
    posggym_agents.make.

    Arguments
    ---------
    policy_name: The name of the policy
    entry_point: The Python entrypoint for initializing an instance of the agent policy.
        Must be a Callable with signature matching `PolicyEntryPoint` or a string
        defining where the entry point function can be imported
        (e.g. module.name:Class).
    version: the policy version
    env_id: Optional ID of the posggym environment that the policy is for.
    env_args: Optional keywords arguments for the environment that the policy is for (if
        it is a environment specific policy). If None then assumes policy can be used
        for the environment with any arguments.
    env_args_id: Optional string representation of the environment keyword arguments. If
        None then will generate an ID from the env_args.
    valid_agent_ids: Optional AgentIDs in environment that policy is compatible with. If
        None then assumes policy can be used for any agent in the environment.
    nondeterministic: Whether this policy is non-deterministic even after seeding.
    kwargs: Additional kwargs, if any, to pass to the agent initializing

    Additional Attributes
    ---------------------
    id: The unique policy identifier made from the env_id, env_args, policy_name, and
        version. Is of the form.
    env_args_id: String representation of the env_args

    """

    policy_name: str
    entry_point: PolicyEntryPoint | str
    version: int | None = field(default=None)

    # Environment attributes
    env_id: str | None = field(default=None)
    env_args: Dict[str, Any] | None = field(default=None)

    # Policy attributes
    valid_agent_ids: List[str] | None = field(default=None)
    nondeterministic: bool = field(default=False)

    # Policy Arguments
    kwargs: Dict = field(default_factory=dict)

    # post-init attributes
    env_args_id: str | None = field(default=None)
    # the unique identifier for the policy spec
    id: str = field(init=False)

    def __post_init__(self):
        """Generate unique spec ID.

        Is called after spec is created.
        """
        if self.env_args is not None and self.env_args_id is None:
            self.env_args_id = get_env_args_id(self.env_args)

        # the unique ID for the policy spec
        self.id = get_policy_id(
            self.env_id, self.env_args_id, self.policy_name, self.version
        )
        # check id is valid
        parse_policy_id(self.id)

        if isinstance(self.valid_agent_ids, list) and len(self.valid_agent_ids) == 0:
            raise error.PolicyRegistrationError(
                f"Invalid PolicySpec for policy with id='{self.id}'. `valid_agent_ids` "
                "must be None or a non-empty list."
            )


def _check_env_id_exists(env_id: str | None, env_args_id: str | None):
    """Check if a env ID exists. If it doesn't, print a helpful error message."""
    if env_id is None:
        return

    env_ids = {spec_.env_id for spec_ in registry.values() if spec_.env_id is not None}
    if env_id not in env_ids:
        suggestion = (
            difflib.get_close_matches(env_id, env_ids, n=1)
            if len(env_ids) > 0
            else None
        )
        suggestion_msg = (
            f"Did you mean: `{suggestion[0]}`?"
            if suggestion
            else f"Have you installed the proper package for {env_id}?"
        )
        raise error.PolicyEnvIDNotFound(
            f"Environment ID {env_id} not found. {suggestion_msg}"
        )

    if env_args_id is None:
        return
    env_args_ids = {
        spec_.env_args_id
        for spec_ in registry.values()
        if (spec_.env_id == env_id and spec_.env_args_id is not None)
    }
    if env_args_id not in env_args_ids:
        suggestion = (
            difflib.get_close_matches(env_args_id, env_args_ids, n=1)
            if len(env_args_ids) > 0
            else None
        )
        suggestion_msg = f"Did you mean: `{suggestion[0]}`?" if suggestion else ""
        raise error.PolicyEnvArgsIDNotFound(
            f"Environment Arguments {env_args_id} for environment ID {env_id} not "
            f"found. {suggestion_msg}"
        )


def _check_name_exists(env_id: str | None, env_args_id: str | None, policy_name: str):
    """Check if policy exists for given env id. If not, print helpful error message."""
    # check if policy_name matches a generic policy
    names = {
        spec_.policy_name.lower(): spec_.policy_name
        for spec_ in registry.values()
        if spec_.env_id is None
    }
    if policy_name in names.values():
        return

    # check env specific policies
    _check_env_id_exists(env_id, env_args_id)
    names = {
        spec_.policy_name.lower(): spec_.policy_name
        for spec_ in registry.values()
        if (
            (spec_.env_id == env_id)
            and (env_args_id is None or spec_.env_args_id == env_args_id)
        )
    }
    if policy_name in names.values():
        return

    suggestion = difflib.get_close_matches(policy_name.lower(), names, n=1)
    env_id_msg = f" for env ID {env_id}" if env_id else ""
    suggestion_msg = f"Did you mean: `{names[suggestion[0]]}`?" if suggestion else ""

    raise error.PolicyNameNotFound(
        f"Policy {policy_name} doesn't exist{env_id_msg}. {suggestion_msg}"
    )


def _check_version_exists(
    env_id: str | None, env_args_id: str | None, policy_name: str, version: int | None
):
    """Check if policy version exists for env ID. Print helpful error message if not.

    This is a complete test whether an policy ID is valid, and will provide the best
    available hints.

    Arguments
    ---------
    env_id: The environment ID
    env_args_ud: The ID of the environment keyword arguments
    policy_name: The policy name
    version: The policy version

    Raises
    ------
    DeprecatedPolicy: The policy doesn't exist but a default version does or the
        policy version is deprecated
    VersionNotFound: The ``version`` used doesn't exist

    """
    if get_policy_id(env_id, env_args_id, policy_name, version) in registry:
        return

    _check_name_exists(env_id, env_args_id, policy_name)
    if version is None:
        return

    message = (
        f"Policy version `v{version}` for policy "
        f"`{get_policy_id(env_id, env_args_id, policy_name, None)}` doesn't exist."
    )

    policy_specs = [
        spec_
        for spec_ in registry.values()
        if (
            spec_.env_id == env_id
            and spec_.env_args_id == env_args_id
            and spec_.policy_name == policy_name
        )
    ]
    policy_specs = sorted(policy_specs, key=lambda spec_: int(spec_.version or -1))

    default_spec = [spec_ for spec_ in policy_specs if spec_.version is None]
    if default_spec:
        message += f" It provides the default version {default_spec[0].id}`."
        if len(policy_specs) == 1:
            raise error.DeprecatedPolicy(message)

    # Process possible versioned environments
    versioned_specs = [spec_ for spec_ in policy_specs if spec_.version is not None]

    latest_spec = max(
        versioned_specs, key=lambda spec: spec.version, default=None  # type: ignore
    )
    if latest_spec is None or latest_spec.version is None:
        return

    if version > latest_spec.version:
        version_list_msg = ", ".join(f"`v{spec_.version}`" for spec_ in policy_specs)
        message += f" It provides versioned policies: [ {version_list_msg} ]."
        raise error.PolicyVersionNotFound(message)

    if version < latest_spec.version:
        raise error.DeprecatedPolicy(
            f"Policy version v{version} for "
            f"`{get_policy_id(env_id, env_args_id, policy_name, None)}` "
            f"is deprecated. Please use `{latest_spec.id}` instead."
        )


def find_highest_version(
    env_id: str | None, env_args_id: str | None, policy_name: str
) -> int | None:
    """Finds the highest registered version of the policy in the registry."""
    version: list[int] = [
        spec_.version
        for spec_ in registry.values()
        if (
            spec_.env_id == env_id
            and spec_.env_args_id == env_args_id
            and spec_.policy_name == policy_name
            and spec_.version is not None
        )
    ]
    return max(version, default=None)


# Global registry of policies. Meant to be accessed through `register` and `make`
registry: dict[str, PolicySpec] = {}


def _check_spec_register(spec: PolicySpec):
    """Checks whether spec is valid to be registered.

    Helper function for `register`.
    """
    global registry
    latest_versioned_spec = max(
        (
            spec_
            for spec_ in registry.values()
            if (
                spec_.env_id == spec.env_id
                and spec_.env_args_id == spec.env_args_id
                and spec_.policy_name == spec.policy_name
                and spec_.version is not None
            )
        ),
        key=lambda spec_: int(spec_.version),  # type: ignore
        default=None,
    )

    unversioned_spec = next(
        (
            spec_
            for spec_ in registry.values()
            if (
                spec_.env_id == spec.env_id
                and spec_.env_args_id == spec.env_args_id
                and spec_.policy_name == spec.policy_name
                and spec_.version is None
            )
        ),
        None,
    )

    if unversioned_spec is not None and spec.version is not None:
        raise error.PolicyRegistrationError(
            "Can't register the versioned policy "
            f"`{spec.id}` when the unversioned policy "
            f"`{unversioned_spec.id}` of the same name already exists."
        )
    elif latest_versioned_spec is not None and spec.version is None:
        raise error.PolicyRegistrationError(
            "Can't register the unversioned policy "
            f"`{spec.id}` when the versioned policy "
            f"`{latest_versioned_spec.id}` of the same name "
            f"already exists. Note: the default behavior is "
            f"that `posggym_agents.make` with the unversioned policy "
            f"will return the latest versioned policy"
        )


def register(
    policy_name: str,
    entry_point: PolicyEntryPoint | str,
    version: int | None = None,
    env_id: str | None = None,
    env_args: Dict[str, Any] | None = None,
    valid_agent_ids: List[str] | None = None,
    nondeterministic: bool = False,
    **kwargs,
):
    """Register a policy with posggym.agents.

    The policy is registered in posggym so it can be used with
    :py:method:`posggym.agents.make`

    Arguments
    ---------
    policy_name: str
      The name of the policy
    entry_point: PolicyEntryPoint | str
      The entry point for creating the policy
    env_id: str, optional
      Optional ID of the posggym environment that the policy is for.
    version: int, optional
      the policy version
    env_args: Dict[str, Any], optional
      Optional keywords arguments for the environment that the policy is for (if
      it is a environment specific policy). If None then assumes policy can be used
      for the environment with any arguments.
    valid_agent_ids: List[str], optional
      Optional AgentIDs in environment that policy is compatible with. If
      None then assumes policy can be used for any agent in the environment.
    nondeterministic: bool
      Whether this policy is non-deterministic even after seeding.
    kwargs:
      Additional kwargs, if any, to pass to the agent initializing

    """
    global registry
    new_spec = PolicySpec(
        policy_name=policy_name,
        entry_point=entry_point,
        version=version,
        env_id=env_id,
        env_args=env_args,
        valid_agent_ids=valid_agent_ids,
        nondeterministic=nondeterministic,
        **kwargs,
    )
    register_spec(new_spec)


def register_spec(spec: PolicySpec):
    """Register a policy spec with posggym-agents.

    Arguments
    ---------
    spec: The policy spec

    """
    global registry
    _check_spec_register(spec)
    if spec.id in registry:
        logger.warn(f"Overriding policy {spec.id} already in registry.")
    registry[spec.id] = spec


def make(id: str | PolicySpec, model: M.POSGModel, agent_id: str, **kwargs) -> Policy:
    """Create an policy according to the given ID.

    To find all available policies use `posggym_agents.agents.registry.keys()` for
    all valid ids.

    Arguments
    ---------
    id: str
      Unique identifier of the policy or a policy spec.
    model: posggym.POSGModel
      The model for the environment the policy will be interacting with.
    agent_id: str
      The ID of the agent the policy will be used for.
    kwargs:
      Additional arguments to pass to the policy constructor.

    Returns
    -------
    policy: posgym.agents.Policy
      An instance of the policy.

    Raises
    ------
    Error:
      If the ``id`` doesn't exist then an error is raised

    """
    if isinstance(id, PolicySpec):
        spec_: PolicySpec = id
    else:
        env_id, env_args_id, policy_name, version = parse_policy_id(id)

        if id not in registry:
            generic_names = {
                spec.policy_name for spec in registry.values() if spec.env_id is None
            }
            if policy_name in generic_names:
                env_id = None
                env_args_id = None

        policy_id = get_policy_id(env_id, env_args_id, policy_name, version)
        spec_ = registry.get(policy_id)  # type: ignore

        latest_version = find_highest_version(env_id, env_args_id, policy_name)
        if (
            version is not None
            and latest_version is not None
            and latest_version > version
        ):
            logger.warn(
                f"The policy {id} is out of date. You should consider "
                f"upgrading to version `v{latest_version}`."
            )

        if version is None and latest_version is not None:
            version = latest_version
            new_policy_id = get_policy_id(env_id, env_args_id, policy_name, version)
            spec_ = registry.get(new_policy_id)  # type: ignore
            logger.warn(
                f"Using the latest versioned policy `{new_policy_id}` "
                f"instead of the unversioned policy `{id}`."
            )

        if spec_ is None:
            _check_version_exists(env_id, env_args_id, policy_name, version)
            raise error.Error(f"No registered policy with id: {id}")

    if spec_.valid_agent_ids and agent_id not in spec_.valid_agent_ids:
        raise error.Error(
            f"Attempted to initialize policy with ID={spec_.id} with invalid "
            f"agent ID '{agent_id}'. Valid agent IDs for this policy are: "
            f"{spec_.valid_agent_ids}."
        )

    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    if spec_.entry_point is None:
        raise error.Error(f"{spec_.id} registered but entry_point is not specified")
    elif callable(spec_.entry_point):
        policy_creator = spec_.entry_point
    else:
        policy_creator = load(spec_.entry_point)

    try:
        policy = policy_creator(model, agent_id, spec_.id, **_kwargs)
    except TypeError as e:
        raise e

    # Copies the policy creation specification and kwargs to add to the
    # policy's specification details
    spec_ = copy.deepcopy(spec_)
    spec_.kwargs = _kwargs
    policy.spec = spec_

    return policy


def spec(id: str) -> PolicySpec:
    """Retrieve the spec for the given policy from the global registry.

    Arguments
    ---------
    id: the policy id.

    Returns
    -------
    spec: the policy spec from the global registry.

    Raises
    ------
    Error: if policy with given ``id`` doesn't exist in global registry.

    """
    spec_ = registry.get(id)
    if spec_ is None:
        env_id, env_args_id, policy_name, version = parse_policy_id(id)
        _check_version_exists(env_id, env_args_id, policy_name, version)

        # No error raised so policy_name-version may be generic
        id = get_policy_id(None, None, policy_name, version)
        _check_version_exists(None, None, policy_name, version)
        spec_ = registry.get(id)

    if spec_ is None:
        raise error.Error(f"No registered policy with id: {id}")
    else:
        assert isinstance(spec_, PolicySpec)
        return spec_


def pprint_registry(
    _registry: dict = registry,
    num_cols: int = 3,
    include_env_ids: List[str] | None = None,
    exclude_env_ids: List[str] | None = None,
    disable_print: bool = False,
) -> str | None:
    """Pretty print the policies in the registry.

    Arguments
    ---------
    _registry: Policy registry to be printed.
    num_cols: Number of columns to arrange policies in, for display.
    include_env_ids: Print only policies for environments with these IDs. If None then
        all environments are included.
    exclude_env_ids: Exclude any policies for environments with thee IDs from being
        printed.
    disable_print: Whether to return a string of all the policy IDs instead of printing
        it to console.

    Returns
    -------
    return_str: formatted str representation of registry, if ``disable_print=True``,
        otherwise returns ``None``.

    """
    # Defaultdict to store policy names according to env_id.
    env_policies = defaultdict(lambda: defaultdict(lambda: []))
    max_justify = 0
    for spec in _registry.values():
        env_id = "Generic" if spec.env_id is None else spec.env_id
        env_policies[env_id][spec.env_args_id].append(
            f"{spec.policy_name}-v{spec.version}"
        )
        max_justify = max(max_justify, len(f"{spec.policy_name}-v{spec.version}"))

    # Iterate through each environment and print policies alphabetically.
    return_str = ""
    for env_id in env_policies:
        if exclude_env_ids is not None and env_id in exclude_env_ids:
            continue
        if include_env_ids is not None and env_id not in include_env_ids:
            continue

        return_str += f"{'=' * 5} {env_id} {'=' * 5}\n"
        for env_args_id, policies in env_policies[env_id].items():
            if env_args_id is not None:
                return_str += f"{'-' * 5} {env_id}/{env_args_id} {'-' * 5}\n"
            # Reference: https://stackoverflow.com/a/33464001
            for count, item in enumerate(sorted(policies), 1):
                return_str += (
                    item.ljust(max_justify) + " "
                )  # Print column with justification.
                # Once all rows printed, switch to new column.
                if count % num_cols == 0 or count == len(policies):
                    return_str = return_str.rstrip(" ") + "\n"
            return_str += "\n"
        return_str += "\n"

    if disable_print:
        return return_str

    print(return_str, end="")
    return None


def get_all_env_policies(
    env_id: str,
    env_args: Dict[str, Any] | str | None = None,
    _registry: Dict = registry,
    include_generic_policies: bool = True,
) -> List[PolicySpec]:
    """Get all PolicySpecs that are associated with a given environment ID.

    Arguments
    ---------
    env_id: The ID of the environment
    env_args: Optional environment arguments or ID string of environment arguments. If
        None, will return all policies for given environment.
    _registry: The policy registry
    include_generic_policies: whether to also return policies that are valid for all
        environments (e.g. the random-v0 policy)

    Returns
    -------
    policy_specs: list of specs for policies associated with given environment.

    """
    return [
        spec
        for spec in _registry.values()
        if (
            (include_generic_policies and spec.env_id is None)
            or (
                spec.env_id == env_id
                and (
                    env_args is None
                    or spec.env_args_id is None
                    or (isinstance(env_args, str) and spec.env_args_id == env_args)
                    or (isinstance(env_args, dict) and spec.env_args == env_args)
                )
            )
        )
    ]


def get_env_agent_policies(
    env_id: str,
    env_args: Dict[str, Any] | None = None,
    _registry: Dict = registry,
    include_generic_policies: bool = True,
) -> Dict[str, List[PolicySpec]]:
    """Get each agent's policy specs associated with given environment.

    Arguments
    ---------
    env_id: The ID of the environment
    env_args: Optional environment arguments. If None, will return all policies for
        given environment.
    _registry: The policy registry
    include_generic_policies: whether to also return policies that are valid for all
        environments (e.g. the random-v0 policy) and environment args

    Returns
    -------
    policy_specs: list of specs for policies associated with given environment.

    """
    env = posggym.make(env_id) if env_args is None else posggym.make(env_id, **env_args)

    policies: Dict[str, List[PolicySpec]] = {i: [] for i in env.possible_agents}
    for spec in get_all_env_policies(
        env_id,
        env_args,
        _registry=_registry,
        include_generic_policies=include_generic_policies,
    ):
        for i in env.possible_agents:
            if spec.valid_agent_ids is None or i in spec.valid_agent_ids:
                policies[i].append(spec)
    return policies


def get_all_envs(
    _registry: Dict = registry,
) -> Dict[str, Dict[str | None, Dict[str, Any] | None]]:
    """Get all the environments that have at least one registered policy.

    Arguments
    ---------
    _registry: The policy registry

    Returns
    -------
    envs: a dictionary with env IDs as keys as list of (env_args, env_args_id) tuples as
        the values.

    """
    envs: Dict[str, Dict[str | None, Dict[str, Any] | None]] = {}
    for spec in _registry.values():
        if spec.env_id is not None:
            envs.setdefault(spec.env_id, {})
            envs[spec.env_id][spec.env_args_id] = spec.env_args
    return envs
