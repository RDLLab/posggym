"""A set of functions for passively checking environment implementations.

Ref:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/utils/passive_env_checker.py

"""
import inspect
from functools import partial
from typing import Callable, Dict, Optional, get_args

import gymnasium.utils.passive_env_checker as gym_passive_env_checker
import numpy as np
from gymnasium import Space, spaces

import posggym
import posggym.model as M
from posggym import error, logger


def check_agent_spaces(
    agent_spaces: Dict[M.AgentID, Space],
    space_type: str,
    check_box_space_fn: Callable[[spaces.Box], None],
):
    """A passive check of environment spaces that should not affect the environment."""
    assert all(isinstance(i, get_args(M.AgentID)) for i in agent_spaces)
    for i, space_i in agent_spaces.items():
        try:
            gym_passive_env_checker.check_space(space_i, space_type, check_box_space_fn)
        except AssertionError as e:
            raise AssertionError(f"Invalid {space_type} space for agent '{i}'.") from e


check_agent_observation_spaces = partial(
    check_agent_spaces,
    space_type="observation",
    check_box_space_fn=gym_passive_env_checker._check_box_observation_space,
)
check_agent_action_spaces = partial(
    check_agent_spaces,
    space_type="action",
    check_box_space_fn=gym_passive_env_checker._check_box_action_space,
)


def check_agent_obs(
    obs: Dict[M.AgentID, M.ObsType],
    observation_spaces: Dict[M.AgentID, Space],
    method_name: str,
):
    """Check that each agent's observation returned by the environment is valid.

    Arguments
    ---------
    obs: The observation for each agent to check
    observation_spaces: The observation spaces for each agent's observation
    method_name: The method name that generated the observation

    """
    for i, obs_i in obs.items():
        try:
            gym_passive_env_checker.check_obs(obs_i, observation_spaces[i], method_name)
        except AssertionError as e:
            raise AssertionError("Invalid observation for agent `{i}`.") from e


def check_reset_obs(obs: Optional[Dict[M.AgentID, M.ObsType]], env: posggym.Env):
    """Check agent observations returned by the environment `reset()` method are valid.

    Arguments
    ---------
    obs: The observation for each agent to check
    env: The environment

    """
    if not env.observation_first:
        assert (
            obs is None
        ), "Expected None from `env.reset()` for `action_first` environment."
        return

    assert isinstance(obs, dict), (
        "Expected observation from `env.reset()` to be a dictionary, mapping agent IDs "
        " to agent obs."
    )
    for i, o_i in obs.items():
        assert (
            i in env.possible_agents
        ), f"Invalid agent ID `{i}`. Possible IDs are {env.possible_agents}."

    check_agent_obs(obs, env.observation_spaces, "reset")


def env_reset_passive_checker(env, **kwargs):
    """A passive check of the `Env.reset` function.

    Investigates the returning reset information and returning the data unchanged.

    """
    signature = inspect.signature(env.reset)
    if "seed" not in signature.parameters and "kwargs" not in signature.parameters:
        logger.warn(
            "posggym requires that `Env.reset` can be passed a `seed` for resetting the"
            " environment random number generator."
        )
    else:
        seed_param = signature.parameters.get("seed")
        # Check the default value is None
        if seed_param is not None and seed_param.default is not None:
            logger.warn(
                "The default seed argument in `Env.reset` should be `None`, otherwise "
                "the environment will by default always be deterministic. "
                f"Actual default: {seed_param}"
            )

    if "options" not in signature.parameters and "kwargs" not in signature.parameters:
        logger.warn(
            "posggym requires that `Env.reset` can be passed `options` to allow the "
            "environment initialisation to be passed additional information."
        )

    # Checks the result of env.reset with kwargs
    result = env.reset(**kwargs)

    if not isinstance(result, tuple):
        logger.warn(
            "The result returned by `env.reset()` was not a tuple of the form "
            "`(obs, info)`, where `obs` is a observation and `info` is a dictionary "
            f"containing additional information. Actual type: `{type(result)}`"
        )
    elif len(result) != 2:
        logger.warn(
            "The result returned by `env.reset()` should be `(obs, info)` by default, "
            "where `obs` is a observation and `info` is a dictionary containing "
            "additional information."
        )
    else:
        obs, info = result
        check_reset_obs(obs, env)
        assert isinstance(info, dict), (
            "The second element returned by `env.reset()` was not a dictionary, "
            f"actual type: {type(info)}"
        )
    return result


def _check_agent_dict(agent_dict, env: posggym.Env, dict_type: str):
    assert isinstance(agent_dict, dict), (
        f"Agent {dict_type} dictionary  must be a dictionary mapping agentID to values."
        f"Actual type: {type(agent_dict)}."
    )
    for i in agent_dict:
        assert i in env.possible_agents, (
            f"Agent {dict_type} dictionary must only contain valid agent IDs: "
            f"invalid ID `{i}`."
        )


def env_step_passive_checker(env: posggym.Env, actions: Dict[M.AgentID, M.ActType]):
    """A passive check for the environment step.

    Investigating the returning data then returning the data unchanged.
    """
    # We don't check the actions as out-of-bounds values are allowed in some
    # environments
    result = env.step(actions)
    assert isinstance(
        result, tuple
    ), f"Expects step result to be a tuple, actual type: {type(result)}"
    if len(result) == 6:
        obs, reward, terminated, truncated, done, info = result

        _check_agent_dict(obs, env, "observation")
        _check_agent_dict(reward, env, "reward")
        _check_agent_dict(terminated, env, "terminated")
        _check_agent_dict(truncated, env, "truncated")
        _check_agent_dict(info, env, "info")

        if not all(isinstance(t_i, (bool, np.bool_)) for t_i in terminated.values()):
            logger.warn(
                "Expects `terminated` signal to be a boolean for every agent, "
                f"actual types: {list(type(t_i) for t_i in terminated.values())}."
            )
        if not all(isinstance(t_i, (bool, np.bool_)) for t_i in truncated.values()):
            logger.warn(
                "Expects `truncated` signal to be a boolean for every agent, "
                f"actual types: {list(type(t_i) for t_i in truncated.values())}."
            )

        if not isinstance(done, (bool, np.bool_)):
            logger.warn(
                "Expects `done` signal returned by `step()` to be a boolean, "
                f"actual type: {type(info)}"
            )
    else:
        raise error.Error(
            "Expected `Env.step` to return a six element tuple, actual number of "
            f"elements returned: {len(result)}."
        )

    check_agent_obs(obs, env.observation_spaces, "step")

    if not (
        all(
            np.issubdtype(type(r_i), np.integer)
            or np.issubdtype(type(r_i), np.floating)
            for r_i in reward.values()
        )
    ):
        logger.warn(
            "The reward returned for each agent by `step()` must be a float, int, "
            "np.integer or np.floating, "
            f"actual types: {list(type(r_i) for r_i in reward.values())}."
        )
    else:
        for i, r_i in reward.items():
            if np.isnan(r_i):  # type: ignore
                logger.warn(f"The reward for agent `{i}` is a NaN value.")
            if np.isinf(r_i):  # type: ignore
                logger.warn(f"The reward for agent `{i}` is an inf value.")

    return result


def _check_render_return(render_mode, render_return):
    """Produces warning if `render_return` doesn't match `render_mode`."""
    gym_passive_env_checker._check_render_return(render_mode, render_return)

    # need to check posggym specific render modes, namely dict renders which
    # return mapping from agentID to agent specific render.
    if render_mode.endswith("_dict"):
        if not isinstance(render_return, dict):
            logger.warn(
                f"Render mode `{render_mode}` should produce a dict, got "
                f"{type(render_return)}"
            )
        else:
            # Check that each item of the dict matches the base render mode
            # Doesn't check dict keys, i.e. agent IDs, are valid
            base_render_mode = render_mode[: -len("_dict")]
            for item in render_return:
                _check_render_return(base_render_mode, item)


def env_render_passive_checker(env: posggym.Env):
    """A passive check of the `Env.render`.

    Checks that render modes/fps in the metadata of the environment are declared.

    This function is exactly the same as
    `gymnasium.utils.passive_env_check.env_render_passive_checker`, we copy it here to
    ensure the updated `_check_render_return` function is used.

    """
    render_modes = env.metadata.get("render_modes")
    if render_modes is None:
        logger.warn(
            "No render modes were declared in the environment "
            "(env.metadata['render_modes'] is None or not defined), you may have "
            "trouble when calling `.render()`."
        )
    else:
        if not isinstance(render_modes, (list, tuple)):
            logger.warn(
                "Expects the render_modes to be a sequence (i.e. list, tuple), "
                f"actual type: {type(render_modes)}"
            )
        elif not all(isinstance(mode, str) for mode in render_modes):
            logger.warn(
                "Expects all render modes to be strings, "
                f"actual types: {[type(mode) for mode in render_modes]}"
            )

        render_fps = env.metadata.get("render_fps")
        # We only require `render_fps` if rendering is actually implemented
        if len(render_modes) > 0:
            if render_fps is None:
                logger.warn(
                    "No render fps was declared in the environment "
                    "(env.metadata['render_fps'] is None or not defined), rendering "
                    "may occur at inconsistent fps."
                )
            else:
                if not (
                    np.issubdtype(type(render_fps), np.integer)
                    or np.issubdtype(type(render_fps), np.floating)
                ):
                    logger.warn(
                        "Expects the `env.metadata['render_fps']` to be an integer "
                        f"or a float, actual type: {type(render_fps)}"
                    )
                else:
                    assert render_fps > 0, (
                        "Expects the `env.metadata['render_fps']` to be greater than "
                        f"zero, actual value: {render_fps}"
                    )

        # env.render is now an attribute with default None
        if len(render_modes) == 0:
            assert env.render_mode is None, (
                "With no render_modes, expects the Env.render_mode to be None, "
                f"actual value: {env.render_mode}"
            )
        else:
            assert env.render_mode is None or env.render_mode in render_modes, (
                "The environment was initialized successfully however with an "
                "unsupported render mode. "
                f"Render mode: {env.render_mode}, modes: {render_modes}"
            )

    result = env.render()
    if env.render_mode is not None:
        _check_render_return(env.render_mode, result)

    return result
