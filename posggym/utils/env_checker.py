"""A set of functions for checking an environment's details.

Based on module in the Gymnasium repository:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/gymnasium/utils/env_checker.py

Which in turn is based on work from:

- Stable Baselines3 repository hosted on GitHub
  (https://github.com/DLR-RM/stable-baselines3/). Original Author: Antonin Raffin.
- warnings/assertions from the PettingZoo repository hosted on GitHub
  (https://github.com/PettingZoo-Team/PettingZoo). Original Author: J K Terry/.

These projects are covered by the MIT License.

"""
import inspect
from copy import deepcopy

import posggym
from posggym import error, logger
from posggym.utils.passive_env_checker import (
    check_rng_equality,
    data_equivalence,
    check_agent_action_spaces,
    check_agent_observation_spaces,
    check_agent_space_limits,
    check_reset_obs,
    env_render_passive_checker,
    env_reset_passive_checker,
    env_step_passive_checker,
)


def check_reset_seed(env: posggym.Env):
    """Check that the environment can be reset with a seed.

    Arguments
    ---------
    env: The environment to check

    Raises
    ------
    AssertionError: The environment cannot be reset with a random seed,
        even though `seed` or `kwargs` appear in the signature.

    """
    signature = inspect.signature(env.reset)
    if "seed" in signature.parameters or (
        "kwargs" in signature.parameters
        and signature.parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    ):
        try:
            obs_1, info = env.reset(seed=123)
            check_reset_obs(obs_1, env.model)

            assert (
                env.unwrapped.model._rng  # pyright: ignore [reportPrivateUsage]
                is not None
            ), (
                "Expects the random number generator to have been generated given a "
                "seed was passed to reset. Mostly likely the environment reset function"
                " does not call `super().reset(seed=seed)`."
            )
            seed_123_rng = deepcopy(
                env.unwrapped.model._rng  # pyright: ignore [reportPrivateUsage]
            )

            obs_2, info = env.reset(seed=123)
            check_reset_obs(obs_2, env.model)

            if env.spec is not None and env.spec.nondeterministic is False:
                assert data_equivalence(obs_1, obs_2), (
                    "Using `env.reset(seed=123)` is non-deterministic as the "
                    "observations are not equivalent."
                )
            check_rng_equality(
                env.unwrapped.model._rng,  # pyright: ignore [reportPrivateUsage]
                seed_123_rng,
                prefix=(
                    "Mostly likely the environment reset function does not call "
                    "`super().reset(seed=seed)` as the random generators are not same "
                    "when the same seeds are passed to `env.reset`. "
                ),
            )

            obs_3, info = env.reset(seed=456)
            check_reset_obs(obs_3, env.model)
            try:
                check_rng_equality(
                    env.unwrapped.model._rng,  # pyright: ignore [reportPrivateUsage]
                    seed_123_rng,
                    prefix="",
                )
            except AssertionError:
                pass
            else:
                raise AssertionError(
                    "Mostly likely the environment reset function does not call "
                    "`super().reset(seed=seed)` as the random number generators are not"
                    " different when different seeds are passed to `env.reset`."
                )

        except TypeError as e:
            raise AssertionError(
                "The environment cannot be reset with a random seed, even though "
                "`seed` or `kwargs` appear in the signature. This should never happen, "
                f"please report this issue. The error was: {e}"
            ) from e

        seed_param = signature.parameters.get("seed")
        # Check the default value is None
        if seed_param is not None and seed_param.default is not None:
            logger.warn(
                "The default seed argument in reset should be `None`, otherwise the "
                "environment will by default always be deterministic. "
                f"Actual default: {seed_param.default}"
            )
    else:
        raise error.Error(
            "The `reset` method does not provide a `seed` or `**kwargs` keyword "
            "argument."
        )


def check_reset_options(env: posggym.Env):
    """Check that the environment can be reset with options.

    Arguments
    ---------
    env: The environment to check

    Raises
    ------
    AssertionError: The environment cannot be reset with options, even though
        `options` or `kwargs` appear in the signature.

    """
    signature = inspect.signature(env.reset)
    if "options" in signature.parameters or (
        "kwargs" in signature.parameters
        and signature.parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    ):
        try:
            env.reset(options={})
        except TypeError as e:
            raise AssertionError(
                "The environment cannot be reset with options, even though `options` "
                "or `**kwargs` appear in the signature. This should never happen, "
                "please report this issue. The error was: {e}"
            ) from e
    else:
        raise error.Error(
            "The `reset` method does not provide an `options` or `**kwargs` keyword "
            "argument."
        )


def check_reset_return_type(env: posggym.Env):
    """Checks that :meth:`reset` correctly returns a tuple of the form `(obs , info)`.

    Arguments
    ---------
    env: The environment to check

    Raises
    ------
    AssertionError depending on spec violation

    """
    result = env.reset()
    assert isinstance(result, tuple), (
        "The result returned by `env.reset()` was not a tuple of the form "
        "`(obs, info)`, where `obs` is a observation and `info` is a dictionary "
        f"containing additional information. Actual type: `{type(result)}`"
    )
    assert (
        len(result) == 2
    ), f"Calling reset method did not return a 2-tuple, actual length: {len(result)}"

    obs, info = result
    check_reset_obs(obs, env.model)
    assert isinstance(info, dict), (
        "The second element returned by `env.reset()` was not a dictionary, "
        f"actual type: {type(info)}"
    )


def check_env(env: posggym.Env, skip_render_check: bool = False):
    """Check that an environment follows posggym API.

    This is an invasive function that calls the environment's reset and step.

    This is particularly useful when using a custom environment.

    TODO Update links

    Please take a look at https://gymnasium.farama.org/content/environment_creation/
    for more information about the API.

    Arguments
    ---------
    env: The posggym environment that will be checked
    skip_render_check: Whether to skip the checks for the render method.
        True by default (useful for the CI)

    """
    more_info_msg = "See COMING SOON for more info."

    assert isinstance(
        env, posggym.Env
    ), f"The environment must inherit from the posggym.Env class. {more_info_msg}"

    if env.unwrapped is not env:
        logger.warn(
            f"The environment ({env}) is different from the unwrapped version "
            f"({env.unwrapped}). This could effect the environment checker as the "
            "environment most likely has a wrapper applied to it. We recommend using "
            "the raw environment for `check_env` using `env.unwrapped`."
        )

    # ============= Check the spaces (observation and action) ================
    assert hasattr(
        env, "action_spaces"
    ), f"The environment must specify agent action spaces. {more_info_msg}"
    check_agent_action_spaces(env.action_spaces)
    check_agent_space_limits(env.action_spaces, "action")

    assert hasattr(
        env, "observation_spaces"
    ), f"The environment must specify agent observation spaces. {more_info_msg}"
    check_agent_observation_spaces(env.observation_spaces)
    check_agent_space_limits(env.observation_spaces, "observation")

    # ==== Check the reset method ====
    check_reset_return_type(env)
    check_reset_seed(env)
    check_reset_options(env)

    # ============ Check the returned values ===============
    env_reset_passive_checker(env)
    env_step_passive_checker(
        env, {i: env.action_spaces[i].sample() for i in env.agents}
    )

    # ==== Check the render method and the declared render modes ====
    if not skip_render_check:
        if env.render_mode is not None:
            env_render_passive_checker(env)

        for render_mode in env.metadata["render_modes"]:
            assert env.spec is not None
            new_env = env.spec.make(render_mode=render_mode)
            new_env.reset()
            if not new_env.observation_first:
                new_env.step(
                    {i: new_env.action_spaces[i].sample() for i in new_env.agents}
                )
            env_render_passive_checker(new_env)
            new_env.close()
