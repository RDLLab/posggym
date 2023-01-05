"""A set of functions for checking a model's details.

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
from typing import get_args, Optional

import posggym.model as M
from posggym import logger
from posggym.utils.passive_env_checker import (
    check_rng_equality,
    data_equivalence,
    check_agent_action_spaces,
    check_agent_obs,
    check_agent_observation_spaces,
    check_agent_space_limits,
    check_state,
    check_state_space,
    model_step_passive_checker
)


def check_initial_state_type(model: M.POSGModel):
    """Checks that :meth:`sample_initial_state` correctly returns a valid state.

    Arguments
    ---------
    model: The model to check

    Returns
    -------
    state: sampled initial state

    Raises
    ------
    AssertionError depending on spec violation

    """
    state = model.sample_initial_state()
    check_state(state, model)
    return state


def check_initial_obs_type(model: M.POSGModel, state: Optional[M.StateType] = None):
    """Checks that :meth:`sample_initial_obs` works correctly.

    Assumes ``model.sample_initial_state()`` works as expected.

    Arguments
    ---------
    model: the model to check
    state: the state to use for check, default is None in which case a new state is
           sampled from the model.

    Raises
    ------
    AssertionError depending on spec violation

    """
    if state is None:
        state = model.sample_initial_state()

    if model.observation_first:
        try:
            obs = model.sample_initial_obs(state)
            check_agent_obs(obs, model.observation_spaces, "sample_initial_obs")
        except NotImplementedError:
            raise AssertionError(
                "Observation first model requires the ``sample_initial_obs`` method to "
                "be implemented."
            )
    else:
        try:
            model.sample_initial_obs(state)
        except (AssertionError, NotImplementedError):
            pass
        else:
            raise AssertionError(
                "Action first model should raise assertion error when "
                "``sample_initial_obs`` method is used."
            )


def check_initial_sampling_seed(model: M.POSGModel):
    """Check that model seeding works correctly for initial conditions.

    Arguments
    ---------
    model: The environment model to check

    Raises
    ------
    AssertionError: The model random seeding doesn't work as expected.

    """
    signature = inspect.signature(model.seed)
    assert (
        "seed" in signature.parameters or (
            "kwargs" in signature.parameters
            and signature.parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD)
    ), "The `seed` method does not provide a `seed` or `**kwargs` keyword argument."
    try:
        model.seed(seed=123)
        state_1 = model.sample_initial_state()
        check_state(state_1, model)
        if model.observation_first:
            obs_1 = model.sample_initial_obs(state_1)
            check_agent_obs(obs_1, model.observation_spaces, "sample_initial_obs")

        assert model._rng is not None, (  # pyright: ignore [reportPrivateUsage]
            "Expects the random number generator to have been generated given a "
            "seed was passed to the `seed` method. Mostly likely the model `seed` "
            "method does not call `super().seed(seed=seed)` or the `rng` property has "
            "not been implemented."
        )
        seed_123_rng = deepcopy(model._rng)  # pyright: ignore [reportPrivateUsage]

        model.seed(seed=123)
        state_2 = model.sample_initial_state()
        check_state(state_2, model)
        if model.observation_first:
            obs_2 = model.sample_initial_obs(state_2)
            check_agent_obs(obs_2, model.observation_spaces, "sample_initial_obs")

        if model.spec is not None and model.spec.nondeterministic is False:
            assert data_equivalence(state_1, state_2), (
                "Using `model.seed(seed=123)` is non-deterministic as the "
                "initial states are not equivalent."
            )
            if model.observation_first:
                assert data_equivalence(obs_1, obs_2), (
                    "Using `model.seed(seed=123)` is non-deterministic as the "
                    "initial observations are not equivalent."
                )

        check_rng_equality(
            model._rng,  # pyright: ignore [reportPrivateUsage]
            seed_123_rng,
            prefix=(
                "Mostly likely the model seed function does not call "
                "`super().seed(seed=seed)` as the random generators are not same "
                "when the same seeds are passed to `model.seed`. "
            ),
        )

        model.seed(seed=456)
        state_3 = model.sample_initial_state()
        check_state(state_3, model)
        if model.observation_first:
            obs_3 = model.sample_initial_obs(state_3)
            check_agent_obs(obs_3, model.observation_spaces, "sample_initial_obs")

        try:
            check_rng_equality(
                model._rng,  # pyright: ignore [reportPrivateUsage]
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
            "The default seed argument in `seed` method should be `None`, otherwise "
            "the model will by default always be deterministic. "
            f"Actual default: {seed_param.default}"
        )


def check_model(model: M.POSGModel):
    """Check that an environment model follows posggym API.

    This is an invasive function that calls the models step.

    This is particularly useful when using a custom environment.

    TODO Update links

    Please take a look at https://gymnasium.farama.org/content/environment_creation/
    for more information about the API.

    Arguments
    ---------
    model: The posggym environment model that will be checked

    """
    more_info_msg = "See COMING SOON for more info."

    assert isinstance(
        model, M.POSGModel
    ), f"The model must inherit from the posggym.POSGModel class. {more_info_msg}"

    # ============= Check agents =============
    assert hasattr(
        model, "possible_agents"
    ), f"The model must specify possible agents tuple. {more_info_msg}"
    assert isinstance(model.possible_agents, tuple), (
        "Model `possible_agents` should be tuple (so it's immutable). "
        f"Actual type: {type(model.possible_agents)}."
    )
    assert all(isinstance(i, get_args(M.AgentID)) for i in model.possible_agents), (
        "Model agent IDs in `possible_agents` must be have type in "
        f"{get_args(M.AgentID)}. "
        f"Actual types: {list(type(i) for i in model.possible_agents)}."
    )

    # ============= Check basic model properties =============
    assert hasattr(
        model, "observation_first"
    ), f"The model must specify whether it's observation first or not. {more_info_msg}"
    assert isinstance(model.observation_first, bool), (
        "Model `observation_first` should be bool. "
        f"Actual type: {type(model.observation_first)}."
    )
    assert hasattr(
        model, "is_symmetric"
    ), f"The model must specify whether it's symmetric or not. {more_info_msg}"
    assert isinstance(
        model.is_symmetric, bool
    ), f"Model `is_symmetric` should be bool. Actual type: {type(model.is_symmetric)}."
    assert all(i in model.reward_ranges for i in model.possible_agents), (
        "Model `reward_ranges` should be defined for all agents in `possible_agents`."
        "Missing: "
        f"{list(i for i in model.possible_agents if i not in model.reward_ranges)}."
    )

    # ============= Check the spaces (action, observation and state) =============
    assert hasattr(
        model, "action_spaces"
    ), f"The model must specify agent action spaces. {more_info_msg}"
    check_agent_action_spaces(model.action_spaces)
    check_agent_space_limits(model.action_spaces, "action")

    assert hasattr(
        model, "observation_spaces"
    ), f"The environment must specify agent observation spaces. {more_info_msg}"
    check_agent_observation_spaces(model.observation_spaces)
    check_agent_space_limits(model.observation_spaces, "observation")

    if hasattr(model, "state_space") and model.state_space is not None:
        check_state_space(model.state_space)

    # ============= Check sampling initial states and observations =============
    check_initial_state_type(model)
    check_initial_obs_type(model)
    check_initial_sampling_seed(model)

    # ============ Check the returned values ===============
    state = model.sample_initial_state()
    model_step_passive_checker(
        model,
        state,
        {i: model.action_spaces[i].sample() for i in model.get_agents(state)}
    )
