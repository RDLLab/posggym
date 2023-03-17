"""Checks that the core posggym environment API is implemented as expected.

Adapted from:
https://github.com/Farama-Foundation/Gymnasium/blob/v0.27.0/tests/test_core.py

"""
from typing import Optional, Dict, Tuple, Any

import numpy as np
from gymnasium.spaces import Box

import posggym.model as M
from posggym import (
    ActionWrapper,
    DefaultEnv,
    Env,
    ObservationWrapper,
    RewardWrapper,
    Wrapper,
)
from posggym.core import WrapperActType, WrapperObsType
from tests.test_model import ExampleModel


class ExampleEnv(DefaultEnv[int, int, int]):
    """Example testing environment."""

    def __init__(self):
        super().__init__(ExampleModel())


def test_posggym_env():
    """Tests general posggym environment API."""
    env = ExampleEnv()

    assert env.metadata == {"render_modes": []}
    assert env.render_mode is None
    assert env.spec is None


class ExampleWrapper(Wrapper):
    """An example testing wrapper."""

    def __init__(self, env: Env[M.StateType, M.ObsType, M.ActType]):
        """Constructor that sets the reward."""
        super().__init__(env)
        self.new_reward = 3

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[M.AgentID, WrapperObsType], Dict[M.AgentID, Dict]]:
        return super().reset(seed=seed, options=options)

    def step(
        self, actions: Dict[M.AgentID, WrapperActType]
    ) -> Tuple[
        Dict[M.AgentID, WrapperObsType],
        Dict[M.AgentID, float],
        Dict[M.AgentID, bool],
        Dict[M.AgentID, bool],
        bool,
        Dict[M.AgentID, Dict],
    ]:
        obs, reward, term, trunc, done, info = self.env.step(actions)  # type: ignore
        reward = {i: self.new_reward for i in reward}
        return obs, reward, term, trunc, done, info


def test_posggym_wrapper():
    """Tests the posggym wrapper API works as expected."""
    env = ExampleEnv()
    wrapper_env = ExampleWrapper(env)

    assert env.metadata == wrapper_env.metadata
    wrapper_env.metadata = {"render_modes": ["rgb_array"]}
    assert env.metadata != wrapper_env.metadata

    assert env.render_mode == wrapper_env.render_mode

    assert env.reward_ranges == wrapper_env.reward_ranges
    wrapper_env.reward_ranges = {i: (-1.0, 1.0) for i in wrapper_env.possible_agents}
    assert env.reward_ranges != wrapper_env.reward_ranges

    assert env.spec == wrapper_env.spec

    assert env.observation_spaces == wrapper_env.observation_spaces
    assert env.action_spaces == wrapper_env.action_spaces
    wrapper_env.observation_spaces = {i: Box(1, 2) for i in wrapper_env.possible_agents}
    wrapper_env.action_spaces = {i: Box(1, 2) for i in wrapper_env.possible_agents}
    assert env.observation_spaces != wrapper_env.observation_spaces
    assert env.action_spaces != wrapper_env.action_spaces

    assert wrapper_env.model is env.model


class ExampleRewardWrapper(RewardWrapper):
    """Example reward wrapper for testing."""

    def rewards(self, rewards: Dict[M.AgentID, float]) -> Dict[M.AgentID, float]:
        return {i: 1 for i in rewards}


class ExampleObservationWrapper(ObservationWrapper):
    """Example observation wrapper for testing."""

    def observations(
        self, obs: Dict[M.AgentID, M.ObsType]
    ) -> Dict[M.AgentID, WrapperObsType]:
        return {i: np.array([1]) for i in obs}  # type: ignore


class ExampleActionWrapper(ActionWrapper):
    """Example action wrapper for testing."""

    def actions(
        self, actions: Dict[M.AgentID, M.ActType]
    ) -> Dict[M.AgentID, WrapperActType]:
        return {i: np.array([1]) for i in actions}  # type: ignore


class ActionWrapperTestEnv(DefaultEnv[int, int, int]):
    """Example environment for tesint action wrapper.

    Step returns the action as an observation.
    """

    def __init__(self):
        super().__init__(ExampleModel())

    def step(
        self, actions: Dict[M.AgentID, int]
    ) -> Tuple[
        Dict[M.AgentID, int],
        Dict[M.AgentID, float],
        Dict[M.AgentID, bool],
        Dict[M.AgentID, bool],
        bool,
        Dict[M.AgentID, Dict],
    ]:
        step = self.model.step(self._state, actions)
        self._step_num += 1
        self._state = step.state
        self._last_obs = step.observations
        self._last_actions = actions
        self._last_rewards = step.rewards
        return (
            actions,
            step.rewards,
            step.terminated,
            step.truncated,
            step.all_done,
            step.info,
        )


def test_wrapper_types():
    """Tests the observation, action and reward wrapper examples."""
    env = ExampleEnv()

    reward_env = ExampleRewardWrapper(env)
    reward_env.reset()
    _, rewards, _, _, _, _ = reward_env.step(0)
    assert all(r_i == 1 for r_i in rewards.values())

    observation_env = ExampleObservationWrapper(env)
    obs, _ = observation_env.reset()
    assert all(o_i == np.array([1]) for o_i in obs.values())
    obs, _, _, _, _, _ = observation_env.step({i: 0 for i in env.possible_agents})
    assert all(o_i == np.array([1]) for o_i in obs.values())

    env = ActionWrapperTestEnv()
    action_env = ExampleActionWrapper(env)
    obs, _, _, _, _, _ = action_env.step({i: 0 for i in env.possible_agents})
    assert all(o_i == np.array([1]) for o_i in obs.values())
