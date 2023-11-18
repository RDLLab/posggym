"""Synchronized vectorized environment class.

Based on Gymnasium Vectorized Environments:
https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/vector/vector_env.py
https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/vector/sync_vector_env.py

"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
from gymnasium.vector.utils import concatenate, create_empty_array
from gymnasium.vector.utils.spaces import batch_space

import posggym


class SyncVectorEnv(posggym.Env):
    """Vectorized environment that serially runs multiple environments.

    This implementation is based on the Gymnasium Vectorized Environments, modified to
    support multiple agents and the POSGGym API.

    A vectorized environment runs multiple independent copies of the same environment
    allowing for batched interactions. To prevent terminated environments waiting until
    all sub-environments are done (terminated or truncated), the vector environment
    autoresets sub-environments when they are done. As a result, the final observaiton
    and info are overwritten by the reset's observation and info. The final observation
    and info of the previous environment is stored in the info dictionary of each agent
    under the keys ``final_observation`` and ``final_info``. See :meth:`step` for more
    information.

    Vector Environments have additional attributes on top of `posggym.Env` as well as,
    slightly modified :attr:`observation_spaces` and :attr:`action_spaces` attributes:

    - :attr:`num_envs` - The number of sub-environments in the vector environment.
    - :attr:`observation_spaces` - The batched observation space of each agent of the
        vector environment.
    - :attr:`single_observation_spaces` - The observation space of each agent in a
        single sub-environment.
    - :attr:`action_spaces` - The batched action space of each agent of the  vector
        environment.
    - :attr:`single_action_spaces` - The action space of each agent in a  single
        sub-environment.


    Notes
    -----
    https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/vector/vector_env.py
    https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/vector/sync_vector_env.py

    """

    def __init__(
        self,
        env_fns: Iterable[Callable[[], posggym.Env]],
        copy: bool = True,
    ):
        """Initialize the vectorized environment.

        Arguments
        ---------
        env_fns:
            iterable of callable functions that create the environments.
        copy:
            If ``True``, then the :meth:`reset` and :meth:`step` methods return a
            copy of the observations.

        """
        self.env_fns = env_fns
        self.envs = [fn() for fn in env_fns]
        self.copy = copy
        self.is_vector_env = True

        self.metadata = self.envs[0].metadata
        self.model = self.envs[0].model

        self.num_envs = len(self.envs)

        self.single_observation_spaces = self.envs[0].observation_spaces
        self.single_action_spaces = self.envs[0].action_spaces

        self._observation_spaces = {
            i: batch_space(self.single_observation_spaces[i], n=self.num_envs)
            for i in self.single_observation_spaces
        }
        self._action_spaces = {
            i: batch_space(self.single_action_spaces[i], n=self.num_envs)
            for i in self.single_action_spaces
        }

        self._check_spaces()

        self.observations = {
            i: create_empty_array(
                self.single_observation_spaces[i], n=self.num_envs, fn=np.zeros
            )
            for i in self.single_observation_spaces
        }
        self._rewards = {
            i: np.zeros((self.num_envs,), dtype=np.float32)
            for i in self.single_observation_spaces
        }
        self._terminateds = {
            i: np.zeros((self.num_envs,), dtype=np.bool_)
            for i in self.single_observation_spaces
        }
        self._truncateds = {
            i: np.zeros((self.num_envs,), dtype=np.bool_)
            for i in self.single_observation_spaces
        }
        self._all_dones = np.zeros((self.num_envs,), dtype=np.bool_)
        self._actions = {i: None for i in self.single_action_spaces}

    def reset(
        self,
        *,
        seed: int | None | List[int] = None,
        options: Dict[str, Any] | None = None,
    ):
        """Reset all environments and return batch of initial observations and info."""
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        self._all_dones[:] = False
        for i in self.single_observation_spaces:
            self._terminateds[i][:] = False
            self._truncateds[i][:] = False

        observations = {i: [] for i in self.single_observation_spaces}
        infos = {i: {} for i in self.single_observation_spaces}

        for env_num, (env, s) in enumerate(zip(self.envs, seed)):
            obs, info = env.reset(seed=s, options=options)
            for i in self.single_observation_spaces:
                observations[i].append(obs[i])
                infos[i] = self._add_info(infos[i], info[i], env_num)

        for i in self.single_observation_spaces:
            self.observations[i] = concatenate(
                self.single_observation_spaces[i],
                observations[i],
                self.observations[i],
            )

        return (deepcopy(self.observations) if self.copy else self.observations), infos

    def step(self, actions):
        """Take a step in all environments with the given actions.

        If any environment is done, then it is reset before taking the next step. The
        final observation and info of the previous environment is stored in the info
        dictionary under the keys ``final_observation`` and ``final_info``.


        Arguments
        ---------
        actions:
            dict mapping agent ID to batch of actions for that agent, with one action
            for each environment. So should be a dict of arrays or lists, with each
            array/list having length equal to the number of environments.

        Returns
        -------
        observations:
            dict mapping agent ID to batch of observations for that agent, with one
            observation for each environment. Each array having shape
            ``(num_envs,) + observation_space.shape``.
        rewards:
            dict mapping agent ID to batch of rewards for that agent, with one reward
            for each environment. Each entry is an array with  shape ``(num_envs,)``.
        terminateds:
            dict mapping agent ID to batch of termination signals for that agent, with
            one termination signal for each environment. Each entry is an array with
            shape ``(num_envs,)``.
        truncateds:
            dict mapping agent ID to batch of truncation signals for that agent, with
            one truncation signal for each environment. Each entry is an array with
            shape ``(num_envs,)``.
        all_dones:
            array of booleans, with one entry for each environment, indicating whether
            the environment is done. Shape is ``(num_envs,)``.
        infos:
            dict mapping agent ID to batch of info objects for that agent, with one
            info object for each environment. Each entry is a dict with one entry for


        The batched environment step results.

        """
        observations = {i: [] for i in self.single_observation_spaces}
        infos = {i: {} for i in self.single_observation_spaces}

        for env_num, env in enumerate(self.envs):
            action = {i: actions[i][env_num] for i in self.single_action_spaces}

            observation, rewards, terminateds, truncateds, all_done, info = env.step(
                action
            )

            if all_done:
                old_observation, old_info = observation, info
                observation, info = env.reset()
                for i in old_observation:
                    info[i]["final_observation"] = old_observation[i]
                    info[i]["final_info"] = old_info[i]

            for i in self.single_observation_spaces:
                observations[i].append(observation[i])
                infos[i] = self._add_info(infos[i], info[i], env_num)
                self._rewards[i][env_num] = rewards[i]
                self._terminateds[i][env_num] = terminateds[i]
                self._truncateds[i][env_num] = truncateds[i]

            self._all_dones[env_num] = all_done

        for i in self.single_observation_spaces:
            self.observations[i] = concatenate(
                self.single_observation_spaces[i],
                observations[i],
                self.observations[i],
            )

        return (
            (deepcopy(self.observations) if self.copy else self.observations),
            deepcopy(self._rewards) if self.copy else self._rewards,
            deepcopy(self._terminateds) if self.copy else self._terminateds,
            deepcopy(self._truncateds) if self.copy else self._truncateds,
            deepcopy(self._all_dones) if self.copy else self._all_dones,
            infos,
        )

    def render(self):
        return self.call("render")

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()

    def call(self, name: str, *args, **kwargs) -> Tuple:
        """Call a method on all environments and return the results."""
        results = []
        for env in self.envs:
            function = getattr(env, name)
            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)
        return tuple(results)

    @property
    def possible_agents(self) -> Tuple[str, ...]:
        return self.envs[0].possible_agents

    @property
    def agents(self):
        return self.call("agents")

    @property
    def state(self):
        return self.call("state")

    @property
    def observation_spaces(self):
        return self._observation_spaces

    @property
    def action_spaces(self):
        return self._action_spaces

    def _add_info(self, infos: dict, info: dict, env_num: int) -> dict:
        """Add env info to the info dictionary of the vectorized environment.

        Given the `info` of a single environment add it to the `infos` dictionary
        which represents all the infos of the vectorized environment.
        Every `key` of `info` is paired with a boolean mask `_key` representing
        whether or not the i-indexed environment has this `info`.

        Arguments:
        ----------
        infos: dict
            the infos of the vectorized environment
        info: dict
            the info coming from the single environment
        env_num: int
            the index of the single environment

        Returns
        -------
        infos: dict
            the (updated) infos of the vectorized environment

        """
        for k in info:
            if k not in infos:
                info_array, array_mask = self._init_info_arrays(type(info[k]))
            else:
                info_array, array_mask = infos[k], infos[f"_{k}"]

            info_array[env_num], array_mask[env_num] = info[k], True
            infos[k], infos[f"_{k}"] = info_array, array_mask
        return infos

    def _init_info_arrays(self, dtype: type) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize the info array.

        Initialize the info array. If the dtype is numeric the info array will have the
        same dtype, otherwise will be an array of `None`. Also, a boolean array of the
        same length is returned. It will be used for assessing which environment has
        info data.

        Arguments
        ---------
        dtype: type
            data type of the info coming from the env.

        Returns
        -------
        array: np.ndarray
            the initialized info array.
        array_mask: np.ndarray
            the initialized boolean array.

        """
        if dtype in [int, float, bool] or issubclass(dtype, np.number):
            array = np.zeros(self.num_envs, dtype=dtype)
        else:
            array = np.zeros(self.num_envs, dtype=object)
            array[:] = None
        array_mask = np.zeros(self.num_envs, dtype=bool)
        return array, array_mask

    def _check_spaces(self) -> bool:
        for env in self.envs:
            for i in self.single_observation_spaces:
                if (
                    i not in env.observation_spaces
                    or env.observation_spaces[i] != self.single_observation_spaces[i]
                ):
                    raise RuntimeError(
                        "Some environments have an observation space different from "
                        f"`{self.single_observation_spaces[i]}`. In order to batch "
                        "observations, the observation spaces from all environments "
                        "must be equal."
                    )

                if (
                    i not in env.action_spaces
                    or env.action_spaces[i] != self.single_action_spaces[i]
                ):
                    raise RuntimeError(
                        "Some environments have an action space different from "
                        f"`{self.single_action_spaces[i]}`. In order to batch actions, "
                        "the action spaces from all environments must be equal."
                    )

        return True
