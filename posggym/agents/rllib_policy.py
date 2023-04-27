"""Policy classes for wrapping Rllib policies to make them posggym-agents compatible."""
from __future__ import annotations

import abc
import os
import os.path as osp
import pickle
import random
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from posggym import logger
from posggym.agents.policy import ActType, ObsType, Policy, PolicyID, PolicyState
from posggym.agents.registration import PolicySpec
from posggym.agents.utils.download import download_from_repo
from posggym.agents.utils.preprocessors import (
    ObsPreprocessor,
    get_flatten_preprocessor,
    identity_preprocessor,
)


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy


if TYPE_CHECKING:
    from ray import rllib

    import posggym.model as M
    from posggym.agents.registration import PolicyEntryPoint
    from posggym.utils.history import AgentHistory


RllibHiddenState = List[Any]


_ACTION_DIST_INPUTS = "action_dist_inputs"
_ACTION_PROB = "action_prob"
_ACTION_LOGP = "action_logp"


class RllibPolicy(Policy[ActType, ObsType]):
    """A Rllib Policy.

    This class essentially acts as wrapper for an Rlib Policy class
    (ray.rllib.policy.policy.Policy).

    Note, if `explore=True` then policy performance may not be perfectly reproducible
    even if seed is set in `reset()` function. This is because Rllib makes use of global
    seeds/RNGS for random, numpy, and torch, and so reproducibility is sensitive to any
    other libraries, policies, etc that also use the global RNGs. It may still result
    in reproducible runs, but behaviour is less isolated.

    If `explore=False` then performance will select the maximizing action each step,
    so will be deterministic.

    """

    def __init__(
        self,
        model: M.POSGModel,
        agent_id: M.AgentID,
        policy_id: PolicyID,
        policy: rllib.policy.policy.Policy,
        preprocessor: Optional[ObsPreprocessor] = None,
        explore: bool = False,
    ):
        self._policy = policy
        if preprocessor is None:
            preprocessor = identity_preprocessor
        self._preprocessor = preprocessor
        self._explore = explore
        super().__init__(model, agent_id, policy_id)

    def step(self, obs: ObsType) -> ActType:
        self._state = self.get_next_state(obs, self._state)
        action = self.sample_action(self._state)
        self._state["action"] = action
        return action

    def reset(self, *, seed: int | None = None):
        super().reset(seed=seed)
        # Rllib doesn't use in-built RNG class, but rather relies on global seed
        # https://github.com/ray-project/ray/blob/master/rllib/utils/debug/deterministic.py
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

            import torch

            # https://pytorch.org/docs/stable/notes/randomness.html
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)

            if hasattr(self._policy, "device") and str(self._policy.device) != "cpu":
                torch.backends.cudnn.benchmark = False
                # See https://github.com/pytorch/pytorch/issues/47672.
                # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
                cuda_version = torch.version.cuda
                if (
                    cuda_version is not None
                    and float(cuda_version) >= 10.2
                    and (
                        "CUBLAS_WORKSPACE_CONFIG" not in os.environ
                        or (
                            os.environ["CUBLAS_WORKSPACE_CONFIG"]
                            not in (":16:8", ":4096:2")
                        )
                    )
                ):
                    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

    def get_initial_state(self) -> PolicyState:
        state = super().get_initial_state()
        state["last_obs"] = None
        state["action"] = None
        state["hidden_state"] = self._policy.get_initial_state()
        state["last_pi_info"] = {}
        return state

    def get_next_state(self, obs: ObsType, state: PolicyState) -> PolicyState:
        action, hidden_state, pi_info = self._compute_action(
            obs, state["hidden_state"], state["action"], explore=self._explore
        )
        return {
            "last_obs": obs,
            "action": action,
            "hidden_state": hidden_state,
            "pi_info": pi_info,
        }

    def sample_action(self, state: PolicyState) -> ActType:
        return state["action"]

    def get_pi(self, state: PolicyState) -> Dict[ActType, float]:
        return self._get_pi_from_info(state["pi_info"])

    @abc.abstractmethod
    def _get_pi_from_info(self, info: Dict[str, Any]) -> Dict[ActType, float]:
        """Get policy distribution from policy info output."""

    def get_value(self, state: PolicyState) -> float:
        return self._get_value_from_info(state["last_pi_info"])

    @abc.abstractmethod
    def _get_value_from_info(self, info: Dict[str, Any]) -> float:
        """Get value from policy info output."""

    def _compute_action(
        self,
        obs: ObsType,
        h_tm1: RllibHiddenState,
        last_action: ActType,
        explore: Optional[bool] = None,
    ) -> Tuple[ActType, RllibHiddenState, Dict[str, Any]]:
        obs = self._preprocessor(obs)
        output = self._policy.compute_single_action(
            obs, h_tm1, prev_action=last_action, explore=explore
        )
        return output

    def _unroll_history(
        self, history: AgentHistory
    ) -> Tuple[ObsType, RllibHiddenState, ActType, Dict[str, Any]]:
        h_tm1 = self._policy.get_initial_state()
        info_tm1: Dict[str, Any] = {}
        a_tp1, o_t = history[-1]

        for a_t, o_t in history:
            a_tp1, h_tm1, info_tm1 = self._compute_action(
                o_t, h_tm1, a_t, explore=self._explore
            )

        h_t, info_t = h_tm1, info_tm1
        # returns:
        # o_t - the final observation in the history
        # h_t - the hidden state after processing o_t, a_t, h_tm1
        # a_tp1 - the next action to perform after processing o_t, a_t, h_tm1
        # info_t - the info returned after processing o_t, a_t, h_tm1
        return o_t, h_t, a_tp1, info_t


class PPORllibPolicy(RllibPolicy[int, ObsType]):
    """A PPO Rllib Policy."""

    VF_PRED = "vf_preds"

    def _get_pi_from_info(self, info: Dict[str, Any]) -> Dict[int, float]:
        logits = info[_ACTION_DIST_INPUTS]
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=0)
        return {a: probs[a] for a in range(len(probs))}

    def _get_value_from_info(self, info: Dict[str, Any]) -> float:
        return info[self.VF_PRED]


def get_rllib_policy_entry_point(policy_file: str) -> PolicyEntryPoint:
    """Get Rllib policy entry point from policy file.

    Arguments
    ---------
    policy_file: the path the rllib policy .pkl file, containing the policy weights and
        configuration information.

    Returns
    -------
    The PolicyEntryPoint function for rllib policy stored in the specified policy file.

    """

    def _entry_point(model: M.POSGModel, agent_id: M.AgentID, policy_id: str, **kwargs):
        preprocessor = get_flatten_preprocessor(model.observation_spaces[agent_id])

        # download policy file from repo if it doesn't already exist
        if not osp.exists(policy_file):
            logger.info(
                f"Local copy of policy file for policy `{policy_id}` not found, so "
                "downloading it from posggym-agents repo and storing local copy for "
                "future use."
            )
            download_from_repo(policy_file, rewrite_existing=False)

        with open(policy_file, "rb") as f:
            data = pickle.load(f)

        action_space = model.action_spaces[agent_id]
        obs_space = model.observation_spaces[agent_id]
        flat_obs_space: spaces.Box = spaces.flatten_space(obs_space)  # type: ignore

        ppo_policy = PPOTorchPolicy(flat_obs_space, action_space, data["config"])
        ppo_policy.set_state(data["state"])

        return PPORllibPolicy(
            model, agent_id, policy_id, ppo_policy, preprocessor, **kwargs
        )

    return _entry_point


def load_rllib_policy_specs_from_files(
    env_id: str,
    env_args: Dict[str, Any] | None,
    policy_file_dir_path: str,
    policy_file_names: List[str],
    version: int = 0,
    valid_agent_ids: Optional[List[M.AgentID]] = None,
    nondeterministic: bool = False,
    **kwargs,
) -> Dict[str, PolicySpec]:
    """Load policy specs for rllib policies from list of policy files.

    Arguments
    ---------
    env_id: ID of the posggym environment that the policy is for.
    env_args: Optional keywords arguments for the environment that the policy is for (if
        it is a environment specific policy). If None then assumes policy can be used
        for the environment with any arguments.
    policy_file_dir_path: path to directory where policy files are located.
    policy_file_names: names of all the policy files to load.
    version: the policy version
    valid_agent_ids: Optional AgentIDs in environment that policy is compatible with. If
        None then assumes policy can be used for any agent in the environment.
    nondeterministic: Whether this policy is non-deterministic even after seeding.
    kwargs: Additional kwargs, if any, to pass to the agent initializing

    Returns
    -------
    Mapping from policy ID to Policy specs for the policy files.

    """
    policy_specs = {}
    for file_name in policy_file_names:
        policy_file = osp.join(policy_file_dir_path, file_name)
        # remove file extension, e.g. .pkl
        policy_name = file_name.split(".")[0]

        spec = PolicySpec(
            policy_name=policy_name,
            entry_point=get_rllib_policy_entry_point(policy_file),
            version=version,
            env_id=env_id,
            env_args=env_args,
            valid_agent_ids=valid_agent_ids,
            nondeterministic=nondeterministic,
            kwargs=kwargs,
        )
        policy_specs[spec.id] = spec
    return policy_specs
